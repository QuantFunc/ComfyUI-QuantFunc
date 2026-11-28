#!/usr/bin/env python3
"""QuantFunc PC Engine installer, resolver, zero-CUDA host guard, and sidecar preload rules
end to end against a FAKE release server.

PC Engine layout (O1 + O6):
  - Consumer 4 SMs ONLY: sm75, sm86, sm89, sm120a (server SMs 80/90/100/103 are refused).
  - Linux & Windows: zero-CUDA host libquantfunc(-12).so / quantfunc(-12).dll + per-SM class folder
    sm{75,86,89,120a}/libquantfunc_kernels(-12).{so,dll} (no cu13/ or cu12/ subdir; Linux pair shares .qf_pair_id).
  - No libquantfunc_attention.so; no libquantfunc_kernels_cuda_sm*.so.
"""
import contextlib
import errno as _errno
import hashlib
import importlib.util
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import types
import urllib.request

_PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("qfe_install_test", os.path.join(_PLUGIN, "qf_engine.py"))
qfe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qfe)

HOSTS = {13: "libquantfunc.so", 12: "libquantfunc-12.so"}
CONSUMER_SETS = {"sm75": [75], "sm86": [86], "sm89": [89], "sm120a": [120]}
WIN_DLLS = {13: "quantfunc.dll", 12: "quantfunc-12.dll"}
bad = 0


def kernel_for(sm=89, major=13, win=False):
    suffix = "-12" if major == 12 else ""
    ext = "dll" if win else "so"
    return f"libquantfunc_kernels{suffix}.{ext}"


def check(label, ok, detail=""):
    global bad
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{(' -> ' + str(detail)) if detail else ''}")
    bad += not ok


def sha(b):
    return hashlib.sha256(b).hexdigest()


class _Resp(io.BytesIO):
    def __init__(self, data, final_url):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}
        self._url = final_url

    def geturl(self):
        return self._url


class Release:
    """Fake PC Engine release repo with zero-CUDA host + consumer per-SM kernel plugins."""

    def __init__(self, version="0.0.14", plugin_req="0.0.08", sets=CONSUMER_SETS):
        self.files, self.fetched, self.offline = {}, [], False
        self.version = version
        body = {}
        for major in (13, 12):
            h = f"HOST-{version}-cu{major}".encode()
            self.files[f"{version}/linux/{HOSTS[major]}"] = h
            body[HOSTS[major]] = sha(h)
            for gset in (sets or CONSUMER_SETS):
                kname = kernel_for(None, major)
                k = f"KERNEL-{version}-{gset}-cu{major}".encode()
                self.files[f"{version}/linux/{gset}/{kname}"] = k
                body[f"{gset}/{kname}"] = sha(k)
        self.manifest = {"schema": 1, "linux": body}
        if sets is not None:
            self.set_sets(json.dumps({"schema": 2, "sets": sets}).encode())
        self.publish()
        self.entries = {
            "0.0.12": {"comfy": "0.0.06", "comfy-12": "0.0.06", "lib": "0.0.12", "lib-12": "0.0.12"},
            version: {"comfy": plugin_req, "comfy-12": plugin_req, "lib": version, "lib-12": version, "kernel_so": True},
        }
        self.files["version.json"] = json.dumps({"linux": self.entries}).encode()

    def set_sets(self, raw, hashed=True):
        self.files[f"{self.version}/linux/sets.json"] = raw
        if hashed:
            self.manifest["linux"]["sets.json"] = sha(raw)
        self.publish()

    def publish(self):
        self.files[f"{self.version}/verify.json"] = json.dumps(self.manifest).encode()

    def open(self, url):
        if self.offline:
            raise OSError("network is unreachable")
        assert url.startswith(qfe._ENGINE_BASE_URL + "/"), url
        rel = url[len(qfe._ENGINE_BASE_URL) + 1:]
        self.fetched.append(rel)
        if rel not in self.files:
            raise OSError(f"404 {rel}")
        return _Resp(self.files[rel], "https://cdn.example/" + rel)


class WinRelease(Release):
    """Windows consumer zero-CUDA host DLL + per-SM kernel DLL release layout."""

    def __init__(self, version="0.0.14", plugin_req="0.0.08"):
        self.files, self.fetched, self.offline, self.version = {}, [], False, version
        self.entries = {"0.0.12": {"comfy": "0.0.06", "comfy-12": "0.0.06", "lib": "0.0.12", "lib-12": "0.0.12"}}
        self.add(version, plugin_req)

    def add(self, version, plugin_req="0.0.08", tamper=None):
        raw = json.dumps({"schema": 2, "sets": CONSUMER_SETS}).encode() + b"\r\n"
        self.files[f"{version}/windows/sets.json"] = raw
        body = {"sets.json": sha(raw)}
        for major, dll in WIN_DLLS.items():
            hdata = f"WINHOST-{version}-cu{major}".encode()
            self.files[f"{version}/windows/{dll}"] = hdata
            body[dll] = sha(b"tampered" if tamper == dll else hdata)
            for gset in CONSUMER_SETS:
                kname = kernel_for(None, major, win=True)
                kdata = f"WINKERNEL-{version}-{gset}-cu{major}".encode()
                self.files[f"{version}/windows/{gset}/{kname}"] = kdata
                body[f"{gset}/{kname}"] = sha(
                    b"tampered" if tamper == f"{gset}/{kname}" else kdata)
        self.files[f"{version}/verify.json"] = json.dumps({"schema": 1, "win32": body}).encode()
        self.entries[version] = {"comfy": plugin_req, "comfy-12": plugin_req, "lib": version, "lib-12": version,
                                 "kernel_so": True}
        self.files["version.json"] = json.dumps({"linux": {}, "win32": self.entries}).encode()


class _FakeMsvcrt(types.ModuleType):
    LK_UNLCK, LK_LOCK, LK_NBLCK = 0, 1, 2

    def __init__(self):
        super().__init__("msvcrt")
        self.calls, self.fail = [], []

    def locking(self, fd, mode, nbytes):
        self.calls.append(({self.LK_UNLCK: "unlock", self.LK_NBLCK: "nblck"}.get(mode, "lock"), nbytes))
        if mode != self.LK_UNLCK and self.fail:
            code = self.fail.pop(0) if mode == self.LK_NBLCK else (
                self.fail.pop(0), getattr(_errno, "EDEADLOCK", _errno.EDEADLK))[1]
            raise OSError(code, os.strerror(code))


@contextlib.contextmanager
def windows():
    saved = qfe._BIN_SUBDIR, qfe._LIB_BASENAME, sys.modules.get("msvcrt")
    fake = _FakeMsvcrt()
    qfe._BIN_SUBDIR, qfe._LIB_BASENAME, sys.modules["msvcrt"] = "windows", "quantfunc.dll", fake
    try:
        yield fake
    finally:
        qfe._BIN_SUBDIR, qfe._LIB_BASENAME = saved[0], saved[1]
        if saved[2] is None:
            sys.modules.pop("msvcrt", None)
        else:
            sys.modules["msvcrt"] = saved[2]


def fake_needed(path):
    """PC Engine O1: host has ZERO CUDA DT_NEEDED; per-SM kernel plugin links CUDA."""
    try:
        data = open(path, "rb").read()
    except OSError:
        return []
    if data.startswith(b"HOST-"):
        return ["libstdc++.so.6", "libpthread.so.0"]
    for major in (13, 12):
        if data.startswith(b"KERNEL-") and data.endswith(f"-cu{major}".encode()):
            return [f"libcudart.so.{major}", f"libcublas.so.{major}"]
    return []


def fake_pair_id(path):
    try:
        parts = open(path, "rb").read().decode().split("-")
    except (OSError, UnicodeDecodeError):
        return None
    return sha(f"{parts[1]}-{parts[-1]}".encode())[:32] if parts[0] in ("HOST", "KERNEL") and len(parts) > 2 else None


def mk_elf(sections):
    names = list(sections) + [".shstrtab"]
    shstr = b"\0" + b"".join(n.encode() + b"\0" for n in names)
    body, offs, name_at, at = b"", {}, {}, 1
    for n in names:
        name_at[n], at = at, at + len(n) + 1
        offs[n] = 64 + len(body)
        body += sections[n] if n in sections else shstr
    head = bytearray(64)
    head[0:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<Q", head, 0x28, 64 + len(body))
    struct.pack_into("<HHH", head, 0x3a, 64, len(names) + 1, len(names))
    shdrs = bytes(64)
    for n in names:
        sh = bytearray(64)
        struct.pack_into("<II", sh, 0, name_at[n], 3 if n == ".shstrtab" else 1)
        struct.pack_into("<QQ", sh, 0x18, offs[n], len(sections[n]) if n in sections else len(shstr))
        shdrs += bytes(sh)
    return bytes(head) + body + shdrs


_STUBBED = ("_engine_http_open", "_torch_cuda_major", "_driver_cuda_major", "_gpu_sm", "_engine_bin_dir",
            "_elf_needed", "_elf_pair_id", "start_engine_install", "install_engine", "_ENGINE_DEVICE")


class Env:
    """Isolated temp bin/ inside a dedicated sandbox parent + stubbed hardware/network."""

    def __init__(self, release, torch_major=13, driver_major=13, sm=89, plugin_version="0.0.08", machine="x86_64"):
        self.release = release
        self.parent = tempfile.mkdtemp(prefix="qf_engine_install_root_")
        self.dir = os.path.join(self.parent, "bin")
        os.makedirs(self.dir)
        with open(os.path.join(self.dir, "version.json"), "w", encoding="utf-8") as f:
            json.dump({"comfy": plugin_version}, f)
        self._saved = {n: getattr(qfe, n) for n in _STUBBED}
        self._machine = qfe.platform.machine
        self._override = os.environ.pop(qfe._ENV_SO_OVERRIDE, None)
        qfe._engine_http_open = release.open
        qfe._torch_cuda_major = lambda: torch_major
        qfe._driver_cuda_major = lambda: driver_major
        qfe._gpu_sm = (lambda device_idx=0: sm[int(device_idx)]) if isinstance(sm, dict) else (lambda device_idx=0: sm)
        qfe._engine_bin_dir = lambda: self.dir
        qfe._elf_needed = fake_needed
        qfe._elf_pair_id = fake_pair_id
        qfe.platform.machine = lambda: machine
        qfe._ENGINE_DEVICE = 0
        qfe._engine_status("idle")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for n, v in self._saved.items():
            setattr(qfe, n, v)
        qfe.platform.machine = self._machine
        os.environ.pop(qfe._ENV_SO_OVERRIDE, None)
        if self._override is not None:
            os.environ[qfe._ENV_SO_OVERRIDE] = self._override
        shutil.rmtree(self.parent, ignore_errors=True)

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def read(self, *parts):
        p = self.path(*parts)
        return open(p, "rb").read() if os.path.isfile(p) else None

    def marker(self, gset="sm89", major=13):
        raw = self.read(f".engine-{gset}-cu{major}.json")
        return json.loads(raw) if raw is not None else None

    def pairs(self):
        return sorted(d for d in os.listdir(self.dir) if qfe._ENGINE_PAIR_RE.fullmatch(d))

    def leftovers(self):
        return sorted(os.path.join(r, f) for r, _, fs in os.walk(self.dir) for f in fs if ".part" in f)


def _join_install(t):
    if t is not None:
        t.join(10)


def main():
    print("=== PC Engine installer + resolver + zero-CUDA host guard ===")

    # 1) All 4 consumer SMs x CUDA 13 & 12: pair lands in its own folder, per-SM kernel before host, marker LAST
    for sm, gset in ((75, "sm75"), (86, "sm86"), (89, "sm89"), (120, "sm120a")):
        for major in (13, 12):
            rel = Release()
            kname = kernel_for(sm, major)
            with Env(rel, torch_major=major, driver_major=13, sm=sm) as env:
                order, real_replace = [], os.replace
                qfe.os.replace = lambda a, b: (order.append(os.path.basename(b)), real_replace(a, b))[1]
                try:
                    qfe.install_engine()
                finally:
                    qfe.os.replace = real_replace
                pair = f"0.0.14-{gset}-cu{major}"
                h = rel.files[f"0.0.14/linux/{HOSTS[major]}"]
                k = rel.files[f"0.0.14/linux/{gset}/{kname}"]
                want = {"version": "0.0.14", "set": gset, "cuda": major, "sms": [sm], "host": HOSTS[major],
                        "kernel": kname, "sha256": {HOSTS[major]: sha(h), kname: sha(k)}}
                m = env.marker(gset, major)
                ok = (env.read(pair, HOSTS[major]) == h and env.read(pair, kname) == k and m == want
                      and order == [kname, HOSTS[major], f".engine-{gset}-cu{major}.json"]
                      and env.read(HOSTS[major]) is None and not env.leftovers())
                check(f"SM {sm} x CUDA {major}: {gset} pair ({kname}), kernel before host, marker LAST", ok,
                      f"order={order} marker={m}")

    # 1b) O1 zero-CUDA host guard at install time: a host ELF linking libcudart/libcublas is REFUSED
    for leaked in ("libcudart.so.13", "libcublas.so.13"):
        rel = Release()
        with Env(rel, sm=89) as env:
            qfe._elf_needed = lambda p, l=leaked: [l] if ".libquantfunc.so" in os.path.basename(p) else []
            try:
                qfe.install_engine()
                msg = "installed!"
            except RuntimeError as e:
                msg = str(e)
            check(f"O1 install guard refuses host linking {leaked} (zero CUDA DT_NEEDED required)",
                  "zero CUDA DT_NEEDED" in msg and env.marker() is None and not env.leftovers(), msg[:80])

    # 2) Server SMs (80/90/100/103) and unsupported SMs (61/87/110/121) are REFUSED (consumer 4-SM scope O6)
    rel_with_server = Release(sets={**CONSUMER_SETS, "sm80": [80], "sm90a": [90], "sm100a": [100], "sm103a": [103]})
    refused_sms = {}
    for sm in (61, 80, 87, 90, 100, 103, 110, 121):
        with Env(rel_with_server, sm=sm) as env:
            try:
                qfe.install_engine()
                refused_sms[sm] = "installed!"
            except qfe.EngineNotInstallable as e:
                refused_sms[sm] = f"SM {sm // 10}.{sm % 10}" in str(e) and not env.pairs()
    check("O6 consumer-only scope: server SMs (80/90/100/103) and unsupported SMs (61/87/110/121) are refused",
          all(v is True for v in refused_sms.values()), refused_sms)

    # 2b) Old driver / CPU torch / non-x86_64
    rel = Release()
    with Env(rel, torch_major=13, driver_major=12) as env:
        try:
            qfe.install_engine()
            check("driver older than torch CUDA installs nothing", False, "installed!")
        except qfe.EngineNotInstallable as e:
            check("driver older than torch CUDA installs nothing", "update the driver" in str(e) and not rel.fetched)
    rel = Release()
    with Env(rel, torch_major=None, driver_major=12) as env:
        try:
            qfe.install_engine()
            got = "installed"
        except qfe.EngineNotInstallable as e:
            got = str(e)
        check("torch CUDA unknown: nothing installed", "CUDA build of PyTorch" in got and not rel.fetched)
    rel = Release()
    with Env(rel, machine="aarch64") as env:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _join_install(qfe.start_engine_install())
        state, detail = qfe.engine_install_status()
        check("aarch64 machine installs nothing, says why once",
              state == "unavailable" and "aarch64" in detail and not rel.fetched and not env.pairs()
              and out.getvalue().count("[qf_native]") == 1)

    # 3) sets.json validation: missing, traversal name, schema != 2, hash mismatch
    rel = Release(sets=None)
    with Env(rel) as env:
        try:
            qfe.install_engine()
            check("release without sets.json refused", False)
        except RuntimeError as e:
            check("release without sets.json refused", "publishes no sets.json" in str(e) and env.marker() is None)
    rel = Release()
    rel.set_sets(json.dumps({"schema": 2, "sets": {"../x": [89]}}).encode())
    with Env(rel) as env:
        try:
            qfe.install_engine()
            check("traversal sets.json class name refused", False)
        except RuntimeError as e:
            check("traversal sets.json class name refused", "GPU-class map" in str(e) and not env.pairs())
    rel = Release()
    rel.set_sets(rel.files["0.0.14/linux/sets.json"] + b" ", hashed=False)
    with Env(rel) as env:
        try:
            qfe.install_engine()
            check("tampered sets.json hash refused", False)
        except RuntimeError as e:
            check("tampered sets.json hash refused", "sets.json" in str(e))
    res = {}
    for label, schema in (("schema 1", 1), ("float 2.0", 2.0)):
        rel = Release()
        rel.set_sets(json.dumps({"schema": schema, "sets": CONSUMER_SETS}).encode())
        with Env(rel, sm=89) as env:
            try:
                qfe.install_engine()
                res[label] = "installed!"
            except RuntimeError as e:
                res[label] = "schema" in str(e) and not env.pairs()
    check("sets.json schema != int 2 refused", res == {"schema 1": True, "float 2.0": True}, res)

    # 4) Present + matching: nothing re-downloaded
    rel = Release()
    with Env(rel) as env:
        qfe.install_engine()
        rel.fetched.clear()
        qfe.install_engine()
        check("present + matching: nothing re-downloaded",
              rel.fetched == ["version.json", "0.0.14/verify.json", "0.0.14/linux/sets.json"], rel.fetched)

    # 5) Upgrade, corrupt download refusal, removing older-than-previous pair, per-CUDA-flavor version selection
    old = Release("0.0.14")
    with Env(old) as env:
        qfe.install_engine()
        new = Release("0.0.15")
        new.files[f"0.0.15/linux/{HOSTS[13]}"] = b"HOST-tampered-cu13"
        qfe._engine_http_open = new.open
        try:
            qfe.install_engine()
            check("corrupt download refused", False)
        except RuntimeError as e:
            check("corrupt download refused; installed pair & marker untouched",
                  "SHA-256" in str(e) and env.marker()["version"] == "0.0.14" and not env.leftovers())
        qfe._engine_http_open = Release("0.0.15").open
        qfe.install_engine()
        after15 = (env.marker()["version"], env.pairs())
        qfe._engine_http_open = Release("0.0.16").open
        qfe.install_engine()
        after16 = (env.marker()["version"], env.pairs())
        check("upgrade keeps 1 previous pair and removes older",
              after15 == ("0.0.15", ["0.0.14-sm89-cu13", "0.0.15-sm89-cu13"])
              and after16 == ("0.0.16", ["0.0.15-sm89-cu13", "0.0.16-sm89-cu13"]), f"{after15} {after16}")

    both = Release("0.0.14")
    newer = Release("0.0.15")
    both.files.update({k: v for k, v in newer.files.items() if k != "version.json"})
    both.entries["0.0.15"] = dict(newer.entries["0.0.15"], **{"comfy-12": "0.0.99"})
    both.files["version.json"] = json.dumps({"linux": both.entries}).encode()
    picked = {}
    for major in (13, 12):
        with Env(both, torch_major=major) as env:
            qfe.install_engine()
            picked[major] = env.marker("sm89", major)["version"]
    check("per CUDA flavor: cu13 takes 0.0.15, cu12 stays on 0.0.14", picked == {13: "0.0.15", 12: "0.0.14"}, picked)

    # 6) All-or-nothing on missing kernel; offline handling; traversal version; HTTPS only
    rel = Release()
    with Env(rel) as env:
        del rel.files[f"0.0.14/linux/sm89/{kernel_for(89, 13)}"]
        try:
            qfe.install_engine()
            check("missing kernel installs nothing", False)
        except OSError:
            check("missing kernel installs nothing (no marker, no host, no temp)",
                  env.marker() is None and env.read("0.0.14-sm89-cu13", HOSTS[13]) is None and not env.leftovers())
    for installed in (True, False):
        rel = Release()
        with Env(rel) as env:
            if installed:
                qfe.install_engine()
            rel.offline = True
            with contextlib.redirect_stdout(io.StringIO()):
                _join_install(qfe.start_engine_install())
            state, detail = qfe.engine_install_status()
            want = "offline" if installed else "failed"
            check(f"offline with{'' if installed else 'out'} installed engine: status {want}",
                  state == want and "unreachable" in detail)
    rel = Release()
    rel.files["version.json"] = json.dumps({"linux": {"../../../tmp/x": {
        "comfy": "0.0.01", "comfy-12": "0.0.01", "lib": "0.0.99", "lib-12": "0.0.99", "kernel_so": True}}}).encode()
    with Env(rel) as env:
        try:
            qfe.install_engine()
            outcome = "installed!"
        except Exception as e:  # noqa: BLE001
            outcome = f"{type(e).__name__}: {e}"
        check("traversal version refused", outcome.startswith("EngineNotInstallable") and not env.pairs())

    import email.message
    import urllib.response

    def _mk_http_resp(url, code=200, body=b"{}", location=None):
        hdrs = email.message.Message()
        if location is not None:
            hdrs["Location"] = location
        r = urllib.response.addinfourl(io.BytesIO(body), hdrs, url, code=code)
        r.msg = "Found" if code == 302 else "OK"
        return r

    http_calls, https_calls = [], []
    routes = {
        "https://hop.example/downgrade": (302, b"", "http://evil.example/version.json"),
        "https://hop.example/bounce": (302, b"", "http://mitm.example/mid"),
        "http://mitm.example/mid": (302, b"", "https://hop.example/ok"),
        "https://hop.example/good-redir": (302, b"", "https://hop.example/ok"),
        "https://hop.example/rel-redir": (302, b"", "/ok"),
        "https://hop.example/ok": (200, b'{"ok":true}', None),
    }

    class _StubHTTP(urllib.request.HTTPHandler):
        def http_open(self, req):
            http_calls.append(req.full_url)
            code, body, loc = routes.get(req.full_url, (200, b"PLAINTEXT", None))
            return _mk_http_resp(req.full_url, code, body, loc)

    class _StubHTTPS(urllib.request.HTTPSHandler):
        def https_open(self, req):
            https_calls.append(req.full_url)
            code, body, loc = routes.get(req.full_url, (404, b"", None))
            return _mk_http_resp(req.full_url, code, body, loc)

    real_build = urllib.request.build_opener
    try:
        urllib.request.build_opener = lambda *handlers: real_build(_StubHTTP, _StubHTTPS, *handlers)
        refused = []
        for url in ("http://www.modelscope.cn/x", "https://hop.example/downgrade", "https://hop.example/bounce"):
            try:
                qfe._engine_http_open(url)
                refused.append(False)
            except RuntimeError as e:
                refused.append("non-HTTPS" in str(e))
        with qfe._engine_http_open("https://hop.example/good-redir") as r1:
            good_body = r1.read()
        with qfe._engine_http_open("https://hop.example/rel-redir") as r2:
            rel_body = r2.read()
        check("hop-by-hop HTTPS guard blocks http://, 302->http, and 302->http->302->https before any plaintext request",
              refused == [True, True, True] and http_calls == []
              and good_body == b'{"ok":true}' and rel_body == b'{"ok":true}',
              f"refused={refused} http_calls={http_calls}")
    finally:
        urllib.request.build_opener = real_build

    # 7) Tampered marker ignored; float cuda rejected; reinstall-once bound; symlinked bin dir
    rel = Release()
    with Env(rel) as env:
        qfe.install_engine()
        mpath, good = env.path(".engine-sm89-cu13.json"), env.marker()
        tampered = {
            "host": dict(good, host="../../evil.so"),
            "version": dict(good, version="../../x"),
            "kernel_legacy": dict(good, kernel="libquantfunc_kernels_cuda_sm89.so"),
            "kernel_wrong_cu": dict(good, kernel="libquantfunc_kernels-12.so"),
            "kernel_traversal": dict(good, kernel="../libquantfunc_kernels.so"),
            "sha256": dict(good, sha256={good["host"]: "zz", good["kernel"]: "zz"}),
            "set": dict(good, set="sm86"),
            "cuda": dict(good, cuda=12),
            "cuda_float": dict(good, cuda=13.0),
            "sms": dict(good, sms=["89"]),
        }
        followed = [f for f, m in tampered.items()
                    if (open(mpath, "w", encoding="utf-8").write(json.dumps(m)),
                        qfe._read_marker(mpath) is not None)[1]]
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(good, f)
        cu12_mpath = env.path(".engine-sm89-cu12.json")
        with open(cu12_mpath, "w", encoding="utf-8") as f:
            json.dump(dict(good, cuda=12, host=HOSTS[12], kernel=kernel_for(89, 13),
                           sha256={HOSTS[12]: good["sha256"][HOSTS[13]],
                                   kernel_for(89, 13): good["sha256"][good["kernel"]]}), f)
        if qfe._read_marker(cu12_mpath) is not None:
            followed.append("cu12_with_cu13_kernel")
        os.remove(cu12_mpath)
        check("tampered marker (including legacy/mismatched kernel name & float cuda) ignored",
              not followed and qfe._installed_pair() == (mpath, good), followed)

        host = env.path("0.0.14-sm89-cu13", HOSTS[13])
        started = []
        qfe.start_engine_install = lambda device_idx=None: started.append(1)
        msg1 = qfe._engine_load_failed(host, OSError("undefined symbol: qf_x"))
        qfe.install_engine()
        msg2 = qfe._engine_load_failed(host, OSError("undefined symbol: qf_x"))
        qfe._engine_load_ok(host)
        check("reinstall-once bound on failed load, cleared on load ok",
              "re-downloaded once" in msg1 and "again after one re-download" in msg2
              and started == [1] and env.read("0.0.14-sm89-cu13", qfe._ENGINE_REINSTALLED) is None)

    # 8) Sidecar preloads: only non-engine DT_NEEDED sidecar preloaded (never per-SM kernel, other host, stray, backup; no libquantfunc_attention.so)
    d = tempfile.mkdtemp(prefix="qf_preload_")
    k89, k89_12 = kernel_for(89, 13), kernel_for(89, 12)
    needed = {
        HOSTS[13]: [k89, "libopencv_core.so.4.6", HOSTS[12]],
        HOSTS[12]: [k89_12, "libopencv_core.so.4.6", HOSTS[13]],
    }
    for f in (HOSTS[13], HOSTS[12], k89, k89_12, "libopencv_core.so.4.6", "libfoo.so", "libquantfunc-deadbeef.so", "notes.txt"):
        open(os.path.join(d, f), "wb").close()
    o_needed = qfe._elf_needed
    qfe._elf_needed = lambda p: needed.get(os.path.basename(p), [])
    saved = (qfe.resolve_so_path, qfe.assert_toolchain_compatible, qfe._engine_cuda_needs,
             qfe.ctypes.CDLL, qfe._bind, qfe._engine_load_ok, qfe._log_lib_fingerprint)
    calls = []
    try:
        got13 = qfe._sidecar_preloads(os.path.join(d, HOSTS[13]))
        got12 = qfe._sidecar_preloads(os.path.join(d, HOSTS[12]))
        check("sidecar preloads (cu13 & cu12): only non-engine DT_NEEDED sidecar (never kernel plugin, other host, stray)",
              got13 == ["libopencv_core.so.4.6"] and got12 == ["libopencv_core.so.4.6"], (got13, got12))
        qfe.assert_toolchain_compatible = lambda p: None
        qfe._engine_cuda_needs = lambda p: set()
        qfe.ctypes.CDLL = lambda p, mode=0: (calls.append(os.path.basename(p)), object())[1]
        qfe._bind = lambda lib: types.SimpleNamespace(qf_set_log_level=lambda level: None)
        qfe._engine_load_ok = lambda p: None
        qfe._log_lib_fingerprint = lambda lib, p, ident: None
        for host_name in (HOSTS[13], HOSTS[12]):
            calls.clear()
            qfe._LIB = None
            qfe.resolve_so_path = lambda hn=host_name: os.path.join(d, hn)
            qfe.load_lib()
            check(f"load_lib ({host_name}) preloads only non-engine sidecar before host (no libquantfunc_attention.so)",
                  calls == ["libopencv_core.so.4.6", host_name], list(calls))
    finally:
        (qfe.resolve_so_path, qfe.assert_toolchain_compatible, qfe._engine_cuda_needs,
         qfe.ctypes.CDLL, qfe._bind, qfe._engine_load_ok, qfe._log_lib_fingerprint) = saved
        qfe._elf_needed, qfe._LIB = o_needed, None
        shutil.rmtree(d, ignore_errors=True)

    # 9) Two consumer GPU classes in one folder (SM 89 + SM 86), device_idx routing, & single-arch process guard
    rel = Release()
    with Env(rel, sm={0: 89, 1: 86}) as env:
        qfe.install_engine(0)
        qfe.install_engine(1)
        resolved = {}
        for dev in (0, 1):
            qfe._ENGINE_DEVICE = dev
            folder = os.path.dirname(qfe.resolve_so_path())
            resolved[dev] = os.path.basename(folder)
        qfe._ENGINE_DEVICE = 0
        for v in ("0.0.15", "0.0.16"):
            qfe._engine_http_open = Release(v).open
            qfe.install_engine(0)
        check("two consumer GPU classes (sm89 + sm86): isolated folders & updates",
              resolved == {0: "0.0.14-sm89-cu13", 1: "0.0.14-sm86-cu13"}
              and env.marker("sm86")["version"] == "0.0.14" and "0.0.14-sm86-cu13" in env.pairs())
        got_arch = {}
        for dev in (0, 1):
            try:
                qfe.make_create_params(weights_path="/m.safetensors", device_id=dev)
                got_arch[dev] = "ok"
            except RuntimeError as e:
                got_arch[dev] = "refused" if "one ComfyUI per GPU architecture" in str(e) and "SM 8.6" in str(e) else str(e)
        check("second GPU architecture in one process refused loudly", got_arch == {0: "ok", 1: "refused"}, got_arch)

    # 10) Local build lock, dev override, .qf_pair_id verification + real ELF section reader
    rel = Release()
    with Env(rel) as env:
        open(env.path(HOSTS[13]), "wb").write(b"LOCAL-BUILD")
        open(env.path(qfe._ENGINE_LOCAL_BUILD_LOCK), "w", encoding="utf-8").close()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _join_install(qfe.start_engine_install())
        check("locked local build survives install",
              not rel.fetched and qfe.engine_install_status()[0] == "local"
              and qfe.resolve_so_path() == os.path.realpath(env.path(HOSTS[13])))
        os.remove(env.path(qfe._ENGINE_LOCAL_BUILD_LOCK))
        os.environ[qfe._ENV_SO_OVERRIDE] = env.path(HOSTS[13])
        check("dev override takes precedence", qfe.resolve_so_path() == os.path.realpath(env.path(HOSTS[13])))
        os.environ[qfe._ENV_SO_OVERRIDE] = env.path("missing.so")
        try:
            qfe.resolve_so_path()
            missing = "loaded!"
        except RuntimeError as e:
            missing = str(e)
        check("missing override refuses loud", "not a file" in missing)
        os.environ.pop(qfe._ENV_SO_OVERRIDE)
        qfe.install_engine()
        check("unlocked bin/linux/libquantfunc.so ignored in favor of installed pair",
              qfe.resolve_so_path() == os.path.realpath(env.path("0.0.14-sm89-cu13", HOSTS[13])))

    with Env(Release(), torch_major=12, driver_major=13, sm=89) as env:
        open(env.path(qfe._ENGINE_LOCAL_BUILD_LOCK), "w", encoding="utf-8").close()
        open(env.path(HOSTS[13]), "wb").write(b"LOCAL-CU13")
        fallback_12 = qfe.resolve_so_path()
        open(env.path(HOSTS[12]), "wb").write(b"LOCAL-CU12")
        open(env.path(kernel_for(89, 12)), "wb").write(b"LOCAL-KERN-12")
        prefer_12 = qfe.resolve_so_path()
        check("local .dev_lib_lock under CUDA 12 prefers libquantfunc-12.so and falls back to libquantfunc.so when absent",
              fallback_12 == os.path.realpath(env.path(HOSTS[13]))
              and prefer_12 == os.path.realpath(env.path(HOSTS[12])), (fallback_12, prefer_12))

    rel = Release()
    k89 = kernel_for(89, 13)
    other = b"KERNEL-0.0.13-sm89-cu13"
    rel.files[f"0.0.14/linux/sm89/{k89}"] = other
    rel.manifest["linux"][f"sm89/{k89}"] = sha(other)
    rel.publish()
    with Env(rel) as env:
        try:
            qfe.install_engine()
            msg = "installed!"
        except RuntimeError as e:
            msg = str(e)
        check("mismatched .qf_pair_id installs nothing", "not one build" in msg and env.marker() is None and not env.leftovers())

    d = tempfile.mkdtemp(prefix="qf_pairid_")
    pid = "0123456789abcdef0123456789abcdef"
    cases = {"good": (mk_elf({".text": b"\x90" * 8, ".qf_pair_id": pid.encode() + b"\0"}), pid),
             "absent": (mk_elf({".text": b"\x90" * 8}), None),
             "uppercase": (mk_elf({".qf_pair_id": pid.upper().encode() + b"\0"}), None),
             "short": (mk_elf({".qf_pair_id": pid[:31].encode() + b"\0"}), None),
             "not ELF": (b"MZ" + bytes(200), None),
             "empty": (b"", None)}
    got = {}
    try:
        for label, (image, _) in cases.items():
            p = os.path.join(d, label.replace(" ", "_"))
            open(p, "wb").write(image)
            got[label] = qfe._elf_pair_id(p)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    check(".qf_pair_id real ELF section reader", got == {k: v for k, (_, v) in cases.items()}, got)

    # 11) Process-bound loaded_so_path, atomic _LIB_PATH before _LIB, concurrent single load
    rel = Release()
    saved = (qfe.assert_toolchain_compatible, qfe.ctypes.CDLL, qfe._bind, qfe._LIB, qfe._LIB_PATH, qfe._FINGERPRINT_PENDING)
    try:
        with Env(rel) as env:
            qfe.install_engine()
            qfe._LIB = qfe._LIB_PATH = None
            qfe.assert_toolchain_compatible = lambda p: None
            qfe.ctypes.CDLL = lambda p, mode=0: object()
            qfe._bind = lambda lib: types.SimpleNamespace(qf_set_log_level=lambda level: None)
            first = qfe.load_lib()
            loaded = qfe.loaded_so_path()
            qfe._engine_http_open = Release("0.0.15").open
            qfe.install_engine()
            moved = qfe.resolve_so_path()
            check("loaded_so_path stays pinned after background upgrade",
                  loaded == os.path.realpath(env.path("0.0.14-sm89-cu13", HOSTS[13]))
                  and moved == os.path.realpath(env.path("0.0.15-sm89-cu13", HOSTS[13]))
                  and qfe.loaded_so_path() == loaded and qfe.load_lib() is first)
            calls_c, got_c = [], []
            slow_real = qfe.resolve_so_path

            def slow_resolve():
                calls_c.append(threading.get_ident())
                time.sleep(0.1)
                return slow_real()
            qfe.resolve_so_path = slow_resolve
            qfe._LIB = qfe._LIB_PATH = None
            try:
                ts = [threading.Thread(target=lambda: got_c.append(qfe.load_lib())) for _ in range(4)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
            finally:
                qfe.resolve_so_path = slow_real
            check("concurrent first load_lib resolves once and shares one handle",
                  len(calls_c) == 1 and len(got_c) == 4 and all(g is got_c[0] for g in got_c))
    finally:
        (qfe.assert_toolchain_compatible, qfe.ctypes.CDLL, qfe._bind, qfe._LIB, qfe._LIB_PATH,
         qfe._FINGERPRINT_PENDING) = saved

    # 12) Disk tamper detected at resolve_so_path; renamed class + pulled release; CUDA 13 -> 12 -> 13 coexistence
    rel = Release()
    with Env(rel) as env:
        qfe.install_engine()
        with open(env.path("0.0.14-sm89-cu13", kernel_for(89, 13)), "ab") as f:
            f.write(b"x")
        started = []
        qfe.start_engine_install = lambda device_idx=None: started.append(1)
        try:
            qfe.resolve_so_path()
            msg = "loaded!"
        except RuntimeError as e:
            msg = str(e)
        check("disk tamper refused at resolve_so_path, marker dropped, one re-download",
              "does not match the SHA-256" in msg and env.marker() is None and started == [1])

    with Env(Release(), sm=89) as env:
        qfe.install_engine()
        qfe._torch_cuda_major = lambda: 12
        qfe.install_engine()
        both_m = (env.marker("sm89", 13) is not None, env.marker("sm89", 12) is not None)
        qfe._torch_cuda_major = lambda: 13
        before = len(env.release.fetched)
        qfe.install_engine()
        refetched = [f for f in env.release.fetched[before:] if f.endswith(".so")]
        check("torch CUDA 13 -> 12 -> 13 keeps both markers and re-downloads nothing",
              both_m == (True, True) and not refetched
              and os.path.relpath(qfe.resolve_so_path(), env.dir) == f"0.0.14-sm89-cu13/{HOSTS[13]}")

    # 13) Windows: all 4 consumer SMs x CUDA 13/12 (zero-CUDA host + per-SM kernel DLL), O1 PE guard, lock wait/error
    got, want = {}, {}
    for sm, gset in ((75, "sm75"), (86, "sm86"), (89, "sm89"), (120, "sm120a")):
        for major, dll in WIN_DLLS.items():
            rel = WinRelease()
            kname = kernel_for(sm, major, win=True)
            with windows() as ms, Env(rel, torch_major=major, driver_major=13, sm=sm, machine="AMD64") as env:
                m = qfe.install_engine() or {}
                folder = f"0.0.14-{gset}-cu{major}"
                got[(sm, major)] = (m.get("set"), m.get("host"), m.get("kernel"),
                                    os.path.relpath(qfe.resolve_so_path(), env.dir), ms.calls, env.leftovers())
                want[(sm, major)] = (gset, dll, kname, f"{folder}/{dll}", [("nblck", 1), ("unlock", 1)], [])
    check("Windows: all 4 consumer SMs x CUDA 13/12 install zero-CUDA host + per-SM kernel DLL, lock, and resolve",
          got == want, {k: v for k, v in got.items() if v != want[k]} or "all 8")
    with windows(), Env(WinRelease(), torch_major=13, sm=89, machine="AMD64") as env:
        o_pe = qfe._pe_imports
        qfe._pe_imports = lambda p: ["cudart64_13.dll"] if ".quantfunc.dll" in os.path.basename(p) else []
        try:
            qfe.install_engine()
            win_o1 = "installed!"
        except RuntimeError as e:
            win_o1 = str(e)
        finally:
            qfe._pe_imports = o_pe
        check("Windows O1 install guard refuses quantfunc.dll importing cudart64_13.dll",
              "zero CUDA PE imports" in win_o1 and env.marker("sm89", 13) is None and not env.leftovers(), win_o1[:80])
    check("addon regex accepts .so, .dll, and .dylib (macOS ready)",
          all(qfe._ENGINE_ADDON_RE.fullmatch(n) for n in ("libextra.so", "extra.dll", "libextra.dylib")))

    real_sleep, slept = qfe.time.sleep, []
    qfe.time.sleep = slept.append
    try:
        with windows() as ms, Env(WinRelease(), torch_major=12, sm=89, machine="AMD64") as env:
            ms.fail = [_errno.EACCES, _errno.EACCES]
            waited = (qfe.install_engine() or {}).get("set"), [c[0] for c in ms.calls], slept[:]
        with windows() as ms, Env(WinRelease(), torch_major=12, sm=89, machine="AMD64") as env:
            ms.fail = [_errno.EINVAL]
            try:
                qfe.install_engine()
                non_eacces = False
            except OSError as e:
                non_eacces = e.errno == _errno.EINVAL and not env.pairs()
    finally:
        qfe.time.sleep = real_sleep
    check("Windows install lock waits only on EACCES and fails loudly on EINVAL",
          waited == ("sm89", ["nblck", "nblck", "nblck", "unlock"], [1, 1]) and non_eacces)

    # 14) Add-ons: valid, best-effort failure, traversal refusal (walking isolated env.parent, NEVER /tmp!)
    ADD = {13: "libquantfunc_extra.so", 12: "libquantfunc_extra-12.so"}

    def with_addons(rel, names=ADD):
        manifest = json.loads(rel.files[f"{rel.version}/verify.json"])
        for major, name in names.items():
            for gset in CONSUMER_SETS:
                data = f"ADDON-{rel.version}-{gset}-cu{major}".encode()
                rel.files[f"{rel.version}/linux/{gset}/{name}"] = data
                manifest["linux"][f"{gset}/{name}"] = sha(data)
        rel.files[f"{rel.version}/verify.json"] = json.dumps(manifest).encode()
        rel.manifest = manifest
        rel.entries[rel.version].update({"addons": [names[13]], "addons-12": [names[12]]})
        rel.files["version.json"] = json.dumps({"linux": rel.entries}).encode()
        return rel

    rel = with_addons(Release())
    evil = ["../escape.so", "sub/x.so", ".hidden.so", HOSTS[13], HOSTS[12],
            kernel_for(89, 13), kernel_for(89, 12), ".engine-sm89-cu13.json"]
    for name in evil:
        rel.files.setdefault(f"0.0.14/linux/sm89/{name}", b"EVIL")
        rel.manifest["linux"].setdefault(qfe._manifest_key("sm89", name), sha(b"EVIL"))
    rel.publish()
    rel.entries["0.0.14"]["addons"] = evil
    rel.files["version.json"] = json.dumps({"linux": rel.entries}).encode()
    with Env(rel, sm=89) as env:
        m = qfe.install_engine()
        planted = [os.path.join(r, f) for r, _, fs in os.walk(env.parent) for f in fs
                   if f in ("escape.so", "x.so", ".hidden.so")]
        check("add-on: path traversal / hidden / marker refused unfetched; host/kernel untouched",
              m["version"] == "0.0.14" and not planted
              and env.read("0.0.14-sm89-cu13", HOSTS[13]) == rel.files[f"0.0.14/linux/{HOSTS[13]}"]
              and env.read("0.0.14-sm89-cu13", kernel_for(89, 13)) == rel.files[f"0.0.14/linux/sm89/{kernel_for(89, 13)}"],
              (planted, rel.fetched))

    # 15) ModelScope 0.0.18 canonical layout (cu13 & cu12): download paths, SHA-256, zero-CUDA host, real .qf_pair_id, regex/legacy rejection
    good_names = ("libquantfunc_kernels.so", "libquantfunc_kernels-12.so")
    bad_names = ("libquantfunc_kernels_cuda_sm89.so", "../libquantfunc_kernels.so",
                 "sm89/libquantfunc_kernels.so", "libquantfunc_kernels-13.so",
                 "libquantfunc_kernels.dll", "libquantfunc.so", "libquantfunc_attention.so")
    check("_ENGINE_KERNEL_RE accepts canonical 0.0.17/0.0.18 Linux names and rejects unexpected/traversal",
          all(qfe._ENGINE_KERNEL_RE.fullmatch(n) for n in good_names)
          and not any(qfe._ENGINE_KERNEL_RE.fullmatch(n) for n in bad_names))

    rel18 = Release("0.0.18")
    hashes18 = rel18.manifest["linux"]
    try:
        qfe._resolve_kernel_for_set("0.0.18", "sm89", 89, 13, {"sm89/LibQuantFunc_Kernels.so": sha(b"CASE")})
        linux_mixed_case = "accepted!"
    except RuntimeError as e:
        linux_mixed_case = str(e)
    with windows():
        win_ci_13 = qfe._resolve_kernel_for_set(
            "0.0.18", "sm89", 89, 13, {"sm89/LibQuantFunc_Kernels.DLL": sha(b"WIN13")})
        win_ci_12 = qfe._resolve_kernel_for_set(
            "0.0.18", "sm89", 89, 12, {"sm89/LibQuantFunc_Kernels-12.DLL": sha(b"WIN12")})
        try:
            qfe._resolve_kernel_for_set(
                "0.0.18", "sm89", 89, 13,
                {"sm89/libquantfunc_kernels.dll": sha(b"A"), "sm89/LibQuantFunc_Kernels.DLL": sha(b"B")})
            win_dup_ci = "accepted!"
        except RuntimeError as e:
            win_dup_ci = str(e)
    check("_manifest_key & _resolve_kernel_for_set: Linux strict case, Windows case-insensitive & duplicate collision refused",
          qfe._manifest_key("sm89", "libquantfunc.so") == "libquantfunc.so"
          and qfe._manifest_key("sm89", "libquantfunc-12.so") == "libquantfunc-12.so"
          and qfe._manifest_key("sm89", "libquantfunc_kernels.so", 13, hashes18) == "sm89/libquantfunc_kernels.so"
          and qfe._manifest_key("sm89", "libquantfunc_kernels-12.so", 12, hashes18) == "sm89/libquantfunc_kernels-12.so"
          and qfe._resolve_kernel_for_set("0.0.18", "sm89", 89, 13, hashes18) == "libquantfunc_kernels.so"
          and qfe._resolve_kernel_for_set("0.0.18", "sm89", 89, 12, hashes18) == "libquantfunc_kernels-12.so"
          and "no per-SM CUDA kernel plugin" in linux_mixed_case
          and win_ci_13 == "LibQuantFunc_Kernels.DLL"
          and win_ci_12 == "LibQuantFunc_Kernels-12.DLL"
          and "no per-SM CUDA kernel plugin" in win_dup_ci,
          (linux_mixed_case, win_ci_13, win_ci_12, win_dup_ci))

    for major, hname, kname in ((13, "libquantfunc.so", "libquantfunc_kernels.so"),
                                (12, "libquantfunc-12.so", "libquantfunc_kernels-12.so")):
        pid = sha(f"0.0.18-cu{major}".encode())[:32]
        other_pid = sha(f"0.0.18-other-cu{major}".encode())[:32]
        elf_h = mk_elf({".text": b"\x90" * 8, ".qf_pair_id": pid.encode() + b"\0"})
        elf_k = mk_elf({".text": b"\xc3" * 8, ".qf_pair_id": pid.encode() + b"\0"})
        elf_k_mismatch = mk_elf({".text": b"\xc3" * 8, ".qf_pair_id": other_pid.encode() + b"\0"})
        elf_k_nopid = mk_elf({".text": b"\xc3" * 8})

        rel = Release("0.0.18")
        rel.files[f"0.0.18/linux/{hname}"] = elf_h
        rel.files[f"0.0.18/linux/sm89/{kname}"] = elf_k
        rel.manifest["linux"][hname] = sha(elf_h)
        rel.manifest["linux"][f"sm89/{kname}"] = sha(elf_k)
        rel.publish()
        with Env(rel, torch_major=major, driver_major=13, sm=89) as env:
            qfe._elf_pair_id = env._saved["_elf_pair_id"]
            m = qfe.install_engine()
            want_fetched = ["version.json", "0.0.18/verify.json", "0.0.18/linux/sets.json",
                            f"0.0.18/linux/{hname}", f"0.0.18/linux/sm89/{kname}"]
            pair = f"0.0.18-sm89-cu{major}"
            check(f"0.0.18 cu{major}: exact download, SHA-256, zero-CUDA host, real .qf_pair_id",
                  rel.fetched == want_fetched and m["host"] == hname and m["kernel"] == kname
                  and env.read(pair, hname) == elf_h and env.read(pair, kname) == elf_k
                  and qfe._pair_intact(m) and env.marker("sm89", major) == m, rel.fetched)

        for target_label, rel_path in (("host", f"0.0.18/linux/{hname}"), ("kernel", f"0.0.18/linux/sm89/{kname}")):
            bad_rel = Release("0.0.18")
            bad_rel.files[f"0.0.18/linux/{hname}"] = elf_h
            bad_rel.files[f"0.0.18/linux/sm89/{kname}"] = elf_k
            bad_rel.manifest["linux"][hname] = sha(elf_h)
            bad_rel.manifest["linux"][f"sm89/{kname}"] = sha(elf_k)
            bad_rel.publish()
            bad_rel.files[rel_path] = b"TAMPERED-" + target_label.encode()
            with Env(bad_rel, torch_major=major, driver_major=13, sm=89) as env:
                qfe._elf_pair_id = env._saved["_elf_pair_id"]
                try:
                    qfe.install_engine()
                    msg = "installed!"
                except RuntimeError as e:
                    msg = str(e)
                check(f"0.0.18 cu{major}: tampered {target_label} SHA-256 refused",
                      "SHA-256" in msg and env.marker("sm89", major) is None and not env.leftovers(), msg[:80])

        leak_rel = Release("0.0.18")
        leak_rel.files[f"0.0.18/linux/{hname}"] = elf_h
        leak_rel.files[f"0.0.18/linux/sm89/{kname}"] = elf_k
        leak_rel.manifest["linux"][hname] = sha(elf_h)
        leak_rel.manifest["linux"][f"sm89/{kname}"] = sha(elf_k)
        leak_rel.publish()
        with Env(leak_rel, torch_major=major, driver_major=13, sm=89) as env:
            qfe._elf_pair_id = env._saved["_elf_pair_id"]
            qfe._elf_needed = lambda p, hn=hname, m=major: [f"libcudart.so.{m}"] if f".{hn}." in os.path.basename(p) else []
            try:
                qfe.install_engine()
                msg = "installed!"
            except RuntimeError as e:
                msg = str(e)
            check(f"0.0.18 cu{major}: zero-CUDA host guard refuses {hname} linking libcudart.so.{major}",
                  "zero CUDA DT_NEEDED" in msg and env.marker("sm89", major) is None and not env.leftovers(), msg[:80])

        for pid_label, bad_k in (("mismatched", elf_k_mismatch), ("missing", elf_k_nopid)):
            pid_rel = Release("0.0.18")
            pid_rel.files[f"0.0.18/linux/{hname}"] = elf_h
            pid_rel.files[f"0.0.18/linux/sm89/{kname}"] = bad_k
            pid_rel.manifest["linux"][hname] = sha(elf_h)
            pid_rel.manifest["linux"][f"sm89/{kname}"] = sha(bad_k)
            pid_rel.publish()
            with Env(pid_rel, torch_major=major, driver_major=13, sm=89) as env:
                qfe._elf_pair_id = env._saved["_elf_pair_id"]
                try:
                    qfe.install_engine()
                    msg = "installed!"
                except RuntimeError as e:
                    msg = str(e)
                check(f"0.0.18 cu{major}: real ELF {pid_label} .qf_pair_id refused",
                      "not one build" in msg and env.marker("sm89", major) is None and not env.leftovers(), msg[:80])

    # Legacy cu13/ subdir or cross-CUDA fallback must be refused
    legacy_rel = Release("0.0.18")
    del legacy_rel.manifest["linux"]["sm89/libquantfunc_kernels.so"]
    legacy_rel.manifest["linux"]["sm89/cu13/libquantfunc_kernels.so"] = sha(b"LEGACY")
    legacy_rel.publish()
    with Env(legacy_rel, torch_major=13, sm=89) as env:
        try:
            qfe.install_engine()
            msg = "installed!"
        except RuntimeError as e:
            msg = str(e)
        check("legacy sm89/cu13/ manifest key without sm89/libquantfunc_kernels.so refused",
              "no per-SM CUDA kernel plugin" in msg and env.marker("sm89", 13) is None and not env.pairs(), msg[:80])

    cu12_missing_rel = Release("0.0.18")
    del cu12_missing_rel.manifest["linux"]["sm89/libquantfunc_kernels-12.so"]
    cu12_missing_rel.publish()
    with Env(cu12_missing_rel, torch_major=12, sm=89) as env:
        try:
            qfe.install_engine()
            msg = "installed!"
        except RuntimeError as e:
            msg = str(e)
        check("cu12 without sm89/libquantfunc_kernels-12.so refused (never falls back to cu13)",
              "no per-SM CUDA kernel plugin" in msg and env.marker("sm89", 12) is None and not env.pairs(), msg[:80])

    # 16) T1: POSIX umask 0002 fresh install hardens bin_dir, pair, host, kernel, addon, and .part pre-replace
    old_umask = os.umask(0o002)
    try:
        rel_u = Release()
        with Env(rel_u, sm=89) as env:
            os.chmod(env.dir, 0o775)
            pre_modes, real_replace = [], os.replace
            qfe.os.replace = lambda a, b: (
                pre_modes.append((os.path.basename(b), os.stat(a).st_mode & 0o777)),
                real_replace(a, b),
            )[1]
            try:
                qfe.install_engine()
            finally:
                qfe.os.replace = real_replace
            pair = "0.0.14-sm89-cu13"
            kname = kernel_for(89, 13)
            mpath = env.path(".engine-sm89-cu13.json")
            modes_ok = (
                (os.stat(env.dir).st_mode & 0o777) == 0o755
                and (os.stat(env.path(pair)).st_mode & 0o777) == 0o755
                and (os.stat(env.path(pair, HOSTS[13])).st_mode & 0o777) == 0o755
                and (os.stat(env.path(pair, kname)).st_mode & 0o777) == 0o755
                and (os.stat(mpath).st_mode & 0o777) == 0o644
                and pre_modes == [(kname, 0o755), (HOSTS[13], 0o755), (".engine-sm89-cu13.json", 0o644)]
            )
            check("T1a umask 0002: fresh install hardens dir/.so to 0755, marker to 0644, .part pre-replace",
                  modes_ok, pre_modes)

        rel_ua = with_addons(Release())
        with Env(rel_ua, sm=89) as env:
            pre_modes, real_replace = [], os.replace
            qfe.os.replace = lambda a, b: (
                pre_modes.append((os.path.basename(b), os.stat(a).st_mode & 0o777)),
                real_replace(a, b),
            )[1]
            try:
                qfe.install_engine()
            finally:
                qfe.os.replace = real_replace
            addon_mode = os.stat(env.path("0.0.14-sm89-cu13", ADD[13])).st_mode & 0o777
            check("T1b umask 0002: addon .part hardened to 0755 before os.replace",
                  addon_mode == 0o755 and (ADD[13], 0o755) in pre_modes, (addon_mode, pre_modes))
    finally:
        os.umask(old_umask)

    # 17) T2: 0775/0664 self-heal on skip branch & resolve_so_path; _harden_posix_mode O_NOFOLLOW/OSError/nt; symlink fail-closed
    rel_heal = with_addons(Release())
    with Env(rel_heal, sm=89) as env:
        qfe.install_engine()
        pair = "0.0.14-sm89-cu13"
        kname = kernel_for(89, 13)
        mpath = env.path(".engine-sm89-cu13.json")
        os.chmod(env.path(pair), 0o775)
        for fn in (HOSTS[13], kname, ADD[13]):
            os.chmod(env.path(pair, fn), 0o664)
        os.chmod(mpath, 0o664)
        rel_heal.fetched.clear()
        qfe.install_engine()
        skip_healed = (
            not any(f.endswith(".so") for f in rel_heal.fetched)
            and (os.stat(env.path(pair)).st_mode & 0o777) == 0o755
            and all((os.stat(env.path(pair, fn)).st_mode & 0o777) == 0o755 for fn in (HOSTS[13], kname, ADD[13]))
            and (os.stat(mpath).st_mode & 0o777) == 0o644
        )
        check("T2a already-matching skip branch heals 0775/0664 dir/.so/addon/marker with 0 binary fetches",
              skip_healed, rel_heal.fetched)

        os.chmod(env.dir, 0o775)
        os.chmod(env.path(pair), 0o775)
        os.chmod(env.path(pair, HOSTS[13]), 0o664)
        os.chmod(env.path(pair, kname), 0o664)
        qfe.resolve_so_path()
        res_healed = (
            (os.stat(env.dir).st_mode & 0o777) == 0o755
            and (os.stat(env.path(pair)).st_mode & 0o777) == 0o755
            and (os.stat(env.path(pair, HOSTS[13])).st_mode & 0o777) == 0o755
            and (os.stat(env.path(pair, kname)).st_mode & 0o777) == 0o755
        )
        check("T2b offline resolve_so_path heals managed 0775/0664 dirs & .so to 0755", res_healed)

        # _harden_posix_mode O_NOFOLLOW symlink refusal, OSError swallow, Windows no-op
        sym_target = os.path.join(env.parent, "sym_target.so")
        open(sym_target, "wb").write(b"X")
        os.chmod(sym_target, 0o664)
        sym_link = os.path.join(env.parent, "sym_link.so")
        os.symlink(sym_target, sym_link)
        qfe._harden_posix_mode(sym_link, 0o755)
        sym_untouched = (os.stat(sym_target).st_mode & 0o777) == 0o664

        saved_fchmod, saved_name, saved_open = qfe.os.fchmod, qfe.os.name, qfe.os.open
        try:
            qfe.os.fchmod = lambda fd, mode: (_ for _ in ()).throw(PermissionError("ro fs"))
            qfe._harden_posix_mode(sym_target, 0o755)
            ro_ok = True
        except OSError:
            ro_ok = False
        finally:
            qfe.os.fchmod = saved_fchmod

        open_calls = []
        try:
            qfe.os.name = "nt"
            qfe.os.open = lambda *a, **k: (open_calls.append(a), saved_open(*a, **k))[1]
            qfe._harden_posix_mode(sym_target, 0o755)
            nt_noop = open_calls == []
        finally:
            qfe.os.name = saved_name
            qfe.os.open = saved_open
        check("T2c _harden_posix_mode refuses symlink via O_NOFOLLOW, swallows OSError, no-op on Windows",
              sym_untouched and ro_ok and nt_noop, (sym_untouched, ro_ok, nt_noop))

    # Symlinked bin_dir and symlinked pair dir must fail closed
    with Env(Release(), sm=89) as env:
        real_bin = os.path.join(env.parent, "real_bin")
        shutil.move(env.dir, real_bin)
        os.symlink(real_bin, env.dir)
        sym_bin_errs = []
        for fn in (qfe.install_engine, qfe.resolve_so_path):
            try:
                fn()
                sym_bin_errs.append("ok!")
            except RuntimeError as e:
                sym_bin_errs.append("refusing symlinked engine bin" in str(e))
        check("T2d symlinked bin_dir rejected fail-closed by install_engine & resolve_so_path",
              sym_bin_errs == [True, True] and not qfe.has_ready_so(), sym_bin_errs)

    with Env(Release(), sm=89) as env:
        qfe.install_engine()
        pair_path = env.path("0.0.14-sm89-cu13")
        real_pair = os.path.join(env.parent, "real_pair")
        shutil.move(pair_path, real_pair)
        os.symlink(real_pair, pair_path)
        sym_pair_errs = []
        for fn in (qfe.install_engine, qfe.resolve_so_path):
            try:
                fn()
                sym_pair_errs.append("ok!")
            except RuntimeError as e:
                sym_pair_errs.append("refusing symlinked engine pair" in str(e))
        check("T2e symlinked pair dir rejected fail-closed by install_engine & resolve_so_path",
              sym_pair_errs == [True, True] and not qfe.has_ready_so(), sym_pair_errs)

    # 18) T3: Non-managed paths (QF_NATIVE_SO_PATH, .dev_lib_lock, QF_KERNEL_PLUGIN_DIR) ZERO chmod/fchmod
    with Env(Release(), sm=89) as env:
        unm_dir = os.path.join(env.parent, "unmanaged")
        os.makedirs(unm_dir, exist_ok=True)
        os.chmod(unm_dir, 0o775)
        unm_host = os.path.join(unm_dir, HOSTS[13])
        unm_kern = os.path.join(unm_dir, kernel_for(89, 13))
        open(unm_host, "wb").write(b"UNM-HOST")
        open(unm_kern, "wb").write(b"UNM-KERN")
        os.chmod(unm_host, 0o664)
        os.chmod(unm_kern, 0o664)
        os.chmod(env.dir, 0o775)
        dev_host = env.path(HOSTS[13])
        open(dev_host, "wb").write(b"DEV-HOST")
        os.chmod(dev_host, 0o664)
        open(env.path(qfe._ENGINE_LOCAL_BUILD_LOCK), "w", encoding="utf-8").close()

        chmod_log = []
        real_chmod, real_fchmod = qfe.os.chmod, qfe.os.fchmod
        qfe.os.chmod = lambda p, m: (chmod_log.append(("chmod", p, m)), real_chmod(p, m))[1]
        qfe.os.fchmod = lambda fd, m: (chmod_log.append(("fchmod", fd, m)), real_fchmod(fd, m))[1]
        saved_kdir = os.environ.get(qfe._ENV_KERNEL_PLUGIN_DIR)
        try:
            os.environ[qfe._ENV_SO_OVERRIDE] = unm_host
            r1 = qfe.resolve_so_path()
            os.environ.pop(qfe._ENV_SO_OVERRIDE, None)

            r2 = qfe.resolve_so_path()

            os.environ[qfe._ENV_KERNEL_PLUGIN_DIR] = unm_dir
            r3 = qfe._select_kernel_plugin(unm_host)

            unm_ok = (
                chmod_log == []
                and r1 == os.path.realpath(unm_host)
                and r2 == os.path.realpath(dev_host)
                and r3 == unm_kern
                and (os.stat(unm_dir).st_mode & 0o777) == 0o775
                and (os.stat(unm_host).st_mode & 0o777) == 0o664
                and (os.stat(unm_kern).st_mode & 0o777) == 0o664
                and (os.stat(env.dir).st_mode & 0o777) == 0o775
                and (os.stat(dev_host).st_mode & 0o777) == 0o664
            )
            check("T3 unmanaged paths (QF_NATIVE_SO_PATH, .dev_lib_lock, QF_KERNEL_PLUGIN_DIR): 0 chmod/fchmod calls",
                  unm_ok, chmod_log)
        finally:
            qfe.os.chmod, qfe.os.fchmod = real_chmod, real_fchmod
            os.environ.pop(qfe._ENV_SO_OVERRIDE, None)
            if saved_kdir is None:
                os.environ.pop(qfe._ENV_KERNEL_PLUGIN_DIR, None)
            else:
                os.environ[qfe._ENV_KERNEL_PLUGIN_DIR] = saved_kdir

    # 19) T4: Same-version SHA reinstall pre-replace failure, mid-replace rollback, and rollback failure
    def _new_sha_release(tag="v2"):
        r = Release("0.0.14")
        h2 = f"HOST-0.0.14-{tag}-cu13".encode()
        k2 = f"KERNEL-0.0.14-sm89-{tag}-cu13".encode()
        r.files[f"0.0.14/linux/{HOSTS[13]}"] = h2
        r.files[f"0.0.14/linux/sm89/{kernel_for(89, 13)}"] = k2
        r.manifest["linux"][HOSTS[13]] = sha(h2)
        r.manifest["linux"][f"sm89/{kernel_for(89, 13)}"] = sha(k2)
        r.publish()
        return r

    def _no_bak_left(env_obj):
        return not any(".part" in f or ".bak-" in f for _r, _d, fs in os.walk(env_obj.dir) for f in fs)

    with Env(Release("0.0.14"), sm=89) as env:
        qfe.install_engine()
        pair = "0.0.14-sm89-cu13"
        kname = kernel_for(89, 13)
        orig_h = env.read(pair, HOSTS[13])
        orig_k = env.read(pair, kname)
        orig_m = env.marker()

        # Pre-replace failure (tampered SHA on new download) preserves old files & marker verbatim
        bad_r = _new_sha_release("badsha")
        bad_r.files[f"0.0.14/linux/{HOSTS[13]}"] = b"CORRUPTED"
        qfe._engine_http_open = bad_r.open
        try:
            qfe.install_engine()
            msg = "installed!"
        except RuntimeError as e:
            msg = str(e)
        check("T4a same-version pre-replace failure preserves old binaries & marker",
              "existing installation was preserved" in msg
              and env.read(pair, HOSTS[13]) == orig_h
              and env.read(pair, kname) == orig_k
              and env.marker() == orig_m
              and _no_bak_left(env), msg[:100])

        # Mid-replace failure: os.replace(host) fails AFTER os.replace(kernel) succeeded
        qfe._engine_http_open = _new_sha_release("midhost").open
        real_replace = qfe.os.replace

        def _fail_host_replace(src, dst):
            if os.path.basename(dst) == HOSTS[13] and ".part" in os.path.basename(src):
                raise OSError("injected host replace failure")
            return real_replace(src, dst)

        qfe.os.replace = _fail_host_replace
        try:
            qfe.install_engine()
            msg = "installed!"
        except RuntimeError as e:
            msg = str(e)
        finally:
            qfe.os.replace = real_replace
        check("T4b mid-replace os.replace(host) failure rolls back kernel & preserves marker",
              "existing installation was preserved" in msg
              and env.read(pair, HOSTS[13]) == orig_h
              and env.read(pair, kname) == orig_k
              and env.marker() == orig_m
              and _no_bak_left(env), msg[:100])

        # Mid-replace failure: _claim() fails AFTER BOTH kernel and host were replaced
        qfe._engine_http_open = _new_sha_release("midclaim").open
        real_write = qfe._engine_write_file

        def _fail_claim_write(path, data):
            if os.path.basename(path) == ".engine-sm89-cu13.json":
                raise OSError("injected claim failure")
            return real_write(path, data)

        qfe._engine_write_file = _fail_claim_write
        try:
            qfe.install_engine()
            msg = "installed!"
        except RuntimeError as e:
            msg = str(e)
        finally:
            qfe._engine_write_file = real_write
        check("T4c mid-replace _claim() failure rolls back both kernel & host and preserves marker",
              "existing installation was preserved" in msg
              and env.read(pair, HOSTS[13]) == orig_h
              and env.read(pair, kname) == orig_k
              and env.marker() == orig_m
              and _no_bak_left(env), msg[:100])

        # Rollback failure: os.replace(host) fails AND restoring kernel backup fails -> removes stale marker
        qfe._engine_http_open = _new_sha_release("rbfail").open

        def _fail_host_and_rollback(src, dst):
            if os.path.basename(dst) == HOSTS[13] and ".part" in os.path.basename(src):
                raise OSError("injected host replace failure")
            if ".bak-" in os.path.basename(src):
                raise OSError("injected rollback failure")
            return real_replace(src, dst)

        qfe.os.replace = _fail_host_and_rollback
        try:
            qfe.install_engine()
            msg = "installed!"
        except RuntimeError as e:
            msg = str(e)
        finally:
            qfe.os.replace = real_replace
        check("T4d rollback failure deletes stale marker and never claims preserved",
              "rollback failed; stale marker removed" in msg
              and "existing installation was preserved" not in msg
              and env.marker() is None
              and _no_bak_left(env), msg[:120])

    # Fresh-install (bak is None) mid-replace failure removes half-installed kernel and leaves no marker
    with Env(Release("0.0.14"), sm=89) as env:
        real_replace = qfe.os.replace

        def _fail_fresh_host_replace(src, dst):
            if os.path.basename(dst) == HOSTS[13] and ".part" in os.path.basename(src):
                raise OSError("injected fresh host replace failure")
            return real_replace(src, dst)

        qfe.os.replace = _fail_fresh_host_replace
        try:
            qfe.install_engine()
            fresh_rb = False
        except OSError as e:
            fresh_rb = "injected fresh host replace failure" in str(e)
        finally:
            qfe.os.replace = real_replace
        pair_dir = env.path("0.0.14-sm89-cu13")
        leftover = os.listdir(pair_dir) if os.path.isdir(pair_dir) else []
        check("T4e fresh-install mid-replace removes half-installed kernel & leaves no marker",
              fresh_rb and leftover == [] and env.marker() is None and _no_bak_left(env),
              (fresh_rb, leftover, env.marker()))

    # 20) T8: Real (unmocked) has_ready_so(device_id) 6+ paths & headless heterogeneous multi-GPU load_lib(device_id=1)
    with Env(Release(), sm={0: 89, 1: 86}) as env:
        qfe.install_engine(1)  # only sm86 installed
        saved_lib = (qfe._LIB, qfe._LIB_PATH, qfe._FINGERPRINT_PENDING, qfe._ENGINE_DEVICE,
                     qfe._engine_platform, qfe.assert_toolchain_compatible, qfe._engine_cuda_needs,
                     qfe.ctypes.CDLL, qfe._bind, qfe._engine_load_ok, qfe._log_lib_fingerprint)
        try:
            qfe._LIB = object()
            p_loaded = qfe.has_ready_so() is True
            qfe._LIB = None

            os.environ[qfe._ENV_SO_OVERRIDE] = env.path("does_not_exist.so")
            p_override_miss = qfe.has_ready_so(device_id=1) is False
            os.environ[qfe._ENV_SO_OVERRIDE] = env.path("0.0.14-sm86-cu13", HOSTS[13])
            p_override_hit = qfe.has_ready_so(device_id=1) is True
            os.environ.pop(qfe._ENV_SO_OVERRIDE, None)

            orig_plat = qfe._engine_platform
            qfe._engine_platform = lambda: None
            p_plat_none_miss = qfe.has_ready_so(device_id=1) is False
            local_p = env.path(HOSTS[13])
            open(local_p, "wb").write(b"LOCAL")
            p_plat_none_hit = qfe.has_ready_so(device_id=1) is True
            os.remove(local_p)
            qfe._engine_platform = orig_plat

            lock_p = env.path(qfe._ENGINE_LOCAL_BUILD_LOCK)
            open(lock_p, "w", encoding="utf-8").close()
            p_lock_miss = qfe.has_ready_so(device_id=1) is False
            local_p = env.path(HOSTS[13])
            open(local_p, "wb").write(b"LOCAL")
            p_lock_hit = qfe.has_ready_so(device_id=1) is True
            os.remove(local_p)
            os.remove(lock_p)

            p_dev0 = qfe.has_ready_so(device_id=0) is False
            p_dev1 = qfe.has_ready_so(device_id=1) is True

            # Tamper sm86 kernel -> has_ready_so returns False, NEVER deletes marker, NEVER calls start_engine_install
            started_probe = []
            qfe.start_engine_install = lambda device_idx=None: started_probe.append(device_idx)
            k86_path = env.path("0.0.14-sm86-cu13", kernel_for(86, 13))
            orig_k86 = open(k86_path, "rb").read()
            open(k86_path, "wb").write(b"TAMPERED")
            p_tamper = qfe.has_ready_so(device_id=1) is False
            p_no_side_fx = env.marker("sm86") is not None and started_probe == []
            open(k86_path, "wb").write(orig_k86)

            # OSError inside probe -> False
            orig_bindir = qfe._engine_bin_dir
            qfe._engine_bin_dir = lambda: (_ for _ in ()).throw(OSError("io fail"))
            p_oserr = qfe.has_ready_so(device_id=1) is False
            qfe._engine_bin_dir = orig_bindir

            check("T8a real has_ready_so all paths (loaded, override-miss/hit, plat-none, lock-miss/hit, multi-gpu, tamper, OSError)",
                  all((p_loaded, p_override_miss, p_override_hit, p_plat_none_miss, p_plat_none_hit,
                       p_lock_miss, p_lock_hit, p_dev0, p_dev1, p_tamper, p_no_side_fx, p_oserr)),
                  (p_loaded, p_override_miss, p_override_hit, p_plat_none_miss, p_plat_none_hit,
                   p_lock_miss, p_lock_hit, p_dev0, p_dev1, p_tamper, p_no_side_fx, p_oserr))

            # Headless heterogeneous multi-GPU load_lib(device_id=1) when _ENGINE_DEVICE == 0
            qfe._ENGINE_DEVICE = 0
            qfe._LIB = qfe._LIB_PATH = None
            qfe.assert_toolchain_compatible = lambda p: None
            qfe._engine_cuda_needs = lambda p: set()
            loaded_paths = []
            qfe.ctypes.CDLL = lambda p, mode=0: (loaded_paths.append(p), object())[1]
            qfe._bind = lambda lib: types.SimpleNamespace(qf_set_log_level=lambda level: None)
            qfe._engine_load_ok = lambda p: None
            qfe._log_lib_fingerprint = lambda lib, p, ident: None

            qfe.load_lib(device_id=1)
            dev1_ok = (
                qfe._ENGINE_DEVICE == 1
                and len(loaded_paths) == 1
                and os.path.basename(os.path.dirname(loaded_paths[0])) == "0.0.14-sm86-cu13"
            )

            # Failure rollback of _ENGINE_DEVICE when load_lib(device_id=1) raises
            qfe._LIB = qfe._LIB_PATH = None
            qfe._ENGINE_DEVICE = 0
            qfe.ctypes.CDLL = lambda p, mode=0: (_ for _ in ()).throw(OSError("dlopen fail"))
            try:
                qfe.load_lib(device_id=1)
                rb_ok = False
            except RuntimeError:
                rb_ok = qfe._ENGINE_DEVICE == 0
            check("T8b multi-GPU load_lib(device_id=1) resolves sm86 and rolls back _ENGINE_DEVICE on error",
                  dev1_ok and rb_ok, (dev1_ok, rb_ok, loaded_paths, qfe._ENGINE_DEVICE))
        finally:
            os.environ.pop(qfe._ENV_SO_OVERRIDE, None)
            (qfe._LIB, qfe._LIB_PATH, qfe._FINGERPRINT_PENDING, qfe._ENGINE_DEVICE,
             qfe._engine_platform, qfe.assert_toolchain_compatible, qfe._engine_cuda_needs,
             qfe.ctypes.CDLL, qfe._bind, qfe._engine_load_ok, qfe._log_lib_fingerprint) = saved_lib

    # 21) T19: Exact 35-symbol qf_capi.py standalone & package re-export verification
    capi_spec = importlib.util.spec_from_file_location("qf_capi_check", os.path.join(_PLUGIN, "qf_capi.py"))
    capi_mod = importlib.util.module_from_spec(capi_spec)
    capi_spec.loader.exec_module(capi_mod)
    pkg_tmp = types.ModuleType("qf_pkg_capi_test")
    pkg_tmp.__path__ = [_PLUGIN]
    sys.modules[pkg_tmp.__name__] = pkg_tmp
    try:
        pkg_qfe = __import__(f"{pkg_tmp.__name__}.qf_engine", fromlist=["qf_engine"])
        all_35 = list(capi_mod.__all__)
        check("T19 qf_capi exposes exact 35 symbols across standalone & package qf_engine plus library_identity",
              len(all_35) == 35
              and all(hasattr(qfe, s) and hasattr(pkg_qfe, s) for s in all_35)
              and callable(getattr(qfe, "library_identity", None))
              and callable(getattr(pkg_qfe, "library_identity", None)),
              (len(all_35), [s for s in all_35 if not (hasattr(qfe, s) and hasattr(pkg_qfe, s))]))
    finally:
        sys.modules.pop(pkg_tmp.__name__, None)
        sys.modules.pop(f"{pkg_tmp.__name__}.qf_engine", None)
        sys.modules.pop(f"{pkg_tmp.__name__}.qf_capi", None)

    # 22) T20: Same-version marker matches remote verify.json, but file on disk is corrupted OR missing -> install_engine self-heals;
    #     AND symlinked pair_have dir is refused BEFORE _sha256_of is ever called.
    rel_heal = Release("0.0.14")
    with Env(rel_heal, sm=89) as env:
        qfe.install_engine()
        pair = "0.0.14-sm89-cu13"
        orig_k = env.read(pair, kernel_for(89, 13))
        # (a) Corrupt kernel on disk while marker JSON still matches remote verify.json
        with open(env.path(pair, kernel_for(89, 13)), "wb") as f:
            f.write(b"CORRUPTED-ON-DISK")
        rel_heal.fetched.clear()
        qfe.install_engine()
        healed_corrupt = (
            env.read(pair, kernel_for(89, 13)) == orig_k
            and qfe._pair_intact(env.marker("sm89", 13))
            and f"0.0.14/linux/sm89/{kernel_for(89, 13)}" in rel_heal.fetched
        )
        # (b) Delete kernel on disk while marker JSON still matches remote verify.json
        os.remove(env.path(pair, kernel_for(89, 13)))
        rel_heal.fetched.clear()
        qfe.install_engine()
        healed_missing = (
            env.read(pair, kernel_for(89, 13)) == orig_k
            and qfe._pair_intact(env.marker("sm89", 13))
            and f"0.0.14/linux/sm89/{kernel_for(89, 13)}" in rel_heal.fetched
        )
        # (c) Symlinked pair_have dir MUST be refused BEFORE _sha256_of reads any file
        real_pair = env.path(pair)
        moved_pair = os.path.join(env.dir, "elsewhere-pair")
        shutil.move(real_pair, moved_pair)
        os.symlink(moved_pair, real_pair)
        sha_calls = []
        orig_sha = qfe._sha256_of
        def _spy_sha(p):
            sha_calls.append(p)
            return orig_sha(p)
        qfe._sha256_of = _spy_sha
        sym_refused = False
        try:
            qfe.install_engine()
        except RuntimeError as e:
            sym_refused = "symlink" in str(e).lower()
        finally:
            qfe._sha256_of = orig_sha
        check("T20 install_engine self-heals corrupted/missing disk files and refuses symlink pair before hashing",
              healed_corrupt and healed_missing and sym_refused and len(sha_calls) == 0,
              (healed_corrupt, healed_missing, sym_refused, sha_calls))

    # 23) T21: Windows locked-DLL rename-aside, stale cleanup, non-Windows/fresh-install guards, and fail-closed rollback
    def _win_rel_bytes(hdata, kdata, addons=None):
        wr = WinRelease("0.0.18")
        wr.files["0.0.18/windows/quantfunc.dll"] = hdata
        wr.files["0.0.18/windows/sm120a/libquantfunc_kernels.dll"] = kdata
        vdoc = json.loads(wr.files["0.0.18/verify.json"])
        vdoc["win32"]["quantfunc.dll"] = sha(hdata)
        vdoc["win32"]["sm120a/libquantfunc_kernels.dll"] = sha(kdata)
        if addons:
            verdoc = json.loads(wr.files["version.json"])
            verdoc["win32"]["0.0.18"]["addons"] = list(addons.keys())
            wr.files["version.json"] = json.dumps(verdoc).encode()
            for aname, adata in addons.items():
                wr.files[f"0.0.18/windows/sm120a/{aname}"] = adata
                vdoc["win32"][f"sm120a/{aname}"] = sha(adata)
        wr.files["0.0.18/verify.json"] = json.dumps(vdoc).encode()
        return wr

    # (a) Non-Windows PermissionError on os.replace MUST NOT rename aside
    rel_lin_perm = Release("0.0.14")
    with Env(rel_lin_perm, sm=89) as env:
        qfe.install_engine()
        pair = "0.0.14-sm89-cu13"
        os.remove(env.path(pair, kernel_for(89, 13)))
        real_replace = os.replace
        def lin_perm(src, dst):
            if os.path.basename(dst) == HOSTS[13]:
                raise PermissionError(13, "POSIX EACCES", dst)
            return real_replace(src, dst)
        qfe.os.replace = lin_perm
        lin_perm_refused = False
        try:
            qfe.install_engine()
        except (PermissionError, RuntimeError) as e:
            lin_perm_refused = "PermissionError" in str(e) or isinstance(e, PermissionError)
        finally:
            qfe.os.replace = real_replace
        no_stale_linux = not any(".stale-" in fn for fn in os.listdir(env.path(pair)))
        check("T21a Linux PermissionError does not trigger Windows rename-aside",
              lin_perm_refused and no_stale_linux, (lin_perm_refused, no_stale_linux))

    with windows():
        # (b) Windows fresh install (dst does not exist) PermissionError MUST NOT rename aside
        wrel_fresh = _win_rel_bytes(b"MZ-HOST-F", b"MZ-KERN-F")
        with Env(wrel_fresh, sm=120, torch_major=13, driver_major=13, machine="AMD64") as env:
            real_replace = os.replace
            def fresh_perm(src, dst):
                if os.path.basename(dst) == "libquantfunc_kernels.dll":
                    raise PermissionError(13, "WinError 5 fresh", dst)
                return real_replace(src, dst)
            qfe.os.replace = fresh_perm
            fresh_perm_raised = False
            try:
                qfe.install_engine()
            except PermissionError:
                fresh_perm_raised = True
            finally:
                qfe.os.replace = real_replace
            check("T21b Windows fresh-install PermissionError (no existing dst) propagates without rename-aside",
                  fresh_perm_raised, fresh_perm_raised)

        # (c) Positive rename-aside with locked stale survivor, legitimate add-on containing '.stale-',
        #     stale symlink/dir immunity, skip-branch & update-branch cleanup, and locked-stale OSError tolerance
        wrel1 = _win_rel_bytes(b"MZ-HOST-V1", b"MZ-KERN-V1", addons={"helper.stale-1.dll": b"ADDON-OK"})
        with Env(wrel1, sm=120, torch_major=13, driver_major=13, machine="AMD64") as env:
            qfe.install_engine()
            pair = "0.0.18-sm120a-cu13"
            pair_dir = env.path(pair)
            marker_path = os.path.join(env.dir, ".engine-sm120a-cu13.json")

            wrel2 = _win_rel_bytes(b"MZ-HOST-V2", b"MZ-KERN-V2", addons={"helper.stale-1.dll": b"ADDON-OK"})
            qfe._engine_http_open = wrel2.open
            real_replace = os.replace
            real_remove = os.remove
            seen_perm = {"count": 0}
            locked_stales = set()

            def win_locked_replace(src, dst):
                if os.path.exists(dst) and os.path.basename(dst) in ("quantfunc.dll", "libquantfunc_kernels.dll"):
                    seen_perm["count"] += 1
                    raise PermissionError(13, "WinError 5 locked DLL", dst)
                return real_replace(src, dst)

            def block_stale_remove(p):
                if ".stale-" in os.path.basename(p):
                    locked_stales.add(os.path.basename(p))
                    raise PermissionError(13, "WinError 5 still mapped", p)
                return real_remove(p)

            qfe.os.replace = win_locked_replace
            qfe.os.remove = block_stale_remove
            try:
                qfe.install_engine()
            finally:
                qfe.os.replace = real_replace
                qfe.os.remove = real_remove
            # Create an unrelated symlink & dir matching stale regex to confirm _cleanup_stale ignores both
            victim_file = os.path.join(env.dir, "outside-victim.bin")
            with open(victim_file, "wb") as vf:
                vf.write(b"KEEP")
            stale_sym = os.path.join(pair_dir, ".quantfunc.dll.stale-999-abcdef01")
            os.symlink(victim_file, stale_sym)
            stale_sub = os.path.join(pair_dir, ".quantfunc.dll.stale-998-abcdef02")
            os.makedirs(stale_sub, exist_ok=True)

            win_rename_ok = (
                seen_perm["count"] == 2
                and len(locked_stales) == 2
                and env.read(pair, "quantfunc.dll") == b"MZ-HOST-V2"
                and env.read(pair, "libquantfunc_kernels.dll") == b"MZ-KERN-V2"
                and env.read(pair, "helper.stale-1.dll") == b"ADDON-OK"
                and qfe._pair_intact(env.marker("sm120a", 13))
            )

            # Skip branch while stale files still locked -> tolerates OSError; then unlocked -> cleans only real stale files
            qfe.os.remove = block_stale_remove
            try:
                qfe.install_engine()
            finally:
                qfe.os.remove = real_remove
            still_has_locked = all(os.path.exists(os.path.join(pair_dir, s)) for s in locked_stales)
            qfe.install_engine()
            skip_cleaned_ok = (
                still_has_locked
                and not any(os.path.exists(os.path.join(pair_dir, s)) for s in locked_stales)
                and os.path.exists(os.path.join(pair_dir, "helper.stale-1.dll"))
                and os.path.exists(victim_file)
                and os.path.islink(stale_sym)
                and os.path.isdir(stale_sub)
            )
            os.unlink(stale_sym)
            os.rmdir(stale_sub)

            # Update branch cleans pre-existing unlocked stale file
            pre_stale = os.path.join(pair_dir, ".quantfunc.dll.stale-777-deadbeef")
            with open(pre_stale, "wb") as sf:
                sf.write(b"OLD-STALE")
            wrel2b = _win_rel_bytes(b"MZ-HOST-V2B", b"MZ-KERN-V2B", addons={"helper.stale-1.dll": b"ADDON-OK"})
            qfe._engine_http_open = wrel2b.open
            qfe.install_engine()
            update_cleaned_ok = (
                not os.path.exists(pre_stale)
                and env.read(pair, "quantfunc.dll") == b"MZ-HOST-V2B"
                and env.read(pair, "helper.stale-1.dll") == b"ADDON-OK"
            )

            # (d) Negative rollback 1: kernel renamed aside, then host fails -> restores kernel V2B
            wrel3 = _win_rel_bytes(b"MZ-HOST-V3", b"MZ-KERN-V3")
            qfe._engine_http_open = wrel3.open
            def win_fail_host_after_kernel_stale(src, dst):
                if os.path.basename(dst) == "libquantfunc_kernels.dll" and os.path.exists(dst):
                    raise PermissionError(13, "WinError 5 locked kernel", dst)
                if os.path.basename(dst) == "quantfunc.dll":
                    raise OSError("disk full on host")
                return real_replace(src, dst)
            qfe.os.replace = win_fail_host_after_kernel_stale
            rb1_ok = False
            try:
                qfe.install_engine()
            except RuntimeError as e:
                rb1_ok = "existing installation was preserved" in str(e)
            finally:
                qfe.os.replace = real_replace
            rb1_ok = rb1_ok and env.read(pair, "quantfunc.dll") == b"MZ-HOST-V2B" and env.read(pair, "libquantfunc_kernels.dll") == b"MZ-KERN-V2B"

            # (e) Negative rollback 2: BOTH kernel & host renamed aside, then _claim fails -> restores BOTH
            orig_claim = qfe._claim
            def fail_claim(*args, **kwargs):
                raise OSError("simulated claim failure")
            qfe.os.replace = win_locked_replace
            qfe._claim = fail_claim
            rb2_ok = False
            try:
                qfe.install_engine()
            except RuntimeError as e:
                rb2_ok = "existing installation was preserved" in str(e)
            finally:
                qfe.os.replace = real_replace
                qfe._claim = orig_claim
            rb2_ok = rb2_ok and env.read(pair, "quantfunc.dll") == b"MZ-HOST-V2B" and env.read(pair, "libquantfunc_kernels.dll") == b"MZ-KERN-V2B"

            # (f) Negative rollback 3: rename dst->stale succeeds, retry replace fails, outer rollback restores dst
            retry_calls = {"k": 0}
            def fail_retry_replace(src, dst):
                if os.path.basename(dst) == "libquantfunc_kernels.dll":
                    retry_calls["k"] += 1
                    if retry_calls["k"] == 1:
                        raise PermissionError(13, "WinError 5 locked", dst)
                    raise OSError("AV locked .part on retry")
                return real_replace(src, dst)
            qfe.os.replace = fail_retry_replace
            rb3_ok = False
            try:
                qfe.install_engine()
            except RuntimeError as e:
                rb3_ok = "existing installation was preserved" in str(e)
            finally:
                qfe.os.replace = real_replace
            rb3_ok = rb3_ok and env.read(pair, "libquantfunc_kernels.dll") == b"MZ-KERN-V2B" and os.path.exists(marker_path)

            # (g) Negative rollback 4 (fail-closed): rename dst->stale succeeds, retry replace fails, AND rename(stale,dst) fails
            #     MUST set rollback_failed=True, remove marker, and NEVER claim "existing installation was preserved"!
            retry_calls["k"] = 0
            real_rename = os.rename
            def fail_restore_rename(src, dst):
                if ".stale-" in os.path.basename(src):
                    raise OSError("simulated restore rename failure")
                return real_rename(src, dst)
            qfe.os.replace = fail_retry_replace
            qfe.os.rename = fail_restore_rename
            rb4_ok = False
            try:
                qfe.install_engine()
            except RuntimeError as e:
                msg = str(e)
                rb4_ok = ("rollback failed" in msg) and ("existing installation was preserved" not in msg)
            finally:
                qfe.os.replace = real_replace
                qfe.os.rename = real_rename
            rb4_ok = rb4_ok and (not os.path.exists(marker_path))

            check("T21c-g Windows rename-aside, stale cleanup, add-on preservation & fail-closed rollback all pass",
                  win_rename_ok and skip_cleaned_ok and update_cleaned_ok and rb1_ok and rb2_ok and rb3_ok and rb4_ok,
                  (win_rename_ok, skip_cleaned_ok, update_cleaned_ok, rb1_ok, rb2_ok, rb3_ok, rb4_ok))

    # 24) C09: Plugin 0.0.09 version.json & Engine 0.0.19 release matrix (0.0.08/0.0.09 pick 0.0.19; 0.0.07 rejects)
    shipped_ver = {}
    for plat_dir in ("linux", "windows"):
        vpath = os.path.join(_PLUGIN, "bin", plat_dir, "version.json")
        with open(vpath, encoding="utf-8") as vf:
            shipped_ver[plat_dir] = json.load(vf)
    check("C09 shipped bin/{linux,windows}/version.json equal {'comfy': '0.0.09'}",
          shipped_ver == {"linux": {"comfy": "0.0.09"}, "windows": {"comfy": "0.0.09"}}, shipped_ver)

    stanza17 = {"comfy": "0.0.07", "comfy-12": "0.0.07", "sdk": "0.0.07", "sdk-12": "0.0.07",
                "lib": "0.0.17", "lib-12": "0.0.17", "kernel_so": True}
    stanza18 = {"comfy": "0.0.08", "comfy-12": "0.0.08", "sdk": "0.0.08", "sdk-12": "0.0.08",
                "lib": "0.0.18", "lib-12": "0.0.18", "kernel_so": True}
    stanza19 = {"comfy": "0.0.08", "comfy-12": "0.0.08", "sdk": "0.0.08", "sdk-12": "0.0.08",
                "lib": "0.0.19", "lib-12": "0.0.19", "kernel_so": True}
    matrix_versions = {"0.0.17": stanza17, "0.0.18": stanza18, "0.0.19": stanza19}
    matrix_no17 = {"0.0.18": stanza18, "0.0.19": stanza19}
    pick_ok = True
    pick_details = {}
    for pver, want_with_17, want_no_17 in (("0.0.09", "0.0.19", "0.0.19"),
                                           ("0.0.08", "0.0.19", "0.0.19"),
                                           ("0.0.07", "0.0.17", None)):
        for major in (13, 12):
            got17 = qfe._engine_pick_version(matrix_versions, pver, major)
            got_no17 = qfe._engine_pick_version(matrix_no17, pver, major)
            pick_details[(pver, major)] = (got17, got_no17)
            if got17 != want_with_17 or got_no17 != want_no_17:
                pick_ok = False
    check("C09 _engine_pick_version matrix: 0.0.08/0.0.09 pick 0.0.19, 0.0.07 rejects 0.0.19/0.0.18 (cu12/cu13)",
          pick_ok, pick_details)

    # Linux & Windows end-to-end 0.0.19 install/upgrade/refusal across 0.0.09 / 0.0.08 / 0.0.07
    rel18_19 = Release("0.0.18", plugin_req="0.0.08")
    rel19 = Release("0.0.19", plugin_req="0.0.08")
    rel18_19.files.update({k: v for k, v in rel19.files.items() if k != "version.json"})
    rel18_19.entries = {"0.0.18": dict(stanza18), "0.0.19": dict(stanza19)}
    rel18_19.files["version.json"] = json.dumps({"linux": rel18_19.entries}).encode()
    e2e_lin = {}
    for major in (13, 12):
        for pver in ("0.0.09", "0.0.08"):
            with Env(rel18_19, torch_major=major, driver_major=major, sm=89, plugin_version=pver) as env:
                qfe.install_engine()
                e2e_lin[(pver, major)] = env.marker("sm89", major)["version"]
        with Env(rel18_19, torch_major=major, driver_major=major, sm=89, plugin_version="0.0.07") as env:
            try:
                qfe.install_engine()
                e2e_lin[("0.0.07", major)] = "installed!"
            except RuntimeError as exc:
                e2e_lin[("0.0.07", major)] = type(exc).__name__
    check("C09 Linux install_engine: 0.0.09 & 0.0.08 install 0.0.19, 0.0.07 refuses (cu12/cu13)",
          all(e2e_lin[(p, m)] == "0.0.19" for p in ("0.0.09", "0.0.08") for m in (13, 12))
          and all(e2e_lin[("0.0.07", m)] == "EngineNotInstallable" for m in (13, 12)), e2e_lin)

    with windows():
        wr18_19 = WinRelease("0.0.18", plugin_req="0.0.08")
        wr18_19.add("0.0.19", plugin_req="0.0.08")
        wr18_19.entries = {"0.0.18": dict(stanza18), "0.0.19": dict(stanza19)}
        wr18_19.files["version.json"] = json.dumps({"linux": {}, "win32": wr18_19.entries}).encode()
        e2e_win = {}
        for major in (13, 12):
            for pver in ("0.0.09", "0.0.08"):
                with Env(wr18_19, torch_major=major, driver_major=major, sm=120,
                         plugin_version=pver, machine="AMD64") as env:
                    qfe.install_engine()
                    e2e_win[(pver, major)] = env.marker("sm120a", major)["version"]
            with Env(wr18_19, torch_major=major, driver_major=major, sm=120,
                     plugin_version="0.0.07", machine="AMD64") as env:
                try:
                    qfe.install_engine()
                    e2e_win[("0.0.07", major)] = "installed!"
                except RuntimeError as exc:
                    e2e_win[("0.0.07", major)] = type(exc).__name__
        check("C09 Windows install_engine: 0.0.09 & 0.0.08 install 0.0.19, 0.0.07 refuses (cu12/cu13)",
              all(e2e_win[(p, m)] == "0.0.19" for p in ("0.0.09", "0.0.08") for m in (13, 12))
              and all(e2e_win[("0.0.07", m)] == "EngineNotInstallable" for m in (13, 12)), e2e_win)

    print("ENGINE_INSTALL:", "PASS" if bad == 0 else f"FAIL ({bad} wrong)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
