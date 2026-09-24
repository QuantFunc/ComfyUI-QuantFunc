#!/usr/bin/env python3
"""Option C (user 2026-09-24 「在原生加载器里实现」): the plugin's engine installer, resolver and preload rule, end to end
against a FAKE release server.

qf_engine is loaded standalone (no ComfyUI, no torch, no GPU, no network). The ONE network entry (_engine_http_open),
torch's CUDA major, the driver's CUDA major, the GPU's SM, the CPU and the ELF DT_NEEDED reader are replaced, and the
plugin's bin/ is a temp dir. The fake release has the layout the engine ships:
  version.json                    {"linux": {"<key>": {"comfy", "comfy-12", "lib", "lib-12", "kernel_so"}}}
  <ver>/verify.json               {"schema", "linux": {"<host>": sha256, "<set>/<kernel>": sha256, "sets.json": sha256}}
  <ver>/linux/sets.json           {"schema": 1, "sets": {"<set>": [sm, ...]}}   the ONLY source of the GPU classes
  <ver>/linux/<host>              one host per CUDA major (every file of a release's major carries one .qf_pair_id)
  <ver>/linux/<set>/<kernel>      one kernel per (GPU class, CUDA major)
Installed: bin/linux/<ver>-<set>-cu<major>/{host, kernel}, then (LAST) the marker bin/linux/.engine-<set>-cu<major>.json.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import types

_PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("qfe_install_test", os.path.join(_PLUGIN, "qf_engine.py"))
qfe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qfe)

HOSTS = {13: "libquantfunc.so", 12: "libquantfunc-12.so"}
KERNELS = {13: "libquantfunc_kernels.so", 12: "libquantfunc_kernels-12.so"}
SETS = {"consumer": [75, 86, 89, 120], "server": [80, 90, 100, 103]}
bad = 0


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
    """A fake QuantFunc/Plugin repo. files: relpath -> bytes; every fetch is recorded."""

    def __init__(self, version="0.0.13", plugin_req="0.0.07", sets=SETS):
        self.files, self.fetched, self.offline = {}, [], False
        self.version = version
        body = {}
        for major in (13, 12):
            h = f"HOST-{version}-cu{major}".encode()
            self.files[f"{version}/linux/{HOSTS[major]}"] = h
            body[HOSTS[major]] = sha(h)
            for gset in SETS:
                k = f"KERNEL-{version}-{gset}-cu{major}".encode()
                self.files[f"{version}/linux/{gset}/{KERNELS[major]}"] = k
                body[f"{gset}/{KERNELS[major]}"] = sha(k)
        self.manifest = {"schema": 1, "linux": body}      # verify_manifest.py's shape: no version key (it is the path)
        if sets is not None:
            self.set_sets(json.dumps({"schema": 1, "sets": sets}).encode())
        self.publish()
        self.entries = {
            "0.0.12": {"comfy": "0.0.06", "comfy-12": "0.0.06", "lib": "0.0.12", "lib-12": "0.0.12"},   # monolithic
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


def fake_needed(path):
    """DT_NEEDED of a fake host: its kernel for the CUDA major it was built for."""
    try:
        data = open(path, "rb").read()
    except OSError:
        return []
    for major in (13, 12):
        if data.startswith(b"HOST-") and data.endswith(f"-cu{major}".encode()):
            return [KERNELS[major], f"libcudart.so.{major}"]
    return []


def fake_pair_id(path):
    """.qf_pair_id of a fake file: one id per (release, CUDA major), as the ship build stamps host and kernels."""
    try:
        parts = open(path, "rb").read().decode().split("-")   # HOST-<ver>-cu<major> / KERNEL-<ver>-<set>-cu<major>
    except (OSError, UnicodeDecodeError):
        return None
    return sha(f"{parts[1]}-{parts[-1]}".encode())[:32] if parts[0] in ("HOST", "KERNEL") and len(parts) > 2 else None


def mk_elf(sections):
    """A minimal ELF64 little-endian image: header, the named sections, .shstrtab and the section headers — what a
    section reader needs, nothing more."""
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
    shdrs = bytes(64)                               # the null section
    for n in names:
        sh = bytearray(64)
        struct.pack_into("<II", sh, 0, name_at[n], 3 if n == ".shstrtab" else 1)
        struct.pack_into("<QQ", sh, 0x18, offs[n], len(sections[n]) if n in sections else len(shstr))
        shdrs += bytes(sh)
    return bytes(head) + body + shdrs


_STUBBED = ("_engine_http_open", "_torch_cuda_major", "_driver_cuda_major", "_gpu_sm", "_engine_bin_dir", "_elf_needed",
            "_elf_pair_id", "start_engine_install", "install_engine", "_ENGINE_DEVICE")


class Env:
    """A machine: a temp plugin bin/ + the stubs. sm: one SM, or {device index: SM}. Restores everything on exit."""

    def __init__(self, release, torch_major=13, driver_major=13, sm=89, plugin_version="0.0.07", machine="x86_64"):
        self.release = release
        self.dir = tempfile.mkdtemp(prefix="qf_engine_install_")
        with open(os.path.join(self.dir, "version.json"), "w") as f:
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
        shutil.rmtree(self.dir, ignore_errors=True)

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def read(self, *parts):
        p = self.path(*parts)
        return open(p, "rb").read() if os.path.isfile(p) else None

    def marker(self, gset="consumer", major=13):
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
    print("=== engine installer + resolver + preload rule (option C) ===")
    # 1) GPU class x CUDA major: the pair lands in its OWN folder, kernel renamed before host, the marker LAST
    for sm, gset in ((89, "consumer"), (120, "consumer"), (80, "server"), (100, "server")):
        for major in (13, 12):
            rel = Release()
            with Env(rel, torch_major=major, driver_major=13, sm=sm) as env:
                order, real_replace = [], os.replace
                qfe.os.replace = lambda a, b: (order.append(os.path.basename(b)), real_replace(a, b))[1]
                try:
                    qfe.install_engine()
                finally:
                    qfe.os.replace = real_replace
                pair = f"0.0.13-{gset}-cu{major}"
                h, k = rel.files[f"0.0.13/linux/{HOSTS[major]}"], rel.files[f"0.0.13/linux/{gset}/{KERNELS[major]}"]
                want = {"version": "0.0.13", "set": gset, "cuda": major, "sms": SETS[gset], "host": HOSTS[major],
                        "kernel": KERNELS[major], "sha256": {HOSTS[major]: sha(h), KERNELS[major]: sha(k)}}
                m = env.marker(gset, major)
                ok = (env.read(pair, HOSTS[major]) == h and env.read(pair, KERNELS[major]) == k and m == want
                      and order == [KERNELS[major], HOSTS[major], f".engine-{gset}-cu{major}.json"]
                      and env.read(HOSTS[major]) is None and not env.leftovers())
                check(f"SM {sm} x CUDA {major}: {gset} pair in its own folder, kernel before host, marker LAST", ok,
                      f"order={order} marker={m}")
    # 2) nothing installed outside the published classes / for an old driver / on a non-x86_64 CPU (G2)
    rel = Release()
    with Env(rel, sm=61) as env:
        try:
            qfe.install_engine()
            check("an SM outside every class installs nothing", False, "installed!")
        except qfe.EngineNotInstallable as e:
            check("an SM outside every class installs nothing, and says why", "SM 61" in str(e) and not env.pairs()
                  and not any(f.startswith("0.0.13/linux/") and not f.endswith("sets.json") for f in rel.fetched), str(e)[:80])
    rel = Release()
    with Env(rel, torch_major=13, driver_major=12) as env:
        try:
            qfe.install_engine()
            check("a driver older than torch's CUDA installs nothing", False, "installed!")
        except qfe.EngineNotInstallable as e:
            check("a driver older than torch's CUDA installs nothing (the driver only CHECKS)",
                  "update the driver" in str(e) and not rel.fetched, str(e)[:80])
    rel = Release()
    with Env(rel, torch_major=None, driver_major=12) as env:
        qfe.install_engine()
        check("torch CUDA unknown: the driver's major chooses the host", env.marker("consumer", 12) is not None
              and env.marker("consumer", 13) is None)
    rel = Release()
    with Env(rel, machine="aarch64") as env:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _join_install(qfe.start_engine_install())
        state, detail = qfe.engine_install_status()
        check("G2: an aarch64 machine installs nothing and says why, once (no download loop)",
              state == "unavailable" and "aarch64" in detail and not rel.fetched and not env.pairs()
              and out.getvalue().count("[qf_native]") == 1, f"{state}: {detail[:70]}")
    # 3) S1: sets.json is the ONLY source of the GPU classes — absent or malformed: refused loud; it decides the class
    rel = Release(sets=None)
    with Env(rel) as env:
        try:
            qfe.install_engine()
            check("a release without sets.json installs nothing", False, "installed!")
        except RuntimeError as e:
            check("a release without sets.json installs nothing, and says so", "publishes no sets.json" in str(e)
                  and not env.pairs() and env.marker() is None, str(e)[:80])
    rel = Release()
    rel.set_sets(json.dumps({"schema": 1, "sets": {"../x": [89]}}).encode())
    with Env(rel) as env:
        try:
            qfe.install_engine()
            check("a sets.json class that is not a plain name is refused", False, "installed!")
        except RuntimeError as e:
            check("a sets.json class that is not a plain name is refused (it would name a folder and a URL segment)",
                  "GPU-class map" in str(e) and not env.pairs(), str(e)[:80])
    rel = Release()
    rel.set_sets(json.dumps({"schema": 1, "sets": {"consumer": [75, 86, 120], "server": [80, 89, 90]}}).encode())
    with Env(rel, sm=89) as env:
        qfe.install_engine()
        check("a published sets.json decides the class (SM 89 moved to server here)",
              env.marker("server") is not None and env.marker("consumer") is None)
    rel.set_sets(rel.files["0.0.13/linux/sets.json"] + b" ", hashed=False)
    with Env(rel, sm=89) as env:
        try:
            qfe.install_engine()
            check("a sets.json that does not match its hash is refused", False, "installed!")
        except RuntimeError as e:
            check("a sets.json that does not match its hash is refused", "sets.json" in str(e), str(e)[:60])
    # 4) present + matching: nothing re-downloaded, the release documents still read (remote-first)
    rel = Release()
    with Env(rel) as env:
        qfe.install_engine()
        rel.fetched.clear()
        qfe.install_engine()
        check("present + matching: nothing re-downloaded, the release documents still read",
              rel.fetched == ["version.json", "0.0.13/verify.json", "0.0.13/linux/sets.json"], rel.fetched)
    # 5) a newer release goes into its own folder and the marker flips last; a corrupt download changes nothing; the
    #    pair a new one replaces stays (a process may be loading it), the one before that goes
    old = Release("0.0.13")
    with Env(old) as env:
        qfe.install_engine()
        new = Release("0.0.14")
        new.files[f"0.0.14/linux/{HOSTS[13]}"] = b"HOST-tampered-cu13"
        qfe._engine_http_open = new.open
        try:
            qfe.install_engine()
            check("a corrupt download is refused", False, "installed!")
        except RuntimeError as e:
            check("a corrupt download is refused; the installed pair and its marker are untouched",
                  "SHA-256" in str(e) and env.marker()["version"] == "0.0.13"
                  and env.read("0.0.13-consumer-cu13", HOSTS[13]) == old.files[f"0.0.13/linux/{HOSTS[13]}"]
                  and not env.leftovers(), str(e)[:80])
        qfe._engine_http_open = Release("0.0.14").open
        qfe.install_engine()
        after14 = (env.marker()["version"], env.pairs())
        qfe._engine_http_open = Release("0.0.15").open
        qfe.install_engine()
        after15 = (env.marker()["version"], env.pairs())
        check("a newer release replaces the installed one; the replaced pair stays, the one before it goes",
              after14 == ("0.0.14", ["0.0.13-consumer-cu13", "0.0.14-consumer-cu13"])
              and after15 == ("0.0.15", ["0.0.14-consumer-cu13", "0.0.15-consumer-cu13"]), f"{after14} {after15}")
    # 5b) per CUDA flavor (the classic rule): 0.0.14's CUDA 12 build needs a newer plugin, so CUDA 12 stays on 0.0.13
    both = Release("0.0.13")
    newer = Release("0.0.14")
    both.files.update({k: v for k, v in newer.files.items() if k != "version.json"})
    both.entries["0.0.14"] = dict(newer.entries["0.0.14"], **{"comfy-12": "0.0.99"})
    both.files["version.json"] = json.dumps({"linux": both.entries}).encode()
    picked = {}
    for major in (13, 12):
        with Env(both, torch_major=major) as env:
            qfe.install_engine()
            picked[major] = env.marker("consumer", major)["version"]
    check("per CUDA flavor: CUDA 13 takes 0.0.14, CUDA 12 (its build needs plugin 0.0.99) stays on 0.0.13",
          picked == {13: "0.0.14", 12: "0.0.13"}, picked)
    # 6) all-or-nothing: the host downloads, the kernel is missing -> no marker, no host in place, no temp left
    rel = Release()
    with Env(rel) as env:
        del rel.files[f"0.0.13/linux/consumer/{KERNELS[13]}"]
        try:
            qfe.install_engine()
            check("a missing kernel installs nothing", False, "installed!")
        except OSError:
            check("a missing kernel installs nothing (no marker, host not in place, no temp left)",
                  env.marker() is None and env.read("0.0.13-consumer-cu13", HOSTS[13]) is None and not env.leftovers())
    # 7) offline: an installed pair stays in use (status offline); none installed -> status failed with the reason
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
            check(f"offline with{'' if installed else 'out'} an installed engine: status {want}",
                  state == want and "unreachable" in detail and (env.marker() is not None) == installed,
                  f"{state}: {detail[:60]}")
    # 8) never a server-supplied path: a traversal "version" is never picked; a kernel name must be a QuantFunc kernel
    rel = Release()
    # valid lib/lib-12 so ONLY the key (the path segment) can stop it
    rel.files["version.json"] = json.dumps({"linux": {"../../../tmp/x": {
        "comfy": "0.0.01", "comfy-12": "0.0.01", "lib": "0.0.99", "lib-12": "0.0.99", "kernel_so": True}}}).encode()
    with Env(rel) as env:
        try:
            qfe.install_engine()
            outcome = "installed!"
        except Exception as e:  # noqa: BLE001 — the property below is what matters, not the exception type
            outcome = f"{type(e).__name__}: {e}"
        check("a traversal version is never picked (nothing fetched below the release root)",
              not any(".." in f for f in rel.fetched) and not env.pairs()
              and outcome.startswith("EngineNotInstallable"), f"{outcome[:60]} fetched={rel.fetched}")
    rel = Release()
    with Env(rel) as env:
        qfe._elf_needed = lambda p: ["libevil.so", "libcudart.so.13"]
        try:
            qfe.install_engine()
            check("a host naming no QuantFunc kernel installs nothing", False, "installed!")
        except RuntimeError as e:
            check("a host naming no QuantFunc kernel installs nothing", "kernel" in str(e) and env.marker() is None,
                  str(e)[:80])
    # 9) HTTPS only: a plain-HTTP URL and a redirect to plain HTTP are both refused by the real network entry
    import urllib.request
    real_urlopen = urllib.request.urlopen
    try:
        urllib.request.urlopen = lambda url, timeout=None: _Resp(b"{}", "http://mirror.example/version.json")
        refused = []
        for url in ("http://www.modelscope.cn/x", qfe._ENGINE_BASE_URL + "/version.json"):
            try:
                qfe._engine_http_open(url)
            except RuntimeError as e:
                refused.append("non-HTTPS" in str(e))
        check("a plain-HTTP URL and a redirect to plain HTTP are refused", refused == [True, True], refused)
    finally:
        urllib.request.urlopen = real_urlopen
    # 10) LOW: every marker field is checked before it names a path — a tampered marker is ignored, never followed
    rel = Release()
    with Env(rel) as env:
        qfe.install_engine()
        mpath, good = env.path(".engine-consumer-cu13.json"), env.marker()
        tampered = {
            "host": dict(good, host="../../evil.so"),
            "version": dict(good, version="../../x"),
            "kernel": dict(good, kernel="/etc/libquantfunc_kernels.so"),
            "sha256": dict(good, sha256={good["host"]: "zz", good["kernel"]: "zz"}),
            "set": dict(good, set="server"),
            "cuda": dict(good, cuda=12),
            "sms": dict(good, sms=["89"]),
        }
        followed = []
        for field, m in tampered.items():
            with open(mpath, "w") as f:
                json.dump(m, f)
            if qfe._read_marker(mpath) is not None or qfe._installed_pair() != (None, None):
                followed.append(field)
        with open(mpath, "w") as f:
            json.dump(good, f)
        check("a tampered marker (any field) is ignored; the intact one is used", not followed
              and qfe._installed_pair() == (mpath, good), f"followed={followed}")
    # 11) the reinstall-once bound: a pair that will not LOAD is re-downloaded once per release, then only reported
    rel = Release()
    with Env(rel) as env:
        qfe.install_engine()
        host = env.path("0.0.13-consumer-cu13", HOSTS[13])
        started = []
        qfe.start_engine_install = lambda device_idx=None: started.append(1)
        msg1 = qfe._engine_load_failed(host, OSError("undefined symbol: qf_kernel_x"))
        flag = env.read("0.0.13-consumer-cu13", qfe._ENGINE_REINSTALLED)
        first = "re-downloaded once" in msg1 and started == [1] and flag == b"0.0.13" and env.marker() is None
        qfe.install_engine()                       # the re-download (same version: into the same folder, flag kept)
        msg2 = qfe._engine_load_failed(host, OSError("undefined symbol: qf_kernel_x"))
        second = "again after one re-download" in msg2 and started == [1] and env.marker() is not None
        qfe._engine_load_ok(host)
        check("a pair that will not load: one re-download per release, then only the error (flag cleared by a load)",
              first and second and env.read("0.0.13-consumer-cu13", qfe._ENGINE_REINSTALLED) is None,
              f"first={first} second={second}")
        link = env.dir + "-link"                   # a symlinked plugin dir (common for custom_nodes)
        os.symlink(env.dir, link)
        try:
            qfe._engine_bin_dir = lambda: link
            msg3 = qfe._engine_load_failed(os.path.realpath(host), OSError("undefined symbol: qf_kernel_x"))
            check("the bound also applies through a symlinked plugin dir", "re-downloaded once" in msg3
                  and started == [1, 1], msg3[:80])
        finally:
            os.remove(link)
    # 12) V1 + A1: ONLY the engine's own DT_NEEDED closure over its folder is preloaded — never another engine build,
    #     either CUDA major's host, a kernel or a backup copy; asserted on the rule AND through load_lib
    d = tempfile.mkdtemp(prefix="qf_preload_")
    needed = {HOSTS[13]: [KERNELS[13], "libquantfunc_attention.so", "libcudart.so.13", HOSTS[12]],
              "libquantfunc_attention.so": ["libopencv_core.so.4.6"]}
    for f in (HOSTS[13], HOSTS[12], KERNELS[13], "libquantfunc_attention.so", "libopencv_core.so.4.6", "libfoo.so",
              "libquantfunc-4c938478.so", "libquantfunc_attention.so.prod-bak", "notes.txt"):
        open(os.path.join(d, f), "wb").close()
    o_needed = qfe._elf_needed
    qfe._elf_needed = lambda p: needed.get(os.path.basename(p), [])
    saved = (qfe.resolve_so_path, qfe.assert_toolchain_compatible, qfe.ctypes.CDLL, qfe._bind, qfe._engine_load_ok,
             qfe._log_lib_fingerprint)
    calls = []
    try:
        got = qfe._sidecar_preloads(os.path.join(d, HOSTS[13]))
        check("only the engine's DT_NEEDED closure is preloaded (not a stray build, the other host, a kernel, a backup)",
              got == ["libopencv_core.so.4.6", "libquantfunc_attention.so"], got)
        qfe._LIB = None
        qfe.resolve_so_path = lambda: os.path.join(d, HOSTS[13])
        qfe.assert_toolchain_compatible = lambda p: None
        qfe.ctypes.CDLL = lambda p, mode=0: (calls.append(os.path.basename(p)), object())[1]
        qfe._bind = lambda lib: types.SimpleNamespace(quantfunc_set_log_level=lambda level: None)
        qfe._engine_load_ok = lambda p: None
        qfe._log_lib_fingerprint = lambda lib, p: None
        qfe.load_lib()
        check("through load_lib: the stray engine build and the other CUDA major's host are never dlopened",
              sorted(calls[:-1]) == ["libopencv_core.so.4.6", "libquantfunc_attention.so"] and calls[-1] == HOSTS[13],
              calls)
    finally:
        (qfe.resolve_so_path, qfe.assert_toolchain_compatible, qfe.ctypes.CDLL, qfe._bind, qfe._engine_load_ok,
         qfe._log_lib_fingerprint) = saved
        qfe._elf_needed, qfe._LIB = o_needed, None
        shutil.rmtree(d, ignore_errors=True)
    # 13) V2/G3: two GPU classes share one plugin folder — separate pairs and markers, the resolver picks by the
    #     device's SM, and updating one class never touches the other's pair
    rel = Release()
    with Env(rel, sm={0: 89, 1: 80}) as env:
        qfe.install_engine(0)
        qfe.install_engine(1)
        resolved = {}
        for dev in (0, 1):
            qfe._ENGINE_DEVICE = dev
            folder = os.path.dirname(qfe.resolve_so_path())
            resolved[dev] = (os.path.basename(folder), open(os.path.join(folder, KERNELS[13]), "rb").read())
        qfe._ENGINE_DEVICE = 0
        server_before = env.read("0.0.13-server-cu13", KERNELS[13])
        for v in ("0.0.14", "0.0.15"):
            qfe._engine_http_open = Release(v).open
            qfe.install_engine(0)
        check("two GPU classes in one folder: each device loads its own class; updating one leaves the other's pair",
              resolved == {0: ("0.0.13-consumer-cu13", rel.files[f"0.0.13/linux/consumer/{KERNELS[13]}"]),
                           1: ("0.0.13-server-cu13", rel.files[f"0.0.13/linux/server/{KERNELS[13]}"])}
              and env.marker("server")["version"] == "0.0.13" and env.read("0.0.13-server-cu13", KERNELS[13]) == server_before
              and "0.0.13-server-cu13" in env.pairs(), env.pairs())
    # 14) V2: an update killed between its two renames (B's probe) — the marker still names the old pair, whose files are
    #     untouched, so the loader gets old host + old kernel, never a cross-release pair
    old = Release("0.0.13")
    with Env(old) as env:
        qfe.install_engine()
        qfe._engine_http_open = Release("0.0.14").open
        real_replace = os.replace

        def killed(a, b):
            if os.path.basename(b) == HOSTS[13] and "0.0.14" in b:
                raise OSError("killed between the two renames (simulated)")
            return real_replace(a, b)
        qfe.os.replace = killed
        try:
            qfe.install_engine()
        except OSError:
            pass
        finally:
            qfe.os.replace = real_replace
        try:
            so = qfe.resolve_so_path()
        except RuntimeError as e:   # a marker naming the torn new pair would be refused here
            so = f"REFUSED: {e}"
        check("an update killed between the renames: the old pair (both files) is what loads",
              (env.marker() or {}).get("version") == "0.0.13"
              and so == os.path.realpath(env.path("0.0.13-consumer-cu13", HOSTS[13]))
              and env.read("0.0.13-consumer-cu13", KERNELS[13]) == old.files[f"0.0.13/linux/consumer/{KERNELS[13]}"], so)
    # 15) V3: a KNOWN mismatch never loads.
    #   (a) the release publishes other bytes for the installed version and the re-download fails: the marker goes
    #       BEFORE the download; the status says why; nothing loads (never "the installed engine stays in use")
    rel = Release()
    with Env(rel) as env:
        qfe.install_engine()
        repub = Release()
        repub.files[f"0.0.13/linux/{HOSTS[13]}"] = b"HOST-0.0.13-republished-cu13"
        repub.manifest["linux"][HOSTS[13]] = sha(repub.files[f"0.0.13/linux/{HOSTS[13]}"])
        repub.publish()

        def flaky(url):
            if url.endswith(".so"):
                raise OSError("timed out")      # the documents arrive, the big download does not
            return repub.open(url)
        qfe._engine_http_open = flaky
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _join_install(qfe.start_engine_install())
        state, detail = qfe.engine_install_status()
        try:
            qfe.resolve_so_path()
            loaded = True
        except RuntimeError:
            loaded = False
        check("a known mismatch whose re-download fails: marker dropped first, status failed with the reason, not loaded",
              state == "failed" and "no longer matches" in detail and env.marker() is None and not loaded
              and "stays in use" not in out.getvalue(), f"{state}: {detail[:70]}")
    #   (b) a file changed on disk since install: the resolver refuses it, drops the marker, starts ONE re-download
    rel = Release()
    with Env(rel) as env:
        qfe.install_engine()
        with open(env.path("0.0.13-consumer-cu13", KERNELS[13]), "ab") as f:
            f.write(b"x")
        started = []
        qfe.start_engine_install = lambda device_idx=None: started.append(1)
        try:
            qfe.resolve_so_path()
            msg = "loaded!"
        except RuntimeError as e:
            msg = str(e)
        check("a pair changed on disk since install is refused (never loaded), its marker dropped, one re-download",
              "does not match the SHA-256" in msg and env.marker() is None and started == [1], msg[:80])
    # 16) G4: the install — and a failed load's re-download — follow ComfyUI's device, never device 0
    rel = Release()
    with Env(rel, sm={0: 89, 1: 80}) as env:
        _join_install(qfe.start_engine_install(1))
        first = env.marker("server") is not None and env.marker("consumer") is None
        host = env.path("0.0.13-server-cu13", HOSTS[13])
        with contextlib.redirect_stdout(io.StringIO()):
            qfe._engine_load_failed(host, OSError("undefined symbol: qf_kernel_x"))
            deadline = time.time() + 10
            while env.marker("server") is None and time.time() < deadline:
                time.sleep(0.05)
        check("G4: install and the load-failure re-download use ComfyUI's device (SM 80 = server), not device 0",
              first and env.marker("server") is not None and env.marker("consumer") is None, env.pairs())
    # 17) A3: a local build (.dev_lib_lock) or the dev override: the installer touches nothing and says so in ONE line,
    #     and that library is what loads. Without the lock, a library in bin/linux/ (a previous updater's copy) is NOT
    #     loaded — the installed pair is. A missing override is an error, never a fall-through.
    rel = Release()
    with Env(rel) as env:
        with open(env.path(HOSTS[13]), "wb") as f:
            f.write(b"LOCAL-BUILD")
        open(env.path(qfe._ENGINE_LOCAL_BUILD_LOCK), "w").close()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _join_install(qfe.start_engine_install())
        state = qfe.engine_install_status()[0]
        check("a locked local build survives an install: nothing fetched or written, one line, it is what loads",
              not rel.fetched and state == "local" and out.getvalue().count("install skipped") == 1
              and env.read(HOSTS[13]) == b"LOCAL-BUILD" and not env.pairs() and env.marker() is None
              and qfe.resolve_so_path() == os.path.realpath(env.path(HOSTS[13])), f"{state} {out.getvalue()[:80]!r}")
        os.remove(env.path(qfe._ENGINE_LOCAL_BUILD_LOCK))
        os.environ[qfe._ENV_SO_OVERRIDE] = env.path(HOSTS[13])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _join_install(qfe.start_engine_install())
        check("the dev override: the installer touches nothing, one line, the override is what loads",
              not rel.fetched and qfe.engine_install_status()[0] == "local" and out.getvalue().count("install skipped") == 1
              and qfe.resolve_so_path() == os.path.realpath(env.path(HOSTS[13])))
        os.environ[qfe._ENV_SO_OVERRIDE] = env.path("no-such-engine.so")
        try:
            qfe.resolve_so_path()
            missing = "loaded something!"
        except RuntimeError as e:
            missing = str(e)
        check("an override naming a missing file is an error, never a fall-through", "not a file" in missing, missing[:70])
        os.environ.pop(qfe._ENV_SO_OVERRIDE)
        qfe.install_engine()
        check("without the lock, an unlocked library in bin/linux/ is ignored: the installed pair loads",
              qfe.resolve_so_path() == os.path.realpath(env.path("0.0.13-consumer-cu13", HOSTS[13]))
              and env.read(HOSTS[13]) == b"LOCAL-BUILD")
    # 18) one build: host and kernel must carry the same .qf_pair_id before either is put in place — a kernel from another
    #     build (its SHA-256 published, so only the id can tell) or a host with no id installs nothing
    rel = Release()
    other = b"KERNEL-0.0.12-consumer-cu13"             # hashes fine, but another release's id
    rel.files[f"0.0.13/linux/consumer/{KERNELS[13]}"] = other
    rel.manifest["linux"][f"consumer/{KERNELS[13]}"] = sha(other)
    rel.publish()
    with Env(rel) as env:
        try:
            qfe.install_engine()
            msg = "installed!"
        except RuntimeError as e:
            msg = str(e)
        check("a host and a kernel of different builds install nothing (no marker, nothing in place, no temp left)",
              "not one build" in msg and env.marker() is None and env.read("0.0.13-consumer-cu13", HOSTS[13]) is None
              and env.read("0.0.13-consumer-cu13", KERNELS[13]) is None and not env.leftovers(), msg[:80])
    rel = Release()
    with Env(rel) as env:
        qfe._elf_pair_id = lambda p: None if os.path.basename(p) == f".{HOSTS[13]}.part" else fake_pair_id(p)
        try:
            qfe.install_engine()
            msg = "installed!"
        except RuntimeError as e:
            msg = str(e)
        check("a host without a pair id installs nothing", "not one build" in msg and env.marker() is None, msg[:80])
    # the REAL reader: the section's bytes up to the NUL, on an ELF image; None for anything else
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
            with open(p, "wb") as f:
                f.write(image)
            got[label] = qfe._elf_pair_id(p)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    check("the .qf_pair_id reader: the id of an ELF that carries one; None when absent, malformed, not ELF or empty",
          got == {k: v for k, (_, v) in cases.items()}, got)
    # 19) G-1 (tests-07 re-CR round 2): the library a process loaded stays ITS library. A newer pair marked while it
    #     runs moves resolve_so_path's answer, but loaded_so_path() (what the pipeline cache keys on) stays, and a later
    #     load_lib() neither re-resolves nor re-hashes the pair.
    rel = Release()
    saved = (qfe.assert_toolchain_compatible, qfe.ctypes.CDLL, qfe._bind, qfe._LIB, qfe._LIB_PATH, qfe._FINGERPRINT_PENDING)
    try:
        with Env(rel) as env:
            qfe.install_engine()
            qfe._LIB = qfe._LIB_PATH = None
            qfe.assert_toolchain_compatible = lambda p: None
            qfe.ctypes.CDLL = lambda p, mode=0: object()
            qfe._bind = lambda lib: types.SimpleNamespace(quantfunc_set_log_level=lambda level: None)
            first = qfe.load_lib()
            loaded = qfe.loaded_so_path()
            qfe._engine_http_open = Release("0.0.14").open
            qfe.install_engine()                       # a background update marks a newer pair
            moved = qfe.resolve_so_path()
            resolves = []
            real_resolve = qfe.resolve_so_path
            qfe.resolve_so_path = lambda: resolves.append(1) or real_resolve()
            again = qfe.load_lib()
            qfe.resolve_so_path = real_resolve
            check("G-1: after a newer pair is marked, the loaded library stays the process's key; nothing re-resolves",
                  loaded == os.path.realpath(env.path("0.0.13-consumer-cu13", HOSTS[13]))
                  and moved == os.path.realpath(env.path("0.0.14-consumer-cu13", HOSTS[13]))
                  and qfe.loaded_so_path() == loaded and again is first and resolves == [],
                  f"loaded={loaded} moved={moved} resolves={len(resolves)}")
            # 20) the path is published BEFORE the library (tests-07 re-CR round 3, B): a concurrent first-load caller
            #     that sees _LIB must see _LIB_PATH too, else its cache key is (None, ...) and one extra pipeline is
            #     built. Traced line by line through a real first load_lib().
            torn = []

            def tracer(frame, event, arg):
                if frame.f_code is not qfe.load_lib.__code__:
                    return None
                if event in ("line", "return") and qfe._LIB is not None and qfe._LIB_PATH is None:
                    torn.append(frame.f_lineno)
                return tracer
            qfe._LIB = qfe._LIB_PATH = None
            sys.settrace(tracer)
            try:
                qfe.load_lib()
            finally:
                sys.settrace(None)
            check("the loaded path is published before the library: no line of load_lib shows _LIB without _LIB_PATH",
                  torn == [] and qfe._LIB is not None and qfe._LIB_PATH is not None, f"torn at lines {torn}")
            # 21) one first load per process (tests-07 re-CR round 4, B): threads racing the first load_lib() while the
            #     resolver is slow (and could answer differently after a marker switch) resolve ONCE and share one library.
            import threading
            calls, got = [], []
            slow_real = qfe.resolve_so_path

            def slow_resolve():
                calls.append(threading.get_ident())
                time.sleep(0.2)
                return slow_real()
            qfe.resolve_so_path = slow_resolve
            qfe._LIB = qfe._LIB_PATH = None
            try:
                ts = [threading.Thread(target=lambda: got.append(qfe.load_lib())) for _ in range(4)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
            finally:
                qfe.resolve_so_path = slow_real
            check("concurrent first loads: one resolve, one library for every caller",
                  len(calls) == 1 and len(got) == 4 and all(g is got[0] for g in got), f"resolves={len(calls)} libs={len(set(map(id, got)))}")
    finally:
        (qfe.assert_toolchain_compatible, qfe.ctypes.CDLL, qfe._bind, qfe._LIB, qfe._LIB_PATH,
         qfe._FINGERPRINT_PENDING) = saved
    print("ENGINE_INSTALL:", "PASS" if bad == 0 else f"FAIL ({bad} wrong)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
