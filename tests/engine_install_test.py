#!/usr/bin/env python3
"""Option C (user 2026-09-24 「在原生加载器里实现」): the plugin's engine installer, resolver and preload rule, end to end
against a FAKE release server.

qf_engine is loaded standalone (no ComfyUI, no torch, no GPU, no network). The ONE network entry (_engine_http_open),
torch's CUDA major, the driver's CUDA major, the GPU's SM, the CPU and the ELF DT_NEEDED reader are replaced, and the
plugin's bin/ is a temp dir. The fake release has the layout the engine ships:
  version.json                    {"linux": {"<key>": {"comfy", "comfy-12", "lib", "lib-12", "kernel_so"}}}
  <ver>/verify.json               {"schema", "linux": {"<host>": sha256, "<set>/<kernel>": sha256, "sets.json": sha256}}
  <ver>/linux/sets.json           {"schema": 2, "sets": {"<set>": [sm, ...]}}   the ONLY source of the GPU classes
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
            self.set_sets(json.dumps({"schema": 2, "sets": sets}).encode())
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


PER_ARCH = {"sm75": [75], "sm80": [80], "sm86": [86], "sm89": [89], "sm90a": [90], "sm100a": [100], "sm103a": [103],
            "sm120a": [120]}   # the per-arch ship's sets.json schema 2 (ship-deps-e PLUGIN-PERARCH-CHANGES.md)


def per_arch_release(version="0.0.14"):
    """The layout the per-arch ship publishes: sets.json schema 2, one kernel .so per architecture and CUDA major."""
    rel = Release(version, sets=None)
    for major in (13, 12):
        for gset in PER_ARCH:
            k = f"KERNEL-{version}-{gset}-cu{major}".encode()
            rel.files[f"{version}/linux/{gset}/{KERNELS[major]}"] = k
            rel.manifest["linux"][f"{gset}/{KERNELS[major]}"] = sha(k)
    rel.set_sets(json.dumps({"schema": 2, "sets": PER_ARCH}).encode())
    return rel


WIN_DLLS = {13: "quantfunc.dll", 12: "quantfunc-12.dll"}
WIN_SETS = {"sm75": [75], "sm86": [86], "sm89": [89], "sm120a": [120]}   # consumer GPUs only (the Windows ship)


class WinRelease(Release):
    """The Windows per-arch layout (verify_manifest fa97df1d5 on ship-win-perarch): {ver}/windows/sets.json — written by a
    Windows echo, so CRLF-terminated — and ONE self-contained DLL per set and CUDA major, {ver}/windows/<set>/<dll> (the
    CLI .exe sits beside it and is never fetched); verify.json section "win32", keyed by those paths."""

    def __init__(self, version="0.0.13", plugin_req="0.0.07"):   # noqa: super().__init__ builds the Linux layout
        self.files, self.fetched, self.offline, self.version = {}, [], False, version
        self.entries = {"0.0.12": {"comfy": "0.0.06", "comfy-12": "0.0.06", "lib": "0.0.12", "lib-12": "0.0.12"}}
        self.add(version, plugin_req)

    def add(self, version, plugin_req="0.0.07", tamper=None):
        """Publish `version`; tamper=<key>: verify.json lists another SHA-256 for that file (a corrupt/tampered DLL)."""
        raw = json.dumps({"schema": 2, "sets": WIN_SETS}).encode() + b"\r\n"
        self.files[f"{version}/windows/sets.json"] = raw
        body = {"sets.json": sha(raw)}
        for gset in WIN_SETS:
            for major, dll in WIN_DLLS.items():
                for name, data in ((dll, f"DLL-{version}-{gset}-cu{major}".encode()), (dll[:-4] + ".exe", b"CLI")):
                    self.files[f"{version}/windows/{gset}/{name}"] = data
                    body[f"{gset}/{name}"] = sha(b"tampered" if tamper == f"{gset}/{name}" else data)
        self.files[f"{version}/verify.json"] = json.dumps({"schema": 1, "win32": body}).encode()
        self.entries[version] = {"comfy": plugin_req, "comfy-12": plugin_req, "lib": version, "lib-12": version,
                                 "kernel_so": True}
        self.files["version.json"] = json.dumps({"linux": {}, "win32": self.entries}).encode()


class _FakeMsvcrt(types.ModuleType):
    """msvcrt on Linux: records the installer's byte-range lock calls."""
    LK_UNLCK, LK_LOCK = 0, 1

    def __init__(self):
        super().__init__("msvcrt")
        self.calls, self.fail = [], []          # fail: errnos the next LK_LOCK calls raise, in order

    def locking(self, fd, mode, nbytes):
        self.calls.append(("lock" if mode == self.LK_LOCK else "unlock", nbytes))
        if mode == self.LK_LOCK and self.fail:
            code = self.fail.pop(0)
            raise OSError(code, os.strerror(code))


@contextlib.contextmanager
def windows():
    """Run qf_engine's Windows paths: its platform constants are read per call; msvcrt is a recorder."""
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
            check("an SM outside every class installs nothing, and says why", "SM 6.1" in str(e) and not env.pairs()
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
        try:
            qfe.install_engine()
            got = "installed"
        except qfe.EngineNotInstallable as e:
            got = str(e)
        # self-CR round 6 (A): the resolver loads torch's major only, so a pair picked by the driver's major would never
        # load — refuse up front, fetch nothing, install nothing
        check("torch CUDA unknown: nothing is installed (no pair the resolver could load), with the reason",
              "CUDA build of PyTorch" in got and not rel.fetched and env.marker("consumer", 12) is None
              and env.marker("consumer", 13) is None, got[:80])
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
    rel.set_sets(json.dumps({"schema": 2, "sets": {"../x": [89]}}).encode())
    with Env(rel) as env:
        try:
            qfe.install_engine()
            check("a sets.json class that is not a plain name is refused", False, "installed!")
        except RuntimeError as e:
            check("a sets.json class that is not a plain name is refused (it would name a folder and a URL segment)",
                  "GPU-class map" in str(e) and not env.pairs(), str(e)[:80])
    rel = Release()
    rel.set_sets(json.dumps({"schema": 2, "sets": {"consumer": [75, 86, 120], "server": [80, 89, 90]}}).encode())
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
        qfe._log_lib_fingerprint = lambda lib, p, ident: None
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
    # 22) a release MOVES an SM to another GPU class (self-CR round 6, A): 0.0.13 installed SM 89 as "consumer"; 0.0.14's
    #     sets.json puts SM 89 in "server". The old consumer marker stays (it still serves that class's other SMs), but
    #     this GPU must load the NEWEST pair, now and on the next start — not the first marker by name.
    old = Release("0.0.13")
    with Env(old, sm=89) as env:
        qfe.install_engine()
        first = qfe.resolve_so_path()
        new = Release("0.0.14")
        new.set_sets(json.dumps({"schema": 2, "sets": {"consumer": [75, 86, 120],
                                                        "server": [80, 89, 90, 100, 103]}}).encode())
        qfe._engine_http_open = new.open
        qfe.install_engine()
        moved = qfe.resolve_so_path()
        qfe.install_engine()                          # the next start
        again = qfe.resolve_so_path()
        want = os.path.realpath(env.path("0.0.14-server-cu13", HOSTS[13]))
        claims = {g: (env.marker(g) or {}).get("sms") for g in ("consumer", "server")}
        check("an SM moved to another GPU class: the pair the installer chose loads, now and on the next start; only its "
              "marker claims the SM, and the old class keeps its pair for its other SMs",
              first == os.path.realpath(env.path("0.0.13-consumer-cu13", HOSTS[13])) and moved == want and again == want
              and os.path.isdir(env.path("0.0.13-consumer-cu13")) and 89 not in claims["consumer"]
              and 86 in claims["consumer"] and 89 in claims["server"],
              f"first={os.path.relpath(first, env.dir)} moved={os.path.relpath(moved, env.dir)} "
              f"again={os.path.relpath(again, env.dir)} claims={claims}")
        # 24) the release is PULLED (self-CR round 7, A): version.json lists 0.0.13 again. The installer keeps the 0.0.13
        #     consumer pair — and the resolver must load that one, not the newer 0.0.14 still on disk.
        qfe._engine_http_open = old.open
        qfe.install_engine()
        try:
            pulled = qfe.resolve_so_path()
        except RuntimeError as e:                  # a regression here is a FAIL line, not a crash of the suite
            pulled = f"refused: {str(e)[:60]}"
        claims = {g: (env.marker(g) or {}).get("sms") for g in ("consumer", "server")}
        check("a pulled release: the pair the installer keeps (0.0.13) is the one that loads; it alone claims the SM",
              pulled == os.path.realpath(env.path("0.0.13-consumer-cu13", HOSTS[13])) and 89 in claims["consumer"]
              and 89 not in (claims["server"] or []), f"pulled={os.path.relpath(pulled, env.dir)} claims={claims}")
    # 25) a hash failure after a move (self-CR round 7, A): the chosen pair is refused and re-downloaded — the old class's
    #     pair must not load in its place (it no longer claims the SM).
    old = Release("0.0.13")
    with Env(old, sm=89) as env:
        qfe.install_engine()
        new = Release("0.0.14")
        new.set_sets(json.dumps({"schema": 2, "sets": {"consumer": [75, 86, 120],
                                                        "server": [80, 89, 90, 100, 103]}}).encode())
        qfe._engine_http_open = new.open
        qfe.install_engine()
        open(env.path("0.0.14-server-cu13", HOSTS[13]), "wb").write(b"CHANGED ON DISK")
        restarted = []
        saved_start = qfe.start_engine_install
        qfe.start_engine_install = lambda *a, **k: restarted.append(1)
        got = []
        try:
            for _ in range(2):                           # this prompt, and the next one while the re-download runs
                try:
                    got.append(os.path.relpath(qfe.resolve_so_path(), env.dir))
                except RuntimeError as e:
                    got.append(f"refused: {str(e)[:60]}")
        finally:
            qfe.start_engine_install = saved_start
        check("a hash failure after a move: refused and re-downloaded; the old class's pair is not loaded instead, on "
              "this prompt or the next", all(g.startswith("refused") for g in got) and restarted[:1] == [1], got)
    # 26) two markers claiming one SM exist only in a folder the installer has not run in since (offline, after an older
    #     plugin): the newest release wins, whichever name sorts first.
    for newer_set, older_set in (("server", "consumer"), ("consumer", "server")):
        older, newer = Release("0.0.13"), Release("0.0.14")
        for rel, gset, extra in ((older, older_set, [75]), (newer, newer_set, [])):
            other = "consumer" if gset == "server" else "server"   # the older class keeps SM 75, so its marker stays
            rel.set_sets(json.dumps({"schema": 2, "sets": {gset: [89, 120] + extra,
                                                           other: [s for s in (75, 80, 86, 90) if s not in extra]}}).encode())
        with Env(older, sm=89) as env:
            qfe.install_engine()
            qfe._engine_http_open = newer.open
            qfe.install_engine()
            stale = env.marker(older_set)                   # an older plugin's folder: its marker still claims 89
            stale["sms"] = sorted(set(stale["sms"]) | {89})
            open(env.path(f".engine-{older_set}-cu13.json"), "w").write(json.dumps(stale))
            got = os.path.relpath(qfe.resolve_so_path(), env.dir)
        check(f"two markers claim the SM (legacy folder, newer release in '{newer_set}'): the newest release loads",
              got == f"0.0.14-{newer_set}-cu13/{HOSTS[13]}", got)
    # 23) a platform with no published engine (macOS; self-CR round 6, A): the library placed in bin/<platform>/ is what
    #     loads there, so a start with it in place fetches nothing and prints nothing; a missing one gets the hint naming
    #     the two published platforms.
    saved = qfe._BIN_SUBDIR, qfe._LIB_BASENAME
    try:
        with Env(Release()) as env:
            qfe._BIN_SUBDIR, qfe._LIB_BASENAME = "darwin", "libquantfunc.dylib"
            open(env.path(qfe._LIB_BASENAME), "wb").write(b"DYLIB")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                placed = qfe.install_engine()
            placed_status, placed_path = qfe.engine_install_status(), os.path.relpath(qfe.resolve_so_path(), env.dir)
            os.remove(env.path(qfe._LIB_BASENAME))
            try:
                qfe.install_engine()
                missing = "installed?"
            except qfe.EngineNotInstallable as e:
                missing = str(e)
            fetched = list(env.release.fetched)
    finally:
        qfe._BIN_SUBDIR, qfe._LIB_BASENAME = saved
    check("no published engine (macOS): a placed library means a quiet start that loads it; a missing one gets the "
          "'Linux and Windows only' hint; nothing is fetched",
          placed is None and buf.getvalue() == "" and "local" in str(placed_status) and placed_path == "libquantfunc.dylib"
          and "Linux and Windows only" in missing and fetched == [],
          f"placed={placed} printed={buf.getvalue()!r} status={placed_status} path={placed_path} missing={missing[:70]!r} "
          f"fetched={fetched}")
    # 27) a crash between the claim and the chosen marker (tests-07 round 6c, A): the other markers give up the SMs FIRST
    #     and the chosen marker is written LAST, so the window leaves NO claimant (refused, reinstalled next start) —
    #     never two for the resolver to guess between.
    old = Release("0.0.13")
    with Env(old, sm=89) as env:
        qfe.install_engine()
        new = Release("0.0.14")
        new.set_sets(json.dumps({"schema": 2, "sets": {"consumer": [75, 86, 120],
                                                        "server": [80, 89, 90, 100, 103]}}).encode())
        qfe._engine_http_open = new.open
        real_write = qfe._engine_write_file

        def crash_on_chosen(path, data):
            if os.path.basename(path) == ".engine-server-cu13.json":
                raise OSError("killed before the chosen marker was written")
            return real_write(path, data)
        qfe._engine_write_file = crash_on_chosen
        try:
            qfe.install_engine()
            crashed = "no crash?"
        except OSError as e:
            crashed = str(e)
        finally:
            qfe._engine_write_file = real_write
        claimants = [n for n in os.listdir(env.dir) if qfe._ENGINE_MARKER_RE.fullmatch(n)
                     and 89 in (json.loads(open(os.path.join(env.dir, n)).read()).get("sms") or [])]
        saved_start = qfe.start_engine_install
        qfe.start_engine_install = lambda *a, **k: None
        try:
            got = os.path.relpath(qfe.resolve_so_path(), env.dir)
        except RuntimeError as e:
            got = f"refused: {str(e)[:50]}"
        finally:
            qfe.start_engine_install = saved_start
        qfe.install_engine()                                   # the next start installs again
        again = os.path.relpath(qfe.resolve_so_path(), env.dir)
        check("a crash between the claim and the chosen marker: no marker claims the SM (refused), and the next start "
              "installs and loads the chosen pair", "killed" in crashed and claimants == [] and got.startswith("refused")
              and again == f"0.0.14-server-cu13/{HOSTS[13]}", f"crash={crashed[:30]} claimants={claimants} got={got} "
              f"again={again}")
    # 28) a marker's cuda must be an int (tests-07 round 6c, B): 13.0 would name a "-cu13.0" folder
    with Env(Release()) as env:
        qfe.install_engine()
        mk = env.marker("consumer", 13)
        mk["cuda"] = 13.0
        open(env.path(".engine-consumer-cu13.json"), "w").write(json.dumps(mk))
        check("a marker whose cuda is a float (13.0) is not a marker", qfe._read_marker(env.path(".engine-consumer-cu13.json"))
              is None)
    # 29) a class RENAMED, then the release PULLED (self-CR round 8, A, rules 1+2): 0.0.14 calls the consumer class
    #     "desktop", so the consumer marker is left claiming nothing and goes with its pair; version.json then lists
    #     0.0.13 again, so "desktop" is left claiming nothing in turn. 0.0.13 must load, and neither the emptied marker
    #     nor its ~2 GB pair may survive (nothing else would ever remove a vanished class).
    old = Release("0.0.13")
    renamed = Release("0.0.14")
    for major in (13, 12):
        k = f"KERNEL-0.0.14-desktop-cu{major}".encode()
        renamed.files[f"0.0.14/linux/desktop/{KERNELS[major]}"] = k
        renamed.manifest["linux"][f"desktop/{KERNELS[major]}"] = sha(k)
    renamed.set_sets(json.dumps({"schema": 2, "sets": {"desktop": [75, 86, 89, 120], "server": [80, 90, 100, 103]}}).encode())
    with Env(old, sm=89) as env:
        qfe.install_engine()
        qfe._engine_http_open = renamed.open
        qfe.install_engine()
        after_rename = (env.marker("consumer") is None, os.path.isdir(env.path("0.0.13-consumer-cu13")))
        qfe._engine_http_open = old.open
        qfe.install_engine()
        try:
            got = os.path.relpath(qfe.resolve_so_path(), env.dir)
        except RuntimeError as e:
            got = f"refused: {str(e)[:60]}"
        check("a renamed class, then the release pulled: 0.0.13 loads; each emptied marker went with its pair",
              got == f"0.0.13-consumer-cu13/{HOSTS[13]}" and after_rename == (True, False)
              and env.marker("desktop") is None and not os.path.exists(env.path("0.0.14-desktop-cu13")),
              f"got={got} after_rename(consumer marker gone, consumer pair still on disk)={after_rename} "
              f"desktop_marker={env.marker('desktop') is not None} desktop_pair={os.path.exists(env.path('0.0.14-desktop-cu13'))}")
    # 30) torch's CUDA major 13 -> 12 -> 13 in ONE folder (self-CR round 8, A, rule 3): a claim is per CUDA major, so the
    #     cu12 install must leave the cu13 marker (same SMs) alone, and switching back re-downloads nothing.
    with Env(Release(), sm=89) as env:
        qfe.install_engine()
        qfe._torch_cuda_major = lambda: 12
        qfe.install_engine()
        both = (env.marker("consumer", 13) is not None, env.marker("consumer", 12) is not None)
        qfe._torch_cuda_major = lambda: 13
        before = len(env.release.fetched)
        qfe.install_engine()
        refetched = [f for f in env.release.fetched[before:] if f.endswith((HOSTS[13], KERNELS[13]))]
        got = os.path.relpath(qfe.resolve_so_path(), env.dir)
        check("torch CUDA 13 -> 12 -> 13 in one folder: both markers kept, the cu13 pair loads, nothing re-downloaded",
              both == (True, True) and env.marker("consumer", 13) is not None and not refetched
              and got == f"0.0.13-consumer-cu13/{HOSTS[13]}", f"both={both} refetched={refetched} got={got}")
    # 31) a STARTING instance whose pair a concurrent install just removed (self-CR round 8, A, LOW; tests-07: fail loud,
    #     re-resolve, never half a pair): this process resolved the consumer pair, then another instance sharing the folder
    #     installed the renamed release, which emptied that class and removed its pair. The load must refuse loudly with
    #     nothing loaded, and the next prompt must resolve again and load the pair that installer chose.
    with Env(old, sm=89) as env:
        qfe.install_engine()
        real_resolve, real_cdll = qfe.resolve_so_path, qfe.ctypes.CDLL
        loaded = []

        def resolve_then_concurrent_install():
            p = real_resolve()
            qfe._engine_http_open = renamed.open       # the other instance's installer, between resolve and load
            qfe.install_engine()
            return p

        def fake_cdll(path, mode=0):
            if not os.path.isfile(path):
                raise OSError(f"{path}: cannot open shared object file: No such file or directory")
            loaded.append(os.path.relpath(path, env.dir))
            return object()
        saved = (qfe.assert_toolchain_compatible, qfe._bind, qfe._LIB, qfe._LIB_PATH, qfe._FINGERPRINT_PENDING,
                 qfe.start_engine_install)
        qfe.assert_toolchain_compatible = lambda so_path: None
        qfe._bind = lambda raw: types.SimpleNamespace(quantfunc_set_log_level=lambda level: None)
        qfe.start_engine_install = lambda *a, **k: None
        qfe._LIB, qfe._LIB_PATH = None, None
        qfe.resolve_so_path, qfe.ctypes.CDLL = resolve_then_concurrent_install, fake_cdll
        try:
            try:
                qfe.load_lib()
                first = "loaded?"
            except RuntimeError as e:
                first = f"refused: {str(e)[:50]}"
            first_loaded, lib_after_refusal = list(loaded), qfe._LIB
            qfe.resolve_so_path = real_resolve
            qfe.load_lib()
            second = qfe._LIB_PATH and os.path.relpath(qfe._LIB_PATH, env.dir)
        finally:
            qfe.resolve_so_path, qfe.ctypes.CDLL = real_resolve, real_cdll
            (qfe.assert_toolchain_compatible, qfe._bind, qfe._LIB, qfe._LIB_PATH, qfe._FINGERPRINT_PENDING,
             qfe.start_engine_install) = saved
        check("a starting instance whose pair a concurrent install removed: refused loudly with nothing loaded; the next "
              "prompt resolves again and loads the chosen pair", first.startswith("refused") and first_loaded == []
              and lib_after_refusal is None and second == f"0.0.14-desktop-cu13/{HOSTS[13]}",
              f"first={first} first_loaded={first_loaded} second={second}")
    # 32) one engine pair per process = one GPU architecture (per-arch kernel sets; tests-07 2026-09-24): ComfyUI's device
    #     (GPU 0, SM 89) installs the sm89 pair; a pipeline on GPU 1 (SM 86) refuses loudly with the hint, GPU 0 is fine.
    #     A class covering both SMs (a multi-arch release) still admits GPU 1: the rule is coverage, not "a second device".
    got = {}
    for rel, label in ((per_arch_release(), "per-arch"), (Release("0.0.13"), "multi-arch")):
        with Env(rel, sm={0: 89, 1: 86}) as env:
            qfe.install_engine()
            for dev in (0, 1):
                try:
                    qfe.make_create_params(model_dir=env.dir, device_idx=dev)
                    got[f"{label}/gpu{dev}"] = "ok"
                except RuntimeError as e:
                    got[f"{label}/gpu{dev}"] = ("refused" if "one ComfyUI per GPU architecture" in str(e)
                                                and "GPU 1 is SM 8.6" in str(e) else f"other: {str(e)[:60]}")
    check("a second GPU architecture in one process is refused loudly with the hint; a GPU the installed class covers "
          "is fine", got == {"per-arch/gpu0": "ok", "per-arch/gpu1": "refused", "multi-arch/gpu0": "ok",
                             "multi-arch/gpu1": "ok"}, got)
    # 33) the per-arch release as published (sets.json schema 2, one kernel .so per architecture): each GPU installs and
    #     loads EXACTLY its architecture's set, in its own folder, and its marker claims only that SM.
    want = {75: "sm75", 80: "sm80", 86: "sm86", 89: "sm89", 90: "sm90a", 100: "sm100a", 103: "sm103a", 120: "sm120a"}
    got = {}
    for sm, gset in want.items():
        with Env(per_arch_release(), sm=sm) as env:
            m = qfe.install_engine() or {}
            got[sm] = (m.get("set"), m.get("sms"), os.path.relpath(qfe.resolve_so_path(), env.dir))
    check("per-arch release: every GPU installs and loads exactly its own architecture's set",
          all(got[sm] == (gset, [sm], f"0.0.14-{gset}-cu13/{HOSTS[13]}") for sm, gset in want.items()), got)
    # 34) an architecture the per-arch release does not publish (SM 8.7 / 11.0 / 12.1 are aarch64 parts): refused by name,
    #     nothing installed; never the nearest architecture's kernel.
    refused = {}
    for sm in (87, 110, 121):
        with Env(per_arch_release(), sm=sm) as env:
            try:
                qfe.install_engine()
                refused[sm] = "installed!"
            except qfe.EngineNotInstallable as e:
                refused[sm] = f"SM {sm // 10}.{sm % 10}" in str(e) and not env.pairs()
    check("an unpublished architecture is refused by name (SM 8.7 / 11.0 / 12.1), never given the nearest kernel",
          refused == {87: True, 110: True, 121: True}, refused)
    # 35) sets.json must be schema 2, an int: the old consumer/server schema 1 and a float 2.0 are refused loudly.
    res = {}
    for label, schema in (("schema 1", 1), ("float 2.0", 2.0)):
        rel = Release()
        rel.set_sets(json.dumps({"schema": schema, "sets": SETS}).encode())
        with Env(rel, sm=89) as env:
            try:
                qfe.install_engine()
                res[label] = "installed!"
            except RuntimeError as e:
                res[label] = "schema" in str(e) and not env.pairs()
    check("a sets.json that is not schema 2 (the old schema 1, or a float 2.0) is refused loudly, nothing installed",
          res == {"schema 1": True, "float 2.0": True}, res)

    # ── Windows: the SAME installer (tests-07 dispatch, user 「根据自己的显卡型号下载对应so」): one DLL per set + CUDA major ──
    # 36) every consumer GPU x CUDA major installs EXACTLY its set's DLL into its own folder, marker last: the fetches are
    #     version.json, verify.json, windows/sets.json and that ONE DLL (never another set, never the CLI .exe), under the
    #     byte-range lock (lock, then unlock), and the resolver loads it. SM 89 + cu12 -> sm89/quantfunc-12.dll.
    got, want = {}, {}
    for sm, gset in ((75, "sm75"), (86, "sm86"), (89, "sm89"), (120, "sm120a")):
        for major, dll in WIN_DLLS.items():
            rel = WinRelease()
            with windows() as ms, Env(rel, torch_major=major, driver_major=13, sm=sm, machine="AMD64") as env:
                m = qfe.install_engine() or {}
                folder = f"0.0.13-{gset}-cu{major}"
                got[(sm, major)] = (m.get("set"), m.get("host"), m.get("kernel"), sorted(m.get("sha256", {})),
                                    env.read(folder, dll) == rel.files[f"0.0.13/windows/{gset}/{dll}"],
                                    os.path.relpath(qfe.resolve_so_path(), env.dir), rel.fetched, ms.calls,
                                    env.marker(gset, major) == m, env.leftovers())
                want[(sm, major)] = (gset, dll, None, [dll], True, f"{folder}/{dll}",
                                     ["version.json", "0.0.13/verify.json", "0.0.13/windows/sets.json",
                                      f"0.0.13/windows/{gset}/{dll}"], [("lock", 1), ("unlock", 1)], True, [])
    check("Windows: each consumer GPU x CUDA major installs exactly its own DLL (SM 89 + cu12 -> sm89/quantfunc-12.dll), "
          "fetches nothing else, locks, and loads it", got == want,
          {k: v for k, v in got.items() if v != want[k]} or "all 8")
    # 37) a wrong SHA-256: nothing of the new release is put in place and the older verified DLL stays installed + loaded.
    rel = WinRelease("0.0.13")
    with windows(), Env(rel, torch_major=12, sm=89, machine="AMD64") as env:
        qfe.install_engine()
        rel.add("0.0.14", tamper="sm89/quantfunc-12.dll")
        try:
            qfe.install_engine()
            outcome = "installed!"
        except RuntimeError as e:
            outcome = "published SHA-256" in str(e)
        kept = (env.marker("sm89", 12) or {}).get("version"), os.path.relpath(qfe.resolve_so_path(), env.dir)
        new_dll = env.read("0.0.14-sm89-cu12", "quantfunc-12.dll")
        left = env.leftovers()
    check("Windows: a DLL whose SHA-256 does not match is never put in place; the older verified DLL stays installed and "
          "loaded", outcome is True and kept == ("0.0.13", "0.0.13-sm89-cu12/quantfunc-12.dll") and new_dll is None
          and left == [], f"{outcome} kept={kept} new={new_dll is not None} left={left}")
    # 38) a GPU with no Windows set (the server classes are Linux-only): refused by name with the published SMs, nothing
    #     installed, never the nearest set's DLL.
    refused = {}
    for sm in (80, 90, 100):
        with windows(), Env(WinRelease(), torch_major=13, sm=sm, machine="AMD64") as env:
            try:
                qfe.install_engine()
                refused[sm] = "installed!"
            except qfe.EngineNotInstallable as e:
                refused[sm] = (f"SM {sm // 10}.{sm % 10}" in str(e) and "7.5, 8.6, 8.9, 12.0" in str(e)
                               and not env.pairs())
    check("Windows: a GPU with no published set is refused by name (with the published SMs), nothing installed",
          refused == {80: True, 90: True, 100: True}, refused)
    # 39) a newer compatible release replaces the older one on the next start (into a NEW folder, the marker flips); the
    #     replaced folder stays until the following install (a process may still have that DLL loaded), and a release
    #     that needs a newer plugin is not taken.
    rel = WinRelease("0.0.13")
    with windows(), Env(rel, torch_major=13, sm=120, machine="AMD64") as env:
        qfe.install_engine()
        rel.add("0.0.14")
        rel.add("0.0.15", plugin_req="0.0.08")                   # needs a newer plugin: never picked by 0.0.07
        m14 = qfe.install_engine() or {}
        after14 = (m14.get("version"), os.path.relpath(qfe.resolve_so_path(), env.dir), env.pairs())
        rel.add("0.0.16")
        qfe.install_engine()
        after16 = env.pairs()
    check("Windows: a newer compatible release replaces the older on the next start (new folder, marker flips); the "
          "replaced folder goes at the following install; one needing a newer plugin is not taken",
          after14 == ("0.0.14", "0.0.14-sm120a-cu13/quantfunc.dll", ["0.0.13-sm120a-cu13", "0.0.14-sm120a-cu13"])
          and after16 == ["0.0.14-sm120a-cu13", "0.0.16-sm120a-cu13"], f"after 0.0.14: {after14}; after 0.0.16: {after16}")
    # 40) the gate: a win32 entry without "kernel_so": true (the classic monolith 0.0.12) is never installed by this
    #     installer, whatever the plugin version; nothing but version.json is fetched.
    rel = WinRelease("0.0.13")
    del rel.entries["0.0.13"]
    rel.files["version.json"] = json.dumps({"linux": {}, "win32": rel.entries}).encode()
    with windows(), Env(rel, torch_major=13, sm=89, machine="AMD64") as env:
        try:
            qfe.install_engine()
            gated = "installed!"
        except qfe.EngineNotInstallable as e:
            gated = "per-architecture layout" in str(e) and rel.fetched == ["version.json"] and not env.pairs()
    check("Windows: only a release gated with kernel_so is installed (the classic 0.0.12 monolith never is)", gated is True,
          f"{gated} fetched={rel.fetched}")
    # 41) the classic updater's bin/windows/quantfunc.dll (left by 0.0.06): WITHOUT the lock it is ignored — the per-set
    #     DLL is installed and loads; WITH bin/windows/.dev_lib_lock it is the local build: nothing fetched, one line.
    rel = WinRelease("0.0.13")
    with windows(), Env(rel, torch_major=12, sm=89, machine="AMD64") as env:
        open(env.path("quantfunc.dll"), "wb").write(b"CLASSIC-0.0.12")
        qfe.install_engine()
        unlocked = os.path.relpath(qfe.resolve_so_path(), env.dir)
    rel = WinRelease("0.0.13")
    with windows(), Env(rel, torch_major=12, sm=89, machine="AMD64") as env:
        open(env.path("quantfunc.dll"), "wb").write(b"LOCAL-BUILD")
        open(env.path(qfe._ENGINE_LOCAL_BUILD_LOCK), "w").close()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            skipped = qfe.install_engine()
        locked = os.path.relpath(qfe.resolve_so_path(), env.dir), rel.fetched, buf.getvalue().count("install skipped")
    check("Windows: a placed quantfunc.dll without the lock is ignored (the per-set DLL loads); under the lock it is the "
          "local build (nothing fetched, one line)",
          unlocked == "0.0.13-sm89-cu12/quantfunc-12.dll" and skipped is None and locked == ("quantfunc.dll", [], 1),
          f"unlocked={unlocked} locked={locked}")
    # 42) one engine per process = one GPU architecture on Windows too: GPU 0 (SM 89) installs sm89; a pipeline on GPU 1
    #     (SM 86) is refused with the hint.
    with windows(), Env(WinRelease(), torch_major=12, sm={0: 89, 1: 86}, machine="AMD64") as env:
        qfe.install_engine()
        second = {}
        for dev in (0, 1):
            try:
                qfe.make_create_params(model_dir=env.dir, device_idx=dev)
                second[dev] = "ok"
            except RuntimeError as e:
                second[dev] = "refused" if "one ComfyUI per GPU architecture" in str(e) else f"other: {str(e)[:60]}"
    check("Windows: a second GPU architecture in one process is refused loudly", second == {0: "ok", 1: "refused"}, second)
    # 43) a Windows marker is ONE file with "kernel": null — a marker naming a kernel (a Linux-shaped or tampered marker
    #     in bin/windows) is not followed: nothing loads from it.
    with windows(), Env(WinRelease(), torch_major=12, sm=89, machine="AMD64") as env:
        m = qfe.install_engine()
        mp = env.path(".engine-sm89-cu12.json")
        good = qfe._read_marker(mp) is not None
        followed = {}
        for label, bad_m in (("kernel named", dict(m, kernel="quantfunc_kernels.dll")),   # only the kernel field
                             ("second sha256", dict(m, sha256=dict(m["sha256"], **{"x.dll": "0" * 64})))):
            open(mp, "w").write(json.dumps(bad_m))
            followed[label] = qfe._read_marker(mp) is not None
        try:
            qfe.resolve_so_path()
            loaded = "loaded!"
        except RuntimeError as e:
            loaded = "no QuantFunc engine is installed" in str(e)
    check("Windows: a marker naming a kernel, or recording a second file, is not a Windows marker and is never followed",
          good and followed == {"kernel named": False, "second sha256": False} and loaded is True,
          f"good={good} followed={followed} loaded={loaded}")
    # 44) the Windows install lock waits ONLY while another installer holds it (the CRT's EDEADLOCK / EACCES after its
    #     own ~10 s of retries); any other error fails the install loudly instead of spinning forever.
    import errno as _errno
    held = getattr(_errno, "EDEADLOCK", _errno.EDEADLK)
    with windows() as ms, Env(WinRelease(), torch_major=12, sm=89, machine="AMD64") as env:
        ms.fail = [held, _errno.EACCES]
        waited = (qfe.install_engine() or {}).get("set"), [c[0] for c in ms.calls]
    with windows() as ms, Env(WinRelease(), torch_major=12, sm=89, machine="AMD64") as env:
        ms.fail = [_errno.EBADF]
        try:
            qfe.install_engine()
            broken = "installed!"
        except OSError as e:
            broken = e.errno == _errno.EBADF and not env.pairs()
    check("Windows: the install lock waits while held (EDEADLOCK/EACCES) and fails loudly on anything else",
          waited == ("sm89", ["lock", "lock", "lock", "unlock"]) and broken is True, f"waited={waited} broken={broken}")
    print("ENGINE_INSTALL:", "PASS" if bad == 0 else f"FAIL ({bad} wrong)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
