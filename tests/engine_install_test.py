#!/usr/bin/env python3
"""Option C (user 2026-09-24 「在原生加载器里实现」): the plugin's engine installer, end to end against a FAKE release server.

qf_engine is loaded standalone (no ComfyUI, no torch, no GPU, no network). The ONE network entry (_engine_http_open),
torch's CUDA major, the driver's CUDA major, the GPU's SM and the ELF DT_NEEDED reader are replaced, and the plugin's
bin/ is a temp dir. The fake release has the layout the engine ships:
  version.json                    {"linux": {"<ver>": {"comfy", "comfy-12", "kernel_so"}}}
  <ver>/verify.json               {"schema", "version", "linux": {"<set>/<file>": sha256}}
  <ver>/linux/<set>/<file>        host + kernel per (GPU class, CUDA major)
Matrix: GPU class x CUDA major x {missing, present, corrupt, offline}, plus the rules each tested both ways.
"""
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile

_PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("qfe_install_test", os.path.join(_PLUGIN, "qf_engine.py"))
qfe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qfe)

HOSTS = {13: "libquantfunc.so", 12: "libquantfunc-12.so"}
KERNELS = {13: "libquantfunc_kernels.so", 12: "libquantfunc_kernels-12.so"}
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

    def __init__(self, version="0.0.13", plugin_req="0.0.07"):
        self.files, self.fetched, self.offline = {}, [], False
        self.version = version
        body = {}
        for major in (13, 12):
            for gset in ("consumer", "server"):
                h = f"HOST-{version}-{gset}-cu{major}".encode()
                k = f"KERNEL-{version}-{gset}-cu{major}".encode()
                self.files[f"{version}/linux/{gset}/{HOSTS[major]}"] = h
                self.files[f"{version}/linux/{gset}/{KERNELS[major]}"] = k
                body[f"{gset}/{HOSTS[major]}"], body[f"{gset}/{KERNELS[major]}"] = sha(h), sha(k)
        self.manifest = {"schema": 1, "version": version, "linux": body}
        self.files[f"{version}/verify.json"] = json.dumps(self.manifest).encode()
        self.files["version.json"] = json.dumps({"linux": {
            "0.0.12": {"comfy": "0.0.06", "comfy-12": "0.0.06", "lib": "0.0.12"},     # monolithic: never picked
            version: {"comfy": plugin_req, "comfy-12": plugin_req, "lib": version, "kernel_so": True},
        }}).encode()

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


class Env:
    """A machine: a temp plugin bin/ + the stubs. Restores qf_engine on exit."""

    def __init__(self, release, torch_major=13, driver_major=13, sm=89, plugin_version="0.0.07"):
        self.release = release
        self.dir = tempfile.mkdtemp(prefix="qf_engine_install_")
        with open(os.path.join(self.dir, "version.json"), "w") as f:
            json.dump({"comfy": plugin_version}, f)
        self._saved = {n: getattr(qfe, n) for n in ("_engine_http_open", "_torch_cuda_major", "_driver_cuda_major",
                                                    "_gpu_sm", "_engine_bin_dir", "_elf_needed", "start_engine_install")}
        qfe._engine_http_open = release.open
        qfe._torch_cuda_major = lambda: torch_major
        qfe._driver_cuda_major = lambda: driver_major
        qfe._gpu_sm = lambda device_idx=0: sm
        qfe._engine_bin_dir = lambda: self.dir
        qfe._elf_needed = fake_needed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for n, v in self._saved.items():
            setattr(qfe, n, v)
        shutil.rmtree(self.dir, ignore_errors=True)

    def read(self, name):
        p = os.path.join(self.dir, name)
        return open(p, "rb").read() if os.path.isfile(p) else None

    def leftovers(self):
        return sorted(f for f in os.listdir(self.dir) if ".part-" in f)


def main():
    print("=== engine installer (option C) ===")
    # 1) GPU class x CUDA major: fresh installs land the right pair, KERNEL renamed before HOST, marker written
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
                m = json.loads(env.read(qfe._ENGINE_MARKER))
                ok = (env.read(HOSTS[major]) == rel.files[f"0.0.13/linux/{gset}/{HOSTS[major]}"]
                      and env.read(KERNELS[major]) == rel.files[f"0.0.13/linux/{gset}/{KERNELS[major]}"]
                      and m == {"version": "0.0.13", "set": gset, "cuda": major, "host": HOSTS[major],
                                "kernel": KERNELS[major]}
                      and order.index(KERNELS[major]) < order.index(HOSTS[major]) and not env.leftovers())
                check(f"SM {sm} x CUDA {major}: {gset} pair installed, kernel renamed before host, marker written", ok,
                      f"order={order} marker={m}")
    # 2) no install outside the published sets / for an old driver; torch CUDA unknown falls back to the driver
    rel = Release()
    with Env(rel, sm=61) as env:
        try:
            qfe.install_engine()
            check("an SM outside both sets installs nothing", False, "installed!")
        except qfe.EngineNotInstallable as e:
            check("an SM outside both sets installs nothing, and says why", "SM 61" in str(e) and not env.read(HOSTS[13])
                  and not any("/linux/" in f for f in rel.fetched), str(e)[:80])
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
        check("torch CUDA unknown: the driver's major chooses the host", env.read(HOSTS[12]) is not None
              and env.read(HOSTS[13]) is None)
    # 3) present: an installed matching pair is kept; the release is still read (remote-first); nothing re-downloaded
    rel = Release()
    with Env(rel) as env:
        qfe.install_engine()
        rel.fetched.clear()
        qfe.install_engine()
        check("present + matching: nothing re-downloaded, release manifests still read",
              rel.fetched == ["version.json", "0.0.13/verify.json"], rel.fetched)
        # corrupt on disk: the host's bytes no longer hash to the manifest -> reinstall
        with open(os.path.join(env.dir, HOSTS[13]), "ab") as f:
            f.write(b"x")
        rel.fetched.clear()
        qfe.install_engine()
        check("present but corrupt on disk: reinstalled", env.read(HOSTS[13]) == rel.files[
            f"0.0.13/linux/consumer/{HOSTS[13]}"] and f"0.0.13/linux/consumer/{HOSTS[13]}" in rel.fetched, rel.fetched)
    # 4) a newer release replaces an older one; a CORRUPT download replaces nothing (never bricks)
    old = Release("0.0.13")
    with Env(old) as env:
        qfe.install_engine()
        new = Release("0.0.14")
        new.files[f"0.0.14/linux/consumer/{HOSTS[13]}"] = b"HOST-tampered-cu13"
        qfe._engine_http_open = new.open
        try:
            qfe.install_engine()
            check("a corrupt download is refused", False, "installed!")
        except RuntimeError as e:
            check("a corrupt download is refused and the old pair stays",
                  "SHA-256" in str(e) and json.loads(env.read(qfe._ENGINE_MARKER))["version"] == "0.0.13"
                  and env.read(HOSTS[13]) == old.files[f"0.0.13/linux/consumer/{HOSTS[13]}"] and not env.leftovers(),
                  str(e)[:80])
        good = Release("0.0.14")
        qfe._engine_http_open = good.open
        qfe.install_engine()
        check("a newer compatible release replaces the installed one",
              json.loads(env.read(qfe._ENGINE_MARKER))["version"] == "0.0.14")
    # 5) all-or-nothing: the host downloads, the kernel is missing -> nothing replaced, no temp files left
    rel = Release()
    with Env(rel) as env:
        del rel.files[f"0.0.13/linux/consumer/{KERNELS[13]}"]
        try:
            qfe.install_engine()
            check("a missing kernel installs nothing", False, "installed!")
        except OSError:
            check("a missing kernel installs nothing (host not renamed, no temp left)",
                  env.read(HOSTS[13]) is None and env.read(qfe._ENGINE_MARKER) is None and not env.leftovers())
    # 6) offline: an installed engine is kept (status offline); none installed -> status failed with the reason
    for installed in (True, False):
        rel = Release()
        with Env(rel) as env:
            if installed:
                qfe.install_engine()
            rel.offline = True
            saved_override = os.environ.pop(qfe._ENV_SO_OVERRIDE, None)   # the dev override would skip the install
            try:
                th = qfe.start_engine_install()
                th.join(10)
            finally:
                if saved_override is not None:
                    os.environ[qfe._ENV_SO_OVERRIDE] = saved_override
            state, detail = qfe.engine_install_status()
            want = "offline" if installed else "failed"
            check(f"offline with{'' if installed else 'out'} an installed engine: status {want}",
                  state == want and "unreachable" in detail and (env.read(HOSTS[13]) is not None) == installed,
                  f"{state}: {detail[:60]}")
    # 7) never a server-supplied path: a traversal "version" is never picked; a kernel name must be a QuantFunc kernel
    rel = Release()
    rel.files["version.json"] = json.dumps({"linux": {"../../../tmp/x": {"comfy": "0.0.01", "kernel_so": True}}}).encode()
    with Env(rel) as env:
        try:
            qfe.install_engine()
            check("a traversal version is refused", False, "installed!")
        except qfe.EngineNotInstallable:
            check("a traversal version is never picked (nothing fetched below the release root)",
                  not any(".." in f for f in rel.fetched) and env.read(HOSTS[13]) is None)
    rel = Release()
    with Env(rel) as env:
        qfe._elf_needed = lambda p: ["libevil.so", "libcudart.so.13"]
        try:
            qfe.install_engine()
            check("a host naming no QuantFunc kernel installs nothing", False, "installed!")
        except RuntimeError as e:
            check("a host naming no QuantFunc kernel installs nothing", "kernel" in str(e) and env.read(HOSTS[13]) is None,
                  str(e)[:80])
    # 8) HTTPS only: a plain-HTTP URL and a redirect to plain HTTP are both refused by the real network entry
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
    # 9) sets.json when published: its hash is checked, and it decides the GPU class
    rel = Release()
    sets = json.dumps({"schema": 1, "sets": {"consumer": [75, 86, 120], "server": [80, 89, 90]}}).encode()
    rel.files["0.0.13/linux/sets.json"] = sets
    rel.manifest["linux"]["sets.json"] = sha(sets)
    rel.files["0.0.13/verify.json"] = json.dumps(rel.manifest).encode()
    with Env(rel, sm=89) as env:
        qfe.install_engine()
        check("a published sets.json decides the class (SM 89 moved to server here)",
              json.loads(env.read(qfe._ENGINE_MARKER))["set"] == "server")
    rel.files["0.0.13/linux/sets.json"] = sets + b" "
    with Env(rel, sm=89) as env:
        try:
            qfe.install_engine()
            check("a sets.json that does not match its hash is refused", False, "installed!")
        except RuntimeError as e:
            check("a sets.json that does not match its hash is refused", "sets.json" in str(e), str(e)[:60])
    # 10) the reinstall-once bound: a pair that will not LOAD is re-downloaded once per release, then only reported
    rel = Release()
    with Env(rel) as env:
        qfe.install_engine()
        host = os.path.join(env.dir, HOSTS[13])
        started = []
        qfe.start_engine_install = lambda device_idx=0: started.append(1)
        msg1 = qfe._engine_load_failed(host, OSError("undefined symbol: qf_kernel_x"))
        flag = env.read(KERNELS[13] + qfe._ENGINE_REINSTALLED_SUFFIX)
        first = ("re-downloaded once" in msg1 and started == [1] and flag == b"0.0.13"
                 and env.read(qfe._ENGINE_MARKER) is None)
        qfe.install_engine()                       # the re-download
        msg2 = qfe._engine_load_failed(host, OSError("undefined symbol: qf_kernel_x"))
        second = "again after one re-download" in msg2 and started == [1] and env.read(qfe._ENGINE_MARKER) is not None
        qfe._engine_load_ok(host)
        check("a pair that will not load: one re-download per release, then only the error (flag cleared by a load)",
              first and second and env.read(KERNELS[13] + qfe._ENGINE_REINSTALLED_SUFFIX) is None,
              f"first={first} second={second}")
        # a symlinked plugin dir (common for custom_nodes): the loader sees the REAL path, the bin dir the link
        link = env.dir + "-link"
        os.symlink(env.dir, link)
        try:
            qfe._engine_bin_dir = lambda: link
            msg3 = qfe._engine_load_failed(os.path.realpath(host), OSError("undefined symbol: qf_kernel_x"))
            check("the bound also applies through a symlinked plugin dir", "re-downloaded once" in msg3
                  and started == [1, 1], msg3[:80])
        finally:
            os.remove(link)
    # 11) the preload rule: a QuantFunc kernel library is never preloaded; other sidecars still are
    entries = ["libquantfunc.so", "libquantfunc_kernels.so", "libquantfunc_kernels-12.so", "libquantfunc_kernels_sm89.so",
               "libquantfunc_attention.so", "libopencv_core.so.4.6", "libquantfunc_attention.so.prod-bak", "notes.txt"]
    got = qfe._sidecar_preloads(entries, "libquantfunc.so")
    check("no QuantFunc kernel library is preloaded; real non-kernel sidecars still are",
          got == ["libopencv_core.so.4.6", "libquantfunc_attention.so"], got)
    print("ENGINE_INSTALL:", "PASS" if bad == 0 else f"FAIL ({bad} wrong)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
