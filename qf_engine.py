"""qf_native.qf_engine -- ctypes bridge to QuantFunc PC Engine (25-symbol qf_* C-API).

Structs and function signatures mirror include/qf_capi.h verbatim.
Host libquantfunc.so has zero CUDA DT_NEEDED imports; CUDA lives in the per-SM
consumer kernel plugin libquantfunc_kernels.so (or libquantfunc_kernels-12.so for CUDA 12).
"""
import builtins
import contextlib
import ctypes
import functools
import glob
import inspect
import json
import logging
import mmap
import operator
import os
import platform
import re
import shutil
import struct
import sys
import threading
import time
import traceback
import types
import urllib.parse
import urllib.request

if __package__:
    from .qf_capi import (
        QF_OK, QF_ERR_INVALID_ARG, QF_ERR_BACKEND, QF_ERR_OOM, QF_ERR_IO,
        QF_ERR_INTERNAL, QF_ERR_BUSY, QF_ERR_AUTH, QF_ERR_UNSUPPORTED,
        QF_DTYPE_FP32, QF_DTYPE_FP16, QF_DTYPE_BF16, QF_DTYPE_INT8, QF_DTYPE_INT32,
        QF_DTYPE_FP8_E4M3, QF_DTYPE_INT4_PACKED, QF_DTYPE_E2M1_PACKED,
        QF_FP32, QF_FP16, QF_BF16,
        QF_ATTN_CAP_FP16, QF_ATTN_CAP_INT8, QF_ATTN_CAP_INT4,
        QF_MAX_STEP_REFS, QF_UINT64_MAX,
        QFAuthConfig, QFDeviceCaps, QFXfmCreateParams, QFVramEstimate,
        QFSessionBeginParams, QFStepParams, QFStepAvParams, QFStepRefsParams,
        ColdNeed, _CAPI_SIGNATURES,
    )
else:
    import importlib.util as _ilu
    _capi_spec = _ilu.spec_from_file_location(
        "qf_capi", os.path.join(os.path.dirname(os.path.abspath(__file__)), "qf_capi.py"))
    _capi = _ilu.module_from_spec(_capi_spec)
    _capi_spec.loader.exec_module(_capi)
    for _sym in (
        "QF_OK", "QF_ERR_INVALID_ARG", "QF_ERR_BACKEND", "QF_ERR_OOM", "QF_ERR_IO",
        "QF_ERR_INTERNAL", "QF_ERR_BUSY", "QF_ERR_AUTH", "QF_ERR_UNSUPPORTED",
        "QF_DTYPE_FP32", "QF_DTYPE_FP16", "QF_DTYPE_BF16", "QF_DTYPE_INT8", "QF_DTYPE_INT32",
        "QF_DTYPE_FP8_E4M3", "QF_DTYPE_INT4_PACKED", "QF_DTYPE_E2M1_PACKED",
        "QF_FP32", "QF_FP16", "QF_BF16",
        "QF_ATTN_CAP_FP16", "QF_ATTN_CAP_INT8", "QF_ATTN_CAP_INT4",
        "QF_MAX_STEP_REFS", "QF_UINT64_MAX",
        "QFAuthConfig", "QFDeviceCaps", "QFXfmCreateParams", "QFVramEstimate",
        "QFSessionBeginParams", "QFStepParams", "QFStepAvParams", "QFStepRefsParams",
        "ColdNeed", "_CAPI_SIGNATURES",
    ):
        globals()[_sym] = getattr(_capi, _sym)
    del _ilu, _capi_spec, _capi, _sym

# Pending LTX reference-audio borrows must survive model/handle GC until native cleanup.
_REFERENCE_AUDIO_PENDING_OWNERS = set()

_IS_WINDOWS = platform.system() == "Windows"
_IS_DARWIN = platform.system() == "Darwin"
_BIN_SUBDIR = "windows" if _IS_WINDOWS else ("darwin" if _IS_DARWIN else "linux")
_LIB_BASENAME = ("quantfunc.dll" if _IS_WINDOWS else
                 ("libquantfunc.dylib" if _IS_DARWIN else "libquantfunc.so"))
_KEYFILE_BASENAME = "config.json"
_ENV_SO_OVERRIDE = "QF_NATIVE_SO_PATH"
_ENV_KEYFILE_OVERRIDE = "QF_NATIVE_KEYFILE"
_ENV_KERNEL_PLUGIN_DIR = "QF_KERNEL_PLUGIN_DIR"
_ENV_ALLOW_UNVERIFIED_TOOLCHAIN = "QF_NATIVE_ALLOW_UNVERIFIED_TOOLCHAIN"

# System V ELF constants for pure-stdlib DT_NEEDED parsing.
_SHT_DYNAMIC = 6
_DT_NULL = 0
_DT_NEEDED = 1

# Consumer 4-SM scope (Turing 75, Ampere 86, Ada 89, Blackwell 120).
_CONSUMER_SMS = (75, 86, 89, 120)


def library_identity(lib):
    """Loaded image identity, not its filename (load_lib retains the CDLL)."""
    handle = getattr(lib, "_handle", None)
    return ("dso", int(handle)) if handle is not None else ("library", lib)


def _bind(lib):
    missing = [name for name, _, _ in _CAPI_SIGNATURES if not hasattr(lib, name)]
    if missing:
        installed = _markers() and not _engine_local_choice()
        hint = "; restart ComfyUI to re-check for the engine" if installed else ""
        raise RuntimeError(
            f"QuantFunc engine/plugin mismatch: library lacks PC Engine C-API symbol(s) "
            f"{', '.join(missing[:5])}{hint}")
    for name, restype, argtypes in _CAPI_SIGNATURES:
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    return lib


_LIB = None
_LIB_PATH = None
_FINGERPRINT_PENDING = None
_LIB_LOCK = threading.Lock()
_LOG_LEVEL = None
_LOG_INFO = 2


def set_log_level(level):
    """Engine console detail, process-wide. Never triggers library loading."""
    global _LOG_LEVEL
    _LOG_LEVEL = int(level)
    if _LIB is not None:
        _LIB.qf_set_log_level(_LOG_LEVEL)
    _emit_fingerprint()


def loaded_so_path():
    """The engine library path loaded in this process, or None before load_lib."""
    return _LIB_PATH


def _emit_fingerprint():
    global _FINGERPRINT_PENDING
    if _FINGERPRINT_PENDING is not None and _LOG_LEVEL is not None and _LOG_LEVEL <= _LOG_INFO:
        lib, path, ident = _FINGERPRINT_PENDING
        _FINGERPRINT_PENDING = None
        _log_lib_fingerprint(lib, path, ident)


def info(msg, *args, flush=True):
    """Plugin detail line; printed only when log level is info or lower."""
    if _LOG_LEVEL is not None and _LOG_LEVEL <= _LOG_INFO:
        say(msg % args if args else msg, flush=flush)


# ── Console-safe output & exception boundary ─────────────────────────────────

def console_safe(text):
    """Escape characters that stdout/stderr code pages cannot encode."""
    for stream in (sys.stdout, sys.stderr):
        enc = getattr(stream, "encoding", None)
        if enc:
            try:
                text = text.encode(enc, "backslashreplace").decode(enc)
            except LookupError:
                text = text.encode("ascii", "backslashreplace").decode("ascii")
    return text


def say(msg, flush=True):
    """print() for plugin lines: never raises on the console's code page."""
    print(console_safe(msg), flush=flush)


class _ConsoleSafe(logging.Filter):
    """Ensure plugin log records are printable on the active console code page."""

    def filter(self, record):
        try:
            text = record.getMessage()
        except (TypeError, ValueError):
            return True
        if record.exc_info:
            text += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip("\n")
            record.exc_info = record.exc_text = None
        if record.stack_info:
            text += "\n" + record.stack_info
            record.stack_info = None
        record.msg, record.args = console_safe(text), None
        return True


def logger(name):
    """logging.getLogger(name) with _ConsoleSafe filter attached."""
    log = logging.getLogger(name)
    if not any(isinstance(f, _ConsoleSafe) for f in log.filters):
        log.addFilter(_ConsoleSafe())
    return log


def _console_safe_one(exc):
    fields = (("strerror", "filename", "filename2") if isinstance(exc, OSError) else
              ("msg", "filename", "text") if isinstance(exc, SyntaxError) else ())
    args, values, notes = exc.args, [getattr(exc, a, None) for a in fields], getattr(exc, "__notes__", None)
    if isinstance(notes, list):
        exc.__notes__ = [console_safe(n) if isinstance(n, str) else n for n in notes]
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001
        return True
    if console_safe(text) == text:
        return True
    exc.args = tuple(console_safe(a) if isinstance(a, str) else a for a in args)
    for attr, value in zip(fields, values):
        if isinstance(value, str):
            setattr(exc, attr, console_safe(value))
    try:
        now = str(exc)
    except Exception:  # noqa: BLE001
        now = ""
    if console_safe(now) == now:
        try:
            exc.qf_console_original = text
        except (AttributeError, TypeError):
            pass
        return True
    exc.args = args
    for attr, value in zip(fields, values):
        if isinstance(value, str):
            setattr(exc, attr, value)
    if isinstance(notes, list):
        exc.__notes__ = notes
    return False


def _group_members(e):
    return e.exceptions if isinstance(e, getattr(builtins, "BaseExceptionGroup", ())) else ()


def _stand_in(e):
    safe = RuntimeError(console_safe("".join(traceback.format_exception_only(type(e), e)).strip()))
    safe.qf_console_original = e
    safe.__cause__ = e.__cause__ if e.__cause__ is not None or e.__suppress_context__ else e.__context__
    safe.__context__ = e.__context__
    safe.__suppress_context__ = e.__suppress_context__
    return safe.with_traceback(e.__traceback__)


def console_safe_exception(exc):
    """Make an exception tree printable on the console code page while preserving types/chains."""
    order, pending, seen = [], [exc], set()
    while pending:
        e = pending.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        order.append((e, _console_safe_one(e)))
        pending += [e.__cause__, e.__context__, *_group_members(e)]
    unsafe = {id(e) for e, ok in order if not ok}
    grown = True
    while grown:
        grown = False
        for e, _ in order:
            if id(e) not in unsafe and any(id(m) in unsafe for m in _group_members(e)):
                unsafe.add(id(e))
                grown = True
    stand = {id(e): _stand_in(e) for e, _ in order if id(e) in unsafe}
    for e in [e for e, _ in order if id(e) not in unsafe] + list(stand.values()):
        for attr in ("__cause__", "__context__"):
            link = getattr(e, attr)
            if link is not None and id(link) in stand:
                setattr(e, attr, stand[id(link)])
    return stand.get(id(exc), exc)


def console_safe_errors(fn):
    """Wrap a ComfyUI entry point so any raised exception is console-safe."""
    if getattr(fn, "__qf_console_safe__", False):
        return fn

    @functools.wraps(fn)
    def wrapper(*args, _safe_exc=console_safe_exception, _is_finalizing=sys.is_finalizing, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if _safe_exc is None or (_is_finalizing is not None and _is_finalizing()):
                raise
            safe = _safe_exc(exc)
            if safe is exc:
                raise
        raise safe
    wrapper.__qf_console_safe__ = True
    try:
        wrapper.__signature__ = inspect.signature(fn)
    except (TypeError, ValueError):
        pass
    return wrapper


def _console_safe_attr(cls, name):
    raw = inspect.getattr_static(cls, name)
    if isinstance(raw, (classmethod, staticmethod)):
        setattr(cls, name, type(raw)(console_safe_errors(raw.__func__)))
    elif isinstance(raw, property):
        fns = (f and console_safe_errors(f) for f in (raw.fget, raw.fset, raw.fdel))
        setattr(cls, name, property(*fns, raw.__doc__))
    elif isinstance(raw, types.FunctionType):
        setattr(cls, name, console_safe_errors(raw))


def console_safe_methods(cls):
    """Wrap every ComfyUI-facing method/property and __init__ of cls."""
    for name in list(vars(cls)):
        if name == "__init__" or not (name.startswith("__") and name.endswith("__")):
            _console_safe_attr(cls, name)
    return cls


_NODE_ENTRY_POINTS = ("INPUT_TYPES", "VALIDATE_INPUTS", "IS_CHANGED", "check_lazy_status")


def console_safe_nodes(mapping):
    """Wrap every registered ComfyUI node class's FUNCTION and lifecycle hooks."""
    for cls in mapping.values():
        for name in (getattr(cls, "FUNCTION", None), *_NODE_ENTRY_POINTS):
            if isinstance(name, str) and hasattr(cls, name):
                _console_safe_attr(cls, name)
    return mapping


# ── Engine installer & resolver (consumer 4-SM single-arch plugins) ──────────

_ENGINE_BASE_URL = "https://www.modelscope.cn/models/QuantFunc/Plugin/resolve/master"
_ENGINE_PLATFORMS = {
    "linux": {"key": "linux", "arches": ("x86_64",), "kernel": True,
              "hosts": {13: "libquantfunc.so", 12: "libquantfunc-12.so"}},
    "windows": {"key": "win32", "arches": ("AMD64",), "kernel": True,
                "hosts": {13: "quantfunc.dll", 12: "quantfunc-12.dll"}},
}
_ENGINE_KERNEL_RE = re.compile(r"libquantfunc_kernels(?:-12)?\.so")
_ENGINE_KERNEL_WIN_RE = re.compile(r"libquantfunc_kernels(?:-12)?\.dll", re.IGNORECASE)
_ENGINE_ADDON_RE = re.compile(r"[A-Za-z0-9][-A-Za-z0-9_.]{0,63}\.(?:so|dll|dylib)")
_ENGINE_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
_ENGINE_SET_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
_ENGINE_SETS_SCHEMA = 2
_ENGINE_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ENGINE_PAIR_ID_RE = re.compile(r"[0-9a-f]{32}")
_ENGINE_MARKER_RE = re.compile(r"\.engine-([a-z][a-z0-9_]{0,31})-cu(\d+)\.json")
_ENGINE_PAIR_RE = re.compile(r"(\d+\.\d+\.\d+)-([a-z][a-z0-9_]{0,31})-cu(\d+)")
_ENGINE_LOCAL_BUILD_LOCK = ".dev_lib_lock"
_ENGINE_INSTALL_LOCKFILE = ".engine-install.lock"
_ENGINE_REINSTALLED = ".reinstalled"
_ENGINE_VERIFY_SCHEMA_MAX = 1
_ENGINE_HTTP_TIMEOUT_S = 120
_ENGINE_PLUGIN_VERSION_FILE = "version.json"

_ENGINE_STATUS = {"state": "idle", "detail": ""}
_ENGINE_STATUS_LOCK = threading.Lock()
_ENGINE_DEVICE = 0


class EngineNotInstallable(RuntimeError):
    """This machine cannot take a published PC Engine build."""


def _engine_status(state, detail=""):
    with _ENGINE_STATUS_LOCK:
        _ENGINE_STATUS.update(state=state, detail=detail)


def engine_install_status():
    """(state, detail): idle | checking | downloading | installed | offline | unavailable | failed | local."""
    with _ENGINE_STATUS_LOCK:
        return _ENGINE_STATUS["state"], _ENGINE_STATUS["detail"]


def _engine_bin_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", _BIN_SUBDIR)


def _engine_platform():
    return _ENGINE_PLATFORMS.get(_BIN_SUBDIR)


def _torch_cuda_major():
    try:
        import torch
        return int(str(torch.version.cuda).split(".")[0]) if torch.version.cuda else None
    except Exception:  # noqa: BLE001
        return None


def _driver_cuda_major():
    try:
        cuda = ctypes.CDLL("nvcuda.dll" if _BIN_SUBDIR == "windows" else "libcuda.so.1")
        v = ctypes.c_int(0)
        if cuda.cuDriverGetVersion(ctypes.byref(v)) != 0:
            return None
        return v.value // 1000
    except Exception:  # noqa: BLE001
        return None


def _gpu_sm(device_idx=0):
    try:
        import torch
        major, minor = torch.cuda.get_device_capability(int(device_idx))
        return major * 10 + minor
    except Exception:  # noqa: BLE001
        return None


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every non-HTTPS redirect hop before any plaintext request is sent."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        resolved = urllib.parse.urljoin(req.full_url, newurl)
        if not resolved.startswith("https://"):
            raise RuntimeError(f"refusing a redirect to a non-HTTPS engine URL: {resolved}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _engine_http_open(url):
    if not url.startswith("https://"):
        raise RuntimeError(f"refusing a non-HTTPS engine URL: {url}")
    opener = urllib.request.build_opener(_HttpsOnlyRedirectHandler)
    resp = opener.open(url, timeout=_ENGINE_HTTP_TIMEOUT_S)
    if not resp.geturl().startswith("https://"):
        resp.close()
        raise RuntimeError(f"refusing a redirect to a non-HTTPS engine URL: {resp.geturl()}")
    return resp


def _engine_http_get(url):
    with _engine_http_open(url) as resp:
        return resp.read()


def _harden_posix_mode(path, mode=0o755):
    """Harden a non-symlink POSIX file or directory without an islink->chmod TOCTOU window."""
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fchmod(fd, mode)
    except OSError:
        pass
    finally:
        os.close(fd)


def _refuse_symlink_dir(path, label):
    if os.path.islink(path):
        raise RuntimeError(f"qf_native: refusing symlinked {label} directory: {path}")


def _engine_fetch_to(url, dest, label):
    import hashlib
    h, done = hashlib.sha256(), 0
    with _engine_http_open(url) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            _engine_status("downloading", f"{label}: {done >> 20} of {total >> 20 if total else '?'} MB")
        f.flush()
        os.fsync(f.fileno())
    _harden_posix_mode(dest, 0o755)
    return h.hexdigest()


def _sha256_of(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _version_key(v):
    return tuple(int(x) for x in v.split("."))


def engine_choice(device_idx=0):
    """(cuda_major, sm) for this machine; raises EngineNotInstallable if unsupported."""
    torch_major, driver_major = _torch_cuda_major(), _driver_cuda_major()
    if torch_major is None:
        raise EngineNotInstallable(
            "PyTorch reports no CUDA version (a CPU-only PyTorch?); the QuantFunc engine needs a "
            "CUDA build of PyTorch; no engine was installed")
    hosts = _engine_platform()["hosts"]
    if torch_major not in hosts:
        raise EngineNotInstallable(
            f"no QuantFunc engine is published for CUDA {torch_major} "
            f"(torch CUDA {torch_major}, driver CUDA {driver_major}); published: CUDA {sorted(hosts)}")
    if driver_major is not None and driver_major < torch_major:
        raise EngineNotInstallable(
            f"your NVIDIA driver runs up to CUDA {driver_major}, but torch uses CUDA {torch_major}: "
            f"update the driver; no engine was installed")
    sm = _gpu_sm(device_idx)
    if sm is None:
        raise EngineNotInstallable("no CUDA GPU is visible to torch; no engine was installed")
    if sm not in _CONSUMER_SMS:
        supported = ", ".join(f"{s // 10}.{s % 10}" for s in _CONSUMER_SMS)
        raise EngineNotInstallable(
            f"no QuantFunc PC Engine is published for this GPU's architecture (SM {sm // 10}.{sm % 10}); "
            f"supported consumer GPUs: SM {supported}; no engine was installed")
    return torch_major, sm


def _engine_pick_version(versions, plugin_version, major):
    suffix = "-12" if major == 12 else ""
    best = best_lib = None
    for key, info_entry in (versions or {}).items():
        if not (isinstance(key, str) and _ENGINE_VERSION_RE.fullmatch(key)
                and isinstance(info_entry, dict) and info_entry.get("kernel_so")):
            continue
        req = info_entry.get("comfy" + suffix, info_entry.get("comfy"))
        lib = info_entry.get("lib" + suffix, info_entry.get("lib", key))
        if not all(isinstance(x, str) and _ENGINE_VERSION_RE.fullmatch(x) for x in (req, lib)):
            continue
        if _version_key(req) <= _version_key(plugin_version) and (
                best is None or _version_key(lib) > _version_key(best_lib)):
            best, best_lib = key, lib
    return best


def _engine_local_choice():
    if os.environ.get(_ENV_SO_OVERRIDE, "").strip():
        return f"{_ENV_SO_OVERRIDE} names the engine library"
    if os.path.exists(os.path.join(_engine_bin_dir(), _ENGINE_LOCAL_BUILD_LOCK)):
        return f"bin/{_BIN_SUBDIR}/{_ENGINE_LOCAL_BUILD_LOCK} keeps the local build bin/{_BIN_SUBDIR}/{_LIB_BASENAME}"
    return None


def _pair_dir(m):
    return f"{m['version']}-{m['set']}-cu{m['cuda']}"


def _kernel_regex(is_pe=None):
    use_win = (_BIN_SUBDIR == "windows") if is_pe is None else bool(is_pe)
    return _ENGINE_KERNEL_WIN_RE if use_win else _ENGINE_KERNEL_RE


def _kernel_filename(major=13, is_pe=None):
    use_win = (_BIN_SUBDIR == "windows") if is_pe is None else bool(is_pe)
    suffix = "-12" if major == 12 else ""
    ext = "dll" if use_win else "so"
    return f"libquantfunc_kernels{suffix}.{ext}"


def _read_marker(path):
    name = _ENGINE_MARKER_RE.fullmatch(os.path.basename(path))
    plat = _engine_platform()
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        files = [m["host"], m["kernel"]] if plat["kernel"] else [m["host"]]
        ok = (name is not None and m["set"] == name.group(1) and type(m["cuda"]) is int
              and m["cuda"] == int(name.group(2))
              and _ENGINE_VERSION_RE.fullmatch(m["version"]) is not None
              and m["host"] == plat["hosts"].get(m["cuda"])
              and (_kernel_regex().fullmatch(m["kernel"]) is not None
                   and m["kernel"].lower() == _kernel_filename(m["cuda"]).lower()
                   if plat["kernel"] else m["kernel"] is None)
              and all(type(s) is int and s in _CONSUMER_SMS for s in m["sms"])
              and sorted(m["sha256"]) == sorted(files)
              and all(_ENGINE_SHA256_RE.fullmatch(h) for h in m["sha256"].values()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return m if ok else None


def _markers():
    d = _engine_bin_dir()
    try:
        names = sorted(n for n in os.listdir(d) if _ENGINE_MARKER_RE.fullmatch(n))
    except OSError:
        return []
    return [(os.path.join(d, n), m) for n in names for m in (_read_marker(os.path.join(d, n)),) if m]


def _installed_pair(device_id=None):
    dev = _ENGINE_DEVICE if device_id is None else int(device_id)
    major, sm = _torch_cuda_major(), _gpu_sm(dev)
    mine = [(p, m) for p, m in _markers() if m["cuda"] == major and sm in m["sms"]]
    return max(mine, key=lambda pm: _version_key(pm[1]["version"])) if mine else (None, None)


def _claim(bin_dir, marker, sms, chosen=None):
    major = int(_ENGINE_MARKER_RE.fullmatch(os.path.basename(marker)).group(2))
    want = sorted(set(sms))
    for p, m in _markers():
        if m["cuda"] != major or p == marker:
            continue
        left = [s for s in m["sms"] if s not in sms]
        if left == m["sms"]:
            continue
        if left:
            _engine_write_file(p, json.dumps(dict(m, sms=left)).encode("utf-8"))
        else:
            os.remove(p)
            shutil.rmtree(os.path.join(bin_dir, _pair_dir(m)), ignore_errors=True)
    if chosen is None:
        chosen = _read_marker(marker)
        if chosen is None or sorted(chosen["sms"]) == want:
            return
        chosen = dict(chosen, sms=want)
    _engine_write_file(marker, json.dumps(chosen).encode("utf-8"))


def _marker_of(so_path):
    here = os.path.realpath(os.path.dirname(so_path))
    return next(((p, m) for p, m in _markers()
                 if os.path.realpath(os.path.join(_engine_bin_dir(), _pair_dir(m))) == here), (None, None))


def _pair_intact(m):
    pair = os.path.join(_engine_bin_dir(), _pair_dir(m))
    try:
        return all(_sha256_of(os.path.join(pair, n)) == h for n, h in m["sha256"].items())
    except OSError:
        return False


def _manifest_key(gpu_set, name, major=None, hashes=None):
    plat = _engine_platform()
    if plat["kernel"] and name in plat["hosts"].values():
        return name
    return f"{gpu_set}/{name}"


def _resolve_kernel_for_set(version, gpu_set, sm, major, hashes):
    """Select the exact consumer per-SM kernel plugin from verify.json without reading host imports."""
    cand = _kernel_filename(major)
    if _BIN_SUBDIR == "windows":
        prefix = f"{gpu_set}/"
        found = [k[len(prefix):] for k in hashes
                 if k.startswith(prefix) and k[len(prefix):].lower() == cand.lower()]
        if len(found) == 1:
            return found[0]
    elif _manifest_key(gpu_set, cand) in hashes:
        return cand
    raise RuntimeError(
        f"the {version} manifest has no per-SM CUDA kernel plugin for {gpu_set} (SM {sm}, CUDA {major})")


def _engine_sets(version, hashes):
    if "sets.json" not in hashes:
        raise RuntimeError(f"the {version} release publishes no sets.json, so its GPU classes are unknown; no engine "
                           f"was installed")
    raw = _engine_http_get(f"{_ENGINE_BASE_URL}/{version}/{_BIN_SUBDIR}/sets.json")
    import hashlib
    if hashlib.sha256(raw).hexdigest() != hashes["sets.json"]:
        raise RuntimeError(f"the {version} sets.json does not match its published SHA-256")
    doc = json.loads(raw)
    schema = doc.get("schema") if isinstance(doc, dict) else None
    if not (type(schema) is int and schema == _ENGINE_SETS_SCHEMA):
        raise RuntimeError(f"the {version} sets.json has schema {schema!r}, and this plugin reads schema "
                           f"{_ENGINE_SETS_SCHEMA} (one kernel plugin per consumer SM); no engine was installed")
    sets = doc.get("sets")
    if not (isinstance(sets, dict) and sets and all(
            isinstance(k, str) and _ENGINE_SET_RE.fullmatch(k) and isinstance(v, list) and v
            and all(type(s) is int for s in v) for k, v in sets.items())):
        raise RuntimeError(f"the {version} sets.json is not a GPU-class map this plugin understands")
    consumer = {k: [s for s in v if s in _CONSUMER_SMS] for k, v in sets.items()}
    consumer = {k: v for k, v in consumer.items() if v}
    if not consumer:
        raise RuntimeError(f"the {version} sets.json publishes no consumer GPU architectures")
    return consumer


def _install_addons(pair, version, info_entry, gpu_set, major, hashes):
    names = info_entry.get("addons-12" if major == 12 else "addons")
    plat = _engine_platform()
    reserved = set(plat["hosts"].values()) if plat else set()
    kpat = _kernel_regex()
    for name in names if isinstance(names, list) else ():
        if not (isinstance(name, str) and _ENGINE_ADDON_RE.fullmatch(name)) or name in reserved or kpat.fullmatch(name):
            continue
        key = _manifest_key(gpu_set, name)
        want, dest, part = hashes.get(key), os.path.join(pair, name), os.path.join(pair, f".{name}.part")
        if want is None:
            continue
        try:
            if not os.path.islink(dest) and os.path.isfile(dest) and _sha256_of(dest) == want:
                _harden_posix_mode(dest, 0o755)
                continue
            if _engine_fetch_to(f"{_ENGINE_BASE_URL}/{version}/{_BIN_SUBDIR}/{key}", part,
                                f"engine {version}: {name}") != want:
                raise RuntimeError(f"{key} does not match its published SHA-256")
            _harden_posix_mode(part, 0o755)
            os.replace(part, dest)
        except Exception as e:  # noqa: BLE001
            for leftover in (part, dest):
                try:
                    os.remove(leftover)
                except OSError:
                    pass
            say(f"[qf_native] engine add-on {name} was not installed ({type(e).__name__}: {e})", flush=True)


def _engine_write_file(path, data):
    tmp = f"{path}.part-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    _harden_posix_mode(tmp, 0o644)
    os.replace(tmp, path)


def install_engine(device_idx=None):
    """Fetch, SHA-256 verify, and atomically install the consumer per-SM engine pair."""
    why = _engine_local_choice()
    if why:
        _engine_status("local", why)
        say(f"[qf_native] QuantFunc engine install skipped: {why}", flush=True)
        return None
    plat = _engine_platform()
    if plat is None:
        if any(os.path.isfile(os.path.join(d, _LIB_BASENAME))
               for d in (_engine_bin_dir(), os.path.dirname(os.path.abspath(__file__)))):
            _engine_status("local", f"bin/{_BIN_SUBDIR}/{_LIB_BASENAME}")
            return None
        raise EngineNotInstallable(
            f"QuantFunc engines are published for Linux and Windows only; put the engine library "
            f"in bin/{_BIN_SUBDIR}/")
    if platform.machine() not in plat["arches"]:
        raise EngineNotInstallable(
            f"QuantFunc engines are published for {'/'.join(plat['arches'])} only, and this "
            f"machine is {platform.machine() or 'unknown'}; no engine was installed")
    bin_dir = _engine_bin_dir()
    _refuse_symlink_dir(bin_dir, "engine bin")
    os.makedirs(bin_dir, exist_ok=True)
    _harden_posix_mode(bin_dir, 0o755)
    with _install_lock(os.path.join(bin_dir, _ENGINE_INSTALL_LOCKFILE)):
        return _install_pair(bin_dir, _ENGINE_DEVICE if device_idx is None else int(device_idx))


@contextlib.contextmanager
def _install_lock(path):
    with open(path, "a+b") as f:
        if _BIN_SUBDIR == "windows":
            import errno
            import msvcrt
            f.seek(0)
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as e:
                    if e.errno != errno.EACCES:
                        raise
                    time.sleep(1)
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
            yield


def _install_pair(bin_dir, device_idx):
    _engine_status("checking")
    major, sm = engine_choice(device_idx)
    try:
        with open(os.path.join(bin_dir, _ENGINE_PLUGIN_VERSION_FILE), encoding="utf-8") as f:
            plugin_version = str(json.load(f)["comfy"])
    except (OSError, ValueError, KeyError) as e:
        raise RuntimeError(f"cannot read this plugin's version from bin/{_BIN_SUBDIR}/{_ENGINE_PLUGIN_VERSION_FILE}: {e}")
    if not _ENGINE_VERSION_RE.fullmatch(plugin_version):
        raise RuntimeError(f"malformed plugin version {plugin_version!r}")
    plat = _engine_platform()
    versions = json.loads(_engine_http_get(f"{_ENGINE_BASE_URL}/version.json")).get(plat["key"])
    version = _engine_pick_version(versions, plugin_version, major)
    if version is None:
        raise EngineNotInstallable(
            f"no published engine with the per-architecture layout is compatible with this plugin "
            f"({plugin_version}, CUDA {major})")
    manifest = json.loads(_engine_http_get(f"{_ENGINE_BASE_URL}/{version}/verify.json"))
    if not (isinstance(manifest, dict) and isinstance(manifest.get("schema"), int)
            and 1 <= manifest["schema"] <= _ENGINE_VERIFY_SCHEMA_MAX and isinstance(manifest.get(plat["key"]), dict)):
        raise RuntimeError(f"the {version} verify.json is not a manifest this plugin understands")
    hashes = manifest[plat["key"]]
    sets = _engine_sets(version, hashes)
    gpu_set = next((k for k, sms in sets.items() if sm in sms), None)
    if gpu_set is None:
        published = ", ".join(f"{s // 10}.{s % 10}" for s in sorted({s for v in sets.values() for s in v}))
        raise EngineNotInstallable(
            f"no QuantFunc engine is published for this GPU's architecture (SM {sm // 10}.{sm % 10}); "
            f"the {version} release has kernels for SM {published}; no engine was installed")
    host = plat["hosts"][major]
    marker = os.path.join(bin_dir, f".engine-{gpu_set}-cu{major}.json")
    have = _read_marker(marker)
    known_bad = None
    if have and have["version"] == version:
        if all(hashes.get(_manifest_key(gpu_set, n, major, hashes)) == h for n, h in have["sha256"].items()):
            pair_have = os.path.join(bin_dir, _pair_dir(have))
            _refuse_symlink_dir(pair_have, "engine pair")
            _harden_posix_mode(pair_have, 0o755)
            for n in have["sha256"]:
                _harden_posix_mode(os.path.join(pair_have, n), 0o755)
            _harden_posix_mode(marker, 0o644)
            _claim(bin_dir, marker, sets[gpu_set])
            _install_addons(pair_have, version, versions[version], gpu_set, major, hashes)
            _engine_status("installed", f"engine {version} ({gpu_set}, CUDA {major})")
            return have
        known_bad, have = version, None
    kernel = _resolve_kernel_for_set(version, gpu_set, sm, major, hashes) if plat["kernel"] else None
    pair = os.path.join(bin_dir, f"{version}-{gpu_set}-cu{major}")
    _refuse_symlink_dir(pair, "engine pair")
    os.makedirs(pair, exist_ok=True)
    _harden_posix_mode(pair, 0o755)
    order = ([kernel] if kernel else []) + [host]
    got, parts, backups, replaced = {}, [], {}, []
    m = {"version": version, "set": gpu_set, "cuda": major, "sms": sets[gpu_set], "host": host, "kernel": kernel,
         "sha256": got}
    rollback_failed = False
    try:
        for name in ((host, kernel) if kernel else (host,)):
            key = _manifest_key(gpu_set, name, major, hashes)
            if not _ENGINE_SHA256_RE.fullmatch(str(hashes.get(key, ""))):
                raise RuntimeError(f"the {version} manifest has no SHA-256 for {key}")
            part_path = os.path.join(pair, f".{name}.part")
            parts.append(part_path)
            got[name] = _engine_fetch_to(
                f"{_ENGINE_BASE_URL}/{version}/{_BIN_SUBDIR}/{key}", part_path, f"engine {version}: {name}")
            if got[name] != hashes[key]:
                raise RuntimeError(f"{key} does not match its published SHA-256 (download corrupt or tampered)")
            _harden_posix_mode(part_path, 0o755)
        if kernel:
            if _BIN_SUBDIR == "windows":
                host_cuda = _host_forbidden_cuda_pe(parts[0])
                if host_cuda:
                    raise RuntimeError(
                        f"the {version} host library unexpectedly imports CUDA ({', '.join(host_cuda)}); "
                        f"PC Engine host must have zero CUDA PE imports")
            else:
                host_cuda = _host_forbidden_cuda(parts[0])
                if host_cuda:
                    raise RuntimeError(
                        f"the {version} host library unexpectedly links CUDA ({', '.join(host_cuda)}); "
                        f"PC Engine host must have zero CUDA DT_NEEDED imports")
                ids = [_elf_pair_id(p) for p in parts]
                if ids[0] is None or ids[0] != ids[1]:
                    raise RuntimeError(
                        f"the {version} host and kernel libraries are not one build (pair ids {ids[0]} / "
                        f"{ids[1]}); nothing was installed")
        for n in order:
            dst = os.path.join(pair, n)
            if os.path.exists(dst):
                bak = os.path.join(pair, f".{n}.bak-{os.getpid()}")
                shutil.copy2(dst, bak)
                _harden_posix_mode(bak, 0o755)
                backups[n] = bak
        for n in order:
            part_path = os.path.join(pair, f".{n}.part")
            _harden_posix_mode(part_path, 0o755)
            os.replace(part_path, os.path.join(pair, n))
            replaced.append(n)
        _install_addons(pair, version, versions[version], gpu_set, major, hashes)
        _claim(bin_dir, marker, sets[gpu_set], chosen=m)
    except Exception as e:
        for n in reversed(replaced):
            dst = os.path.join(pair, n)
            bak = backups.get(n)
            try:
                if bak is not None and os.path.exists(bak):
                    os.replace(bak, dst)
                    _harden_posix_mode(dst, 0o755)
                elif os.path.exists(dst):
                    os.remove(dst)
            except OSError:
                rollback_failed = True
        if rollback_failed:
            try:
                os.remove(marker)
            except OSError:
                pass
            raise RuntimeError(
                f"the installed engine {version} update failed ({type(e).__name__}: {e}) and "
                f"rollback failed; stale marker removed so mismatched libraries will not load") from e
        if known_bad:
            raise RuntimeError(
                f"the installed engine {known_bad} no longer matches its published SHA-256 and the "
                f"re-download failed ({type(e).__name__}: {e}); existing installation was preserved") from e
        raise
    finally:
        for p in parts + list(backups.values()):
            try:
                os.remove(p)
            except OSError:
                pass
    keep = {_pair_dir(m), have and _pair_dir(have)}
    for d in os.listdir(bin_dir):
        p = _ENGINE_PAIR_RE.fullmatch(d)
        if p and p.group(2) == gpu_set and int(p.group(3)) == major and d not in keep:
            shutil.rmtree(os.path.join(bin_dir, d), ignore_errors=True)
    _engine_status("installed", f"engine {version} ({gpu_set}, CUDA {major})")
    say(f"[qf_native] installed QuantFunc engine {version} for {gpu_set} GPUs, CUDA {major}", flush=True)
    return m


def _engine_load_failed(so_path, err):
    detail = str(err)[:400]
    mpath, m = _marker_of(so_path)
    if not m:
        return f"the engine library failed to load: {detail}"
    flag = os.path.join(os.path.dirname(mpath), _pair_dir(m), _ENGINE_REINSTALLED)
    if os.path.exists(flag):
        return (f"the engine library {m['version']} failed to load again after one re-download: {detail}. "
                f"It is not re-downloaded again for this release: check the NVIDIA driver and the GPU.")
    _engine_write_file(flag, m["version"].encode("utf-8"))
    try:
        os.remove(mpath)
    except OSError:
        pass
    start_engine_install()
    return (f"the engine library {m['version']} failed to load: {detail}. It is being re-downloaded once; "
            f"queue the prompt again when the console says the engine is installed.")


def _engine_load_ok(so_path):
    mpath, m = _marker_of(so_path)
    if m:
        try:
            os.remove(os.path.join(os.path.dirname(mpath), _pair_dir(m), _ENGINE_REINSTALLED))
        except OSError:
            pass


def start_engine_install(device_idx=None):
    """Run install_engine() on a daemon thread at plugin import."""
    global _ENGINE_DEVICE
    if device_idx is not None:
        _ENGINE_DEVICE = int(device_idx)

    def _run():
        try:
            install_engine()
        except EngineNotInstallable as e:
            _engine_status("unavailable", str(e))
            say(f"[qf_native] QuantFunc engine not installed: {e}", flush=True)
        except Exception as e:  # noqa: BLE001
            kept = _installed_pair()[1]
            _engine_status("offline" if kept else "failed", f"{type(e).__name__}: {e}")
            say(f"[qf_native] QuantFunc engine update failed ({type(e).__name__}: {e}); "
                f"{'the installed engine ' + kept['version'] + ' stays in use' if kept else 'no engine is installed'}",
                flush=True)

    t = threading.Thread(target=_run, name="qf-engine-install", daemon=True)
    t.start()
    return t


def _local_host_names(plat):
    names = []
    if plat is not None:
        preferred = plat["hosts"].get(_torch_cuda_major())
        if preferred:
            names.append(preferred)
    if _LIB_BASENAME not in names:
        names.append(_LIB_BASENAME)
    return names


def has_ready_so(device_id=None):
    """Read-only probe mirroring resolve_so_path() priority without deleting markers or downloading."""
    if _LIB is not None:
        return True
    try:
        override = os.environ.get(_ENV_SO_OVERRIDE, "").strip()
        if override:
            return os.path.isfile(override)
        bin_dir = _engine_bin_dir()
        pkg = os.path.dirname(os.path.abspath(__file__))
        plat = _engine_platform()
        local = os.path.exists(os.path.join(bin_dir, _ENGINE_LOCAL_BUILD_LOCK))
        if local or plat is None:
            names = _local_host_names(plat)
            return any(os.path.isfile(os.path.join(d, name)) for name in names for d in (bin_dir, pkg))
        if os.path.islink(bin_dir):
            return False
        _, m = _installed_pair(device_id=device_id)
        if not m:
            return False
        pair = os.path.join(bin_dir, _pair_dir(m))
        if os.path.islink(pair):
            return False
        return _pair_intact(m)
    except OSError:
        return False


def resolve_so_path():
    """Resolve the engine library path without any workflow-serializable input."""
    pkg = os.path.dirname(os.path.abspath(__file__))
    bin_dir = _engine_bin_dir()
    override = os.environ.get(_ENV_SO_OVERRIDE, "").strip()
    if override:
        if not os.path.isfile(override):
            raise RuntimeError(f"qf_native: {_ENV_SO_OVERRIDE} names {override!r}, which is not a file")
        return os.path.realpath(override)
    plat = _engine_platform()
    local = os.path.exists(os.path.join(bin_dir, _ENGINE_LOCAL_BUILD_LOCK))
    if local or plat is None:
        names = _local_host_names(plat)
        for name in names:
            for d in (bin_dir, pkg):
                c = os.path.join(d, name)
                if os.path.isfile(c):
                    return os.path.realpath(c)
        missing = " or ".join(f"bin/{_BIN_SUBDIR}/{n}" for n in names)
        raise RuntimeError(
            "qf_native: no engine library found: "
            + (f"bin/{_BIN_SUBDIR}/{_ENGINE_LOCAL_BUILD_LOCK} marks a local build, but "
               f"{missing} is not there" if local else
               f"put {_LIB_BASENAME} in the package bin/{_BIN_SUBDIR}/, or set {_ENV_SO_OVERRIDE}=<abs path>"))
    _refuse_symlink_dir(bin_dir, "engine bin")
    mpath, m = _installed_pair()
    if m:
        pair = os.path.join(bin_dir, _pair_dir(m))
        _refuse_symlink_dir(pair, "engine pair")
        if _pair_intact(m):
            _harden_posix_mode(bin_dir, 0o755)
            _harden_posix_mode(pair, 0o755)
            for n in m["sha256"]:
                _harden_posix_mode(os.path.join(pair, n), 0o755)
            return os.path.realpath(os.path.join(pair, m["host"]))
        try:
            os.remove(mpath)
        except OSError:
            pass
        start_engine_install()
        raise RuntimeError(
            f"qf_native: the installed QuantFunc engine {m['version']} does not match the SHA-256 "
            f"recorded when it was installed (a file changed on disk), so it is NOT loaded. It is being "
            f"re-downloaded: queue the prompt again when the console says the engine is installed.")
    state, detail = engine_install_status()
    if state in ("checking", "downloading"):
        raise RuntimeError(
            f"qf_native: the QuantFunc engine library is still downloading ({detail}). Queue the prompt "
            f"again when the console says the engine is installed.")
    raise RuntimeError(
        f"qf_native: no QuantFunc engine is installed for this GPU ({state}: {detail or 'no install attempted'}). "
        f"The plugin installs it automatically at ComfyUI start; the console says why it did not. A local build: put "
        f"{_LIB_BASENAME} in bin/{_BIN_SUBDIR}/ and create bin/{_BIN_SUBDIR}/{_ENGINE_LOCAL_BUILD_LOCK}.")


# ── Pure-stdlib ELF / PE readers & zero-CUDA-host toolchain guard ────────────

def _elf_sections(data):
    if data[:4] != b"\x7fELF":
        return []
    is64 = data[4] == 2
    end = "<" if data[5] == 1 else ">"
    if is64:
        shoff = struct.unpack_from(end + "Q", data, 0x28)[0]
        shentsize, shnum, shstrndx = struct.unpack_from(end + "HHH", data, 0x3a)
    else:
        shoff = struct.unpack_from(end + "I", data, 0x20)[0]
        shentsize, shnum, shstrndx = struct.unpack_from(end + "HHH", data, 0x2e)
    raw = []
    for i in range(shnum):
        b = shoff + i * shentsize
        if is64:
            raw.append(struct.unpack_from(end + "II", data, b)
                       + struct.unpack_from(end + "QQ", data, b + 0x18)
                       + struct.unpack_from(end + "I", data, b + 0x28))
        else:
            raw.append(struct.unpack_from(end + "II", data, b)
                       + struct.unpack_from(end + "III", data, b + 0x10))
    names = raw[shstrndx][2] if shstrndx < len(raw) else None
    return [("" if names is None else _cstr(data, names + n), t, o, s, l) for (n, t, o, s, l) in raw]


def _cstr(data, off):
    z = data.find(b"\x00", off)
    if z < 0:
        raise ValueError("unterminated ELF string")
    return data[off:z].decode("latin-1")


def _elf_map(path, read):
    try:
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            return read(data)
    except Exception:  # noqa: BLE001
        return None


def _elf_needed(path):
    """DT_NEEDED sonames of an ELF binary (pure stdlib; [] on parse error)."""
    def read(data):
        secs = _elf_sections(data)
        dyn = next(((o, s, l) for (_, t, o, s, l) in secs if t == _SHT_DYNAMIC), None)
        if not dyn or dyn[2] >= len(secs):
            return []
        dyn_off, dyn_sz, dyn_link = dyn
        dynstr_off = secs[dyn_link][2]
        is64 = data[4] == 2
        end = "<" if data[5] == 1 else ">"
        needed = []
        for off in range(dyn_off, dyn_off + dyn_sz, 16 if is64 else 8):
            tag, val = struct.unpack_from(end + ("qQ" if is64 else "iI"), data, off)
            if tag == _DT_NULL:
                break
            if tag == _DT_NEEDED:
                needed.append(_cstr(data, dynstr_off + val))
        return needed
    return _elf_map(path, read) or []


def _elf_pair_id(path):
    def read(data):
        sec = next(((o, s) for (n, _, o, s, _) in _elf_sections(data) if n == ".qf_pair_id"), None)
        pid = data[sec[0]:sec[0] + sec[1]].split(b"\x00")[0].decode("ascii") if sec else ""
        return pid if _ENGINE_PAIR_ID_RE.fullmatch(pid) else None
    return _elf_map(path, read)


def _is_elf(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except Exception:  # noqa: BLE001
        return False


def _is_pe(path):
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"MZ"
    except Exception:  # noqa: BLE001
        return False


def _pe_imports(path):
    """Lowercased DLL names imported by a Windows PE image (import + delay-load tables)."""
    def read(data):
        if data[:2] != b"MZ":
            return []
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            return []
        nsec, opt_size = struct.unpack_from("<H", data, pe + 6)[0], struct.unpack_from("<H", data, pe + 20)[0]
        opt = pe + 24
        dirs = opt + (112 if struct.unpack_from("<H", data, opt)[0] == 0x20B else 96)
        secs = [struct.unpack_from("<IIII", data, opt + opt_size + 40 * i + 8) for i in range(nsec)]

        def at(rva):
            return next(ptr + rva - va for vsize, va, raw, ptr in secs if va <= rva < va + max(vsize, raw))

        names = []
        for index, size, name_at in ((1, 20, 12), (13, 32, 4)):
            rva = struct.unpack_from("<I", data, dirs + 8 * index)[0]
            if not rva:
                continue
            off = at(rva)
            while True:
                name_rva = struct.unpack_from("<I", data, off + name_at)[0]
                if not name_rva:
                    break
                names.append(_cstr(data, at(name_rva)).lower())
                off += size
        return names
    return _elf_map(path, read) or []


_HOST_FORBIDDEN_CUDA_RE = re.compile(r"lib(?:cudart|cublas|cublasLt)\.so\.\d+")
_HOST_FORBIDDEN_CUDA_PE_RE = re.compile(r"(?:cudart|cublas(?:lt)?)64_\d+\.dll", re.IGNORECASE)
_CUDA_LIB_RE = re.compile(
    r"lib(?:cudart|cublas|cublasLt|cusolver|cusolverMg|cusparse|cufft|cufftw|curand|nvrtc|nvJitLink|cudnn\w*)\.so\.\d+")
_RTLD_DI_LINKMAP = 2


def _host_forbidden_cuda(so_path):
    """Return any forbidden CUDA runtime/cuBLAS DT_NEEDED entries on the Linux host libquantfunc.so."""
    return [n for n in _elf_needed(so_path) if _HOST_FORBIDDEN_CUDA_RE.fullmatch(n)]


def _host_forbidden_cuda_pe(dll_path):
    """Return any forbidden CUDA runtime/cuBLAS PE imports on the Windows host quantfunc.dll."""
    return [n for n in _pe_imports(dll_path) if _HOST_FORBIDDEN_CUDA_PE_RE.fullmatch(n)]


def _host_cuda_major(so_path, is_pe=None):
    use_win = _is_pe(so_path) if is_pe is None else bool(is_pe)
    stem = os.path.splitext(os.path.basename(so_path))[0]
    return 12 if (stem.lower() if use_win else stem).endswith("-12") else 13


def _find_kernel_in_dir(d, want, pat, is_pe):
    try:
        entries = sorted(n for n in os.listdir(d) if pat.fullmatch(n)
                         and os.path.isfile(os.path.join(d, n)))
    except OSError:
        return None
    for n in entries:
        if (n.lower() == want.lower()) if is_pe else (n == want):
            return os.path.join(d, n)
    return None


def _select_kernel_plugin(so_path):
    """Locate the paired CUDA kernel plugin (QF_KERNEL_PLUGIN_DIR -> marker -> beside host)."""
    is_pe = _is_pe(so_path)
    want = _kernel_filename(_host_cuda_major(so_path, is_pe), is_pe)
    pat = _kernel_regex(is_pe)
    so_dir = os.path.dirname(os.path.realpath(so_path))
    env_dir = os.environ.get(_ENV_KERNEL_PLUGIN_DIR, "").strip()
    if env_dir:
        found = _find_kernel_in_dir(env_dir, want, pat, is_pe)
        if found is not None:
            return found
    _, marker = _marker_of(so_path)
    if marker and marker.get("kernel") and (
            marker["kernel"].lower() == want.lower() if is_pe else marker["kernel"] == want):
        p = os.path.join(so_dir, marker["kernel"])
        if os.path.isfile(p):
            return p
    return _find_kernel_in_dir(so_dir, want, pat, is_pe)


def _so_cuda_major(bin_path):
    """CUDA major dynamically linked by an ELF kernel plugin or Windows PE DLL, or None."""
    for lib in _elf_needed(bin_path):
        m = re.match(r"libcudart\.so\.(\d+)", lib)
        if m:
            return int(m.group(1))
    imports = _pe_imports(bin_path)
    for pat in (r"cudart64_(\d+)\.dll", r"cublas(?:lt)?64_(\d+)\.dll"):
        majors = {int(m.group(1)) for m in (re.fullmatch(pat, n) for n in imports) if m}
        if majors:
            return majors.pop() if len(majors) == 1 else None
    return None


def assert_toolchain_compatible(so_path):
    """Fail-closed CUDA toolchain guard (O1 zero-CUDA host + per-SM kernel plugin match).

    On both Linux (ELF) and Windows (PE):
      1. Host library (libquantfunc.so / quantfunc.dll) MUST have zero CUDA runtime/cuBLAS imports.
      2. The selected per-SM kernel plugin (libquantfunc_kernels(?:-12)?.{so,dll}) MUST
         exist beside the host and link the same CUDA major as torch.
    """
    if os.environ.get(_ENV_ALLOW_UNVERIFIED_TOOLCHAIN, "").strip().lower() in ("1", "true", "yes"):
        return
    torch_major = _torch_cuda_major()
    is_elf, is_pe = _is_elf(so_path), _is_pe(so_path)
    if not is_elf and not is_pe:
        raise RuntimeError(
            f"qf_native: REFUSING to load - the CUDA-toolchain compatibility check reads Linux (ELF) and "
            f"Windows (PE) engine libraries only, and the engine binary ({so_path}) is a "
            f"{platform.system() or 'non-Linux'} binary whose CUDA version cannot be read here. torch is built for "
            f"CUDA {torch_major or 'none / CPU-only'}. Ensure your torch and the engine binary use the SAME CUDA "
            f"major, then set {_ENV_ALLOW_UNVERIFIED_TOOLCHAIN}=1 to proceed.")
    leaked = _host_forbidden_cuda(so_path) if is_elf else _host_forbidden_cuda_pe(so_path)
    if leaked:
        kind = "DT_NEEDED" if is_elf else "PE"
        raise RuntimeError(
            f"qf_native: REFUSING to load - host library ({so_path}) links CUDA ({', '.join(leaked)}). "
            f"QuantFunc PC Engine host library must have zero CUDA {kind} imports; CUDA belongs "
            f"exclusively in the per-SM kernel plugin.")
    target_bin = _select_kernel_plugin(so_path)
    if target_bin is None:
        want_name = _kernel_filename(_host_cuda_major(so_path, is_pe), is_pe)
        raise RuntimeError(
            f"qf_native: REFUSING to load - missing consumer CUDA kernel plugin "
            f"({want_name}) beside {so_path}. Place the plugin for your "
            f"GPU architecture (SM 7.5/8.6/8.9/12.0) and torch CUDA {torch_major or 'none / CPU-only'} in "
            f"that folder, restart ComfyUI to auto-install, or set {_ENV_ALLOW_UNVERIFIED_TOOLCHAIN}=1.")
    so_major = _so_cuda_major(target_bin)
    if torch_major is None or so_major is None:
        raise RuntimeError(
            f"qf_native: REFUSING to load - cannot verify the engine's CUDA toolchain matches torch's "
            f"(torch CUDA={torch_major or 'none / CPU-only'}, engine library CUDA major="
            f"{so_major if so_major is not None else 'undeterminable'} from {target_bin}). "
            f"Install a torch + engine built for the SAME CUDA major, "
            f"or set {_ENV_ALLOW_UNVERIFIED_TOOLCHAIN}=1 if you KNOW this combination is safe.")
    if torch_major != so_major:
        raise RuntimeError(
            f"qf_native: REFUSING to load - CUDA toolchain MISMATCH. torch is built for CUDA {torch_major} "
            f"but the engine kernel plugin ({target_bin}) links CUDA {so_major}. Use an engine built for "
            f"CUDA {torch_major}, or a torch built for CUDA {so_major}. "
            f"(Override only if you know it is safe: {_ENV_ALLOW_UNVERIFIED_TOOLCHAIN}=1.)")


def _sidecar_preloads(so_path):
    """Non-engine sidecar libraries in so_path's folder reachable via host DT_NEEDED."""
    d = os.path.dirname(so_path)
    try:
        present = set(os.listdir(d))
    except OSError:
        return []
    never = set((_engine_platform() or {}).get("hosts", {}).values()) | {os.path.basename(so_path)}
    want, todo = set(), [os.path.basename(so_path)]
    while todo:
        for n in _elf_needed(os.path.join(d, todo.pop())):
            if n in present and n not in want and n not in never and not _ENGINE_KERNEL_RE.fullmatch(n):
                want.add(n)
                todo.append(n)
    return sorted(want)


class _LinkMap(ctypes.Structure):
    _fields_ = [("l_addr", ctypes.c_void_p), ("l_name", ctypes.c_char_p)]


def _linker_path(soname):
    if _BIN_SUBDIR != "linux":
        return None
    try:
        lib = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)
        dlinfo = ctypes.CDLL(None).dlinfo
    except (OSError, AttributeError):
        return None
    dlinfo.argtypes, dlinfo.restype = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p], ctypes.c_int
    lm = ctypes.POINTER(_LinkMap)()
    if dlinfo(lib._handle, _RTLD_DI_LINKMAP, ctypes.byref(lm)) != 0 or not lm or not lm.contents.l_name:
        return None
    return os.fsdecode(lm.contents.l_name)


def _engine_cuda_needs(so_path):
    """CUDA toolkit sonames needed by the selected per-SM kernel plugin and local sidecar closure."""
    seeds = [so_path]
    plugin = _select_kernel_plugin(so_path)
    if plugin is not None:
        seeds.append(plugin)
    seen = {os.path.realpath(p) for p in seeds}
    todo = list(seeds)
    cuda = set()
    while todo:
        cur = todo.pop()
        cur_dir = os.path.dirname(cur)
        for n in _elf_needed(cur):
            if _CUDA_LIB_RE.fullmatch(n):
                cuda.add(n)
            else:
                cand = os.path.join(cur_dir, n)
                rp = os.path.realpath(cand)
                if rp not in seen and os.path.isfile(cand):
                    seen.add(rp)
                    todo.append(cand)
    return cuda


def _torch_cuda_plan(needs, linker, major):
    if major is None:
        return set(), set(), []
    anchors = {os.path.dirname(p) for p in (linker(f"libcudart.so.{major}"), linker(f"libcublas.so.{major}")) if p}
    dirs = set(anchors)
    parents = [os.path.dirname(os.path.dirname(a)) for a in anchors if os.path.basename(a) == "lib"]
    for parent in {p for p in parents if parents.count(p) > 1}:
        dirs.update(os.path.realpath(d) for d in glob.glob(os.path.join(parent, "*", "lib")))
    provided, preloads = set(), []
    for soname in sorted(needs):
        bound = linker(soname)
        if bound and os.path.dirname(os.path.realpath(bound)) in dirs:
            provided.add(soname)
            continue
        path = next((os.path.join(d, soname) for d in sorted(dirs) if os.path.exists(os.path.join(d, soname))), None)
        if path:
            provided.add(soname)
            if not bound:
                preloads.append(path)
            dirs.add(os.path.dirname(os.path.realpath(path)))
    return dirs, provided, preloads


def _assert_torch_cuda_family(provided, dirs, linker):
    for soname in sorted(provided):
        bound = linker(soname)
        if bound and os.path.dirname(os.path.realpath(bound)) not in dirs:
            raise RuntimeError(
                f"qf_native: REFUSING the engine - it links {soname}, and this process resolves it to "
                f"{os.path.realpath(bound)}, which is not torch's copy (torch's CUDA libraries are in "
                f"{', '.join(sorted(dirs))}). One process must use one cuBLAS/cuSOLVER set.")


def load_lib(device_id=None):
    """Load and bind the 25-symbol PC Engine host library once per process."""
    global _LIB, _LIB_PATH, _FINGERPRINT_PENDING, _ENGINE_DEVICE
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        prev_dev = _ENGINE_DEVICE
        if device_id is not None:
            _ENGINE_DEVICE = int(device_id)
        try:
            so_path = resolve_so_path()
            assert_toolchain_compatible(so_path)
            torch_major = _torch_cuda_major() if "torch" in sys.modules else None
            cuda_dirs, cuda_provided, cuda_preloads = _torch_cuda_plan(
                _engine_cuda_needs(so_path), _linker_path, torch_major)
            for p in cuda_preloads:
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_LOCAL)
                except OSError as e:
                    raise RuntimeError(f"qf_native: could not load torch's own {p}: {e}") from e
            so_dir = os.path.dirname(so_path)
            pending = [f for f in _sidecar_preloads(so_path) if f not in cuda_provided]
            for _ in range(max(1, len(pending))):
                still = []
                for f in pending:
                    try:
                        ctypes.CDLL(os.path.join(so_dir, f), mode=ctypes.RTLD_LOCAL)
                    except OSError:
                        still.append(f)
                if not still:
                    break
                pending = still
            ident = _file_identity(so_path)
            try:
                lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                raise RuntimeError(f"qf_native: {_engine_load_failed(so_path, e)}") from e
            _engine_load_ok(so_path)
            _assert_torch_cuda_family(cuda_provided, cuda_dirs, _linker_path)
            bound = _bind(lib)
            _LIB_PATH = so_path
            _LIB = bound
            if _LOG_LEVEL is not None:
                _LIB.qf_set_log_level(_LOG_LEVEL)
            _FINGERPRINT_PENDING = (_LIB, so_path, ident if _file_identity(so_path) == ident else None)
            _emit_fingerprint()
        except BaseException:
            _ENGINE_DEVICE = prev_dev
            raise
    return _LIB


def _file_identity(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def _log_lib_fingerprint(lib, so_path, ident):
    try:
        import hashlib
        h = hashlib.md5()
        with open(so_path, "rb") as fh:
            st = os.fstat(fh.fileno())
            if ident is None or (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) != ident:
                raise RuntimeError("the file at this path changed after the engine loaded it")
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        ver = "?"
        try:
            ver = (lib.qf_version() or b"?").decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        info("[qf_native] engine lib: %s  size=%d  mtime=%s  md5=%s  qf_version=%s"
             % (so_path, st.st_size, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                h.hexdigest(), ver), flush=True)
    except Exception as e:  # noqa: BLE001
        info("[qf_native] engine lib: %s (fingerprint unavailable: %r)" % (so_path, e), flush=True)


def last_err(lib):
    """Engine's last error text, console-safe."""
    e = lib.qf_last_error()
    return console_safe(e.decode("utf-8", "replace")) if e else "(none)"


def _enc(s):
    return s.encode("utf-8") if isinstance(s, str) else s


# ── Transformer creation, VRAM estimate & live handle ────────────────────────

_FORBIDDEN_CREATE_KEYS = frozenset({
    "cache_mode", "cache_thresh", "step_cache", "block_cache",
    "step_cache_thresh", "block_cache_thresh",
    "cat_keep_ratio", "cat_external_signal",
    "sparse", "sparse_cdf", "sol", "sol_s2", "s2_cubin_path",
    "quality", "video_enhance", "audio_enhance",
    "kernel_plugin_path", "denoise_only", "transformer_precision", "lora",
    "controlnet", "tiled_vae", "text_precision", "cpu_offload", "layer_offload",
})


def _refuse_session_knobs_in_create(config_json):
    """Reject runtime session knobs or retired keys anywhere inside create config_json."""
    cfg = config_json
    if isinstance(cfg, (str, bytes, bytearray)):
        try:
            cfg = json.loads(cfg)
        except Exception:  # noqa: BLE001
            return

    def _scan(obj):
        if isinstance(obj, dict):
            if any(k in _FORBIDDEN_CREATE_KEYS for k in obj):
                return True
            return any(_scan(v) for v in obj.values())
        if isinstance(obj, list):
            return any(_scan(v) for v in obj)
        return False

    if _scan(cfg):
        raise RuntimeError(
            "qf_native: runtime session or retired key must never appear in create config_json "
            "(it would poison the transformer cache identity and trigger full rebuilds).")


def _refuse_second_arch(device_id):
    if device_id == _ENGINE_DEVICE or _engine_platform() is None or _engine_local_choice():
        return
    sm, m = _gpu_sm(device_id), _installed_pair()[1]
    if sm is None or m is None or sm in m["sms"]:
        return
    raise RuntimeError(
        f"qf_native: GPU {device_id} is SM {sm // 10}.{sm % 10}, but this ComfyUI runs the QuantFunc engine "
        f"built for SM {' / '.join(f'{s // 10}.{s % 10}' for s in m['sms'])} (GPU {_ENGINE_DEVICE}). "
        f"Run one ComfyUI per GPU architecture (CUDA_VISIBLE_DEVICES).")


def make_create_params(*, weights_path, device_id=0, config_json=None):
    """Build QFXfmCreateParams without loading a library or mutating caller dicts."""
    device_id = operator.index(device_id)
    if not 0 <= device_id < (1 << 31):
        raise ValueError("device_id must fit nonnegative int32")
    _refuse_second_arch(device_id)
    _refuse_session_knobs_in_create(config_json)
    if isinstance(config_json, dict):
        config_json = dict(config_json)
    try:
        cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(cdir, exist_ok=True)
        if config_json is None:
            config_json = {"cache_dir": cdir}
        elif isinstance(config_json, dict):
            config_json.setdefault("cache_dir", cdir)
        else:
            parsed = json.loads(config_json)
            if isinstance(parsed, dict) and "cache_dir" not in parsed:
                parsed["cache_dir"] = cdir
                config_json = json.dumps(parsed)
    except Exception:  # noqa: BLE001
        pass
    p = QFXfmCreateParams(struct_size=ctypes.sizeof(QFXfmCreateParams), device_id=device_id)
    p._keep = [_enc(weights_path)]
    p.weights_path = p._keep[0]
    if config_json is not None:
        cj = json.dumps(config_json) if isinstance(config_json, dict) else config_json
        p._keep.append(_enc(cj))
        p.config_json = p._keep[-1]
    return p


def estimate_vram(lib, create_params):
    """Header-only VRAM estimate via qf_xfm_estimate_vram (no GPU allocation)."""
    fn = getattr(lib, "qf_xfm_estimate_vram", None)
    if fn is None:
        return ColdNeed(None, None)
    est = QFVramEstimate(struct_size=ctypes.sizeof(QFVramEstimate))
    st = fn(ctypes.byref(create_params), ctypes.byref(est))
    if st == QF_OK and (est.resident_weight_bytes or est.working_set_floor_bytes):
        return ColdNeed(
            int(est.resident_weight_bytes),
            None,
            int(est.min_offload_bytes),
            int(est.working_set_floor_bytes),
            int(est.recommended_offload_mode),
        )
    if st == QF_ERR_INVALID_ARG:
        raise RuntimeError(f"QuantFunc cold VRAM estimate refused: {last_err(lib)}")
    return ColdNeed(None, f"status {st}: {last_err(lib)}")


def query_device_caps(lib, device_id=0):
    """Query hardware capabilities via qf_device_query_caps."""
    caps = QFDeviceCaps(struct_size=ctypes.sizeof(QFDeviceCaps))
    st = lib.qf_device_query_caps(int(device_id), ctypes.byref(caps))
    if st != QF_OK:
        raise RuntimeError(f"qf_device_query_caps failed st={st}: {last_err(lib)}")
    return caps


def create_transformer(lib, *, create_params=None, **create_kwargs):
    """Create a pure transformer handle via qf_xfm_create."""
    if create_params is not None and create_kwargs:
        raise ValueError("use retained create_params or create keywords, not both")
    p = make_create_params(**create_kwargs) if create_params is None else create_params
    if not isinstance(p, QFXfmCreateParams):
        raise TypeError("creation requires QFXfmCreateParams")
    _refuse_session_knobs_in_create(p.config_json)
    handle = ctypes.c_void_p()
    st = lib.qf_xfm_create(ctypes.byref(p), ctypes.byref(handle))
    if st != QF_OK or not handle:
        detail = last_err(lib)
        if st == QF_ERR_AUTH:
            raise RuntimeError(
                f"QuantFunc API key missing or rejected: enter a valid key (qf_...) in the loader's api_key "
                f"field, set QUANTFUNC_API_KEY, or update bin/{_BIN_SUBDIR}/{_KEYFILE_BASENAME} ({detail}).")
        if st == QF_ERR_UNSUPPORTED:
            raise RuntimeError(
                f"QuantFunc PC Engine unsupported configuration or GPU (consumer SM 7.5/8.6/8.9/12.0 required): {detail}")
        raise RuntimeError(f"QuantFunc engine failed to load transformer (status {st}): {detail}")
    return handle


class QFEngineHandle:
    """Owns the loaded library, live qf_xfm_t handle, and single open session."""

    @classmethod
    def create(cls, lib, *, capacity_bytes=0, create_params=None, **create_kwargs):
        xfm = create_transformer(lib, create_params=create_params, **create_kwargs)
        return cls(lib, xfm, capacity_bytes=capacity_bytes)

    def __init__(self, lib, xfm, capacity_bytes=0):
        self.lib = lib
        self.xfm = xfm
        self.capacity_bytes = int(capacity_bytes)
        self.host_weight_bytes = int(capacity_bytes)
        self.current_session = None
        self.step_count = 0
        self.sampler_step_count = 0
        self.applied_lora_sig = "[]"
        self.applied_api_key = None

    def set_api_key(self, api_key):
        """Hot-switch API key in place via qf_auth_set_api_key (no rebuild)."""
        if self.xfm is None:
            raise RuntimeError("QuantFunc set_api_key: no live transformer")
        st = self.lib.qf_auth_set_api_key(self.xfm, api_key.encode("utf-8"))
        if st != QF_OK:
            raise RuntimeError(f"QuantFunc could not switch the API key (status {st}): {last_err(self.lib)}")

    def set_loras(self, loras):
        """Hot-swap active LoRA stack in place via qf_xfm_set_loras (JSON array, [] clears)."""
        if self.xfm is None:
            raise RuntimeError("QuantFunc set_loras: no live transformer")
        payload = [{"path": e["path"], "scale": float(e["scale"])} for e in (loras or ())]
        st = self.lib.qf_xfm_set_loras(self.xfm, json.dumps(payload).encode("utf-8"))
        if st != QF_OK:
            raise RuntimeError(f"QuantFunc qf_xfm_set_loras failed (status {st}): {last_err(self.lib)}")

    def unload_vram(self, bytes_requested=QF_UINT64_MAX):
        """Release up to bytes_requested of engine VRAM via qf_xfm_unload_vram."""
        if self.xfm is None:
            return 0
        req = operator.index(bytes_requested)
        if not 0 <= req <= QF_UINT64_MAX:
            raise ValueError("unload_vram bytes_requested must fit uint64")
        freed = ctypes.c_uint64(0)
        st = self.lib.qf_xfm_unload_vram(self.xfm, req, ctypes.byref(freed))
        if st != QF_OK:
            raise RuntimeError(f"qf_xfm_unload_vram failed (status {st}): {last_err(self.lib)}")
        return int(freed.value)

    def retain_reference_audio(self, owner):
        if getattr(self, "_reference_audio_pending", False):
            raise RuntimeError("qf_native LTX-AV: prior reference begin has no proven copy completion")
        self._reference_audio_owner = owner
        self._reference_audio_pending = owner is not None
        if owner is not None:
            _REFERENCE_AUDIO_PENDING_OWNERS.add(self)

    def reference_audio_pending(self):
        return getattr(self, "_reference_audio_pending", False)

    def reference_audio_copy_completed(self):
        self._reference_audio_pending = False
        _REFERENCE_AUDIO_PENDING_OWNERS.discard(self)

    def unload_sync(self):
        """Explicit between-call synchronization fence via qf_xfm_unload_vram(0)."""
        if self.xfm is None:
            raise RuntimeError("qf_native: synchronous release needs a live transformer")
        self.unload_vram(0)
        self.reference_audio_copy_completed()
        self._reference_audio_owner = None
        return QF_OK

    def end_session_if_open(self):
        """Close the open session if any. Retains current_session on refused/raising end."""
        if self.current_session is None:
            return (False, True)
        ok = True
        try:
            st = self.lib.qf_xfm_session_end(self.current_session)
            if st != QF_OK:
                ok = False
                try:
                    say(f"[qf_native] WARNING qf_xfm_session_end returned status={st}: "
                        f"{last_err(self.lib)}", flush=True)
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            ok = False
            try:
                say(f"[qf_native] WARNING qf_xfm_session_end raised: {exc!r}", flush=True)
            except Exception:  # noqa: BLE001
                pass
        if ok:
            self.current_session = None
            self._reference_audio_owner = None
            self.reference_audio_copy_completed()
        return (True, ok)

    def resident_vram_bytes(self):
        """Query engine resident VRAM bytes via qf_xfm_resident_vram_bytes."""
        if self.xfm is None:
            return 0
        out = ctypes.c_uint64(0)
        st = self.lib.qf_xfm_resident_vram_bytes(self.xfm, ctypes.byref(out))
        if st != QF_OK:
            raise RuntimeError(f"QuantFunc residency query failed: {last_err(self.lib)}")
        return int(out.value)

    def vram_need_bytes(self, latent_shape):
        """Query additional VRAM needed for latent_shape via qf_xfm_vram_need_bytes."""
        if self.xfm is None:
            return 0
        dims = [int(d) for d in latent_shape]
        if not dims or any(d <= 0 for d in dims):
            return 0
        arr = (ctypes.c_int64 * len(dims))(*dims)
        out = ctypes.c_uint64(0)
        st = self.lib.qf_xfm_vram_need_bytes(self.xfm, arr, len(dims), ctypes.byref(out))
        if st != QF_OK:
            raise RuntimeError(f"QuantFunc demand query failed: {last_err(self.lib)}")
        return int(out.value)

    def destroy(self):
        if self.reference_audio_pending():
            raise RuntimeError("qf_native LTX-AV: reference copy pending; explicit successful native cleanup required")
        self.end_session_if_open()
        if self.xfm is not None:
            try:
                self.lib.qf_xfm_destroy(self.xfm)
            except Exception:  # noqa: BLE001
                pass
            self.xfm = None
