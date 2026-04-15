"""QuantFunc library setup: CUDA detection, DLL selection, dependency resolution.

On startup:
1. Detect CUDA version (13 vs 12) from nvidia-smi or torch
2. Select the matching DLL (quantfunc.dll / quantfunc-12.dll)
3. Test-load the DLL to check for missing dependencies
4. On Windows: if deps are missing, download the dep zip from ModelScope
5. Return the resolved DLL path
"""

import ctypes
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import zipfile

logger = logging.getLogger("QuantFunc.LibSetup")

_IS_WINDOWS = platform.system() == "Windows"
_BIN_SUBDIR = "windows" if _IS_WINDOWS else "linux"
_MODELSCOPE_REPO = "QuantFunc/Plugin"


def _get_bin_dir() -> str:
    """Return the bin/<platform>/ directory path."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg_dir, "bin", _BIN_SUBDIR)


def detect_cuda_major() -> int:
    """Detect the CUDA major version this system can run.
    Returns 13, 12, or 0 (unknown).

    Primary signal: nvidia-smi's "CUDA Version" header (driver-supported max).
    This is the authoritative answer — it's what the user sees when they run
    nvidia-smi, and the worker ships its own runtime libs via the dep zip,
    so anything the driver supports will load.

    Fallbacks (only if nvidia-smi unavailable, e.g. no GPU at install time):
    driver_version → CUDA cap, installed toolkit dirs, CUDA_PATH, torch.
    """
    # Primary: nvidia-smi "CUDA Version" header
    try:
        out = subprocess.check_output(
            ["nvidia-smi"], timeout=5, stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
        m = re.search(r'CUDA Version:\s*(\d+)', out)
        if m:
            return int(m.group(1))
    except Exception:
        pass

    # Fallback: driver_version → CUDA cap
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip()
        driver_ver = int(out.split(".")[0]) if out else 0
        if driver_ver >= 560:
            return 13
        if driver_ver >= 525:
            return 12
    except Exception:
        pass

    # Fallback: installed toolkit dirs (scan all cuda-X.Y, take max)
    candidates = []
    if _IS_WINDOWS:
        base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.isdir(base):
            try:
                for entry in os.listdir(base):
                    m = re.match(r'v(\d+)', entry)
                    if m and os.path.isdir(os.path.join(base, entry)):
                        candidates.append(int(m.group(1)))
            except OSError:
                pass
    else:
        for prefix in ("/usr/local", "/opt", "/usr/lib"):
            if not os.path.isdir(prefix):
                continue
            try:
                for entry in os.listdir(prefix):
                    m = re.match(r'cuda-(\d+)', entry)
                    if m and os.path.isdir(os.path.join(prefix, entry)):
                        candidates.append(int(m.group(1)))
            except OSError:
                pass
        for link in ("/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda"):
            if os.path.islink(link) or os.path.isdir(link):
                try:
                    real = os.path.basename(os.path.realpath(link))
                    m = re.search(r'(\d+)', real)
                    if m:
                        candidates.append(int(m.group(1)))
                except Exception:
                    pass
    if candidates:
        return max(candidates)

    # Fallback: CUDA_PATH env hint
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        target = os.path.realpath(cuda_path) if os.path.exists(cuda_path) else cuda_path
        m = re.search(r'(\d+)', os.path.basename(target.rstrip("\\/")))
        if m:
            return int(m.group(1))

    # Last resort: torch's bundled cuda
    try:
        import torch
        cuda_ver = torch.version.cuda or ""
        m = re.match(r'(\d+)', cuda_ver)
        if m:
            return int(m.group(1))
    except Exception:
        pass

    return 0


def select_cuda_major() -> int:
    """Pick the effective CUDA major for binary selection.

    detect_cuda_major() reports what's installed on disk, but on RTX 50-series
    (SM120+) the GPU REQUIRES CUDA 13 PTX regardless. The cu13 dep zip ships
    its own libcudart.so.13/libcublas.so.13/etc., so a system with only cu12
    toolkit installed can still run the cu13 binary. Trust the GPU.
    """
    detected = detect_cuda_major()
    try:
        gpu_sm = _detect_gpu_sm()
    except Exception:
        gpu_sm = 0
    if gpu_sm >= 120 and detected != 13:
        logger.info("SM%d GPU detected, forcing CUDA 13 (detected %d)", gpu_sm, detected)
        return 13
    return detected if detected > 0 else 13


def get_lib_names(cuda_major: int):
    """Return (dll_name, cli_name) based on CUDA version.
    CUDA 13 (default): quantfunc.dll / quantfunc.exe
    CUDA 12:           quantfunc-12.dll / quantfunc-12.exe
    """
    if cuda_major <= 12:
        if _IS_WINDOWS:
            return "quantfunc-12.dll", "quantfunc-12.exe"
        else:
            return "libquantfunc-12.so", "quantfunc-12"
    else:
        if _IS_WINDOWS:
            return "quantfunc.dll", "quantfunc.exe"
        else:
            return "libquantfunc.so", "quantfunc"


def get_dep_zip_name(cuda_major: int) -> str:
    """Return dependency zip filename for the given CUDA version and platform."""
    plat = "win32" if _IS_WINDOWS else "linux"
    if cuda_major <= 12:
        return f"cu12-dep-{plat}.zip"
    else:
        return f"cu13-dep-{plat}.zip"


def _collect_dll_dirs(dll_path: str) -> list:
    """Collect directories that may contain DLL dependencies.

    Mirrors the same scanning logic as worker.py _load_dll() so that
    the test result accurately predicts whether the worker can load.
    """
    dll_dir = os.path.dirname(os.path.abspath(dll_path))
    extra_dirs = [dll_dir]

    if not _IS_WINDOWS:
        # Linux: collect CUDA lib directories
        cuda_path = os.environ.get("CUDA_PATH", "")
        if not cuda_path:
            # Auto-detect highest CUDA toolkit
            for ver in [13, 12]:
                candidate = f"/usr/local/cuda-{ver}"
                if os.path.isdir(candidate):
                    cuda_path = candidate
                    break
            if not cuda_path and os.path.isdir("/usr/local/cuda"):
                cuda_path = "/usr/local/cuda"
        if cuda_path:
            for sub in ["lib64", "lib"]:
                d = os.path.join(cuda_path, sub)
                if os.path.isdir(d):
                    extra_dirs.append(d)

        # LD_LIBRARY_PATH dirs containing CUDA .so files
        for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
            if p and os.path.isdir(p) and p not in extra_dirs:
                extra_dirs.append(p)

        return extra_dirs

    # CUDA toolkit bin
    cuda_path = os.environ.get("CUDA_PATH", "")
    if not cuda_path:
        base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.isdir(base):
            versions = sorted(os.listdir(base), reverse=True)
            for v in versions:
                if os.path.isdir(os.path.join(base, v, "bin")):
                    cuda_path = os.path.join(base, v)
                    break
    if cuda_path:
        for sub in [os.path.join("bin", "x64"), "bin"]:
            d = os.path.join(cuda_path, sub)
            if os.path.isdir(d):
                extra_dirs.append(d)

    # cuDNN: add ALL cuda-version subdirs
    cudnn_base = os.path.join(
        os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA", "CUDNN")
    if os.path.isdir(cudnn_base):
        for ver in sorted(os.listdir(cudnn_base), reverse=True):
            ver_dir = os.path.join(cudnn_base, ver, "bin")
            if os.path.isdir(ver_dir):
                for sub in sorted(os.listdir(ver_dir), reverse=True):
                    x64 = os.path.join(ver_dir, sub, "x64")
                    if os.path.isdir(x64):
                        extra_dirs.append(x64)
                break  # highest cuDNN version only

    # PATH dirs containing CUDA/cuDNN/OpenCV DLLs
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if not p or not os.path.isdir(p):
            continue
        try:
            files = os.listdir(p)
        except OSError:
            continue
        if any(f.startswith(("cublas", "cudart", "cudnn", "cusolver",
                             "curand", "opencv")) and f.endswith(".dll")
               for f in files):
            extra_dirs.append(p)

    return extra_dirs


def _test_load_dll(dll_path: str) -> tuple:
    """Try to load the DLL in a subprocess. Returns (success, error_message).

    Uses a subprocess to avoid false positives: the parent process (ComfyUI)
    has PyTorch's CUDA/cuDNN DLLs already loaded in memory, so an in-process
    test would succeed even when dependency DLLs are missing on disk.  The
    subprocess replicates the same path scanning as the worker for accuracy.
    """
    if not os.path.exists(dll_path):
        return False, f"File not found: {dll_path}"

    # Collect the same dirs the worker would use
    extra_dirs = _collect_dll_dirs(dll_path)
    dirs_json = json.dumps(extra_dirs)

    script = (
        "import ctypes, json, os, sys, platform\n"
        "dll_path = sys.argv[1]\n"
        "dirs = json.loads(sys.argv[2])\n"
        "if platform.system() == 'Windows' and hasattr(os, 'add_dll_directory'):\n"
        "    for d in dirs:\n"
        "        try:\n"
        "            os.add_dll_directory(d)\n"
        "        except OSError:\n"
        "            pass\n"
        "os.environ['PATH'] = os.pathsep.join(dirs) + os.pathsep + os.environ.get('PATH', '')\n"
        "if platform.system() != 'Windows':\n"
        "    os.environ['LD_LIBRARY_PATH'] = os.pathsep.join(dirs) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')\n"
        "try:\n"
        "    lib = ctypes.CDLL(dll_path)\n"
        "    lib.quantfunc_version.restype = ctypes.c_char_p\n"
        "    v = lib.quantfunc_version()\n"
        "    print(v.decode('utf-8') if v else '')\n"
        "except Exception as e:\n"
        "    print('ERROR:' + str(e), file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, dll_path, dirs_json],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            ver = result.stdout.strip()
            return True, ver if ver else "unknown"
        else:
            err = result.stderr.strip()
            if err.startswith("ERROR:"):
                err = err[6:]
            return False, err or "subprocess exited with code {}".format(result.returncode)
    except subprocess.TimeoutExpired:
        return False, "DLL test load timed out"
    except Exception as e:
        return False, f"subprocess test failed: {e}"


_MODELSCOPE_RAW_URL = "https://www.modelscope.cn/models/QuantFunc/Plugin/resolve/master"


def _ensure_modelscope():
    """Install modelscope SDK if not available."""
    try:
        import modelscope  # noqa: F401
        return True
    except ImportError:
        print("[QuantFunc] Installing modelscope SDK...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "modelscope", "-q"],
                stdout=subprocess.DEVNULL,
            )
            print("[QuantFunc] modelscope installed successfully")
            return True
        except Exception as e:
            print(f"[QuantFunc] Failed to install modelscope: {e}")
            return False


def _download_from_modelscope(file_path: str):
    """Download a file from ModelScope. Tries SDK then direct HTTP.
    Returns local file path or None.
    """
    # Method 1: modelscope SDK (auto-install if needed)
    if _ensure_modelscope():
        try:
            from modelscope.hub.file_download import model_file_download
            return model_file_download(model_id=_MODELSCOPE_REPO, file_path=file_path)
        except Exception as e:
            print(f"[QuantFunc] modelscope download failed: {e}")

    # Method 2: direct HTTP fallback
    url = f"{_MODELSCOPE_RAW_URL}/{file_path}"
    try:
        import urllib.request
        print(f"[QuantFunc] Trying direct download: {url}")
        tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(file_path))
        urllib.request.urlretrieve(url, tmp_path)
        return tmp_path
    except Exception as e:
        print(f"[QuantFunc] Direct download also failed: {e}")
        print(f"[QuantFunc] Please download manually: {url}")
        return None


def _download_dep_zip(cuda_major: int, dest_dir: str) -> bool:
    """Download and extract CUDA dependency zip from ModelScope."""
    zip_name = get_dep_zip_name(cuda_major)
    print(f"[QuantFunc] Downloading dependencies ({zip_name})...")

    local_path = _download_from_modelscope(zip_name)
    if not local_path or not os.path.exists(local_path):
        print(f"[QuantFunc] Extract to: {dest_dir}")
        return False

    try:
        print(f"[QuantFunc] Extracting to {dest_dir}...")
        os.makedirs(dest_dir, exist_ok=True)
        with zipfile.ZipFile(local_path, "r") as zf:
            zf.extractall(dest_dir)
        print(f"[QuantFunc] Dependencies installed")
        return True
    except Exception as e:
        print(f"[QuantFunc] Extract failed: {e}")
        return False


def _detect_gpu_sm() -> int:
    """Detect GPU compute capability (SM version). Returns e.g. 120, 89, 86, or 0."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip()
        # Parse "12.0" → 120, "8.9" → 89
        for line in out.split("\n"):
            line = line.strip()
            if "." in line:
                major, minor = line.split(".")[:2]
                return int(major) * 10 + int(minor)
    except Exception:
        pass

    # Fallback: try torch
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            return cap[0] * 10 + cap[1]
    except Exception:
        pass

    return 0


def resolve_library() -> str:
    """Main entry point: detect CUDA, select DLL, ensure deps, return DLL path.

    Returns the absolute path to the quantfunc DLL ready for loading.
    Raises RuntimeError if SM120+ GPU detected with CUDA 12 (unsupported).
    """
    bin_dir = _get_bin_dir()
    cuda_major = select_cuda_major()
    dll_name, _ = get_lib_names(cuda_major)
    dll_path = os.path.join(bin_dir, dll_name)

    # Fallback: if version-specific DLL doesn't exist, try the default
    if not os.path.exists(dll_path):
        default_dll = "quantfunc.dll" if _IS_WINDOWS else "libquantfunc.so"
        fallback = os.path.join(bin_dir, default_dll)
        if os.path.exists(fallback):
            logger.info("CUDA %d DLL not found (%s), falling back to %s",
                       cuda_major, dll_name, default_dll)
            dll_path = fallback
            dll_name = default_dll

    if not os.path.exists(dll_path):
        logger.warning("No DLL found at %s", dll_path)
        return dll_path  # auto_update.py will download it

    # Quick test-load to log status (dep download is handled by WorkerManager)
    success, msg = _test_load_dll(dll_path)
    if success:
        logger.info("DLL loaded OK: %s (v%s, CUDA %d)", dll_name, msg, cuda_major)
    else:
        # Log but don't fail — WorkerManager._ensure_worker will download deps
        # and retry when the worker actually fails to load.
        logger.info("DLL pre-check: %s may have missing deps (%s), "
                    "will resolve on first use", dll_name, msg)

    return dll_path
