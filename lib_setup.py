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
import shutil
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
    """Detect CUDA major version available on this system.
    Returns 13, 12, or 0 (unknown).
    """
    # Method 1: check CUDA_PATH environment
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        m = re.search(r'v(\d+)', cuda_path)
        if m:
            return int(m.group(1))

    # Method 2: nvidia-smi CUDA version
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip()
        # Driver >= 560 supports CUDA 13, >= 525 supports CUDA 12
        driver_ver = int(out.split(".")[0]) if out else 0
        if driver_ver >= 560:
            return 13
        elif driver_ver >= 525:
            return 12
    except Exception:
        pass

    # Method 3: check installed CUDA toolkit
    if _IS_WINDOWS:
        base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.isdir(base):
            versions = sorted(os.listdir(base), reverse=True)
            for v in versions:
                m = re.match(r'v(\d+)', v)
                if m:
                    return int(m.group(1))
    else:
        for ver in [13, 12]:
            if shutil.which(f"nvcc") or os.path.exists(f"/usr/local/cuda-{ver}"):
                return ver

    # Method 4: try torch
    try:
        import torch
        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda or ""
            m = re.match(r'(\d+)', cuda_ver)
            if m:
                return int(m.group(1))
    except Exception:
        pass

    return 0


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
    """Return dependency zip filename for the given CUDA version."""
    if cuda_major <= 12:
        return "cu12-dep-win32.zip"
    else:
        return "cu13-dep-win32.zip"


def _test_load_dll(dll_path: str) -> tuple:
    """Try to load the DLL. Returns (success, error_message)."""
    if not os.path.exists(dll_path):
        return False, f"File not found: {dll_path}"

    dll_dir = os.path.dirname(os.path.abspath(dll_path))

    if _IS_WINDOWS and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(dll_dir)
        except OSError:
            pass

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = dll_dir + os.pathsep + old_path

    try:
        lib = ctypes.CDLL(dll_path)
        lib.quantfunc_version.restype = ctypes.c_char_p
        version = lib.quantfunc_version().decode()
        # Unload by deleting reference (Windows may keep it loaded)
        del lib
        return True, version
    except OSError as e:
        return False, str(e)
    finally:
        os.environ["PATH"] = old_path


def _download_dep_zip(cuda_major: int, dest_dir: str) -> bool:
    """Download and extract CUDA dependency zip from ModelScope."""
    zip_name = get_dep_zip_name(cuda_major)
    print(f"[QuantFunc] Downloading {zip_name} from ModelScope...")

    try:
        from modelscope.hub.file_download import model_file_download
        local_path = model_file_download(
            model_id=_MODELSCOPE_REPO,
            file_path=zip_name,
        )

        print(f"[QuantFunc] Extracting {zip_name} to {dest_dir}...")
        os.makedirs(dest_dir, exist_ok=True)
        with zipfile.ZipFile(local_path, "r") as zf:
            zf.extractall(dest_dir)

        print(f"[QuantFunc] Dependencies installed to {dest_dir}")
        return True
    except Exception as e:
        print(f"[QuantFunc] Failed to download dependencies: {e}")
        return False


def resolve_library() -> str:
    """Main entry point: detect CUDA, select DLL, ensure deps, return DLL path.

    Returns the absolute path to the quantfunc DLL ready for loading.
    """
    bin_dir = _get_bin_dir()
    cuda_major = detect_cuda_major()
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

    # Test-load to check dependencies
    success, msg = _test_load_dll(dll_path)
    if success:
        logger.info("DLL loaded OK: %s (v%s, CUDA %d)", dll_name, msg, cuda_major)
        return dll_path

    # Load failed — likely missing CUDA/cuDNN shared libraries
    if not _IS_WINDOWS:
        cuda_pkg = "cuda-toolkit-13-1" if cuda_major >= 13 else "cuda-toolkit-12-6"
        print(f"[QuantFunc] Cannot load {dll_name}: {msg}")
        print(f"[QuantFunc] Install CUDA and cuDNN dependencies:")
        print(f"[QuantFunc]   # CUDA Toolkit")
        print(f"[QuantFunc]   sudo apt install {cuda_pkg}  # or: conda install cuda-toolkit")
        print(f"[QuantFunc]   # cuDNN 9.x")
        print(f"[QuantFunc]   sudo apt install libcudnn9-cuda-{'13' if cuda_major >= 13 else '12'}")
        print(f"[QuantFunc]   # Or download from: https://developer.nvidia.com/cudnn")
        return dll_path

    logger.warning("DLL load failed (%s), attempting dependency download...", msg)

    # Download dep zip and extract to bin dir
    if _download_dep_zip(cuda_major, bin_dir):
        # Retry load
        success2, msg2 = _test_load_dll(dll_path)
        if success2:
            print(f"[QuantFunc] DLL loaded successfully after dep install (v{msg2})")
        else:
            print(f"[QuantFunc] DLL still cannot load after dep install: {msg2}")
            print(f"[QuantFunc] Please install CUDA Toolkit and cuDNN manually.")
    else:
        print(f"[QuantFunc] Could not download dependencies. Please install manually:")
        print(f"[QuantFunc]   - CUDA Toolkit {'13.x' if cuda_major >= 13 else '12.x'}")
        print(f"[QuantFunc]   - cuDNN 9.x")

    return dll_path
