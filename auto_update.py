"""Auto-update QuantFunc shared library from ModelScope.

On plugin startup:
1. Read current plugin version from bin/<platform>/version.json ("comfy" field)
2. Read current lib version by calling quantfunc_version() from the .so/.dll
   - Uses a subprocess to avoid locking the DLL in the main process (Windows)
   - If the lib doesn't exist or can't be loaded, lib version = None (needs download)
3. Fetch remote version.json from ModelScope QuantFunc/Plugin repo
4. Find the highest lib version whose "comfy" requirement <= current plugin version
5. If that lib version > local lib version (or local is None), download it

Remote version.json structure:
{
  "linux": {
    "0.0.02": { "comfy": "0.0.01", "lib": "0.0.02" },
    "0.0.01": { "comfy": "0.0.01", "lib": "0.0.01" }
  },
  "win32": { ... }
}

Local bin/<platform>/version.json:
{ "comfy": "0.0.01" }
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Dict, Optional, Tuple

logger = logging.getLogger("QuantFunc.AutoUpdate")

_MODELSCOPE_REPO = "QuantFunc/Plugin"
_IS_WINDOWS = platform.system() == "Windows"
_PLATFORM = "win32" if _IS_WINDOWS else "linux"

def _get_lib_name() -> str:
    """Get the correct library filename based on CUDA version."""
    try:
        from .lib_setup import detect_cuda_major, get_lib_names
        cuda_major = detect_cuda_major()
        lib_name, _ = get_lib_names(cuda_major)
        return lib_name
    except Exception:
        return "quantfunc.dll" if _IS_WINDOWS else "libquantfunc.so"

_LIB_NAME = _get_lib_name()


def _get_bin_dir() -> str:
    """Return the bin/<platform>/ directory path."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg_dir, "bin", "windows" if _IS_WINDOWS else "linux")


def _read_comfy_version() -> str:
    """Read current plugin version from bin/<platform>/version.json."""
    version_file = os.path.join(_get_bin_dir(), "version.json")
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("comfy", "0.0.00")
    except Exception:
        return "0.0.00"


def _read_lib_version() -> Optional[str]:
    """Read lib version by spawning a subprocess that loads the library.

    Uses a subprocess to avoid locking the DLL in the main process,
    which would prevent file replacement on Windows.
    Returns version string or None if lib doesn't exist or can't be loaded.
    """
    lib_path = os.path.join(_get_bin_dir(), _LIB_NAME)
    if not os.path.exists(lib_path):
        return None

    # Spawn a short-lived subprocess to read the version
    # This avoids loading the DLL into our process (which locks it on Windows)
    script = (
        "import ctypes, sys, os\n"
        "try:\n"
        "    lib = ctypes.CDLL(sys.argv[1])\n"
        "    lib.quantfunc_version.restype = ctypes.c_char_p\n"
        "    lib.quantfunc_version.argtypes = []\n"
        "    v = lib.quantfunc_version()\n"
        "    print(v.decode('utf-8') if v else '')\n"
        "except Exception as e:\n"
        "    print('', file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, lib_path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            ver = result.stdout.strip()
            return ver if ver else None
    except Exception as e:
        logger.debug("Cannot read lib version via subprocess: %s", e)

    return None


def _parse_version(v: str) -> list:
    """Parse version string to list of ints for comparison."""
    parts = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return parts


def _ver_cmp(a: str, b: str) -> int:
    """Compare two version strings. Returns -1, 0, or 1."""
    ap, bp = _parse_version(a), _parse_version(b)
    max_len = max(len(ap), len(bp))
    ap.extend([0] * (max_len - len(ap)))
    bp.extend([0] * (max_len - len(bp)))
    if ap < bp:
        return -1
    elif ap > bp:
        return 1
    return 0


def _fetch_remote_versions() -> Optional[Dict]:
    """Fetch version.json from ModelScope.
    Returns the platform dict or None.
    """
    try:
        from modelscope.hub.file_download import model_file_download
        local_path = model_file_download(
            model_id=_MODELSCOPE_REPO,
            file_path="version.json",
        )
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.debug("Failed to fetch remote version.json: %s", e)
        return None

    return data.get(_PLATFORM)


def _get_cuda_suffix() -> str:
    """Return version.json key suffix based on CUDA version: '' for CUDA 13, '-12' for CUDA 12."""
    try:
        from .lib_setup import detect_cuda_major
        cuda_major = detect_cuda_major()
        return "-12" if cuda_major <= 12 else ""
    except Exception:
        return ""

_CUDA_SUFFIX = _get_cuda_suffix()


def _find_best_compatible_version(
    remote_versions: Dict, comfy_version: str, local_lib: Optional[str]
) -> Optional[Tuple[str, Dict]]:
    """Find the highest lib version compatible with the current plugin.

    Uses CUDA-specific version keys: "lib" + "comfy" for CUDA 13,
    "lib-12" + "comfy-12" for CUDA 12.

    Eligible if:
      1. "comfy" requirement <= comfy_version
      2. "lib" version > local_lib (or local_lib is None = not downloaded)

    Returns (version_key, info_dict) or None.
    """
    lib_key = "lib" + _CUDA_SUFFIX        # "lib" or "lib-12"
    comfy_key = "comfy" + _CUDA_SUFFIX    # "comfy" or "comfy-12"

    best_key = None
    best_lib = None
    best_info = None

    for version_key, info in remote_versions.items():
        required_comfy = info.get(comfy_key, info.get("comfy", "0.0.00"))
        lib_version = info.get(lib_key, info.get("lib", version_key))

        # Plugin must be new enough
        if _ver_cmp(required_comfy, comfy_version) > 0:
            continue

        # Must be an upgrade (or first download)
        if local_lib is not None and _ver_cmp(lib_version, local_lib) <= 0:
            continue

        # Pick the highest
        if best_lib is None or _ver_cmp(lib_version, best_lib) > 0:
            best_key = version_key
            best_lib = lib_version
            best_info = info

    if best_key:
        return best_key, best_info
    return None


def _download_lib(version_key: str, info: Dict) -> bool:
    """Download the shared library for the given version from ModelScope."""
    bin_dir = _get_bin_dir()
    lib_version = info.get("lib", version_key)

    # Remote path uses "windows" or "linux" as subdirectory on ModelScope
    remote_subdir = "windows" if _IS_WINDOWS else "linux"
    remote_path = "{}/{}/{}".format(version_key, remote_subdir, _LIB_NAME)

    print("[QuantFunc] Downloading {} v{} from ModelScope...".format(_LIB_NAME, lib_version))

    try:
        from modelscope.hub.file_download import model_file_download
        local_path = model_file_download(
            model_id=_MODELSCOPE_REPO,
            file_path=remote_path,
        )

        # Ensure bin dir exists
        os.makedirs(bin_dir, exist_ok=True)

        dest = os.path.join(bin_dir, _LIB_NAME)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=bin_dir, suffix=".tmp")
        try:
            os.close(tmp_fd)
            shutil.copy2(local_path, tmp_path)

            if _IS_WINDOWS:
                # DLL may be locked by worker process — use backup+rename strategy
                backup = dest + ".bak"
                try:
                    if os.path.exists(backup):
                        os.remove(backup)
                    if os.path.exists(dest):
                        os.rename(dest, backup)
                    os.rename(tmp_path, dest)
                    # Clean up backup
                    try:
                        if os.path.exists(backup):
                            os.remove(backup)
                    except OSError:
                        pass  # backup cleanup is best-effort
                except OSError as e:
                    print(
                        "[QuantFunc] Cannot replace {} (file locked?): {}. "
                        "Update saved as pending, will apply on next restart.".format(
                            _LIB_NAME, e
                        )
                    )
                    pending = dest + ".update"
                    if os.path.exists(pending):
                        os.remove(pending)
                    os.rename(tmp_path, pending)
                    return False
            else:
                # Linux: os.replace is atomic
                os.replace(tmp_path, dest)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        print(
            "[QuantFunc] Updated {} to v{}. Restart ComfyUI to use the new version.".format(
                _LIB_NAME, lib_version
            )
        )
        return True

    except Exception as e:
        print("[QuantFunc] Failed to download update: {}".format(e))
        return False


def _apply_pending_update():
    """On startup, apply pending .update file if it exists (Windows lock workaround)."""
    bin_dir = _get_bin_dir()
    dest = os.path.join(bin_dir, _LIB_NAME)
    pending = dest + ".update"
    if os.path.exists(pending):
        try:
            os.replace(pending, dest)
            print("[QuantFunc] Applied pending update for {}".format(_LIB_NAME))
        except OSError as e:
            logger.debug("[QuantFunc] Cannot apply pending update: %s", e)


def _check_and_update():
    """Check for updates and download if available. Runs in background thread."""
    try:
        _apply_pending_update()

        comfy_version = _read_comfy_version()
        local_lib = _read_lib_version()

        if local_lib is None:
            print(
                "[QuantFunc] No library found, checking ModelScope for download "
                "(plugin v{})...".format(comfy_version)
            )
        else:
            print(
                "[QuantFunc] Checking for updates (plugin v{}, lib v{})...".format(
                    comfy_version, local_lib
                )
            )

        remote_versions = _fetch_remote_versions()
        if remote_versions is None:
            print("[QuantFunc] Could not reach ModelScope, skipping update check")
            return

        result = _find_best_compatible_version(remote_versions, comfy_version, local_lib)

        if result is None:
            if local_lib:
                print("[QuantFunc] Library is up to date (v{})".format(local_lib))
            else:
                print("[QuantFunc] No compatible library version found on ModelScope")
            return

        best_key, best_info = result
        best_lib = best_info.get("lib", best_key)
        if local_lib:
            print("[QuantFunc] Update available: v{} -> v{}".format(local_lib, best_lib))
        else:
            print("[QuantFunc] Downloading library v{}...".format(best_lib))
        _download_lib(best_key, best_info)

    except Exception as e:
        print("[QuantFunc] Update check failed: {}".format(e))


def check_for_updates():
    """Launch background update check. Non-blocking, safe to call on startup."""
    t = threading.Thread(target=_check_and_update, daemon=True, name="QuantFunc-UpdateCheck")
    t.start()
