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

import errno
import hashlib
import json
import logging
import os
import platform
import re
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
        from .lib_setup import select_cuda_major, get_lib_names
        cuda_major = select_cuda_major()
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


_MODELSCOPE_RAW_URL = "https://www.modelscope.cn/models/QuantFunc/Plugin/resolve/master"

# --- SHA-256 integrity verification (from plugin 0.0.12) ---
# Each shipped engine lib has a sha256 recorded in a per-version manifest
# {version}/verify.json on ModelScope. On startup we fetch it (remote-first,
# local cache only when the network is blocked), compare the local lib, and
# self-heal (re-download, verify-before-replace) on a mismatch. Never blocks
# loading; never touches a locally-compiled dev build.
_VERIFY_FLOOR = "0.0.12"          # manifests only exist from this version onward
_VERIFY_SCHEMA_MAX = 1            # highest manifest schema this code understands


def _ensure_modelscope():
    """Return True if the modelscope SDK is importable; do NOT install it at runtime.

    Runtime `pip install` from a custom node is prohibited by the ComfyUI registry
    security policy (it bypasses the user's package manager). modelscope is declared
    in requirements.txt and installed at install time by ComfyUI-Manager / pip; if it
    is missing we fail gracefully with an actionable message instead of installing.
    """
    try:
        import modelscope  # noqa: F401
        return True
    except ImportError:
        print("[QuantFunc] modelscope is not installed; ModelScope features are "
              "unavailable. Install it with `pip install modelscope` (declared in "
              "requirements.txt) and restart ComfyUI.")
        return False


def _fetch_remote_versions() -> Optional[Dict]:
    """Fetch version.json from ModelScope.
    Auto-installs modelscope SDK if needed, falls back to direct HTTP.
    Returns the platform dict or None.
    """
    data = None

    # Method 1: modelscope SDK (auto-install)
    if _ensure_modelscope():
        try:
            from modelscope.hub.file_download import model_file_download
            local_path = model_file_download(
                model_id=_MODELSCOPE_REPO,
                file_path="version.json",
            )
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print("[QuantFunc] modelscope download failed: {}".format(e))

    # Method 2: direct HTTP fallback
    if data is None:
        try:
            import urllib.request
            url = "{}/version.json".format(_MODELSCOPE_RAW_URL)
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print("[QuantFunc] Direct download also failed: {}".format(e))
            print("[QuantFunc] Manual download: {}".format(_MODELSCOPE_RAW_URL))
            return None

    return data.get(_PLATFORM) if data else None


def _get_cuda_suffix() -> str:
    """Return version.json key suffix based on CUDA version: '' for CUDA 13, '-12' for CUDA 12."""
    try:
        from .lib_setup import select_cuda_major
        cuda_major = select_cuda_major()
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

    # Find the highest remote lib version first. If local is already newer
    # (e.g. a locally compiled dev build), skip update entirely so we don't
    # accidentally downgrade a freshly built binary.
    if local_lib is not None:
        highest_remote = None
        for info in remote_versions.values():
            lib_version = info.get(lib_key, info.get("lib", "0.0.00"))
            if highest_remote is None or _ver_cmp(lib_version, highest_remote) > 0:
                highest_remote = lib_version
        if highest_remote is not None and _ver_cmp(local_lib, highest_remote) > 0:
            logger.debug("Local lib %s is newer than highest remote %s, skipping update",
                         local_lib, highest_remote)
            return None

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

    download_url = "{}/{}".format(_MODELSCOPE_RAW_URL, remote_path)
    print("[QuantFunc] Downloading {} v{} ...".format(_LIB_NAME, lib_version))

    local_path = None

    # Method 1: modelscope SDK (auto-installed)
    if _ensure_modelscope():
        try:
            from modelscope.hub.file_download import model_file_download
            local_path = model_file_download(
                model_id=_MODELSCOPE_REPO,
                file_path=remote_path,
            )
        except Exception as e:
            print("[QuantFunc] modelscope download failed: {}".format(e))

    # Method 2: direct HTTP fallback
    if local_path is None or not os.path.exists(str(local_path)):
        try:
            import urllib.request
            print("[QuantFunc] Trying direct download: {}".format(download_url))
            tmp_fd, tmp_dl = tempfile.mkstemp(suffix=".download")
            os.close(tmp_fd)
            urllib.request.urlretrieve(download_url, tmp_dl)
            local_path = tmp_dl
        except Exception as e:
            print("[QuantFunc] Download failed: {}".format(e))
            print("[QuantFunc] Please download manually:")
            print("[QuantFunc]   {}".format(download_url))
            print("[QuantFunc] Place in: {}".format(bin_dir))
            return False

    try:

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


# --------------------------------------------------------------------------- #
# SHA-256 integrity verification (from plugin 0.0.12)
# --------------------------------------------------------------------------- #
def _verify_cache_path(version: str) -> str:
    """Per-version manifest cache; filename IS the version key (no envelope)."""
    return os.path.join(_get_bin_dir(), "verify-{}.json".format(version))


def _verify_state_path() -> str:
    """Heal-state marker that bounds the self-heal to one heavy re-download."""
    return os.path.join(_get_bin_dir(), "verify.state")


def _sha256_file(path: str) -> Optional[str]:
    """Return the file's sha256 hex digest, or None on any IO error."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _atomic_write_json(path: str, obj: Dict) -> None:
    """Write JSON atomically (temp on the SAME dir/fs as path, then os.replace)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise


def _fetch_remote_verify(version: str) -> Tuple[str, Optional[Dict]]:
    """Fetch {version}/verify.json from ModelScope, remote-first.

    Returns (status, data) where status is one of:
      "ok"          - data is the parsed manifest dict (also written to the cache)
      "no_manifest" - HTTP 404: the server has no manifest for this version -> SKIP, do NOT use cache
      "corrupt"     - 200 but body is not valid JSON / not a dict -> SKIP, do NOT use cache
      "blocked"     - network down / non-404 HTTP error -> caller may fall back to the local cache

    urllib is the PRIMARY path (uncached -> truly remote-first + real HTTP status);
    the modelscope SDK is a secondary REACH only when blocked (proxy/auth-only envs).
    """
    import urllib.request
    import urllib.error

    url = "{}/{}/verify.json".format(_MODELSCOPE_RAW_URL, version)
    status, data = "blocked", None
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            raw = resp.read()
        # A 200 whose body we have but can't decode/parse is corruption, NOT a network block.
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except ValueError:  # JSONDecodeError and UnicodeDecodeError are both ValueError
            return "corrupt", None
        if not isinstance(parsed, dict):
            return "corrupt", None
        status, data = "ok", parsed
    except urllib.error.HTTPError as e:
        if getattr(e, "code", None) == 404:
            return "no_manifest", None
        status, data = "blocked", None
    except Exception:
        status, data = "blocked", None

    if status == "blocked" and _ensure_modelscope():
        # degraded mode (network down): try the SDK as a secondary reach (may be cache-stale)
        try:
            from modelscope.hub.file_download import model_file_download
            p = model_file_download(
                model_id=_MODELSCOPE_REPO, file_path="{}/verify.json".format(version)
            )
            with open(p, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            if isinstance(parsed, dict):
                status, data = "ok", parsed
        except Exception:
            pass

    if status == "ok":
        try:
            _atomic_write_json(_verify_cache_path(version), data)
        except Exception:
            pass  # cache write is best-effort
    return status, data


def _read_cached_verify(version: str) -> Optional[Dict]:
    """Read the per-version cache; any error or non-dict -> None (never raises)."""
    try:
        with open(_verify_cache_path(version), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _load_verify_for(version: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Remote-first; fall back to the local cache ONLY when the network is blocked."""
    status, data = _fetch_remote_verify(version)
    if status == "ok":
        return data, "remote"
    if status == "blocked":
        cached = _read_cached_verify(version)
        if cached is not None:
            return cached, "cache"
    return None, None


def _heal_state_read() -> Tuple[Optional[str], Optional[str]]:
    """Return (version, expected) of the last bounded heal attempt, or (None, None)."""
    try:
        with open(_verify_state_path(), "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            return (d.get("version"), d.get("expected"))
    except Exception:
        pass
    return (None, None)


def _heal_state_write(version: str, expected: str) -> None:
    try:
        _atomic_write_json(_verify_state_path(), {"version": version, "expected": expected})
    except Exception:
        pass


def _heal_state_clear() -> None:
    try:
        os.remove(_verify_state_path())
    except OSError:
        pass


def _heal_download(version: str, expected: str) -> str:
    """Re-download the official lib and install it ONLY if its sha matches `expected`.

    VERIFY-BEFORE-REPLACE + EXDEV-safe + disk-safe. Returns one of:
      "healed"   - downloaded, sha matched, replaced the lib (restart to use)
      "pending"  - downloaded, sha matched, but the DLL was locked (Windows) -> .update pending
      "unhealed" - a heavy transfer happened but did NOT yield a matching lib
                   (sha mismatch OR a non-space install error) -> caller MARKS to bound re-downloads
      "failed"   - cheap/transient: no/low disk, 404, network error, truncated, temp-read error
                   -> caller does NOT mark (safe to retry next startup)
    Never destroys the existing lib unless a verified replacement is in place.
    """
    bin_dir = _get_bin_dir()
    dest = os.path.join(bin_dir, _LIB_NAME)
    remote_subdir = "windows" if _IS_WINDOWS else "linux"
    remote_path = "{}/{}/{}".format(version, remote_subdir, _LIB_NAME)
    download_url = "{}/{}".format(_MODELSCOPE_RAW_URL, remote_path)

    # Pre-flight free space: peak need is ~2x lib (existing dest kept + bindir temp copy) + margin.
    try:
        existing = os.path.getsize(dest) if os.path.exists(dest) else (64 << 20)
        if shutil.disk_usage(bin_dir).free < existing * 2 + (64 << 20):
            print("[QuantFunc] verify: not enough free space in {} to heal; skipping".format(bin_dir))
            return "failed"
    except OSError:
        pass  # if we cannot stat, proceed; the bounded install try still protects

    scratch = None
    scratch_is_ours = False
    bindir_tmp = None
    try:
        # Download to a scratch temp. urllib first (own scratch + Content-Length), SDK fallback.
        try:
            import urllib.request
            fd, scratch = tempfile.mkstemp(suffix=".heal")
            scratch_is_ours = True  # mark BEFORE close so an (unlikely) close raise still cleans up
            os.close(fd)
            with urllib.request.urlopen(download_url, timeout=60) as resp:
                clen = resp.headers.get("Content-Length")
                clen = int(clen) if (clen and clen.isdigit()) else None
                got = 0
                with open(scratch, "wb") as out:
                    while True:
                        buf = resp.read(1 << 20)
                        if not buf:
                            break
                        out.write(buf)
                        got += len(buf)
            if clen is not None and got != clen:
                print("[QuantFunc] verify: heal download truncated ({}/{} bytes)".format(got, clen))
                return "failed"
        except Exception:
            # Remove our own partial scratch BEFORE nulling it, else the finally can't reach it.
            if scratch and scratch_is_ours and os.path.exists(scratch):
                try:
                    os.remove(scratch)
                except OSError:
                    pass
            scratch, scratch_is_ours = None, False
            if _ensure_modelscope():
                try:
                    from modelscope.hub.file_download import model_file_download
                    scratch = model_file_download(model_id=_MODELSCOPE_REPO, file_path=remote_path)
                except Exception:
                    scratch = None
            if not scratch or not os.path.exists(str(scratch)):
                return "failed"

        actual = _sha256_file(scratch)
        if actual is None:
            return "failed"
        if actual != expected:
            return "unhealed"  # wrong publish / corrupt download -> bounded by caller

        # Verified-good bytes. Install into bin_dir (same fs as dest). Bound copy2+replace together.
        try:
            os.makedirs(bin_dir, exist_ok=True)
            fd, bindir_tmp = tempfile.mkstemp(dir=bin_dir, suffix=".tmp")
            os.close(fd)
            shutil.copy2(scratch, bindir_tmp)
            if _IS_WINDOWS:
                # DLL may be locked by a worker — reproduce _download_lib's backup+rename dance.
                backup = dest + ".bak"
                try:
                    if os.path.exists(backup):
                        os.remove(backup)
                    if os.path.exists(dest):
                        os.rename(dest, backup)
                    os.rename(bindir_tmp, dest)
                    try:
                        if os.path.exists(backup):
                            os.remove(backup)
                    except OSError:
                        pass
                    return "healed"
                except OSError:
                    pending = dest + ".update"
                    if os.path.exists(pending):
                        os.remove(pending)
                    os.rename(bindir_tmp, pending)
                    return "pending"
            else:
                os.replace(bindir_tmp, dest)
                return "healed"
        except OSError as e:
            if getattr(e, "errno", None) == errno.ENOSPC:
                # Transient: disk filled after the pre-flight check. Don't mark -> retry once freed.
                print("[QuantFunc] verify: no disk space to install the verified lib; will retry")
                return "failed"
            print("[QuantFunc] verify: install of verified lib failed: {}".format(e))
            return "unhealed"  # heavy transfer done but no matching lib installed -> bounded
    finally:
        for p, ours in ((bindir_tmp, True), (scratch, scratch_is_ours)):
            if ours and p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def _verify_local_lib() -> None:
    """Verify the installed lib's sha256 against the per-version ModelScope manifest.

    Self-heals (bounded, verify-before-replace) on a mismatch. SKIPs (never blocks
    loading, never touches the file) on any 'cannot verify' condition: a dev build
    (unreadable lib version), a pre-0.0.12 lib, no reachable manifest, a newer manifest
    schema, an unregistered artifact, or an unreadable file.
    """
    local_lib = _read_lib_version()
    if local_lib is None:
        return  # dev build / unreadable -> never touch
    if not re.fullmatch(r"\d+(?:\.\d+)*", local_lib):
        return  # defense-in-depth: only a clean dotted-numeric version feeds the URL / cache path
    if _ver_cmp(local_lib, _VERIFY_FLOOR) < 0:
        return  # no manifest exists before the floor

    data, source = _load_verify_for(local_lib)
    if data is None:
        print("[QuantFunc] verify: no usable manifest for v{} (not published / unreachable / "
              "corrupt); skipping integrity check".format(local_lib))
        return

    schema = data.get("schema", 1)
    if not isinstance(schema, int) or isinstance(schema, bool) or schema > _VERIFY_SCHEMA_MAX:
        print("[QuantFunc] verify: manifest schema {} newer than supported ({}); "
              "skipping".format(schema, _VERIFY_SCHEMA_MAX))
        return

    platform_map = data.get(_PLATFORM)
    if not isinstance(platform_map, dict):
        return
    expected = platform_map.get(_LIB_NAME)
    if not isinstance(expected, str) or not expected:
        return  # this artifact is not registered for this version

    bin_dir = _get_bin_dir()
    dest = os.path.join(bin_dir, _LIB_NAME)
    actual = _sha256_file(dest)
    if actual is None:
        return
    if actual == expected:
        print("[QuantFunc] verify: {} integrity OK (v{}, sha256 {}…, manifest from {})".format(
            _LIB_NAME, local_lib, actual[:16], source))
        return

    # ---- MISMATCH: bounded, verify-before-replace self-heal ----
    if os.path.exists(dest + ".update"):
        print("[QuantFunc] verify: SHA mismatch, but a verified update is pending; "
              "it applies on the next restart")
        return
    if _heal_state_read() == (local_lib, expected):
        print("[QuantFunc] verify: SHA mismatch persists after a heal attempt for v{} — "
              "the published manifest may be wrong; NOT re-downloading. Delete {} to retry."
              .format(local_lib, _verify_state_path()))
        return

    print("[QuantFunc] verify: SHA-256 MISMATCH for {} (v{}, manifest from {})".format(
        _LIB_NAME, local_lib, source))
    print("[QuantFunc]   expected (ModelScope): {}...".format(expected[:16]))
    print("[QuantFunc]   local file:            {}...".format(actual[:16]))
    print("[QuantFunc] verify: re-downloading the official artifact...")

    result = _heal_download(local_lib, expected)
    if result == "healed":
        _heal_state_clear()
        print("[QuantFunc] verify: downloaded a verified official {}. "
              "Restart ComfyUI to use it.".format(_LIB_NAME))
    elif result == "pending":
        print("[QuantFunc] verify: verified official {} saved as pending; "
              "applies on the next restart.".format(_LIB_NAME))
    elif result == "unhealed":
        _heal_state_write(local_lib, expected)
        print("[QuantFunc] verify: the official download did NOT match the manifest "
              "(wrong publish / corrupt / install error); keeping the current {} "
              "untouched.".format(_LIB_NAME))
    else:  # "failed"
        print("[QuantFunc] verify: could not download/install the official {} "
              "(offline / 404 / low disk / truncated); keeping current.".format(_LIB_NAME))


def _run_update_check() -> None:
    """The plugin-version-driven auto-update flow (extracted from _check_and_update).

    Its internal early-returns no longer skip the integrity verify, which runs
    separately in _check_and_update.
    """
    comfy_version = _read_comfy_version()
    local_lib = _read_lib_version()
    lib_path = os.path.join(_get_bin_dir(), _LIB_NAME)
    lib_file_exists = os.path.exists(lib_path)

    if local_lib is None and not lib_file_exists:
        print(
            "[QuantFunc] No library found, checking ModelScope for download "
            "(plugin v{})...".format(comfy_version)
        )
    elif local_lib is None and lib_file_exists:
        # Library file exists but version can't be read (e.g. locally compiled
        # build, or incompatible binary). Don't overwrite it.
        print("[QuantFunc] Library exists but version unreadable, skipping update")
        return
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


def _check_and_update():
    """Check for updates, download if available, then verify lib integrity.

    Runs in a background thread. Two isolated phases: the plugin-version-driven
    update flow, then the SHA-256 integrity verify. The verify runs regardless of
    whether the update flow reached ModelScope (it has its own fetch + cache
    fallback), and neither phase masks the other.
    """
    try:
        _apply_pending_update()
        _run_update_check()
    except Exception as e:
        print("[QuantFunc] Update check failed: {}".format(e))

    try:
        _verify_local_lib()
    except Exception as e:
        print("[QuantFunc] Integrity verify failed (non-fatal): {}".format(e))


def check_for_updates():
    """Launch update check. Blocks on first-time download, background for updates.

    When the library doesn't exist yet (first install), the download runs
    synchronously so the DLL is ready before the user can trigger a node.
    For subsequent update checks the thread runs in the background.
    """
    lib_path = os.path.join(_get_bin_dir(), _LIB_NAME)
    lib_missing = not os.path.exists(lib_path)

    t = threading.Thread(target=_check_and_update, daemon=True, name="QuantFunc-UpdateCheck")
    t.start()

    if lib_missing:
        # Block until download finishes so the worker doesn't race against it
        t.join(timeout=120)
