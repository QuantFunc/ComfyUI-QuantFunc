"""QuantFunc ComfyUI nodes: pipeline config, model loader, LoRA, LoRA config, and inference.

Data flow:
  (PipelineConfig) ──config──→ ModelLoader ──pipeline──→ (LoRA) ──→ (LoRA Config) ──→ Generate → IMAGE

PipelineConfig provides advanced init options (optional — without it, auto_optimize defaults apply).
ModelLoader outputs a pipeline config. LoRA nodes append lora paths (chainable).
LoRA Config sets merge strategy. Generate materializes the pipeline (cached) and runs inference.

The quantfunc engine runs in a separate worker process to isolate its CUDA runtime
from ComfyUI's PyTorch (avoids DLL version conflicts on Windows).
"""

import atexit
import ctypes
import hashlib
import json
import logging
import numpy as np
import os
import platform
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time

# ============================================================================
# Library path resolution
# ============================================================================

_IS_WINDOWS = platform.system() == "Windows"
_BIN_SUBDIR = "windows" if _IS_WINDOWS else "linux"

def _resolve_lib_path():
    """Find the quantfunc shared library.
    Uses lib_setup to detect CUDA version and select the correct DLL.
    """
    # Environment override takes priority
    env_path = os.environ.get("QUANTFUNC_LIB", "")
    if env_path and os.path.exists(env_path):
        return os.path.abspath(env_path)

    try:
        from .lib_setup import resolve_library
        return resolve_library()
    except RuntimeError:
        # Intentional fatal config errors (e.g. SM120 GPU without CUDA 13)
        # must surface, not silently fall back to a mismatched default name.
        raise
    except Exception as e:
        logging.getLogger("QuantFunc").warning("lib_setup failed: %s, using default", e)

    # Fallback: default name
    pkg_dir = os.path.dirname(__file__)
    lib_name = "quantfunc.dll" if _IS_WINDOWS else "libquantfunc.so"
    return os.path.join(pkg_dir, "bin", _BIN_SUBDIR, lib_name)

_LIB_PATH = _resolve_lib_path()
_WORKER_PY = os.path.join(os.path.dirname(__file__), "worker.py")


def _get_available_devices():
    """Detect available CUDA GPU devices. Returns list of string device IDs."""
    devices = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                devices.append("{}: {}".format(i, name))
    except Exception:
        pass
    if not devices:
        # Fallback: try nvidia-smi
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode().strip()
            for line in out.split("\n"):
                line = line.strip()
                if line:
                    devices.append(line.replace(", ", ": "))
        except Exception:
            pass
    return devices if devices else ["0: GPU"]


_AVAILABLE_DEVICES = _get_available_devices()


def _load_lib_config():
    """Load config.json from the same directory as the quantfunc library binary.
    Returns dict with server_url and api_key (empty strings if not found).
    """
    config_path = os.path.join(os.path.dirname(_LIB_PATH), "config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
            return {
                "server_url": cfg.get("server_url", ""),
                "api_key": cfg.get("api_key", ""),
            }
    except Exception as e:
        logging.debug("[QuantFunc] Failed to load %s: %s", config_path, e)
    return {"server_url": "", "api_key": ""}


def _make_cache_key(cfg):
    """Build a cache key from pipeline config.
    Excludes api_key and server_url — changing auth credentials should not
    force pipeline recreation (use set_api_key for hot-swap instead).
    """
    opts = dict(cfg.get("options", {}))
    opts.pop("api_key", None)
    opts.pop("server_url", None)
    parts = json.dumps({
        "model_dir": cfg.get("model_dir", ""),
        "transformer": cfg.get("transformer", ""),
        "backend": cfg.get("backend", "svdq"),
        "precision": cfg.get("precision", "int4"),
        "scheduler": cfg.get("scheduler", ""),
        "device": cfg.get("device", 0),
        "options": opts,
    }, sort_keys=True)
    return hashlib.sha256(parts.encode()).hexdigest()[:16]


# ============================================================================
# Worker Manager — manages worker subprocess
# ============================================================================

_dep_download_lock = threading.Lock()
_dep_downloading = False  # True while download is in progress
_dep_downloaded = False   # True after dep download attempted (success or fail)


if not _IS_WINDOWS:
    try:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _PR_SET_PDEATHSIG = 1

        def _linux_die_with_parent():
            """preexec_fn: kernel will send SIGTERM to this worker when its
            parent process (ComfyUI) dies, even from SIGKILL or crash."""
            _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        _linux_die_with_parent = None
else:
    _linux_die_with_parent = None


def _kill_stale_workers(dll_path):
    """Find and kill leftover worker.py subprocesses from previous ComfyUI
    runs that didn't exit cleanly. Identified by matching the --dll-path arg
    in their cmdline. Must only match OUR workers, not other tools.
    """
    my_pid = os.getpid()
    worker_script = os.path.abspath(_WORKER_PY)
    dll_abs = os.path.abspath(dll_path)
    candidates = []

    if _IS_WINDOWS:
        # psutil would be cleanest but we can't assume it. Use WMIC.
        # /format:list output is blocks of Key=Value lines separated by blank
        # lines. Key order inside a block is not guaranteed.
        try:
            out = subprocess.check_output(
                ["wmic", "process", "where",
                 "name='python.exe' or name='pythonw.exe'",
                 "get", "ProcessId,CommandLine", "/format:list"],
                stderr=subprocess.DEVNULL, timeout=5).decode("utf-8", "replace")
            block = {}
            def flush_block():
                cmd = block.get("CommandLine", "")
                pid_str = block.get("ProcessId", "").strip()
                if pid_str.isdigit() and cmd:
                    pid = int(pid_str)
                    if (pid != my_pid and worker_script in cmd
                            and dll_abs in cmd):
                        candidates.append(pid)
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    if block:
                        flush_block()
                        block = {}
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    block[k.strip()] = v
            if block:
                flush_block()
        except Exception as e:
            logging.debug("[QuantFunc] stale-worker scan (wmic) failed: %s", e)
    else:
        # Linux: walk /proc
        try:
            for pid_entry in os.listdir("/proc"):
                if not pid_entry.isdigit():
                    continue
                pid = int(pid_entry)
                if pid == my_pid:
                    continue
                try:
                    with open("/proc/%d/cmdline" % pid, "rb") as f:
                        cmdline = f.read().decode("utf-8", "replace")
                except (OSError, IOError):
                    continue
                # Args are NUL-separated; also match if joined with spaces
                if worker_script in cmdline and dll_abs in cmdline:
                    candidates.append(pid)
        except Exception as e:
            logging.debug("[QuantFunc] stale-worker scan (/proc) failed: %s", e)

    if not candidates:
        return
    logging.warning("[QuantFunc] Killing %d stale worker process(es) from previous run: %s",
                    len(candidates), candidates)
    for pid in candidates:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    # Give them 3 seconds to exit gracefully, then SIGKILL leftovers
    time.sleep(3)
    for pid in candidates:
        try:
            os.kill(pid, 0)  # still alive?
        except OSError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            logging.warning("[QuantFunc] SIGKILL'd stale worker pid %d", pid)
        except OSError:
            pass


class WorkerManager:
    """Manages a QuantFunc worker subprocess with isolated CUDA libraries."""

    def __init__(self):
        self._process = None
        self._stdin = None
        self._stdout = None
        self._stderr_thread = None
        self._loaded_keys = set()          # keys currently alive in worker
        self._api_keys = {}                # cache_key -> api_key last set
        self._node_refs = {}               # generate_node_id -> cache_key (owner tracking)
        self._req_counter = 0
        self._lock = threading.Lock()

    # ── Worker lifecycle ──

    def _build_worker_env(self, dll_dir):
        """Build environment dict for the worker subprocess."""
        env = os.environ.copy()
        if _IS_WINDOWS:
            extra = [dll_dir]
            cuda_path = env.get("CUDA_PATH", "")
            if cuda_path:
                cuda_bin = os.path.join(cuda_path, "bin")
                if os.path.isdir(cuda_bin):
                    extra.insert(0, cuda_bin)
            env["PATH"] = os.pathsep.join(extra) + os.pathsep + env.get("PATH", "")
        else:
            ld_parts = [dll_dir]
            cuda_path = env.get("CUDA_PATH", "/usr/local/cuda")
            lib64 = os.path.join(cuda_path, "lib64")
            if os.path.isdir(lib64):
                ld_parts.append(lib64)
            existing = env.get("LD_LIBRARY_PATH", "")
            if existing:
                ld_parts.append(existing)
            env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)
        return env

    def _start_worker(self, dll_path, env):
        """Start worker subprocess and wait for ready signal.
        Returns (success, error_message).
        """
        python_exe = os.environ.get("QUANTFUNC_PYTHON", "") or sys.executable
        cmd = [python_exe, _WORKER_PY, "--dll-path", dll_path]

        # Kill leftover workers from a previous ComfyUI run that didn't exit
        # cleanly (SIGKILL / reboot / force-quit). They still hold VRAM.
        _kill_stale_workers(dll_path)

        creation_flags = 0
        preexec = None
        if _IS_WINDOWS:
            creation_flags = subprocess.CREATE_NO_WINDOW
        else:
            # Linux: ask the kernel to send SIGTERM to the worker whenever our
            # process dies (PR_SET_PDEATHSIG). Without this, if ComfyUI gets
            # SIGKILL'd or crashes, the worker is reparented to init and keeps
            # holding ~15 GB of VRAM until the user manually kills it.
            preexec = _linux_die_with_parent

        logging.info("[QuantFunc] Starting worker: %s (python=%s)",
                     " ".join(cmd[:4]), python_exe)

        try:
            self._process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env, creationflags=creation_flags,
                preexec_fn=preexec)
        except Exception as e:
            return False, (f"Failed to start worker process: {e}\n"
                           f"Python: {python_exe}\n"
                           f"Set QUANTFUNC_PYTHON env var to a working Python 3.8+ path.")

        self._stdin = self._process.stdin
        self._stdout = self._process.stdout

        self._stderr_thread = threading.Thread(
            target=self._stderr_reader, daemon=True)
        self._stderr_thread.start()

        ready = self._read_response(timeout=60)
        if not isinstance(ready, dict) or ready.get("type") != "ready":
            try:
                self._process.kill()
                _, stderr_out = self._process.communicate(timeout=5)
                stderr_msg = stderr_out.decode(errors="replace")[-500:] if stderr_out else ""
            except Exception:
                stderr_msg = ""
            self._process = None
            return False, (f"Worker failed to start (timeout or crash).\n"
                           f"Python: {python_exe}\n"
                           f"DLL: {dll_path}\n"
                           f"Worker stderr: {stderr_msg}\n"
                           f"Hint: Set QUANTFUNC_PYTHON env var to a Python with ctypes + numpy.")

        logging.info("[QuantFunc] Worker ready (version %s, pid %d)",
                     ready.get("version", "?"), self._process.pid)
        return True, ""

    @staticmethod
    def _try_download_deps(dll_path):
        """Download dependency DLLs if not already attempted. Thread-safe.
        Returns True if deps were newly downloaded.
        Raises RuntimeError if another thread is currently downloading.
        """
        global _dep_downloading, _dep_downloaded
        if _dep_downloaded:
            return False
        acquired = _dep_download_lock.acquire(blocking=False)
        if not acquired:
            # Another thread is downloading right now
            raise RuntimeError(
                "[QuantFunc] 依赖库正在下载中，请稍后再试。\n"
                "Dependency libraries are being downloaded. Please try again shortly.")
        try:
            if _dep_downloaded:
                return False
            _dep_downloading = True
            try:
                from .lib_setup import select_cuda_major, _download_dep_zip
                cuda_major = select_cuda_major()
                bin_dir = os.path.dirname(os.path.abspath(dll_path))
                logging.warning("[QuantFunc] Worker failed to load DLL, "
                                "downloading dependency libraries...")
                result = _download_dep_zip(cuda_major, bin_dir)
                _dep_downloaded = True
                return result
            except Exception as e:
                logging.error("[QuantFunc] Dependency download failed: %s", e)
                _dep_downloaded = True
                return False
            finally:
                _dep_downloading = False
        finally:
            _dep_download_lock.release()

    def _ensure_worker(self):
        """Start worker process if not running.
        On first load failure, downloads deps and retries once.
        """
        if self._process is not None and self._process.poll() is None:
            return

        # If deps are being downloaded by another thread, fail fast
        if _dep_downloading:
            raise RuntimeError(
                "[QuantFunc] 依赖库正在下载中，请稍后再试。\n"
                "Dependency libraries are being downloaded. Please try again shortly.")

        if self._process is not None:
            logging.warning("[QuantFunc] Worker process died, restarting...")
            self._loaded_keys.clear()
            self._api_keys.clear()
            self._node_refs.clear()

        dll_path = _LIB_PATH
        if not os.path.exists(dll_path):
            raise RuntimeError(
                f"QuantFunc library not found: {dll_path}\n"
                f"The auto-download may still be in progress or may have failed.\n"
                f"Check the ComfyUI console for download status messages.")
        dll_dir = os.path.dirname(os.path.abspath(dll_path))
        env = self._build_worker_env(dll_dir)

        # First attempt
        ok, err = self._start_worker(dll_path, env)
        if ok:
            return

        # First failure may be missing dependency libs — download and retry
        if self._try_download_deps(dll_path):
            logging.info("[QuantFunc] Dependencies installed, retrying worker...")
            env = self._build_worker_env(dll_dir)  # rebuild (deps now in dll_dir)
            ok2, err2 = self._start_worker(dll_path, env)
            if ok2:
                return
            raise RuntimeError(err2)

        raise RuntimeError(err)

    def _stderr_reader(self):
        """Forward worker's stderr to logging and cache recent error lines."""
        self._recent_stderr = []
        try:
            for line in self._process.stderr:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logging.info("[QuantFunc-worker] %s", text)
                    # Cache recent lines with error/warning keywords for crash diagnostics
                    if any(k in text.lower() for k in ("error", "fatal", "exception", "oom", "abort", "segfault")):
                        self._recent_stderr.append(text)
                        if len(self._recent_stderr) > 10:
                            self._recent_stderr.pop(0)
        except Exception:
            pass

    def _kill_worker(self):
        if self._process is not None:
            pid = self._process.pid
            # First try graceful SIGTERM
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                pass
            # Then SIGKILL if still alive
            if self._process.poll() is None:
                try:
                    self._process.kill()
                    self._process.wait(timeout=5)
                except Exception:
                    pass
            # Last resort: os.kill (handles CUDA driver stuck in uninterruptible sleep)
            if self._process.poll() is None:
                try:
                    import signal
                    os.kill(pid, signal.SIGKILL)
                    self._process.wait(timeout=3)
                except Exception:
                    logging.error("[QuantFunc] Failed to kill worker pid %d — "
                                  "process may be stuck in CUDA driver (D state). "
                                  "GPU resources may remain occupied until reboot.", pid)
            self._process = None
            self._loaded_keys.clear()
            self._api_keys.clear()
            self._node_refs.clear()

    # ── IPC ──

    def _next_req_id(self):
        self._req_counter += 1
        return self._req_counter

    def _send_command(self, cmd):
        """Send a JSON command to worker."""
        data = json.dumps(cmd, ensure_ascii=True).encode("utf-8") + b"\n"
        self._stdin.write(data)
        self._stdin.flush()

    _SENTINEL_TIMEOUT = "__timeout__"
    _SENTINEL_WORKER_DIED = "__worker_died__"

    def _read_response(self, timeout=600):
        """Read one JSON line from worker stdout."""
        # Simple blocking read with timeout via thread
        result = [self._SENTINEL_TIMEOUT]  # default = timeout
        def reader():
            try:
                line = self._stdout.readline()
                if line:
                    result[0] = json.loads(line.decode("utf-8").strip())
                else:
                    # Empty line = worker process exited (stdout closed)
                    result[0] = self._SENTINEL_WORKER_DIED
            except Exception as e:
                result[0] = {"type": "error", "error_message": str(e)}
        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout=timeout)
        return result[0]

    def _read_binary(self, n_bytes):
        """Read exactly n_bytes from worker stdout."""
        data = b""
        while len(data) < n_bytes:
            chunk = self._stdout.read(n_bytes - len(data))
            if not chunk:
                raise RuntimeError("Worker stdout closed during binary read")
            data += chunk
        return data

    def _call(self, cmd, progress_cb=None, timeout=1800):
        """Send command and collect response, relaying progress."""
        self._send_command(cmd)

        while True:
            resp = self._read_response(timeout=timeout)
            if resp is self._SENTINEL_TIMEOUT:
                self._kill_worker()
                raise RuntimeError(f"Worker timeout after {timeout}s — the operation took too long. "
                                   "Try a smaller resolution or fewer steps.")
            if resp is self._SENTINEL_WORKER_DIED:
                # Worker crashed — read cached stderr error lines
                import time
                time.sleep(0.5)  # let stderr reader finish flushing
                stderr_lines = getattr(self, "_recent_stderr", [])
                error_detail = stderr_lines[-1] if stderr_lines else ""
                # Also check exit code
                exit_code = None
                try:
                    exit_code = self._process.poll()
                except Exception:
                    pass
                self._kill_worker()
                if not error_detail:
                    error_detail = f"Worker process crashed (exit code: {exit_code}, no error details captured)"
                else:
                    error_detail = f"Worker crashed (exit code: {exit_code}): {error_detail}"
                logging.error(f"[QuantFunc] {error_detail}")
                raise RuntimeError(error_detail)

            msg_type = resp.get("type", "")

            if msg_type == "progress":
                if progress_cb:
                    progress_cb(resp.get("step", 0), resp.get("total", 0))
                continue

            if msg_type == "result":
                status = resp.get("status", "")
                if status == "cancelled":
                    raise InterruptedError("Generation cancelled")
                if status == "error":
                    error_msg = resp.get("error_message", "Unknown worker error")
                    error_code = resp.get("error_code", -1)
                    # Kill worker on CUDA/OOM/internal errors — CUDA state may be
                    # corrupted and the process will hold GPU memory indefinitely.
                    # Auth errors (code 7) are recoverable — don't kill.
                    if error_code not in (7,):  # QUANTFUNC_ERROR_AUTH
                        logging.warning("[QuantFunc] C API error (code %d), killing worker "
                                        "to release GPU resources: %s", error_code, error_msg[:200])
                        self._kill_worker()
                    raise RuntimeError(error_msg)
                return resp

            # Unknown message type, skip
            continue

    # ── Public API ──

    def set_api_key(self, cache_key, api_key):
        """Hot-swap API key on the specified pipeline (no pipeline recreation)."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            if cache_key not in self._loaded_keys:
                return
            self._set_api_key_locked(cache_key, api_key)

    def _unload_others_locked(self, keep_key):
        """Offload all pipelines except keep_key to CPU, to free VRAM before
        loading/running keep_key. Must be called with self._lock held."""
        others = [k for k in self._loaded_keys if k != keep_key]
        for k in others:
            try:
                self._call({"cmd": "unload", "req_id": self._next_req_id(),
                            "cache_key": k}, timeout=60)
                logging.info("[QuantFunc] Offloaded pipeline %s to CPU", k[:8])
            except Exception as e:
                logging.warning("[QuantFunc] Failed to offload %s: %s", k[:8], e)

    def ensure_pipeline(self, cfg, node_id=None, alive_node_ids=None):
        """Ensure pipeline matching cfg is loaded in worker. Returns its cache key.
        `node_id` is the caller's ComfyUI UNIQUE_ID. If provided, we track which
        node references which cache key and destroy pipelines that no node
        references anymore (e.g. when a loader node's transformer path changes).
        `alive_node_ids` is the set of Generate node ids present in the current
        workflow — any _node_refs entry not in this set is considered stale
        (node was deleted from the workflow) and its pipeline is destroyed if
        no one else references it.
        """
        with self._lock:
            self._ensure_worker()

            key = _make_cache_key(cfg)
            opts = cfg.get("options", {})
            new_api_key = opts.get("api_key", "")

            # Evict stale refs from deleted/disconnected Generate nodes
            if alive_node_ids is not None:
                stale = [nid for nid in self._node_refs if nid not in alive_node_ids]
                for nid in stale:
                    dropped_key = self._node_refs.pop(nid)
                    logging.info("[QuantFunc] Dropping stale node ref %s -> %s",
                                 nid, dropped_key[:8] if dropped_key else None)
                    if (dropped_key and
                            dropped_key not in self._node_refs.values() and
                            dropped_key != key and
                            dropped_key in self._loaded_keys):
                        logging.info("[QuantFunc] Destroying orphan pipeline %s (no nodes reference it)",
                                     dropped_key[:8])
                        try:
                            self._call({"cmd": "destroy",
                                        "req_id": self._next_req_id(),
                                        "cache_key": dropped_key}, timeout=30)
                        except Exception as e:
                            logging.warning("[QuantFunc] destroy(%s) failed: %s", dropped_key[:8], e)
                        self._loaded_keys.discard(dropped_key)
                        self._api_keys.pop(dropped_key, None)

            # Update node→key ownership and release orphaned pipelines
            if node_id is not None:
                old_key = self._node_refs.get(node_id)
                self._node_refs[node_id] = key
                if old_key and old_key != key:
                    # If no other node still references the old key, destroy it
                    if old_key not in self._node_refs.values() and old_key in self._loaded_keys:
                        logging.info("[QuantFunc] Destroying orphan pipeline %s (node %s changed config)",
                                     old_key[:8], node_id)
                        try:
                            self._call({"cmd": "destroy",
                                        "req_id": self._next_req_id(),
                                        "cache_key": old_key}, timeout=30)
                        except Exception as e:
                            logging.warning("[QuantFunc] destroy(%s) failed: %s", old_key[:8], e)
                        self._loaded_keys.discard(old_key)
                        self._api_keys.pop(old_key, None)

            if key in self._loaded_keys:
                # Pipeline already loaded — check if API key changed
                if new_api_key and new_api_key != self._api_keys.get(key, ""):
                    self._set_api_key_locked(key, new_api_key)
                return key

            if self._loaded_keys:
                logging.info("[QuantFunc] New pipeline requested, offloading %d existing to CPU...",
                             len(self._loaded_keys))
                # Free VRAM before creating new pipeline
                self._unload_others_locked(keep_key=key)

            # Build create command
            create_cmd = {
                "cmd": "create",
                "req_id": self._next_req_id(),
                "cache_key": key,
                "model_dir": cfg.get("model_dir", ""),
                "transformer_path": cfg.get("transformer", ""),
                "scheduler_config": cfg.get("scheduler", "") or None,
                "model_backend": cfg.get("backend", "svdq"),
                "svdq_precision": cfg.get("precision", "int4"),
                "device_idx": cfg.get("device", 0),
                "config_json": json.dumps(opts),
            }

            logging.info(f"[QuantFunc] create_cmd: model_dir={create_cmd['model_dir']!r}, "
                         f"transformer={create_cmd['transformer_path']!r}, "
                         f"scheduler={create_cmd['scheduler_config']!r}, "
                         f"backend={create_cmd['model_backend']!r}, "
                         f"device={create_cmd['device_idx']!r}, "
                         f"config_json={create_cmd['config_json']!r}")

            self._call(create_cmd, timeout=1800)
            self._loaded_keys.add(key)
            self._api_keys[key] = new_api_key
            logging.info("[QuantFunc] Pipeline ready (%d loaded).", len(self._loaded_keys))
            return key

    def _set_api_key_locked(self, cache_key, api_key):
        """Internal: set API key while already holding self._lock."""
        cmd = {
            "cmd": "set_api_key",
            "req_id": self._next_req_id(),
            "cache_key": cache_key,
            "api_key": api_key,
        }
        self._call(cmd, timeout=30)
        self._api_keys[cache_key] = api_key
        logging.info("[QuantFunc] API key updated (hot-swap).")

    def text_to_image(self, cache_key, prompt, height, width, steps, seed,
                      guidance_scale, options_json=None, pbar=None):
        """Generate text-to-image on the specified pipeline. Returns [H, W, 3] float32 numpy array."""
        with self._lock:
            self._ensure_worker()

            # Before running, make sure only the target pipeline is GPU-resident
            self._unload_others_locked(keep_key=cache_key)

            def on_progress(step, total):
                if pbar is not None:
                    pbar.update(1)

            cmd = {
                "cmd": "text_to_image",
                "req_id": self._next_req_id(),
                "cache_key": cache_key,
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "options_json": options_json,
            }

            resp = self._call(cmd, progress_cb=on_progress, timeout=600)
            return self._read_image(resp)

    def image_to_image(self, cache_key, prompt, ref_paths, height, width, steps, seed,
                       true_cfg_scale=4.0, negative_prompt="",
                       options_json=None, pbar=None):
        """Generate image-to-image on the specified pipeline. Returns [H, W, 3] float32 numpy array."""
        with self._lock:
            self._ensure_worker()

            # Before running, make sure only the target pipeline is GPU-resident
            self._unload_others_locked(keep_key=cache_key)

            def on_progress(step, total):
                if pbar is not None:
                    pbar.update(1)

            cmd = {
                "cmd": "image_to_image",
                "req_id": self._next_req_id(),
                "cache_key": cache_key,
                "prompt": prompt,
                "ref_image_paths": ref_paths,
                "height": height,
                "width": width,
                "num_steps": steps,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "options_json": options_json,
            }

            resp = self._call(cmd, progress_cb=on_progress, timeout=600)
            return self._read_image(resp)

    def export_model(self, cfg, export_path):
        """Export model via worker."""
        with self._lock:
            self._ensure_worker()

            # Destroy all loaded pipelines first to free VRAM
            if self._loaded_keys:
                self._call({"cmd": "destroy", "req_id": self._next_req_id()})
                self._loaded_keys.clear()
                self._api_keys.clear()

            opts = dict(cfg.get("options", {}))
            sched = cfg.get("scheduler", "")
            if sched:
                opts["scheduler_config"] = sched

            cmd = {
                "cmd": "export",
                "req_id": self._next_req_id(),
                "model_dir": cfg.get("model_dir", ""),
                "export_path": export_path,
                "transformer_path": cfg.get("transformer", ""),
                "model_backend": cfg.get("backend", "svdq"),
                "svdq_precision": cfg.get("precision", "int4"),
                "device_idx": cfg.get("device", 0),
                "config_json": json.dumps(opts),
            }

            self._call(cmd, timeout=1800)

    def cancel(self):
        """Send cancel signal to worker."""
        if self._process and self._process.poll() is None:
            try:
                cmd = json.dumps({"cmd": "cancel", "req_id": 0}).encode("utf-8") + b"\n"
                self._stdin.write(cmd)
                self._stdin.flush()
            except Exception:
                pass

    def unload_pipeline(self, cache_key=None):
        """Offload models from GPU to CPU, freeing VRAM. Pipelines stay alive for fast reload.
        If cache_key given, unload that one; otherwise unload all."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            if not self._loaded_keys:
                return
            cmd = {"cmd": "unload", "req_id": self._next_req_id()}
            if cache_key is not None:
                if cache_key not in self._loaded_keys:
                    return
                cmd["cache_key"] = cache_key
            try:
                self._call(cmd, timeout=30)
                logging.info("[QuantFunc] Models offloaded to CPU — VRAM freed (%s)",
                             cache_key if cache_key else "all")
            except Exception as e:
                logging.warning("[QuantFunc] Unload failed: %s", e)

    def destroy_all(self):
        """Destroy all loaded pipelines (keep worker alive)."""
        with self._lock:
            if self._process and self._process.poll() is None and self._loaded_keys:
                try:
                    self._call({"cmd": "destroy", "req_id": self._next_req_id()}, timeout=30)
                except Exception:
                    pass
                self._loaded_keys.clear()
                self._api_keys.clear()

    def shutdown(self):
        """Shutdown worker process."""
        with self._lock:
            if self._process and self._process.poll() is None:
                try:
                    # Try graceful shutdown via IPC first
                    cmd = json.dumps({"cmd": "shutdown", "req_id": 0}).encode("utf-8") + b"\n"
                    self._stdin.write(cmd)
                    self._stdin.flush()
                    self._process.wait(timeout=10)
                except Exception:
                    # IPC failed (broken pipe, etc.) — use signal-based kill
                    self._kill_worker()
            self._process = None
            self._loaded_keys.clear()
            self._api_keys.clear()

    def _read_image(self, resp):
        """Read binary image data following a result response."""
        n_bytes = resp.get("image_bytes", 0)
        w = resp.get("image_width", 0)
        h = resp.get("image_height", 0)
        if n_bytes == 0 or w == 0 or h == 0:
            raise RuntimeError("No image data in response")
        raw = self._read_binary(n_bytes)
        fmt = resp.get("image_format", "rgb_float32")
        if fmt == "rgb_uint8":
            # uint8 [0,255] → float32 [0,1], 4x less IPC data than float32
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
        else:
            # Legacy float32 path
            arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w, 3).copy()
        return arr


_manager = WorkerManager()
atexit.register(_manager.shutdown)


# ============================================================================
# Hook into ComfyUI model management — auto-unload when other nodes need VRAM
# ============================================================================

try:
    import comfy.model_management as _mm

    _original_free_memory = _mm.free_memory

    # Threshold to distinguish real VRAM requests from ComfyUI's blanket
    # unload_all_models() which passes 1e30.  256 TB is far beyond any real
    # GPU memory request but still well below 1e30.
    _UNREALISTIC_VRAM_REQUEST = 256 * 1024 * 1024 * 1024 * 1024  # 256 TB

    def _hooked_free_memory(memory_required, device, keep_loaded=[], **kwargs):
        if _manager._loaded_keys and memory_required > 0:
            # Ignore blanket "free everything" calls (e.g. unload_all_models
            # passes 1e30).  These happen after every prompt execution when
            # smart memory management is disabled and would needlessly destroy
            # the pipeline, forcing expensive re-creation on the next run.
            if memory_required >= _UNREALISTIC_VRAM_REQUEST:
                logging.debug("[QuantFunc] Ignoring blanket free_memory call "
                              f"(requested {memory_required:.2e} bytes)")
            else:
                # Only unload if there isn't enough free VRAM to satisfy the request
                try:
                    import torch
                    free_vram, _ = torch.cuda.mem_get_info(device)
                except Exception:
                    free_vram = 0
                if free_vram < memory_required:
                    logging.info("[QuantFunc] Auto-offloading pipelines to free VRAM for other models "
                                 f"(need {memory_required // 1024**2} MB, free {free_vram // 1024**2} MB)")
                    _manager.unload_pipeline()  # offload all to CPU, keep alive
        return _original_free_memory(memory_required, device, keep_loaded=keep_loaded, **kwargs)

    _mm.free_memory = _hooked_free_memory
except Exception:
    pass


# ============================================================================
# Node: QuantFunc Pipeline Config
# ============================================================================

class QuantFuncPipelineConfig:
    """Advanced pipeline configuration for model initialization.

    VRAM/offload strategy is chosen automatically by libquantfunc based on
    your GPU's free VRAM and the loaded model size — no offload knobs are
    exposed here. Old workflows that still set cpu_offload / layer_offload /
    adaptive_offload / offload_compression will continue to load (libquantfunc
    silently ignores those config keys with a one-time deprecation warning),
    but the values have no effect.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Tile-based VAE decoding to reduce VRAM (auto-enabled at high resolution)"}),
                "attention_backend": (["auto", "sage", "flash", "sdpa"], {"default": "auto",
                                      "tooltip": "Attention implementation: auto picks best for your GPU"}),
                "precision": (["bf16", "fp16"], {"default": "bf16", "tooltip": "Compute precision for pipeline"}),
                "text_precision": (["int4", "int8", "fp4", "fp8", "fp16"], {"default": "int4",
                                    "tooltip": "Text encoder quantization precision (fp4 requires SM120+/Blackwell)"}),
                "vision_quant": (["int8", "int4", "fp8", "fp4", "fp16"], {"default": "int8",
                                  "tooltip": "Vision encoder quantization (int8 = INT8 weights + FP16 compute, best quality/size tradeoff)"}),
            },
            "optional": {
                "vae_tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64,
                                  "tooltip": "VAE tile size in pixels (0 = auto)"}),
                "pinned_memory_limit": ("STRING", {"default": "", "tooltip": "Max pinned CPU memory: '60%', '48G', '48M', or empty for auto"}),
            }
        }

    RETURN_TYPES = ("QUANTFUNC_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build_config"
    CATEGORY = "QuantFunc"

    def build_config(self, tiled_vae, attention_backend, precision, text_precision,
                     vision_quant="int8", vae_tile_size=0, pinned_memory_limit=""):
        config = {
            "tiled_vae": tiled_vae,
            "attention_backend": attention_backend,
            "precision": precision,
            "text_precision": text_precision,
            "vision_quant": vision_quant,
        }

        if vae_tile_size > 0:
            config["vae_tile_size"] = vae_tile_size

        pinned = pinned_memory_limit if isinstance(pinned_memory_limit, str) and pinned_memory_limit else ""
        if pinned:
            config["pinned_memory_limit"] = pinned

        return (config,)


class QuantFuncModelLoader:
    """Load a QuantFunc model. Uses auto_optimize by default.
    Connect a PipelineConfig node to override init settings.
    Edit mode is auto-detected when ref_image is connected to Generate.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dir": ("STRING", {"default": "", "tooltip": "Base model directory (contains model_index.json)"}),
                "transformer_path": ("STRING", {"default": "", "tooltip": "Transformer weights path (safetensors file or directory)"}),
                "model_backend": (["svdq", "lighting"], {"default": "svdq"}),
                "device": (_AVAILABLE_DEVICES,),
            },
            "optional": {
                "config": ("QUANTFUNC_CONFIG", {"tooltip": "Advanced pipeline config (from PipelineConfig node). If not connected, uses auto_optimize defaults."}),
                "api_key": ("STRING", {"default": "", "tooltip": "QuantFunc API key for model authentication (e.g. qf_xxx). Server URL is read from config.json next to the library."}),
                "scheduler_config": ("STRING", {"default": "", "tooltip": "Scheduler JSON config path (for Lightning models)"}),
                "precision_config": ("STRING", {"default": "", "tooltip": "Per-layer precision config JSON path (Lighting backend only)"}),
                "prequant_weights": ("STRING", {"default": "", "tooltip": "Pre-quantized modulation weights safetensors path (Lighting backend only)"}),
                "fused_mod": ("BOOLEAN", {"default": False, "tooltip": "Fused INT8 SiLU+GEMV+bias+split6 for W8A8 modulation layers (Lighting backend only)"}),
                "act_quant_mode": (["absmax", "mse"], {
                    "default": "absmax",
                    "tooltip": "Activation quantization scale algorithm (Lighting backend only):\n"
                               "• absmax — fast, scale = absmax/7 (default)\n"
                               "• mse — search ~5 candidates for min MSE (+1dB quality, ~8% slower)",
                }),
                "manual_unload_model": ("BOOLEAN", {"default": False, "tooltip": "Activate to manually unload the model and free GPU memory. No image will be generated."}),
            }
        }

    RETURN_TYPES = ("QUANTFUNC_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "QuantFunc"

    def load_model(self, model_dir, transformer_path, model_backend,
                   device, config=None, manual_unload_model=False,
                   api_key="", scheduler_config="",
                   precision_config="", prequant_weights="",
                   fused_mod=False, act_quant_mode="absmax", **kwargs):
        scheduler_config = scheduler_config.strip() if isinstance(scheduler_config, str) else ""
        # Validate: scheduler_config must be a file path (not a bare number or random text)
        if scheduler_config and not os.path.exists(scheduler_config):
            logging.warning(f"[QuantFunc] scheduler_config path does not exist: {scheduler_config!r}, ignoring")
            scheduler_config = ""
        precision_config = precision_config.strip() if isinstance(precision_config, str) else ""
        prequant_weights = prequant_weights.strip() if isinstance(prequant_weights, str) else ""
        transformer_path = transformer_path if isinstance(transformer_path, str) and transformer_path else ""

        api_key = api_key.strip() if isinstance(api_key, str) else ""
        if api_key.lower() == "none":
            api_key = ""

        # Load server_url (and fallback api_key) from config.json next to the library
        lib_config = _load_lib_config()
        if not api_key:
            api_key = lib_config.get("api_key", "")
        server_url = lib_config.get("server_url", "")

        options = {"auto_optimize": True}
        if precision_config:
            options["precision_config"] = precision_config
        if prequant_weights:
            options["mod_weights"] = prequant_weights
        if fused_mod:
            options["fused_mod"] = True
        if api_key:
            options["api_key"] = api_key
        if server_url:
            options["server_url"] = server_url
        if model_backend == "lighting":
            options.setdefault("rotation_block_size", 256)
        options["act_quant_mode"] = act_quant_mode

        text_precision = "int4"
        if config and isinstance(config, dict):
            config = dict(config)  # copy to avoid mutating cached PipelineConfig output
            text_precision = config.pop("text_precision", text_precision)
            options.update(config)

        cfg = {
            "model_dir": model_dir,
            "transformer": transformer_path,
            "backend": model_backend,
            "precision": text_precision,
            "scheduler": scheduler_config,
            "device": int(device.split(":")[0]) if isinstance(device, str) else device,
            "options": options,
            "unload": manual_unload_model,
        }
        return (cfg,)


# ============================================================================
# Node: QuantFunc Model Auto Loader
# ============================================================================

def _get_auto_loader_dropdowns():
    """Get dropdown options from resource cache (loaded at import time)."""
    try:
        from .model_auto_loader import get_transformer_options
        return get_transformer_options()
    except Exception:
        return ["None"]


def _get_prequant_dropdowns():
    """Get prequant weight dropdown options from resource cache."""
    try:
        from .model_auto_loader import get_prequant_options
        return get_prequant_options()
    except Exception:
        return ["None"]


def _get_precision_config_dropdowns():
    """Get precision config dropdown options from resource cache."""
    try:
        from .model_auto_loader import get_precision_config_options
        return get_precision_config_options()
    except Exception:
        return ["None"]


class QuantFuncModelAutoLoader:
    """Auto-download and load QuantFunc models.

    Selects the correct GPU variant (50x-below/50x-above) automatically.
    Downloads base model, transformer, prequant weights, and precision config
    from HuggingFace or ModelScope on first use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from .model_auto_loader import MODEL_SERIES_LIST, _DATA_SOURCES
        transformer_opts = _get_auto_loader_dropdowns()
        return {
            "required": {
                "model_series": (MODEL_SERIES_LIST, {"tooltip": "Model series to download and load"}),
                "model_backend": (["svdq", "lighting"], {"default": "svdq"}),
                "device": (_AVAILABLE_DEVICES,),
                "data_source": (_DATA_SOURCES, {"default": "modelscope", "tooltip": "Download source: modelscope (China) or huggingface"}),
            },
            "optional": {
                "transformer": (transformer_opts, {"default": "None", "tooltip": "Transformer model variant. Format: Series/name. Select None to use base model's default transformer."}),
                "config": ("QUANTFUNC_CONFIG", {"tooltip": "Advanced pipeline config (from PipelineConfig node)"}),
                "api_key": ("STRING", {"default": "", "tooltip": "QuantFunc API key for model authentication"}),
                "act_quant_mode": (["absmax", "mse"], {
                    "default": "absmax",
                    "tooltip": "Activation quantization scale algorithm (Lighting backend only):\n"
                               "• absmax — fast, scale = absmax/7 (default)\n"
                               "• mse — search ~5 candidates for min MSE (+1dB quality, ~8% slower)",
                }),
                "manual_unload_model": ("BOOLEAN", {"default": False, "tooltip": "Activate to manually unload the model and free GPU memory."}),
            }
        }

    RETURN_TYPES = ("QUANTFUNC_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "QuantFunc"

    def load_model(self, model_series, model_backend, device, data_source,
                   config=None, manual_unload_model=False,
                   transformer="None", api_key="",
                   act_quant_mode="absmax",
                   **kwargs):
        from .model_auto_loader import (
            detect_gpu_variant, download_base_model,
            download_transformer, resolve_transformer_selection,
        )

        # ── GPU variant & base model ──
        gpu_variant = detect_gpu_variant()
        model_dir = download_base_model(model_series, gpu_variant, data_source)

        # ── Transformer (download if selected, otherwise use base model's) ──
        transformer_path = ""
        if transformer and transformer != "None":
            t_series, t_name = resolve_transformer_selection(transformer, model_series)
            if t_series and t_name:
                transformer_path = download_transformer(t_series, t_name, data_source)

        # ── Build pipeline config (same structure as ModelLoader) ──
        api_key = api_key.strip() if isinstance(api_key, str) else ""
        if api_key.lower() == "none":
            api_key = ""

        lib_config = _load_lib_config()
        if not api_key:
            api_key = lib_config.get("api_key", "")
        server_url = lib_config.get("server_url", "")

        options = {"auto_optimize": True}
        if api_key:
            options["api_key"] = api_key
        if server_url:
            options["server_url"] = server_url
        if model_backend == "lighting":
            options.setdefault("rotation_block_size", 256)
        options["act_quant_mode"] = act_quant_mode

        text_precision = "int4"
        if config and isinstance(config, dict):
            config = dict(config)  # copy to avoid mutating cached PipelineConfig output
            text_precision = config.pop("text_precision", text_precision)
            options.update(config)

        cfg = {
            "model_dir": model_dir,
            "transformer": transformer_path,
            "backend": model_backend,
            "precision": text_precision,
            "device": int(device.split(":")[0]) if isinstance(device, str) else device,
            "options": options,
            "unload": manual_unload_model,
        }
        return (cfg,)


# ============================================================================
# Node: QuantFunc Prequant Auto Loader
# ============================================================================

class QuantFuncPrequantAutoLoader:
    """Auto-download prequant weights from HuggingFace or ModelScope.

    Outputs a file path string that can be connected to ModelLoader's
    prequant_weights input. When not connected, ModelLoader falls back
    to its own text input field.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from .model_auto_loader import _DATA_SOURCES
        prequant_opts = _get_prequant_dropdowns()
        return {
            "required": {
                "prequant": (prequant_opts, {"default": "None", "tooltip": "Pre-quantized modulation weights. Format: Series/name. Select None to skip."}),
                "data_source": (_DATA_SOURCES, {"default": "modelscope", "tooltip": "Download source: modelscope (China) or huggingface"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prequant_weights",)
    FUNCTION = "load_prequant"
    CATEGORY = "QuantFunc"

    def load_prequant(self, prequant, data_source):
        if not prequant or prequant == "None":
            return ("",)

        from .model_auto_loader import resolve_selection_no_series, download_prequant

        pq_series, pq_name = resolve_selection_no_series(prequant, "Prequant")
        if not pq_series or not pq_name:
            return ("",)

        path = download_prequant(pq_series, pq_name, data_source)
        return (path,)


# ============================================================================
# Node: QuantFunc Precision Config Auto Loader
# ============================================================================

class QuantFuncPrecisionConfigAutoLoader:
    """Auto-download precision config from HuggingFace or ModelScope.

    Outputs a file path string that can be connected to ModelLoader's
    precision_config input. When not connected, ModelLoader falls back
    to its own text input field.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from .model_auto_loader import _DATA_SOURCES
        pc_opts = _get_precision_config_dropdowns()
        return {
            "required": {
                "precision_config": (pc_opts, {"default": "None", "tooltip": "Per-layer precision config JSON. Format: Series/name. Select None to skip."}),
                "data_source": (_DATA_SOURCES, {"default": "modelscope", "tooltip": "Download source: modelscope (China) or huggingface"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("precision_config",)
    FUNCTION = "load_precision_config"
    CATEGORY = "QuantFunc"

    def load_precision_config(self, precision_config, data_source):
        if not precision_config or precision_config == "None":
            return ("",)

        from .model_auto_loader import resolve_selection_no_series, download_precision_config

        pc_series, pc_name = resolve_selection_no_series(precision_config, "Precision config")
        if not pc_series or not pc_name:
            return ("",)

        path = download_precision_config(pc_series, pc_name, data_source)
        return (path,)


# ============================================================================
# Node: QuantFunc Base Series Model Auto Loader
# ============================================================================

class QuantFuncBaseSeriesModelAutoLoader:
    """Auto-download base model from QuantFunc model series.

    Selects the correct GPU variant (50x-below/50x-above) automatically.
    Downloads the base model from QuantFunc series repos on first use.
    Outputs the local model directory path as a string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from .model_auto_loader import MODEL_SERIES_LIST, _DATA_SOURCES
        return {
            "required": {
                "model_series": (MODEL_SERIES_LIST, {"tooltip": "QuantFunc model series"}),
                "data_source": (_DATA_SOURCES, {"default": "modelscope", "tooltip": "Download source: modelscope (China) or huggingface"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_dir",)
    FUNCTION = "load_base_model"
    CATEGORY = "QuantFunc"

    def load_base_model(self, model_series, data_source):
        from .model_auto_loader import detect_gpu_variant, download_base_model
        gpu_variant = detect_gpu_variant()
        model_dir = download_base_model(model_series, gpu_variant, data_source)
        return (model_dir,)


# ============================================================================
# Node: QuantFunc Base Model Auto Loader
# ============================================================================

def _get_diffusers_model_options():
    """Recursively scan ComfyUI/models/diffusers/ for model directories (containing model_index.json)."""
    try:
        dm_dir = os.path.join(_get_comfyui_dir(), "models", "diffusers")
        if os.path.isdir(dm_dir):
            dirs = []
            for root, subdirs, filenames in os.walk(dm_dir):
                if "model_index.json" in filenames:
                    rel = os.path.relpath(root, dm_dir)
                    dirs.append(rel.replace("\\", "/"))
            if dirs:
                return ["None"] + sorted(dirs)
    except Exception:
        pass
    return ["None"]


class QuantFuncBaseModelAutoLoader:
    """Load base models from ComfyUI/models/diffusers/ directory.

    Scans for subdirectories containing model_index.json.
    Outputs the local model directory path as a string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        model_opts = _get_diffusers_model_options()
        return {
            "required": {
                "model_dir": (model_opts, {"tooltip": "Base model from models/diffusers/"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_dir",)
    FUNCTION = "load_base_model"
    CATEGORY = "QuantFunc"

    def load_base_model(self, model_dir):
        if not model_dir or model_dir == "None":
            return ("",)

        full_path = os.path.join(_get_comfyui_dir(), "models", "diffusers", model_dir)
        if not os.path.isdir(full_path):
            raise RuntimeError("Model directory not found: {}".format(full_path))
        return (full_path,)


# ============================================================================
# Node: QuantFunc Base Model Auto Loader with Download
# ============================================================================

def _get_base_model_repo_dropdowns():
    """Get base model repo dropdown options from cache."""
    try:
        from .model_auto_loader import get_base_model_repo_options
        return get_base_model_repo_options()
    except Exception:
        return ["None"]


class QuantFuncBaseModelAutoLoaderWithDownload:
    """Auto-discover and download base models from ModelScope/HuggingFace.

    Searches upstream repos for available base models and downloads
    to ComfyUI/models/diffusers/ on first use.
    Outputs the local model directory path as a string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from .model_auto_loader import _DATA_SOURCES
        repo_opts = _get_base_model_repo_dropdowns()
        return {
            "required": {
                "base_model_repo": (repo_opts, {"tooltip": "Upstream base model repository. Auto-discovered from ModelScope."}),
                "data_source": (_DATA_SOURCES, {"default": "modelscope", "tooltip": "Download source: modelscope (China) or huggingface"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_dir",)
    FUNCTION = "load_base_model"
    CATEGORY = "QuantFunc"

    def load_base_model(self, base_model_repo, data_source):
        if not base_model_repo or base_model_repo == "None":
            return ("",)

        from .model_auto_loader import download_base_model_to_diffusers
        path = download_base_model_to_diffusers(base_model_repo, data_source)
        return (path,)


# ============================================================================
# Node: QuantFunc Transformer Auto Loader
# ============================================================================

def _get_local_transformer_file_options():
    """Recursively scan models/QuantFunc/transformer/ for .safetensors files."""
    try:
        from .model_auto_loader import get_models_dir
        tf_dir = os.path.join(get_models_dir(), "transformer")
        if os.path.isdir(tf_dir):
            files = []
            for root, _, filenames in os.walk(tf_dir):
                for f in filenames:
                    if f.endswith(".safetensors"):
                        rel = os.path.relpath(os.path.join(root, f), tf_dir)
                        files.append(rel.replace("\\", "/"))
            if files:
                return ["None"] + sorted(files)
    except Exception:
        pass
    return ["None"]


class QuantFuncTransformerAutoLoader:
    """Auto-load transformer weights from models/QuantFunc/transformer/ directory.

    Recursively scans for .safetensors files and presents them as a dropdown.
    Outputs the file path as a string for connecting to ModelLoader's transformer input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        tf_opts = _get_local_transformer_file_options()
        return {
            "required": {
                "transformer_file": (tf_opts, {"tooltip": "Transformer weights from models/QuantFunc/transformer/"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transformer_path",)
    FUNCTION = "load_transformer"
    CATEGORY = "QuantFunc"

    def load_transformer(self, transformer_file):
        if not transformer_file or transformer_file == "None":
            return ("",)

        from .model_auto_loader import get_models_dir
        tf_path = os.path.join(get_models_dir(), "transformer", transformer_file)
        if not os.path.exists(tf_path):
            raise RuntimeError("Transformer file not found: {}".format(tf_path))
        return (tf_path,)


# ============================================================================
# Node: QuantFunc LoRA Auto Loader
# ============================================================================

def _get_comfyui_dir():
    """Return the ComfyUI root directory (parent of custom_nodes)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_lora_file_options():
    """Recursively scan ComfyUI/models/loras/ for .safetensors files."""
    try:
        lora_dir = os.path.join(_get_comfyui_dir(), "models", "loras")
        if os.path.isdir(lora_dir):
            files = []
            for root, _, filenames in os.walk(lora_dir):
                for f in filenames:
                    if f.endswith(".safetensors"):
                        rel = os.path.relpath(os.path.join(root, f), lora_dir)
                        files.append(rel.replace("\\", "/"))
            if files:
                return ["None"] + sorted(files)
    except Exception:
        pass
    return ["None"]


class QuantFuncLoRAAutoLoader:
    """Auto-load LoRA weights from models/QuantFunc/lora/ directory.

    Scans the lora directory for .safetensors files and presents them
    as a dropdown. Appends the selected LoRA to the pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_opts = _get_lora_file_options()
        return {
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "lora_file": (lora_opts, {"tooltip": "LoRA weights from models/loras/"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                           "tooltip": "LoRA weight scale (1.0 = full strength)"}),
            },
        }

    RETURN_TYPES = ("QUANTFUNC_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "add_lora"
    CATEGORY = "QuantFunc"

    def add_lora(self, pipeline, lora_file, scale):
        cfg = dict(pipeline)
        cfg["options"] = dict(cfg.get("options", {}))

        if lora_file and lora_file != "None":
            lora_path = os.path.join(_get_comfyui_dir(), "models", "loras", lora_file)
            if not os.path.exists(lora_path):
                raise RuntimeError("LoRA file not found: {}".format(lora_path))
            loras = list(cfg["options"].get("lora", []))
            if scale != 1.0:
                loras.append("{}:{}".format(lora_path, scale))
            else:
                loras.append(lora_path)
            cfg["options"]["lora"] = loras

        return (cfg,)


# ============================================================================
# Node: QuantFunc LoRA
# ============================================================================

class QuantFuncLoRALoader:
    """Append a LoRA to the pipeline. Chain multiple LoRA nodes together."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "lora_path": ("STRING", {"default": "", "tooltip": "Path to LoRA safetensors file"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                           "tooltip": "LoRA weight scale (1.0 = full strength)"}),
            },
        }

    RETURN_TYPES = ("QUANTFUNC_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "add_lora"
    CATEGORY = "QuantFunc"

    def add_lora(self, pipeline, lora_path, scale):
        cfg = dict(pipeline)
        cfg["options"] = dict(cfg.get("options", {}))

        if lora_path:
            loras = list(cfg["options"].get("lora", []))
            if scale != 1.0:
                loras.append(f"{lora_path}:{scale}")
            else:
                loras.append(lora_path)
            cfg["options"]["lora"] = loras

        return (cfg,)


# ============================================================================
# Node: QuantFunc LoRA Config
# ============================================================================

class QuantFuncLoRAConfig:
    """Configure LoRA merge strategy. Place after LoRA nodes, before Generate."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "max_rank": ("INT", {"default": 512, "min": 1, "max": 1024, "step": 1,
                              "tooltip": "Maximum LoRA rank for SVD merge (higher = more accurate, more VRAM)"}),
                "merge_method": (["auto", "itc", "awsvd", "rop", "concat"], {"default": "auto",
                                  "tooltip": "auto: best for model type; itc: IT+C; awsvd: activation-weighted SVD; rop: ROP+W; concat: concatenate weights"}),
            },
        }

    RETURN_TYPES = ("QUANTFUNC_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "configure"
    CATEGORY = "QuantFunc"

    def configure(self, pipeline, max_rank, merge_method):
        cfg = dict(pipeline)
        cfg["options"] = dict(cfg.get("options", {}))

        cfg["options"]["lora_max_rank"] = max_rank
        if merge_method == "concat":
            cfg["options"]["lora_concat"] = True
            cfg["options"]["lora_merge_method"] = "auto"
        else:
            cfg["options"]["lora_merge_method"] = merge_method
            cfg["options"]["lora_concat"] = False

        return (cfg,)


# ============================================================================
# Node: QuantFunc Generate
# ============================================================================

class QuantFuncGenerate:
    """Generate an image. Creates/reuses a cached pipeline from the config.
    Edit mode is auto-detected when ref_image is connected.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "prompt": ("STRING", {"default": "A cute cat", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 64}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            },
            "optional": {
                "ref_images": ("QUANTFUNC_IMAGE_LIST", {"tooltip": "Reference images for edit mode (from ImageList node)"}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "true_cfg_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (["euler", "heun", "dpm++2m", "dpm++2m_sde", "euler_a", "ddim"], {
                    "default": "euler",
                    "tooltip": "Sampling algorithm:\n"
                               "• euler — 1st order, fast, deterministic\n"
                               "• heun — 2nd order, higher quality, 2x slower\n"
                               "• dpm++2m — 2nd order multistep, deterministic\n"
                               "• dpm++2m_sde — dpm++2m + noise (use sampler_eta)\n"
                               "• euler_a — euler + noise (use sampler_eta)\n"
                               "• ddim — classic DDIM, deterministic (eta=0) or stochastic (eta>0)",
                }),
                "sampler_eta": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Noise scale for stochastic samplers (dpm++2m_sde, euler_a, ddim).\n"
                               "0 = deterministic (no effect). Only used by stochastic samplers.\n"
                               "Recommended 0.2~0.5 for ≤20 steps. Higher eta needs more steps.",
                }),
                "unload_every_time": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload all models from GPU to CPU after generation, freeing VRAM.\n"
                               "Models stay in CPU RAM for fast reload on the next run.\n"
                               "Enable this when sharing VRAM with other ComfyUI nodes.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "workflow_prompt": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "QuantFunc"

    def generate(self, pipeline, prompt, width, height, steps, seed,
                 guidance_scale, ref_images=None,
                 negative_prompt="", true_cfg_scale=4.0,
                 sampler_name="euler", sampler_eta=0.0,
                 unload_every_time=False, unique_id=None,
                 workflow_prompt=None):
        import torch

        # Handle unload request
        if pipeline.get("unload"):
            _manager.destroy_all()
            from PIL import Image as PILImage, ImageDraw
            msg_img = PILImage.new("RGB", (512, 128), color=(40, 40, 40))
            draw = ImageDraw.Draw(msg_img)
            draw.text((20, 45), "Model unloaded successfully.", fill=(200, 200, 200))
            msg_np = np.array(msg_img, dtype=np.float32) / 255.0
            msg_tensor = torch.from_numpy(msg_np).unsqueeze(0)
            logging.info("[QuantFunc] Model unloaded successfully.")
            return (msg_tensor,)

        # Auto-detect edit mode from ref_images
        cfg = dict(pipeline)
        cfg["options"] = dict(cfg.get("options", {}))
        # Unpack ImageList dict format
        ref_img_resize = "720"
        if ref_images is not None and isinstance(ref_images, dict):
            # New: ref_img_resize ("720" / "1024" / "origin")
            # Backwards compat: old workflows may still send keep_ref_img_size (bool)
            if "ref_img_resize" in ref_images:
                ref_img_resize = ref_images["ref_img_resize"]
            elif ref_images.get("keep_ref_img_size"):
                ref_img_resize = "1024"
            ref_images = ref_images["images"]
        if ref_images is not None:
            cfg["options"]["edit_mode"] = True

        # Collect live QuantFuncGenerate node ids from the current workflow.
        # Exclude nodes whose required `pipeline` input isn't connected — those
        # never execute and should release their cached pipeline refs.
        alive_ids = None
        if isinstance(workflow_prompt, dict):
            alive_ids = set()
            for nid, node in workflow_prompt.items():
                if not (isinstance(node, dict) and node.get("class_type") == "QuantFuncGenerate"):
                    continue
                pipe_in = node.get("inputs", {}).get("pipeline")
                # Connected inputs are [source_node_id, output_slot]; anything
                # else (None / missing) means dangling.
                if not (isinstance(pipe_in, list) and len(pipe_in) == 2):
                    continue
                alive_ids.add(nid)
        cache_key = _manager.ensure_pipeline(cfg, node_id=unique_id,
                                             alive_node_ids=alive_ids)

        # Create ComfyUI progress bar
        pbar = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(steps)
        except Exception:
            pass

        try:
            if ref_images is not None:
                # Save each ref image to temp file
                from PIL import Image
                tmp_paths = []
                for img_tensor in ref_images:
                    for i in range(img_tensor.shape[0]):
                        fd, tmp_path = tempfile.mkstemp(suffix=".bmp")
                        os.close(fd)
                        img_np = (img_tensor[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                        Image.fromarray(img_np).save(tmp_path)
                        tmp_paths.append(tmp_path)

                neg = negative_prompt if isinstance(negative_prompt, str) and negative_prompt else ""
                i2i_opts = {}
                if sampler_name != "euler":
                    i2i_opts["sampler"] = sampler_name
                if sampler_eta > 0.0:
                    i2i_opts["eta"] = sampler_eta
                i2i_opts["ref_img_resize"] = ref_img_resize
                i2i_opts_json = json.dumps(i2i_opts) if i2i_opts else None
                arr = _manager.image_to_image(
                    cache_key=cache_key,
                    prompt=prompt, ref_paths=tmp_paths,
                    height=height, width=width, steps=steps, seed=seed,
                    true_cfg_scale=true_cfg_scale, negative_prompt=neg,
                    options_json=i2i_opts_json, pbar=pbar)
            else:
                t2i_opts = {}
                neg = negative_prompt if isinstance(negative_prompt, str) and negative_prompt else ""
                if neg and true_cfg_scale > 1.0:
                    t2i_opts["negative_prompt"] = neg
                    t2i_opts["true_cfg_scale"] = true_cfg_scale
                t2i_opts["sampler"] = sampler_name
                if sampler_eta > 0.0:
                    t2i_opts["eta"] = sampler_eta
                opts_json = json.dumps(t2i_opts) if t2i_opts else None
                logging.info("[QuantFunc] t2i sampler_name=%s, sampler_eta=%s, opts_json=%s",
                             sampler_name, sampler_eta, opts_json)

                arr = _manager.text_to_image(
                    cache_key=cache_key,
                    prompt=prompt, height=height, width=width,
                    steps=steps, seed=seed, guidance_scale=guidance_scale,
                    options_json=opts_json, pbar=pbar)

            if unload_every_time:
                _manager.unload_pipeline(cache_key)

            return (torch.from_numpy(arr).unsqueeze(0),)  # [1, H, W, 3]

        except InterruptedError:
            logging.info("[QuantFunc] Generation interrupted, returning blank image.")
            blank = torch.zeros(1, height, width, 3, dtype=torch.float32)
            return (blank,)


# ============================================================================
# Node: QuantFunc Image List
# ============================================================================

class QuantFuncImageList:
    """Reference images for edit mode. Single or multiple images supported."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"image{i}": ("IMAGE",) for i in range(2, 11)}
        optional["ref_img_resize"] = (["720", "1024", "origin"], {
            "default": "720",
            "tooltip": "Reference image resize mode (edit pipelines):\n"
                       "  720  — long-side cap at 720 px (default, fastest)\n"
                       "  1024 — long-side cap at 1024 px (better quality, slower)\n"
                       "  origin — keep original size, only floor each dim to a multiple of 16",
        })
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("QUANTFUNC_IMAGE_LIST",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "QuantFunc"

    def combine(self, image1, ref_img_resize="720", **kwargs):
        images = [image1]
        for i in range(2, 11):
            img = kwargs.get(f"image{i}")
            if img is not None:
                images.append(img)
        return ({"images": images, "ref_img_resize": ref_img_resize},)


# ============================================================================
# Node: QuantFunc Export
# ============================================================================

class QuantFuncExport:
    """Export a pre-quantized model directory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "export_path": ("STRING", {"default": "", "tooltip": "Output directory for exported model"}),
                "export_mode": (["all", "custom"], {
                    "default": "all",
                    "tooltip": "'all' copies entire model (vae, tokenizer, etc.) for standalone use; 'custom' selects individual components"
                }),
            },
            "optional": {
                "export_transformer": ("BOOLEAN", {"default": True}),
                "export_text_encoder": ("BOOLEAN", {"default": False}),
                "export_vision_encoder": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "export_model"
    CATEGORY = "QuantFunc"

    def export_model(self, pipeline, export_path, export_mode="all",
                     export_transformer=True, export_text_encoder=False,
                     export_vision_encoder=False):
        if not export_path:
            raise ValueError("export_path is required")

        if export_mode == "all":
            components = ["all"]
        else:
            components = []
            if export_transformer:
                components.append("transformer")
            if export_text_encoder:
                components.append("text_encoder")
            if export_vision_encoder:
                components.append("vision_encoder")
            if not components:
                raise ValueError("At least one component must be selected for export")

        # Inject export_models into pipeline config options
        if "options" not in pipeline:
            pipeline["options"] = {}
        pipeline["options"]["export_models"] = ",".join(components)

        _manager.export_model(pipeline, export_path)
        logging.info("[QuantFunc] Export complete: %s", export_path)
        return {}


# ============================================================================
# Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "QuantFuncPipelineConfig": QuantFuncPipelineConfig,
    "QuantFuncModelLoader": QuantFuncModelLoader,
    "QuantFuncModelAutoLoader": QuantFuncModelAutoLoader,
    "QuantFuncPrequantAutoLoader": QuantFuncPrequantAutoLoader,
    "QuantFuncPrecisionConfigAutoLoader": QuantFuncPrecisionConfigAutoLoader,
    "QuantFuncBaseSeriesModelAutoLoader": QuantFuncBaseSeriesModelAutoLoader,
    "QuantFuncBaseModelAutoLoader": QuantFuncBaseModelAutoLoader,
    "QuantFuncBaseModelAutoLoaderWithDownload": QuantFuncBaseModelAutoLoaderWithDownload,
    "QuantFuncTransformerAutoLoader": QuantFuncTransformerAutoLoader,
    "QuantFuncLoRAAutoLoader": QuantFuncLoRAAutoLoader,
    "QuantFuncLoRALoader": QuantFuncLoRALoader,
    "QuantFuncLoRAConfig": QuantFuncLoRAConfig,
    "QuantFuncGenerate": QuantFuncGenerate,
    "QuantFuncImageList": QuantFuncImageList,
    "QuantFuncExport": QuantFuncExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuantFuncPipelineConfig": "QuantFunc Pipeline Config",
    "QuantFuncModelLoader": "QuantFunc Model Loader",
    "QuantFuncModelAutoLoader": "QuantFunc Model Auto Loader",
    "QuantFuncPrequantAutoLoader": "QuantFunc Prequant Auto Loader",
    "QuantFuncPrecisionConfigAutoLoader": "QuantFunc Precision Config Auto Loader",
    "QuantFuncBaseSeriesModelAutoLoader": "QuantFunc Base Series Model Auto Loader",
    "QuantFuncBaseModelAutoLoader": "QuantFunc Base Model Auto Loader",
    "QuantFuncBaseModelAutoLoaderWithDownload": "QuantFunc Base Model Auto Loader with Download",
    "QuantFuncTransformerAutoLoader": "QuantFunc Transformer Auto Loader",
    "QuantFuncLoRAAutoLoader": "QuantFunc LoRA Auto Loader",
    "QuantFuncLoRALoader": "QuantFunc LoRA",
    "QuantFuncLoRAConfig": "QuantFunc LoRA Config",
    "QuantFuncGenerate": "QuantFunc Generate",
    "QuantFuncImageList": "QuantFunc Image List",
    "QuantFuncExport": "QuantFunc Export",
}
