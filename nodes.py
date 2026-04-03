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
import hashlib
import json
import logging
import numpy as np
import os
import platform
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


class WorkerManager:
    """Manages a QuantFunc worker subprocess with isolated CUDA libraries."""

    def __init__(self):
        self._process = None
        self._stdin = None
        self._stdout = None
        self._stderr_thread = None
        self._current_key = None
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
            cuda_path = env.get("CUDA_PATH", "/usr/local/cuda")
            lib64 = os.path.join(cuda_path, "lib64")
            if os.path.isdir(lib64):
                env["LD_LIBRARY_PATH"] = lib64 + ":" + env.get("LD_LIBRARY_PATH", "")
        return env

    def _start_worker(self, dll_path, env):
        """Start worker subprocess and wait for ready signal.
        Returns (success, error_message).
        """
        python_exe = os.environ.get("QUANTFUNC_PYTHON", "") or sys.executable
        cmd = [python_exe, _WORKER_PY, "--dll-path", dll_path]

        creation_flags = 0
        if _IS_WINDOWS:
            creation_flags = subprocess.CREATE_NO_WINDOW

        logging.info("[QuantFunc] Starting worker: %s (python=%s)",
                     " ".join(cmd[:4]), python_exe)

        try:
            self._process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env, creationflags=creation_flags)
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
        if ready is None or ready.get("type") != "ready":
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
                from .lib_setup import detect_cuda_major, _download_dep_zip
                cuda_major = detect_cuda_major()
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
        On first DLL load failure (Windows), downloads deps and retries once.
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
            self._current_key = None

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

        # On Windows, first failure may be missing dep DLLs — download and retry
        if _IS_WINDOWS and self._try_download_deps(dll_path):
            logging.info("[QuantFunc] Dependencies installed, retrying worker...")
            env = self._build_worker_env(dll_dir)  # rebuild (deps now in dll_dir)
            ok2, err2 = self._start_worker(dll_path, env)
            if ok2:
                return
            raise RuntimeError(err2)

        raise RuntimeError(err)

    def _stderr_reader(self):
        """Forward worker's stderr to logging."""
        try:
            for line in self._process.stderr:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logging.info("[QuantFunc-worker] %s", text)
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
            self._current_key = None

    # ── IPC ──

    def _next_req_id(self):
        self._req_counter += 1
        return self._req_counter

    def _send_command(self, cmd):
        """Send a JSON command to worker."""
        data = json.dumps(cmd, ensure_ascii=True).encode("utf-8") + b"\n"
        self._stdin.write(data)
        self._stdin.flush()

    def _read_response(self, timeout=600):
        """Read one JSON line from worker stdout."""
        # Simple blocking read with timeout via thread
        result = [None]
        def reader():
            try:
                line = self._stdout.readline()
                if line:
                    result[0] = json.loads(line.decode("utf-8").strip())
            except Exception as e:
                result[0] = {"type": "error", "error_message": str(e)}
        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            return None  # timeout
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

    def _call(self, cmd, progress_cb=None, timeout=600):
        """Send command and collect response, relaying progress."""
        self._send_command(cmd)

        while True:
            resp = self._read_response(timeout=timeout)
            if resp is None:
                self._kill_worker()
                raise RuntimeError("Worker timeout")

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

    def set_api_key(self, api_key):
        """Hot-swap API key on the loaded pipeline (no pipeline recreation)."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return  # no worker running
            cmd = {
                "cmd": "set_api_key",
                "req_id": self._next_req_id(),
                "api_key": api_key,
            }
            self._call(cmd, timeout=30)
            self._current_api_key = api_key

    def ensure_pipeline(self, cfg):
        """Ensure pipeline matching cfg is loaded in worker."""
        with self._lock:
            self._ensure_worker()

            key = _make_cache_key(cfg)
            opts = cfg.get("options", {})
            new_api_key = opts.get("api_key", "")

            if key == self._current_key:
                # Pipeline config unchanged — check if API key changed
                if new_api_key and new_api_key != getattr(self, "_current_api_key", ""):
                    self._set_api_key_locked(new_api_key)
                return  # reuse existing pipeline

            if self._current_key is not None:
                logging.info("[QuantFunc] Config changed, recreating pipeline...")

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
            self._current_key = key
            self._current_api_key = new_api_key
            logging.info("[QuantFunc] Pipeline ready.")

    def _set_api_key_locked(self, api_key):
        """Internal: set API key while already holding self._lock."""
        cmd = {
            "cmd": "set_api_key",
            "req_id": self._next_req_id(),
            "api_key": api_key,
        }
        self._call(cmd, timeout=30)
        self._current_api_key = api_key
        logging.info("[QuantFunc] API key updated (hot-swap).")

    def text_to_image(self, prompt, height, width, steps, seed,
                      guidance_scale, options_json=None, pbar=None):
        """Generate text-to-image. Returns [H, W, 3] float32 numpy array."""
        with self._lock:
            self._ensure_worker()

            def on_progress(step, total):
                if pbar is not None:
                    pbar.update(1)

            cmd = {
                "cmd": "text_to_image",
                "req_id": self._next_req_id(),
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

    def image_to_image(self, prompt, ref_paths, height, width, steps, seed,
                       true_cfg_scale=4.0, negative_prompt="",
                       options_json=None, pbar=None):
        """Generate image-to-image. Returns [H, W, 3] float32 numpy array."""
        with self._lock:
            self._ensure_worker()

            def on_progress(step, total):
                if pbar is not None:
                    pbar.update(1)

            cmd = {
                "cmd": "image_to_image",
                "req_id": self._next_req_id(),
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

            # Destroy loaded pipeline first to free VRAM
            if self._current_key is not None:
                self._call({"cmd": "destroy", "req_id": self._next_req_id()})
                self._current_key = None

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

    def destroy_all(self):
        """Destroy loaded pipeline (keep worker alive)."""
        with self._lock:
            if self._process and self._process.poll() is None and self._current_key is not None:
                try:
                    self._call({"cmd": "destroy", "req_id": self._next_req_id()}, timeout=30)
                except Exception:
                    pass
                self._current_key = None

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
            self._current_key = None

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

    def _hooked_free_memory(memory_required, device, keep_loaded=[], **kwargs):
        if _manager._current_key is not None and memory_required > 0:
            # Only unload if there isn't enough free VRAM to satisfy the request
            try:
                import torch
                free_vram, _ = torch.cuda.mem_get_info(device)
            except Exception:
                free_vram = 0
            if free_vram < memory_required:
                logging.info("[QuantFunc] Auto-unloading pipelines to free VRAM for other models "
                             f"(need {memory_required // 1024**2} MB, free {free_vram // 1024**2} MB)")
                _manager.destroy_all()
        return _original_free_memory(memory_required, device, keep_loaded=keep_loaded, **kwargs)

    _mm.free_memory = _hooked_free_memory
except Exception:
    pass


# ============================================================================
# Node: QuantFunc Pipeline Config
# ============================================================================

class QuantFuncPipelineConfig:
    """Advanced pipeline configuration for model initialization.
    Overrides auto_optimize defaults. If not connected, model uses auto_optimize with no overrides.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cpu_offload": ("BOOLEAN", {"default": True, "tooltip": "Offload idle models to CPU to save VRAM"}),
                "layer_offload": ("BOOLEAN", {"default": False, "tooltip": "Per-layer offload for transformer (slower but uses less VRAM)"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Tile-based VAE decoding to reduce VRAM (auto-enabled at high resolution)"}),
                "attention_backend": (["auto", "sage", "flash", "sdpa"], {"default": "auto",
                                      "tooltip": "Attention implementation: auto picks best for your GPU"}),
                "precision": (["bf16", "fp16"], {"default": "bf16", "tooltip": "Compute precision for pipeline"}),
                "text_precision": (["int4", "int8", "fp4", "fp8", "fp16"], {"default": "int4",
                                    "tooltip": "Text encoder quantization precision (fp4 requires SM120+/Blackwell)"}),
            },
            "optional": {
                "adaptive_offload": (["off", "normal", "aggressive"], {"default": "aggressive",
                                     "tooltip": "Adaptive GPU caching: aggressive keeps more blocks on GPU between runs"}),
                "offload_compression": (["none", "auto", "int8", "fp8"], {"default": "auto",
                                        "tooltip": "Compress offloaded weights to reduce PCIe transfer time"}),
                "vae_tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64,
                                  "tooltip": "VAE tile size in pixels (0 = auto)"}),
                "pinned_memory_limit": ("STRING", {"default": "", "tooltip": "Max pinned CPU memory: '60%', '48G', '48M', or empty for auto"}),
            }
        }

    RETURN_TYPES = ("QUANTFUNC_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build_config"
    CATEGORY = "QuantFunc"

    def build_config(self, cpu_offload, layer_offload, tiled_vae, attention_backend,
                     precision, text_precision, adaptive_offload="aggressive",
                     offload_compression="auto", vae_tile_size=0, pinned_memory_limit=""):
        config = {
            "cpu_offload": cpu_offload,
            "layer_offload": layer_offload,
            "tiled_vae": tiled_vae,
            "attention_backend": attention_backend,
            "precision": precision,
            "text_precision": text_precision,
        }
        if adaptive_offload == "off":
            config["adaptive_offload"] = ""
        else:
            config["adaptive_offload"] = adaptive_offload

        config["offload_compression"] = offload_compression

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
                   fused_mod=False, **kwargs):
        scheduler_config = scheduler_config.strip() if isinstance(scheduler_config, str) else ""
        # Validate: scheduler_config must be a file path (not a bare number or random text)
        if scheduler_config and not os.path.exists(scheduler_config):
            logging.warning(f"[QuantFunc] scheduler_config path does not exist: {scheduler_config!r}, ignoring")
            scheduler_config = ""
        precision_config = precision_config.strip() if isinstance(precision_config, str) else ""
        prequant_weights = prequant_weights.strip() if isinstance(prequant_weights, str) else ""
        transformer_path = transformer_path if isinstance(transformer_path, str) and transformer_path else ""

        api_key = api_key.strip() if isinstance(api_key, str) else ""

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

        text_precision = "int4"
        if config and isinstance(config, dict):
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
                "scheduler_config": ("STRING", {"default": "", "tooltip": "Scheduler JSON config path (for Lightning models)"}),
                "manual_unload_model": ("BOOLEAN", {"default": False, "tooltip": "Activate to manually unload the model and free GPU memory."}),
            }
        }

    RETURN_TYPES = ("QUANTFUNC_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "QuantFunc"

    def load_model(self, model_series, model_backend, device, data_source,
                   config=None, manual_unload_model=False,
                   transformer="None", api_key="", scheduler_config="",
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
        scheduler_config = scheduler_config.strip() if isinstance(scheduler_config, str) else ""
        if scheduler_config and not os.path.exists(scheduler_config):
            logging.warning("[QuantFunc] scheduler_config path does not exist: %r, ignoring",
                            scheduler_config)
            scheduler_config = ""

        api_key = api_key.strip() if isinstance(api_key, str) else ""

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

        text_precision = "int4"
        if config and isinstance(config, dict):
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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "QuantFunc"

    def generate(self, pipeline, prompt, width, height, steps, seed,
                 guidance_scale, ref_images=None,
                 negative_prompt="", true_cfg_scale=4.0,
                 sampler_name="euler", sampler_eta=0.0):
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
        if ref_images is not None:
            cfg["options"]["edit_mode"] = True

        _manager.ensure_pipeline(cfg)

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
                i2i_opts_json = json.dumps(i2i_opts) if i2i_opts else None
                arr = _manager.image_to_image(
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
                    prompt=prompt, height=height, width=width,
                    steps=steps, seed=seed, guidance_scale=guidance_scale,
                    options_json=opts_json, pbar=pbar)

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

    def combine(self, image1, **kwargs):
        images = [image1]
        for i in range(2, 11):
            img = kwargs.get(f"image{i}")
            if img is not None:
                images.append(img)
        return (images,)


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
    "QuantFuncLoRALoader": "QuantFunc LoRA",
    "QuantFuncLoRAConfig": "QuantFunc LoRA Config",
    "QuantFuncGenerate": "QuantFunc Generate",
    "QuantFuncImageList": "QuantFunc Image List",
    "QuantFuncExport": "QuantFunc Export",
}
