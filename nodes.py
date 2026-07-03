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
import datetime as _datetime
import hashlib
import json
import logging
import numpy as np
import os
import platform
import queue as _queue
import re as _re
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time

# Worker stderr noise filter: every line lands in the per-worker temp log
# (see _stderr_reader); only lines that match these patterns are echoed to
# the ComfyUI console. Keeps the user informed of per-stage timings + errors
# without the engine's verbose info dump (every block-quant warn, every
# auth handshake, every VAE-DIAG trace).
_WORKER_CONSOLE_KEEP = _re.compile(
    r'\[load\] '                          # component load timings
    r'|Progress: \d+/\d+'                 # per-step diffusion progress
    r'|\[total\]'                         # end-to-end timing
    r'|\[\d+/\d+\] [a-z_]+:'              # per-stage component time (e.g. [3/5] vae_encoder: 178 ms)
    r'|Saving image|Done!'                # image save
    r'|Bundle export complete'            # export
    r'|Pipeline type:'                    # pipeline kind
    r'|Strategy:'                         # auto-optimize summary
    r'|Error|Failed|fatal|exception|terminate|segfault'  # error keywords
    r'|out of memory|OOM'                 # OOM
    r'|Tensor .* not found'               # missing tensor
    r'|abort'                             # abort
)


class _WorkerLogWriter:
    """Singleton background writer for worker stderr logs.

    Why a global thread?  The stderr reader per-WorkerManager is on the
    critical path of every engine log line (info, warn, debug). Doing the
    file write + flush() inline meant a syscall per line, blocking the
    reader thread; if the kernel pipe buffer fills (~64 KB on Linux) the
    engine's own stderr write() blocks, stalling generation.

    With a dedicated writer:
      - readers only `put_nowait` into a bounded queue (microseconds)
      - one OS thread owns all file handles; ~64 KB block-buffered writes
        + flush every 2 s of idle/work and on error keywords
      - multi-pipeline isolation: each WorkerManager registers its own
        log path; the writer keys file handles by path and never mixes
        streams across workers
      - bounded queue (8192 entries): if the writer is briefly behind,
        readers drop the oldest entries rather than blocking the worker
        (we prefer "lost log lines" over "stalled generation")
    """
    _instance = None
    _init_lock = threading.Lock()

    @classmethod
    def get(cls):
        if cls._instance is not None:
            return cls._instance
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._q: "_queue.Queue[tuple[str, str|None, bool]]" = _queue.Queue(maxsize=8192)
        self._fhs: "dict[str, object]" = {}
        self._stop = threading.Event()
        self._t = threading.Thread(
            target=self._loop, daemon=True, name="QuantFuncLogWriter")
        self._t.start()
        atexit.register(self._shutdown)

    def open(self, path: str) -> None:
        """Pre-open log file. Idempotent."""
        if path in self._fhs:
            return
        try:
            self._fhs[path] = open(
                path, "w", buffering=64 * 1024,
                encoding="utf-8", errors="replace")
        except Exception as e:
            logging.warning("[QuantFunc-worker] log open failed for %s: %s", path, e)

    def write(self, path: str, text: str, *, flush: bool = False) -> None:
        """Enqueue a write. Non-blocking — drops oldest if queue full."""
        try:
            self._q.put_nowait((path, text, flush))
        except _queue.Full:
            # Backpressure: drop one to free a slot, then enqueue. Avoids
            # blocking the stderr reader on transient writer stalls.
            try:
                self._q.get_nowait()
            except _queue.Empty:
                pass
            try:
                self._q.put_nowait((path, text, flush))
            except _queue.Full:
                pass

    def close(self, path: str) -> None:
        """Schedule final flush + close of one path's handle."""
        try:
            # text=None signals "close this fd"
            self._q.put_nowait((path, None, True))
        except _queue.Full:
            pass

    def _loop(self) -> None:
        last_flush = time.monotonic()
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=1.0)
            except _queue.Empty:
                # Idle: periodic flush so `tail -f` sees recent progress
                now = time.monotonic()
                if now - last_flush > 2.0:
                    for fh in list(self._fhs.values()):
                        try: fh.flush()
                        except Exception: pass
                    last_flush = now
                continue
            path, text, do_flush = item
            fh = self._fhs.get(path)
            if fh is None:
                # Lazy open if writer race ahead of open()
                try:
                    fh = open(path, "a", buffering=64 * 1024,
                              encoding="utf-8", errors="replace")
                    self._fhs[path] = fh
                except Exception:
                    continue
            if text is None:
                # Close request
                try:
                    fh.flush(); fh.close()
                except Exception: pass
                self._fhs.pop(path, None)
                continue
            try:
                fh.write(text)
                fh.write("\n")
                if do_flush:
                    fh.flush()
                    last_flush = time.monotonic()
            except Exception:
                pass
            # Opportunistic periodic flush (cheap monotonic compare per item)
            now = time.monotonic()
            if now - last_flush > 2.0:
                for f in list(self._fhs.values()):
                    try: f.flush()
                    except Exception: pass
                last_flush = now

    def _shutdown(self) -> None:
        self._stop.set()
        try:
            self._t.join(timeout=2.0)
        except Exception:
            pass
        # Drain any remaining items + close all handles
        while True:
            try:
                path, text, _ = self._q.get_nowait()
            except _queue.Empty:
                break
            fh = self._fhs.get(path)
            if fh is None or text is None:
                continue
            try: fh.write(text + "\n")
            except Exception: pass
        for fh in self._fhs.values():
            try: fh.flush(); fh.close()
            except Exception: pass
        self._fhs.clear()

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


def _detect_model_backend(transformer_path: str, model_dir: str) -> str:
    """Auto-detect svdq vs lighting from transformer safetensors (metadata first,
    then tensor-key fingerprint). Returns "svdq" or "lighting"; defaults to
    "lighting" when nothing matches the SVDQ signature (Lighting handles both
    FP16-base runtime-quant and `lighting_precomputed` reload).

    SVDQ signature (Nunchaku export):
      - metadata `quantization_config.method == "svdquant"`
      - or tensor keys like `transformer_blocks.0.attn.to_qkv.qweight`
        (BARE `qweight`, no underscore prefix) co-existing with
        `*.lora_down` / `*.smooth_orig` sidecars

    Lighting signature:
      - metadata method `lighting_precomputed` / `lighting` / `flux2klein_runtime`
      - or tensor keys with `_qweight_w4a4` / `_qweight` (underscore prefix
        → QuantFunc Lighting export's persistent buffers)
      - or only FP16 `*.weight` tensors (FP16 base, runtime-quantize)
    """
    # Resolve probe target: explicit transformer arg → model_dir/transformer/
    candidates = []
    if transformer_path:
        if os.path.isfile(transformer_path):
            candidates.append(transformer_path)
        elif os.path.isdir(transformer_path):
            try:
                candidates += sorted(
                    os.path.join(transformer_path, f)
                    for f in os.listdir(transformer_path)
                    if f.endswith(".safetensors")
                )[:1]
            except OSError:
                pass
    if not candidates and model_dir:
        xfm_dir = os.path.join(model_dir, "transformer")
        if os.path.isdir(xfm_dir):
            try:
                candidates += sorted(
                    os.path.join(xfm_dir, f) for f in os.listdir(xfm_dir)
                    if f.endswith(".safetensors")
                )[:1]
            except OSError:
                pass
    if not candidates:
        return "lighting"  # safe fallback — engine errors clearly if file missing

    probe = candidates[0]
    try:
        from .format_adapters.tools.safetensors_io import read_safetensors_header
    except Exception as e:
        logging.debug("[QuantFunc] backend detect: cannot import helpers: %s", e)
        return "lighting"

    # ONE header read — JSON-only, no tensor data. Typical < 1 MB even for
    # 17 GB transformers; ~5-20 ms on warm cache.
    try:
        header = read_safetensors_header(probe)
    except Exception as e:
        logging.debug("[QuantFunc] backend detect: header read failed: %s", e)
        return "lighting"

    # 1. Metadata `method` — ground truth when present.
    meta = header.get("__metadata__", {}) or {}
    qc_str = meta.get("quantization_config", "")
    if qc_str:
        try:
            method = json.loads(qc_str).get("method", "")
            if method == "svdquant":
                return "svdq"
            if method in ("lighting_precomputed", "lighting", "flux2klein_runtime"):
                return "lighting"
        except json.JSONDecodeError:
            pass
    if meta.get("model_class", "").find("Nunchaku") >= 0:
        return "svdq"

    # 2. Tensor-key fingerprint — first ~200 keys from the same header dict.
    seen_underscore_qweight = False
    seen_bare_qweight = False
    seen_lora_sidecar = False
    scanned = 0
    for k in header:
        if k == "__metadata__":
            continue
        scanned += 1
        if scanned > 200:
            break
        if "._qweight" in k:
            seen_underscore_qweight = True
        elif k.endswith(".qweight"):
            seen_bare_qweight = True
        if k.endswith(".lora_down") or ".lora_down." in k or k.endswith(".smooth_orig"):
            seen_lora_sidecar = True
    if seen_underscore_qweight:
        return "lighting"
    if seen_bare_qweight and seen_lora_sidecar:
        return "svdq"

    return "lighting"


def _read_class_name(json_path):
    """Best-effort read of `_class_name` from a diffusers config/model_index JSON.
    Returns "" on any failure (missing file, bad JSON)."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return str(json.load(f).get("_class_name", "") or "")
    except Exception:
        return ""


def _pipeline_video_family(cfg):
    """Return 'wan' | 'ltx' | None — a best-effort, ADVISORY classification of the
    loaded pipeline's video family, used ONLY to pick the plugin's node behavior
    (video path vs image path). It reads the same SOURCES in the same ORDER as the
    engine's detectPipelineKind() (PipelineLoader.cpp): model_dir/model_index.json
    `_class_name` first, then transformer/config.json `_class_name` (only when the
    former is empty).

    The MATCH here is a deliberate SUPERSET of the engine's registrars, NOT a strict
    mirror: it treats any `_class_name` starting with "Wan"/"LTX" (plus the
    transformer classes WanTransformer3DModel / LTX2VideoTransformer3DModel) as that
    family. wan_detect (WanVideoPipeline.cpp, widened 2026-07-02) now ALSO prefix-
    matches any `Wan…` pipeline_class OR transformer_class=="WanTransformer3DModel",
    so a Wan I2V-A14B dir whose _class_name is "WanImageToVideoPipeline" IS both
    classified "wan" here AND loaded by the engine; ltx2_pipeline_detect
    (LTX2VideoPipeline.cpp) still exact-matches pipeline_class=="LTX2Pipeline" only
    (no transformer-class fallback), so this superset is still broader on the LTX
    side. A match HERE is not a hard guarantee the engine can load the model, but for
    the Wan family it now aligns; anything Wan/LTX-shaped is routed to the video path
    so a genuinely unloadable dir surfaces a clear LOUD video/load error (the engine
    throws at create time, and generate_video throws on a
    non-video pipeline) rather than the wrong image-node behavior — never a silent
    misroute. Returns None for every image pipeline (QwenImage/ZImage/Klein/
    Ideogram/Flux), keeping the image node's video branch guarded (byte-unchanged)."""
    model_dir = str((cfg or {}).get("model_dir", "") or "")
    pipeline_cls = ""
    transformer_cls = ""
    if model_dir:
        pipeline_cls = _read_class_name(os.path.join(model_dir, "model_index.json"))
        if not pipeline_cls:
            transformer_cls = _read_class_name(
                os.path.join(model_dir, "transformer", "config.json"))
    # `_arch` hint the loader may have recorded is a cheap secondary signal.
    arch = str((cfg or {}).get("_arch", "") or "")

    def _is_wan(s):
        return s.startswith("Wan")
    def _is_ltx(s):
        return s.startswith("LTX") or s.startswith("Ltx")

    if _is_wan(pipeline_cls) or transformer_cls == "WanTransformer3DModel" or _is_wan(arch):
        return "wan"
    if (_is_ltx(pipeline_cls) or transformer_cls == "LTX2VideoTransformer3DModel"
            or _is_ltx(arch)):
        return "ltx"
    return None


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
            # signal.SIGKILL is absent on Windows (raises AttributeError, which
            # would escape an `except OSError`); fall back to SIGTERM, which on
            # Windows os.kill maps to TerminateProcess anyway. Mirrors the
            # broad `except Exception` guard used by _kill_worker below.
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            logging.warning("[QuantFunc] SIGKILL'd stale worker pid %d", pid)
        except Exception:
            pass


def _make_progress_preview_cbs(pbar, latent_preview):
    """Return (on_progress, on_preview) callbacks for a worker generation.

    When latent preview is active, BOTH paths use ProgressBar.update_absolute
    (idempotent on the step counter, so a preview's update doesn't double-count
    the progress that the "progress" message already advanced). When inactive,
    the legacy update(1) increment is preserved unchanged. on_preview pushes the
    PIL image through ComfyUI's live-preview channel via update_absolute's third
    arg, exactly like ComfyUI's own samplers (latent_preview.py)."""
    preview_active = bool(latent_preview)
    max_size = 256
    if preview_active:
        try:
            max_size = int(latent_preview.get("max_size", 256))
        except Exception:
            max_size = 256

    def on_progress(step, total):
        if pbar is None:
            return
        if preview_active:
            try:
                pbar.update_absolute(int(step), int(total))
            except Exception:
                pbar.update(1)
        else:
            pbar.update(1)

    on_preview = None
    if preview_active:
        def on_preview(step, total, img):   # noqa: F811
            if pbar is None:
                return
            try:
                pbar.update_absolute(int(step), int(total), ("JPEG", img, max_size))
            except Exception:
                pass

    return on_progress, on_preview


def _find_downstream_latent_preview(workflow_prompt, my_node_id):
    """Return the node id of a 'QuantFunc Latent Preview' viewer wired to THIS
    generate node's `latent_preview` output, or None.

    The viewer carries no config — its mere presence enables the live preview,
    which is routed to its node id so the per-step latent2rgb frames render on
    it instead of on the generate node.
    """
    if not isinstance(workflow_prompt, dict) or my_node_id is None:
        return None
    me = str(my_node_id)
    for nid, node in workflow_prompt.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "QuantFuncLatentPreview":
            continue
        # A connected input is [source_node_id, source_output_slot]; match any
        # input fed by THIS node (only our latent_preview output reaches it).
        for v in node.get("inputs", {}).values():
            if isinstance(v, list) and len(v) == 2 and str(v[0]) == me:
                return str(nid)
    return None


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
        # cache_key -> unload_mode policy. Read by the ComfyUI free_memory hook
        # to decide how to respond to VRAM pressure from other plugins. Only
        # two values are ever stored (see QuantFuncGenerate, activate_unload
        # bool -> string): "none" = refuse all release (stay pinned in RAM);
        # "gpu+cpu" (default) = offload GPU on pressure, disk-page the RAM
        # backup, destroy only as a terminal fallback.
        self._unload_modes = {}            # cache_key -> "none" | "gpu+cpu"
        self._req_counter = 0
        self._lock = threading.Lock()
        # LRU bookkeeping for graded eviction: cache_key -> monotonically
        # increasing use sequence (higher = more recently used). Updated on
        # every ensure_pipeline call (load or cache-hit). Oldest evicted first.
        self._last_used = {}
        self._use_seq = 0
        # Upper bound on simultaneously-resident pipelines. A blanket
        # free_memory no longer destroys pipelines (it offloads + disk-pages,
        # fully reversible), so unbounded growth across many distinct models is
        # bounded HERE instead: a newly-loaded pipeline that pushes the count
        # over the cap evicts the least-recently-used UNREFERENCED pipeline.
        # <=0 means unbounded. Env override QUANTFUNC_MAX_RESIDENT_PIPELINES.
        try:
            self._max_resident = int(os.environ.get(
                "QUANTFUNC_MAX_RESIDENT_PIPELINES", "3"))
        except (ValueError, TypeError):
            self._max_resident = 3

    # ── Graded eviction: measured, reversible-first (offload → disk-page →
    #    destroy). See respond_to_free_memory for the policy. ──

    def _touch_lru(self, cache_key):
        """Mark cache_key most-recently-used. Plain dict/int ops under the GIL;
        callers already hold self._lock where ordering matters."""
        self._use_seq += 1
        self._last_used[cache_key] = self._use_seq

    def _lru_order(self, keys):
        """Return `keys` sorted oldest-used first (eviction order). Keys never
        touched sort oldest."""
        return sorted(keys, key=lambda k: self._last_used.get(k, -1))

    @staticmethod
    def _free_vram_bytes(device):
        """Free VRAM on `device` in bytes, or None if torch/CUDA unavailable.
        Driver-level free is global per device across processes, so it reflects
        the worker subprocess's frees too."""
        try:
            import torch
            free_b, _total = torch.cuda.mem_get_info(device)
            return int(free_b)
        except Exception:
            return None

    @staticmethod
    def _host_ram_available_bytes():
        """Available host RAM in bytes (Linux), tightened against a cgroup
        memory cap (v2 memory.max/.current, v1 limit/usage) when present —
        mirrors the engine's getAvailableMemoryBytes(). None if unknown."""
        avail = None
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) * 1024  # kB -> bytes
                        break
        except Exception:
            avail = None
        for max_path, cur_path in (
                ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory.current"),
                ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                 "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
            try:
                with open(max_path) as f:
                    raw = f.read().strip()
                if raw in ("max", ""):
                    continue
                limit = int(raw)
                if limit <= 0 or limit >= (1 << 62):
                    continue
                with open(cur_path) as f:
                    used = int(f.read().strip())
                cg_avail = max(0, limit - used)
                avail = cg_avail if avail is None else min(avail, cg_avail)
            except Exception:
                continue
        return avail

    def respond_to_free_memory(self, memory_required, device, blanket):
        """Graded, MEASURED response to a GENUINE external free_memory request.

        Two independent resource axes, cheapest-to-restore first:
            rung 1  offload GPU -> CPU      (ms..s reload)  — the VRAM remedy
            rung 2  release_backup -> disk  (~2-5s reload)  — the HOST-RAM remedy
            rung 3  destroy                 (terminal; ~15-25s rebuild)
        VRAM axis: a free_memory request is a VRAM request; rung 1 offload frees
        the GPU and is the COMPLETE remedy (a finite request stops the moment
        measured free VRAM satisfies it; a blanket offloads all). VRAM pressure
        NEVER triggers destroy — destroying an already-offloaded pipeline frees
        zero VRAM, so it would be pure friendly-fire that rebuilds coexisting
        pipelines every run.
        HOST-RAM axis: rung 2 disk-pages backups when RAM is low (or on a
        blanket "free everything"). rung 3 destroy is the LAST resort, gated
        SOLELY on genuine host-RAM exhaustion (still < _HOST_RAM_CRITICAL_BYTES
        after disk-paging) — independent of whether the call was finite or
        blanket. On a roomy box it never fires.
        unload_mode "none" pipelines refuse all release (skipped entirely)."""
        with self._lock:
            if not self._loaded_keys:
                return
            order = self._lru_order(list(self._loaded_keys))
            modes = dict(self._unload_modes)

        def _eligible(k):
            return modes.get(k, "gpu+cpu") != "none"

        def _satisfied():
            if blanket:
                return False
            free_b = self._free_vram_bytes(device)
            return free_b is not None and free_b >= memory_required

        # rung 1 — offload GPU -> CPU, LRU first. This is the COMPLETE remedy for
        # a VRAM request: it returns the resident pipelines' GPU memory and is
        # fully reversible (fast reload). Stop early once a finite VRAM request
        # is measured-satisfied (a blanket request offloads everything).
        for k in order:
            if not _eligible(k):
                continue
            if _satisfied():
                break
            self.unload_pipeline(k, sync=True)

        # rung 2 — disk-page the CPU backups (madvise) to reclaim PHYSICAL HOST
        # RAM. This is the RAM-pressure remedy, NOT a VRAM one (it frees no
        # VRAM). Fire on a blanket "free everything" request, or when host RAM
        # is running low. release_backup is a no-op unless the pipeline was
        # built with activate_unload (gpu+cpu).
        ram_avail = self._host_ram_available_bytes()
        ram_low = ram_avail is not None and ram_avail < _HOST_RAM_LOW_BYTES
        if blanket or ram_low:
            for k in order:
                if not _eligible(k):
                    continue
                self.release_backup_pipeline(k)

        # rung 3 — DESTROY is reserved for genuine HOST-RAM EXHAUSTION only (the
        # "resources truly insufficient" case): host RAM STILL critically low
        # AFTER disk-paging. A VRAM shortfall NEVER triggers destroy — offload
        # (rung 1) already returned the GPU memory, and destroying an already-
        # offloaded pipeline frees ZERO additional VRAM while forcing a costly
        # rebuild next use. That VRAM-triggered destroy was the friendly-fire
        # ("误伤") that rebuilt coexisting pipelines every run. Destroy LRU-first
        # (most-recently-used = likely the active pipeline = destroyed last),
        # only while RAM stays critical. Iterates the lock-captured `order`
        # snapshot (destroy_pipeline guards a key already gone).
        ram_avail = self._host_ram_available_bytes()
        if ram_avail is not None and ram_avail < _HOST_RAM_CRITICAL_BYTES:
            for k in order:
                ram_now = self._host_ram_available_bytes()
                if ram_now is None or ram_now >= _HOST_RAM_CRITICAL_BYTES:
                    break
                if not _eligible(k):
                    continue
                logging.warning(
                    "[QuantFunc] host RAM critically low (%d MB) after disk-page "
                    "— destroying LRU pipeline %s to reclaim its backup (genuine "
                    "resource exhaustion, not a VRAM friendly-fire)",
                    ram_now // 1024**2, k[:8])
                self.destroy_pipeline(k)

    def _enforce_resident_cap_locked(self, just_loaded_key):
        """Destroy least-recently-used UNREFERENCED pipelines while the resident
        count exceeds self._max_resident. MUST be called holding self._lock.
        Never evicts the pipeline just loaded, one a live node still references,
        or an unload_mode="none" (refuse-release) pipeline."""
        if self._max_resident <= 0:
            return
        referenced = set(self._node_refs.values())
        while len(self._loaded_keys) > self._max_resident:
            victims = [k for k in self._lru_order(list(self._loaded_keys))
                       if k != just_loaded_key
                       and k not in referenced
                       and self._unload_modes.get(k, "gpu+cpu") != "none"]
            if not victims:
                break  # everything resident is referenced/pinned — leave it
            victim = victims[0]
            try:
                self._call({"cmd": "destroy", "req_id": self._next_req_id(),
                            "cache_key": victim}, timeout=60)
            except Exception as e:
                # Worker may still hold it — keep our state truthful (don't drop
                # the key) and stop spinning; a later load retries the evict.
                logging.warning("[QuantFunc] cap-evict destroy(%s) failed: %s "
                                "— leaving resident", victim[:8], e)
                break
            self._loaded_keys.discard(victim)
            self._api_keys.pop(victim, None)
            self._unload_modes.pop(victim, None)
            self._last_used.pop(victim, None)
            logging.info("[QuantFunc] Resident cap %d exceeded — destroyed "
                         "LRU unreferenced pipeline %s",
                         self._max_resident, victim[:8])

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

        # Allocate this worker's log path NOW (before the stderr reader
        # spawns) so we can print it to the ComfyUI console at the same
        # init point as "Starting worker:" / "Worker ready". Pipeline
        # isolation: per-pid path → each WorkerManager owns its own file.
        log_ts = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._worker_log_path = os.path.join(
            tempfile.gettempdir(),
            f"quantfunc_worker_{self._process.pid}_{log_ts}.log")
        _WorkerLogWriter.get().open(self._worker_log_path)
        logging.info("[QuantFunc] Worker log: %s", self._worker_log_path)

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
            self._unload_modes.clear()
            self._last_used.clear()
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
        """Forward worker's stderr to console + per-worker temp log.

        ComfyUI console gets only the lines we'd want a user to see
        (per-stage timings, progress, errors, OOM). The full stream goes
        to /tmp/quantfunc_worker_<pid>_<ts>.log so engine debug output is
        still available when diagnosing.
        """
        self._recent_stderr = []
        # Log path + writer-side file handle are set up by _start_worker
        # so the path can be printed to the ComfyUI console at the same
        # `[QuantFunc] Worker log: ...` moment as `Starting worker:` /
        # `Worker ready`. Defensive default: if the field is missing (e.g.
        # legacy entry into this method), still write somewhere sane.
        log_path = getattr(self, "_worker_log_path", None) or os.path.join(
            tempfile.gettempdir(),
            f"quantfunc_worker_unknown_"
            f"{_datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        writer = _WorkerLogWriter.get()
        if not getattr(self, "_worker_log_path", None):
            writer.open(log_path)
            self._worker_log_path = log_path
            logging.info("[QuantFunc] Worker log: %s", log_path)
        _ERROR_FLUSH_KW = ("error", "fatal", "exception", "oom",
                            "abort", "segfault", "terminate")
        try:
            for line in self._process.stderr:
                text = line.decode("utf-8", errors="replace").rstrip()
                if not text:
                    continue
                # Detect error keywords (cheap substring scan) — used both
                # for crash-diagnostic cache and to request immediate flush
                # on the writer thread (so a subsequent crash leaves a
                # complete log on disk).
                is_error = False
                low = text.lower()
                for k in _ERROR_FLUSH_KW:
                    if k in low:
                        is_error = True
                        break
                if is_error:
                    self._recent_stderr.append(text)
                    if len(self._recent_stderr) > 10:
                        self._recent_stderr.pop(0)
                # All disk I/O is delegated to the global writer thread;
                # this hot path only does substring/regex + a non-blocking
                # queue put. No syscalls per line → stderr pipe stays
                # drained → engine never blocks on stderr write().
                writer.write(log_path, text, flush=is_error)
                # Console echo only for key lines (full stream in log_path)
                if _WORKER_CONSOLE_KEEP.search(text):
                    logging.info("[QuantFunc-worker] %s", text)
        except Exception:
            pass
        finally:
            writer.close(log_path)

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
                    # SIGKILL is absent on Windows — without the getattr fallback
                    # this raises AttributeError that the `except` below would
                    # swallow, turning the last-resort kill into a silent no-op.
                    # SIGTERM via os.kill maps to TerminateProcess on Windows.
                    os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
                    self._process.wait(timeout=3)
                except Exception:
                    logging.error("[QuantFunc] Failed to kill worker pid %d — "
                                  "process may be stuck in CUDA driver (D state). "
                                  "GPU resources may remain occupied until reboot.", pid)
            self._process = None
            self._loaded_keys.clear()
            self._api_keys.clear()
            self._unload_modes.clear()
            self._last_used.clear()
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

    def _call(self, cmd, progress_cb=None, preview_cb=None, timeout=1800):
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

            if msg_type == "preview":
                # Per-step latent preview (engine-decoded latent2rgb). Decode the
                # base64 RGB and hand it to the preview callback. Never let a
                # preview failure interrupt the generation.
                if preview_cb:
                    try:
                        import base64
                        from PIL import Image
                        w = int(resp.get("width", 0)); h = int(resp.get("height", 0))
                        b64 = resp.get("rgb_b64")
                        if w > 0 and h > 0 and b64:
                            raw = base64.b64decode(b64)
                            img = Image.frombytes("RGB", (w, h), raw)
                            preview_cb(resp.get("step", 0), resp.get("total", 0), img)
                    except Exception as e:
                        logging.debug("[QuantFunc] latent preview decode failed: %s", e)
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
        loading/running keep_key. Must be called with self._lock held.

        Uses sync=True so the worker's quantfunc_unload_sync blocks until
        VRAM is actually released — the caller is about to load or run
        keep_key and can't race with an async offload of the others."""
        others = [k for k in self._loaded_keys if k != keep_key]
        for k in others:
            try:
                self._call({"cmd": "unload", "req_id": self._next_req_id(),
                            "cache_key": k, "sync": True}, timeout=60)
                logging.info("[QuantFunc] Evicted pipeline %s to free VRAM for %s",
                             k[:8], keep_key[:8] if keep_key else "?")
            except Exception as e:
                logging.warning("[QuantFunc] Failed to evict %s: %s", k[:8], e)

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
            opts = dict(cfg.get("options", {}))
            # Route the transformer's coalesced backup through disk-file-
            # backed mmap whenever the caller declared they might later
            # unload. The backing store is a real file (/var/tmp/
            # quantfunc_backup_XXXXXX, or $QUANTFUNC_BACKUP_DIR / $TMPDIR
            # if set), released on clean process exit. When the Generate
            # node sets activate_unload=True and ComfyUI later requests a
            # free_memory, we trigger releaseRamPages() on that backup so
            # the kernel writes dirty pages to disk and reclaims ~13 GB of
            # physical RAM — visible to other plugins. Next gen page-
            # faults pages back in from disk.
            # When activate_unload=False (default in the Generate node),
            # the backup stays in pinned RAM and no disk file is created;
            # the pipeline refuses ComfyUI's free_memory requests.
            opts.setdefault("activate_unload", False)
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
                        self._unload_modes.pop(dropped_key, None)
                        self._last_used.pop(dropped_key, None)

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
                        self._unload_modes.pop(old_key, None)
                        self._last_used.pop(old_key, None)

            if key in self._loaded_keys:
                # Pipeline already loaded — check if API key changed
                if new_api_key and new_api_key != self._api_keys.get(key, ""):
                    self._set_api_key_locked(key, new_api_key)
                self._touch_lru(key)
                return key

            if self._loaded_keys:
                logging.info("[QuantFunc] New pipeline requested, offloading %d existing to CPU...",
                             len(self._loaded_keys))
                # Free VRAM before creating new pipeline
                self._unload_others_locked(keep_key=key)

            # The C engine's tokenizer loader reads tokenizer/vocab.json +
            # merges.txt and cannot parse the fused HF tokenizer.json. Models
            # that ship only tokenizer.json (e.g. ideogram-4-fp8) would fail at
            # load with "Failed to open vocab.json"; derive the split files from
            # the model's own tokenizer.json here. Path-independent: this is the
            # single create chokepoint, so it covers raw model dirs that bypass
            # the format-adapter staging. Idempotent (no-op once split present).
            self._ensure_engine_tokenizer(cfg.get("model_dir", ""))

            # Build create command
            create_cmd = {
                "cmd": "create",
                "req_id": self._next_req_id(),
                "cache_key": key,
                "model_dir": cfg.get("model_dir", ""),
                "transformer_path": cfg.get("transformer", ""),
                "scheduler_config": cfg.get("scheduler", "") or None,
                "model_backend": cfg.get("backend", "svdq"),
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
            self._touch_lru(key)
            self._enforce_resident_cap_locked(just_loaded_key=key)
            logging.info("[QuantFunc] Pipeline ready (%d loaded).", len(self._loaded_keys))
            return key

    @staticmethod
    def _ensure_engine_tokenizer(model_dir):
        """Backfill engine-native vocab.json/merges.txt from a fused
        tokenizer.json when a model dir ships only the latter. Best-effort +
        idempotent; never raises (model load proceeds and the engine reports if
        the tokenizer is genuinely absent)."""
        if not model_dir:
            return
        try:
            from .format_adapters.tools.hf_layout import ensure_engine_tokenizer
            ensure_engine_tokenizer(os.path.join(model_dir, "tokenizer"))
        except Exception as e:
            logging.getLogger("QuantFunc").debug(
                "tokenizer split-file ensure skipped for %s: %s", model_dir, e)

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
                      guidance_scale, options_json=None, pbar=None,
                      latent_preview=None):
        """Generate text-to-image on the specified pipeline. Returns (images, masks) — see _read_image.
        `latent_preview` (dict with max_size/every_n_steps, already injected into
        options_json by the caller) enables per-step live preview."""
        with self._lock:
            self._ensure_worker()
            # Print the worker log path on EVERY generation (the worker persists
            # across generations, so the one-time startup print scrolls away).
            logging.info("[QuantFunc] Worker log: %s",
                         getattr(self, "_worker_log_path", None) or "(not set)")

            # Before running, make sure only the target pipeline is GPU-resident
            self._unload_others_locked(keep_key=cache_key)

            on_progress, on_preview = _make_progress_preview_cbs(pbar, latent_preview)

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
            if latent_preview:
                cmd["latent_preview"] = True

            resp = self._call(cmd, progress_cb=on_progress,
                              preview_cb=on_preview, timeout=600)
            return self._read_image(resp)

    def text_to_video(self, cache_key, prompt, height, width, steps, seed,
                      guidance_scale, num_frames, options_json=None, pbar=None):
        """#344 — LTX-2 text-to-video (+ audio). Returns (frames, audio) where
        frames is a [N, H, W, 3] float32 numpy array and audio is None or
        {"waveform": np[C, N], "sample_rate": int}."""
        with self._lock:
            self._ensure_worker()
            self._unload_others_locked(keep_key=cache_key)
            on_progress, _ = _make_progress_preview_cbs(pbar, None)
            # num_frames/fps ride in options_json (mirrors the C-API t2v contract).
            opts = json.loads(options_json) if options_json else {}
            opts["num_frames"] = int(num_frames)
            cmd = {
                "cmd": "text_to_video",
                "req_id": self._next_req_id(),
                "cache_key": cache_key,
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "options_json": json.dumps(opts),
            }
            resp = self._call(cmd, progress_cb=on_progress, timeout=1800)
            return self._read_video(resp)

    def _read_video(self, resp):
        """Read frames (+ optional audio) from a text_to_video response."""
        n = int(resp.get("num_frames", 0))
        w = int(resp.get("width", 0)); h = int(resp.get("height", 0))
        if n == 0 or w == 0 or h == 0:
            raise RuntimeError("No video frames in response")
        _MAX = 16384
        if w > _MAX or h > _MAX or n > 100000:
            raise RuntimeError(f"Worker reported implausible video {n}x{w}x{h}")
        nbytes = n * h * w * 3
        fpath = resp.get("frame_shm_path")
        if not fpath:
            raise RuntimeError("No frame_shm_path in video response")
        try:
            with open(fpath, "rb") as f:
                raw = f.read(nbytes)
        finally:
            try: os.unlink(fpath)
            except OSError: pass
        if len(raw) != nbytes:
            raise RuntimeError(f"Video frame bytes {len(raw)} != expected {nbytes}")
        frames = (np.frombuffer(raw, dtype=np.uint8).reshape(n, h, w, 3).astype(np.float32) / 255.0)
        # Audio (None for video-only). Worker buffer is PLANAR/channel-major [C, N].
        audio = None
        a = resp.get("audio")
        if a:
            ch = int(a["channels"]); ns = int(a["num_samples"]); sr = int(a["sample_rate"])
            ap = a.get("shm_path")
            # Sanity-bound the engine-reported sizes before allocating (mirror the
            # frame guard) — a corrupt response must not drive a huge np/file alloc.
            if ap and 0 < ch <= 64 and 0 < ns <= 100_000_000:
                want = ch * ns * 4  # float32
                try:
                    with open(ap, "rb") as f:
                        araw = f.read(want)
                finally:
                    try: os.unlink(ap)
                    except OSError: pass
                if len(araw) == want:
                    wav = np.frombuffer(araw, dtype=np.float32).reshape(ch, ns)
                    audio = {"waveform": wav, "sample_rate": sr}
                else:
                    print(f"[QuantFunc] video audio partial read "
                          f"({len(araw)}/{want} bytes) — dropping audio", file=sys.stderr)
            elif ap:
                print(f"[QuantFunc] implausible audio dims ch={ch} ns={ns} "
                      f"— dropping audio", file=sys.stderr)
        return frames, audio

    def image_to_video(self, cache_key, prompt, ref_paths, height, width, steps, seed,
                       guidance_scale, num_frames, negative_prompt="",
                       options_json=None, pbar=None):
        """#329/#330 — image-to-video via the generic quantfunc_image_to_video C-API.
        The FIRST ref path is the first-frame condition. Whether the engine actually
        honors it is the ENGINE's decision (Wan i2v-capable checkpoints do; a t2v-only
        Wan checkpoint throws; LTX i2v is not yet engine-wired → the ref is ignored,
        the node warns). Returns (frames, audio) like text_to_video — frames is a
        [N, H, W, 3] float32 numpy array and audio is None or
        {"waveform": np[C, N], "sample_rate": int}."""
        with self._lock:
            self._ensure_worker()
            self._unload_others_locked(keep_key=cache_key)
            on_progress, _ = _make_progress_preview_cbs(pbar, None)
            # num_frames/fps ride in options_json (mirrors the C-API i2v contract).
            opts = json.loads(options_json) if options_json else {}
            opts["num_frames"] = int(num_frames)
            cmd = {
                "cmd": "image_to_video",
                "req_id": self._next_req_id(),
                "cache_key": cache_key,
                "prompt": prompt,
                "ref_image_paths": ref_paths,
                "height": height,
                "width": width,
                "num_steps": steps,
                # Video CFG rides in guidance_scale; the worker maps it onto the i2i
                # struct's true_cfg_scale (the field the engine's i2v reads).
                "guidance_scale": guidance_scale,
                "negative_prompt": negative_prompt or "",
                "seed": seed,
                "options_json": json.dumps(opts),
            }
            resp = self._call(cmd, progress_cb=on_progress, timeout=1800)
            return self._read_video(resp)

    def image_to_image(self, cache_key, prompt, ref_paths, height, width, steps, seed,
                       true_cfg_scale=1.0, negative_prompt="",
                       options_json=None, pbar=None,
                       mask_path=None, mask_strength=1.0, mask_grow=6,
                       mask_blur=0.0, mask_no_snap=False, latent_preview=None):
        """Generate image-to-image on the specified pipeline. Returns (images, masks) — see _read_image.
        Optional inpaint: pass `mask_path` to a pixel-space mask PNG (white=inpaint, black=preserve).
        Mirrors ComfyUI SetLatentNoiseMask + GrowMask + MaskBlur + VAEEncodeForInpaint.grow_mask_by.
        `latent_preview` enables per-step live preview (config already in options_json).
        """
        with self._lock:
            self._ensure_worker()
            # Print the worker log path on EVERY generation (the worker persists
            # across generations, so the one-time startup print scrolls away).
            logging.info("[QuantFunc] Worker log: %s",
                         getattr(self, "_worker_log_path", None) or "(not set)")

            # Before running, make sure only the target pipeline is GPU-resident
            self._unload_others_locked(keep_key=cache_key)

            on_progress, on_preview = _make_progress_preview_cbs(pbar, latent_preview)

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
            if mask_path:
                cmd["mask_path"] = mask_path
                cmd["mask_strength"] = float(mask_strength)
                cmd["mask_grow"] = int(mask_grow)
                cmd["mask_blur"] = float(mask_blur)
                cmd["mask_no_snap"] = bool(mask_no_snap)
            if latent_preview:
                cmd["latent_preview"] = True

            resp = self._call(cmd, progress_cb=on_progress,
                              preview_cb=on_preview, timeout=600)
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
                self._unload_modes.clear()
                self._last_used.clear()

            opts = dict(cfg.get("options", {}))
            sched = cfg.get("scheduler", "")
            if sched:
                opts["scheduler_config"] = sched

            # Same tokenizer split-file backfill as the create path: export also
            # loads the model and needs vocab.json/merges.txt for the engine.
            self._ensure_engine_tokenizer(cfg.get("model_dir", ""))

            cmd = {
                "cmd": "export",
                "req_id": self._next_req_id(),
                "model_dir": cfg.get("model_dir", ""),
                "export_path": export_path,
                "transformer_path": cfg.get("transformer", ""),
                "model_backend": cfg.get("backend", "svdq"),
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

    def unload_pipeline(self, cache_key=None, sync=False):
        """Offload models from GPU to CPU, freeing VRAM. Pipelines stay alive for fast reload.
        If cache_key given, unload that one; otherwise unload all.

        sync=True → blocks until VRAM is actually freed (skips the 3-second
        grace period in the worker). Use this when another component is
        about to allocate VRAM (cross-pipeline eviction, ComfyUI's
        free_memory hook for third-party models). Default sync=False is
        fire-and-forget so the caller can return immediately."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            if not self._loaded_keys:
                return
            cmd = {"cmd": "unload", "req_id": self._next_req_id(), "sync": sync}
            if cache_key is not None:
                if cache_key not in self._loaded_keys:
                    return
                cmd["cache_key"] = cache_key
            try:
                # Sync unload can take up to ~5 s (grace skip + D2H + trim);
                # async returns almost instantly.
                self._call(cmd, timeout=60 if sync else 30)
                logging.info("[QuantFunc] Models offloaded to CPU — VRAM freed (%s%s)",
                             cache_key if cache_key else "all",
                             " sync" if sync else "")
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
                self._unload_modes.clear()
                self._last_used.clear()

    def release_backup_pipeline(self, cache_key):
        """Release physical RAM pages of the mmap-backed CPU offload_backup
        while keeping the backing files (and the pipeline handle). Next
        generate() page-faults pages in (~2-5s via OS page cache / disk)
        instead of full re-init (~15-25s). Used by unload_mode='gpu+cpu'.

        Pair with unload_pipeline: first unload GPU → CPU (mmap), then
        release_backup to free the physical RAM while keeping content on
        disk for fast reload.

        No-op if the DLL predates quantfunc_release_backup, or if the
        pipeline wasn't created with activate_unload=True (e.g. export
        mode or the user unticked the widget)."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            if cache_key not in self._loaded_keys:
                return
            try:
                self._call({"cmd": "release_backup",
                            "req_id": self._next_req_id(),
                            "cache_key": cache_key}, timeout=30)
                logging.info("[QuantFunc] Pipeline %s backup released — RAM pages returned to OS",
                             cache_key[:8] if cache_key else "None")
            except Exception as e:
                logging.warning("[QuantFunc] release_backup_pipeline(%s) failed: %s",
                                cache_key[:8] if cache_key else "None", e)

    def destroy_pipeline(self, cache_key):
        """Fully destroy a single pipeline (frees GPU + CPU/RAM).
        Next use will recreate from scratch — slower than unload_pipeline but
        reclaims the ~10 GB pinned coalesced backup held in system RAM.
        Used as a last resort / escape hatch."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            if cache_key not in self._loaded_keys:
                return
            try:
                self._call({"cmd": "destroy",
                            "req_id": self._next_req_id(),
                            "cache_key": cache_key}, timeout=60)
                self._loaded_keys.discard(cache_key)
                self._api_keys.pop(cache_key, None)
                self._unload_modes.pop(cache_key, None)
                self._last_used.pop(cache_key, None)
                logging.info("[QuantFunc] Pipeline %s destroyed — GPU + RAM released",
                             cache_key[:8] if cache_key else "None")
            except Exception as e:
                logging.warning("[QuantFunc] destroy_pipeline(%s) failed: %s",
                                cache_key[:8] if cache_key else "None", e)

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
            self._unload_modes.clear()
            self._last_used.clear()

    def _read_image(self, resp):
        """Read image data from worker response.

        Returns (images, masks):
          images: float32 ndarray [N, H, W, 3] in [0,1]  (N==1 for single-image
                  results — byte-identical to the legacy path).
          masks:  float32 ndarray [N, H, W] in [0,1] (per-image alpha), or None when
                  the result is RGB (no alpha).
        Prefers /dev/shm (zero-copy) over the stdout pipe."""
        n_bytes = resp.get("image_bytes", 0)
        w = resp.get("image_width", 0)
        h = resp.get("image_height", 0)
        if n_bytes == 0 or w == 0 or h == 0:
            raise RuntimeError("No image data in response")
        # Bound + consistency-check the worker-reported sizes BEFORE the read /
        # reshape: a desynced or compromised worker could otherwise force an
        # unbounded blocking read, or a reshape mismatch deep in numpy.
        N = int(resp.get("image_count", 1))
        C = int(resp.get("image_channels", 3))
        fmt = resp.get("image_format", "rgb_float32")
        _MAX_IMG_DIM = 16384
        if w > _MAX_IMG_DIM or h > _MAX_IMG_DIM:
            raise RuntimeError(f"Worker reported implausible image dims {w}x{h}")
        if N < 1 or N > 4096 or C not in (3, 4):
            raise RuntimeError(f"Worker reported implausible count/channels N={N} C={C}")
        if fmt == "rgb_uint8":
            expected = w * h * 3
        elif fmt == "layers_f32":
            expected = N * h * w * C * 4
        else:  # legacy single float32 RGB
            expected = w * h * 3 * 4
        if n_bytes != expected:
            raise RuntimeError(
                f"Worker image_bytes={n_bytes} != expected {expected} "
                f"(fmt={fmt} {w}x{h} N={N} C={C})")
        shm_path = resp.get("image_shm_path")
        if shm_path:
            try:
                with open(shm_path, "rb") as f:
                    raw = f.read(n_bytes)
            finally:
                try:
                    os.unlink(shm_path)
                except OSError:
                    pass
        else:
            raw = self._read_binary(n_bytes)
        if fmt == "rgb_uint8":
            # uint8 [0,255] → float32 [0,1], 4x less IPC data than float32
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
            return arr[None, ...].copy(), None          # [1,H,W,3], no alpha
        if fmt == "layers_f32":
            # All N images stacked, [N,H,W,C] float32 in [0,1]. Split RGB + alpha.
            flat = np.frombuffer(raw, dtype=np.float32).reshape(N, h, w, C)
            images = np.ascontiguousarray(flat[..., :3])               # [N,H,W,3]
            masks = np.ascontiguousarray(flat[..., 3]) if C == 4 else None  # [N,H,W]
            return images, masks
        # Legacy single float32 RGB
        arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w, 3).copy()
        return arr[None, ...], None


_manager = WorkerManager()
atexit.register(_manager.shutdown)


# ============================================================================
# Hook into ComfyUI model management — auto-unload when other nodes need VRAM
# ============================================================================

# Below this much available host RAM, proactively disk-page (madvise) idle
# pipeline backups so QuantFunc doesn't push the box toward the OOM-killer.
# Env override QUANTFUNC_HOST_RAM_LOW_BYTES.
try:
    _HOST_RAM_LOW_BYTES = int(os.environ.get(
        "QUANTFUNC_HOST_RAM_LOW_BYTES", str(4 * 1024 ** 3)))
except (ValueError, TypeError):
    _HOST_RAM_LOW_BYTES = 4 * 1024 ** 3

# Host RAM below this (after disk-paging) is genuine exhaustion: only THEN may a
# pipeline be destroyed (last resort) — never to satisfy a VRAM request. Env
# override QUANTFUNC_HOST_RAM_CRITICAL_BYTES.
try:
    _HOST_RAM_CRITICAL_BYTES = int(os.environ.get(
        "QUANTFUNC_HOST_RAM_CRITICAL_BYTES", str(1536 * 1024 ** 2)))
except (ValueError, TypeError):
    _HOST_RAM_CRITICAL_BYTES = 1536 * 1024 ** 2

# Ladder invariant: the reversible disk-page rung (LOW) must trigger BEFORE the
# destructive rung (CRITICAL). A misconfigured CRITICAL > LOW would let destroy
# fire at a RAM level where disk-paging hasn't even run yet — clamp to preserve
# reversible-first ordering.
if _HOST_RAM_CRITICAL_BYTES > _HOST_RAM_LOW_BYTES:
    logging.warning("[QuantFunc] QUANTFUNC_HOST_RAM_CRITICAL_BYTES (%d) > "
                    "_HOST_RAM_LOW_BYTES (%d); clamping CRITICAL to LOW to keep "
                    "disk-page-before-destroy ordering",
                    _HOST_RAM_CRITICAL_BYTES, _HOST_RAM_LOW_BYTES)
    _HOST_RAM_CRITICAL_BYTES = _HOST_RAM_LOW_BYTES

# Captured original (pre-patch) comfy.model_management.free_memory; set inside
# the try below, stays None if comfy is unavailable. free_comfy_native_models()
# calls THIS so the plugin's own VRAM cleanup never trips the QF hook.
_original_free_memory = None

try:
    import comfy.model_management as _mm

    _original_free_memory = _mm.free_memory

    # Threshold to distinguish real VRAM requests from ComfyUI's blanket
    # unload_all_models() which passes 1e30.  256 TB is far beyond any real
    # GPU memory request but still well below 1e30.
    _UNREALISTIC_VRAM_REQUEST = 256 * 1024 * 1024 * 1024 * 1024  # 256 TB

    def _hooked_free_memory(memory_required, device, keep_loaded=[], **kwargs):
        # A QuantFunc pipeline is torn down only as much as the MEASURED
        # resource situation demands, cheapest-to-restore rung first
        # (offload GPU->CPU -> disk-page RAM -> destroy), and ONLY for a genuine
        # EXTERNAL request. The plugin's own pre-load cleanup of ComfyUI-native
        # torch weights goes through free_comfy_native_models() (the un-hooked
        # original free_memory), so it never reaches here -- that self-inflicted
        # blanket-destroy was why two pipelines rebuilt each other every run.
        if _manager._loaded_keys and memory_required > 0:
            blanket = memory_required >= _UNREALISTIC_VRAM_REQUEST
            try:
                _manager.respond_to_free_memory(memory_required, device, blanket)
            except Exception as e:
                logging.warning(
                    "[QuantFunc] respond_to_free_memory failed: %s", e)
        return _original_free_memory(memory_required, device, keep_loaded=keep_loaded, **kwargs)

    _mm.free_memory = _hooked_free_memory
    logging.info("[QuantFunc] free_memory hook installed — graded eviction "
                 "(offload->disk-page->destroy), self-cleanup bypasses hook "
                 "[pipeline-coexist build]")
except Exception:
    pass


def free_comfy_native_models():
    """Free ONLY ComfyUI's own native torch models (dead weight before a QF
    worker load) WITHOUT tripping the QuantFunc free_memory hook.

    The plugin used to call comfy's unload_all_models() directly, but that
    routes through our patched free_memory as a "blanket" (1e30) request, which
    the hook interpreted as "destroy every QuantFunc pipeline" -- so running a
    second pipeline destroyed the first (and vice-versa), forcing a full
    rebuild every run even with activate_unload on. We instead call the
    captured ORIGINAL free_memory, which clears comfy-native torch weights only;
    sibling QuantFunc pipelines are left to ensure_pipeline's offload-and-
    coexist (and the hook acts only on GENUINE external VRAM pressure).
    """
    try:
        import comfy.model_management as mm
        # Call the captured ORIGINAL (pre-patch) free_memory so this clears
        # comfy-native torch weights WITHOUT routing through our hook (which
        # would offload QuantFunc pipelines). _original_free_memory is assigned
        # immediately after `import comfy.model_management` succeeds at module
        # load, so it is None only when comfy was un-importable — in which case
        # the `import` just above also raises and the except swallows it. The
        # `or mm.free_memory` fallback is therefore unreachable belt-and-braces.
        (_original_free_memory or mm.free_memory)(1e30, mm.get_torch_device())
        mm.soft_empty_cache()
    except Exception as e:
        logging.debug("[QuantFunc] free_comfy_native_models failed: %s", e)


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
                "text_precision": (["int4", "int8", "fp4", "fp8", "fp16", "bf16"], {"default": "int4",
                                    "tooltip": "Text encoder quantization precision (fp4 requires SM120+/Blackwell; "
                                               "bf16 = full-precision BF16 weights, used by Klein-base models)"}),
                "vision_quant": (["int8", "int4", "fp8", "fp4", "fp16"], {"default": "int8",
                                  "tooltip": "Vision encoder quantization (int8 = INT8 weights + FP16 compute, best quality/size tradeoff)"}),
                "vae_precision": (["auto", "fp16", "fp8", "int8", "bf16"], {"default": "auto",
                                  "tooltip": "VAE precision (auto picks fp8/int8 on SM120+ via cuDNN, falls back to fp16 elsewhere; "
                                             "bf16 = full-precision BF16, used by Klein-base models)"}),
                "act_quant_mode": (["absmax", "auto", "mse"], {"default": "absmax",
                                  "tooltip": "Activation quantization scale algorithm (Lighting backend, INT4 only):\n"
                                             "• absmax (default) — fast single-pass, scale = absmax/7\n"
                                             "• auto — engine picks: MSE-search when rotation>0, else absmax\n"
                                             "• mse — search ~5 candidates for min MSE (+1dB quality, ~8% slower load)\n"
                                             "FP4 / INT8 / FP8 ignore this setting (kernel uses its own scaling)."}),
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
                     vision_quant="int8", vae_precision="auto", act_quant_mode="absmax",
                     vae_tile_size=0, pinned_memory_limit=""):
        config = {
            "tiled_vae": tiled_vae,
            "attention_backend": attention_backend,
            "precision": precision,
            "text_precision": text_precision,
            "vision_quant": vision_quant,
            "vae_precision": vae_precision,
            "act_quant_mode": act_quant_mode,
        }

        if vae_tile_size > 0:
            config["vae_tile_size"] = vae_tile_size

        pinned = pinned_memory_limit if isinstance(pinned_memory_limit, str) and pinned_memory_limit else ""
        if pinned:
            config["pinned_memory_limit"] = pinned

        return (config,)


def _first_safetensors(dir_path: str) -> str:
    """First .safetensors file inside `dir_path`, alphabetical. Returns ""
    when the directory doesn't exist or has none.
    """
    if not dir_path or not os.path.isdir(dir_path):
        return ""
    try:
        for f in sorted(os.listdir(dir_path)):
            if f.endswith(".safetensors"):
                return os.path.join(dir_path, f)
    except OSError:
        pass
    return ""


def _resolve_to_safetensors(p: str) -> str:
    """Accepts a `.safetensors` file path or a directory and returns a
    concrete file path. Empty / missing → "".
    """
    if not p or not isinstance(p, str):
        return ""
    p = p.strip()
    if not p:
        return ""
    if os.path.isfile(p):
        return p
    if os.path.isdir(p):
        return _first_safetensors(p)
    return ""


def _build_model_refs(model_dir: str, transformer_path: str,
                       prequant_weights: str = "") -> tuple:
    """Common helper for ModelLoader / ModelAutoLoader: produce three
    `_QFPathStub` instances typed as comfy MODEL / CLIP / VAE so they plug
    directly into QuantFunc Build Pipeline (same socket types as ComfyUI's
    UNETLoader / CLIPLoader / VAELoader). Mirrors format_adapters' Pick*.
    """
    from .nodes_format_adapters import _QFPathStub

    prequant_weights = prequant_weights.strip() if isinstance(prequant_weights, str) else ""

    # Resolve concrete file paths inside the standard HF model_dir layout.
    # User-provided transformer_path may be either a file or a directory —
    # both are normalised to a concrete .safetensors here.
    xfm_path = _resolve_to_safetensors(transformer_path) or _first_safetensors(
        os.path.join(model_dir, "transformer"))
    te_path  = _first_safetensors(os.path.join(model_dir, "text_encoder"))
    vae_path = _first_safetensors(os.path.join(model_dir, "vae"))

    if not xfm_path:
        raise RuntimeError(
            "QuantFunc Model Loader: no transformer .safetensors found. "
            f"transformer_path={transformer_path!r}, "
            f"model_dir={model_dir!r}/transformer/ has no .safetensors. "
            "Provide an explicit transformer_path or check model_dir layout.")

    backend = _detect_model_backend(xfm_path, model_dir)
    logging.info("[QuantFunc] model_backend → %s (xfm=%s)",
                  backend, os.path.basename(xfm_path))

    model_stub = _QFPathStub(xfm_path, kind="transformer")
    clip_stub  = _QFPathStub(te_path,  kind="te")
    vae_stub   = _QFPathStub(vae_path, kind="vae")
    # Stash QuantFunc-specific hints onto the model stub so BuildPipeline
    # can recover model_dir context, backend, and prequant sidecar.
    model_stub.qf_model_dir = model_dir
    model_stub.qf_backend_hint = backend
    if prequant_weights:
        model_stub.qf_prequant_weights = prequant_weights
    return (model_stub, clip_stub, vae_stub)


class QuantFuncModelLoader:
    """Load a QuantFunc model — outputs three handles (transformer / text_encoder
    / vae), mirroring ComfyUI's Load Checkpoint shape. Wire all three into
    `QuantFunc Build Pipeline`, which carries device + advanced runtime config.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dir": ("STRING", {"default": "", "tooltip": "Base model directory (contains model_index.json)"}),
                "transformer_path": ("STRING", {"default": "", "tooltip": "Transformer weights path (safetensors file or directory)"}),
            },
            "optional": {
                "prequant_weights": ("STRING", {
                    "default": "",
                    "tooltip": "Pre-quantized modulation weights safetensors path (Lighting backend only)",
                }),
            },
        }

    # Output comfy native MODEL/CLIP/VAE types so this loader interoperates
    # with QuantFunc Build Pipeline (and any other node accepting these
    # sockets — e.g. comfy CheckpointLoaderSimple / UNETLoader / CLIPLoader
    # / VAELoader output the same types). The values are `_QFPathStub`
    # objects carrying qf_source_path; non-QuantFunc consumers will crash
    # on these stubs by design (per format_adapters convention).
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_model"
    CATEGORY = "QuantFunc"

    def load_model(self, model_dir, transformer_path,
                   prequant_weights="", **kwargs):
        return _build_model_refs(model_dir, transformer_path, prequant_weights)


# ============================================================================
# Node: QuantFunc Wan Combine Experts (A14B two-transformer combine-picker)
# ============================================================================

def _wan_combine_stage_root() -> str:
    """Root for staged two-expert Wan dirs. Honors $QUANTFUNC_CACHE_DIR (the same
    override the rest of the adapter layer respects), else ComfyUI temp, else the
    system temp dir — NEVER the plugin tree (forbidden staging ground; the ~56 GB
    output would bloat/pollute the install). The dominant-size outputs are freshly
    REWRITTEN (dequant), not hardlinked, so a same-volume-as-source root is not
    required. NOTE: ComfyUI wipes its temp dir on every restart — set
    QUANTFUNC_CACHE_DIR (or the node's output_dir) to PERSIST the ~56 GB stage
    across restarts instead of re-dequantizing each time."""
    env = os.environ.get("QUANTFUNC_CACHE_DIR")
    if env:
        return os.path.join(env, "qf_wan_combined")
    try:
        import folder_paths
        return os.path.join(folder_paths.get_temp_directory(), "qf_wan_combined")
    except Exception:
        import tempfile
        return os.path.join(tempfile.gettempdir(), "qf_wan_combined")


class QuantFuncWanCombineExperts:
    """Combine two single-file Wan2.2-A14B experts (high-noise + low-noise) into a
    single two-expert model_dir the QuantFunc engine loads.

    Wan2.2-A14B is a DUAL-transformer model: a high-noise expert drives the early
    (high-noise) denoise steps and a low-noise expert the later steps, switching at
    `boundary_ratio`. It is distributed as TWO separate single-file checkpoints,
    not a diffusers dir. This node stages them into the engine's expected layout
    (`transformer/` = high, `transformer_2/` = low, shared `text_encoder`/`vae`/
    `tokenizer`/`scheduler` from a Wan diffusers dir, and a synthesized
    `model_index.json` carrying `boundary_ratio`), then outputs the staged
    `model_dir` STRING — wire it into `QuantFunc Model Loader`.

    The expert modality is auto-detected from the weights (t2v: in==out channels;
    A14B channel-concat i2v: in>out) and written into the synthesized transformer
    configs. The model_index always carries `_class_name="WanPipeline"` — a value
    the engine's family detect accepts (it accepts any `Wan…`-prefixed class) —
    because the engine dispatches t2v-vs-i2v by the transformer's CHANNELS, not the
    pipeline class string. (t2v is the path exercised end-to-end here; i2v staging
    emits the same loadable class + i2v channels, relying on the engine's
    channel-driven i2v dispatch.)

    ComfyUI/original-Wan single-file keys are remapped to diffusers keys; fp8
    `*_scaled` experts are dequantized to fp16 (the engine's Wan transformer
    factory does not yet consume fp8 inline); the Wan2.1 16-ch VAE config is
    absent-key-fixed. `boundary_ratio` defaults to AUTO (inherit the shared
    model_index's published value when its modality matches the experts —
    the boundary is modality-specific while the shared components are not —
    else t2v 0.875 / i2v 0.9, the official Wan-AI release values). Staging is cached (idempotent on identical sources),
    atomic (tmp+rename — a concurrent/aborted run can never leave a partial dir
    that looks complete), and pre-checks free disk against the exact output size.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_noise_expert": ("STRING", {"default": "", "tooltip":
                    "Path to the HIGH-noise A14B expert (.safetensors single file)"}),
                "low_noise_expert": ("STRING", {"default": "", "tooltip":
                    "Path to the LOW-noise A14B expert (.safetensors single file)"}),
                "shared_components": ("STRING", {"default": "", "tooltip":
                    "Path to a Wan diffusers dir supplying the shared vae / "
                    "text_encoder / tokenizer / scheduler (+ transformer config base)"}),
                "boundary_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,
                    "step": 0.005, "tooltip":
                    "Denoise fraction of the high→low expert switch. 0 = AUTO: "
                    "inherit the shared model_index's published value when its "
                    "modality matches the experts (else the experts' published "
                    "default: t2v 0.875 / i2v 0.9 — verified from the official "
                    "Wan-AI A14B releases). Set >0 only to override explicitly."}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "", "tooltip":
                    "Staging output dir (empty = $QUANTFUNC_CACHE_DIR, else ComfyUI "
                    "temp — which ComfyUI WIPES on every restart, forcing a "
                    "re-dequant). Point at a roomy PERSISTENT disk — a staged A14B "
                    "is ~56 GB."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_dir",)
    FUNCTION = "combine"
    CATEGORY = "QuantFunc"

    def combine(self, high_noise_expert, low_noise_expert, shared_components,
                boundary_ratio, output_dir="", **kwargs):
        from .format_adapters.comfyui_wan_remap import stage_two_expert, _fingerprint

        high = (high_noise_expert or "").strip()
        low = (low_noise_expert or "").strip()
        shared = (shared_components or "").strip()
        if not high or not low or not shared:
            raise RuntimeError(
                "QuantFunc Wan Combine Experts: high_noise_expert, "
                "low_noise_expert and shared_components are all required.")

        # Widget sentinel: 0 = AUTO (inherit the shared model_index's published
        # boundary_ratio, else the modality's published default) — the module
        # resolves it; an explicit >0 value is an override. 0 never reaches the
        # engine (whose boundary_ratio>0 gate would silently drop the low expert).
        boundary = float(boundary_ratio) if boundary_ratio and boundary_ratio > 0 \
            else None

        out = (output_dir or "").strip()
        if not out:
            fp = _fingerprint([os.path.abspath(high), os.path.abspath(low),
                               os.path.abspath(shared)],
                              extra="{}".format(boundary))
            out = os.path.join(_wan_combine_stage_root(), fp)

        # The engine's Wan transformer factory does not consume fp8 inline, so
        # staging always dequantizes to fp16 (the only engine-loadable form today).
        model_dir = stage_two_expert(high, low, shared, out,
                                     boundary_ratio=boundary)
        logging.info("[QuantFunc] Wan two-expert staged → %s", model_dir)
        return (model_dir,)


# ============================================================================
# Node: QuantFunc Wan Combine Experts (Auto) — dropdown of scanned A14B sets
# ============================================================================

_WAN_A14B_NO_EXPERTS = "[no Wan A14B experts found]"
_WAN_A14B_NO_SHARED = "[no Wan diffusers dir found]"
_wan_a14b_choice_cache = None  # (experts_map, shared_map) | None (unscanned)


def _wan_a14b_scan_roots():
    """Model dirs to scan for Wan A14B weights: the standard ComfyUI model folders
    the experts/diffusers dirs live in (resolved via folder_paths so
    extra_model_paths.yaml is honored) + the QuantFunc model_cache
    ($QUANTFUNC_CACHE_DIR, else ~/model_cache). Deduped, existing dirs only."""
    roots = []
    try:
        import folder_paths
        for key in ("diffusion_models", "unet", "checkpoints", "diffusers"):
            try:
                roots.extend(folder_paths.get_folder_paths(key))
            except Exception:  # noqa: BLE001 — a key not registered in this install
                pass
    except Exception:  # noqa: BLE001 — folder_paths unavailable (non-ComfyUI test)
        pass
    cache_env = os.environ.get("QUANTFUNC_CACHE_DIR") or \
        os.path.join(os.path.expanduser("~"), "model_cache")
    roots.append(cache_env)
    seen, out = set(), []
    for r in roots:
        if not r:
            continue
        real = os.path.realpath(r)
        if real in seen or not os.path.isdir(real):
            continue
        seen.add(real)
        out.append(r)
    return out


def _get_wan_a14b_choices(force_rescan=False):
    """Memoized (experts_map, shared_map) — each {display_label: abs_path} — of the
    scanned Wan A14B expert files + Wan diffusers dirs. Pure filesystem scan
    (delegates to comfyui_wan_remap.list_wan_a14b_choices)."""
    global _wan_a14b_choice_cache
    if _wan_a14b_choice_cache is not None and not force_rescan:
        return _wan_a14b_choice_cache
    experts, shared = {}, {}
    try:
        from .format_adapters.comfyui_wan_remap import list_wan_a14b_choices
        experts, shared = list_wan_a14b_choices(_wan_a14b_scan_roots())
    except Exception as e:  # noqa: BLE001 — never let a scan error break node load
        logging.warning("[QuantFunc] Wan A14B scan failed: %s", e)
    _wan_a14b_choice_cache = (experts, shared)
    return _wan_a14b_choice_cache


def refresh_wan_a14b_choices():
    """Drop the memo so the next INPUT_TYPES re-scans (dropdown refresh)."""
    global _wan_a14b_choice_cache
    _wan_a14b_choice_cache = None


def _wan_a14b_expert_dropdown():
    labels = list(_get_wan_a14b_choices()[0].keys())
    return labels if labels else [_WAN_A14B_NO_EXPERTS]


def _wan_a14b_shared_dropdown():
    labels = list(_get_wan_a14b_choices()[1].keys())
    return labels if labels else [_WAN_A14B_NO_SHARED]


class QuantFuncWanCombineExpertsAuto:
    """Load a Wan2.2-A14B two-expert model by picking the weights from dropdowns —
    outputs MODEL / CLIP / VAE straight into `QuantFunc Build Pipeline`.

    Pick the HIGH-noise expert, the LOW-noise expert, and a shared Wan diffusers
    dir (vae / text_encoder / tokenizer / scheduler) from dropdowns of your scanned
    model files — no path typing. The two experts are combined into the engine's
    two-transformer layout INTERNALLY (cached), and the node hands the three
    standard MODEL / CLIP / VAE handles to Build Pipeline exactly like a native
    loader — you never see or manage a staging dir.

    Dropdowns: the expert lists show every Wan single-file transformer with a
    high/low filename hint (header-confirmed; the single TI2V-5B is excluded); the
    shared list shows Wan diffusers dirs with A14B (dual-transformer) dirs first so
    a 5B dir's different VAE can't shadow the 14B shared components. boundary_ratio
    defaults to AUTO (t2v 0.875 / i2v 0.9). The manual `QuantFunc Wan Combine
    Experts` node (hand-typed paths) is unchanged and remains for paths outside the
    scanned roots.

    Verification note (验证契约): the node → staged-dir → MODEL/CLIP/VAE → Build
    Pipeline handoff is verified (the whole staged two-expert dir, incl.
    transformer_2/ + boundary, reaches the engine). The end-to-end two-expert
    Wan VIDEO GENERATION on real GPU is the user's local ComfyUI run — not yet
    machine-verified here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        exp = _wan_a14b_expert_dropdown()
        sh = _wan_a14b_shared_dropdown()
        return {
            "required": {
                "high_noise_expert": (exp, {"tooltip":
                    "The HIGH-noise A14B expert (drives the early/high-noise denoise "
                    "steps). Pick the *high*-noise checkpoint here."}),
                "low_noise_expert": (exp, {"tooltip":
                    "The LOW-noise A14B expert (drives the later/low-noise steps). "
                    "Pick the *low*-noise checkpoint here."}),
                "shared_components": (sh, {"tooltip":
                    "A Wan diffusers dir supplying the shared vae / text_encoder / "
                    "tokenizer / scheduler. A14B-family dirs are listed first (a 5B "
                    "dir's VAE differs — prefer the 14B one)."}),
            },
            "optional": {
                "boundary_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,
                    "step": 0.005, "tooltip":
                    "High→low expert switch fraction. 0 = AUTO (t2v 0.875 / i2v 0.9, "
                    "or the shared model_index's published value when the modality "
                    "matches). Set >0 only to override."}),
            },
        }

    # Same MODEL / CLIP / VAE socket shape as the native loaders + the QuantFunc
    # Pick* nodes → plugs straight into QuantFunc Build Pipeline.
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    CATEGORY = "QuantFunc"

    @classmethod
    def VALIDATE_INPUTS(cls, high_noise_expert=None, low_noise_expert=None,
                        shared_components=None):
        # Dropdown lists are scanned lazily and can differ from a saved workflow's
        # values (files added/removed); resolve at run time with a clear error.
        return True

    def load(self, high_noise_expert, low_noise_expert, shared_components,
             boundary_ratio=0.0, **kwargs):
        from .format_adapters.comfyui_wan_remap import stage_two_expert, _fingerprint

        # Re-scan at run time so a file added since the graph opened still resolves.
        experts, shared = _get_wan_a14b_choices(force_rescan=True)
        high = experts.get(high_noise_expert)
        low = experts.get(low_noise_expert)
        sh = shared.get(shared_components)
        missing = []
        if not high:
            missing.append(f"high_noise_expert={high_noise_expert!r}")
        if not low:
            missing.append(f"low_noise_expert={low_noise_expert!r}")
        if not sh:
            missing.append(f"shared_components={shared_components!r}")
        if missing:
            raise RuntimeError(
                "QuantFunc Wan Combine Experts (Auto): could not resolve "
                + ", ".join(missing) + ". Put the A14B experts (high+low single "
                "files) + a Wan diffusers dir under the ComfyUI model dirs "
                "(diffusion_models/unet/checkpoints/diffusers) or "
                "$QUANTFUNC_CACHE_DIR, then reopen the graph to rescan.")

        # 0 = AUTO (the module resolves the published boundary); >0 = explicit override.
        boundary = float(boundary_ratio) if boundary_ratio and boundary_ratio > 0 \
            else None

        # INTERNAL, cached: combine the two experts into the engine's two-transformer
        # model_dir (never surfaced to the user). Keyed on the resolved sources so
        # identical picks share one staged dir. Same call the manual node makes.
        fp = _fingerprint([os.path.abspath(high), os.path.abspath(low),
                           os.path.abspath(sh)], extra="{}".format(boundary))
        staged_dir = os.path.join(_wan_combine_stage_root(), fp)
        model_dir = stage_two_expert(high, low, sh, staged_dir,
                                     boundary_ratio=boundary)
        logging.info("[QuantFunc] Wan A14B two-expert staged (internal) -> %s", model_dir)

        # Output the three standard handles → Build Pipeline. BuildPipeline's
        # hf_native adapter walks the transformer file's parent chain to the
        # model_index.json and loads the WHOLE staged dir, so transformer_2/ +
        # boundary_ratio (the two-expert架构) are preserved.
        return _build_model_refs(model_dir, "")


# ============================================================================
# Node: QuantFunc Model Auto Loader
# ============================================================================

def _get_auto_loader_dropdowns():
    """Get dropdown options from resource cache (loaded at import time)."""
    try:
        from .model_auto_loader import get_transformer_options
        return get_transformer_options()
    except Exception:
        # Fallback must still contain the INPUT_TYPES default ([auto-detect]) so
        # ComfyUI never flags the saved default as "value not in list".
        from .model_auto_loader import AUTO_DETECT
        return [AUTO_DETECT, "None"]


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


# Catalog-driven combos (transformer / prequant / precision_config /
# base_model_repo) are resolved and downloaded by the node at run time, so a
# saved-workflow value that isn't currently in the network-dependent, lazily
# refreshed dropdown must not hard-fail ComfyUI's "Value not in list" check.
# Each node below names ONLY its catalog input in VALIDATE_INPUTS, which makes
# ComfyUI skip the combo check for that one input (execution.py validate_inputs:
# `x not in validate_function_inputs`) while STILL validating the static combos
# (model_series / data_source) and any future numeric ranges. The value is
# resolved at execution time, surfacing a clear error only if truly unresolvable.


class QuantFuncModelAutoLoader:
    """Auto-download and load QuantFunc models.

    Selects the correct GPU variant (50x-below/50x-above) automatically.
    Downloads base model, transformer, prequant weights, and precision config
    from HuggingFace or ModelScope on first use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from .model_auto_loader import MODEL_SERIES_LIST, _DATA_SOURCES, AUTO_DETECT
        transformer_opts = _get_auto_loader_dropdowns()
        return {
            "required": {
                "model_series": (MODEL_SERIES_LIST, {"tooltip": "Model series to download and load"}),
                "data_source": (_DATA_SOURCES, {"default": "modelscope", "tooltip": "Download source: modelscope (China) or huggingface"}),
            },
            "optional": {
                "transformer": (transformer_opts, {"default": AUTO_DETECT, "tooltip": "Transformer weight. [auto-detect] (default) picks the highest-tier weight your DEFAULT GPU (CUDA device 0) can run; weights that need a newer GPU are hidden. None = use the base model's default transformer. Or pick an explicit Series/name. NOTE: auto-detect targets device 0 — if you route the pipeline to a WEAKER non-default GPU via Build Pipeline's device input, pick a matching lower-tier weight explicitly."}),
            },
        }

    # Output comfy native MODEL/CLIP/VAE types — same rationale as
    # QuantFuncModelLoader above: interoperates with BuildPipeline and any
    # node accepting these sockets.
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_model"
    CATEGORY = "QuantFunc"

    @classmethod
    def VALIDATE_INPUTS(cls, transformer=None):
        return True  # catalog combo resolved at run time; see note above

    def load_model(self, model_series, data_source,
                   transformer="None", **kwargs):
        from .model_auto_loader import (
            detect_gpu_variant, download_base_model,
            download_transformer, resolve_transformer_selection,
        )

        # ── GPU variant & base model ──
        gpu_variant = detect_gpu_variant(model_series)
        model_dir = download_base_model(model_series, gpu_variant, data_source)

        # ── Transformer (download if selected, otherwise use base model's) ──
        transformer_path = ""
        if transformer and transformer != "None":
            t_series, t_name = resolve_transformer_selection(transformer, model_series)
            if t_series and t_name:
                transformer_path = download_transformer(t_series, t_name, data_source)

        return _build_model_refs(model_dir, transformer_path)


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

    @classmethod
    def VALIDATE_INPUTS(cls, prequant=None):
        return True  # catalog combo resolved at run time; see note above

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

    @classmethod
    def VALIDATE_INPUTS(cls, precision_config=None):
        return True  # catalog combo resolved at run time; see note above

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
        gpu_variant = detect_gpu_variant(model_series)
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

    @classmethod
    def VALIDATE_INPUTS(cls, base_model_repo=None):
        return True  # catalog combo resolved at run time; see note above

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


def _get_controlnet_options():
    """Scan ComfyUI/models/controlnet/ for ControlNet weights (#324), mirroring
    ComfyUI's single-file convention (and the LoRA scanner).

    A ControlNet is a standalone `.safetensors` checkpoint — the engine loads a
    single file directly. We therefore list every `.safetensors` (recursive),
    PLUS any subdir that ships a `config.json` (e.g. InstantX Qwen-ControlNet-
    Union, whose config.json carries the exact injection topology) — for those
    we list the DIRECTORY so the engine reads config.json (its nested weight is
    not double-listed). Selecting either works: the engine's loader accepts a
    file OR a directory.
    """
    entries = []
    try:
        import folder_paths
        # ComfyUI-registered single-file checkpoints — honors EVERY registered
        # controlnet location (models/controlnet AND models/t2i_adapter) and a
        # custom --base-directory, exactly like ComfyUI's own ControlNetLoader.
        entries.extend(folder_paths.get_filename_list("controlnet"))
        # Plus InstantX-style model DIRECTORIES (ship config.json), which
        # get_filename_list doesn't enumerate — scan each registered base.
        for base in folder_paths.get_folder_paths("controlnet"):
            if not os.path.isdir(base):
                continue
            for name in sorted(os.listdir(base)):
                full = os.path.join(base, name)
                if os.path.isdir(full) and os.path.exists(os.path.join(full, "config.json")):
                    entries.append(name)
    except Exception:
        # Fallback: folder_paths unavailable → manual scan of the default dir.
        try:
            cn_dir = os.path.join(_get_comfyui_dir(), "models", "controlnet")
            for root, _, files in os.walk(cn_dir):
                for f in files:
                    if f.endswith(".safetensors"):
                        entries.append(os.path.relpath(
                            os.path.join(root, f), cn_dir).replace("\\", "/"))
        except Exception:
            pass
    return (["None"] + sorted(dict.fromkeys(entries))) if entries else ["None"]


def _resolve_controlnet_selection(name):
    """Resolve a ControlNet dropdown selection to an absolute path via ComfyUI's
    folder_paths (traversal-safe, base-dir + t2i_adapter aware). Handles both
    single-file entries and InstantX-style config DIRECTORIES. Returns "" if not
    found (so a tampered/`../` workflow value can't escape the registered bases)."""
    try:
        import folder_paths
        p = folder_paths.get_full_path("controlnet", name)  # validates containment
        if p and os.path.exists(p):
            return p
        for base in folder_paths.get_folder_paths("controlnet"):
            rbase = os.path.realpath(base)
            cand = os.path.realpath(os.path.join(base, name))
            if (cand == rbase or cand.startswith(rbase + os.sep)) and os.path.isdir(cand):
                return cand
    except Exception:
        pass
    rbase = os.path.realpath(os.path.join(_get_comfyui_dir(), "models", "controlnet"))
    cand = os.path.realpath(os.path.join(rbase, name))
    if (cand == rbase or cand.startswith(rbase + os.sep)) and os.path.exists(cand):
        return cand
    return ""


def _write_qfraw_image(img_tensor, staging_dir):
    """Write a ComfyUI IMAGE tensor ([B,H,W,3] or [H,W,3], float [0,1]) as a
    'QFRAW01' raw-RGB blob that the engine's ImageUtils::load_image reads
    directly, skipping cv::imread. The binary layout is identical to the
    ref-image staging loop; this helper intentionally uses a pure-numpy
    conversion (a single control image doesn't need that loop's cv2 SIMD
    fast-path). Returns the temp file path; the caller MUST unlink it. None on
    failure.
    """
    try:
        t = img_tensor
        if hasattr(t, "dim") and t.dim() == 4:
            t = t[0]
        arr_f32 = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        img_np = (np.clip(arr_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
        h, w = int(img_np.shape[0]), int(img_np.shape[1])
        fd, path = tempfile.mkstemp(suffix=".qfraw", dir=staging_dir)
        os.close(fd)
        header = b"QFRAW01\x00" + h.to_bytes(4, "little") + w.to_bytes(4, "little")
        with open(path, "wb") as f:
            f.write(header)
            f.write(img_np.tobytes())
        return path
    except Exception as e:
        logging.warning("[QuantFunc] control image staging failed: %s", e)
        return None


# ControlNet family → (_class_name for the engine's backend matcher, controlnet_arch).
# Keep this in sync with `_detect_controlnet_class` (the auto-sniff path): both
# express the same family→class mapping (one keyed by the arch dropdown, the
# other by safetensors keys).
_CONTROLNET_ARCH_TO_CLASS = {
    "instantx_qwen": ("QwenImageControlNetModel", None),
    "pai-fun": ("QwenImageControlTransformer2DModel", "pai-fun"),
    "zimage-fun": (None, "zimage-fun"),
}


def _detect_controlnet_class(safetensors_path):
    """Sniff a ControlNet checkpoint's safetensors header keys to identify its
    family → (_class_name, controlnet_arch). Cheap: reads only the JSON header.
    Returns (None, None) if unknown."""
    try:
        import struct
        with open(safetensors_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            # safetensors key-headers are a few KB; cap the read so a crafted /
            # corrupt length can't trigger a multi-GB allocation (DoS).
            if n <= 0 or n > 100 * 1024 * 1024:
                return (None, None)
            hdr = json.loads(f.read(n))
        joined = "\n".join(k for k in hdr if k != "__metadata__")
        # Order matters — most-specific marker first. ZImage is keyed on its
        # distinctive two-stream `control_noise_refiner` (NOT the ambiguous
        # `control_layers`, which can also appear in PAI Fun checkpoints).
        if "controlnet_x_embedder" in joined or "controlnet_blocks" in joined:
            return ("QwenImageControlNetModel", None)               # InstantX
        if "control_noise_refiner" in joined:
            return (None, "zimage-fun")                             # ZImage Fun (two-stream)
        if "control_" in joined:
            return ("QwenImageControlTransformer2DModel", "pai-fun")  # PAI Fun-Control
    except Exception as e:
        logging.warning("[QuantFunc] ControlNet family sniff failed: %s", e)
    return (None, None)


def _resolve_controlnet_model(model_path, arch):
    """Return a ControlNet model path the engine can load.

    The engine reads `<model>/config.json` ONLY when `model` is a DIRECTORY; a
    bare single-file ControlNet has no readable config.json, so the engine leaves
    its `cn_config` JSON-null and crashes in the backend matcher (json value() on
    null), AND InstantX can't be matched without `_class_name`. So for a single
    FILE we STAGE a tiny directory: a symlink to the weight (named
    diffusion_pytorch_model.safetensors) + a synthesized config.json carrying the
    family's `_class_name`/`controlnet_arch` (the InstantX topology already
    matches the engine defaults). A sibling config.json, if present, is copied
    verbatim. Directories pass through unchanged. (Note: for ZImage Fun the
    pipeline forces its arch from the BASE transformer config, so the staged
    config.json is harmlessly unused there — the symlinked weight is what loads.)"""
    try:
        if not model_path or not os.path.isfile(model_path):
            return model_path  # dir (or missing) → engine handles it
        # Prefer a real sibling config.json; else synthesize from arch / sniff.
        sibling = os.path.join(os.path.dirname(model_path), "config.json")
        cfg_out = None
        if os.path.exists(sibling):
            try:
                with open(sibling) as _f:
                    cfg_out = json.load(_f)
            except Exception:
                cfg_out = None
        if cfg_out is None:
            if arch and arch != "auto":
                cls_name, cn_arch = _CONTROLNET_ARCH_TO_CLASS.get(arch, (None, None))
            else:
                cls_name, cn_arch = _detect_controlnet_class(model_path)
            cfg_out = {}
            if cls_name:
                cfg_out["_class_name"] = cls_name
            if cn_arch:
                cfg_out["controlnet_arch"] = cn_arch
        # Stable staging dir keyed by (abspath, mtime, arch) — idempotent reuse.
        st = os.stat(model_path)
        key = hashlib.sha256(
            "{}|{}|{}".format(os.path.abspath(model_path), int(st.st_mtime), arch).encode()
        ).hexdigest()[:16]
        stage_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "cache", "controlnet_staging")
        stage_dir = os.path.join(stage_root, key)
        os.makedirs(stage_dir, exist_ok=True)
        link = os.path.join(stage_dir, "diffusion_pytorch_model.safetensors")
        try:
            if not os.path.lexists(link):
                os.symlink(os.path.abspath(model_path), link)
        except FileExistsError:
            pass  # concurrent stage created it; same key → same target, harmless
        # Always write SOME config.json so cn_config is a non-null object even for
        # an unknown family (turns the engine's value()-on-null crash into a clean
        # "no backend matched" at worst).
        with open(os.path.join(stage_dir, "config.json"), "w") as _f:
            json.dump(cfg_out, _f, indent=2)
        logging.info("[QuantFunc] ControlNet staged %s → %s (config=%s)",
                     os.path.basename(model_path), stage_dir, cfg_out)
        return stage_dir
    except Exception as e:
        logging.warning("[QuantFunc] ControlNet staging failed (%s); using raw path", e)
        return model_path


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


_CONTROLNET_ARCHS = ["auto", "instantx_qwen", "pai-fun", "zimage-fun"]
# Engine ControlNet is implemented ONLY for these main-pipeline families
# (instantx_qwen/pai-fun → QwenImage; zimage-fun → ZImage). On any other pipeline
# (Flux2Klein / Ideogram4) the `controlnet` option is a silent no-op, so we warn.
_CONTROLNET_SUPPORTED_ARCH_PREFIXES = ("QwenImage", "ZImage")
_CONTROLNET_ARCH_TOOLTIP = (
    "ControlNet architecture. 'auto' = the engine detects it from the model's\n"
    "config / _class_name (recommended). Force one only if auto mis-detects:\n"
    "  instantx_qwen — InstantX Qwen-Image ControlNet (Union: canny/depth/pose/…)\n"
    "  pai-fun — PAI Qwen-Image Fun-Control\n"
    "  zimage-fun — Z-Image Fun-Control"
)


def _apply_controlnet_cfg(pipeline, model_path, arch):
    """Inject a single ControlNet model into the pipeline config (load-time),
    mirroring the LoRA loader's pipeline→pipeline pass-through. The engine reads
    component_opts['controlnet'] = {model, arch?} at create and AUTOMATICALLY
    uses the pipeline's own transformer as the ControlNet's base (no separate
    base_transformer needed — it's the same model you loaded in the model loader)."""
    cfg = dict(pipeline)
    cfg["options"] = dict(cfg.get("options", {}))
    if model_path:
        # Guard: warn loudly when the running pipeline can't use ControlNet so a
        # wired ControlNet doesn't silently do nothing. The engine implements it
        # only for QwenImage + ZImage; on Klein / Ideogram it is a no-op. We only
        # warn when the pipeline arch is KNOWN and unsupported (never false-warn).
        pl_arch = str(cfg.get("_arch", ""))
        if pl_arch and not any(pl_arch.startswith(p) for p in _CONTROLNET_SUPPORTED_ARCH_PREFIXES):
            logging.warning(
                "[QuantFunc] ControlNet is NOT supported for '%s' pipelines — the "
                "engine implements ControlNet only for QwenImage and ZImage, so this "
                "ControlNet will be IGNORED. (controlnet=%s)", pl_arch, model_path)
        # A single-file ControlNet has no engine-readable config.json → stage a
        # dir (symlink + synthesized config.json with _class_name) so the engine
        # can match + load it. Directories pass through unchanged.
        model_path = _resolve_controlnet_model(model_path, arch)
        cn = {"model": model_path}
        if arch and arch != "auto":
            cn["arch"] = arch
        cfg["options"]["controlnet"] = cn
    else:
        # Empty path == detach any ControlNet previously chained in.
        cfg["options"].pop("controlnet", None)
    return (cfg,)


class QuantFuncControlNetLoader:
    """Attach a ControlNet to the pipeline (#324). Flow mirrors the LoRA loader:
    wire a QUANTFUNC_PIPELINE in, give the ControlNet weights path, get a
    QUANTFUNC_PIPELINE out. The engine loads ONE ControlNet; the control IMAGE +
    type + scale are per-generate on the Generate node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "controlnet_path": ("STRING", {"default": "",
                    "tooltip": "Path to a ControlNet .safetensors checkpoint (standalone "
                               "weight, like ComfyUI), or an InstantX model directory "
                               "(ships config.json for the exact topology)."}),
                "arch": (_CONTROLNET_ARCHS, {"default": "auto", "tooltip": _CONTROLNET_ARCH_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("QUANTFUNC_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "add_controlnet"
    CATEGORY = "QuantFunc"

    def add_controlnet(self, pipeline, controlnet_path, arch="auto"):
        return _apply_controlnet_cfg(pipeline, controlnet_path, arch)


class QuantFuncControlNetAutoLoader:
    """Auto-load a ControlNet from ComfyUI/models/controlnet/ (#324). Scans that
    folder for .safetensors checkpoints (and InstantX config-dirs) and offers a
    dropdown. Flow mirrors the LoRA Auto Loader (pipeline in → pipeline out).
    """

    @classmethod
    def INPUT_TYPES(cls):
        cn_opts = _get_controlnet_options()
        return {
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "controlnet": (cn_opts, {"tooltip": "ControlNet weights from models/controlnet/"}),
                "arch": (_CONTROLNET_ARCHS, {"default": "auto", "tooltip": _CONTROLNET_ARCH_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("QUANTFUNC_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "add_controlnet"
    CATEGORY = "QuantFunc"

    def add_controlnet(self, pipeline, controlnet, arch="auto"):
        model_path = ""
        if controlnet and controlnet != "None":
            model_path = _resolve_controlnet_selection(controlnet)
            if not model_path:
                raise RuntimeError("ControlNet model not found: {}".format(controlnet))
        return _apply_controlnet_cfg(pipeline, model_path, arch)


class QuantFuncControlImage:
    """ControlNet control input (#324). This node LOADS the control map image and
    CONFIGURES how the ControlNet uses it (type / strength / guidance window),
    then bundles them into one `control_image` output. Wire that output into the
    QuantFunc Generate node's `control_image` input. The ControlNet MODEL itself
    is loaded separately via 'QuantFunc ControlNet Loader' into the pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The control map — e.g. a canny / depth / pose image "
                               "(from a preprocessor or Load Image)."}),
                "control_type": (["canny", "depth", "pose", "soft_edge"], {
                    "default": "canny",
                    "tooltip": "What kind of control map `image` is (union ControlNets accept "
                               "any of these). Must match the map you feed in."}),
                "control_scale": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "ControlNet conditioning strength (0 = off). ⚠ Too HIGH washes "
                               "out / noises the image: the engine applies the control residual "
                               "strongly, and few-step LIGHTING models amplify it further.\n"
                               "Recommended START values (raise gradually for stronger control):\n"
                               "  PAI Fun-Control  ~0.2-0.4 (1.0 → pure noise on a lighting model)\n"
                               "  InstantX / ZImage ~0.5-0.9\n"
                               "If the output is washed-out/noisy, LOWER this first."}),
            },
            "optional": {
                "control_guidance_start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Inject control only from this fraction of the schedule "
                               "onward (diffusers control_guidance_start). 0 = first step."}),
                "control_guidance_end": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Stop injecting control after this fraction (diffusers "
                               "control_guidance_end). 1 = last step. Lower (e.g. 0.7) to "
                               "cut late-step control grain on long runs."}),
            },
        }

    RETURN_TYPES = ("QUANTFUNC_CONTROL",)
    RETURN_NAMES = ("control_image",)
    FUNCTION = "build"
    CATEGORY = "QuantFunc"

    def build(self, image, control_type, control_scale,
              control_guidance_start=0.0, control_guidance_end=1.0):
        return ({
            "image": image,
            "control_type": str(control_type),
            "control_scale": float(control_scale),
            "control_guidance_start": float(control_guidance_start),
            "control_guidance_end": float(control_guidance_end),
        },)


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


class QuantFuncLayeredConfig:
    """Configure Qwen-Image-Layered decomposition. Place between the pipeline builder
    and Generate. Sets how many RGBA layers the model decomposes the input into — the
    official `layers` parameter (variable 3/4/8...; default 4). It is a PER-GENERATION
    setting (passed per-call like the official diffusers `layers`), so changing it does
    NOT rebuild the pipeline — the next generation just decomposes into the new count.
    Only affects QwenImageLayered models; harmless on others. Wire QuantFunc Generate's
    `image`/`mask` outputs into a QuantFunc Layer Viewer to inspect the resulting layers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "layers": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1,
                            "tooltip": "Number of RGBA layers to decompose into "
                                       "(Qwen-Image-Layered `layers`; official default 4, "
                                       "supports 3/4/8...). Higher = finer decomposition + slower."}),
            },
        }

    RETURN_TYPES = ("QUANTFUNC_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "configure"
    CATEGORY = "QuantFunc"

    def configure(self, pipeline, layers):
        cfg = dict(pipeline)
        # Top-level key (NOT under "options") so it sits OUTSIDE _make_cache_key →
        # changing the layer count does NOT rebuild/reload the pipeline. QuantFunc
        # Generate reads it and passes it per-generation; the engine applies it via its
        # per-call layered_layers override (matches the official per-call `layers`).
        cfg["_layered_layers"] = int(layers)
        return (cfg,)


# ============================================================================
# Node: QuantFunc Generate
# ============================================================================

def _reinhard_color_match(target_hwc, reference_hwc, strength):
    """Plugin-side Reinhard color transfer in Lab space — mirrors the engine's
    apply_reinhard_color_match (src/ImageUtils.cpp). Matches `target`'s per-channel
    Lab mean/std to `reference`, then blends by `strength`. Done in Python (no C++
    change): the engine has the same algorithm but no i2i-options wiring for it.
    target_hwc / reference_hwc: float32 H×W×C RGB in [0,1]. Returns float32 HWC."""
    if strength <= 0.0:
        return target_hwc
    try:
        import cv2
    except Exception:
        logging.warning("[QuantFunc] color_match needs opencv-python; skipping")
        return target_hwc
    tgt = np.clip(target_hwc, 0.0, 1.0).astype(np.float32)
    ref = np.clip(reference_hwc, 0.0, 1.0).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_RGB2Lab)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2Lab)
    # Per-channel mean/std via OpenCV's SIMD meanStdDev (the same call the engine
    # uses); then the affine match + blend in-place (no large temporaries). ~2x
    # faster than a per-channel numpy loop and avoids the engine's manual
    # per-pixel tensor<->cv::Mat scalar copies.
    t_mean, t_std = cv2.meanStdDev(tgt_lab)
    r_mean, r_std = cv2.meanStdDev(ref_lab)
    t_mean = t_mean.reshape(3).astype(np.float32); t_std = t_std.reshape(3).astype(np.float32)
    r_mean = r_mean.reshape(3).astype(np.float32); r_std = r_std.reshape(3).astype(np.float32)
    scale = np.where(t_std > 1e-6, r_std / t_std, 1.0).astype(np.float32)
    tgt_lab -= t_mean; tgt_lab *= scale; tgt_lab += r_mean    # out = (lab - tm)*scale + rm
    out = cv2.cvtColor(tgt_lab, cv2.COLOR_Lab2RGB)
    if strength < 1.0:                                         # out = (1-s)*tgt + s*out
        out = cv2.addWeighted(tgt, 1.0 - strength, out, strength, 0.0)
    return np.clip(out, 0.0, 1.0, out=out)


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
                "true_cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.1, "tooltip": "Classical CFG (needs a negative prompt). 1.0 = OFF (default) — correct for distilled / few-step models. Raise (e.g. 4.0) only for base / non-distilled models."}),
                "sampler_name": ([
                    # --- deterministic (names match ComfyUI KSAMPLER_NAMES) ---
                    "euler", "heun", "heunpp2", "dpmpp_2m", "lms",
                    # --- stochastic / ancestral ---
                    "dpmpp_2m_sde", "euler_ancestral", "ddim",
                    # --- 2nd-pass: new deterministic ---
                    "dpm_2", "ipndm", "ipndm_v", "res_multistep", "gradient_estimation",
                    # --- 2nd-pass: new stochastic / ancestral ---
                    "dpm_2_ancestral", "dpmpp_2s_ancestral", "dpmpp_sde",
                    "dpmpp_3m_sde", "dpmpp_2m_sde_heun",
                    # --- distilled / consistency ---
                    "lcm", "res_multistep_ancestral",
                    # --- SA-Solver (stochastic Adams predictor-corrector) ---
                    "sa_solver", "sa_solver_pece",
                ], {
                    "default": "euler",
                    "tooltip": (
                        "Sampling algorithm (#326 — 23 samplers).\n"
                        "\n"
                        "DETERMINISTIC (no eta/s_noise effect):\n"
                        "  euler — 1st order, fast\n"
                        "  heun — 2nd order, 2x slower\n"
                        "  heunpp2 — Heun++ (up to 3rd order)\n"
                        "  dpmpp_2m — 2nd-order multistep (recommended for quality)\n"
                        "  dpm_2 — DPM-Solver 2nd order (2 NFE/step)\n"
                        "  lms — linear multistep (use sampler_solver_order 1-4)\n"
                        "  ipndm / ipndm_v — Adams implicit multistep\n"
                        "  res_multistep — RES 2nd-order multistep\n"
                        "  gradient_estimation — Euler + gradient momentum\n"
                        "  ddim — DDIM (eta=0: deterministic)\n"
                        "\n"
                        "STOCHASTIC / ANCESTRAL (use sampler_eta + sampler_s_noise):\n"
                        "  euler_ancestral — Euler ancestral\n"
                        "  dpm_2_ancestral — DPM-2 ancestral\n"
                        "  dpmpp_2s_ancestral — DPM++(2S) ancestral\n"
                        "  dpmpp_sde — DPM++(SDE) stochastic\n"
                        "  dpmpp_2m_sde — DPM++(2M) SDE\n"
                        "  dpmpp_2m_sde_heun — dpmpp_2m_sde + Heun corrector\n"
                        "  dpmpp_3m_sde — DPM++(3M) SDE\n"
                        "  res_multistep_ancestral — RES 2nd-order + noise\n"
                        "  lcm — LCM (distilled / lightning models)\n"
                        "\n"
                        "SA-SOLVER (use sampler_eta, sampler_s_noise,\n"
                        "           sampler_predictor_order, sampler_corrector_order):\n"
                        "  sa_solver — stochastic Adams predictor\n"
                        "  sa_solver_pece — sa_solver + corrector eval\n"
                    ),
                }),
                "sampler_eta": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Noise scale (eta) for stochastic samplers.\n"
                        "0 = deterministic (no effect on deterministic samplers).\n"
                        "Applies to: euler_ancestral, dpm_2_ancestral, dpmpp_2s_ancestral,\n"
                        "  dpmpp_sde, dpmpp_2m_sde*, dpmpp_3m_sde, ddim, sa_solver*.\n"
                        "Recommended: 0.3~0.7 for 20+ steps; higher eta needs more steps."
                    ),
                }),
                "sampler_s_noise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": (
                        "SDE noise multiplier (s_noise). Default 1.0 = ComfyUI default.\n"
                        "Only used by SDE / ancestral samplers (same list as eta).\n"
                        "0 = no noise injection. >1 = more stochastic (grainier).\n"
                        "Mirrors ComfyUI SamplerDPMPP_SDE / SamplerDPMPP_2M_SDE s_noise."
                    ),
                }),
                "sampler_solver_order": ("INT", {
                    "default": 4, "min": 1, "max": 4, "step": 1,
                    "tooltip": (
                        "Multistep order for lms (Adams-Bashforth order 1-4).\n"
                        "Only used by: lms (caps the linear-multistep history).\n"
                        "Default 4 = ComfyUI default. Lower order = faster, less accurate."
                    ),
                }),
                "sampler_predictor_order": ("INT", {
                    "default": 3, "min": 1, "max": 4, "step": 1,
                    "tooltip": (
                        "SA-Solver predictor Adams order (1-4). Default 3 = ComfyUI default.\n"
                        "Only used by: sa_solver, sa_solver_pece.\n"
                        "Higher order = more accurate but needs more history steps."
                    ),
                }),
                "sampler_corrector_order": ("INT", {
                    "default": 4, "min": 1, "max": 4, "step": 1,
                    "tooltip": (
                        "SA-Solver corrector Adams order (1-4). Default 4 = ComfyUI default.\n"
                        "Only used by: sa_solver, sa_solver_pece.\n"
                        "Higher order = more accurate correction step."
                    ),
                }),
                "scheduler": ([
                    # ComfyUI SCHEDULER_HANDLERS — 9 types (#334). `normal` is the
                    # shipped anchor (native FlowMatchEuler flow curve, byte-identical
                    # to legacy → zero regression). The other 8 reshape the sigma CURVE
                    # from the same flow sigma(t) + the model's sigma table.
                    "normal", "karras", "exponential", "sgm_uniform", "simple",
                    "ddim_uniform", "beta", "linear_quadratic", "kl_optimal",
                ], {
                    "default": "normal",
                    "tooltip": (
                        "Noise SCHEDULE — the sigma CURVE shape (#334). Orthogonal to\n"
                        "the sampler (sampler = how to integrate each step; scheduler =\n"
                        "what the sigma curve looks like). Names match ComfyUI's\n"
                        "scheduler dropdown exactly.\n"
                        "\n"
                        "  normal — native FlowMatchEuler flow curve (default, unchanged)\n"
                        "  karras — Karras et al. 2022 (rho=7)\n"
                        "  exponential — geometric (exp-linspace) spacing in sigma\n"
                        "  sgm_uniform — ComfyUI normal(sgm=True)\n"
                        "  simple — stride the model's own sigma table\n"
                        "  ddim_uniform — uniform index stride (DDIM)\n"
                        "  beta — beta.ppf(a=b=0.6) over the sigma table\n"
                        "  linear_quadratic — ComfyUI linear-quadratic (Mochi)\n"
                        "  kl_optimal — KL-optimal schedule (sd-webui #15608)\n"
                        "\n"
                        "NOTE: distilled / few-step models on explicit-sigma schedules\n"
                        "(e.g. Klein) IGNORE the scheduler TYPE; it shapes the curve for\n"
                        "base models on the set_timesteps(mu) path (QwenImage / ZImage)."
                    ),
                }),
                "control_image": ("QUANTFUNC_CONTROL", {
                    "tooltip": "ControlNet control input (#324) from a 'QuantFunc Control "
                               "Image' node (it carries the control map + type + strength + "
                               "guidance). Requires a ControlNet loaded via 'QuantFunc "
                               "ControlNet Loader'. t2i ONLY (ignored on edit / img2img)."}),
                "fbcache": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.3, "step": 0.01,
                    "tooltip": (
                        "FBCache — First-Block Cache (TeaCache-style step skipping). "
                        "Speeds up generation by reusing the transformer's cached "
                        "block-1..N residual on steps where the model is changing "
                        "little.\n\n"
                        "Value = the ACCUMULATED relative-L1 change budget of block-0's "
                        "output. While the running sum stays BELOW this, the remaining "
                        "transformer blocks are SKIPPED for that step (cheap reuse); "
                        "once it crosses, a full step runs and the budget resets.\n\n"
                        "0.0 = OFF (default, bit-exact).  Higher = more steps skipped = "
                        "faster but progressively LOSSY (softer / less prompt-faithful). "
                        "Typical 0.03-0.12; tune per model + step count (few-step "
                        "distilled models need a smaller value).\n\n"
                        "This is the COND (positive-prompt) branch threshold. For models "
                        "that run sequential true-CFG (a cond pass on the prompt + an "
                        "uncond pass on the negative prompt — Ideogram4, QwenImage-Layered) "
                        "the UNCOND branch has its own 'fbcache_uncond' threshold below "
                        "(0 = uncond not cached). Single-cache models (QwenImage t2i, "
                        "ZImage, Wan) use only this cond value.\n\n"
                        "Create-time knob: the engine reads it PER MODEL FAMILY "
                        "(QwenImage / ZImage / Ideogram4 / Wan lighting). The active "
                        "model auto-consumes the matching key; families with no FBCache "
                        "path (e.g. Klein) ignore it. Changing this value recreates the "
                        "pipeline."
                    ),
                }),
                "fbcache_uncond": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.3, "step": 0.01,
                    "tooltip": (
                        "FBCache UNCOND threshold — the step-skip budget for the "
                        "UNCONDITIONAL (negative-prompt) CFG branch.\n\n"
                        "Only meaningful for models that run sequential true-CFG with a "
                        "SEPARATE uncond pass: Ideogram4 and QwenImage-Layered. There the "
                        "cond pass uses 'fbcache' (above) and the uncond pass uses this "
                        "value, each with its OWN cache so the two branches never pollute "
                        "each other.\n\n"
                        "0.0 = OFF (default, the uncond branch is NEVER cached — bit-exact "
                        "for it). Single-cache models (QwenImage t2i, ZImage, Wan) have no "
                        "uncond branch and IGNORE this value, so the default leaves them "
                        "unchanged.\n\n"
                        "Independent of 'fbcache': you can cache cond only (fbcache>0, "
                        "fbcache_uncond=0), both, or neither (both 0 = FBCache fully OFF, "
                        "byte-identical). The uncond velocity is the slower-varying signal, "
                        "so a slightly HIGHER uncond threshold is often safe. Changing this "
                        "recreates the pipeline."
                    ),
                }),
                "vram_budget": (
                    ["off", "10%", "20%", "30%", "40%", "50%",
                     "60%", "70%", "80%", "90%", "100%"],
                    {
                        "default": "100%",
                        "tooltip": "限制本 pipeline 可用的显存上限,按所选设备物理总显存的"
                                   "百分比计。对应引擎的 workspace VRAM budget"
                                   "(vram_budget_mb):引擎会「按这个大小的显卡来规划 + 强制」"
                                   "offload / 分块 / VAE tiling 等策略。\n\n"
                                   "100%(默认)/ off:不限制,用满整张卡。\n"
                                   "例:80% 在 24G 卡上 ≈ 19.2G;在 12G 卡上 ≈ 9.6G。\n\n"
                                   "Create-time 旋钮:改这个值会重建 pipeline。",
                    },
                ),
                "activate_unload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Let ComfyUI unload this pipeline when it needs VRAM for "
                               "coexisting plugins.\n\n"
                               "False (default): never unload. Weights stay pinned in RAM "
                               "(~17 GB) after each generate; fastest subsequent runs (0 "
                               "overhead). ComfyUI's free_memory requests are refused.\n\n"
                               "True: listen to ComfyUI's memory-pressure signals. On "
                               "pressure — another plugin needs VRAM → offload GPU and "
                               "madvise the disk-backed transformer backup to return "
                               "~13 GB of RAM to the OS. On 'Free Model and Node Cache' → "
                               "full destroy. Next run page-faults the backup from disk "
                               "(~5-15 s on SSD). Enables the disk-backed coalesced backup "
                               "at pipeline creation, so a ~13 GB file will exist in "
                               "$QUANTFUNC_BACKUP_DIR / $TMPDIR / /var/tmp / /tmp while "
                               "the pipeline is alive (unlinked on clean exit).",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "workflow_prompt": "PROMPT"},
        }

    # Output order: image · mask · latent_preview. MIGRATION: mask and
    # latent_preview swapped slots vs an earlier build (was image ·
    # latent_preview · mask) — workflows saved against the old order must
    # re-wire those two ports. The IMAGE output (slot 0) is unchanged.
    RETURN_TYPES = ("IMAGE", "MASK", "QF_LATENT_PREVIEW")
    RETURN_NAMES = ("image", "mask", "latent_preview")
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "QuantFunc"

    # Legacy sampler names the engine still accepts (the dropdown was renamed to
    # ComfyUI's canonical spelling). VALIDATE_INPUTS tolerates them so a workflow
    # saved before the rename still queues instead of failing the COMBO check.
    _LEGACY_SAMPLER_ALIASES = {"euler_a", "dpm++2m", "dpm++2m_sde"}

    @classmethod
    def VALIDATE_INPUTS(cls, sampler_name=None, **kwargs):
        if sampler_name is None:
            return True
        # Resolve the dropdown list from whichever section holds sampler_name
        # (it's "optional"), defensively — never KeyError if it's ever moved.
        it = cls.INPUT_TYPES()
        valid = set()
        for section in ("optional", "required"):
            spec = it.get(section, {}).get("sampler_name")
            if spec:
                valid = set(spec[0])
                break
        if sampler_name not in valid and sampler_name not in cls._LEGACY_SAMPLER_ALIASES:
            return "Unknown sampler_name: {}".format(sampler_name)
        return True

    @staticmethod
    def _resolve_vram_budget_mb(vram_budget, device):
        """vram_budget dropdown ("80%" / "off") → engine vram_budget_mb (MiB),
        or None (= no cap) for off / >=100% / unresolvable. `device` is the
        selected card's index — the SAME one the worker runs on, so the percent
        is taken of THAT card's physical VRAM (correct on heterogeneous /
        masked-GPU boxes). Fails open (None), never crashes a generate."""
        import torch
        vb = str(vram_budget or "off").strip().rstrip("%")
        if not (vb.isdigit() and int(vb) > 0):
            return None
        pct = int(vb)
        if pct >= 100:
            # 100% = whole card = no real cap → treat as off (don't inject); an
            # explicit budget == total still trips the engine's budget-run mode
            # (skips VRAM-probe cache writes) for no benefit.
            return None
        try:
            dev = int(device)
            if not torch.cuda.is_available():
                raise RuntimeError("torch reports no CUDA device")
            count = torch.cuda.device_count()
            if not (0 <= dev < count):
                raise IndexError(
                    "device index %d out of range (visible cuda count=%d); "
                    "check CUDA_VISIBLE_DEVICES" % (dev, count))
            props = torch.cuda.get_device_properties(dev)
            total_mib = props.total_memory // (1024 * 1024)
            budget_mib = int(total_mib * pct / 100)
            logging.info(
                "[QuantFunc] vram_budget=%d%% of cuda:%d (%s, total %d MiB) "
                "→ vram_budget_mb=%d", pct, dev, props.name, total_mib, budget_mib)
            return budget_mib
        except Exception as e:
            logging.warning(
                "[QuantFunc] vram_budget=%d%%: cannot resolve device %r VRAM "
                "(%s) — running WITHOUT the cap (unlimited).", pct, device, e)
            return None

    def _generate_video_single_frame(self, cfg, prompt, width, height, steps, seed,
                                     guidance_scale, negative_prompt, unique_id, family):
        """#329/#330 — render ONE image from a video (Wan / LTX) pipeline via a
        1-frame text-to-video. Returns the image node's 3-tuple
        (IMAGE, MASK, latent_preview) with the single frame as a [1,H,W,3] IMAGE +
        an opaque mask. We request num_frames=1; the engine may still decode more
        than one frame (a video VAE has a minimum temporal decode — Wan TI2V-5B
        returns 4), so we slice frames[:1] to guarantee exactly one image."""
        import torch
        cache_key = _manager.ensure_pipeline(cfg, node_id=unique_id)
        opts = {}
        neg = negative_prompt if (isinstance(negative_prompt, str) and negative_prompt) else ""
        if neg:
            opts["negative_prompt"] = neg
        pbar = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(steps)
        except Exception:
            pass
        logging.info("[QuantFunc] image node → video pipeline (%s): 1-frame t2v %dx%d",
                     family, width, height)
        frames, _audio = _manager.text_to_video(
            cache_key, prompt, height, width, steps, seed, float(guidance_scale),
            1, options_json=(json.dumps(opts) if opts else None), pbar=pbar)
        # frames: [1, H, W, 3] float32 [0,1] — take the single frame as the IMAGE.
        image = torch.from_numpy(frames[:1])                      # [1, H, W, 3]
        mask = torch.ones(image.shape[:3], dtype=torch.float32)   # [1, H, W] opaque
        return (image, mask, None)

    def generate(self, pipeline, prompt, width, height, steps, seed,
                 guidance_scale, ref_images=None,
                 negative_prompt="", true_cfg_scale=1.0,
                 sampler_name="euler", sampler_eta=0.0,
                 sampler_s_noise=1.0, sampler_solver_order=4,
                 sampler_predictor_order=3, sampler_corrector_order=4,
                 scheduler="normal",
                 control_image=None,
                 vram_budget="100%",
                 activate_unload=False, unload_mode=None, unload_every_time=None,
                 fbcache=0.0, fbcache_uncond=0.0,
                 unique_id=None, workflow_prompt=None):
        # Backwards-compat with older saved workflows that used the
        # unload_mode dropdown or the unload_every_time bool: both are
        # collapsed onto a single activate_unload bool — true if the user
        # asked to release under pressure (any non-"none" legacy value),
        # false otherwise.  Silent; the new widget is the source of truth.
        if unload_mode is not None:
            activate_unload = activate_unload or (unload_mode != "none")
        if unload_every_time:
            activate_unload = True
        # Internal _unload_modes bookkeeping still uses the string "gpu+cpu"
        # / "none" keys the free_memory hook understands; map the bool.
        unload_mode_internal = "gpu+cpu" if activate_unload else "none"
        import torch

        # Auto-detect edit mode from ref_images
        cfg = dict(pipeline)
        cfg["options"] = dict(cfg.get("options", {}))
        # #329/#330 — a VIDEO pipeline (Wan / LTX) loaded into the IMAGE Generate
        # node produces a SINGLE image = a 1-frame text-to-video. Detect the video
        # family and route to the video C-API (num_frames=1) instead of the image
        # t2i/i2i path — the t2i C-API throws on a video pipeline (WanVideoPipeline /
        # LTX2VideoPipeline have no image path). Guarded: None (every image pipeline)
        # → the branch is skipped and the image path below is BYTE-UNCHANGED.
        _video_family = _pipeline_video_family(cfg)
        if _video_family is not None:
            # The image node produces a 1-frame t2v for a video pipeline; a wired
            # ref_images/edit input is NOT used on this path (i2v-via-image-node is
            # out of scope — use the dedicated 'QuantFunc Generate Video' node's
            # first_frame for image-to-video). Warn LOUDLY so it isn't silently
            # dropped.
            if ref_images is not None:
                logging.warning(
                    "[QuantFunc] %s is a VIDEO pipeline loaded into the image "
                    "'QuantFunc Generate' node → producing a SINGLE frame (1-frame "
                    "text-to-video); the wired ref_images are IGNORED here. For "
                    "image-to-video, use the 'QuantFunc Generate Video' node's "
                    "`first_frame` input.", _video_family)
            return self._generate_video_single_frame(
                cfg, prompt, width, height, steps, seed, guidance_scale,
                negative_prompt, unique_id, _video_family)
        # Qwen-Image-Layered guidance — NON-intrusive: we NEVER override your widget
        # values (you control steps / true_cfg / negative_prompt / resolution). We only
        # warn when the current settings are likely to under-perform vs the official
        # reference (num_inference_steps=50, true_cfg_scale=4.0, negative_prompt=" ",
        # 640/1024 resolution bucket). Plugin-only.
        if str(cfg.get("_arch", "") or "").startswith("QwenImageLayered"):
            if steps < 20:
                logging.warning("[QuantFunc] Layered: steps=%d is low — the official ComfyUI template "
                                "uses 20 steps + CFG 2.5 + 640px (the model's original 50/4.0 is ~2x "
                                "slower); fewer than 20 may not form clean layers.", steps)
            if true_cfg_scale > 1.0 and not (isinstance(negative_prompt, str) and negative_prompt):
                logging.warning("[QuantFunc] Layered: true_cfg_scale=%.1f>1 but negative_prompt is "
                                "empty — the engine applies CFG only with a non-empty negative prompt "
                                "(official uses ' '); CFG will NOT engage as-is.", true_cfg_scale)
            _area = int(width) * int(height)
            if not (560 * 560 <= _area <= 1120 * 1120):
                logging.warning("[QuantFunc] Layered: %dx%d is outside the recommended 640/1024 "
                                "resolution buckets — try ~640x640 or ~1024x1024.", width, height)
        # Propagate activate_unload into the pipeline's comp_opts so the
        # C++ side builds the transformer's coalesced backup on disk
        # (enabling future releaseRamPages) instead of in pinned RAM.
        # Must be set at create time — the coalesced backup's storage
        # medium is baked in during warmup and can't be switched later
        # without a 13 GB re-pack.
        cfg["options"]["activate_unload"] = bool(activate_unload)
        # VRAM workspace budget (create-time): inject only when a real cap is
        # resolved → default "off" leaves cfg["options"] untouched (no spurious
        # recreate), mirroring the fbcache>0 gate below.
        _budget_mib = self._resolve_vram_budget_mb(vram_budget, cfg.get("device", 0))
        if _budget_mib is not None:
            cfg["options"]["vram_budget_mb"] = _budget_mib
        # FBCache (First-Block Cache, TeaCache-style step skipping) — CREATE-TIME
        # comp_opts the engine reads PER MODEL FAMILY. Two model-agnostic knobs,
        # mirroring the engine's #472 split: `fbcache` = the COND (positive-prompt)
        # branch threshold, `fbcache_uncond` = the UNCOND (negative-prompt) branch.
        # Sequential true-CFG families (Ideogram4, QwenImage-Layered) keep a
        # SEPARATE cache per branch; single-cache families (QwenImage t2i, ZImage,
        # Wan) only have a cond/primary cache and ignore the uncond keys.
        #
        # Each key is consumed ONLY by its own model's factory, so setting every
        # family key to the same branch value stays model-agnostic and never
        # enables FBCache on a model that isn't loaded. Map to the engine's ACTUAL
        # accepted keys (verified against the engine):
        #   COND  → generic "fbcache" (Ideogram4Pipeline + QwenImage-Layered xfm
        #           factory read this first), plus the per-family cond/single keys
        #           qwenimage_fbcache (QwenImage t2i), zimage_fbcache (ZImage),
        #           wan_fbcache_thresh (Wan), ideogram4_fbcache_cond (Ideogram4
        #           legacy cond fallback).
        #   UNCOND→ generic "fbcache_uncond" (Ideogram4Pipeline + QwenImage-Layered
        #           xfm factory), plus ideogram4_fbcache (Ideogram4 legacy uncond
        #           fallback). Single-cache families have no uncond key.
        # Setting the generic "fbcache"/"fbcache_uncond" keys is what reaches the
        # QwenImage-Layered dual-slot FBCache (it reads ONLY the generic keys).
        #
        # Each branch is gated independently (>0): both default 0.0 leaves
        # cfg["options"] byte-identical → no perturbation of the pipeline cache key
        # / no spurious recreate, and a model's existing output is unchanged.
        fbc_cond = float(fbcache or 0.0)
        fbc_uncond = float(fbcache_uncond or 0.0)
        if fbc_cond > 0.0:
            cfg["options"]["fbcache"] = fbc_cond              # generic cond (Ideogram4 + QwenImage-Layered)
            cfg["options"]["qwenimage_fbcache"] = fbc_cond    # QwenImage t2i single cache
            cfg["options"]["zimage_fbcache"] = fbc_cond       # ZImage single cache
            cfg["options"]["ideogram4_fbcache_cond"] = fbc_cond  # Ideogram4 legacy cond fallback
            cfg["options"]["wan_fbcache_thresh"] = fbc_cond   # Wan single cache
        if fbc_uncond > 0.0:
            cfg["options"]["fbcache_uncond"] = fbc_uncond     # generic uncond (Ideogram4 + QwenImage-Layered)
            cfg["options"]["ideogram4_fbcache"] = fbc_uncond  # Ideogram4 legacy uncond fallback
        # Unpack ImageList dict format
        ref_img_resize = "720"
        ref_img_resize_others = "720"
        # Inpaint defaults (overridden if mask present in ref_images dict).
        inpaint_mask = None
        inpaint_strength = 1.0
        inpaint_grow = 6
        inpaint_blur = 0.0
        inpaint_no_snap = False
        color_match = 0.0
        # #293 t2i img2img (SDEdit): set when the ImageList carries an init_img.
        img2img_mode = False
        img2img_strength = 0.6
        if ref_images is not None and isinstance(ref_images, dict):
            # New: ref_img_resize ("720" / "1024" / "origin")
            # Backwards compat: old workflows may still send keep_ref_img_size (bool)
            if "ref_img_resize" in ref_images:
                ref_img_resize = ref_images["ref_img_resize"]
            elif ref_images.get("keep_ref_img_size"):
                ref_img_resize = "1024"
            ref_img_resize_others = ref_images.get("ref_img_resize_others", "720")
            # Inpaint payload (optional): mask tensor + 4 knobs.
            inpaint_mask = ref_images.get("mask")
            inpaint_strength = ref_images.get("mask_strength", 1.0)
            inpaint_grow = ref_images.get("mask_grow", 6)
            inpaint_blur = ref_images.get("mask_blur", 0.0)
            inpaint_no_snap = ref_images.get("mask_no_snap", False)
            color_match = float(ref_images.get("color_match", 0.0))
            img2img_mode = bool(ref_images.get("img2img", False))
            img2img_strength = float(ref_images.get("img2img_strength", 0.6))
            ref_images = ref_images["images"]
        # edit_mode controls which Pipeline class the C++ engine instantiates:
        #   - QwenImage / QwenImageEdit are SEPARATE classes; QwenImageEditPipeline
        #     only supports generate_edit() and rejects generate() at runtime.
        #     So edit_mode MUST track ref_images presence for these.
        #   - Klein uses ONE pipeline class for both modes (edit_mode=true just
        #     pre-loads VAE encoder); toggling triggers recreate but t2i still
        #     works since the pipeline serves both.
        # Set conditionally — accept the recreate cost when toggling ref_images.
        # #293 t2i img2img: an init_img is NOT a vision/Kontext edit — it must
        # build the T2I pipeline (edit_mode=False) so the C-API image_to_image
        # takes the SDEdit img2img route through generate(). enable_img2img opts
        # the t2i pipeline into loading its VAE encoder (Klein already has one).
        cfg["options"]["edit_mode"] = (ref_images is not None) and not img2img_mode
        if img2img_mode:
            cfg["options"]["enable_img2img"] = True

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

        # Latent preview is now a DOWNSTREAM viewer node wired to THIS node's
        # `latent_preview` OUTPUT. Connecting one == enabling preview (it has no
        # widgets). We (a) route the engine's per-step latent2rgb frames to ITS
        # node id so the live preview renders THERE (not on this node) and
        # (b) turn on the cheap engine-side preview decode with fixed defaults.
        lp_node_id = _find_downstream_latent_preview(workflow_prompt, unique_id)
        lp_cfg = ({"every_n_steps": 1, "max_size": 256, "method": "latent2rgb"}
                  if lp_node_id is not None else None)

        # Create ComfyUI progress bar. Route progress + preview frames to the
        # connected viewer node when present; else default to this node.
        pbar = None
        try:
            from comfy.utils import ProgressBar
            try:
                pbar = (ProgressBar(steps, node_id=lp_node_id)
                        if lp_node_id else ProgressBar(steps))
            except TypeError:
                pbar = ProgressBar(steps)  # older ComfyUI: no node_id kwarg
        except Exception:
            pass

        # /dev/shm QFRAW staging files — RAM-backed (tmpfs); MUST be unlinked
        # once the engine has consumed them (see the `finally` below) or every
        # edit/inpaint generation leaks a raw RGB blob (+ mask) into RAM.
        #
        # Staging dir: /dev/shm (tmpfs/RAM) on Linux; falls back to the platform
        # temp dir on Windows / when shm is unavailable. Mirrors worker.py's
        # output-path guard — without this, mkstemp(dir="/dev/shm") resolves to
        # D:\dev\shm on Windows and raises FileNotFoundError.
        _staging_dir = "/dev/shm" if (os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)) else tempfile.gettempdir()
        tmp_paths = []
        mask_path = None
        control_path = None  # #324 ControlNet staging blob (t2i); unlinked in finally
        try:
            if ref_images is not None:
                # Write each ref image as a "QFRAW01" raw uint8 RGB blob to
                # /dev/shm. Backend ImageUtils::load_image detects the magic
                # and skips cv::imread entirely — ~80 ms saved per 1728×2304
                # ref vs the prior BMP encode + cv::imread decode round-trip.
                # Format: 8-byte magic "QFRAW01\0" + uint32 H + uint32 W +
                # H*W*3 uint8 RGB bytes.
                tmp_paths = []
                try:
                    import cv2
                    _have_cv2 = True
                except ImportError:
                    _have_cv2 = False
                from PIL import Image  # output preview + RGBA-pipeline ref staging
                # Most pipelines take an RGB reference (engine load_image channels=3)
                # and use the fast QFRAW01 raw-RGB blob. But an RGBA-input pipeline
                # (QwenImageLayered → its VAE encoder input_channels=4, so the engine
                # calls load_image with channels=4) BYPASSES the RGB-only qfraw decoder
                # and would fall to cv::imread on a raw blob → "Failed to load image".
                # For those, stage a real PNG instead: the engine's already-correct
                # cv::imread(IMREAD_UNCHANGED) path loads it at 4 channels (opaque
                # alpha for an RGB source, real alpha for an RGBA source). No engine
                # change; RGB pipelines keep the fast qfraw blob.
                _RGBA_INPUT_ARCHS = ("QwenImageLayered",)  # arches whose refImageChannels()==4
                pipeline_wants_rgba = str(cfg.get("_arch", "") or "").startswith(_RGBA_INPUT_ARCHS)
                # #411 defense-in-depth: cfg["_arch"] can still be the plain "QwenImage"
                # fallback if the per-file fingerprint missed the layered identity (flat
                # lighting export / non-shard-1 shard / model_dir-as-path). Mirror the
                # engine's OWN source of truth — model_dir/model_index.json _class_name
                # == QwenImageLayeredPipeline (the signal it uses for refImageChannels()
                # ==4) — plus a "layered" model_dir marker. A false positive only stages
                # a PNG (engine loads it at 3 OR 4 channels — harmless); a false negative
                # crashes with "Failed to load image .qfraw".
                if not pipeline_wants_rgba:
                    _md = str(cfg.get("model_dir", "") or "")
                    try:
                        _mi = os.path.join(_md, "model_index.json")
                        if os.path.isfile(_mi):
                            with open(_mi, "r", encoding="utf-8") as _mf:
                                if json.load(_mf).get("_class_name") == "QwenImageLayeredPipeline":
                                    pipeline_wants_rgba = True
                    except Exception:
                        pass
                    if not pipeline_wants_rgba and "layered" in os.path.basename(_md).lower():
                        pipeline_wants_rgba = True
                for img_tensor in ref_images:
                    for i in range(img_tensor.shape[0]):
                        suffix = ".png" if pipeline_wants_rgba else ".qfraw"
                        fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=_staging_dir)
                        os.close(fd)
                        # Track the path BEFORE the write so a mid-write failure
                        # (e.g. %TEMP% disk-full on Windows — now real disk, not
                        # tmpfs) is still unlinked by the `finally` below. The
                        # mask path (further down) already orders it this way.
                        tmp_paths.append(tmp_path)
                        t = img_tensor[i]
                        # ComfyUI IMAGE is [B, H, W, C] FP32 in [0,1]. Convert
                        # to HWC uint8 [0,255]. cv2.convertScaleAbs is the
                        # fast SIMD path; numpy fallback is fine.
                        arr_f32 = t.numpy() if t.device.type == "cpu" else t.cpu().numpy()
                        if _have_cv2:
                            img_np = cv2.convertScaleAbs(arr_f32, alpha=255.0)
                        else:
                            img_np = (arr_f32 * 255).clip(0, 255).astype(np.uint8)
                        h, w = img_np.shape[0], img_np.shape[1]
                        c = img_np.shape[2] if img_np.ndim == 3 else 1
                        if pipeline_wants_rgba:
                            # Real PNG → engine cv::imread(IMREAD_UNCHANGED) loads it
                            # at 4ch. PIL writes RGB(A) in the correct order (no BGR
                            # swap), matching ComfyUI's IMAGE convention. compress_level
                            # low — /dev/shm is RAM, so favor encode speed over size.
                            mode = "RGBA" if c == 4 else "RGB"
                            Image.fromarray(np.ascontiguousarray(img_np), mode).save(
                                tmp_path, format="PNG", compress_level=1)
                        else:
                            header = b"QFRAW01\x00" + h.to_bytes(4, "little") + w.to_bytes(4, "little")
                            with open(tmp_path, "wb") as f:
                                f.write(header)
                                # tobytes() is C-contiguous HWC uint8 — RGB order
                                # matches ComfyUI's IMAGE convention, so no swap.
                                f.write(img_np.tobytes())
                neg = negative_prompt if isinstance(negative_prompt, str) and negative_prompt else ""
                i2i_opts = {}
                # Qwen-Image-Layered layer count (from QuantFunc Layered Config, stashed
                # at cfg["_layered_layers"] OUTSIDE the cache key). Passed PER-GENERATION
                # → the engine's per-call layered_layers override applies it with NO
                # pipeline rebuild. Absent / non-layered → engine uses its default (4).
                _layered_layers = cfg.get("_layered_layers")
                if isinstance(_layered_layers, int) and _layered_layers >= 1:
                    i2i_opts["layered_layers"] = _layered_layers
                # #293 t2i img2img: strength → C-API options_json edit_strength →
                # setEditImg2ImgStrength → flow-match SDEdit init in t2i generate().
                if img2img_mode:
                    i2i_opts["edit_strength"] = img2img_strength
                if sampler_name != "euler":
                    i2i_opts["sampler"] = sampler_name
                if sampler_eta > 0.0:
                    i2i_opts["eta"] = sampler_eta
                # Sampler modifier params (#326 node param surface). Emit
                # each only when non-default so the engine log stays clean.
                if sampler_s_noise != 1.0:
                    i2i_opts["s_noise"] = sampler_s_noise
                if sampler_solver_order != 4:
                    i2i_opts["solver_order"] = sampler_solver_order
                if sampler_predictor_order != 3:
                    i2i_opts["predictor_order"] = sampler_predictor_order
                if sampler_corrector_order != 4:
                    i2i_opts["corrector_order"] = sampler_corrector_order
                # Scheduler TYPE (#334). Emit only when != "normal" so the
                # default path stays byte-identical to legacy (no key → engine
                # uses the native FlowMatchEuler flow curve = the `normal` anchor).
                if scheduler and scheduler != "normal":
                    i2i_opts["scheduler"] = scheduler
                # Per-image resize: 主图 (refs[0]) uses main_image_resize,
                # 参考图 2~10 (refs[1..]) use ref_image_resize_others.
                # Build per-image array for C++ backend.
                num_refs = len(tmp_paths)
                if num_refs > 1 and ref_img_resize != ref_img_resize_others:
                    resize_arr = [ref_img_resize] + [ref_img_resize_others] * (num_refs - 1)
                    i2i_opts["ref_img_resize"] = resize_arr
                else:
                    i2i_opts["ref_img_resize"] = ref_img_resize
                if lp_cfg:
                    i2i_opts["latent_preview"] = lp_cfg
                i2i_opts_json = json.dumps(i2i_opts) if i2i_opts else None

                # Inpaint mask: ComfyUI MASK is [B, H, W] float32 in [0,1]
                # (white=mask). Save first slice as raw uint8 L8 with
                # "QFRAWL1" magic header — backend skips PNG decode entirely.
                # Format: 8-byte magic + uint32 H + uint32 W + H*W uint8.
                mask_path = None
                if inpaint_mask is not None and inpaint_strength > 0.0:
                    fd, mask_path = tempfile.mkstemp(suffix=".qfraw", dir=_staging_dir)
                    os.close(fd)
                    m = inpaint_mask
                    # Accept [B,H,W] / [B,1,H,W] / [H,W]
                    if m.dim() == 4: m = m[0, 0]
                    elif m.dim() == 3: m = m[0]
                    m_np = (m.detach().cpu().clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
                    h, w = m_np.shape[0], m_np.shape[1]
                    header = b"QFRAWL1\x00" + h.to_bytes(4, "little") + w.to_bytes(4, "little")
                    with open(mask_path, "wb") as f:
                        f.write(header)
                        f.write(m_np.tobytes())

                images_np, masks_np = _manager.image_to_image(
                    cache_key=cache_key,
                    prompt=prompt, ref_paths=tmp_paths,
                    height=height, width=width, steps=steps, seed=seed,
                    true_cfg_scale=true_cfg_scale, negative_prompt=neg,
                    options_json=i2i_opts_json, pbar=pbar,
                    mask_path=mask_path,
                    mask_strength=inpaint_strength,
                    mask_grow=inpaint_grow,
                    mask_blur=inpaint_blur,
                    mask_no_snap=inpaint_no_snap,
                    latent_preview=lp_cfg)
                # color_match (潜在色彩匹配): plugin-side Reinhard post-decode against
                # the main reference image. Engine has the same algorithm but no
                # i2i-options wiring, so we do it here — no C++ change. Single-image
                # edit only — a multi-layer (QwenImageLayered) result has no defined
                # per-layer reference, so color_match is skipped for N>1.
                if color_match > 0.0 and ref_images and images_np.shape[0] == 1:
                    try:
                        ref0 = ref_images[0]
                        ref_np = (ref0[0] if ref0.dim() == 4 else ref0).detach().cpu().numpy()
                        images_np = images_np.copy()
                        images_np[0] = _reinhard_color_match(images_np[0], ref_np, color_match)
                    except Exception as _cm_e:
                        logging.warning("[QuantFunc] color_match skipped: %s", _cm_e)
            else:
                t2i_opts = {}
                neg = negative_prompt if isinstance(negative_prompt, str) and negative_prompt else ""
                if neg and true_cfg_scale > 1.0:
                    t2i_opts["negative_prompt"] = neg
                    t2i_opts["true_cfg_scale"] = true_cfg_scale
                t2i_opts["sampler"] = sampler_name
                if sampler_eta > 0.0:
                    t2i_opts["eta"] = sampler_eta
                # Sampler modifier params (#326 node param surface). Emit
                # each only when non-default so the engine log stays clean.
                if sampler_s_noise != 1.0:
                    t2i_opts["s_noise"] = sampler_s_noise
                if sampler_solver_order != 4:
                    t2i_opts["solver_order"] = sampler_solver_order
                if sampler_predictor_order != 3:
                    t2i_opts["predictor_order"] = sampler_predictor_order
                if sampler_corrector_order != 4:
                    t2i_opts["corrector_order"] = sampler_corrector_order
                # Scheduler TYPE (#334). Emit only when != "normal" so the
                # default path stays byte-identical to legacy (no key → engine
                # uses the native FlowMatchEuler flow curve = the `normal` anchor).
                if scheduler and scheduler != "normal":
                    t2i_opts["scheduler"] = scheduler
                # #324 ControlNet (t2i only — the engine clears control on the
                # edit/i2i path). control_image is a QUANTFUNC_CONTROL bundle
                # from the 'QuantFunc Control Image' node: {image, control_type,
                # control_scale, control_guidance_start/end}. Stage its image as
                # a QFRAW01 blob and pass path + type + scale + guidance via
                # options_json. No-op in the engine if no ControlNet was loaded.
                if isinstance(control_image, dict) and control_image.get("image") is not None:
                    control_path = _write_qfraw_image(control_image["image"], _staging_dir)
                    if control_path:
                        t2i_opts["control_image"] = control_path
                        t2i_opts["control_type"] = control_image.get("control_type", "canny")
                        t2i_opts["control_scale"] = float(control_image.get("control_scale", 1.0))
                        t2i_opts["control_guidance_start"] = float(
                            control_image.get("control_guidance_start", 0.0))
                        t2i_opts["control_guidance_end"] = float(
                            control_image.get("control_guidance_end", 1.0))
                if lp_cfg:
                    t2i_opts["latent_preview"] = lp_cfg
                opts_json = json.dumps(t2i_opts) if t2i_opts else None
                logging.info(
                    "[QuantFunc] t2i sampler=%s scheduler=%s eta=%.2f s_noise=%.2f "
                    "solver_order=%d pred_ord=%d corr_ord=%d opts=%s",
                    sampler_name, scheduler, sampler_eta, sampler_s_noise,
                    sampler_solver_order, sampler_predictor_order,
                    sampler_corrector_order, opts_json)

                images_np, masks_np = _manager.text_to_image(
                    cache_key=cache_key,
                    prompt=prompt, height=height, width=width,
                    steps=steps, seed=seed, guidance_scale=guidance_scale,
                    options_json=opts_json, pbar=pbar, latent_preview=lp_cfg)

            # Persist the mode so the ComfyUI free_memory hook can look it up
            # for this pipeline. The hook inspects _unload_modes[k]:
            #   "gpu+cpu" (activate_unload=True)  → release on memory pressure
            #   "none"    (activate_unload=False) → refuse release requests
            # No per-gen action needed — the hook is the sole trigger.
            with _manager._lock:
                _manager._unload_modes[cache_key] = unload_mode_internal

            out = torch.from_numpy(images_np)  # [N, H, W, 3] (N==1 for single-image)
            # Per-image alpha → MASK output (slot 1, the 2nd port). None (RGB
            # result) → opaque ones, so any downstream MASK consumer is safe.
            out_mask = (torch.from_numpy(masks_np) if masks_np is not None
                        else torch.ones(out.shape[:3], dtype=torch.float32))  # [N, H, W]
            # 3rd output (slot 2) = wire to the Latent Preview viewer node. It
            # only anchors the connection (so we can locate the viewer + route
            # the live per-step frames to its node id) and triggers the viewer's
            # execution; the viewer shows ONLY those live frames, so the payload
            # is a lightweight sentinel (NOT the full image tensor — that would
            # be serialized over the wire for nothing). None when no viewer wired.
            lp_out = True if lp_node_id is not None else None
            return (out, out_mask, lp_out)  # image [N,H,W,3], mask [N,H,W], latent_preview

        except InterruptedError:
            logging.info("[QuantFunc] Generation interrupted, returning blank image.")
            blank = torch.zeros(1, height, width, 3, dtype=torch.float32)
            return (blank, torch.ones(1, height, width, dtype=torch.float32), None)
        finally:
            # Unlink the /dev/shm QFRAW staging files. By image_to_image's return
            # the engine has already read them; on interrupt they're unused. Either
            # way they must not accumulate in tmpfs (RAM). Output shm is freed
            # separately in the reader path.
            for _p in tmp_paths:
                try:
                    os.unlink(_p)
                except OSError:
                    pass
            if control_path:
                try:
                    os.unlink(control_path)
                except OSError:
                    pass
            if mask_path:
                try:
                    os.unlink(mask_path)
                except OSError:
                    pass


# ============================================================================
# Node: QuantFunc Image List
# ============================================================================

class QuantFuncImageList:
    """Reference images for edit mode. Single or multiple images supported.
    Optional MASK input enables inpaint (mirrors ComfyUI SetLatentNoiseMask):
    only the white region of the mask is regenerated; black region is preserved.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        # ── 参考图 2-10 ──
        for i in range(2, 11):
            optional[f"ref_image_{i}"] = ("IMAGE", {
                "tooltip": f"第 {i} 张参考图(可选)。最多 10 张。",
            })
        # ── mask 组:main_image_mask 放在 mask_config 下,两个一起 ──
        # mask_config = MASK 高级配置(可选,不连接走默认)。
        # 默认 = QuantFuncMaskConfig 的默认 = strength=1.0, grow=6, blur=0.0, no_snap=False。
        optional["mask_config"] = ("QUANTFUNC_MASK_CONFIG", {
            "tooltip": "可选:用 QuantFunc Mask Config 节点配置遮罩高级参数"
                       "(strength / grow / blur / no_snap)。不连接 = 全用默认值。",
        })
        # main_image_mask = 主图遮罩(可选,触发 inpaint),放在 mask_config 下成组
        optional["main_image_mask"] = ("MASK", {
            "tooltip": "主图的 inpaint 遮罩(白=重绘,黑=保留,等价 ComfyUI "
                       "SetLatentNoiseMask)。只有白色区域被模型重绘;"
                       "黑色区域通过后处理 snap 完整保留原图。",
        })
        # ── img2img 源图:init_img 放在最下,与下方 strength(强度)成组 ──
        # 连接它即让 t2i 走 img2img:VAE 编码源图 → 按 strength 起始 sigma → denoise
        # (保留源图结构按 prompt 改)。与 edit 主图/参考图互斥(t2i pipeline,非 vision-edit)。
        optional["init_img"] = ("IMAGE", {
            "tooltip": "t2i img2img (以图生图) 源图。连接 → 文生图从这张图出发"
                       "(保留构图/结构,按 prompt 改);留空 = 纯文生图 / 用主图走 edit。"
                       "强度见下方 init_img_strength。与 edit 的 main_image 互斥。",
        })
        # ── 数值参数(widget)──
        # init_img_strength = 图生图强度(原 init_denoise),紧挨上方 init_img 成组
        optional["init_img_strength"] = ("FLOAT", {
            "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
            "tooltip": "图生图强度 (0~1,默认 0.6,仅 init_img 连接时生效):\n"
                       "  低 (0.2~0.4) — 强保留源图,仅微调\n"
                       "  0.6        — 平衡(推荐)\n"
                       "  1.0        — 完全重画(== 纯文生图,忽略源图)\n"
                       "对齐 diffusers img2img strength。",
        })
        # main_image_resize = 主图缩放
        optional["main_image_resize"] = (["720", "1024", "origin"], {
            "default": "720",
            "tooltip": "主图缩放模式:\n"
                       "  720  — 长边裁到 720 px(默认,最快)\n"
                       "  1024 — 长边裁到 1024 px(更高质量,稍慢)\n"
                       "  origin — 保留原尺寸,只把每边对齐到 16 的倍数",
        })
        # ref_image_resize_others = 参考图 2~10 缩放
        optional["ref_image_resize_others"] = (["720", "1024", "origin"], {
            "default": "720",
            "tooltip": "参考图 2~10 的缩放模式:\n"
                       "  720  — 长边裁到 720 px(默认,省 VRAM)\n"
                       "  1024 — 长边裁到 1024 px\n"
                       "  origin — 保留原尺寸(图大可能 OOM)",
        })
        # color_match = 潜在色彩匹配(插件侧 Reinhard 后处理,镜像引擎算法)
        optional["color_match"] = ("FLOAT", {
            "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
            "tooltip": "潜在色彩匹配强度 (0~1,默认 0):解码后把输出色彩分布匹配到"
                       "主参考图(Reinhard,Lab 空间):\n"
                       "  0.0     — 不校正(最锐利,可能色偏)\n"
                       "  0.3~0.5 — 平衡(推荐)\n"
                       "  1.0     — 完全匹配(色彩最忠实,细节略软)",
        })
        return {
            # main_image is OPTIONAL now: an img2img-only workflow connects
            # init_img instead. combine() requires exactly one of the two.
            "required": {},
            "optional": {
                # main_image = edit 模式的主图(被遮罩区域将被重绘)
                "main_image": ("IMAGE", {"tooltip": "edit 模式的主图(被遮罩区域将被重绘)。"
                                                    "纯 img2img 用 init_img 代替。"}),
                **optional,
            },
        }

    RETURN_TYPES = ("QUANTFUNC_IMAGE_LIST",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "QuantFunc"

    def combine(self, main_image=None, main_image_resize="720",
                ref_image_resize_others="720",
                main_image_mask=None, mask_config=None,
                init_img=None, init_img_strength=0.6, **kwargs):
        # #293 t2i img2img (SDEdit): init_img connected → t2i pipeline starts
        # from the VAE-encoded source (NOT vision/Kontext edit). Mutually
        # exclusive with the edit main_image; init_img wins if both are set.
        if init_img is not None:
            return ({
                "images": [init_img],          # the single source image
                "img2img": True,               # → QuantFuncGenerate: t2i img2img route
                "img2img_strength": float(init_img_strength),
                "ref_img_resize": "origin",    # encode source at the output resolution
                "ref_img_resize_others": "origin",
                "color_match": 0.0,
            },)
        if main_image is None:
            raise ValueError(
                "QuantFuncImageList: connect either main_image (edit) or "
                "init_img (t2i img2img).")
        images = [main_image]
        for i in range(2, 11):
            img = kwargs.get(f"ref_image_{i}")
            if img is not None:
                images.append(img)
        # Internal dict keys keep the legacy `ref_img_resize` / `mask` names so
        # QuantFuncGenerate.generate (which reads them) needs no parallel rename.
        out = {
            "images": images,
            "ref_img_resize": main_image_resize,
            "ref_img_resize_others": ref_image_resize_others,
            "color_match": float(kwargs.get("color_match", 0.0)),
        }
        if main_image_mask is not None:
            # Auto-align mask to main_image's pixel dims. Lets users wire any
            # ImageScale / Resize node into main_image without needing a
            # parallel resize for the mask (ComfyUI doesn't have a dedicated
            # MASK resize node — only Image-side scalers). main_image is
            # [B, H, W, C], main_image_mask is [B, H, W] or [B, 1, H, W].
            mask_t = main_image_mask
            img_h, img_w = main_image.shape[1], main_image.shape[2]
            mh = mask_t.shape[-2]
            mw = mask_t.shape[-1]
            if mh != img_h or mw != img_w:
                import torch.nn.functional as F
                m4 = mask_t.unsqueeze(1) if mask_t.dim() == 3 else mask_t
                m4 = F.interpolate(m4, size=(img_h, img_w),
                                    mode="bilinear", align_corners=False)
                mask_t = m4.squeeze(1) if mask_t.dim() == 3 else m4
            out["mask"] = mask_t
            # Defaults match QuantFuncMaskConfig defaults exactly.
            cfg = mask_config if isinstance(mask_config, dict) else {}
            out["mask_strength"] = float(cfg.get("mask_strength", 1.0))
            out["mask_grow"]     = int(cfg.get("mask_grow", 6))
            out["mask_blur"]     = float(cfg.get("mask_blur", 0.0))
            out["mask_no_snap"]  = bool(cfg.get("mask_no_snap", False))
        return (out,)


# ============================================================================
# Node: QuantFunc Mask Config (advanced inpaint knobs, optional)
# ============================================================================

class QuantFuncMaskConfig:
    """Inpaint MASK 高级配置(可选)。把 4 个边界参数打包成一个输出,接到
    QuantFunc Image List 的 `mask_config` 入口。不接就走默认值,跟 ComfyUI
    的 VAEEncodeForInpaint / SetLatentNoiseMask 保持一致。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "遮罩强度倍数 (0..1)。0 = 关闭 inpaint。",
                }),
                "mask_grow": ("INT", {
                    "default": 6, "min": 0, "max": 64, "step": 1,
                    "tooltip": "遮罩像素膨胀 N(对齐 ComfyUI VAEEncodeForInpaint "
                               "grow_mask_by 默认 6)。让接缝过渡更自然。",
                }),
                "mask_blur": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 64.0, "step": 0.5,
                    "tooltip": "遮罩高斯模糊 sigma(像素;对齐 ComfyUI MaskBlur)。"
                               "0 = 边界硬切。",
                }),
                "mask_no_snap": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "关闭最后一步对未遮罩区域的 snap 回原图。"
                               "默认关闭(snap 开,跟 ComfyUI 一致)。"
                               "开启 = 让模型决定整张图,边界过渡更柔和但"
                               "保留区会轻微飘移。",
                }),
            },
        }

    RETURN_TYPES = ("QUANTFUNC_MASK_CONFIG",)
    RETURN_NAMES = ("mask_config",)
    FUNCTION = "build"
    CATEGORY = "QuantFunc"

    def build(self, mask_strength, mask_grow, mask_blur, mask_no_snap):
        return ({
            "mask_strength": float(mask_strength),
            "mask_grow": int(mask_grow),
            "mask_blur": float(mask_blur),
            "mask_no_snap": bool(mask_no_snap),
        },)


# ============================================================================
# Node: QuantFunc Mask Scale By (mirrors ComfyUI ImageScaleBy, for MASK)
# ============================================================================

class QuantFuncMaskScaleBy:
    """按比例缩放 MASK,对称 ComfyUI 自带 ImageScaleBy(自带的不接 MASK 类型)。
    用法:LoadImageMask → QuantFunc Mask Scale By(同主图的 scale_by)→ main_image_mask"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "要缩放的遮罩"}),
                "scale_by": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01,
                    "tooltip": "缩放比例。要和主图的 ImageScaleBy 设成一样的值。",
                }),
                "method": (["bilinear", "nearest", "bicubic"], {
                    "default": "bilinear",
                    "tooltip": "插值方式。bilinear 默认平滑,nearest 保留硬边,"
                               "bicubic 最高质量但稍慢。",
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "scale_by"
    CATEGORY = "QuantFunc"

    def scale_by(self, mask, scale_by, method):
        import torch.nn.functional as F
        # ComfyUI MASK is [B, H, W]. Add channel dim for interpolate.
        m4 = mask.unsqueeze(1) if mask.dim() == 3 else mask
        h, w = m4.shape[-2], m4.shape[-1]
        new_h = max(1, int(round(h * scale_by)))
        new_w = max(1, int(round(w * scale_by)))
        kwargs = {"size": (new_h, new_w), "mode": method}
        if method in ("bilinear", "bicubic"):
            kwargs["align_corners"] = False
        out = F.interpolate(m4, **kwargs)
        out = out.clamp(0.0, 1.0)
        if mask.dim() == 3:
            out = out.squeeze(1)
        return (out,)


# ============================================================================
# Node: QuantFunc Export
# ============================================================================

class QuantFuncExport:
    """Export a pre-quantized model directory.

    Two output formats:
      - diffusers (formerly "separated") — HF-style directory:
                     transformer/, text_encoder/, vae/, ... each component
                     its own safetensors. The traditional layout,
                     compatible with `--model-dir <out>` reload.
      - comfy_checkpoint (formerly "bundle", aka 全家桶) — single
                     safetensors with all components packed under
                     per-component prefixes (model.diffusion_model.*,
                     text_encoder.*, vae.*, vision_encoder.*). One file,
                     loadable directly as a ComfyUI checkpoint via the
                     QuantFunc plugin's bundled-checkpoint adapter.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "export_path": ("STRING", {
                    "default": "",
                    "tooltip": "diffusers 模式: 输出到这个目录, 每个组件一个 safetensors\n"
                               "comfy_checkpoint 模式: 输出到这个目录下的单个 model.safetensors (全家桶)",
                }),
                "export_format": (["diffusers", "comfy_checkpoint"], {
                    "default": "diffusers",
                    "tooltip": "diffusers = HF 标准目录(每个组件独立文件)\n"
                               "comfy_checkpoint = 单文件 safetensors 打包所有组件 (全家桶)\n"
                               "  ↳ comfy_checkpoint 模式下 export_mode 强制为 'all'\n"
                               "    (transformer + text_encoder + vae + vision_encoder)",
                }),
                "export_mode": (["all", "custom"], {
                    "default": "all",
                    "tooltip": "(仅 diffusers) 'all' 复制整个模型(vae、tokenizer 等)用于独立使用; "
                               "'custom' 选择单个组件",
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

    def export_model(self, pipeline, export_path,
                     export_format="diffusers", export_mode="all",
                     export_transformer=True, export_text_encoder=False,
                     export_vision_encoder=False):
        if not export_path:
            raise ValueError("export_path is required")

        if "options" not in pipeline:
            pipeline["options"] = {}

        # Accept both the new UI labels (diffusers / comfy_checkpoint) and the
        # legacy labels (separated / bundle) so old workflow JSON keeps working.
        # Engine-side string is unchanged: "bundle" or absent (= separated).
        is_bundle = export_format in ("comfy_checkpoint", "bundle")

        if is_bundle:
            # Bundle (= ComfyUI checkpoint) always packs the whole pipeline —
            # per-component selection doesn't apply. Force `all` so the worker
            # doesn't filter components and break the layout.
            pipeline["options"]["export_format"] = "bundle"
            pipeline["options"]["export_models"] = "all"
        else:
            pipeline["options"].pop("export_format", None)  # default = separated/diffusers
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
            pipeline["options"]["export_models"] = ",".join(components)

        _manager.export_model(pipeline, export_path)
        logging.info("[QuantFunc] Export complete (%s): %s", export_format, export_path)
        return {}


# ============================================================================
# Registration
# ============================================================================

class QuantFuncLatentPreview:
    """Live latent-preview viewer (no settings — just a display).

    Wire the `latent_preview` OUTPUT of a 'QuantFunc Generate' node into this
    node. Connecting it is all it takes to ENABLE the preview. During
    generation the engine decodes a cheap per-step latent2rgb image of the
    forming latent (negligible overhead vs the transformer step) and streams it
    ONTO THIS node, so you watch the image form here instead of on the generate
    node. When the run finishes, this node shows the final image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_preview": ("QF_LATENT_PREVIEW",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "show"
    OUTPUT_NODE = True
    CATEGORY = "QuantFunc"

    def show(self, latent_preview=None, unique_id=None):
        # Pure live viewer. The per-step latent2rgb frames are streamed onto
        # THIS node's id by the engine preview callback during generation, and
        # the frontend keeps the last frame shown. We deliberately do NOT
        # persist a second copy via {"ui":{"images"}} — that produced a 2nd
        # stacked image. The live latent frame is the only picture shown here.
        return {}


class QuantFuncGenerateVideo:
    """Wan / LTX video generation (text-to-video, + image-to-video on capable
    checkpoints). Outputs the frame batch as ComfyUI IMAGE and, for AV models
    (LTX-2), the vocoder waveform as ComfyUI AUDIO.

    Family-agnostic: works with any loaded video pipeline (Wan 2.2 TI2V-5B /
    I2V-A14B, LTX-2). Wan produces no audio → the AUDIO output is None (silent
    playback); LTX-2 emits an audio track. Wire an optional `start_image` IMAGE
    (ComfyUI Wan convention) to condition the clip on a starting frame (image-to-
    video) — absent → text-to-video. NOTE on i2v support: it is the ENGINE that
    decides — a Wan i2v-capable checkpoint honors the start image (a t2v-only Wan
    checkpoint throws a clear error); LTX i2v is not yet wired in the engine, so on
    LTX the start_image is currently IGNORED (a loud warning is logged) and a t2v
    clip is produced. Inputs use the ComfyUI Wan/LTX names (`length` = frame count,
    `start_image` = first-frame condition). Feed `frames` (+ `audio`) into
    'QuantFunc Video Preview' for in-graph A/V playback, or the core CreateVideo →
    SaveVideo (or VHS) nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            # WIDGET-ORDER BACK-COMPAT (graph-format .json workflows): ComfyUI restores
            # a saved node's `widgets_values` by ARRAY POSITION for classic dict-
            # INPUT_TYPES nodes, NOT by name. This node (#344) originally shipped the
            # widget order [prompt, width, height, num_frames, steps, guidance_scale,
            # seed, negative_prompt]. To keep an OLD saved graph from misaligning its
            # steps/cfg/seed into the wrong slots, we preserve those positions exactly:
            # the frame-count widget stays at slot 4 (just RENAMED num_frames→`length`,
            # which is semantically the same value, so an old array maps correctly),
            # steps/guidance_scale/seed keep their slots, negative_prompt stays the
            # first optional widget (slot 8), and any NEW widget (fps) is APPENDED after
            # it so a shorter old array never reaches it. `start_image` is an IMAGE
            # SOCKET (not a positional widget), so it doesn't shift widget indices.
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                # `length` = frame count (ComfyUI Wan/LTX convention, cf. the core
                # WanImageToVideo/EmptyLTXVLatentVideo `length` input) — RENAMED IN PLACE
                # from the old `num_frames` (SAME widget slot 4 → old saved values map
                # correctly). Wan needs (length-1)%4==0 (e.g. 49=48+1, or the Wan-native
                # 81); LTX needs (length-1)%8==0 (e.g. 49, or the LTX-native 97). 49 fits both.
                "length": ("INT", {"default": 49, "min": 1, "max": 257, "step": 4,
                    "tooltip": "Number of frames (ComfyUI Wan/LTX 'length'; formerly "
                               "'num_frames'). Wan: (length-1)%4==0 (49, 81, …); "
                               "LTX: (length-1)%8==0 (49, 97, …)."}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                # negative_prompt stays the FIRST optional widget (slot 8, as in #344).
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                # fps APPENDED after negative_prompt (new slot 9) so an old 8-slot saved
                # array never lands a value here (falls to the method default 24.0).
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Playback fps (rides in options_json; Wan + LTX both read it)."}),
                # `start_image` = the first-frame condition (ComfyUI Wan i2v convention,
                # cf. the core WanImageToVideo `start_image` input). IMAGE = a socket, not
                # a widget → does not consume a widgets_values slot.
                "start_image": ("IMAGE", {"tooltip": "Optional first-frame condition "
                    "(ComfyUI 'start_image') → image-to-video. Honored by Wan i2v-capable "
                    "checkpoints; LTX i2v is not yet engine-wired (start_image ignored on "
                    "LTX → t2v). Absent → text-to-video."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    # frames (IMAGE) — universal, for upscale/interpolate/preview; audio (AUDIO) —
    # None for Wan, the vocoder track for LTX-2; video (VIDEO) — a native ComfyUI
    # VIDEO so it plugs STRAIGHT into the core `Save Video` node in one wire (no
    # `Create Video` bridge). `Video Combine` / `Preview Image` still take `frames`.
    RETURN_TYPES = ("IMAGE", "AUDIO", "VIDEO")
    RETURN_NAMES = ("frames", "audio", "video")
    FUNCTION = "generate_video"
    CATEGORY = "QuantFunc"

    def generate_video(self, pipeline, prompt, width, height, length, steps,
                       guidance_scale, seed, fps=24.0,
                       negative_prompt="", start_image=None, unique_id=None):
        import torch
        # `length` was renamed IN PLACE from the old required `num_frames` (same widget
        # slot — see the INPUT_TYPES widget-order note), so an old graph-format saved
        # workflow's slot-4 value maps here correctly.
        length = int(length)
        cfg = dict(pipeline)
        cfg["options"] = dict(cfg.get("options", {}))
        family = _pipeline_video_family(cfg)  # 'wan' | 'ltx' | None (advisory/logging)
        cache_key = _manager.ensure_pipeline(cfg, node_id=unique_id)
        # length (frame count) rides in the manager call; fps + negative_prompt ride
        # in options_json (the C-API t2v/i2v contract).
        opts = {}
        if fps and float(fps) > 0.0:
            opts["fps"] = float(fps)
        neg = negative_prompt if (isinstance(negative_prompt, str) and negative_prompt) else ""
        if neg:
            opts["negative_prompt"] = neg
        opts_json = json.dumps(opts) if opts else None
        pbar = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(steps)
        except Exception:
            pass

        # image-to-video when a start_image is wired; else text-to-video. The start
        # image is staged as a QFRAW01 raw-RGB blob (engine load_image reads it
        # directly) and passed as the sole ref path (= first-frame condition).
        staging_dir = "/dev/shm" if (os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)) \
            else tempfile.gettempdir()
        start_image_path = None
        try:
            if start_image is not None:
                # LTX i2v (first-frame conditioning) is NOT yet wired in the engine
                # (LTX2VideoPipeline::generate_video ignores cond_images — "P5"), so on
                # an LTX pipeline the start_image is silently discarded and a plain t2v
                # clip is produced. Warn LOUDLY rather than silently mislead. Wan I2V is
                # supported by capable Wan checkpoints (a t2v-only Wan checkpoint throws
                # a clear engine error instead of ignoring it). We still forward the
                # start_image so a future LTX-i2v-capable engine works unchanged.
                if family == "ltx":
                    logging.warning(
                        "[QuantFunc] LTX i2v (start_image conditioning) is not yet "
                        "supported by the engine — the start_image will likely be "
                        "IGNORED and a text-to-video clip produced. (Wan I2V is "
                        "supported on capable Wan checkpoints.)")
                start_image_path = _write_qfraw_image(start_image, staging_dir)
                if not start_image_path:
                    raise RuntimeError("Failed to stage start_image for image-to-video")
                logging.info("[QuantFunc] video i2v (%s): start_image conditioned, "
                             "%d frames @ %.1f fps", family or "auto", length, fps)
                frames, audio = _manager.image_to_video(
                    cache_key, prompt, [start_image_path], height, width, steps, seed,
                    float(guidance_scale), length, negative_prompt=neg,
                    options_json=opts_json, pbar=pbar)
            else:
                logging.info("[QuantFunc] video t2v (%s): %d frames @ %.1f fps",
                             family or "auto", length, fps)
                frames, audio = _manager.text_to_video(
                    cache_key, prompt, height, width, steps, seed, float(guidance_scale),
                    length, options_json=opts_json, pbar=pbar)
        finally:
            if start_image_path:
                try:
                    os.unlink(start_image_path)
                except OSError:
                    pass

        image = torch.from_numpy(frames)                 # [N, H, W, 3] float32 [0,1] = IMAGE
        audio_out = None
        if audio is not None:
            wav = torch.from_numpy(audio["waveform"]).unsqueeze(0)  # [1, C, N]
            audio_out = {"waveform": wav, "sample_rate": int(audio["sample_rate"])}
        video_out = _frames_to_video(image, audio_out, fps)
        return (image, audio_out, video_out)


def _frames_to_video(image, audio_out, fps):
    """Wrap the IMAGE frame batch (+ optional AUDIO) as a native ComfyUI VIDEO so the
    node's `video` output plugs STRAIGHT into the core `Save Video` node (one wire,
    no `Create Video`). Graceful: returns None on an older ComfyUI without the VIDEO
    API — `frames`/`audio` are still emitted, so `Video Combine`/`Create Video` work."""
    try:
        from comfy_api.input_impl import VideoFromComponents
        from comfy_api.latest import VideoComponents
        from fractions import Fraction
        return VideoFromComponents(VideoComponents(
            images=image, audio=audio_out, frame_rate=Fraction(float(fps) or 24.0)))
    except Exception as e:  # noqa: BLE001 — ComfyUI without the native VIDEO type
        logging.debug("[QuantFunc] native VIDEO output unavailable (%s); "
                      "frames/audio still emitted", e)
        return None


def _encode_video_preview(frames_u8, fps, audio, out_path, container_fmt):
    """Encode a [N,H,W,3] uint8 RGB frame batch (+ optional audio) into a
    browser-playable clip at `out_path`.

    audio: None or {"waveform": np.float32 [C, N], "sample_rate": int}.
    Primary encoder is PyAV (bundles ffmpeg libs → muxes video + audio in ONE
    container with NO external ffmpeg binary). Falls back to cv2 VideoWriter
    (video-only) when PyAV is unavailable. Returns True if audio was muxed, False
    if the clip is video-only. Raises when NO encoder is available."""
    N, H, W = int(frames_u8.shape[0]), int(frames_u8.shape[1]), int(frames_u8.shape[2])
    # yuv420p (h264/vp9) needs even width/height — crop the last row/col if odd.
    if W % 2:
        frames_u8 = frames_u8[:, :, :W - 1, :]; W -= 1
    if H % 2:
        frames_u8 = frames_u8[:, :H - 1, :, :]; H -= 1
    rate = max(1, int(round(float(fps) or 24.0)))

    # ---- Primary: PyAV (video + optional audio, single container) ----
    # Recipe mirrors ComfyUI core's VideoFromComponents.save_to (h264 + per-frame
    # reformat('yuv420p'); audio = planar 'fltp' [C,N] frame with pts=0). CRUCIAL:
    # BOTH streams are created UP FRONT, before muxing any packet — the muxer
    # writes the container header on the first mux and assigns each stream a
    # time_base then; an audio stream added AFTER the first video mux keeps
    # time_base=0 → PyAV crashes "Cannot rebase to zero time" at the audio mux
    # (verified on PyAV 17.1.0).
    container = None
    try:
        import av
        from fractions import Fraction
        vcodec = "h264" if container_fmt == "mp4" else "libvpx-vp9"
        acodec = "aac" if container_fmt == "mp4" else "libopus"
        container = av.open(out_path, mode="w")
        vstream = container.add_stream(vcodec, rate=Fraction(rate, 1))
        vstream.width = W
        vstream.height = H
        vstream.pix_fmt = "yuv420p"

        # Prepare the audio stream BEFORE encoding any video (see note above).
        astream = None
        wav = None
        layout = "stereo"
        sr = 16000
        if audio is not None:
            try:
                wav = np.asarray(audio["waveform"], dtype=np.float32)  # [C, N] planar
                if wav.ndim == 1:
                    wav = wav[None, :]
                ch = int(wav.shape[0])
                if ch > 2:                       # downmix >2ch to stereo (keep 0,1)
                    wav = np.ascontiguousarray(wav[:2]); ch = 2
                else:
                    wav = np.ascontiguousarray(wav)
                sr = int(audio.get("sample_rate", 16000)) or 16000
                layout = {1: "mono", 2: "stereo"}.get(ch, "stereo")
                astream = container.add_stream(acodec, rate=sr, layout=layout)
            except Exception as ae:
                logging.warning("[QuantFunc] video preview: audio stream setup "
                                "failed (%s) — video-only", ae)
                astream = None

        # Encode + mux the video frames.
        for i in range(N):
            vframe = av.VideoFrame.from_ndarray(np.ascontiguousarray(frames_u8[i]),
                                                format="rgb24").reformat(format="yuv420p")
            for pkt in vstream.encode(vframe):
                container.mux(pkt)
        for pkt in vstream.encode():  # flush video
            container.mux(pkt)

        # Encode + mux the audio (stream already registered up front).
        has_audio = False
        if astream is not None:
            try:
                aframe = av.AudioFrame.from_ndarray(wav, format="fltp", layout=layout)
                aframe.sample_rate = sr
                aframe.pts = 0
                container.mux(astream.encode(aframe))
                container.mux(astream.encode(None))  # flush audio
                has_audio = True
            except Exception as ae:
                logging.warning("[QuantFunc] video preview: audio mux failed "
                                "(%s) — video-only", ae)
        container.close()
        return has_audio
    except Exception as e:
        logging.warning("[QuantFunc] video preview: PyAV encode failed (%s) — "
                        "falling back to cv2 (video-only)", e)
        # Close the open container's fd + drop the partially-written file so the
        # cv2 fallback re-opens a clean out_path (no fd/partial-file leak).
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        try:
            if os.path.exists(out_path):
                os.unlink(out_path)
        except OSError:
            pass

    # ---- Fallback: cv2 VideoWriter (video-only) ----
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if container_fmt == "mp4" else "VP80"))
        vw = cv2.VideoWriter(out_path, fourcc, float(rate), (W, H))
        if not vw.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        for i in range(N):
            bgr = cv2.cvtColor(np.ascontiguousarray(frames_u8[i]), cv2.COLOR_RGB2BGR)
            vw.write(bgr)
        vw.release()
        if audio is not None:
            logging.warning("[QuantFunc] video preview: cv2 fallback cannot mux "
                            "audio — clip is silent (install PyAV for A/V).")
        return False
    except Exception as e2:
        raise RuntimeError(
            "No usable video encoder — install PyAV (`pip install av`) or a cv2 "
            f"with video support. PyAV+cv2 both failed: {e2}")


class QuantFuncVideoPreview:
    """In-graph A/V playback for QuantFunc video output. Wire the `frames` IMAGE
    (and optional `audio` AUDIO) from 'QuantFunc Generate Video'.

    Encodes the frames to a browser-playable clip (H.264 mp4 by default, muxing the
    AUDIO track when present via PyAV — no external ffmpeg binary needed) into
    ComfyUI's temp dir and returns ComfyUI's NATIVE video-preview payload
    ({"ui": {"images": [...], "animated": (True,)}}), the SAME mechanism the core
    SaveVideo / SaveWEBM nodes use (ui.PreviewVideo) — so ComfyUI renders its own
    <video> player with controls + audio on this node, no custom widget. Wan clips
    (no audio) play silently. The `frames` remain a standard IMAGE elsewhere in the
    graph (this node is a terminal viewer), so the core CreateVideo/SaveVideo and
    VHS nodes still work off the same output. For the fully-official flow you can
    instead wire frames+audio into CreateVideo → SaveVideo."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Playback frame rate. Match the value used on Generate Video."}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Optional audio track (LTX-2). Muxed into "
                    "the clip when present; Wan has none → silent."}),
                "container": (["mp4", "webm"], {"default": "mp4"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "QuantFunc"

    def preview(self, frames, fps, audio=None, container="mp4", unique_id=None):
        import random
        import folder_paths

        imgs = frames.detach().cpu().numpy()            # [N,H,W,3] float [0,1]
        if imgs.ndim == 3:
            imgs = imgs[None, ...]
        frames_u8 = (np.clip(imgs, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        N, H, W = frames_u8.shape[0], frames_u8.shape[1], frames_u8.shape[2]

        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        ext = "webm" if container == "webm" else "mp4"
        prefix = "qfvideo_" + "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
        filename = f"{prefix}.{ext}"
        out_path = os.path.join(temp_dir, filename)

        # Convert a ComfyUI AUDIO dict ({"waveform": [B,C,N] tensor, "sample_rate"})
        # to the planar numpy [C, N] the encoder expects (first batch item).
        audio_np = None
        if isinstance(audio, dict) and audio.get("waveform") is not None:
            try:
                wf = audio["waveform"]
                w = wf.detach().cpu().numpy() if hasattr(wf, "detach") else np.asarray(wf)
                if w.ndim == 3:      # [B, C, N] → first batch
                    w = w[0]
                elif w.ndim == 1:    # [N] → mono
                    w = w[None, :]
                audio_np = {"waveform": w.astype(np.float32),
                            "sample_rate": int(audio.get("sample_rate", 16000))}
            except Exception as e:
                logging.warning("[QuantFunc] video preview: bad AUDIO input (%s) — silent", e)

        try:
            _encode_video_preview(frames_u8, fps, audio_np, out_path, ext)
        except Exception as e:
            logging.error("[QuantFunc] video preview encode failed: %s", e)
            return {"ui": {"text": [f"video preview encode failed: {e}"]}}

        # ComfyUI NATIVE video-preview payload (== ui.PreviewVideo.as_dict()): a
        # SavedResult {filename, subfolder, type} list under "images" + animated=(True,).
        # The frontend infers a video from the .mp4/.webm extension and renders its
        # own <video> player (audio via the muxed track). `format` is an extra hint
        # some frontend builds use to pick the video widget.
        return {"ui": {"images": [{
            "filename": filename,
            "subfolder": "",
            "type": "temp",
            "format": f"video/{ext}",
        }], "animated": (True,)}}


class QuantFuncLayerViewer:
    """Interactive viewer for QwenImageLayered's N RGBA layers.

    Wire QuantFuncGenerate's `image` output to `images` (and `mask` → `masks` for true
    transparency). The node body shows a thumbnail list of all layers + a main canvas:
    click a layer to isolate it; the default (nothing selected) shows the alpha-over
    composite of every layer. Non-layered inputs (N==1) just show that one image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "masks": ("MASK", {"tooltip": "Per-layer alpha (wire QuantFunc Generate's `mask`)."}),
                "save_layers": ("BOOLEAN", {"default": False,
                    "tooltip": "Also write the N RGBA layers + the composite to the output/ folder."}),
                "filename_prefix": ("STRING", {"default": "QF_layer"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "view"
    OUTPUT_NODE = True
    CATEGORY = "QuantFunc"

    @staticmethod
    def _to_rgba_uint8(img_hw3, mask_hw):
        """[H,W,3] float + optional [H,W] alpha → [H,W,4] uint8 (opaque if no mask)."""
        rgb = (np.clip(img_hw3, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        if mask_hw is not None:
            a = (np.clip(mask_hw, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        else:
            a = np.full(rgb.shape[:2], 255, np.uint8)
        return np.dstack([rgb, a])

    @staticmethod
    def _composite(rgba_list):
        """Alpha-over the layers bottom→top → [H,W,4] uint8 with STRAIGHT (unassociated)
        alpha, which is how PIL writes an RGBA PNG. Accumulate in PREMULTIPLIED form (the
        numerically clean way to iterate Porter-Duff 'over'), then UN-premultiply the RGB
        at the end — saving premultiplied RGB under PIL's straight-alpha PNG would darken
        semi-transparent regions (the browser canvas display path is unaffected; it
        composites straight RGBA via source-over)."""
        out = np.zeros((*rgba_list[0].shape[:2], 4), np.float32)  # premultiplied accumulator
        for rgba in rgba_list:
            f = rgba.astype(np.float32) / 255.0
            a = f[..., 3:4]
            out[..., :3] = f[..., :3] * a + out[..., :3] * (1.0 - a)
            out[..., 3:4] = a + out[..., 3:4] * (1.0 - a)
        alpha = out[..., 3:4]
        rgb = np.divide(out[..., :3], alpha, out=np.zeros_like(out[..., :3]), where=alpha > 1e-6)
        straight = np.concatenate([rgb, alpha], axis=-1)
        return (np.clip(straight, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def view(self, images, masks=None, save_layers=False, filename_prefix="QF_layer"):
        import random
        from PIL import Image
        import folder_paths

        imgs = images.detach().cpu().numpy()                       # [N,H,W,3]
        N, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[2]
        msk = masks.detach().cpu().numpy() if masks is not None else None  # [M,H,W]|None

        rgba_list = [
            self._to_rgba_uint8(imgs[i], msk[i] if (msk is not None and i < msk.shape[0]) else None)
            for i in range(N)
        ]

        # Save each layer as an RGBA PNG to ComfyUI's temp dir for the live viewer.
        temp_dir = folder_paths.get_temp_directory()
        prefix = "qflayer_" + "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
        full_dir, fname, counter, subfolder, _ = folder_paths.get_save_image_path(prefix, temp_dir, W, H)
        ui_layers = []
        for i, rgba in enumerate(rgba_list):
            file = f"{fname}_{counter:05}_{i:03}.png"
            Image.fromarray(rgba, "RGBA").save(os.path.join(full_dir, file), compress_level=1)
            ui_layers.append({"filename": file, "subfolder": subfolder, "type": "temp"})

        # Optional permanent save → output/ (each layer + the composite).
        if save_layers:
            out_dir = folder_paths.get_output_directory()
            o_full, o_name, o_counter, _o_sub, _ = folder_paths.get_save_image_path(filename_prefix, out_dir, W, H)
            for i, rgba in enumerate(rgba_list):
                Image.fromarray(rgba, "RGBA").save(
                    os.path.join(o_full, f"{o_name}_{o_counter:05}_layer{i:03}.png"))
            Image.fromarray(self._composite(rgba_list), "RGBA").save(
                os.path.join(o_full, f"{o_name}_{o_counter:05}_composite.png"))

        return {"ui": {"qf_layers": ui_layers, "qf_size": [W, H]}}


NODE_CLASS_MAPPINGS = {
    "QuantFuncLayerViewer": QuantFuncLayerViewer,
    "QuantFuncGenerateVideo": QuantFuncGenerateVideo,
    "QuantFuncVideoPreview": QuantFuncVideoPreview,
    "QuantFuncPipelineConfig": QuantFuncPipelineConfig,
    "QuantFuncModelLoader": QuantFuncModelLoader,
    "QuantFuncWanCombineExperts": QuantFuncWanCombineExperts,
    "QuantFuncWanCombineExpertsAuto": QuantFuncWanCombineExpertsAuto,
    "QuantFuncModelAutoLoader": QuantFuncModelAutoLoader,
    # QuantFuncBuildPipeline lives in nodes_format_adapters.py (one canonical
    # implementation; loaded after this map → __init__.py's update() lifts it
    # under the same registration key).
    "QuantFuncPrequantAutoLoader": QuantFuncPrequantAutoLoader,
    "QuantFuncPrecisionConfigAutoLoader": QuantFuncPrecisionConfigAutoLoader,
    "QuantFuncBaseSeriesModelAutoLoader": QuantFuncBaseSeriesModelAutoLoader,
    "QuantFuncBaseModelAutoLoader": QuantFuncBaseModelAutoLoader,
    "QuantFuncBaseModelAutoLoaderWithDownload": QuantFuncBaseModelAutoLoaderWithDownload,
    "QuantFuncTransformerAutoLoader": QuantFuncTransformerAutoLoader,
    "QuantFuncLoRAAutoLoader": QuantFuncLoRAAutoLoader,
    "QuantFuncLoRALoader": QuantFuncLoRALoader,
    "QuantFuncControlNetLoader": QuantFuncControlNetLoader,
    "QuantFuncControlNetAutoLoader": QuantFuncControlNetAutoLoader,
    "QuantFuncControlImage": QuantFuncControlImage,
    "QuantFuncLoRAConfig": QuantFuncLoRAConfig,
    "QuantFuncLayeredConfig": QuantFuncLayeredConfig,
    "QuantFuncGenerate": QuantFuncGenerate,
    "QuantFuncLatentPreview": QuantFuncLatentPreview,
    "QuantFuncImageList": QuantFuncImageList,
    "QuantFuncMaskConfig": QuantFuncMaskConfig,
    "QuantFuncMaskScaleBy": QuantFuncMaskScaleBy,
    "QuantFuncExport": QuantFuncExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuantFuncPipelineConfig": "QuantFunc Pipeline Config",
    "QuantFuncModelLoader": "QuantFunc Model Loader",
    "QuantFuncWanCombineExperts": "QuantFunc Wan Combine Experts (A14B two-transformer)",
    "QuantFuncWanCombineExpertsAuto": "QuantFunc Wan Combine Experts (Auto)",
    "QuantFuncModelAutoLoader": "QuantFunc Model Auto Loader",
    # QuantFuncBuildPipeline display name is set in nodes_format_adapters.py.
    "QuantFuncPrequantAutoLoader": "QuantFunc Prequant Auto Loader",
    "QuantFuncPrecisionConfigAutoLoader": "QuantFunc Precision Config Auto Loader",
    "QuantFuncBaseSeriesModelAutoLoader": "QuantFunc Base Series Model Auto Loader",
    "QuantFuncBaseModelAutoLoader": "QuantFunc Base Model Auto Loader",
    "QuantFuncBaseModelAutoLoaderWithDownload": "QuantFunc Base Model Auto Loader with Download",
    "QuantFuncTransformerAutoLoader": "QuantFunc Transformer Auto Loader",
    "QuantFuncLoRAAutoLoader": "QuantFunc LoRA Auto Loader",
    "QuantFuncLoRALoader": "QuantFunc LoRA",
    "QuantFuncControlNetLoader": "QuantFunc ControlNet Loader",
    "QuantFuncControlNetAutoLoader": "QuantFunc ControlNet Auto Loader",
    "QuantFuncControlImage": "QuantFunc Control Image",
    "QuantFuncLoRAConfig": "QuantFunc LoRA Config",
    "QuantFuncLayeredConfig": "QuantFunc Layered Config",
    "QuantFuncGenerate": "QuantFunc Generate",
    "QuantFuncLayerViewer": "QuantFunc Layer Viewer",
    "QuantFuncGenerateVideo": "QuantFunc Generate Video (Wan / LTX · t2v + i2v)",
    "QuantFuncVideoPreview": "QuantFunc Video Preview (A/V)",
    "QuantFuncLatentPreview": "QuantFunc Latent Preview",
    "QuantFuncImageList": "QuantFunc Image List",
    "QuantFuncMaskConfig": "QuantFunc Mask Config",
    "QuantFuncMaskScaleBy": "QuantFunc Mask Scale By",
    "QuantFuncExport": "QuantFunc Export",
}
