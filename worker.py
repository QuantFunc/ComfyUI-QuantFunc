#!/usr/bin/env python3
"""QuantFunc worker process — runs in a separate process with isolated CUDA libraries.

Communicates with the parent (ComfyUI nodes.py) via stdin/stdout:
  - stdin:  JSON lines (commands)
  - stdout: JSON lines (responses, progress) + raw binary (image data)
  - stderr: log messages (forwarded to parent's console)

This isolation prevents DLL conflicts between ComfyUI's PyTorch CUDA 12.x
and QuantFunc's CUDA 13.x on Windows.
"""

import ctypes
import json
import os
import platform
import queue
import struct
import sys
import threading
import traceback

# ============================================================================
# Binary I/O (stdin/stdout in binary mode)
# ============================================================================

_stdout_lock = threading.Lock()

def _init_binary_io():
    """Switch stdin/stdout to raw binary mode."""
    if platform.system() == "Windows":
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    # Reopen as unbuffered binary streams
    global _stdin, _stdout
    _stdin = os.fdopen(sys.stdin.fileno(), "rb", closefd=False)
    _stdout = os.fdopen(sys.stdout.fileno(), "wb", 0, closefd=False)


def send_json(obj):
    """Send a JSON object as a single line to stdout."""
    data = json.dumps(obj, ensure_ascii=True).encode("utf-8") + b"\n"
    with _stdout_lock:
        _stdout.write(data)
        _stdout.flush()


def send_binary(data: bytes):
    """Send raw binary data to stdout."""
    with _stdout_lock:
        _stdout.write(data)
        _stdout.flush()


def read_command():
    """Read one JSON command from stdin. Returns None on EOF."""
    line = _stdin.readline()
    if not line:
        return None
    return json.loads(line.decode("utf-8").strip())


def log(msg):
    """Write to stderr (forwarded to parent's console)."""
    sys.stderr.write(f"[worker] {msg}\n")
    sys.stderr.flush()


# ============================================================================
# ctypes bindings (mirrors quantfunc.h)
# ============================================================================

_lib = None

class _Pipeline(ctypes.Structure):
    pass
class _Image(ctypes.Structure):
    pass
class _Video(ctypes.Structure):   # #344 LTX-2 t2v handle (frames + audio)
    pass

PIPE_PTR = ctypes.POINTER(_Pipeline)
IMG_PTR = ctypes.POINTER(_Image)
VID_PTR = ctypes.POINTER(_Video)
PROGRESS_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
# Latent-preview callback (mirrors quantfunc.h latent_preview_callback):
#   int (*)(int step, int total, int width, int height,
#           const unsigned char *rgb, void *user_data)
LATENT_PREVIEW_CB = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_void_p)

class InitParams(ctypes.Structure):
    _fields_ = [
        ("model_dir",        ctypes.c_char_p),
        ("transformer_path", ctypes.c_char_p),
        ("vae_path",         ctypes.c_char_p),
        ("text_encoder_path",ctypes.c_char_p),
        ("tokenizer_path",   ctypes.c_char_p),
        ("scheduler_config", ctypes.c_char_p),
        ("model_backend",    ctypes.c_char_p),
        ("device_idx",       ctypes.c_int),
        ("config_json",      ctypes.c_char_p),
    ]

class T2IParams(ctypes.Structure):
    _fields_ = [
        ("prompt",            ctypes.c_char_p),
        ("height",            ctypes.c_int),
        ("width",             ctypes.c_int),
        ("num_steps",         ctypes.c_int),
        ("guidance_scale",    ctypes.c_float),
        ("seed",              ctypes.c_int64),
        ("options_json",      ctypes.c_char_p),
        ("progress_callback", PROGRESS_CB),
        ("callback_user_data",ctypes.c_void_p),
        # Per-step latent preview (mirrors quantfunc_t2i_params_t). NULL = off.
        ("latent_preview_callback", LATENT_PREVIEW_CB),
        ("latent_preview_user_data", ctypes.c_void_p),
    ]

class I2IParams(ctypes.Structure):
    _fields_ = [
        ("prompt",            ctypes.c_char_p),
        ("ref_image_paths",   ctypes.POINTER(ctypes.c_char_p)),
        ("num_ref_images",    ctypes.c_int),
        ("height",            ctypes.c_int),
        ("width",             ctypes.c_int),
        ("num_steps",         ctypes.c_int),
        ("true_cfg_scale",    ctypes.c_float),
        ("negative_prompt",   ctypes.c_char_p),
        ("seed",              ctypes.c_int64),
        ("options_json",      ctypes.c_char_p),
        ("progress_callback", PROGRESS_CB),
        ("callback_user_data",ctypes.c_void_p),
        # Inpaint (mirrors include/quantfunc.h additions). NULL/empty mask_path
        # = no inpaint. Convention: white = inpaint, black = preserve.
        ("mask_path",         ctypes.c_char_p),
        ("mask_strength",     ctypes.c_float),
        ("mask_grow",         ctypes.c_int),
        ("mask_blur",         ctypes.c_float),
        ("mask_no_snap",      ctypes.c_int),
        # Per-step latent preview (mirrors quantfunc_i2i_params_t). NULL = off.
        ("latent_preview_callback", LATENT_PREVIEW_CB),
        ("latent_preview_user_data", ctypes.c_void_p),
        # Brush/context conditioning (QwenImage-Layered-Control-V2).
        # Appended at END of struct — existing callers zero-init → NULL = disabled.
        # Must match quantfunc_i2i_params_t::context_image_path in include/quantfunc.h.
        ("context_image_path", ctypes.c_char_p),
    ]

class ExportParams(ctypes.Structure):
    _fields_ = [
        ("model_dir",        ctypes.c_char_p),
        ("export_path",      ctypes.c_char_p),
        ("transformer_path", ctypes.c_char_p),
        ("model_backend",    ctypes.c_char_p),
        ("device_idx",       ctypes.c_int),
        ("config_json",      ctypes.c_char_p),
    ]


def _load_dll(dll_path):
    """Load quantfunc DLL with isolated CUDA library path."""
    global _lib

    dll_dir = os.path.dirname(os.path.abspath(dll_path))

    if platform.system() == "Windows":
        # Collect all directories that may contain DLL dependencies
        extra_dirs = [dll_dir]

        # CUDA toolkit bin (set by parent or system)
        # CUDA 13+ puts DLLs in bin/x64/, older versions in bin/
        cuda_path = os.environ.get("CUDA_PATH", "")
        if not cuda_path:
            # Auto-detect: find highest version CUDA toolkit
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

        # cuDNN: installed separately, scan common locations
        # Add ALL cuda-version subdirs so both CUDA 12 and 13 DLLs are found
        cudnn_base = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                  "NVIDIA", "CUDNN")
        if os.path.isdir(cudnn_base):
            for ver in sorted(os.listdir(cudnn_base), reverse=True):
                # cuDNN 9.x puts DLLs in bin/<cuda_ver>/x64/
                ver_dir = os.path.join(cudnn_base, ver, "bin")
                if os.path.isdir(ver_dir):
                    for sub in sorted(os.listdir(ver_dir), reverse=True):
                        x64 = os.path.join(ver_dir, sub, "x64")
                        if os.path.isdir(x64):
                            extra_dirs.append(x64)
                    break  # use first (highest) cuDNN version

        # Scan PATH for directories with CUDA/cuDNN/OpenCV DLLs
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

        # Register via add_dll_directory (Python 3.8+) AND PATH
        for d in extra_dirs:
            d = os.path.abspath(d)
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(d)
                    log(f"  add_dll_directory: {d}")
                except OSError as e:
                    log(f"  add_dll_directory FAILED: {d} ({e})")
        os.environ["PATH"] = os.pathsep.join(extra_dirs) + os.pathsep + os.environ.get("PATH", "")

    if platform.system() != "Windows":
        # Ensure dll_dir is in LD_LIBRARY_PATH so extracted dep .so files are found
        dll_dir_abs = os.path.abspath(dll_dir)
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if dll_dir_abs not in ld_path.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = dll_dir_abs + os.pathsep + ld_path
            log(f"  Added {dll_dir_abs} to LD_LIBRARY_PATH")

    log(f"Loading DLL: {dll_path}")
    try:
        _lib = ctypes.CDLL(dll_path)
    except OSError as e:
        log(f"CDLL failed: {e}")
        ld_info = os.environ.get("LD_LIBRARY_PATH", "N/A") if platform.system() != "Windows" else "N/A"
        log(f"Extra dirs searched: {extra_dirs if platform.system() == 'Windows' else ld_info}")
        raise

    # Function signatures
    _lib.quantfunc_version.restype = ctypes.c_char_p
    _lib.quantfunc_last_error.restype = ctypes.c_char_p
    _lib.quantfunc_set_log_level.restype = None
    _lib.quantfunc_set_log_level.argtypes = [ctypes.c_int]
    _lib.quantfunc_create.restype = ctypes.c_int
    _lib.quantfunc_create.argtypes = [ctypes.POINTER(InitParams), ctypes.POINTER(PIPE_PTR)]
    _lib.quantfunc_destroy.restype = None
    _lib.quantfunc_destroy.argtypes = [PIPE_PTR]
    _lib.quantfunc_text_to_image.restype = ctypes.c_int
    _lib.quantfunc_text_to_image.argtypes = [PIPE_PTR, ctypes.POINTER(T2IParams), ctypes.POINTER(IMG_PTR)]
    _lib.quantfunc_image_to_image.restype = ctypes.c_int
    _lib.quantfunc_image_to_image.argtypes = [PIPE_PTR, ctypes.POINTER(I2IParams), ctypes.POINTER(IMG_PTR)]
    _lib.quantfunc_export.restype = ctypes.c_int
    _lib.quantfunc_export.argtypes = [ctypes.POINTER(ExportParams)]
    _lib.quantfunc_image_width.restype = ctypes.c_int
    _lib.quantfunc_image_width.argtypes = [IMG_PTR]
    _lib.quantfunc_image_height.restype = ctypes.c_int
    _lib.quantfunc_image_height.argtypes = [IMG_PTR]
    _lib.quantfunc_image_float_data.restype = ctypes.POINTER(ctypes.c_float)
    _lib.quantfunc_image_float_data.argtypes = [IMG_PTR]
    _lib.quantfunc_image_data.restype = ctypes.POINTER(ctypes.c_uint8)
    _lib.quantfunc_image_data.argtypes = [IMG_PTR]
    # Multi-image / RGBA accessors (QwenImageLayered → N RGBA layers). Guarded so an
    # older .so without these symbols still works (falls back to single RGB N=1/C=3).
    if hasattr(_lib, "quantfunc_image_count"):
        _lib.quantfunc_image_count.restype = ctypes.c_int
        _lib.quantfunc_image_count.argtypes = [IMG_PTR]
    if hasattr(_lib, "quantfunc_image_channels"):
        _lib.quantfunc_image_channels.restype = ctypes.c_int
        _lib.quantfunc_image_channels.argtypes = [IMG_PTR]
    _lib.quantfunc_image_destroy.restype = None
    _lib.quantfunc_image_destroy.argtypes = [IMG_PTR]

    # #344 — LTX-2 text-to-video (+ audio). Optional: only present in engines that
    # ship the video API (wrapped in try so older DLLs still load for t2i).
    global _HAS_VIDEO
    _HAS_VIDEO = False
    try:
        _lib.quantfunc_text_to_video.restype = ctypes.c_int
        _lib.quantfunc_text_to_video.argtypes = [PIPE_PTR, ctypes.POINTER(T2IParams), ctypes.POINTER(VID_PTR)]
        _lib.quantfunc_video_num_frames.restype = ctypes.c_int
        _lib.quantfunc_video_num_frames.argtypes = [VID_PTR]
        _lib.quantfunc_video_fps.restype = ctypes.c_float
        _lib.quantfunc_video_fps.argtypes = [VID_PTR]
        _lib.quantfunc_video_frame.restype = IMG_PTR    # borrowed (owned by the video)
        _lib.quantfunc_video_frame.argtypes = [VID_PTR, ctypes.c_int]
        _lib.quantfunc_video_audio_data.restype = ctypes.POINTER(ctypes.c_float)
        _lib.quantfunc_video_audio_data.argtypes = [VID_PTR]
        _lib.quantfunc_video_audio_num_samples.restype = ctypes.c_int
        _lib.quantfunc_video_audio_num_samples.argtypes = [VID_PTR]
        _lib.quantfunc_video_audio_channels.restype = ctypes.c_int
        _lib.quantfunc_video_audio_channels.argtypes = [VID_PTR]
        _lib.quantfunc_video_sample_rate.restype = ctypes.c_int
        _lib.quantfunc_video_sample_rate.argtypes = [VID_PTR]
        _lib.quantfunc_video_destroy.restype = None
        _lib.quantfunc_video_destroy.argtypes = [VID_PTR]
        _HAS_VIDEO = True
    except AttributeError:
        log("video API not present in this DLL (t2i-only engine)")

    # #329/#330 — image-to-video (first ref image = first-frame condition). Reuses
    # the i2i param struct; num_frames/fps ride in options_json. Guarded SEPARATELY
    # from _HAS_VIDEO so a t2v-only engine still enables t2v (i2v is a superset add).
    global _HAS_I2V
    _HAS_I2V = False
    if _HAS_VIDEO and hasattr(_lib, "quantfunc_image_to_video"):
        _lib.quantfunc_image_to_video.restype = ctypes.c_int
        _lib.quantfunc_image_to_video.argtypes = [PIPE_PTR, ctypes.POINTER(I2IParams), ctypes.POINTER(VID_PTR)]
        _HAS_I2V = True

    # Optional: quantfunc_set_api_key (may not exist in older DLLs)
    try:
        _lib.quantfunc_set_api_key.restype = ctypes.c_int
        _lib.quantfunc_set_api_key.argtypes = [PIPE_PTR, ctypes.c_char_p]
    except AttributeError:
        pass  # Old DLL without set_api_key

    # Optional: quantfunc_unload (may not exist in older DLLs)
    try:
        _lib.quantfunc_unload.restype = ctypes.c_int
        _lib.quantfunc_unload.argtypes = [PIPE_PTR]
    except AttributeError:
        pass  # Old DLL without unload

    # Optional: quantfunc_unload_sync (blocks until VRAM actually released)
    try:
        _lib.quantfunc_unload_sync.restype = ctypes.c_int
        _lib.quantfunc_unload_sync.argtypes = [PIPE_PTR]
    except AttributeError:
        pass  # Old DLL without unload_sync

    # Optional: quantfunc_release_backup (added for ComfyUI gpu+cpu unload mode)
    try:
        _lib.quantfunc_release_backup.restype = ctypes.c_int
        _lib.quantfunc_release_backup.argtypes = [PIPE_PTR]
    except AttributeError:
        pass  # Old DLL without release_backup

    version = _lib.quantfunc_version().decode()
    log(f"Loaded DLL version {version} from {dll_path}")


def _get_error():
    """Get last error string from DLL."""
    err = _lib.quantfunc_last_error()
    return err.decode() if err else "unknown error"


# ============================================================================
# Pipeline state
# ============================================================================

_pipelines = {}       # Dict[cache_key: str, PIPE_PTR]
_cancel_flag = threading.Event()

# Keep reference to current callback to prevent GC during ctypes call
_current_cb = None


def _get_pipeline(msg, req_id):
    """Look up pipeline by cache_key. Send error if missing; return None.
    Fallback: if cache_key is missing and exactly one pipeline loaded, use it
    (backwards-compat for older callers that don't pass cache_key)."""
    key = msg.get("cache_key")
    if key is None:
        if len(_pipelines) == 1:
            return next(iter(_pipelines.values()))
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": -1,
                   "error_message": "cache_key required (loaded: {})".format(
                       list(_pipelines.keys()))})
        return None
    pipe = _pipelines.get(key)
    if pipe is None:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": -1,
                   "error_message": "No pipeline for cache_key={!r} (loaded: {})".format(
                       key, list(_pipelines.keys()))})
        return None
    return pipe


def _make_progress_cb(req_id):
    """Create a progress callback that sends progress messages and checks cancel."""
    global _current_cb

    @PROGRESS_CB
    def cb(step, total, user_data):
        send_json({"type": "progress", "req_id": req_id, "step": step, "total": total})
        if _cancel_flag.is_set():
            return 1  # cancel
        return 0

    _current_cb = cb  # prevent GC
    return cb


_current_preview_cb = None


def _make_latent_preview_cb(req_id):
    """Create a latent-preview callback that ships each step's small RGB preview
    (engine-decoded latent2rgb) to the parent as base64 JSON. The parent feeds
    it to ComfyUI's live-preview channel. Never aborts generation on failure."""
    global _current_preview_cb
    import base64

    @LATENT_PREVIEW_CB
    def cb(step, total, width, height, rgb_ptr, user_data):
        try:
            n = int(width) * int(height) * 3
            if n > 0 and rgb_ptr:
                buf = ctypes.string_at(rgb_ptr, n)
                send_json({"type": "preview", "req_id": req_id,
                           "step": int(step), "total": int(total),
                           "width": int(width), "height": int(height),
                           "rgb_b64": base64.b64encode(buf).decode("ascii")})
        except Exception:
            pass  # a preview failure must never affect generation
        if _cancel_flag.is_set():
            return 1  # cancel
        return 0

    _current_preview_cb = cb  # prevent GC
    return cb


def _extract_and_send_image(img_ptr, req_id):
    """Extract image data from C API and hand off to parent.

    For Linux: write raw uint8 RGB bytes to /dev/shm (tmpfs RAM disk) and
    send the file path to parent in JSON. Saves a ~3MB bytes() allocation
    on this side and a ~3MB stdout pipe transfer — both walk the same RAM
    but the pipe requires multiple read() syscalls in parent, while /dev/shm
    is a single open+read+unlink round-trip. Typically ~15-20 ms savings.

    Falls back to legacy stdout-binary when /dev/shm isn't available."""
    w = _lib.quantfunc_image_width(img_ptr)
    h = _lib.quantfunc_image_height(img_ptr)
    # Multi-image / RGBA results (QwenImageLayered → N RGBA layers). Guarded so an
    # older .so without count/channels falls back to the legacy single RGB image.
    n_imgs = _lib.quantfunc_image_count(img_ptr) if hasattr(_lib, "quantfunc_image_count") else 1
    n_ch = _lib.quantfunc_image_channels(img_ptr) if hasattr(_lib, "quantfunc_image_channels") else 3

    if n_imgs <= 1 and n_ch == 3:
        # Legacy fast path — byte-identical to before: image 0, uint8 RGB.
        uint8_ptr = _lib.quantfunc_image_data(img_ptr)
        n_bytes = h * w * 3
        payload = bytes(memoryview(
            ctypes.cast(uint8_ptr, ctypes.POINTER(ctypes.c_uint8 * n_bytes))[0]))
        img_fmt = "rgb_uint8"
    else:
        # Multi-layer / RGBA: float32 [N,H,W,C] in [0,1], all N images stacked, sent
        # raw — no conversion here (the parent has numpy; the worker stays numpy-free).
        fptr = _lib.quantfunc_image_float_data(img_ptr)
        n_floats = n_imgs * h * w * n_ch
        n_bytes = n_floats * 4
        payload = bytes(memoryview(
            ctypes.cast(fptr, ctypes.POINTER(ctypes.c_float * n_floats))[0]))
        img_fmt = "layers_f32"

    meta = {
        "type": "result", "req_id": req_id, "status": "ok",
        "image_width": w, "image_height": h, "image_bytes": n_bytes,
        "image_count": n_imgs, "image_channels": n_ch, "image_format": img_fmt,
    }

    use_shm = os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)
    if use_shm:
        shm_path = f"/dev/shm/qf_out_{os.getpid()}_{req_id}.raw"
        with open(shm_path, "wb") as f:
            f.write(payload)
        _lib.quantfunc_image_destroy(img_ptr)
        meta["image_shm_path"] = shm_path
        send_json(meta)
        return

    # Legacy fallback (Windows / no /dev/shm): stdout binary
    _lib.quantfunc_image_destroy(img_ptr)
    send_json(meta)
    send_binary(payload)


# ============================================================================
# Command handlers
# ============================================================================

QUANTFUNC_OK = 0
QUANTFUNC_ERROR_CANCELLED = 6


def handle_create(msg):
    req_id = msg["req_id"]
    key = msg.get("cache_key", "")

    # Reuse existing pipeline for this key (idempotent)
    if key and key in _pipelines:
        send_json({"type": "result", "req_id": req_id, "status": "ok",
                   "cache_key": key, "reused": True})
        return

    params = InitParams()
    params.model_dir = msg["model_dir"].encode() if msg.get("model_dir") else None
    params.transformer_path = msg["transformer_path"].encode() if msg.get("transformer_path") else None
    params.vae_path = msg["vae_path"].encode() if msg.get("vae_path") else None
    params.text_encoder_path = msg["text_encoder_path"].encode() if msg.get("text_encoder_path") else None
    params.tokenizer_path = msg["tokenizer_path"].encode() if msg.get("tokenizer_path") else None
    params.scheduler_config = msg["scheduler_config"].encode() if msg.get("scheduler_config") else None
    params.model_backend = msg.get("model_backend", "svdq").encode()
    params.device_idx = msg.get("device_idx", 0)
    params.config_json = msg["config_json"].encode() if msg.get("config_json") else None

    pipe = PIPE_PTR()
    status = _lib.quantfunc_create(ctypes.byref(params), ctypes.byref(pipe))
    if status != QUANTFUNC_OK:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": status, "error_message": _get_error()})
        return

    _pipelines[key] = pipe
    send_json({"type": "result", "req_id": req_id, "status": "ok", "cache_key": key})


def handle_text_to_image(msg):
    req_id = msg["req_id"]
    pipe = _get_pipeline(msg, req_id)
    if pipe is None:
        return

    _cancel_flag.clear()
    cb = _make_progress_cb(req_id)

    t2i = T2IParams()
    t2i.prompt = msg["prompt"].encode()
    t2i.height = msg.get("height", 1024)
    t2i.width = msg.get("width", 1024)
    t2i.num_steps = msg.get("num_steps", 8)
    t2i.guidance_scale = msg.get("guidance_scale", 0.0)
    t2i.seed = msg.get("seed", 0)
    t2i.options_json = msg["options_json"].encode() if msg.get("options_json") else None
    t2i.progress_callback = cb
    t2i.callback_user_data = None
    if msg.get("latent_preview"):
        pcb = _make_latent_preview_cb(req_id)
        t2i.latent_preview_callback = pcb
        t2i.latent_preview_user_data = None

    img = IMG_PTR()
    status = _lib.quantfunc_text_to_image(pipe, ctypes.byref(t2i), ctypes.byref(img))

    if status == QUANTFUNC_ERROR_CANCELLED:
        send_json({"type": "result", "req_id": req_id, "status": "cancelled"})
        return
    if status != QUANTFUNC_OK:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": status, "error_message": _get_error()})
        return

    _extract_and_send_image(img, req_id)


def handle_image_to_image(msg):
    req_id = msg["req_id"]
    pipe = _get_pipeline(msg, req_id)
    if pipe is None:
        return

    _cancel_flag.clear()
    cb = _make_progress_cb(req_id)

    ref_paths = msg.get("ref_image_paths", [])
    num_refs = len(ref_paths)
    ref_encoded = [p.encode() for p in ref_paths]
    ref_arr = (ctypes.c_char_p * num_refs)(*ref_encoded) if num_refs > 0 else None

    i2i = I2IParams()
    i2i.prompt = msg["prompt"].encode()
    i2i.ref_image_paths = ref_arr
    i2i.num_ref_images = num_refs
    i2i.height = msg.get("height", 1024)
    i2i.width = msg.get("width", 1024)
    i2i.num_steps = msg.get("num_steps", 4)
    i2i.true_cfg_scale = msg.get("true_cfg_scale", 1.0)
    neg = msg.get("negative_prompt")
    i2i.negative_prompt = neg.encode() if neg else None
    i2i.seed = msg.get("seed", 0)
    i2i.options_json = msg["options_json"].encode() if msg.get("options_json") else None
    i2i.progress_callback = cb
    i2i.callback_user_data = None
    if msg.get("latent_preview"):
        pcb = _make_latent_preview_cb(req_id)
        i2i.latent_preview_callback = pcb
        i2i.latent_preview_user_data = None
    # Inpaint plumbing — node serializes the MASK to a temp PNG, passes path.
    mp = msg.get("mask_path")
    i2i.mask_path = mp.encode() if mp else None
    i2i.mask_strength = float(msg.get("mask_strength", 1.0))
    i2i.mask_grow = int(msg.get("mask_grow", 6))
    i2i.mask_blur = float(msg.get("mask_blur", 0.0))
    i2i.mask_no_snap = int(bool(msg.get("mask_no_snap", False)))

    img = IMG_PTR()
    status = _lib.quantfunc_image_to_image(pipe, ctypes.byref(i2i), ctypes.byref(img))

    if status == QUANTFUNC_ERROR_CANCELLED:
        send_json({"type": "result", "req_id": req_id, "status": "cancelled"})
        return
    if status != QUANTFUNC_OK:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": status, "error_message": _get_error()})
        return

    _extract_and_send_image(img, req_id)


def _extract_and_send_video(vid, req_id):
    """#344 — extract the LTX-2 video (frames + audio) and hand to the parent.

    Frames: concatenated raw uint8 RGB (N*H*W*3) written to /dev/shm; the parent
    reshapes to [N,H,W,3]. Audio (if any): host FP32 PLANAR/channel-major
    (channels*num_samples) written to /dev/shm; the parent reshapes to [C,N]. Both
    use the same shm round-trip as the image path. The video handle (and its frames'
    GPU/CPU buffers) are destroyed here."""
    import os
    n = _lib.quantfunc_video_num_frames(vid)
    fps = float(_lib.quantfunc_video_fps(vid))
    use_shm = os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)
    base = ("/dev/shm/" if use_shm else "/tmp/") + f"qf_vid_{os.getpid()}_{req_id}"

    # Frames → one contiguous uint8 RGB blob [N,H,W,3].
    w = h = 0
    frame_path = base + "_frames.raw"
    # O_NOFOLLOW defeats a symlink planted at the predictable /dev/shm path (refuses
    # to follow a symlink final component); O_TRUNC tolerates a stale regular file.
    _ffd = os.open(frame_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(_ffd, "wb") as f:
        for i in range(n):
            img = _lib.quantfunc_video_frame(vid, i)   # borrowed
            if not img:
                continue
            fw = _lib.quantfunc_image_width(img)
            fh = _lib.quantfunc_image_height(img)
            w, h = fw, fh
            nb = fh * fw * 3
            u8 = _lib.quantfunc_image_data(img)
            view = ctypes.cast(u8, ctypes.POINTER(ctypes.c_uint8 * nb))[0]
            f.write(bytes(memoryview(view)))

    result = {"type": "result", "req_id": req_id, "status": "ok",
              "num_frames": n, "width": w, "height": h, "fps": fps,
              "frame_format": "rgb_uint8", "frame_shm_path": frame_path,
              "audio": None}

    # Audio (LTX-2 AV checkpoints only; NULL for video-only).
    if _HAS_VIDEO:
        a_ptr = _lib.quantfunc_video_audio_data(vid)
        ns = _lib.quantfunc_video_audio_num_samples(vid)
        ch = _lib.quantfunc_video_audio_channels(vid)
        sr = _lib.quantfunc_video_sample_rate(vid)
        if a_ptr and ns > 0 and ch > 0:
            nfloat = ch * ns
            a_path = base + "_audio.raw"
            view = ctypes.cast(a_ptr, ctypes.POINTER(ctypes.c_float * nfloat))[0]
            _afd = os.open(a_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(_afd, "wb") as f:
                f.write(bytes(memoryview(view)))
            result["audio"] = {"num_samples": ns, "channels": ch,
                               "sample_rate": sr, "shm_path": a_path}

    _lib.quantfunc_video_destroy(vid)
    send_json(result)


def handle_text_to_video(msg):
    """#344 — LTX-2 text-to-video (+ audio). Mirrors handle_text_to_image: same
    T2IParams (num_frames/fps come through options_json), but dispatches the video
    C-API and returns frames + audio."""
    req_id = msg["req_id"]
    if not _HAS_VIDEO:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": -1,
                   "error_message": "This QuantFunc engine build has no video API "
                                    "(needs the LTX-2 t2v + audio C-API, #344+)."})
        return
    pipe = _get_pipeline(msg, req_id)
    if pipe is None:
        return
    _cancel_flag.clear()
    cb = _make_progress_cb(req_id)

    t2i = T2IParams()
    t2i.prompt = msg["prompt"].encode()
    t2i.height = msg.get("height", 512)
    t2i.width = msg.get("width", 512)
    t2i.num_steps = msg.get("num_steps", 30)
    t2i.guidance_scale = float(msg.get("guidance_scale", 4.0))
    t2i.seed = msg.get("seed", 0)
    t2i.options_json = msg["options_json"].encode() if msg.get("options_json") else None
    t2i.progress_callback = cb
    t2i.callback_user_data = None
    t2i.latent_preview_callback = LATENT_PREVIEW_CB()  # NULL
    t2i.latent_preview_user_data = None

    vid = VID_PTR()
    status = _lib.quantfunc_text_to_video(pipe, ctypes.byref(t2i), ctypes.byref(vid))
    if status == QUANTFUNC_ERROR_CANCELLED:
        send_json({"type": "result", "req_id": req_id, "status": "cancelled"})
        return
    if status != QUANTFUNC_OK:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": status, "error_message": _get_error()})
        return

    _extract_and_send_video(vid, req_id)


def handle_image_to_video(msg):
    """#329/#330 — image-to-video (Wan I2V / LTX i2v). Mirrors handle_text_to_video
    but takes the i2i param struct: the FIRST ref image is the first-frame
    condition. num_frames/fps ride in options_json; guidance rides in
    true_cfg_scale (the i2i struct has no guidance_scale field — the engine's
    quantfunc_image_to_video reads guidance from true_cfg_scale)."""
    req_id = msg["req_id"]
    if not _HAS_VIDEO or not _HAS_I2V:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": -1,
                   "error_message": "This QuantFunc engine build has no image-to-video "
                                    "API (needs quantfunc_image_to_video, #329/#330+)."})
        return
    pipe = _get_pipeline(msg, req_id)
    if pipe is None:
        return
    _cancel_flag.clear()
    cb = _make_progress_cb(req_id)

    ref_paths = msg.get("ref_image_paths", [])
    num_refs = len(ref_paths)
    ref_encoded = [p.encode() for p in ref_paths]
    ref_arr = (ctypes.c_char_p * num_refs)(*ref_encoded) if num_refs > 0 else None

    i2i = I2IParams()
    i2i.prompt = msg["prompt"].encode()
    i2i.ref_image_paths = ref_arr
    i2i.num_ref_images = num_refs
    i2i.height = msg.get("height", 512)
    i2i.width = msg.get("width", 512)
    i2i.num_steps = msg.get("num_steps", 30)
    # Video CFG rides in true_cfg_scale (see docstring). >0 → engine uses it.
    i2i.true_cfg_scale = float(msg.get("guidance_scale", 5.0))
    neg = msg.get("negative_prompt")
    i2i.negative_prompt = neg.encode() if neg else None
    i2i.seed = msg.get("seed", 0)
    i2i.options_json = msg["options_json"].encode() if msg.get("options_json") else None
    i2i.progress_callback = cb
    i2i.callback_user_data = None
    # No inpaint / latent-preview / brush conditioning on the video path.
    i2i.mask_path = None
    i2i.latent_preview_callback = LATENT_PREVIEW_CB()  # NULL
    i2i.latent_preview_user_data = None

    vid = VID_PTR()
    status = _lib.quantfunc_image_to_video(pipe, ctypes.byref(i2i), ctypes.byref(vid))
    if status == QUANTFUNC_ERROR_CANCELLED:
        send_json({"type": "result", "req_id": req_id, "status": "cancelled"})
        return
    if status != QUANTFUNC_OK:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": status, "error_message": _get_error()})
        return

    _extract_and_send_video(vid, req_id)


def handle_export(msg):
    req_id = msg["req_id"]

    params = ExportParams()
    params.model_dir = msg["model_dir"].encode() if msg.get("model_dir") else None
    params.export_path = msg["export_path"].encode() if msg.get("export_path") else None
    params.transformer_path = msg["transformer_path"].encode() if msg.get("transformer_path") else None
    params.model_backend = msg.get("model_backend", "svdq").encode()
    params.device_idx = msg.get("device_idx", 0)
    params.config_json = msg["config_json"].encode() if msg.get("config_json") else None

    status = _lib.quantfunc_export(ctypes.byref(params))
    if status != QUANTFUNC_OK:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": status, "error_message": _get_error()})
        return
    send_json({"type": "result", "req_id": req_id, "status": "ok"})


def handle_set_api_key(msg):
    req_id = msg["req_id"]
    pipe = _get_pipeline(msg, req_id)
    if pipe is None:
        return

    api_key = msg.get("api_key", "")
    if not hasattr(_lib, "quantfunc_set_api_key"):
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": -1, "error_message": "DLL does not support set_api_key"})
        return

    status = _lib.quantfunc_set_api_key(pipe, api_key.encode() if api_key else None)
    if status != QUANTFUNC_OK:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": status, "error_message": _get_error()})
        return
    send_json({"type": "result", "req_id": req_id, "status": "ok"})


def handle_unload(msg):
    """Offload pipeline(s) from GPU to CPU.
    If cache_key given, unload that one; otherwise unload all.

    msg["sync"] = True → use quantfunc_unload_sync (blocks until VRAM is
    actually freed; skips the 3-second grace period). Used for cross-
    pipeline eviction and ComfyUI's free_memory hook, where the caller
    is about to allocate VRAM and can't race with our async offload.

    Default (sync=False) uses fire-and-forget quantfunc_unload — the
    background thread has a 3-second grace period that a new generate()
    on the same pipeline can cancel to keep the model resident."""
    req_id = msg["req_id"]

    if not hasattr(_lib, "quantfunc_unload"):
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": -1, "error_message": "DLL does not support unload"})
        return

    sync = bool(msg.get("sync", False))
    fn = _lib.quantfunc_unload_sync if (sync and hasattr(_lib, "quantfunc_unload_sync")) \
                                    else _lib.quantfunc_unload

    key = msg.get("cache_key")
    if key is not None:
        pipe = _pipelines.get(key)
        if pipe is None:
            send_json({"type": "result", "req_id": req_id, "status": "error",
                       "error_code": -1,
                       "error_message": "No pipeline for cache_key={!r}".format(key)})
            return
        targets = [(key, pipe)]
    else:
        targets = list(_pipelines.items())

    for k, pipe in targets:
        status = fn(pipe)
        if status != QUANTFUNC_OK:
            send_json({"type": "result", "req_id": req_id, "status": "error",
                       "error_code": status,
                       "error_message": "unload({}) failed: {}".format(k, _get_error())})
            return
    send_json({"type": "result", "req_id": req_id, "status": "ok",
               "unloaded": [k for k, _ in targets], "sync": sync})


def handle_release_backup(msg):
    """Release mmap-backed offload_backup physical pages (keeps backing files,
    RAM returned to OS). Used by unload_mode=gpu+cpu to free the ~15 GB CPU
    backup while preserving fast reload."""
    req_id = msg["req_id"]
    if not hasattr(_lib, "quantfunc_release_backup"):
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": -1,
                   "error_message": "DLL does not support release_backup"})
        return
    key = msg.get("cache_key")
    if key is not None:
        pipe = _pipelines.get(key)
        if pipe is None:
            send_json({"type": "result", "req_id": req_id, "status": "error",
                       "error_code": -1,
                       "error_message": "No pipeline for cache_key={!r}".format(key)})
            return
        targets = [(key, pipe)]
    else:
        targets = list(_pipelines.items())
    for k, pipe in targets:
        status = _lib.quantfunc_release_backup(pipe)
        if status != QUANTFUNC_OK:
            send_json({"type": "result", "req_id": req_id, "status": "error",
                       "error_code": status,
                       "error_message": "release_backup({}) failed: {}".format(k, _get_error())})
            return
    send_json({"type": "result", "req_id": req_id, "status": "ok",
               "released": [k for k, _ in targets]})


def handle_destroy(msg):
    """Destroy pipeline(s).
    If cache_key given, destroy that one; otherwise destroy all."""
    req_id = msg["req_id"]
    key = msg.get("cache_key")
    if key is not None:
        pipe = _pipelines.pop(key, None)
        if pipe is not None:
            _lib.quantfunc_destroy(pipe)
        send_json({"type": "result", "req_id": req_id, "status": "ok",
                   "destroyed": [key] if pipe is not None else []})
        return
    destroyed = list(_pipelines.keys())
    for _, pipe in _pipelines.items():
        _lib.quantfunc_destroy(pipe)
    _pipelines.clear()
    send_json({"type": "result", "req_id": req_id, "status": "ok",
               "destroyed": destroyed})


# ============================================================================
# Stdin reader thread (handles cancel commands out-of-band)
# ============================================================================

_command_queue = queue.Queue()


def _stdin_reader():
    """Background thread: reads commands from stdin, dispatches cancel immediately."""
    while True:
        try:
            cmd = read_command()
            if cmd is None:
                # Parent process died (EOF on stdin)
                _command_queue.put({"cmd": "shutdown", "req_id": 0})
                break
            if cmd.get("cmd") == "cancel":
                _cancel_flag.set()
            else:
                _command_queue.put(cmd)
        except Exception as e:
            log(f"stdin reader error: {e}")
            _command_queue.put({"cmd": "shutdown", "req_id": 0})
            break


# ============================================================================
# Main loop
# ============================================================================

HANDLERS = {
    "create":         handle_create,
    "text_to_image":  handle_text_to_image,
    "image_to_image": handle_image_to_image,
    "text_to_video":  handle_text_to_video,   # #344 LTX-2 t2v + audio
    "image_to_video": handle_image_to_video,  # #329/#330 Wan/LTX i2v (first-frame)
    "export":         handle_export,
    "set_api_key":    handle_set_api_key,
    "unload":         handle_unload,
    "release_backup": handle_release_backup,
    "destroy":        handle_destroy,
}


def _cleanup_and_exit(signum=None, frame=None):
    """Clean up all pipelines and exit. Called on SIGTERM/SIGINT."""
    sig_name = f" (signal {signum})" if signum else ""
    log(f"Cleanup{sig_name}: destroying {len(_pipelines)} pipeline(s)...")
    try:
        if _lib is not None:
            for _, pipe in _pipelines.items():
                _lib.quantfunc_destroy(pipe)
            _pipelines.clear()
    except Exception as e:
        log(f"Cleanup error: {e}")
    # Force exit — don't let CUDA atexit handlers hang
    os._exit(0)


def main():
    import argparse
    import signal
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll-path", required=True, help="Path to quantfunc DLL")
    parser.add_argument("--log-level", type=int, default=2, help="Log level (0=trace, 2=info, 6=off)")
    args = parser.parse_args()

    _init_binary_io()

    # Install signal handlers for graceful shutdown (release GPU before dying)
    signal.signal(signal.SIGTERM, _cleanup_and_exit)
    signal.signal(signal.SIGINT, _cleanup_and_exit)

    # Linux: ask the kernel to send SIGTERM to this worker whenever the parent
    # process (ComfyUI) dies — covers SIGKILL / crash / force-quit that skip
    # parent's atexit handlers. Harmless if preexec_fn already set it.
    if platform.system() == "Linux":
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            PR_SET_PDEATHSIG = 1
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        except Exception as e:
            log(f"prctl(PR_SET_PDEATHSIG) failed: {e}")

    log(f"Starting worker (pid={os.getpid()}, dll={args.dll_path})")
    _load_dll(args.dll_path)
    # Worker uses stdout for IPC. Redirect DLL logs to stderr so they don't
    # corrupt the JSON protocol. Parent forwards worker stderr to its console.
    try:
        _lib.quantfunc_set_log_stderr.restype = None
        _lib.quantfunc_set_log_stderr.argtypes = [ctypes.c_int]
        _lib.quantfunc_set_log_stderr(args.log_level)
        log("DLL logs redirected to stderr")
    except AttributeError:
        # Old DLL without set_log_stderr — fall back to disabling logs
        _lib.quantfunc_set_log_level(6)
        log("DLL logs disabled (old DLL without stderr redirect)")

    # Start stdin reader thread
    reader = threading.Thread(target=_stdin_reader, daemon=True)
    reader.start()

    # Send ready signal
    send_json({"type": "ready", "version": _lib.quantfunc_version().decode()})

    # Main command loop
    while True:
        try:
            msg = _command_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        cmd = msg.get("cmd", "")
        req_id = msg.get("req_id", 0)

        if cmd == "shutdown":
            log(f"Shutting down ({len(_pipelines)} pipeline(s))")
            for _, pipe in _pipelines.items():
                _lib.quantfunc_destroy(pipe)
            _pipelines.clear()
            break

        if cmd == "ping":
            send_json({"type": "result", "req_id": req_id, "status": "pong"})
            continue

        handler = HANDLERS.get(cmd)
        if handler is None:
            send_json({"type": "result", "req_id": req_id, "status": "error",
                       "error_code": -1, "error_message": f"Unknown command: {cmd}"})
            continue

        try:
            handler(msg)
        except Exception as e:
            log(f"Handler error: {traceback.format_exc()}")
            send_json({"type": "result", "req_id": req_id, "status": "error",
                       "error_code": -1, "error_message": str(e)[:500]})

    log("Worker exited")


if __name__ == "__main__":
    main()
