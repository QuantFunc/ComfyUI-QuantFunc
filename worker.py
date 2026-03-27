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

PIPE_PTR = ctypes.POINTER(_Pipeline)
IMG_PTR = ctypes.POINTER(_Image)
PROGRESS_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)

class InitParams(ctypes.Structure):
    _fields_ = [
        ("model_dir",        ctypes.c_char_p),
        ("transformer_path", ctypes.c_char_p),
        ("vae_path",         ctypes.c_char_p),
        ("text_encoder_path",ctypes.c_char_p),
        ("tokenizer_path",   ctypes.c_char_p),
        ("scheduler_config", ctypes.c_char_p),
        ("model_backend",    ctypes.c_char_p),
        ("svdq_precision",   ctypes.c_char_p),
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
    ]

class ExportParams(ctypes.Structure):
    _fields_ = [
        ("model_dir",        ctypes.c_char_p),
        ("export_path",      ctypes.c_char_p),
        ("transformer_path", ctypes.c_char_p),
        ("model_backend",    ctypes.c_char_p),
        ("svdq_precision",   ctypes.c_char_p),
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
        cudnn_base = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                  "NVIDIA", "CUDNN")
        if os.path.isdir(cudnn_base):
            for ver in sorted(os.listdir(cudnn_base), reverse=True):
                # cuDNN 9.x puts DLLs in bin/<cuda_ver>/x64/
                ver_dir = os.path.join(cudnn_base, ver, "bin")
                if os.path.isdir(ver_dir):
                    for sub in os.listdir(ver_dir):
                        x64 = os.path.join(ver_dir, sub, "x64")
                        if os.path.isdir(x64):
                            extra_dirs.append(x64)
                            break  # use first (highest) cuda version subdir
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

    log(f"Loading DLL: {dll_path}")
    try:
        _lib = ctypes.CDLL(dll_path)
    except OSError as e:
        log(f"CDLL failed: {e}")
        log(f"Extra dirs searched: {extra_dirs if platform.system() == 'Windows' else 'N/A'}")
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
    _lib.quantfunc_image_destroy.restype = None
    _lib.quantfunc_image_destroy.argtypes = [IMG_PTR]

    # Optional: quantfunc_set_api_key (may not exist in older DLLs)
    try:
        _lib.quantfunc_set_api_key.restype = ctypes.c_int
        _lib.quantfunc_set_api_key.argtypes = [PIPE_PTR, ctypes.c_char_p]
    except AttributeError:
        pass  # Old DLL without set_api_key

    version = _lib.quantfunc_version().decode()
    log(f"Loaded DLL version {version} from {dll_path}")


def _get_error():
    """Get last error string from DLL."""
    err = _lib.quantfunc_last_error()
    return err.decode() if err else "unknown error"


# ============================================================================
# Pipeline state
# ============================================================================

_pipeline = None      # PIPE_PTR
_cache_key = None     # str
_cancel_flag = threading.Event()

# Keep reference to current callback to prevent GC during ctypes call
_current_cb = None


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


def _extract_and_send_image(img_ptr, req_id):
    """Extract image data from C API, send as JSON + binary.
    Uses uint8 RGB (3 bytes/pixel) instead of float32 (12 bytes/pixel) for 4x less IPC data."""
    w = _lib.quantfunc_image_width(img_ptr)
    h = _lib.quantfunc_image_height(img_ptr)
    uint8_ptr = _lib.quantfunc_image_data(img_ptr)
    n_bytes = h * w * 3  # uint8 RGB

    buf = (ctypes.c_uint8 * n_bytes).from_address(ctypes.addressof(uint8_ptr.contents))
    raw = bytes(buf)

    _lib.quantfunc_image_destroy(img_ptr)

    send_json({
        "type": "result", "req_id": req_id, "status": "ok",
        "image_width": w, "image_height": h, "image_bytes": n_bytes,
        "image_format": "rgb_uint8",
    })
    send_binary(raw)


# ============================================================================
# Command handlers
# ============================================================================

QUANTFUNC_OK = 0
QUANTFUNC_ERROR_CANCELLED = 6


def handle_create(msg):
    global _pipeline, _cache_key
    req_id = msg["req_id"]

    # Destroy existing pipeline if any
    if _pipeline is not None:
        _lib.quantfunc_destroy(_pipeline)
        _pipeline = None
        _cache_key = None

    params = InitParams()
    params.model_dir = msg["model_dir"].encode() if msg.get("model_dir") else None
    params.transformer_path = msg["transformer_path"].encode() if msg.get("transformer_path") else None
    params.vae_path = msg["vae_path"].encode() if msg.get("vae_path") else None
    params.text_encoder_path = msg["text_encoder_path"].encode() if msg.get("text_encoder_path") else None
    params.tokenizer_path = msg["tokenizer_path"].encode() if msg.get("tokenizer_path") else None
    params.scheduler_config = msg["scheduler_config"].encode() if msg.get("scheduler_config") else None
    params.model_backend = msg.get("model_backend", "svdq").encode()
    params.svdq_precision = msg.get("svdq_precision", "int4").encode()
    params.device_idx = msg.get("device_idx", 0)
    params.config_json = msg["config_json"].encode() if msg.get("config_json") else None

    pipe = PIPE_PTR()
    status = _lib.quantfunc_create(ctypes.byref(params), ctypes.byref(pipe))
    if status != QUANTFUNC_OK:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": status, "error_message": _get_error()})
        return

    _pipeline = pipe
    _cache_key = msg.get("cache_key", "")
    send_json({"type": "result", "req_id": req_id, "status": "ok", "cache_key": _cache_key})


def handle_text_to_image(msg):
    req_id = msg["req_id"]
    if _pipeline is None:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": -1, "error_message": "No pipeline loaded"})
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

    img = IMG_PTR()
    status = _lib.quantfunc_text_to_image(_pipeline, ctypes.byref(t2i), ctypes.byref(img))

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
    if _pipeline is None:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": -1, "error_message": "No pipeline loaded"})
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
    i2i.true_cfg_scale = msg.get("true_cfg_scale", 4.0)
    neg = msg.get("negative_prompt")
    i2i.negative_prompt = neg.encode() if neg else None
    i2i.seed = msg.get("seed", 0)
    i2i.options_json = msg["options_json"].encode() if msg.get("options_json") else None
    i2i.progress_callback = cb
    i2i.callback_user_data = None

    img = IMG_PTR()
    status = _lib.quantfunc_image_to_image(_pipeline, ctypes.byref(i2i), ctypes.byref(img))

    if status == QUANTFUNC_ERROR_CANCELLED:
        send_json({"type": "result", "req_id": req_id, "status": "cancelled"})
        return
    if status != QUANTFUNC_OK:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": status, "error_message": _get_error()})
        return

    _extract_and_send_image(img, req_id)


def handle_export(msg):
    req_id = msg["req_id"]

    params = ExportParams()
    params.model_dir = msg["model_dir"].encode() if msg.get("model_dir") else None
    params.export_path = msg["export_path"].encode() if msg.get("export_path") else None
    params.transformer_path = msg["transformer_path"].encode() if msg.get("transformer_path") else None
    params.model_backend = msg.get("model_backend", "svdq").encode()
    params.svdq_precision = msg.get("svdq_precision", "int4").encode()
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
    if _pipeline is None:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": -1, "error_message": "No pipeline loaded"})
        return

    api_key = msg.get("api_key", "")
    if not hasattr(_lib, "quantfunc_set_api_key"):
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": -1, "error_message": "DLL does not support set_api_key"})
        return

    status = _lib.quantfunc_set_api_key(_pipeline, api_key.encode() if api_key else None)
    if status != QUANTFUNC_OK:
        send_json({"type": "result", "req_id": req_id, "status": "error",
                   "error_code": status, "error_message": _get_error()})
        return
    send_json({"type": "result", "req_id": req_id, "status": "ok"})


def handle_destroy(msg):
    global _pipeline, _cache_key
    req_id = msg["req_id"]
    if _pipeline is not None:
        _lib.quantfunc_destroy(_pipeline)
        _pipeline = None
        _cache_key = None
    send_json({"type": "result", "req_id": req_id, "status": "ok"})


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
    "export":         handle_export,
    "set_api_key":    handle_set_api_key,
    "destroy":        handle_destroy,
}


def _cleanup_and_exit(signum=None, frame=None):
    """Clean up pipeline and exit. Called on SIGTERM/SIGINT."""
    global _pipeline
    sig_name = f" (signal {signum})" if signum else ""
    log(f"Cleanup{sig_name}: destroying pipeline...")
    try:
        if _pipeline is not None and _lib is not None:
            _lib.quantfunc_destroy(_pipeline)
            _pipeline = None
            log("Pipeline destroyed")
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
            log("Shutting down")
            if _pipeline is not None:
                _lib.quantfunc_destroy(_pipeline)
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
