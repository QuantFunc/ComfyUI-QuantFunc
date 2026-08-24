"""qf_cloud_te_node — QuantFunc CLOUD text-encoder LOADER (CLIP-shaped, official-node plug-in).

LOADER FORM (user 2026-08-24 "只做loader形式 对接官方的其他插件"): this node outputs a
CLIP-shaped object that wires into the OFFICIAL MiniMax H3 nodes' `clip` input
(MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo / CLIPTextEncode). The PROMPT and the
reference images stay on the OFFICIAL nodes — this loader carries NO text/image inputs.
When the official node calls `clip.tokenize(prompt, images=...)` /
`tokenize(prompt, minimax_ref_items=...)` + `clip.encode_from_tokens_scheduled(tokens)`,
the shim ships text + reference IMAGES to the remote QuantFunc TE worker through the
engine C-API (`quantfunc_te_cloud_encode`) and returns the conditioning. No local CLIP
weights, no GPU text-encoder.

Auth rides the ENGINE convention (no widget): env QUANTFUNC_API_KEY / QF_API_KEY, else the
bundled keyfile (bin/<plat>/config.json, QF_NATIVE_KEYFILE override); server_url = env
QF_SERVER_URL else the engine default. All upload/signing/crypto lives inside the .so
(design v11 security boundary) — the node only loops one bounded C-API call
(pending → resume via a durable task_id) until terminal or budget.
"""
import ctypes
import hashlib
import json
import os
import tempfile
import time

import numpy as np
import torch

from . import qf_engine

# Optional ComfyUI cancel hook — absent when imported outside ComfyUI (e.g. tests).
try:
    import comfy.model_management as _mm  # type: ignore
except Exception:  # pragma: no cover
    _mm = None

_DEFAULT_SERVER_URL = "https://service.quantfunc.com"
# Worker model ids the cloud service serves for the native lane. Dropdown (user order:
# model_id 要下拉框). Extend as the serving lane deploys more TE workers.
_MODEL_CHOICES = ["qwen3-vl-32b"]
_DTYPE_CHOICES = {"fp32": qf_engine.QF_FP32, "fp8_e4m3": qf_engine.QF_FP8_E4M3}
_PER_CALL_WAIT_MS = 5000   # per-call wait sub-step budget (ms); the shim loops for the overall budget
_POLL_INTERVAL_S = 1.0     # sleep between resume polls (s)


def _read_auth():
    """Engine auth convention (mirrors __init__._read_auth; kept local to avoid a
    package-circular import — the package __init__ imports THIS module)."""
    key = os.environ.get("QUANTFUNC_API_KEY", "") or os.environ.get("QF_API_KEY", "")
    surl = os.environ.get("QF_SERVER_URL", _DEFAULT_SERVER_URL)
    override = os.environ.get(qf_engine._ENV_KEYFILE_OVERRIDE, "").strip()
    keyfile = override or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "bin", qf_engine._BIN_SUBDIR, qf_engine._KEYFILE_BASENAME)
    if not key and os.path.exists(keyfile):
        try:
            c = json.load(open(keyfile))
            key, surl = c.get("api_key", ""), c.get("server_url", surl)
        except Exception:  # noqa: BLE001
            pass
    return key, surl


_RESUME_CACHE_PATH = os.path.join(tempfile.gettempdir(), "qf_cloud_te_resume.json")


def _resume_cache_load() -> dict:
    try:
        return json.load(open(_RESUME_CACHE_PATH))
    except Exception:  # noqa: BLE001 — absent/corrupt cache = empty (worst case: one extra task)
        return {}


def _resume_cache_store(d: dict):
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(_RESUME_CACHE_PATH))
        with os.fdopen(fd, "w") as f:
            json.dump(d, f)
        os.replace(tmp, _RESUME_CACHE_PATH)
    except Exception:  # noqa: BLE001 — cache is an optimization, never fail the encode over it
        pass


def _request_key(model_id, output_dtype, text, ref_paths) -> str:
    """Content-addressed identity of one encode request — the loader-form replacement for the
    removed resume_task_id widget: an IDENTICAL re-run auto-resumes its still-open cloud task
    instead of submitting a second billable one. Ref images hash by CONTENT (paths are temp)."""
    h = hashlib.sha256()
    for part in (model_id, "\0", output_dtype, "\0", text):
        h.update(part.encode("utf-8"))
    for p in ref_paths:
        h.update(b"\0")
        try:
            h.update(open(p, "rb").read())
        except OSError:
            h.update(p.encode("utf-8"))
    return h.hexdigest()


def _last_error(lib) -> str:
    try:
        e = lib.quantfunc_last_error()
        return e.decode("utf-8", "replace") if e else "(no detail)"
    except Exception:
        return "(no detail)"


def _save_ref_images(images) -> list:
    """ComfyUI IMAGE tensors ([B,H,W,C] float[0,1], or a list of them) → temp PNG paths
    (positional order preserved — <Picture i> ordinals follow this order)."""
    from PIL import Image
    paths = []
    if images is None:
        return paths
    if torch.is_tensor(images):
        images = [images]
    for img in images:
        arr = img.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[None, ...]
        for i in range(arr.shape[0]):
            im = (np.clip(arr[i], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            fd, p = tempfile.mkstemp(prefix="qf_cloud_te_ref_", suffix=".png")
            os.close(fd)
            Image.fromarray(im[..., :3]).save(p)
            paths.append(p)
    return paths


def _read_tensor(lib, handle) -> torch.Tensor:
    """Read a quantfunc_tensor_t handle → float32 torch tensor [1, seq, hidden]."""
    ndim = ctypes.c_int32()
    dims = (ctypes.c_int64 * 8)()
    dtype = ctypes.c_int32()
    if lib.quantfunc_tensor_info(handle, ctypes.byref(ndim), dims, ctypes.byref(dtype)) != qf_engine.QUANTFUNC_OK:
        raise RuntimeError("cloud TE: tensor_info failed: " + _last_error(lib))
    shape = [int(dims[i]) for i in range(ndim.value)]
    numel = 1
    for d in shape:
        numel *= d
    elem = 4 if dtype.value == 0 else 1
    raw = (ctypes.c_ubyte * (numel * elem))()
    if lib.quantfunc_tensor_read(handle, raw, len(raw)) != qf_engine.QUANTFUNC_OK:
        raise RuntimeError("cloud TE: tensor_read failed: " + _last_error(lib))
    payload = bytes(raw)
    if dtype.value == 0:  # float32, C-contiguous
        arr = np.frombuffer(payload, dtype=np.float32).reshape(shape).copy()
        return torch.from_numpy(arr)
    # dtype == 1: fp8_e4m3 payload + per-token scales (len = seq = dims[-2]).
    seq = shape[-2] if len(shape) >= 2 else 1
    sc = (ctypes.c_float * seq)()
    if lib.quantfunc_tensor_read_scales(handle, sc, seq) != qf_engine.QUANTFUNC_OK:
        raise RuntimeError("cloud TE: tensor_read_scales failed: " + _last_error(lib))
    scales = torch.from_numpy(np.frombuffer(bytes(sc), dtype=np.float32).copy())
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("cloud TE: fp8_e4m3 output needs torch>=2.1 (float8_e4m3fn); "
                           "use output_dtype='fp32' instead.")
    e4m3 = torch.frombuffer(bytearray(payload), dtype=torch.uint8).view(torch.float8_e4m3fn)
    x = e4m3.float().reshape(shape)                       # decode e4m3 → f32
    return x * scales.reshape(1, seq, 1)                  # de-quantize per token


def _cloud_encode(text, ref_paths, model_id, output_dtype, timeout_seconds, device_idx):
    """One text (+ordered ref image paths) → [1,seq,hidden] fp32 tensor via the engine
    C-API, looping resume→terminal. A pending/timeout raise carries a resumable task_id
    (the content-keyed cache auto-resumes an identical re-run); a terminal failure DROPS
    the cache entry — that task is dead, a re-run submits fresh."""
    lib = qf_engine.load_lib()
    if not hasattr(lib, "quantfunc_te_cloud_encode"):
        raise RuntimeError("This libquantfunc.so has no cloud-TE support — rebuild/update "
                           "the engine (quantfunc_te_cloud_encode absent).")
    api_key, server_url = _read_auth()
    if not api_key:
        raise RuntimeError("cloud TE: no API key — set QUANTFUNC_API_KEY (or bundle "
                           "bin/<plat>/config.json). The loader takes no key widget by design.")

    def _cancel(done, total, user):
        try:
            if _mm is not None and _mm.processing_interrupted():
                return 1
        except Exception:
            pass
        return 0
    cb = qf_engine.QF_TE_CLOUD_PROGRESS_CB(_cancel)

    p = lib.quantfunc_default_te_cloud_params()
    p.server_url = server_url.encode("utf-8")
    p.api_key = api_key.encode("utf-8")
    p.device_idx = int(device_idx)
    p.model_id = model_id.encode("utf-8")
    p.text = text.encode("utf-8")
    c_refs = None
    if ref_paths:
        c_refs = (ctypes.c_char_p * len(ref_paths))(*[s.encode("utf-8") for s in ref_paths])
    p.ref_paths = c_refs if c_refs is not None else ctypes.cast(None, ctypes.POINTER(ctypes.c_char_p))
    p.n_refs = len(ref_paths)
    p.output_dtype = _DTYPE_CHOICES[output_dtype]
    p.wait_ms = _PER_CALL_WAIT_MS
    p.progress_callback = cb
    p.callback_user_data = None

    # Loader-form resume (no widget): an IDENTICAL request re-run resumes its still-open
    # cloud task via a content-keyed local cache instead of submitting a second billable task.
    req_key = _request_key(model_id, output_dtype, text, ref_paths)
    cache = _resume_cache_load()
    cached_tid = cache.get(req_key, "")
    resume_bytes = cached_tid.encode("utf-8") if cached_tid else None
    if resume_bytes:
        p.resume_task_id = resume_bytes  # keep bytes alive via resume_bytes

    out_result = ctypes.c_void_p()
    out_pending = ctypes.c_int(0)
    task_buf = ctypes.create_string_buffer(64)   # Snowflake task_id <= 20 bytes
    deadline = time.time() + float(timeout_seconds)

    def _drop_cache_entry():
        c = _resume_cache_load()
        if c.pop(req_key, None) is not None:
            _resume_cache_store(c)

    try:
        while True:
            rc = lib.quantfunc_te_cloud_encode(
                ctypes.byref(p), ctypes.byref(out_result),
                ctypes.byref(out_pending), task_buf, len(task_buf))
            tid = task_buf.value.decode("utf-8", "replace")
            if tid and tid != cache.get(req_key):
                cache[req_key] = tid
                _resume_cache_store(cache)   # persist BEFORE any raise → a re-run resumes
            if rc != qf_engine.QUANTFUNC_OK:
                _drop_cache_entry()          # terminal failure — resuming it is pointless
                err = _last_error(lib)
                if tid:
                    err += f" (task_id={tid})"
                raise RuntimeError("cloud TE encode failed: " + err)
            if out_pending.value == 0:
                break
            if tid:
                p.resume_task_id = task_buf.value  # keep bytes alive via task_buf
            if _mm is not None:
                _mm.throw_exception_if_processing_interrupted()
            if time.time() > deadline:
                raise RuntimeError(
                    f"cloud TE: timed out after {timeout_seconds}s (task still processing; "
                    f"task_id={tid}). Re-running the SAME workflow auto-resumes this task "
                    f"(content-keyed local cache) — no second billable submit.")
            time.sleep(_POLL_INTERVAL_S)
        if not out_result:
            _drop_cache_entry()   # terminal (same out_pending==0 edge as success) — entry is dead
            raise RuntimeError("cloud TE: terminal state but no result tensor: " + _last_error(lib))
        _drop_cache_entry()                  # success — the task is consumed
        return _read_tensor(lib, out_result)
    finally:
        if out_result:
            try:
                lib.quantfunc_tensor_destroy(out_result)
            except Exception:
                pass
        for pth in ref_paths:
            try:
                os.remove(pth)
            except OSError:
                pass


class QFCloudTEClip:
    """CLIP-shaped shim: the official nodes call tokenize(...) + encode_from_tokens_scheduled(...)
    on it exactly as on a local comfy CLIP; the encode runs on the cloud TE worker.

    Supported official callers (verified against comfy_extras/nodes_minimax_h3.py + CLIPTextEncode):
    - tokenize(prompt)                                → text-only encode
    - tokenize(prompt, images=[...])                  → i2v keyframe images ride to the worker
    - tokenize(prompt, minimax_ref_items=[...])       → r2v <Picture i> IMAGE refs ride to the worker;
                                                        video/audio ref items REFUSE LOUD (the cloud
                                                        worker takes image refs — use the local CLIP
                                                        for video/audio references)
    """

    def __init__(self, model_id, output_dtype, timeout_seconds, device_idx):
        self._model_id = model_id
        self._output_dtype = output_dtype
        self._timeout = timeout_seconds
        self._device_idx = device_idx

    # ── comfy CLIP interface (the subset the official nodes use) ──
    def tokenize(self, text, return_word_ids=False, **kwargs):
        images = kwargs.get("images", None)
        ref_items = kwargs.get("minimax_ref_items", None)
        ref_tensors = []
        if images is not None:
            ref_tensors.extend(images if isinstance(images, (list, tuple)) else [images])
        if ref_items:
            for it in ref_items:
                t = it.get("type")
                if t == "image":
                    ref_tensors.append(it["data"])
                else:
                    raise RuntimeError(
                        f"QuantFunc cloud TE: reference type '{t}' is not cloud-encodable "
                        f"(the worker takes IMAGE refs only) — use the local CLIP loader for "
                        f"video/audio references.")
        return {"text": text, "ref_tensors": ref_tensors}

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict=None, show_pbar=True):
        ref_paths = _save_ref_images(tokens.get("ref_tensors") or None)
        emb = _cloud_encode(tokens["text"], ref_paths, self._model_id,
                            self._output_dtype, self._timeout, self._device_idx)
        meta = dict(add_dict or {})
        return [[emb, meta]]

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        cond = self.encode_from_tokens_scheduled(tokens)[0][0]
        if return_dict:
            return {"cond": cond, "pooled_output": None}
        if return_pooled:
            return cond, None
        return cond

    def clone(self, *_args, **_kwargs):
        # Tolerant of real-CLIP clone kwargs (e.g. nodes_hooks.py clone(disable_dynamic=True)) —
        # the shim is stateless-per-encode, so an equivalent instance is always a valid clone.
        return QFCloudTEClip(self._model_id, self._output_dtype, self._timeout, self._device_idx)

    # NOTE deliberately NO `patcher` attribute: a None stub let comfy internals crash
    # three lines downstream (`clip.patcher.forced_hooks` -> raw NoneType AttributeError,
    # delta-CR R5). Absent, access falls to __getattr__'s message-bearing refusal and
    # hasattr() probes stay False.
    def add_hooks_to_dict(self, d):
        return d

    def __getattr__(self, name):
        # LOUD refusal for the rest of the real-CLIP surface (LoraLoader's add_patches,
        # CLIPSetLastLayer's clip_layer, ...): AttributeError (so benign hasattr() probes
        # stay False) with a message naming the boundary instead of a bare traceback.
        raise AttributeError(
            f"QuantFunc cloud TE CLIP shim has no '{name}' — it supports the official "
            f"conditioning nodes only (tokenize / encode_from_tokens[_scheduled]). "
            f"LoRA-on-CLIP / CLIP hooks / set-last-layer need a local CLIP loader.")


class QuantFuncCloudTELoader:
    """Load a CLOUD text encoder as a CLIP — wire it into the official MiniMax H3 nodes'
    `clip` input (Image to Video / Reference to Video / CLIPTextEncode). Prompt + reference
    images stay on the official nodes; encoding runs on the remote QuantFunc TE worker
    (no local CLIP weights, no GPU text-encoder). Auth = engine convention
    (QUANTFUNC_API_KEY env / bundled keyfile) — no key widget."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (_MODEL_CHOICES, {"default": _MODEL_CHOICES[0],
                             "tooltip": "TE model served by the QuantFunc cloud worker."}),
            },
            "optional": {
                "output_dtype": (list(_DTYPE_CHOICES.keys()), {"default": "fp32",
                                 "tooltip": "Wire format of the returned embedding. fp8_e4m3 "
                                            "halves the download; both decode to the same "
                                            "conditioning path."}),
                "timeout_seconds": ("INT", {"default": 600, "min": 1, "max": 36000}),
                "device_idx": ("INT", {"default": 0, "min": 0, "max": 15}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"
    CATEGORY = "QuantFunc/cloud"

    def load(self, model_id, output_dtype="fp32", timeout_seconds=600, device_idx=0):
        lib = qf_engine.load_lib()
        if not hasattr(lib, "quantfunc_te_cloud_encode"):
            raise RuntimeError("This libquantfunc.so has no cloud-TE support — rebuild/update "
                               "the engine (quantfunc_te_cloud_encode absent).")
        api_key, _ = _read_auth()
        if not api_key:
            raise RuntimeError("cloud TE: no API key — set QUANTFUNC_API_KEY (or bundle "
                               "bin/<plat>/config.json).")
        return (QFCloudTEClip(model_id, output_dtype, int(timeout_seconds), int(device_idx)),)


NODE_CLASS_MAPPINGS = {"QuantFuncCloudTELoader": QuantFuncCloudTELoader}
NODE_DISPLAY_NAME_MAPPINGS = {"QuantFuncCloudTELoader": "QuantFunc Cloud TE Loader"}
