"""qf_cloud_te_node — ComfyUI node for the QuantFunc CLOUD text-encode.

The node ONLY calls the engine C-API (quantfunc_te_cloud_encode) and reads the
returned tensor handle. It contains NO upload / download / signing / crypto / R2
logic — all of that lives inside the compiled .so (design v11 security boundary).

Call model: the C-API does one bounded unit of work per call and reports either a
terminal result or "still pending" with a durable task_id; this node loops,
re-passing the task_id (resume) until terminal or its own overall budget. A cloud
task_id survives process restart, so a run interrupted mid-encode can be resumed.
"""
import ctypes
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

_DTYPE_CHOICES = {"fp32": qf_engine.QF_FP32, "fp8_e4m3": qf_engine.QF_FP8_E4M3}
_PER_CALL_WAIT_MS = 5000   # per-call wait sub-step budget (ms); the node loops for the overall budget
_POLL_INTERVAL_S = 1.0     # sleep between resume polls (s)


def _last_error(lib) -> str:
    try:
        e = lib.quantfunc_last_error()
        return e.decode("utf-8", "replace") if e else "(no detail)"
    except Exception:
        return "(no detail)"


def _save_ref_images(image_batch) -> list[str]:
    """ComfyUI IMAGE tensor [B,H,W,C] float[0,1] → temp PNG paths (positional order)."""
    from PIL import Image
    paths = []
    arr = image_batch.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None, ...]
    for i in range(arr.shape[0]):
        im = (np.clip(arr[i], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        fd, p = tempfile.mkstemp(prefix="qf_cloud_te_ref_", suffix=".png")
        os.close(fd)
        Image.fromarray(im).save(p)
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
                           "request output_dtype='fp32' instead.")
    e4m3 = torch.frombuffer(bytearray(payload), dtype=torch.uint8).view(torch.float8_e4m3fn)
    x = e4m3.float().reshape(shape)                       # decode e4m3 → f32
    return x * scales.reshape(1, seq, 1)                  # de-quantize per token


class QuantFuncCloudTEEncode:
    """Encode text (+ optional reference images) to a conditioning tensor via a
    remote QuantFunc TE worker — no local model / GPU."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "model_id": ("STRING", {"default": ""}),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "output_dtype": (list(_DTYPE_CHOICES.keys()), {"default": "fp32"}),
                "timeout_seconds": ("INT", {"default": 600, "min": 1, "max": 36000}),
            },
            "optional": {
                "ref_images": ("IMAGE",),
                "device_idx": ("INT", {"default": 0, "min": 0, "max": 15}),
                "resume_task_id": ("STRING", {"default": "", "tooltip":
                    "Resume a prior task by its id (surfaced in a previous run's error/timeout) "
                    "instead of submitting a NEW billable task. Leave empty for a fresh encode."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "QuantFunc/cloud"

    def encode(self, server_url, api_key, model_id, text, output_dtype,
               timeout_seconds, ref_images=None, device_idx=0, resume_task_id=""):
        lib = qf_engine.load_lib()
        if not hasattr(lib, "quantfunc_te_cloud_encode"):
            raise RuntimeError("This libquantfunc.so has no cloud-TE support — rebuild/update "
                               "the engine (quantfunc_te_cloud_encode absent).")
        if not (server_url and api_key and model_id and text):
            raise RuntimeError("cloud TE: server_url, api_key, model_id and text are all required.")

        ref_paths = _save_ref_images(ref_images) if ref_images is not None else []
        # Keep python refs to the encoded bytes alive for the whole call.
        c_refs = None
        if ref_paths:
            arr = (ctypes.c_char_p * len(ref_paths))(*[p.encode("utf-8") for p in ref_paths])
            c_refs = arr

        # Cancel hook — ComfyUI interrupt → non-zero return aborts the transfer.
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
        p.ref_paths = c_refs if c_refs is not None else ctypes.cast(None, ctypes.POINTER(ctypes.c_char_p))
        p.n_refs = len(ref_paths)
        p.output_dtype = _DTYPE_CHOICES[output_dtype]
        p.wait_ms = _PER_CALL_WAIT_MS
        p.progress_callback = cb
        p.callback_user_data = None
        # Resume a prior task (from a previous failed/interrupted run) instead of submitting a
        # NEW billable task. The struct keeps the bytes alive; the loop overwrites it on pending.
        resume_task_id = (resume_task_id or "").strip()
        if resume_task_id:
            p.resume_task_id = resume_task_id.encode("utf-8")

        out_result = ctypes.c_void_p()
        out_pending = ctypes.c_int(0)
        task_buf = ctypes.create_string_buffer(64)   # Snowflake task_id <= 20 bytes
        deadline = time.time() + float(timeout_seconds)
        try:
            while True:
                rc = lib.quantfunc_te_cloud_encode(
                    ctypes.byref(p), ctypes.byref(out_result),
                    ctypes.byref(out_pending), task_buf, len(task_buf))
                if rc != qf_engine.QUANTFUNC_OK:
                    # The engine writes out_task_id whenever a task EXISTS (even on a terminal
                    # error) — surface it so the user can resume WITHOUT re-billing.
                    err = _last_error(lib)
                    tid = task_buf.value.decode("utf-8", "replace")
                    if tid:
                        err += (f" — a task already exists (task_id={tid}); to retry WITHOUT a "
                                f"second billable submit, re-run with resume_task_id={tid}")
                    raise RuntimeError("cloud TE encode failed: " + err)
                if out_pending.value == 0:
                    break
                # still processing → resume the SAME task on the next call.
                tid = task_buf.value.decode("utf-8", "replace")
                if tid:
                    p.resume_task_id = task_buf.value  # keep bytes alive via task_buf
                if _mm is not None:
                    try:
                        _mm.throw_exception_if_processing_interrupted()
                    except Exception:
                        raise
                if time.time() > deadline:
                    raise RuntimeError(f"cloud TE: timed out after {timeout_seconds}s "
                                       f"(task still processing; task_id={tid}). Re-run to resume.")
                time.sleep(_POLL_INTERVAL_S)

            if not out_result:
                raise RuntimeError("cloud TE: terminal state but no result tensor: " + _last_error(lib))
            emb = _read_tensor(lib, out_result)
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

        # ComfyUI CONDITIONING = list of [tensor, meta]. H3 TE conditioning has no pooled.
        return ([[emb, {}]],)


class QuantFuncH3AddReference:
    """Attach ONE reference image to an existing H3 conditioning (`minimax_refs`) —
    the DiT-side half of the official MiniMaxH3ReferenceToVideo, for conditioning
    produced WITHOUT the local CLIP (e.g. QuantFuncCloudTEEncode).

    The official node does two inseparable things: (1) present the refs to the
    Qwen3-VL text encoder (<Picture i> tokens) and (2) VAE-encode each ref into a
    `minimax_refs` block the DiT consumes every step. With the cloud TE, half (1)
    rides QuantFuncCloudTEEncode's `ref_images` input (SAME images, SAME order —
    <Picture 1> = first chained reference), and this node replicates half (2)
    byte-for-byte from the official image branch (aspect-preserving down-scale,
    32-px canvas rounding, vae.encode, same block keys). Chain one node per
    reference, exactly like MiniMaxH3AddGuide chains guides. Image refs only —
    video/audio references need the local-CLIP path (the cloud worker takes image
    refs)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "vae": ("VAE",),
                "ref_image": ("IMAGE",),
                "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32,
                          "tooltip": "The GENERATION width — used only by 'match' sizing (scale the ref down to the generation's pixel area)."}),
                "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "ref_image_size": (["match", "max"], {"default": "match",
                    "tooltip": "'match' scales the ref (down only, keeping aspect) to the generation's pixel area; 'max' uses the reference pipeline's 2048px short edge for best identity fidelity (slower — ref tokens ride every step)."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "add_reference"
    CATEGORY = "QuantFunc/cloud"

    # mirrors comfy_extras/nodes_minimax_h3.py (CANVAS_MULTIPLE=32, REF_IMAGE_SHORT_EDGE=2048)
    _CANVAS_MULTIPLE = 32
    _REF_SHORT_EDGE = 2048

    def add_reference(self, positive, vae, ref_image, width, height, ref_image_size="match"):
        import math
        import comfy.utils
        import node_helpers

        h, w = ref_image.shape[1], ref_image.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((width * height) / (w * h)))
        else:
            scale = min(1.0, self._REF_SHORT_EDGE / min(w, h))
        cm = self._CANVAS_MULTIPLE
        tw = max(cm, round(w * scale / cm) * cm)
        th = max(cm, round(h * scale / cm) * cm)
        # official _resize: [B,H,W,C] -> lanczos -> [1,th,tw,3], stretch (crop disabled)
        samples = ref_image[:1, ..., :3].movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, tw, th, "lanczos", "disabled")
        resized = samples.movedim(1, -1)
        z = vae.encode(resized)
        block = {"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z}
        refs = list(positive[0][1].get("minimax_refs", []))
        refs.append(block)
        return (node_helpers.conditioning_set_values(positive, {"minimax_refs": refs}),)


NODE_CLASS_MAPPINGS = {
    "QuantFuncCloudTEEncode": QuantFuncCloudTEEncode,
    "QuantFuncH3AddReference": QuantFuncH3AddReference,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QuantFuncCloudTEEncode": "QuantFunc Cloud TE Encode",
    "QuantFuncH3AddReference": "QuantFunc H3 Add Reference (cloud TE)",
}
