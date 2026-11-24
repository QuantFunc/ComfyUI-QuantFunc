"""qf_qwenimage21_modelpatcher -- Qwen-Image-2.1 (t2i + edit) family seam for QuantFunc PC Engine.

Plugin-layer QualityLaw guard:
  - quality_enhance=True  -> ratio 1.0
  - quality_enhance=False -> ratio 0.85 ONLY when !has_refs && batch == 1 && single_cond_group;
    edit (reference images), CFG > 1, and batch > 1 ALWAYS emit 1.0.
"""
import ctypes
import os

import comfy.model_base
import comfy.supported_models

from . import qf_engine as qfe
from . import qf_modelpatcher as qfmp


FAMILY = "qwenimage21"
_CAT_KEEP_QI21 = 0.85


def matches(pipeline_class, transformer_class=""):
    return (str(pipeline_class) == "QwenImage21Pipeline"
            or str(transformer_class) == "QwenImage21Transformer2DModel")


@qfe.console_safe_methods
class QFQwenImage21Model(qfmp.QFImageSessionModel, comfy.model_base.QwenImage21):
    _TAG = "qwenimage21"
    _NO_COND_HINT = "wire TextEncodeQwenImage21"
    _ENGINE_IGNORED_COND_KEYS = (
        "attention_mask", "reference_latents_method", "cross_attn_controlnet",
        "noise_concat", "concat_latent_image", "concat_mask",
    )

    def _reset_run_cond_state(self):
        super()._reset_run_cond_state()
        self._run_has_refs = False
        self._run_batch = 1
        self._run_single_cond_group = True
        self._run_cond_groups = 0

    @property
    def current_patcher(self):
        return getattr(self, "_current_patcher", None)

    @current_patcher.setter
    def current_patcher(self, patcher):
        self._current_patcher = patcher

    def _effective_cat_ratio(self, quality_enhance):
        if quality_enhance:
            return 1.0
        if (getattr(self, "_run_has_refs", False)
                or getattr(self, "_run_batch", 1) != 1
                or not getattr(self, "_run_single_cond_group", True)):
            return 1.0
        return _CAT_KEEP_QI21

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        try:
            self._run_cond_groups = getattr(self, "_run_cond_groups", 0) + 1
            if kwargs.get("reference_latents") is not None:
                self._run_has_refs = True
            return out
        except BaseException:
            self._reset_run_cond_state()
            self._qf_conds_done = True
            raise

    def _check_conds(self, kwargs):
        dev = getattr(self, "device", None)
        dev_id = dev.index if getattr(dev, "type", None) == "cuda" and dev.index is not None else None
        if kwargs.get("reference_latents") is not None and not hasattr(qfe.load_lib(device_id=dev_id), "qf_xfm_step_refs"):
            raise RuntimeError(
                "qf_native qwenimage21: reference images are wired (TextEncodeQwenImage21 images + vae), but "
                "the loaded QuantFunc engine has no qf_xfm_step_refs - update the QuantFunc engine.")

    def _prepare_begin_context(self, xin, ctx, transformer_options, kwargs):
        batch = int(xin.shape[0])
        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        cuuids = transformer_options.get("uuids") if isinstance(transformer_options, dict) else None
        self._run_has_refs = bool(kwargs.get("ref_latents")) or getattr(self, "_run_has_refs", False)
        self._run_batch = batch
        self._run_single_cond_group = (
            batch == 1
            and (cou is None or len(cou) <= 1)
            and (cuuids is None or len(cuuids) <= 1)
            and getattr(self, "_run_cond_groups", 1) <= 1
        )

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        if (isinstance(transformer_options, dict) and transformer_options.get("qwen_image21_cache")
                and not getattr(self, "_qi21_cache_noted", False)):
            self._qi21_cache_noted = True
            qfe.say(
                "[qf_native] qwenimage21: QwenImage21Cache settings do not apply to the QuantFunc engine "
                "(sampling is unaffected)", flush=True)
        return super()._apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    def _denoise_group(self, p, i, xi, ci, kwargs, step_index, ctx_key):
        refs = kwargs.get("ref_latents", None) or []
        if not refs:
            return super()._denoise_group(p, i, xi, ci, kwargs, step_index, ctx_key)
        n = len(refs)
        if n > qfe.QF_MAX_STEP_REFS:
            self._qf.end_session_if_open()
            raise RuntimeError(
                f"qf_native qwenimage21: at most {qfe.QF_MAX_STEP_REFS} reference latents supported (got {n})")
        slots = (list(kwargs.get("image_slots", None) or []) + [int(ci.shape[1])] * n)[:n]
        ref_i = [(r[i:i + 1] if int(r.shape[0]) > 1 else r).to(device=xi.device, dtype=xi.dtype).contiguous()
                 for r in refs]
        if any(t.dim() != 4 for t in ref_i):
            self._qf.end_session_if_open()
            raise RuntimeError(
                f"qf_native qwenimage21: reference latents must be 4-D [1, C, h, w]; got "
                f"{[tuple(t.shape) for t in ref_i]}")
        rp = qfe.QFStepRefsParams()
        ctypes.memset(ctypes.byref(rp), 0, ctypes.sizeof(rp))
        rp.struct_size = ctypes.sizeof(rp)
        rp.base = p
        ptrs = (ctypes.c_void_p * n)(*[t.data_ptr() for t in ref_i])
        flat = [int(v) for t in ref_i for v in t.shape]
        dims = (ctypes.c_int32 * len(flat))(*flat)
        sl = (ctypes.c_int32 * n)(*[int(v) for v in slots])
        rp.num_refs = n
        rp.ref_latents = ctypes.cast(ptrs, ctypes.POINTER(ctypes.c_void_p))
        rp.ref_dims = ctypes.cast(dims, ctypes.POINTER(ctypes.c_int32))
        rp.image_slots = ctypes.cast(sl, ctypes.POINTER(ctypes.c_int32))
        self._call_denoise_step(
            rp, f"qwenimage21 qf_xfm_step_refs[step={step_index},group={i},key={ctx_key},refs={n}]",
            fn_name="qf_xfm_step_refs")


def register(deps):
    def build(transformer1_path, bundle_dir=None, lora_entries=(), pinned_memory=False):
        config_path = os.path.join(bundle_dir, "transformer", "config.json") if bundle_dir else ""
        if not os.path.isfile(config_path):
            raise RuntimeError(f"qf_native qwenimage21: transformer config missing: {config_path}")
        return qfmp.family_build(
            deps, transformer1_path,
            {"model_arch": "qwenimage21", "config_path": config_path},
            comfy.supported_models.QwenImage21,
            {"image_model": "qwen_image21", "disable_unet_model_creation": True},
            QFQwenImage21Model,
            f"[qf_native] loaded QuantFuncQwenImage21Loader package={os.path.basename(transformer1_path)} "
            f"(create deferred)",
            pinned_memory=pinned_memory,
        )(list(lora_entries))

    return build
