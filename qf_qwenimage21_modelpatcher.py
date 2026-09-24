"""qf_qwenimage21_modelpatcher — the Qwen-Image-2.1 (t2i + image edit) family seam (model + builder).

The Krea2 shape (the shared image seam, qf_modelpatcher.QFImageSessionModel): comfy owns CLIP
(TextEncodeQwenImage21 → Qwen3-VL-8B last-hidden [B, S, 4096]) / VAE (qwen_image_2.1_vae) / sampler / CFG; the
engine owns ONLY the 32-block denoise via the generic external session (QwenImage21Pipeline::prepareExternalDenoise /
quantfunc_denoise_begin + quantfunc_denoise_step). comfy's own QwenImage21Transformer2DModel receives
(x [B,64,H/16,W/16] normalized latent, timestep = sigma, context) per cond group — exactly what this seam forwards,
so cond and latent pass through unchanged (cast to bf16, the engine's activation dtype). The engine prunes a
one-cond-group session without references only (text-to-image, img2img, mask inpainting); edit, CFG > 1 and
batch > 1 run full.
Reference-image EDIT: TextEncodeQwenImage21 (with a VAE + images) appends reference_latents to both conds and
reports image_slots per cond; model_base.QwenImage.extra_conds normalizes the references (process_latent_in).
Both ride every step through quantfunc_denoise_step_refs — the engine splices them exactly like comfy's own
build_sequence (generate_edit's forwardEdit). Anything else the seam cannot consume is refused loud.
"""
import ctypes
import os

import comfy.model_base
import comfy.supported_models

from . import qf_engine as qfe
from . import qf_modelpatcher as qfmp


FAMILY = "qwenimage21"


def matches(pipeline_class, transformer_class=""):
    """The engine's own detector (src/QwenImage21Pipeline.cpp qwenimage21_detect)."""
    return (str(pipeline_class) == "QwenImage21Pipeline"
            or str(transformer_class) == "QwenImage21Transformer2DModel")


class QFQwenImage21Model(qfmp.QFImageSessionModel, comfy.model_base.QwenImage21):
    _TAG = "qwenimage21"
    _VAE_S = 16   # latent_formats.QwenImage21 spacial_downscale_ratio; session W/H = latent * 16
    _LATENT_CHANNELS = 64
    _NO_COND_HINT = "wire TextEncodeQwenImage21"
    # comfy model_base.QwenImage(21).extra_conds consumables — the RULE: a key is accepted only if the engine
    # consumes it or the SAMPLER honours it; every other wired key is refused LOUD, never silently dropped (the
    # wan reject-list discipline). CONSUMED: reference_latents + image_slots (quantfunc_denoise_step_refs).
    # SAMPLER-HONOURED: denoise_mask (SetLatentNoiseMask inpainting is KSamplerX0Inpaint's blend, never a QI2.1
    # model input). REFUSED: attention_mask (the engine forward is mask-blind), reference_latents_method (QI2.1
    # has one splice rule — a method from e.g. FluxKontextMultiReferenceLatentMethod would do nothing), and
    # cross_attn_controlnet / noise_concat / concat_latent_image / concat_mask (channels QI2.1 has no input for).
    # Audited by tests/reject_list_completeness.py (QwenImage21 row, the parent chain walked).
    _ENGINE_IGNORED_COND_KEYS = ("attention_mask", "reference_latents_method", "cross_attn_controlnet",
                                 "noise_concat", "concat_latent_image", "concat_mask")

    # comfy model_base.QwenImage21.current_patcher's setter drives the TORCH model's prefix K/V cache
    # (diffusion_model.reset_prefix_cache) — the engine owns its own prefix cache inside the session,
    # so the stub must not be asked for one.
    @property
    def current_patcher(self):
        return getattr(self, "_current_patcher", None)

    @current_patcher.setter
    def current_patcher(self, patcher):
        self._current_patcher = patcher

    def get_dynamic_vram__units(self):
        return [], []   # no torch blocks to page — the engine holds the weights

    def _check_conds(self, kwargs):
        # probe the loaded library, NOT self._qf.lib (that materializes the lazy engine = creates the pipeline early)
        if kwargs.get("reference_latents") is not None and \
                not hasattr(qfe.load_lib(), "quantfunc_denoise_step_refs"):
            raise RuntimeError(
                "qf_native qwenimage21: reference images are wired (TextEncodeQwenImage21 images + vae), but "
                "the loaded QuantFunc engine has no quantfunc_denoise_step_refs - update the QuantFunc engine "
                "to a build with Qwen-Image-2.1 edit support.")

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        # QwenImage21Cache (the official edit template wires it) configures comfy's OWN torch prefix cache; the
        # QuantFunc engine's prefix cache is a create-time engine option (off by default), so say it once.
        if isinstance(transformer_options, dict) and transformer_options.get("qwen_image21_cache") \
                and not getattr(self, "_qi21_cache_noted", False):
            self._qi21_cache_noted = True
            qfe.say("[qf_native] qwenimage21: QwenImage21Cache settings do not apply to the QuantFunc engine "
                    "(its prefix K/V cache is an engine option, off by default) - sampling is unaffected",
                    flush=True)
        return super()._apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    def _denoise_group(self, p, i, xi, ci, kwargs, step_index, ctx_key):
        # Reference-image EDIT: the references (already normalized by extra_conds' process_latent_in) and this
        # cond's slots; comfy build_sequence fills missing slots with the text length (reference after the text).
        refs = kwargs.get("ref_latents", None) or []
        if not refs:
            return super()._denoise_group(p, i, xi, ci, kwargs, step_index, ctx_key)
        n = len(refs)
        slots = (list(kwargs.get("image_slots", None) or []) + [int(ci.shape[1])] * n)[:n]
        ref_i = [(r[i:i + 1] if int(r.shape[0]) > 1 else r).to(device=xi.device, dtype=xi.dtype)
                 .contiguous() for r in refs]            # CONDList repeats/concats per batch item; ABI: dtype = the latent's
        if any(t.dim() != 4 for t in ref_i):             # the ABI packs exactly [1, C, h, w] per reference
            self._qf.end_session_if_open()               # never strand the session begun above
            raise RuntimeError("qf_native qwenimage21: reference latents must be 4-D image latents [1, C, h, w]; got "
                               f"{[tuple(t.shape) for t in ref_i]} - feed Qwen-Image-2.1 references (TextEncodeQwenImage21 + its VAE)")
        rp = qfe.DenoiseStepRefsParams()
        ctypes.memset(ctypes.byref(rp), 0, ctypes.sizeof(rp))
        rp.struct_size = ctypes.sizeof(rp)
        rp.base = p
        ptrs = (ctypes.c_void_p * n)(*[t.data_ptr() for t in ref_i])
        dims = (ctypes.c_int32 * (4 * n))(*[int(v) for t in ref_i for v in t.shape])
        sl = (ctypes.c_int32 * n)(*[int(v) for v in slots])
        rp.num_refs = n
        rp.ref_latents = ctypes.cast(ptrs, ctypes.POINTER(ctypes.c_void_p))
        rp.ref_dims = ctypes.cast(dims, ctypes.POINTER(ctypes.c_int32))
        rp.image_slots = ctypes.cast(sl, ctypes.POINTER(ctypes.c_int32))
        self._call_denoise_step(
            rp, f"qwenimage21 denoise_step_refs[step={step_index},group={i},key={ctx_key},refs={n}]",
            fn_name="quantfunc_denoise_step_refs")


def register(deps):
    def build(transformer1_path, bundle_dir=None, lora_entries=()):
        """File-based Qwen-Image-2.1 (t2i + edit) — the Krea2 single-expert staging pattern: stage the shipped
        config bundle (configs/qwen-image-2.1-*/, model_index.json only — the engine synthesizes the
        per-component configs from the reference arch) + symlink the transformer file; engine create
        runs denoise_only=True (TE + VAE weights skipped — comfy's TextEncodeQwenImage21 owns
        conditioning, comfy's VAEDecode decodes)."""
        model_dir = qfmp.stage_denoise_only_package(bundle_dir, transformer1_path)
        return qfmp.family_build(
            deps, model_dir, {"denoise_only": True}, comfy.supported_models.QwenImage21,
            {"image_model": "qwen_image21", "disable_unet_model_creation": True}, QFQwenImage21Model,
            f"[qf_native] loaded QuantFuncNativeLoader (Qwen-Image-2.1 svdq) package={os.path.basename(transformer1_path)} "
            f"capacity=native Prepared query (create deferred)")(list(lora_entries))

    return build
