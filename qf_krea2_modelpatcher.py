"""qf_krea2_modelpatcher — the Krea-2 (image t2i) family seam (model + builder).

The FIRST image family on the native seam (user 2026-08-29): same denoise_only shape as the
video families — comfy owns CLIP(Qwen3-VL taps)/VAE/sampler; the engine owns ONLY the
28-block single-stream MMDiT denoise via the generic external session
(Krea2Pipeline::prepareExternalDenoise / quantfunc_denoise_begin + quantfunc_denoise_step).
comfy's krea2 cond arrives as (B, seq, txtlayers*txtdim) — comfy/ldm/krea2/model.py:304 —
which is byte-identical to the engine's expected [1, S, L*td] staging, so cond passes through.
The session machinery is the shared image seam (qf_modelpatcher.QFImageSessionModel); comfy's krea2
rides the Wan21 VIDEO latent format ([B,C,T=1,H,W]), whose singleton T the seam squeezes for the ABI.
"""
import os

import comfy.model_base
import comfy.supported_models

from . import qf_engine as qfe
from . import qf_modelpatcher as qfmp


FAMILY = "krea2"


def matches(pipeline_class, transformer_class=""):
    """The engine's own detector: Krea2Pipeline / Krea2Transformer2DModel."""
    return (str(pipeline_class).startswith("Krea2")
            or str(transformer_class) == "Krea2Transformer2DModel")


@qfe.console_safe_methods   # an exception leaving it is console-safe (#738)
class QFKrea2Model(qfmp.QFImageSessionModel, comfy.model_base.Krea2):
    _TAG = "krea2"
    _NO_COND_HINT = "wire a krea2 CLIPTextEncode"
    _REFUSED_HINT = " (reference/edit conditioning is not part of the krea2 t2i seam)"
    # comfy model_base.Krea2.extra_conds consumables this seam cannot honor — refuse LOUD,
    # never silently drop (the wan reject-list discipline; reference_latents = the krea2
    # ref2img channel, no field in the generic denoise ABI).
    # + the BaseModel inpaint/concat channels (concat_latent_image/concat_mask/denoise_mask): the
    # reject_list_completeness audit (Krea2 row, registered 2026-09-22) found them unguarded.
    _ENGINE_IGNORED_COND_KEYS = ("reference_latents", "reference_latents_method",
                                 "cross_attn_controlnet", "noise_concat",
                                 "concat_latent_image", "concat_mask", "denoise_mask")


def register(deps):
    def build(transformer1_path, bundle_dir=None, lora_entries=(), pinned_memory=False):
        """File-based Krea-2 Turbo t2i — the H3 single-expert staging pattern: stage the
        shipped config bundle (configs/krea2-turbo-*/, minimal official skeleton) + symlink
        the transformer file; engine create runs denoise_only=True (TE + VAE weights
        skipped — comfy's krea2 CLIP owns conditioning, comfy's VAEDecode decodes; the
        engine reads the staged configs for session geometry only)."""
        model_dir = qfmp.stage_denoise_only_package(bundle_dir, transformer1_path)
        return qfmp.family_build(
            deps, model_dir, {"denoise_only": True}, comfy.supported_models.Krea2,
            {"image_model": "krea2", "disable_unet_model_creation": True}, QFKrea2Model,
            f"[qf_native] loaded QuantFuncNativeLoader (Krea-2 t2i svdq) package={os.path.basename(transformer1_path)} "
            f"capacity=native Prepared query (create deferred)", pinned_memory)(list(lora_entries))

    return build
