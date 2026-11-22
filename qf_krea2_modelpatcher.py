"""qf_krea2_modelpatcher -- Krea-2 (image t2i) family seam for QuantFunc PC Engine."""
import os

import comfy.model_base
import comfy.supported_models

from . import qf_engine as qfe
from . import qf_modelpatcher as qfmp


FAMILY = "krea2"


def matches(pipeline_class, transformer_class=""):
    return (str(pipeline_class).startswith("Krea2")
            or str(transformer_class) == "Krea2Transformer2DModel")


@qfe.console_safe_methods
class QFKrea2Model(qfmp.QFImageSessionModel, comfy.model_base.Krea2):
    _TAG = "krea2"
    _NO_COND_HINT = "wire a krea2 CLIPTextEncode"
    _REFUSED_HINT = " (reference/edit conditioning is not part of the krea2 t2i seam)"
    _ENGINE_IGNORED_COND_KEYS = (
        "reference_latents", "reference_latents_method",
        "cross_attn_controlnet", "noise_concat",
        "concat_latent_image", "concat_mask", "denoise_mask",
    )


def register(deps):
    def build(transformer1_path, bundle_dir=None, lora_entries=(), pinned_memory=False):
        config_path = os.path.join(bundle_dir, "transformer", "config.json") if bundle_dir else ""
        if not os.path.isfile(config_path):
            raise RuntimeError(f"qf_native krea2: transformer config missing: {config_path}")
        return qfmp.family_build(
            deps, transformer1_path,
            {"model_arch": "krea2", "config_path": config_path},
            comfy.supported_models.Krea2,
            {"image_model": "krea2", "disable_unet_model_creation": True},
            QFKrea2Model,
            f"[qf_native] loaded QuantFuncKrea2Loader package={os.path.basename(transformer1_path)} (create deferred)",
            pinned_memory=pinned_memory,
        )(list(lora_entries))

    return build
