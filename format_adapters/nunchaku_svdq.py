"""Adapter for Nunchaku / MIT upstream SVDQuant INT4 transformer files.

Detection:
  - safetensors __metadata__["model_class"] contains "Nunchaku" OR
  - __metadata__["quantization_config"] JSON has "method": "svdquant"

Layout:
  These files are transformer-only single-file safetensors with rank-32 LoRA
  decomposition baked in (proj_down / proj_up / smooth_factor / qweight /
  wscales). The QuantFunc engine's `backend=svdq` path natively understands
  the Nunchaku key naming via [QwenImageTransformer.cpp:1098-1111] which
  remaps `lora_down → proj_down`, `lora_up → proj_up`, `smooth → smooth_factor`.

  We must:
    - route to backend="svdq" (not "lighting")
    - pass the safetensors absolute path via cfg["transformer"], NOT through
      a staging dir's transformer/ subdir (engine reads metadata directly)
    - still build a staging dir with TE / VAE / scheduler / tokenizer for
      the rest of the pipeline
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .base import BuildContext, FormatAdapter, SourceBundle, StagingResult
from .factory import adapter
from .tools import read_safetensors_metadata, fingerprint_arch_from_keys

logger = logging.getLogger(__name__)


def _detect_nunchaku(path: str) -> tuple[bool, str]:
    """Return (is_nunchaku, transformer_class_name) by reading metadata."""
    try:
        meta = read_safetensors_metadata(path)
    except Exception:
        return (False, "")
    model_class = meta.get("model_class", "")
    if "Nunchaku" in model_class:
        # Pull diffusers _class_name from embedded config JSON if present
        try:
            cfg = json.loads(meta.get("config", "{}"))
            return (True, cfg.get("_class_name", ""))
        except Exception:
            return (True, "")
    qc_str = meta.get("quantization_config", "")
    try:
        qc = json.loads(qc_str) if isinstance(qc_str, str) else qc_str
        if isinstance(qc, dict) and qc.get("method") == "svdquant":
            try:
                cfg = json.loads(meta.get("config", "{}"))
                return (True, cfg.get("_class_name", ""))
            except Exception:
                return (True, "")
    except Exception:
        pass
    return (False, "")


@adapter(priority=90)  # above ComfyUIDiffusionModelAdapter (50), below PrequantOurs (100)
class NunchakuSVDQAdapter(FormatAdapter):
    """Single-file Nunchaku/MIT SVDQuant INT4 transformer (transformer-only).

    Engine routes via backend=svdq; SVDQ loader natively handles
    Nunchaku key naming (proj_down / proj_up / smooth_factor).
    """

    @classmethod
    def detect(cls, sources: SourceBundle) -> bool:
        if sources.checkpoint is not None:
            return False  # bundled-checkpoint has its own adapter
        if sources.transformer is None:
            return False
        is_nunchaku, _ = _detect_nunchaku(sources.transformer.path)
        return is_nunchaku

    def adapt(self, sources: SourceBundle, staging_dir: Path,
              context: BuildContext) -> StagingResult:
        from .tools.hf_layout import (
            HFLayout, copy_tokenizer_bundle,
            bundled_te_config, bundled_vae_config, assert_vae_matches_arch,
        )
        from .comfyui_clip import _detect_te_prefix
        from .comfyui_vae import _detect_vae_prefix, _is_comfyui_flat_vae

        assert sources.transformer is not None
        xfm_path = sources.transformer.path
        _, xfm_class = _detect_nunchaku(xfm_path)

        # Arch detection: prefer metadata _class_name, fall back to QwenImage.
        # Nunchaku ships QwenImage variants and (newer) Z-Image-Turbo SVDQ files.
        # A ZImageTransformer2DModel must NOT stay mis-tagged QwenImage (#408) —
        # that picks the wrong VAE/TE/tokenizer config + ZImagePipeline routing.
        arch = "QwenImage"
        if xfm_class:
            if "ZImage" in xfm_class or "Z_Image" in xfm_class or "ZImagePipeline" in xfm_class:
                arch = "ZImage"
            elif "Edit" in xfm_class:
                arch = "QwenImageEdit"
        # QwenImageLayered reuses the base QwenImageTransformer2DModel class, so the
        # `xfm_class` metadata CANNOT tell it apart from base QwenImage — the only
        # stable on-disk signal is the key fingerprint (its distinctive
        # `addition_t_embedding` weight, see arch_fingerprint._detect_arch_by_keys).
        # Confirm it whenever we'd otherwise default to base QwenImage. Only ever
        # UPGRADES QwenImage → QwenImageLayered, so the ZImage/Edit class-string
        # detection (#408) is untouched. Needed so the layered RGBA 4-ch VAE config +
        # QwenImageLayeredPipeline routing are selected instead of the base 3-ch path.
        if arch == "QwenImage" and fingerprint_arch_from_keys(xfm_path) == "QwenImageLayered":
            arch = "QwenImageLayered"

        layout = HFLayout(staging_dir)

        # IMPORTANT: do NOT add the transformer to the staging dir. The svdq
        # backend reads the file directly via cfg["transformer"]; staging
        # only carries TE / VAE / scheduler / tokenizer.
        # We do still write a transformer/config.json so model_index.json
        # makes the loader believe the layout is intact. Use minimal config
        # (engine reads from safetensors metadata anyway).
        layout.add_transformer(xfm_path, config={
            "_class_name": xfm_class or "QwenImageTransformer2DModel"
        })

        # Text encoder (optional — user usually wires a separate Qwen2.5-VL)
        if sources.text_encoder is not None:
            te_path = sources.text_encoder.path
            te_prefix = _detect_te_prefix(te_path)
            te_cfg = bundled_te_config(arch) or {
                "_class_name": "Qwen2_5VLForConditionalGeneration",
            }
            if arch == "Krea2":      # not reachable today; keeps the sweep total
                from .tools.krea2_fp8_te import guard_krea2_te_fp8
                guard_krea2_te_fp8(te_path)
            layout.add_text_encoder(te_path, config=te_cfg)
            if te_prefix:
                layout.set_key_strip("te", te_prefix)
        else:
            # Bundled TE config still helps engine build correct shape; no weights.
            te_cfg = bundled_te_config(arch) or {
                "_class_name": "Qwen2_5VLForConditionalGeneration",
            }
            if arch == "Krea2":      # not reachable today; keeps the sweep total
                from .tools.krea2_fp8_te import guard_krea2_te_fp8
                guard_krea2_te_fp8(xfm_path)
            layout.add_text_encoder(xfm_path, config=te_cfg)
            logger.warning("[nunchaku_svdq] no text_encoder source — engine will fail "
                           "unless cfg.transformer-side TE is provided downstream")

        # VAE (optional — user usually wires a separate VAE)
        vae_cfg = bundled_vae_config(arch) or {"_class_name": "AutoencoderKL"}
        if sources.vae is not None:
            vae_path = sources.vae.path
            # QwenImageLayered needs a 4-channel RGBA VAE: fail LOUD on a wrong
            # (3-ch) wired VAE instead of a cryptic engine conv_out [4]vs[3] crash.
            assert_vae_matches_arch(arch, vae_path)
            vae_prefix = _detect_vae_prefix(vae_path)
            layout.add_vae(vae_path, config=vae_cfg)
            if vae_prefix:
                layout.set_key_filter("vae", vae_prefix)  # bundle slice: filter by `vae.` prefix
            # Only ComfyUI FLAT-named VAEs (standalone qwen_image_vae.safetensors
            # or bundle-embedded vae.* slices: conv1.weight, head.0.gamma) need the
            # engine's ComfyUIVAEAliasProvider (flat ↔ diffusers translation). A
            # diffusers-native VAE (diffusion_pytorch_model.safetensors with
            # post_quant_conv.*/encoder.conv_in.*) is ALREADY what the engine asks
            # for; aliasing it remaps post_quant_conv.bias → conv2.bias → "not
            # found" (#408). Gate the flag on the actual key shape.
            if _is_comfyui_flat_vae(vae_path, vae_prefix):
                layout.set_extra("vae_comfyui_alias", True)
            else:
                logger.info("[nunchaku_svdq] VAE is diffusers-native (%s) — engine "
                            "reads it directly, ComfyUIVAEAliasProvider NOT applied",
                            os.path.basename(str(vae_path)))
        else:
            layout.add_vae(xfm_path, config=vae_cfg)
            logger.warning("[nunchaku_svdq] no vae source — generation will fail "
                           "unless one is provided downstream")

        # Tokenizer bundle (required — Nunchaku files don't carry one)
        copy_tokenizer_bundle(arch, layout.tokenizer_dir())

        # Scheduler config (optional)
        layout.add_scheduler(sources.scheduler_config, arch=arch)

        # Method hint signals BuildPipeline to switch backend → svdq and
        # pass the original transformer file path via cfg["transformer"].
        layout.set_method("prequant_svdq_separate")
        layout.set_extra("svdq_transformer_path", str(xfm_path))

        layout.write_quantfunc_config()
        from .tools.hf_layout import krea2_is_distilled as _krea2_is_distilled
        layout.write_model_index(
            arch, is_distilled=_krea2_is_distilled(arch, sources.transformer.path))

        logger.info("[nunchaku_svdq] arch=%s xfm_class=%r staging=%s",
                     arch, xfm_class, staging_dir)
        return StagingResult(
            model_dir=str(staging_dir),
            arch=arch,
            method_hint="prequant_svdq_separate",
            cleanup_dir=str(staging_dir),
        )
