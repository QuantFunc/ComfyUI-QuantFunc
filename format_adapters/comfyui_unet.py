"""Adapter for ComfyUI single-file UNETLoader / Load Diffusion Model output.

Two sub-cases, one adapter:

1. **ComfyUI-prefixed** (original path): the file carries
   ``model.diffusion_model.X`` or ``diffusion_model.X`` prefix keys.
   These are the standard UNETLoader / Load Diffusion Model outputs.

2. **Bare BFL / diffusers layout** (new, general fix): the file has NO
   ComfyUI prefix but the arch can be identified by
   ``fingerprint_arch_from_keys()`` (e.g. raw BFL Klein files with bare
   ``double_blocks.``/``single_blocks.`` keys, or bare diffusers QwenImage
   files). This covers any precision — BF16, FP8, INT4, FP4 — as long as
   the arch fingerprint recognises the key pattern.

In both sub-cases the file contains ONLY transformer weights; TE and VAE
come from separate files (LoadCLIP / LoadVAE nodes).

Detection priority is below PrequantOurs and NVFP4Disk: this is the
"generic" single-file transformer format applied when no more-specific
adapter matched.

Adaptation:
  staging/transformer/diffusion_pytorch_model.safetensors  → symlink to source
  staging/transformer/config.json                           (synthesized)
  staging/quantfunc_config.json:
    {"method": "online_quant",
     "transformer_key_strip": "<prefix>"}   # empty string for bare layout
  + symlink whatever TE / VAE was provided (handled by their own adapters
    via cooperative co-adaptation — see factory.build_pipeline_inputs).
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BuildContext, FormatAdapter, SourceBundle, StagingResult
from .factory import adapter
from .tools import (
    fingerprint_arch_from_keys,
    read_safetensors_keys,
)
from .tools.krea2_fp8_te import guard_krea2_te_fp8
from .tools.hf_layout import (
    HFLayout,
    ARCH_TO_TRANSFORMER_CLASS,
    copy_tokenizer_bundle,
    krea2_is_distilled,
    bundled_te_config,
    bundled_vae_config,
    assert_vae_matches_arch,
)

logger = logging.getLogger(__name__)


# Ordered: first hit wins. We strip whichever prefix is present.
CANDIDATE_PREFIXES = [
    "model.diffusion_model.",   # Flux/SDXL/Qwen UNETLoader output
    "diffusion_model.",          # Some Qwen variants
]


def _detect_transformer_prefix(file_path: str) -> str:
    """Return the matching prefix, or "" if none."""
    sample = []
    for k in read_safetensors_keys(file_path):
        sample.append(k)
        if len(sample) >= 50:
            break
    for px in CANDIDATE_PREFIXES:
        if any(k.startswith(px) for k in sample):
            return px
    return ""


def _detect_krea2_te_prefix(te_path: str) -> str:
    """Prefix to strip from a Krea-2 Qwen3-VL 4B text-encoder file.

    The engine's krea2 TE factory (ComponentImpl.cpp) accepts the tower under
    `language_model.*` (full Qwen3-VL source — KEEP; it prepends the prefix
    itself) or bare `embed_tokens.weight` (QF export). A ComfyUI standalone
    Qwen3-VL text tower is stored under `model.*` → strip to bare so the
    export-layout probe fires. Returns "" when no strip is needed.
    """
    keys = list(read_safetensors_keys(te_path))
    if any(k.startswith("language_model.") for k in keys):
        return ""                       # full source layout — engine handles it
    if any(k == "model.embed_tokens.weight" or k.startswith("model.") for k in keys):
        return "model."                 # text-only tower → strip to bare
    return ""                            # already bare


# --------------------------------------------------------------------------- #
# Krea-2 fp8 text-encoder acceptance. NOTE every "the engine dequantizes / throws"
# statement in this file and in tools/krea2_fp8_te.py describes the engine build
# carrying the krea2 TE fp8 dequant (branch fix/krea2-te-fp8-dequant, d44f01b8) —
# NOT engine main, which hardcodes skip_fp8_dequant=true and would byte-reinterpret
# the fp8 into its BF16 container. See ENGINE DEPENDENCY below.
# The guard lives in tools/krea2_fp8_te.py
# because `arch == "Krea2"` is reachable from MORE than this adapter (bundled
# checkpoints, hf-native synthesised staging). A guard wired into only one route
# is not a guard. See that module for the engine contract + blind spots; the
# sweep-lock test fails the suite if a Krea-2 TE staging route skips it.
#
# ENGINE DEPENDENCY (read before touching the guard):
#   Routing an fp8 Qwen3-VL TE into staging is ONLY safe on an engine whose krea2
#   TE factory runs the generic DequantFP8Provider (fp8 -> bf16 container -> int4)
#   AND whose DequantFP8Provider carries the per-tensor DEFAULT-DENY scale guard.
#   That engine change is NOT yet on engine main — it lives on the engine branch
#   `fix/krea2-te-fp8-dequant` (d44f01b8) pending merge + ship. An engine WITHOUT
#   it loads the krea2 TE with `skip_fp8_dequant=true` and byte-reinterprets the
#   fp8 bytes into the BF16 container => SILENT GARBAGE.
#   => This plugin change MUST NOT be RELEASED ahead of that engine build.
#      A VERSION gate is not possible: the engine exposes only
#      `quantfunc_version()`, and the fixed build reports the SAME version string
#      (0.0.12) as the unfixed shipped one, so it cannot discriminate.
#      The project's REAL coupling mechanism is the SHA-256 ship-manifest
#      (tests/scripts/verify_manifest.py + auto_update.py::_verify_local_lib):
#      it pins a plugin release to a specific engine-binary SHA-256 and
#      self-heals on mismatch. When this plugin version ships, its verify.json
#      MUST require the engine build that contains the krea2 TE fp8 dequant.
#      Until then the safe sequence is ship-engine-then-plugin.
# --------------------------------------------------------------------------- #


@adapter(priority=50)
class ComfyUIDiffusionModelAdapter(FormatAdapter):
    """Single-file transformer: ComfyUI UNETLoader prefix OR bare BFL/diffusers.

    Two acceptance paths (mutually exclusive, both land in this adapter):

    Path A — ComfyUI-prefixed: file has ``model.diffusion_model.`` or
      ``diffusion_model.`` prefix → ``_detect_transformer_prefix`` returns
      non-empty. Classic UNETLoader / Load Diffusion Model output.

    Path B — Bare arch: file has NO ComfyUI prefix but
      ``fingerprint_arch_from_keys()`` identifies a known architecture
      (Flux2Klein, QwenImage, ZImage, …) from the bare key patterns.
      Handles raw BFL-layout single-file models at any precision (BF16,
      FP8, FP4, INT4) that the user loaded via "QuantFunc Pick Diffusion
      Model (zero-load)".

    In both paths the file contains ONLY transformer weights; TE and VAE
    come from separate Load nodes (co-adapted by this adapter).
    """

    @classmethod
    def detect(cls, sources: SourceBundle) -> bool:
        if sources.checkpoint is not None:
            return False                 # bundled-checkpoint has its own adapter
        if sources.transformer is None:
            return False
        try:
            path = sources.transformer.path
            # Path A: standard ComfyUI prefix
            if _detect_transformer_prefix(path):
                return True
            # Path B: no prefix, but arch is identifiable from key patterns.
            # Use the pre-scanned arch from FileRef when available (avoids an
            # extra header read on the hot path); fall back to a fresh scan.
            arch = sources.transformer.arch or fingerprint_arch_from_keys(path)
            return bool(arch)
        except Exception:
            return False

    def adapt(self, sources: SourceBundle, staging_dir: Path,
              context: BuildContext) -> StagingResult:
        assert sources.transformer is not None
        xfm_path = sources.transformer.path
        prefix = _detect_transformer_prefix(xfm_path)
        # NOTE: for bare BFL/diffusers layout (Path B) prefix is "".
        # Do NOT fall back to "model.diffusion_model." — the engine's
        # Flux2Klein / QwenImage / ZImage load paths already understand the
        # bare key layout; adding a bogus prefix-strip would corrupt loading.
        # For Path A (ComfyUI-prefixed), prefix is the detected string.
        arch = (fingerprint_arch_from_keys(xfm_path)
                or sources.transformer.arch
                or "")
        if not arch:
            raise RuntimeError(
                f"Could not identify transformer architecture for "
                f"{xfm_path}. Inspect the safetensors header — file may "
                f"be from an unsupported family. (Refusing to fall back "
                f"to a hardcoded arch since that produces silent garbage "
                f"output when wrong.)")

        layout = HFLayout(staging_dir)

        # ZImage BFL-style files use `layers.N.attention.qkv` (fused) +
        # `final_layer.*` / `x_embedder.*` (no `all_` prefix). The C++
        # engine wants HF-diffusers naming (split QKV + `all_final_layer.2-1.*`
        # / `all_x_embedder.2-1.*`). Materialise a remapped safetensors at
        # staging time — one full read+write pass; subsequent runs reuse.
        remap_used = False
        if arch == "ZImage":
            try:
                from .zimage_bfl_remap import is_bfl_zimage, stage_bfl_zimage
            except Exception:
                is_bfl_zimage = lambda *_: False  # fall back if import fails
                stage_bfl_zimage = None
            if is_bfl_zimage(xfm_path):
                logger.info("[comfyui_unet] ZImage BFL layout detected; "
                             "remapping qkv/out/q_norm/k_norm and final_layer "
                             "/ x_embedder paths to diffusers naming")
                layout.add_transformer_remapped(
                    xfm_path,
                    remap_fn=stage_bfl_zimage and (lambda s, d:
                        stage_bfl_zimage(str(s), d.parent, force=False)),
                    config={"_class_name": ARCH_TO_TRANSFORMER_CLASS.get(arch, "")})
                # The remapped file is HF-diffusers native (no prefix to strip).
                remap_used = True

        # Krea-2 single-file (BFL/akira `blocks.N.attn.wq` + `txtfusion.*`): the
        # engine's Krea2TransformerLighting reads diffusers-internal names
        # (`transformer_blocks.N.attn.to_q` / `ff.gate` / `norm1` /
        # `scale_shift_table` / `img_in` / `txt_in` / `time_embed` /
        # `time_mod_proj` / `final_layer` / `text_fusion`), so a zero-copy
        # key_remap.json manifest translates the BFL layout on load (the engine's
        # fresh-quant DequantFP8Provider handles the FP8 weights). The int8
        # ConvRot format is rejected fail-loud inside the remap (unsupported).
        if not remap_used and arch == "Krea2":
            from .comfyui_krea2_remap import is_krea2_bfl, stage_krea2
            if is_krea2_bfl(xfm_path):
                logger.info("[comfyui_unet] Krea-2 BFL layout detected; writing "
                             "zero-copy key_remap.json (blocks/attn/mlp/mod/txtfusion "
                             "→ transformer_blocks/to_q/ff/scale_shift_table/text_fusion)")
                layout.add_transformer_remapped(
                    xfm_path,
                    remap_fn=lambda s, d: stage_krea2(str(s), d.parent, force=False),
                    config={"_class_name": ARCH_TO_TRANSFORMER_CLASS.get(arch, "")})
                remap_used = True

        if not remap_used:
            # Transformer (symlink + on-load prefix strip)
            layout.add_transformer(
                xfm_path,
                config={"_class_name": ARCH_TO_TRANSFORMER_CLASS.get(arch, "")})
            # Only write a key_strip when there is actually a prefix to strip.
            # For bare BFL / diffusers layout (prefix == "") we skip this —
            # a key_strip of "" in quantfunc_config.json would cause the engine
            # to match every key with an empty prefix (= every key) and remove
            # nothing, which is a no-op but may trigger unexpected code paths.
            if prefix:
                layout.set_key_strip("transformer", prefix)

        # Text encoder (if provided)
        if sources.text_encoder is not None:
            from .comfyui_clip import _detect_te_prefix
            te_path = sources.text_encoder.path
            if arch == "Krea2":
                # Krea-2's Qwen3-VL 4B text tower: the engine's krea2 TE factory
                # expects the tower under `language_model.*` (full Qwen3-VL
                # source) or bare `embed_tokens.weight` (QF export). A ComfyUI
                # Qwen3-VL text-encoder file uses `model.*` → strip to bare so
                # the engine's export-layout probe fires; a full-source
                # `language_model.*` file is kept as-is (engine prepends it).
                # The engine's krea2 TE factory now dequantizes the per-tensor
                # F32 fp8_scaled Qwen3-VL TE (DequantFP8Provider, fp8→bf16→int4),
                # so that layout is ALLOWED through; a per-channel/block/non-F32
                # fp8 the engine can't dequant still fails loud here.
                guard_krea2_te_fp8(te_path)
                te_prefix = _detect_krea2_te_prefix(te_path)
            else:
                te_prefix = _detect_te_prefix(te_path)
            te_class = ("Qwen2_5VLForConditionalGeneration"
                        if arch == "QwenImageEdit"
                        else "Qwen3VLForConditionalGeneration" if arch == "Krea2"
                        else "Qwen3ForCausalLM")
            # Use bundled full TE config when available — it carries
            # hidden_size / num_attention_heads / head_dim / etc. that the
            # C++ engine needs to allocate the right tensor shapes. Minimal
            # `{_class_name: ...}` triggers a fallback `head_dim = hidden /
            # num_heads` (= 80 for ZImage), producing wrong q_proj shape.
            te_cfg = bundled_te_config(arch) or {"_class_name": te_class}
            layout.add_text_encoder(te_path, config=te_cfg)
            if te_prefix:
                layout.set_key_strip("te", te_prefix)

        # VAE (if provided)
        if sources.vae is not None:
            from .comfyui_vae import _detect_vae_prefix
            vae_path = sources.vae.path
            # Fail LOUD if the wired VAE's channel count is wrong for this arch
            # (QwenImageLayered needs a 4-channel RGBA VAE) — a clear, actionable
            # error instead of the engine's cryptic deep conv_out [4]vs[3] crash.
            assert_vae_matches_arch(arch, vae_path)
            vae_prefix = _detect_vae_prefix(vae_path)
            # ZImage / SDXL-style BFL VAE files use `mid.attn_1.{q,k,v}` +
            # `up.N.block.M` etc.; HF AutoencoderKL uses
            # `mid_block.attentions.0.to_{q,k,v}` + `up_blocks.N.resnets.M`.
            # Detect and remap on the fly.
            try:
                from .zimage_bfl_remap import is_bfl_vae, stage_bfl_vae
            except Exception:
                is_bfl_vae = lambda *_: False
                stage_bfl_vae = None
            if stage_bfl_vae is not None and is_bfl_vae(Path(vae_path)):
                logger.info("[comfyui_unet] BFL-style VAE detected; "
                             "remapping mid/up/down/nin_shortcut paths to "
                             "HF AutoencoderKL naming")
                layout.add_vae_remapped(
                    vae_path,
                    remap_fn=lambda s, d: stage_bfl_vae(str(s), d.parent, force=False),
                    config={"_class_name": "AutoencoderKL"})
            else:
                # A standalone "Pick VAE" file carries NO config.json. Hardcoding
                # AutoencoderKL here built the wrong 2D decoder against a 3D
                # AutoencoderKLQwenImage VAE → engine copy_ overflow / "conv_in.bias
                # not found" (#257, customer production crash). Use the bundled
                # per-arch VAE config (with _class_name + latents_mean/std) so a
                # standalone qwen_image_vae paired with a QwenImage(Edit) transformer
                # is staged correctly — same pattern the bundled/SVDQ adapters use.
                # Falls back to AutoencoderKL only for archs with no bundled config.
                layout.add_vae(
                    vae_path,
                    config=bundled_vae_config(arch) or {"_class_name": "AutoencoderKL"})
                if vae_prefix:
                    layout.set_key_strip("vae", vae_prefix)

        # Tokenizer bundle (mandatory — ComfyUI files don't carry one)
        copy_tokenizer_bundle(arch, layout.tokenizer_dir())

        # Scheduler config (optional)
        layout.add_scheduler(sources.scheduler_config, arch=arch)

        # Hints + index
        layout.set_method("online_quant")
        # Separate UNet/CLIP/VAE files carry no per-component precision
        # metadata — propagate the user's choice so TE/VAE actually quantize.
        layout.apply_user_precisions(
            text_precision=context.text_precision,
            vae_precision=context.vae_precision)
        layout.write_quantfunc_config()
        # Krea-2: derive the distillation marker from a real signal (shared with
        # every other synthesising adapter) — never a silent default.
        layout.write_model_index(
            arch, is_distilled=krea2_is_distilled(arch, xfm_path))

        path_label = "comfyui-prefix" if prefix else "bare-bfl/diffusers"
        logger.info("[comfyui_unet] arch=%s prefix=%r path=%s staging=%s",
                     arch, prefix, path_label, staging_dir)
        return StagingResult(
            model_dir=str(staging_dir),
            arch=arch,
            method_hint="online_quant",
            cleanup_dir=str(staging_dir),
        )
