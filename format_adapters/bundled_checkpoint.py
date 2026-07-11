"""Adapter for single-file checkpoints that bundle multiple components.

A "bundled checkpoint" is a single safetensors file whose tensor keys group
into multiple top-level components (transformer + text encoder + VAE) under
distinct prefixes — e.g. Qwen-Rapid-SFW-v22 with `model.diffusion_model.*`
+ `text_encoders.qwen25_7b.*` + `vae.*`. Some packagers call these
"all-in-one" or "everything bundle"; we use a neutral name here so the
abstraction generalises to any future packaging that follows the same
pattern.

Detection:
  Match the first scheme in `BUNDLE_SCHEMES` whose three component
  prefixes are all present in the file's key list.

Adaptation:
  - Symlink the same physical file once per component subdir
  - Write per-component prefix FILTERs in quantfunc_config.json so each
    component's TensorsProvider sees only its slice (zero data copy)
  - Synthesize per-component config.json
  - Copy bundled tokenizer
  - Set method=online_quant (these files are typically FP8 mixed → INT4 quant)

Memory pages of the underlying file are mmap-shared across the per-component
symlinks (the OS deduplicates). Disk write footprint is ~5 MB (configs +
tokenizer); zero weight data is copied.

Adding a new bundle layout: append a dict to `BUNDLE_SCHEMES`. The schema
is `{name, transformer, te, vae}` where each prefix string identifies one
component. No adapter changes needed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .base import BuildContext, FormatAdapter, SourceBundle, StagingResult
from .factory import adapter
from .tools.safetensors_io import (
    has_keys_starting_with,
    read_safetensors_keys,
)
from .tools.hf_layout import (
    krea2_is_distilled,
    HFLayout,
    ARCH_TO_TRANSFORMER_CLASS,
    copy_tokenizer_bundle,
    bundled_te_config,
    bundled_vae_config,
    assert_vae_matches_arch,
)

logger = logging.getLogger(__name__)


# Each entry maps a component-prefix-set "scheme" → per-component prefixes.
# We try schemes in order; first scheme whose all prefixes are present wins.
# Append new layouts here without touching adapter logic.
#
# `vision` is optional — only present in edit-mode prequant exports where
# vision_encoder is independently quantized (separate Module instance from TE).
BUNDLE_SCHEMES = [
    {   # QuantFunc native flat bundle (writer = src/Export.cpp::exportBundle).
        # Arch-agnostic prefixes; arch detection happens via transformer keys.
        "name": "qf_flat_bundle",
        "transformer": "model.diffusion_model.",
        "te":          "text_encoder.",
        "vae":         "vae.",
        "vision":      "vision_encoder.",
    },
    {   # Qwen-Rapid bundle (upstream ComfyUI ecosystem)
        "name": "qwen25_bundle",
        "transformer": "model.diffusion_model.",
        # te MUST include the trailing `model.` segment. Qwen3TextEncoderGEMM
        # registers its children without a `model.` prefix (embed_tokens.weight
        # / layers.N / norm), so the PrefixFilterProvider's prepend must be the
        # FULL bridge from file root to the start of the C++ module hierarchy.
        # The qwen25 bundle stores the language sub-tree at
        # `text_encoders.qwen25_7b.transformer.model.*` (visual sub-tree is at
        # `text_encoders.qwen25_7b.transformer.visual.*`). Without `model.` the
        # bridge points one level too shallow; engine queries become
        # `…transformer.layers.0.attn.q_proj.weight` (does NOT exist) → Module
        # `partial=true` silent-skip (Module.h:145-147) → empty offload_backup
        # → Tensor::scalar_size() throws `map::at` at Tensor.h:2021 — the
        # #135/#140/#134/#25-recurrence map::at INTERNAL signature.
        "te":          "text_encoders.qwen25_7b.transformer.model.",
        "vae":         "vae.",
        # Vision tower lives at `text_encoders.qwen25_7b.transformer.visual.*`
        # (NOT under `.model.`, the LLM sub-tree). The engine's AIO vision path
        # uses Qwen25VLVisionTensorsProvider(add_visual_prefix=true) → it queries
        # `visual.blocks.N…` and its presence probe checks
        # `visual.blocks.0.attn.qkv.weight` (ComponentImpl.cpp:7400), so the
        # PrefixFilter prepend (ComponentImpl.cpp:1306-1310) must strip ONLY to
        # `transformer.` and KEEP the `visual.` segment. WITHOUT this key,
        # adapt() falls back to ve_prefix = te (`…transformer.model.`, the LLM
        # sub-tree) → the filter matches ZERO vision weights → engine warns
        # "no weights found in safetensors — skipping" (ComponentImpl.cpp:7404)
        # → null vision encoder → generate_edit null-deref / worker kill.
        "vision":      "text_encoders.qwen25_7b.transformer.",
    },
    {   # Qwen variants with qwen2vl naming
        "name": "qwen2vl_bundle",
        "transformer": "model.diffusion_model.",
        "te":          "text_encoders.qwen2vl.transformer.",
        "vae":         "vae.",
        # Same visual-subtree rule as qwen25_bundle: KEEP `visual.`, strip only
        # to `transformer.`. Here `te` already stops at `transformer.` so the
        # default ve_prefix would coincidentally resolve correctly, but declare
        # `vision` EXPLICITLY so a future `te` change (e.g. adding a `.model.`
        # segment to mirror qwen25) cannot silently re-break the vision tower.
        "vision":      "text_encoders.qwen2vl.transformer.",
    },
    {   # SD legacy bundles (out of MVP scope but detect for graceful error)
        "name": "sd_legacy_bundle",
        "transformer": "model.diffusion_model.",
        "te":          "cond_stage_model.",
        "vae":         "first_stage_model.",
    },
]


def _select_scheme(file_path: str) -> dict | None:
    """Pick the first scheme whose required xfm + te + vae prefixes are present.

    Optional `vision` slot is recorded for the matched scheme but does not
    contribute to selection. qf_flat_bundle is checked first; the upstream
    text_encoders.* schemes contain a longer prefix that does NOT collide
    with the bare `text_encoder.` of qf_flat_bundle, so detection is unique.
    """
    # Build dedup'd probe prefix list (include `vision` so we know if it exists)
    all_prefixes = set()
    for s in BUNDLE_SCHEMES:
        all_prefixes.add(s["transformer"])
        all_prefixes.add(s["te"])
        all_prefixes.add(s["vae"])
        if "vision" in s:
            all_prefixes.add(s["vision"])
    seen = has_keys_starting_with(file_path, list(all_prefixes))
    for s in BUNDLE_SCHEMES:
        if {s["transformer"], s["te"], s["vae"]}.issubset(seen):
            return s
    return None


def _read_bundle_flat_metadata(file_path: str) -> dict:
    """Read full safetensors metadata dict (qf_flat_bundle's flat key-prefixed
    entries like `transformer.method`, `text_encoder.text_precision`, etc.).
    Returns empty dict on failure.
    """
    try:
        from .tools.safetensors_io import read_safetensors_metadata
        return read_safetensors_metadata(file_path) or {}
    except Exception:
        return {}


def _build_nested_component_config(flat_meta: dict) -> dict:
    """Translate flat `<comp>.<key>=<val>` bundle metadata to the nested
    quantfunc_config.json structure that PipelineLoader.cpp expects, e.g.::

        flat_meta = {
            "transformer.method": "lighting_precomputed",
            "transformer.rotation_block_size": "256",
            "text_encoder.text_precision": "int4",
            "text_encoder.text_rotation_block_size": "256",
            "text_encoder.use_rotation": "true",
            "text_encoder.prequantized": "true",
            "vae.vae_precision": "auto",
        }
        →
        {
          "transformer": {"method": "lighting_precomputed",
                          "rotation_block_size": 256},
          "text_encoder": {"text_precision": "int4",
                            "text_rotation_block_size": 256,
                            "use_rotation": True,
                            "prequantized": True},
          "vae": {"vae_precision": "auto"},
        }
    """
    nested: dict = {}
    for k, v in flat_meta.items():
        # Skip top-level non-component keys
        if "." not in k:
            continue
        # Skip obfuscation-related and bundle-internal markers
        if k.startswith("obf.") or k == "format" or k == "quantfunc_obfuscated":
            continue
        # Skip the embedded full-config blob — handled separately by add_transformer
        if k == "transformer.config_json":
            continue
        comp, _, sub = k.partition(".")
        if comp not in ("transformer", "text_encoder", "vision_encoder",
                         "vae", "vae_encoder", "vae_decoder"):
            continue
        # Coerce stringy types: numerics, booleans
        cooked: object = v
        if isinstance(v, str):
            if v == "true":
                cooked = True
            elif v == "false":
                cooked = False
            else:
                # Try int / float
                try:
                    cooked = int(float(v)) if "." in v and float(v).is_integer() \
                                          else int(v)
                except (TypeError, ValueError):
                    try:
                        cooked = float(v)
                    except (TypeError, ValueError):
                        cooked = v
        nested.setdefault(comp, {})[sub] = cooked
    return nested


def _read_bundle_xfm_method(file_path: str) -> str:
    """Read `transformer.method` from a qf_flat_bundle's safetensors metadata.

    Returns "" when not present (caller picks default). Distinguishes prequant
    bundles (`lighting_precomputed`) from runtime-quant bundles where the bundle
    still holds FP16 base weights (`online_quant`). For the qf_flat_bundle
    export path the writer (src/Export.cpp::exportBundle) hardcodes
    `transformer.method=lighting_precomputed`.
    """
    try:
        from .tools.safetensors_io import read_safetensors_metadata
        meta = read_safetensors_metadata(file_path) or {}
        return meta.get("transformer.method", "")
    except Exception:
        return ""


def _detect_arch_from_bundle(file_path: str, scheme: dict) -> str:
    """Identify pipeline arch from bundle contents.

    For arch-agnostic schemes (qf_flat_bundle), fingerprint via the
    `model.diffusion_model.*` transformer keys — same logic as the
    standalone single-file UNet adapter uses. This covers ZImage / QwenImage /
    QwenImageEdit / Flux2Klein generically.

    For upstream Qwen-family schemes (qwen25_bundle / qwen2vl_bundle), fall
    back to the heuristic that distinguished QwenImage vs QwenImageEdit by
    filename "edit" hint, since both share Qwen2.5-VL TE keys.
    """
    if scheme["name"] == "qf_flat_bundle":
        # qf_flat_bundle writes `arch=<ZImage|QwenImage|QwenImageEdit|Flux2Klein>`
        # in metadata. Trust that first — the alphabetical key heuristic in
        # _detect_arch_by_keys looks at only the first 500 keys, which on a
        # 1398-key bundle may miss arch-specific signatures (cap_embedder
        # comes after `attention.*` in alphabetical order).
        try:
            from .tools.safetensors_io import read_safetensors_metadata
            meta = read_safetensors_metadata(file_path) or {}
            arch_meta = meta.get("arch", "")
            if arch_meta:
                return arch_meta
        except Exception as e:
            logger.debug("[bundle_ckpt] qf_flat arch metadata read failed: %s", e)
        # Fallback: transformer-key fingerprint.
        try:
            from .tools.arch_fingerprint import fingerprint_arch_from_keys
            arch = fingerprint_arch_from_keys(file_path) or ""
            if arch:
                return arch
        except Exception as e:
            logger.warning("[bundle_ckpt] qf_flat fingerprint failed: %s", e)
        # Fallback: filename hint
        fname = os.path.basename(file_path).lower()
        if "edit" in fname:
            return "QwenImageEdit"
        if "klein" in fname or "flux" in fname:
            return "Flux2Klein"
        if "z" in fname.split("_")[0] or "zimage" in fname.replace("-", ""):
            return "ZImage"
        return "QwenImage"

    # Upstream qwen25_bundle / qwen2vl_bundle: filename "edit" hint first,
    # else inspect content. Community-distributed Qwen-Rapid AIO bundles
    # often include the full Edit weight set (vae.encoder.* + Qwen2.5-VL
    # visual.*) without "edit" in the name — re-exporting those as T2I
    # silently drops the encoder weights, then user can't run edit mode.
    # Surface the Edit capability instead.
    fname = os.path.basename(file_path).lower()
    if "edit" in fname:
        return "QwenImageEdit"
    try:
        from .tools.safetensors_io import read_safetensors_header
        h = read_safetensors_header(file_path)
        has_vae_encoder = any(k.startswith("vae.encoder.") for k in h
                              if k != "__metadata__")
        has_visual = any(".visual." in k for k in h
                         if k != "__metadata__")
        if has_vae_encoder and has_visual:
            logger.info(
                "[bundle_ckpt] qwen25_bundle content sniff: vae.encoder + "
                "visual.* present → arch=QwenImageEdit (filename had no hint)")
            return "QwenImageEdit"
    except Exception as e:
        logger.debug("[bundle_ckpt] qwen25_bundle content sniff failed: %s", e)
    return "QwenImage"


@adapter(priority=80)
class BundledCheckpointAdapter(FormatAdapter):
    @classmethod
    def detect(cls, sources: SourceBundle) -> bool:
        if sources.checkpoint is None:
            return False
        try:
            return _select_scheme(sources.checkpoint.path) is not None
        except Exception:
            return False

    def adapt(self, sources: SourceBundle, staging_dir: Path,
              context: BuildContext) -> StagingResult:
        assert sources.checkpoint is not None
        ckpt_path = sources.checkpoint.path
        scheme = _select_scheme(ckpt_path)
        if scheme is None:
            raise RuntimeError(
                f"Bundled checkpoint detection inconsistency for {ckpt_path}")

        if scheme["name"] == "sd_legacy_bundle":
            raise NotImplementedError(
                "SD1.5/SDXL checkpoint format detected — not supported "
                "(QuantFunc engine only supports Qwen / Flux2 / ZImage). "
                "Use a Qwen-format bundled checkpoint instead.")

        arch = _detect_arch_from_bundle(ckpt_path, scheme)
        logger.info("[bundle_ckpt] scheme=%s arch=%s",
                     scheme["name"], arch)

        layout = HFLayout(staging_dir)

        # Symlink the same physical file into transformer + TE component
        # subdirs.  Each component reads through PrefixFilterProvider with its
        # own slice of the mmapped file, with on-the-fly FP8→FP16 dequant.
        # qf_flat_bundle embeds the original transformer/config.json under
        # `transformer.config_json` so variant-specific dims (Klein-4B vs
        # Klein-9B etc.) survive the round-trip; otherwise the C++ factory
        # falls back to the dominant-variant defaults and shape-asserts.
        xfm_cfg = {"_class_name": ARCH_TO_TRANSFORMER_CLASS.get(arch, "")}
        if scheme["name"] == "qf_flat_bundle":
            try:
                from .tools.safetensors_io import read_safetensors_metadata
                meta = read_safetensors_metadata(ckpt_path) or {}
                cfg_blob = meta.get("transformer.config_json", "")
                if cfg_blob:
                    import json as _json
                    xfm_cfg = _json.loads(cfg_blob)
            except Exception as e:
                logger.warning("[bundle_ckpt] couldn't restore xfm config: %s", e)
        layout.add_transformer(ckpt_path, config=xfm_cfg)
        # Bundled configs carry full hidden_size / vocab_size / num_layers etc.
        # so the C++ TE loader builds the correct model shape (Qwen2.5-VL 7B
        # for Edit, Qwen3 2.5B for non-Edit).  Fallback to minimal config if
        # bundle missing — will likely fail at load with a clear shape error.
        te_cfg = bundled_te_config(arch) or {
            "_class_name": ("Qwen2_5VLForConditionalGeneration"
                            if arch == "QwenImageEdit" else "Qwen3ForCausalLM"),
        }
        # Krea-2 fp8 TE: a bundle stages the TE as a PREFIX SLICE of this same
        # file, so guard that slice (scanning the whole file would false-refuse on
        # the transformer's own fp8). `prequantized` comes from the bundle's OWN
        # untrusted metadata and the engine reads it as skip_fp8_dequant=true —
        # which would byte-reinterpret fp8 as BF16 (silent garbage), so the guard
        # refuses that combination.
        if arch == "Krea2":
            from .tools.krea2_fp8_te import guard_krea2_te_fp8
            _flat = _read_bundle_flat_metadata(ckpt_path)
            _preq = str(_flat.get("text_encoder.prequantized", "")).lower() in (
                "true", "1", "yes")
            guard_krea2_te_fp8(ckpt_path, prefix=scheme["te"],
                               prequantized_hint=_preq)
        layout.add_text_encoder(ckpt_path, config=te_cfg)
        layout.set_key_filter("transformer", scheme["transformer"])
        layout.set_key_filter("te",          scheme["te"])
        # Vision tower: by default share TE prefix (Qwen2.5-VL has both LLM
        # head and ViT in one weight set under text_encoders.qwen25_7b.* /
        # text_encoder.* depending on scheme).  qf_flat_bundle additionally
        # supports an OPTIONAL `vision_encoder.` prefix populated only when
        # vision was independently quantized (Lighting prequant edit-mode
        # exports). Probe the file for that prefix and prefer it when present.
        ve_prefix = scheme["te"]
        if "vision" in scheme:
            try:
                from .tools.safetensors_io import has_keys_starting_with
                seen = has_keys_starting_with(ckpt_path, [scheme["vision"]])
                if scheme["vision"] in seen:
                    ve_prefix = scheme["vision"]
            except Exception:
                pass
        layout.set_key_filter("ve", ve_prefix)

        # qf_flat_bundle exports VAE with HF-diffusers naming directly (the
        # C++ engine walks Module tree which uses HF naming internally), so
        # NO ComfyUIVAEAliasProvider remap is needed. Upstream qwen25_bundle
        # / qwen2vl_bundle still hold ComfyUI-flat naming inside `vae.*` and
        # need the alias translation.
        is_qf_flat = (scheme["name"] == "qf_flat_bundle")

        # VAE: prefer caller-provided separate VAE if connected, otherwise
        # use the bundle's embedded VAE.
        vae_cfg = bundled_vae_config(arch) or {"_class_name": "AutoencoderKL"}
        if sources.vae is not None:
            from .comfyui_vae import _detect_vae_prefix
            vae_path = sources.vae.path
            # A separately-wired VAE can be the wrong channel count for this arch
            # (e.g. a QwenImageLayered qf_flat_bundle + the standard 3-ch
            # qwen_image_vae): fail LOUD instead of a cryptic engine conv_out crash.
            assert_vae_matches_arch(arch, vae_path)
            vae_prefix = _detect_vae_prefix(vae_path)
            layout.add_vae(vae_path, config=vae_cfg)
            if vae_prefix:
                layout.set_key_filter("vae", vae_prefix)
            # External standalone VAE: ComfyUI flat naming; alias-translate.
            layout.set_extra("vae_comfyui_alias", True)
            logger.info("[bundle_ckpt] using separate VAE: %s (prefix=%r)",
                         sources.vae.path, vae_prefix)
        else:
            layout.add_vae(ckpt_path, config=vae_cfg)
            layout.set_key_filter("vae", scheme["vae"])
            if not is_qf_flat:
                layout.set_extra("vae_comfyui_alias", True)
                logger.info("[bundle_ckpt] using bundle-embedded VAE via ComfyUIVAEAliasProvider")
            else:
                logger.info("[bundle_ckpt] using qf_flat bundle VAE (HF-diffusers naming, no alias)")

        # Tokenizer bundle (mandatory)
        copy_tokenizer_bundle(arch, layout.tokenizer_dir())

        # Scheduler config (optional)
        layout.add_scheduler(sources.scheduler_config, arch=arch)

        # method: qf_flat_bundle exports are already prequant (Lighting writes
        # quantized weights to the bundle); upstream FP8-mixed bundles need
        # runtime INT4 quantization at load time.
        if is_qf_flat:
            flat_meta = _read_bundle_flat_metadata(ckpt_path)
            xfm_method = flat_meta.get("transformer.method", "")
            # Keep the metadata's `transformer.method` value on disk as-is —
            # C++ engine matches against e.g. "lighting_precomputed" /
            # "lighting" / "svdq" in its quantization_config parser.
            method = xfm_method or "lighting_precomputed"
            layout.set_method(method)
            # Plugin-internal method_hint: qf_flat_bundle is always a
            # quantfunc-produced bundle, so translate the disk `method`
            # to the canonical bundle hint (auto-derive whitelist + backend
            # dispatch are table-driven on this).
            if xfm_method in ("svdq",):
                # Future qf_flat_bundle with svdq xfm (not yet produced)
                method_hint = "prequant_svdq_separate"
            else:
                method_hint = "prequant_lighting_bundle"
            # Translate flat per-component metadata into nested quantfunc_config
            # so PipelineLoader.cpp can pick up text_rotation_block_size,
            # prequantized, vision_quant, etc. (it reads nc["text_encoder"]
            # / nc["vision_encoder"] as dicts).
            nested = _build_nested_component_config(flat_meta)
            for comp_name, comp_cfg in nested.items():
                if comp_cfg:
                    layout.set_extra(comp_name, comp_cfg)
            # `model.safetensors` alias (Lighting prequant loader expects
            # exactly that filename) is now created unconditionally inside
            # HFLayout.add_transformer, no per-adapter workaround needed.
        else:
            layout.set_method("online_quant")
            method_hint = "online_quant"
            # Upstream FP8-mixed bundles (qwen25_bundle / qwen2vl_bundle) carry
            # no per-component metadata — propagate user-selected precisions
            # from BuildContext so TE/VAE actually get quantized at load time.
            # Without this the C++ TE defaults to FP16 and the export bloats
            # by ~15 GB (FP16 TE + duplicated fused buffers).
            layout.apply_user_precisions(
                text_precision=context.text_precision,
                vae_precision=context.vae_precision)

        layout.write_quantfunc_config()
        layout.write_model_index(
            arch, is_distilled=krea2_is_distilled(arch, ckpt_path))

        return StagingResult(
            model_dir=str(staging_dir),
            arch=arch,
            method_hint=method_hint,
            cleanup_dir=str(staging_dir),
        )
