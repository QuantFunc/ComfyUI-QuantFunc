"""Architecture fingerprinting from safetensors headers.

Files written by the official diffusers / our export tools usually carry
`_class_name` in their __metadata__["config"] field. ComfyUI ecosystem files
often lack this — we fall back to detecting architecture from key patterns.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .safetensors_io import (
    read_safetensors_header,
    read_safetensors_metadata,
)


# Map _class_name → our internal arch tag
CLASS_NAME_TO_ARCH = {
    "QwenImagePipeline":            "QwenImage",
    "QwenImageEditPipeline":         "QwenImageEdit",
    "QwenImageTransformer2DModel":   "QwenImage",  # may be edit, refined later
    "ZImagePipeline":                "ZImage",
    "ZImageTurboPipeline":           "ZImage",
    "Flux2KleinPipeline":            "Flux2Klein",
    "Flux2Transformer2DModel":       "Flux2Klein",
    "Ideogram4Pipeline":             "Ideogram4",
    "Ideogram4Transformer2DModel":   "Ideogram4",
    # Krea-2 / Krea-2-Turbo (single-stream 28-block MMDiT). A diffusers-layout
    # export carries the class in metadata; the common ComfyUI single-files are
    # BFL/akira layout with NO _class_name → the key-pattern arm below catches them.
    "Krea2Transformer2DModel":       "Krea2",
    "Krea2Pipeline":                 "Krea2",
    # Qwen-Image-Layered reuses QwenImageTransformer2DModel for its transformer
    # (so the TRANSFORMER class can't disambiguate — only the PIPELINE class can).
    "QwenImageLayeredPipeline":      "QwenImageLayered",
}


def _detect_arch_by_keys(keys: list[str]) -> str:
    """Heuristic match by characteristic key patterns.

    QwenImage transformers use `transformer_blocks.X.attn.to_q` (60 layers).
    ZImage diffusers uses `noise_refiner.X.attention.to_q`/`layers.X.*`.
    ZImage BFL uses `layers.X.attention.qkv` (fused) + `cap_embedder.*`
    + `context_refiner.X.*` + `final_layer.*` (no `all_` prefix).
    Flux2-Klein BFL uses `single_blocks.X` + `double_blocks.X`.
    Flux2-Klein diffusers uses `single_transformer_blocks.X` +
    `transformer_blocks.X` (8 doubles + 24 singles, distinguishable from
    QwenImage's pure `transformer_blocks.0..59`).
    """
    # Klein BFL: distinctive single_blocks + double_blocks pattern
    has_single = any(k.startswith("single_blocks.") or
                      ".single_blocks." in k for k in keys[:200])
    has_double = any(k.startswith("double_blocks.") or
                      ".double_blocks." in k for k in keys[:200])
    if has_single and has_double:
        return "Flux2Klein"

    # Klein diffusers: single_transformer_blocks.X + transformer_blocks.X
    has_single_xfm = any("single_transformer_blocks." in k for k in keys[:500])
    if has_single_xfm:
        return "Flux2Klein"

    # Krea-2 (28-block single-stream MMDiT): the BFL/akira reference layout uses
    # `txtfusion.*` (the Qwen3-VL layer-tap fusion tower: layerwise_blocks /
    # refiner_blocks / projector) — a top-level name NO other supported arch
    # carries — together with single-stream `blocks.N.attn.wq`/`blocks.N.mod.lin`
    # (adaLN-single modulation), NOT the diffusers `transformer_blocks.*` naming.
    # Checked BEFORE the generic `transformer_blocks` block-count fallback and the
    # ZImage/Qwen paths: Krea-2 shares the bare `blocks.N` prefix with Wan, but Wan
    # has `blocks.N.self_attn`/`attn1` (dual attention, no `.attn.wq`, no `.mod.lin`,
    # no `txtfusion`) — so the txtfusion + attn.wq/mod.lin double-signal is
    # collision-free vs Klein(single/double_blocks), Wan, Qwen(transformer_blocks),
    # ZImage(cap_embedder), Ideogram4(llm_cond_*), QwenImageLayered(addition_t_embedding).
    has_txtfusion = any("txtfusion." in k for k in keys)
    has_krea_block = any(
        re.search(r"(^|\.)blocks\.\d+\.(attn\.(wq|to_gate|gate)|mod\.lin)", k)
        for k in keys)
    if has_txtfusion and has_krea_block:
        return "Krea2"

    # Ideogram-4: `llm_cond_proj`/`llm_cond_norm` (the LLM-conditioning
    # projection read by Ideogram4TransformerLighting) is unique to this arch —
    # no other supported model carries it. Substring match tolerates the
    # `unconditional_transformer.` prefix of the dual-transformer checkpoints.
    # The input_proj + adaln_proj + t_embedding trio is a strong backup signal.
    # Checked BEFORE ZImage because both share `layers.X.attention.qkv`, but
    # only Ideogram-4 has llm_cond_* / the projection trio.
    has_llm_cond = any("llm_cond_proj" in k or "llm_cond_norm" in k for k in keys)
    has_ideo_trio = (any("input_proj" in k for k in keys)
                     and any("adaln_proj" in k for k in keys)
                     and any("t_embedding.mlp_" in k for k in keys))
    if has_llm_cond or has_ideo_trio:
        return "Ideogram4"

    # ZImage signature (both BFL and diffusers): cap_embedder + context_refiner.
    # Distinct from any other arch (no other model has these top-level names).
    has_cap = any("cap_embedder.0" in k or k.endswith("cap_embedder.0.weight")
                  for k in keys[:500])
    has_ctx_refiner = any("context_refiner." in k for k in keys[:500])
    if has_cap and has_ctx_refiner:
        return "ZImage"

    # Qwen-Image-Layered: a Qwen-Image variant (same QwenImageTransformer2DModel
    # class + transformer_blocks.X structure) distinguished by use_additional_t_cond,
    # which adds the `time_text_embed.addition_t_embedding` weight that base
    # Qwen-Image lacks. Checked BEFORE the generic Qwen block-count path so it
    # doesn't collapse to plain "QwenImage". NOTE: that key lives only in shard-1 —
    # `fingerprint_arch_from_keys` adds a shard-independent sibling-config fallback
    # for sharded checkpoints where another shard is read.
    if any("addition_t_embedding" in k for k in keys):
        return "QwenImageLayered"

    # Qwen / fallback: pure transformer_blocks.X (no single_transformer_blocks)
    block_indices: set[int] = set()
    for k in keys:
        m = re.search(r"transformer_blocks\.(\d+)\.", k)
        if m:
            block_indices.add(int(m.group(1)))
    if block_indices:
        max_idx = max(block_indices)
        if max_idx >= 50:
            return "QwenImage"   # 60 layers
        if max_idx >= 20:
            # Ambiguous: could be ZImage diffusers without cap_embedder
            # in the first 500 keys (sharded files often spread it). Prefer
            # ZImage if any noise_refiner / context_refiner anchor present.
            if any("noise_refiner." in k or "context_refiner." in k
                   for k in keys):
                return "ZImage"
            return "ZImage"      # 30 layers (still plausible default)
    return ""


def _is_qwen_layered_sibling(file_path: str | Path) -> bool:
    """Shard-independent Qwen-Image-Layered probe for a diffusers-dir model.

    Qwen-Image-Layered (QwenImageLayeredPipeline) reuses the base
    QwenImageTransformer2DModel class + `transformer_blocks.X` structure, so the
    only stable on-disk signal is the diffusers config flag `use_additional_t_cond`
    (its distinctive `time_text_embed.addition_t_embedding` weight lives only in
    shard-1, so a read of any other shard would mis-detect plain "QwenImage").
    Confirm via the sibling transformer config.json / the pipeline model_index.json
    sitting next to the weights. Returns False for standalone (non-diffusers) files.
    """
    try:
        p = Path(file_path)
        cfg = p.parent / "config.json"
        if cfg.is_file():
            c = json.loads(cfg.read_text(encoding="utf-8"))
            if (c.get("use_additional_t_cond") is True
                    and "QwenImageTransformer2DModel" in str(c.get("_class_name", ""))):
                return True
        mi = p.parent.parent / "model_index.json"   # <root>/transformer/<shard> -> <root>/model_index.json
        if mi.is_file():
            if json.loads(mi.read_text(encoding="utf-8")).get("_class_name") == "QwenImageLayeredPipeline":
                return True
    except Exception:
        pass
    return False


def fingerprint_arch_from_keys(file_path: str | Path) -> str:
    """Return arch tag ("QwenImage"/"QwenImageEdit"/"QwenImageLayered"/"Flux2Klein"/"ZImage"/"Ideogram4" or "")."""
    # 0. Diffusers-dir override (shard-independent): Qwen-Image-Layered can only be
    #    told apart from base Qwen-Image by its config flag, not the metadata class
    #    (shared) — and its distinctive key is shard-1-only. Check the sibling first.
    if _is_qwen_layered_sibling(file_path):
        return "QwenImageLayered"
    try:
        meta = read_safetensors_metadata(file_path)
    except Exception:
        return ""

    # 1. metadata.config._class_name
    cfg_str = meta.get("config", "")
    if cfg_str:
        try:
            cfg = json.loads(cfg_str)
            cn = cfg.get("_class_name", "")
            if cn in CLASS_NAME_TO_ARCH:
                return CLASS_NAME_TO_ARCH[cn]
        except Exception:
            pass

    # 2. metadata._class_name (some exports put it at top level)
    cn = meta.get("_class_name", "")
    if cn in CLASS_NAME_TO_ARCH:
        return CLASS_NAME_TO_ARCH[cn]

    # 3. Key fingerprint
    try:
        h = read_safetensors_header(file_path)
        keys = [k for k in h.keys() if k != "__metadata__"]
        return _detect_arch_by_keys(keys)
    except Exception:
        return ""


# QuantFunc metadata markers — presence of any one indicates our prequant export
QUANTFUNC_MARKERS = {
    "method",                     # "lighting" | "svdq"
    "quantfunc_obfuscated",       # "true" / our obfuscated export
    "text_rotation_block_size",   # H256 rotation marker
    "precision_config",           # exported precision config
}


def fingerprint_kind_from_metadata(file_path: str | Path) -> str:
    """Return one of:
        "prequant_lighting_separate"  — QuantFunc metadata present, skip quantization
        "nvfp4_disk"                  — NVIDIA NVFP4 _quantization_metadata present
        "raw_fp4"                     — first tensor is F4_E2M1 (other 4-bit native)
        "raw_fp8"                     — first tensor is F8_E4M3 / F8_E5M2
        "raw_fp8_mixed"               — F8 with mixed BF16 (UNETLoader-style)
        "raw_int8"                    — first tensor is I8
        "raw_highprec"                — F16 / BF16 / F32 → online-quant
        ""                            — unknown
    """
    try:
        meta = read_safetensors_metadata(file_path)
    except Exception:
        return ""

    # Highest priority: QuantFunc-written metadata
    if any(m in meta for m in QUANTFUNC_MARKERS):
        return "prequant_lighting_separate"

    # NVFP4 disk format (NVIDIA)
    qmd = meta.get("_quantization_metadata", "")
    if qmd:
        try:
            obj = json.loads(qmd)
            layers = obj.get("layers", {})
            if any(l.get("format") == "nvfp4" for l in layers.values()):
                return "nvfp4_disk"
        except Exception:
            pass

    # Fall back to first-tensor dtype
    try:
        from .safetensors_io import count_tensors_by_dtype
        counts = count_tensors_by_dtype(file_path)
    except Exception:
        return ""

    if not counts:
        return ""

    # 4-bit native (no metadata, raw FP4)
    if "F4_E2M1" in counts:
        return "raw_fp4"

    # FP8 variants
    has_fp8 = "F8_E4M3" in counts or "F8_E5M2" in counts
    has_bf16_or_fp16 = "BF16" in counts or "F16" in counts
    if has_fp8:
        # If majority is FP8 but BF16 also present → mixed (norms / embeddings)
        if has_bf16_or_fp16 and counts.get("BF16", 0) + counts.get("F16", 0) < sum(counts.values()) // 2:
            return "raw_fp8_mixed"
        return "raw_fp8"

    if "I8" in counts:
        return "raw_int8"

    if "BF16" in counts or "F16" in counts or "F32" in counts:
        return "raw_highprec"

    return ""
