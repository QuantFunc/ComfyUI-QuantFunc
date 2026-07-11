"""BFL/akira → engine-internal key remap for the Krea-2 single-stream transformer.

ZERO-DATA-COPY design (identical mechanism to ``zimage_bfl_remap``): the ~13 GB
safetensors is NEVER rewritten. The staging dir gets a symlink to the original
file plus a tiny ``key_remap.json`` manifest that the C++ ``KeyAliasingProvider``
reads (auto-discovered at ``<transformer>/key_remap.json`` by
``PipelineLoader``), serving zero-copy mmap views on ``getTensor(internal_key)``.

Why a remap is needed (measured):
  The common ComfyUI Krea-2 single-files (krea2_turbo_*.safetensors) carry the
  BFL/akira REFERENCE layout — ``blocks.N.attn.wq`` / ``blocks.N.mod.lin`` /
  ``txtfusion.*`` / ``first`` / ``last`` / ``tmlp`` / ``tproj`` / ``txtmlp`` —
  whereas the engine's Krea2TransformerLighting registers DIFFUSERS-internal
  names (``transformer_blocks.N.attn.to_q`` / ``ff.gate`` / ``norm1`` /
  ``scale_shift_table`` / ``img_in`` / ``txt_in`` / ``time_embed`` /
  ``time_mod_proj`` / ``final_layer`` / ``text_fusion``). The engine's base
  weight ``loadParams`` reads ONLY the internal names (the BFL ``blocks.*``
  scheme is wired only in the LoRA loader, ``ModelType::Krea2``), so a plugin
  remap is required — exactly the ``zimage_bfl_remap`` pattern.

Precision:
  * FP8 (krea2_turbo_fp8_scaled): F8_E4M3 weights + scalar ``.weight_scale``.
    The engine's fresh-quant load path wraps the source with
    ``DequantFP8Provider`` (INNER of ``KeyAliasingProvider``), so it dequantizes
    F8→BF16 using the SOURCE ``.weight_scale`` and re-quantizes to the user's
    chosen precision. => the manifest only renames ``.weight`` / norms / biases;
    the ``.weight_scale`` siblings are consumed in source-key space and need NO
    manifest entry.
  * INT8-ConvRot (krea2_turbo_int8_convrot): I8 weights + per-channel
    ``.weight_scale`` + a ``comfy_quant`` U8 descriptor encoding a ConvRot
    rotation. The engine's fresh-quant path has NO int8 dequant, and the
    rotation cannot be inverted without decoding ``comfy_quant`` — dequantizing
    naively would produce SILENT GARBAGE. So this format is fingerprint-
    recognized as Krea-2 but the remap REJECTS it fail-loud (no silent garbage).

The key map is authoritative against the engine registration
(``src/gemm/lighting/Krea2TransformerLighting.cpp``) + the LoRA ``ModelType::
Krea2`` BFL→internal table (``src/lora/LoRALoader.cpp:951-985``).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from .tools.fs_util import link_or_copy

logger = logging.getLogger(__name__)

# ONE source of truth for the engine's fp8 scale-sibling suffixes (kWeightScaleSuffixes)
from .tools.krea2_fp8_te import FP8_SCALE_SUFFIXES
from .tools.safetensors_io import read_safetensors_header


# Optional leading prefix on the source keys (bare BFL single-files have none;
# some ComfyUI exports wrap with these). Stripped for the mapping LOGIC only —
# the manifest always references the FULL original source key.
_CANDIDATE_PREFIXES = ("model.diffusion_model.", "diffusion_model.")

# BFL/akira per-block submodule tail  →  engine-internal submodule tail.
# Shared by the main image blocks (blocks.N) AND the text-fusion blocks
# (txtfusion.{layerwise,refiner}_blocks.N) — both are Krea2Block/Krea2TextFusion
# with the SAME Krea2Attention (to_q/to_k/to_v/to_gate/to_out.0 + norm_q/norm_k)
# and Krea2SwiGLU (gate/up/down) submodules. RMSNorm registers ".weight".
_BLOCK_TAIL_RENAME = {
    "attn.wq.weight":               "attn.to_q.weight",
    "attn.wk.weight":               "attn.to_k.weight",
    "attn.wv.weight":               "attn.to_v.weight",
    "attn.wo.weight":               "attn.to_out.0.weight",
    "attn.gate.weight":             "attn.to_gate.weight",
    "attn.qknorm.qnorm.scale":      "attn.norm_q.weight",
    "attn.qknorm.knorm.scale":      "attn.norm_k.weight",
    "mlp.gate.weight":              "ff.gate.weight",
    "mlp.up.weight":                "ff.up.weight",
    "mlp.down.weight":              "ff.down.weight",
    "prenorm.scale":                "norm1.weight",
    "postnorm.scale":               "norm2.weight",
    # "mod.lin"  handled separately (reshape [6H] -> [6, H] scale_shift_table)
}

# Top-level (non-block) BFL key  →  engine-internal key. The Sequential indices
# (tmlp.0/2, tproj.1, txtmlp.0/1/3) skip the non-parametric activation slots —
# verified against the real checkpoint header + the engine's ctor registration:
#   img_in / txt_in.{norm,linear_1,linear_2} / time_embed.{linear_1,linear_2} /
#   time_mod_proj / final_layer.{norm,linear,scale_shift_table}.
_TOP_RENAME = {
    "first.weight":            "img_in.weight",
    "first.bias":              "img_in.bias",
    "last.linear.weight":      "final_layer.linear.weight",
    "last.linear.bias":        "final_layer.linear.bias",
    "last.modulation.lin":     "final_layer.scale_shift_table",  # [2, H] both sides
    "last.norm.scale":         "final_layer.norm.weight",
    "tmlp.0.weight":           "time_embed.linear_1.weight",
    "tmlp.0.bias":             "time_embed.linear_1.bias",
    "tmlp.2.weight":           "time_embed.linear_2.weight",
    "tmlp.2.bias":             "time_embed.linear_2.bias",
    "tproj.1.weight":          "time_mod_proj.weight",
    "tproj.1.bias":            "time_mod_proj.bias",
    "txtmlp.0.scale":          "txt_in.norm.weight",
    "txtmlp.1.weight":         "txt_in.linear_1.weight",
    "txtmlp.1.bias":           "txt_in.linear_1.bias",
    "txtmlp.3.weight":         "txt_in.linear_2.weight",
    "txtmlp.3.bias":           "txt_in.linear_2.bias",
}

# adaLN-single per-block modulation table: source is flat [6*H]; engine registers
# scale_shift_table as [6, H]. A metadata-only reshape (row-major bytes identical).
_MODULATION_ROWS = 6

# ConvRot int8 marker sibling — presence means the (unsupported) int8_convrot
# format (see module docstring).
_COMFY_QUANT_SUFFIX = ".comfy_quant"


def _manifest_cache_dir() -> Path:
    """Stable per-user cache for built manifests (KB-scale metadata JSON — NOT
    weight copies — so this persistent location is death-rule compliant).
    Override via env ``QF_MANIFEST_CACHE_DIR``; default ``<plugin>/cache/manifests``."""
    env = os.environ.get("QF_MANIFEST_CACHE_DIR")
    if env:
        d = Path(env)
    else:
        plugin_root = Path(__file__).resolve().parent.parent
        d = plugin_root / "cache" / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(src: Path, kind: str) -> Path:
    st = src.stat()
    safe = src.name.replace("/", "_").replace(" ", "_")
    return _manifest_cache_dir() / f"{kind}_{safe}_{st.st_size}_{st.st_mtime_ns}.json"


def _read_header(src_path: Path) -> dict:
    """Return the safetensors header dict (without __metadata__).

    Reuses the SHARED reader (tools/safetensors_io.py), which carries the
    _MAX_HEADER_BYTES DoS guard — the 8-byte length prefix is UNTRUSTED, so a
    crafted file declaring an exabyte header would make a naive `f.read(n)`
    attempt a multi-GB allocation (OOM/hang). Mirrors comfyui_wan_remap.py's
    already-hardened helper; do NOT re-inline a raw struct/read here.
    """
    hdr = dict(read_safetensors_header(src_path))
    hdr.pop("__metadata__", None)
    return hdr


def _logical_key(src_key: str) -> str:
    """Strip an optional leading wrapper prefix for mapping logic."""
    for px in _CANDIDATE_PREFIXES:
        if src_key.startswith(px):
            return src_key[len(px):]
    return src_key


def is_krea2_bfl(src_path: Path | str) -> bool:
    """Detect the Krea-2 BFL/akira single-file layout (needs the remap).

    True only for the BFL layout (bare ``blocks.N.attn.wq`` + ``txtfusion.*``),
    NOT an already-diffusers Krea-2 export (``transformer_blocks.N.attn.to_q``),
    which the engine loads directly with no remap."""
    try:
        hdr = _read_header(Path(src_path))
    except Exception as e:
        logger.debug("is_krea2_bfl: header read failed: %s", e)
        return False
    keys = [_logical_key(k) for k in hdr.keys()]
    has_txtfusion = any(k.startswith("txtfusion.") for k in keys)
    has_bfl_block = any(
        re.match(r"blocks\.\d+\.(attn\.(wq|to_gate|gate)|mod\.lin)", k) for k in keys)
    # Guard: a diffusers export uses transformer_blocks.* — don't claim those.
    has_diffusers = any(k.startswith("transformer_blocks.") for k in keys)
    return has_txtfusion and has_bfl_block and not has_diffusers


def build_krea2_xfm_manifest(src_path: Path) -> dict:
    """Build the BFL→internal ``key_remap.json`` manifest (target→source).

    Returns ``{"rename": {...}, "row_slice": {}, "reshape": {...}}``. Raises on
    the unsupported int8_convrot format, or if any source weight/param key is
    left UNMAPPED (completeness guard against silently dropping a weight →
    engine empty-weight crash / wrong output).
    """
    hdr = _read_header(Path(src_path))
    rename: dict[str, str] = {}
    reshape: dict[str, dict] = {}
    unmapped: list[str] = []

    def map_block(prefix_internal: str, idx: str, tail: str, src_key: str,
                  allow_mod: bool) -> bool:
        """Map one blocks.N tail. Returns True if handled/skipped."""
        if tail in _BLOCK_TAIL_RENAME:
            rename[f"{prefix_internal}.{idx}.{_BLOCK_TAIL_RENAME[tail]}"] = src_key
            return True
        if allow_mod and tail == "mod.lin":
            shape = hdr[src_key].get("shape") or []
            numel = shape[0] if len(shape) == 1 else 0
            if numel <= 0 or numel % _MODULATION_ROWS != 0:
                raise RuntimeError(
                    f"Krea-2 remap: {src_key!r} mod.lin expected 1-D [6*H], got {shape}")
            reshape[f"{prefix_internal}.{idx}.scale_shift_table"] = {
                "source": src_key,
                "shape": [_MODULATION_ROWS, numel // _MODULATION_ROWS],
            }
            return True
        return False

    for src_key in hdr.keys():
        k = _logical_key(src_key)

        # FP8 per-tensor scale sibling — consumed by DequantFP8Provider in the
        # SOURCE key space (inner of KeyAliasingProvider); no manifest entry.
        # BOTH engine suffixes (kWeightScaleSuffixes = .scale_weight / .weight_scale):
        # the ComfyUI `_scaled` convention emits `.scale_weight`, and recognising
        # only one would send it to `unmapped` → a hard refusal of a checkpoint the
        # engine can actually dequantize.
        if any(k.endswith(sfx) for sfx in FP8_SCALE_SUFFIXES):
            continue
        # ConvRot int8 marker — the format is unsupported (see docstring).
        if k.endswith(_COMFY_QUANT_SUFFIX):
            raise RuntimeError(
                "Krea-2 remap: this checkpoint is the ComfyUI ConvRot INT8 format "
                "(has `comfy_quant` markers). Its int8 weights carry a ConvRot "
                "rotation that QuantFunc cannot invert, and the engine has no int8 "
                "dequant on the fresh-quant path — loading it would produce garbage. "
                "Use the FP8 Krea-2 checkpoint (krea2_turbo_fp8_scaled) instead.")

        # Main image blocks: blocks.N.<tail>  (adaLN-single => mod.lin present)
        m = re.match(r"blocks\.(\d+)\.(.+)$", k)
        if m and map_block("transformer_blocks", m.group(1), m.group(2), src_key,
                           allow_mod=True):
            continue

        # Text-fusion blocks: txtfusion.{layerwise,refiner}_blocks.N.<tail>
        m = re.match(r"txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(.+)$", k)
        if m and map_block(f"text_fusion.{m.group(1)}", m.group(2), m.group(3),
                           src_key, allow_mod=False):
            continue

        # Text-fusion projector.
        if k == "txtfusion.projector.weight":
            rename["text_fusion.projector.weight"] = src_key
            continue

        # Top-level.
        if k in _TOP_RENAME:
            rename[_TOP_RENAME[k]] = src_key
            continue

        unmapped.append(src_key)

    if unmapped:
        raise RuntimeError(
            "Krea-2 remap: %d source key(s) left UNMAPPED (would be silently "
            "dropped → wrong output). First few: %s" % (
                len(unmapped), unmapped[:8]))

    return {"rename": rename, "row_slice": {}, "reshape": reshape}


def stage_krea2(src_path: str, staging_transformer_dir: Path,
                *, force: bool = False) -> Path:
    """Stage a Krea-2 BFL transformer with zero data copy.

    Writes:
      <staging_transformer_dir>/diffusion_pytorch_model.safetensors  (symlink)
      <staging_transformer_dir>/key_remap.json                       (~KB manifest)
    """
    src = Path(src_path).resolve()
    dst = staging_transformer_dir / "diffusion_pytorch_model.safetensors"
    manifest_path = staging_transformer_dir / "key_remap.json"

    if (dst.exists() or dst.is_symlink()) and force:
        dst.unlink()
    if not (dst.exists() or dst.is_symlink()):
        staging_transformer_dir.mkdir(parents=True, exist_ok=True)
        link_or_copy(src, dst)

    if manifest_path.exists() and not force:
        return dst

    cache_path = _cache_key(src, "krea2_xfm")
    if cache_path.exists() and not force:
        try:
            os.symlink(str(cache_path), str(manifest_path))
        except OSError:
            import shutil
            shutil.copyfile(str(cache_path), str(manifest_path))
        logger.info("[comfyui_krea2_remap] %s → manifest from cache", src.name)
        return dst

    manifest = build_krea2_xfm_manifest(src)
    cache_path.write_text(json.dumps(manifest, indent=2))
    try:
        os.link(str(cache_path), str(manifest_path))
    except OSError:
        import shutil
        shutil.copyfile(str(cache_path), str(manifest_path))
    logger.info("[comfyui_krea2_remap] %s → manifest built+cached "
                "(%d renames, %d reshapes, %.1f KB)",
                src.name, len(manifest["rename"]), len(manifest["reshape"]),
                cache_path.stat().st_size / 1024)
    return dst
