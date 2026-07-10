"""ComfyUI / original-Wan single-file → diffusers-key remap + two-expert (A14B) staging.

Wan2.2-A14B ships as TWO separate single-file checkpoints (a *high-noise* expert
and a *low-noise* expert), NOT a diffusers dir. The QuantFunc engine, however,
loads the A14B two-expert architecture from a diffusers *model_dir* that contains
`transformer/` (high) + `transformer_2/` (low) + shared `text_encoder/` `vae/`
`tokenizer/` `scheduler/` + a `model_index.json` carrying `boundary_ratio`
(the mid-denoise expert-switch threshold).

This module bridges the two: it stages the two single-file experts into that
diffusers layout so `QuantFuncModelLoader` / `QuantFunc Build Pipeline` can load
them unchanged. Three concerns are handled:

  1. KEY REMAP — the ComfyUI/original-Wan single-file uses original-Wan key
     naming (`blocks.N.self_attn/cross_attn/ffn.0/text_embedding/time_projection/
     head.head/modulation`); the engine wants DIFFUSERS keys (`blocks.N.attn1/
     attn2/ffn.net.0.proj/condition_embedder.*/proj_out/scale_shift_table`).
     `remap_key()` is derived empirically and verified key-exact against the
     diffusers Wan2.2-A14B reference (1908 single-file tensors → 1095 diffusers
     keys, 0 missing / 0 extra; the fp8 `.scale_weight`/`.scale_input` siblings
     and the `scaled_fp8` marker are dropped — the engine reads scales inline).

  2. FP8 → FP16 DEQUANT — ComfyUI `*_fp8_scaled` experts store the attention/FFN
     linears as F8_E4M3 with per-tensor `.scale_weight`. The engine's Wan video
     transformer factory does not (yet) wire the shared DequantFP8Provider, so
     fp8 weights would load into fp16 slots (size mismatch). Until that 1-line
     engine wrap lands, we dequant to fp16 on the plugin side (the ONLY staging
     mode). Once the engine wires DequantFP8 + a KeyAliasingProvider for the Wan
     factory, the zero-copy `key_remap.json` manifest (`build_wan_xfm_manifest`)
     becomes usable — no data copy, fp8 preserved.

  3. VAE ABSENT-KEY FIX — the Wan2.1 16-ch VAE (used by the whole Wan-14B family)
     ships a diffusers `config.json` that OMITS `decoder_base_dim` / `is_residual`
     / `patch_size`. The engine mis-defaults ABSENT keys to Wan2.2/5B values
     (decoder_base_dim 256, is_residual true, patch_size 2) → `conv_in [1024]` vs
     the real `[384]` → fatal shape mismatch. We synthesize a config that makes
     these keys EXPLICIT (mirroring the diffusers `AutoencoderKLWan` defaults for
     absent keys: decoder_base_dim=base_dim, is_residual=false, patch_size=1) so
     the engine parser reads them correctly. Only-fill-if-absent → byte-safe for a
     VAE that already carries them (Wan2.2/5B).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import struct
import time
from contextlib import contextmanager
from pathlib import Path

from .tools.fs_util import link_or_copy
from .tools.safetensors_io import read_safetensors_header, _MAX_HEADER_BYTES

logger = logging.getLogger(__name__)


# ============================================================================
# Key remap  (original-Wan / ComfyUI single-file  →  diffusers)
# ============================================================================

def remap_key(k: str) -> str:
    """Translate one original-Wan key to its diffusers-WanTransformer3DModel name.

    Pass-through for keys already diffusers-shaped or engine-neutral
    (`patch_embedding.*`). Verified key-exact vs the diffusers A14B reference.
    """
    m = re.match(r"blocks\.(\d+)\.(.*)", k)
    if m:
        n, rest = m.group(1), m.group(2)
        rest = re.sub(r"^self_attn\.norm_([qk])\.", r"attn1.norm_\1.", rest)
        rest = re.sub(r"^self_attn\.o\.", "attn1.to_out.0.", rest)
        rest = re.sub(r"^self_attn\.([qkv])\.", r"attn1.to_\1.", rest)
        rest = re.sub(r"^cross_attn\.norm_([qk])\.", r"attn2.norm_\1.", rest)
        rest = re.sub(r"^cross_attn\.o\.", "attn2.to_out.0.", rest)
        rest = re.sub(r"^cross_attn\.([qkv])\.", r"attn2.to_\1.", rest)
        rest = re.sub(r"^ffn\.0\.", "ffn.net.0.proj.", rest)
        rest = re.sub(r"^ffn\.2\.", "ffn.net.2.", rest)
        rest = re.sub(r"^norm3\.", "norm2.", rest)
        rest = re.sub(r"^modulation$", "scale_shift_table", rest)
        return f"blocks.{n}.{rest}"
    tbl = {
        "text_embedding.0.": "condition_embedder.text_embedder.linear_1.",
        "text_embedding.2.": "condition_embedder.text_embedder.linear_2.",
        "time_embedding.0.": "condition_embedder.time_embedder.linear_1.",
        "time_embedding.2.": "condition_embedder.time_embedder.linear_2.",
        "time_projection.1.": "condition_embedder.time_proj.",
        "head.head.": "proj_out.",
    }
    for a, b in tbl.items():
        if k.startswith(a):
            return b + k[len(a):]
    if k == "head.modulation":
        return "scale_shift_table"
    return k  # patch_embedding.*, and anything already diffusers-shaped


def _remap_wan_vae_resnet_subkey(r: str) -> str:
    """Original-Wan VAE resnet sub-keys -> diffusers AutoencoderKLWan names.

    The original block is an nn.Sequential `residual` ([norm,SiLU,conv,norm,SiLU,
    dropout,conv] -> indices 0/2/3/6) + optional `shortcut`; diffusers names them
    norm1/conv1/norm2/conv2/conv_shortcut.
    """
    r = re.sub(r"^residual\.0\.gamma$", "norm1.gamma", r)
    r = re.sub(r"^residual\.2\.", "conv1.", r)
    r = re.sub(r"^residual\.3\.gamma$", "norm2.gamma", r)
    r = re.sub(r"^residual\.6\.", "conv2.", r)
    r = re.sub(r"^shortcut\.", "conv_shortcut.", r)
    return r


def remap_wan_vae_key(k: str) -> str:
    """Translate one original-Wan / ComfyUI VAE key to its diffusers
    AutoencoderKLWan name (the naming the engine's Wan VAE loader reads).

    Derived empirically from the ComfyUI `wan2.2_vae.safetensors` (196 keys,
    original naming: conv1/conv2, {en,de}coder.{conv1,head,middle,{up,down}samples})
    vs the diffusers `Wan2.2-TI2V-5B-Diffusers/vae` reference (196 keys:
    {quant,post_quant}_conv, conv_{in,out}, norm_out, mid_block, {up,down}_blocks)
    and VALIDATED key-exact + shape-exact against that reference (see
    tests/test_wan_5b_single_file.py). Structure-driven (resample/time_conv =>
    the block's {up,down}sampler; residual/shortcut => resnets.J), so it holds
    for any Wan VAE using the original naming, not just the 5B channel widths.
    Pass-through for keys already diffusers-shaped.
    """
    if k.startswith("conv1."):
        return "quant_conv." + k[len("conv1."):]
    if k.startswith("conv2."):
        return "post_quant_conv." + k[len("conv2."):]
    m = re.match(r"^(encoder|decoder)\.(.*)$", k)
    if not m:
        return k
    side, rest = m.group(1), m.group(2)
    if rest.startswith("conv1."):
        rest = "conv_in." + rest[len("conv1."):]
    elif rest == "head.0.gamma":
        rest = "norm_out.gamma"
    elif rest.startswith("head.2."):
        rest = "conv_out." + rest[len("head.2."):]
    else:
        mm = re.match(r"^middle\.(\d)\.(.*)$", rest)
        if mm:
            i, sub = int(mm.group(1)), mm.group(2)
            if i == 1:   # attention block: norm/to_qkv/proj names are shared
                rest = f"mid_block.attentions.0.{sub}"
            else:        # middle.0 -> resnets.0, middle.2 -> resnets.1
                rest = (f"mid_block.resnets.{0 if i == 0 else 1}."
                        f"{_remap_wan_vae_resnet_subkey(sub)}")
        else:
            mu = re.match(r"^upsamples\.(\d+)\.upsamples\.(\d+)\.(.*)$", rest)
            md = re.match(r"^downsamples\.(\d+)\.downsamples\.(\d+)\.(.*)$", rest)
            if mu:
                blk, j, sub = mu.group(1), mu.group(2), mu.group(3)
                if sub.startswith(("resample.", "time_conv.")):
                    rest = f"up_blocks.{blk}.upsampler.{sub}"
                else:
                    rest = (f"up_blocks.{blk}.resnets.{j}."
                            f"{_remap_wan_vae_resnet_subkey(sub)}")
            elif md:
                blk, j, sub = md.group(1), md.group(2), md.group(3)
                if sub.startswith(("resample.", "time_conv.")):
                    rest = f"down_blocks.{blk}.downsampler.{sub}"
                else:
                    rest = (f"down_blocks.{blk}.resnets.{j}."
                            f"{_remap_wan_vae_resnet_subkey(sub)}")
    return f"{side}.{rest}"


def _is_scale_sibling(k: str) -> bool:
    return k.endswith(".scale_weight") or k.endswith(".scale_input")


def _is_droppable(k: str) -> bool:
    """Keys that must NOT appear in the staged diffusers file: the ComfyUI fp8
    marker + the per-tensor scale siblings (consumed inline during dequant)."""
    return k == "scaled_fp8" or _is_scale_sibling(k)


# ============================================================================
# safetensors header IO  (stdlib only — no torch needed for detect/manifest)
# ============================================================================

def _read_header(path: str | Path) -> dict:
    # Reuse the shared reader (carries a _MAX_HEADER_BYTES DoS guard); drop the
    # __metadata__ pseudo-entry so callers iterate tensor keys only.
    hdr = dict(read_safetensors_header(path))
    hdr.pop("__metadata__", None)
    return hdr


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ============================================================================
# Detection + modality
# ============================================================================

def is_comfyui_wan_single_file(path: str | Path) -> bool:
    """True for an original-Wan single-file transformer (t2v or i2v expert).

    Signature: `blocks.N.self_attn.*` + `blocks.N.ffn.0.*` + a `patch_embedding`
    (original-Wan naming, pre-remap). Cheap header-only read.
    """
    try:
        hdr = _read_header(path)
    except Exception as e:  # noqa: BLE001
        logger.debug("is_comfyui_wan_single_file: header read failed: %s", e)
        return False
    keys = list(hdr.keys())
    has_self_attn = any(re.search(r"^blocks\.\d+\.self_attn\.", k) for k in keys)
    has_ffn0 = any(re.search(r"^blocks\.\d+\.ffn\.0\.", k) for k in keys)
    has_patch = any(k.startswith("patch_embedding") for k in keys)
    return has_self_attn and has_ffn0 and has_patch


def detect_wan_modality(path: str | Path) -> tuple[int, int]:
    """Return (in_channels, out_channels) for a Wan expert.

    in_channels  = `patch_embedding.weight` shape[1]  (Conv3d [out,in,kt,kh,kw]).
    out_channels = `head.head.weight` / `proj_out.weight` shape[0] // (pt*ph*pw).
                   Falls back to in_channels when the head shape can't be resolved
                   (t2v: in == out). i2v A14B is in>out (channel-concat).
    """
    hdr = _read_header(path)
    # ONE pass over the header: in_channels + patch volume (kt*kh*kw) both come
    # from the same patch_embedding.weight Conv3d shape [out,in,kt,kh,kw].
    in_ch, pt, ph, pw = 0, 1, 1, 1
    for k, info in hdr.items():
        if k.startswith("patch_embedding") and k.endswith(".weight"):
            shp = info["shape"]
            if len(shp) == 5:
                in_ch, pt, ph, pw = shp[1], shp[2], shp[3], shp[4]
            break
    if not in_ch:
        raise RuntimeError(f"{path}: no patch_embedding.weight — not a Wan expert?")
    out_ch = in_ch
    for k in ("head.head.weight", "proj_out.weight"):
        if k in hdr:
            rows = hdr[k]["shape"][0]
            vol = max(pt * ph * pw, 1)
            if rows % vol == 0:
                out_ch = rows // vol
            break
    return in_ch, out_ch


def _count_layers(hdr: dict) -> int:
    idx = -1
    for k in hdr:
        m = re.match(r"blocks\.(\d+)\.", k)
        if m:
            idx = max(idx, int(m.group(1)))
    return idx + 1


# ============================================================================
# Zero-copy manifest builder (FUTURE — engine KeyAliasingProvider + DequantFP8)
# ============================================================================

def build_wan_xfm_manifest(src_path: str | Path) -> dict:
    """Build a `key_remap.json` manifest (diffusers_key → source_key rename).

    Consumed by the engine's KeyAliasingProvider for a ZERO-DATA-COPY view of the
    single-file. Only valid once the Wan transformer factory wires both the
    KeyAliasingProvider AND the DequantFP8Provider (so fp8 blocks load directly).
    Until then use the dequant-rewrite path (`remap_dequant_file`).
    """
    hdr = _read_header(src_path)
    rename: dict[str, str] = {}
    for src_key in hdr:
        if _is_droppable(src_key):
            continue
        rename[remap_key(src_key)] = src_key
    return {"rename": rename, "row_slice": {}, "reshape": {}}


# ============================================================================
# Physical rewrites
# ============================================================================

# safetensors dtype → element byte width (for planning the streamed output header).
_ST_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
                   "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U64": 8, "U32": 4, "U16": 2,
                   "U8": 1, "BOOL": 1}


def remap_dequant_file(src: str | Path, dst: str | Path,
                       key_fn=remap_key,
                       extra_drop: tuple = ()) -> int:
    """Remap keys AND dequant fp8 (F8_E4M3 × scale_weight → fp16). Needs torch.

    `key_fn` maps each kept source key to its output name (default: the Wan
    TRANSFORMER remap). `extra_drop` names additional exact keys to omit beyond
    the fp8 scale siblings/marker (e.g. ComfyUI's embedded `spiece_model` blob
    in the umt5 text-encoder single-file). Defaults are byte-identical to the
    original two-argument behaviour.

    The working path for the current engine (Wan factory has no DequantFP8). The
    scale siblings + `scaled_fp8` marker are consumed here and NOT written.

    TRULY memory-bounded: each tensor is read with a plain `seek`+`read` of just its
    byte slice (NO persistent whole-file mmap) and the output safetensors is written
    incrementally, freeing each tensor. Peak RSS scales with ONE tensor's working set
    (its fp8 bytes + fp32 transient + product + torch runtime) — MEASURED ~2.1 GB
    total converting a real 14 GB A14B expert, independent of model size — vs a naive
    load-all-and-accumulate that needed ~70 GB, and an mmap variant that kept the
    whole ~14 GB source page-cache-resident. Safe on a modest-RAM host.
    """
    import torch

    src, dst = str(src), str(dst)
    st2torch = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
                "BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn,
                "I64": torch.int64, "I32": torch.int32, "I16": torch.int16,
                "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
    # F8_E5M2 only when this torch actually has the dtype — silently reinterpreting
    # E5M2 bits as E4M3 would be numerically wrong; absent → the plan loop's
    # unsupported-dtype check below fails LOUD instead.
    if hasattr(torch, "float8_e5m2"):
        st2torch["F8_E5M2"] = torch.float8_e5m2
    total = os.path.getsize(src)
    with open(src, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        if hlen > _MAX_HEADER_BYTES or hlen > total - 8:
            raise RuntimeError(f"{src}: implausible safetensors header length {hlen}")
        raw = json.loads(fh.read(hlen))
    raw.pop("__metadata__", None)
    data_start = 8 + hlen
    data_region = total - data_start

    def _slice(k, info):
        off = info.get("data_offsets")
        if not (isinstance(off, (list, tuple)) and len(off) == 2
                and isinstance(off[0], int) and isinstance(off[1], int)
                and 0 <= off[0] <= off[1] <= data_region):
            raise RuntimeError(f"{src}: tensor {k!r} data_offsets {off!r} out of range")
        return off[0], off[1]

    # Plan the output header + per-tensor read plan (source order → kept, fp8→F16).
    scale_slices = {k: (_slice(k, v), v["dtype"])
                    for k, v in raw.items() if k.endswith(".scale_weight")}
    # ComfyUI's fp8-scaled marker: when present, EVERY fp8 .weight must carry a
    # .scale_weight sibling — a missing one means the checkpoint is malformed and a
    # bare fp8→fp16 cast would be ~20× wrong. Fail LOUD at plan time (before any write).
    has_scaled_marker = "scaled_fp8" in raw
    out_hdr: dict = {}
    plan: list = []   # (out_key, is_fp8, src_dt, (s,e), scale_key_or_None, out_bytes)
    cursor = 0
    for k, info in raw.items():
        if _is_droppable(k) or k in extra_drop:
            continue
        src_dt = info["dtype"]
        if src_dt not in st2torch:
            raise RuntimeError(f"{src}: tensor {k!r} unsupported dtype {src_dt!r}")
        is_fp8 = src_dt in ("F8_E4M3", "F8_E5M2")
        out_dt = "F16" if is_fp8 else src_dt
        s, e = _slice(k, info)
        numel = 1
        for sh in info["shape"]:
            if not isinstance(sh, int) or sh < 0:
                raise RuntimeError(f"{src}: tensor {k!r} invalid shape entry {sh!r}")
            numel *= sh
        size = numel * _ST_DTYPE_BYTES[out_dt]
        sk = (k[: -len(".weight")] + ".scale_weight"
              if is_fp8 and k.endswith(".weight") else None)
        if has_scaled_marker and sk is not None and sk not in scale_slices:
            raise RuntimeError(
                f"{src}: fp8 tensor {k!r} carries the 'scaled_fp8' marker but has NO "
                f"'{sk}' sibling — a bare fp8→fp16 cast would be numerically wrong "
                f"(unscaled). The checkpoint is malformed; refusing to dequantize.")
        out_key = key_fn(k)
        out_hdr[out_key] = {"dtype": out_dt, "shape": info["shape"],
                            "data_offsets": [cursor, cursor + size]}
        plan.append((out_key, is_fp8, src_dt, (s, e), sk, size))
        cursor += size

    nh = json.dumps(out_hdr, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(nh) % 8)) % 8
    nh += b" " * pad
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    def _read(fh, s, e, st_dt):
        fh.seek(data_start + s)
        # raw bytes → uint8 tensor → reinterpret as the stored dtype (no numpy fp8 need)
        return torch.frombuffer(bytearray(fh.read(e - s)),
                                dtype=torch.uint8).view(st2torch[st_dt])

    with open(src, "rb") as fh, open(dst, "wb") as out:
        out.write(struct.pack("<Q", len(nh)))
        out.write(nh)
        for out_key, is_fp8, src_dt, (s, e), sk, size in plan:
            t = _read(fh, s, e, src_dt)
            if is_fp8:
                w = t.to(torch.float32)
                # The plan loop already REJECTED a marker-carrying file with a missing
                # scale sibling (fail-loud); here a sibling absent from scale_slices can
                # only mean a legitimately-unscaled fp8 file (no 'scaled_fp8' marker).
                # A real READ error on an existing sibling still propagates.
                if sk is not None and sk in scale_slices:
                    (ss, se), sdt = scale_slices[sk]
                    w = w * _read(fh, ss, se, sdt).to(torch.float32)
                t = w.to(torch.float16)
            b = t.contiguous().view(torch.uint8).numpy().tobytes()
            if len(b) != size:
                raise RuntimeError(
                    f"{src}: {out_key} produced {len(b)} bytes != planned {size}")
            out.write(b)
            del t
    return len(plan)


def estimate_dequant_output_bytes(src: str | Path, extra_drop: tuple = ()) -> int:
    """Exact DATA bytes `remap_dequant_file` will write for `src` (kept tensors,
    fp8→F16 widening applied) — the same plan math, header-only, no torch. Used for
    the free-disk pre-check; the output json header (~100 KB) rides in the caller's
    safety margin."""
    hdr = _read_header(src)   # DoS-guarded
    total = 0
    for k, info in hdr.items():
        if _is_droppable(k) or k in extra_drop:
            continue
        src_dt = info["dtype"]
        if src_dt not in _ST_DTYPE_BYTES:
            raise RuntimeError(f"{src}: tensor {k!r} unsupported dtype {src_dt!r}")
        out_dt = "F16" if src_dt in ("F8_E4M3", "F8_E5M2") else src_dt
        numel = 1
        for sh in info["shape"]:
            if not isinstance(sh, int) or sh < 0:
                raise RuntimeError(f"{src}: tensor {k!r} invalid shape entry {sh!r}")
            numel *= sh
        total += numel * _ST_DTYPE_BYTES[out_dt]
    return total


# ============================================================================
# Config synthesis
# ============================================================================

# Published boundary_ratio defaults per modality — VERIFIED 2026-07-02 against the
# OFFICIAL upstream model_index.json of both A14B releases (two independent sources,
# huggingface.co + modelscope.cn, byte-identical):
#   Wan-AI/Wan2.2-T2V-A14B-Diffusers → "boundary_ratio": 0.875
#   Wan-AI/Wan2.2-I2V-A14B-Diffusers → "boundary_ratio": 0.9   (also the local ref)
# Used ONLY when neither the user nor the shared model_index supplies a value.
_PUBLISHED_BOUNDARY_T2V = 0.875
_PUBLISHED_BOUNDARY_I2V = 0.9

# The synthesized model_index ALWAYS carries _class_name="WanPipeline" — for BOTH
# t2v and i2v experts. Rationale (engine-verified 2026-07-02):
#   * The engine's family detect (wan_detect, WanVideoPipeline.cpp) accepts ANY
#     `Wan…`-prefixed pipeline class (`in.pipeline_class.rfind("Wan",0)==0`), so
#     "WanPipeline" is loadable everywhere (as is the published
#     "WanImageToVideoPipeline"). We synthesize from scratch and have no published
#     class to preserve, so we write the canonical "WanPipeline".
#   * t2v-vs-i2v behavior is CHANNEL-driven in the engine (is_i2v = in_channels >
#     latent channels; the VAE encoder loads when xfm_in > xfm_out), so the modality
#     information lives in the transformer config's in/out_channels we synthesize —
#     the pipeline-level class string plays no role in it.
# We own this synthesized file, so we write the canonical loadable value.
_ENGINE_WAN_PIPELINE_CLASS = "WanPipeline"


def synthesize_transformer_config(base_config: dict, in_ch: int, out_ch: int,
                                  num_layers: int) -> dict:
    """Base diffusers WanTransformer3DModel config with the expert's real
    channel/depth dims applied (so a t2v expert isn't mislabelled i2v etc.)."""
    cfg = dict(base_config)
    cfg["_class_name"] = "WanTransformer3DModel"
    cfg["in_channels"] = in_ch
    cfg["out_channels"] = out_ch
    if num_layers > 0:
        cfg["num_layers"] = num_layers
    return cfg


def synthesize_vae_config(base_vae_config: dict) -> dict:
    """Make the Wan2.1 16-ch VAE's ABSENT decoder keys explicit (only-fill-if-
    absent → byte-safe for a VAE that already carries them, e.g. Wan2.2/5B)."""
    cfg = dict(base_vae_config)
    base_dim = cfg.get("base_dim", 96)
    cfg.setdefault("decoder_base_dim", base_dim)
    cfg.setdefault("is_residual", False)
    cfg.setdefault("patch_size", 1)
    cfg.setdefault("out_channels", 3)
    cfg.setdefault("scale_factor_temporal", 4)
    cfg.setdefault("scale_factor_spatial", 8)
    return cfg


def synthesize_model_index(base_model_index: dict | None, class_name: str,
                           boundary_ratio: float) -> dict:
    """Two-expert Wan model_index with `boundary_ratio` (mid-denoise expert
    switch) + both transformer entries + the shared component wiring."""
    mi = dict(base_model_index) if base_model_index else {}
    mi["_class_name"] = class_name
    mi["boundary_ratio"] = float(boundary_ratio)
    mi.setdefault("_diffusers_version", "0.35.0.dev0")
    mi["transformer"] = ["diffusers", "WanTransformer3DModel"]
    mi["transformer_2"] = ["diffusers", "WanTransformer3DModel"]
    mi.setdefault("vae", ["diffusers", "AutoencoderKLWan"])
    mi.setdefault("text_encoder", ["transformers", "UMT5EncoderModel"])
    mi.setdefault("tokenizer", ["transformers", "T5TokenizerFast"])
    mi.setdefault("scheduler", ["diffusers", "UniPCMultistepScheduler"])
    return mi


# ============================================================================
# Staging
# ============================================================================

_SHARED_SUBDIRS = ("text_encoder", "tokenizer", "scheduler")


def _fingerprint(paths: list[str], extra: str = "") -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode())
        try:
            st = os.stat(p)
            h.update(str(st.st_size).encode())
            h.update(str(int(st.st_mtime)).encode())
        except OSError:
            pass
    h.update(extra.encode())
    return h.hexdigest()[:16]


def _stage_shared_dir(src: str, dst: str) -> None:
    """Materialise a shared-component DIRECTORY at `dst` pointing at `src`: a symlink
    (zero-copy) where the OS permits unprivileged dir symlinks, else a copy (Windows
    without Developer Mode — `link_or_copy` is file-only, so dirs handle it here).
    Only ever removes a pre-existing SYMLINK at `dst`; a real directory already there
    (a prior copy into this — guarded — out_dir) is kept idempotently, never deleted."""
    if os.path.islink(dst):
        os.unlink(dst)               # removing a symlink never touches its target
    elif os.path.isdir(dst):
        return                        # already materialised by a prior stage of this dir
    try:
        os.symlink(src, dst)
    except OSError:                   # e.g. Windows without Developer Mode
        shutil.copytree(src, dst)


def _is_within(child: str, parent: str) -> bool:
    """True if `parent` is `child` or an ancestor of it (realpath-normalized).
    Uses commonpath (not string prefix) so edge cases like parent="/" or a shared
    leading name fragment are handled correctly; different drives → not within."""
    try:
        child = os.path.realpath(child)
        parent = os.path.realpath(parent)
        return os.path.commonpath([child, parent]) == parent
    except ValueError:
        return False   # e.g. different Windows drives / mixed abs+rel


def _assert_safe_out_dir(out_dir: str, protected: list[str]) -> None:
    """Guard the destructive staging writes: (1) `out_dir` must not overlap (equal /
    ancestor / descendant of) any source path — else a re-stage could clobber the
    experts / shared model; (2) an existing NON-EMPTY `out_dir` must carry our own
    `.qf_stage_complete` marker — refuse to write into a directory of unknown
    (user-owned) content."""
    for p in protected:
        if not p:
            continue
        if _is_within(out_dir, p) or _is_within(p, out_dir):
            raise RuntimeError(
                f"refusing to stage into {out_dir!r}: it overlaps a source path {p!r}. "
                f"Choose an output_dir separate from the experts and shared components.")
    if os.path.isdir(out_dir) and os.listdir(out_dir) \
            and not os.path.isfile(os.path.join(out_dir, ".qf_stage_complete")):
        raise RuntimeError(
            f"refusing to stage into non-empty {out_dir!r} that was not created by this "
            f"node (no .qf_stage_complete marker) — it may hold your own files; choose an "
            f"empty / dedicated output_dir.")


def resolve_boundary_ratio(requested, base_model_index: dict | None,
                           in_ch: int, out_ch: int,
                           shared_modality: tuple[int, int] | None = None
                           ) -> tuple[float, str]:
    """Resolve the effective boundary_ratio + a provenance label.

    Precedence (C1 — never silently override the model's published value):
      1. explicit `requested` (user override) — must be in (0, 1]; 0/negative is
         REJECTED (the engine's `boundary_ratio > 0` gate would silently drop the
         28 GB low-noise expert — C2);
      2. the shared model_index's own `boundary_ratio` (the published value) —
         inherited ONLY when the shared dir's modality matches the experts'
         (`shared_modality` = the shared transformer config's (in,out) channels).
         The boundary is MODALITY-SPECIFIC (t2v 0.875 vs i2v 0.9) while the shared
         components (vae/TE/tokenizer/scheduler) are byte-identical across the two
         A14B releases — so pairing t2v experts with the i2v diffusers dir is a
         legitimate, common setup whose model_index carries the OTHER modality's
         boundary. Inheriting it silently would be wrong → fall through to the
         experts' own published default, with a loud warning;
      3. the experts' modality's published default (t2v 0.875 / i2v 0.9).
    """
    if requested is not None:
        r = float(requested)
        if not (0.0 < r <= 1.0):
            raise RuntimeError(
                f"boundary_ratio={r} is invalid: must be in (0, 1]. A value of 0 would "
                f"make the engine silently ignore the staged low-noise expert "
                f"(single-expert gate). Use None/auto to inherit the published value.")
        return r, "explicit override"
    experts_i2v = in_ch > out_ch
    if base_model_index:
        b = base_model_index.get("boundary_ratio")
        if isinstance(b, (int, float)) and 0.0 < float(b) <= 1.0:
            if (shared_modality is not None
                    and (shared_modality[0] > shared_modality[1]) != experts_i2v):
                logger.warning(
                    "[comfyui_wan_remap] shared_components model_index carries "
                    "boundary_ratio=%s but its transformer config is the OTHER "
                    "modality (shared in/out=%s vs experts %s) — NOT inheriting; "
                    "using the experts' published %s default instead.",
                    b, shared_modality, (in_ch, out_ch),
                    "i2v" if experts_i2v else "t2v")
            else:
                return float(b), "inherited from shared model_index"
    if experts_i2v:
        return _PUBLISHED_BOUNDARY_I2V, "published i2v default"
    return _PUBLISHED_BOUNDARY_T2V, "published t2v default"


# Free-disk safety margin over the EXACT planned output bytes (covers the output
# json headers ~100 KB each, configs, and filesystem overhead). 2 GiB.
_FREE_SPACE_MARGIN_BYTES = 2 * 1024 ** 3

# Sentinel written FIRST into a staging tmp dir so stale-tmp cleanup can prove the
# dir is OURS before removing it (never delete by name-pattern alone).
_TMP_SENTINEL = ".qf_stage_tmp"

# Suffix of the per-out_dir exclusive staging lockfile (SF1).
_STAGE_LOCK_SUFFIX = ".lock"

# Rollback-rename retry budget for a swap-in failure: covers TRANSIENT fault classes
# (e.g. a Windows AV scanner briefly holding the dir) — a PERSISTENT fault (ro-remount,
# EIO) cannot be retried away and degrades to the loud compound-fault error below.
_ROLLBACK_RETRIES = 3
_ROLLBACK_RETRY_DELAY_S = 0.1


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness, used only in the SAFE direction (a pid that looks alive
    just SKIPS a cleanup — a disk leak, never a deletion of live work).
    POSIX: signal-0 probe (EPERM ⇒ alive, ESRCH ⇒ dead). Windows: OpenProcess query
    — NEVER os.kill(pid, 0) there, which calls TerminateProcess(pid, exit_code=0)
    and would KILL a live stager (the sig arg is the exit code on Windows)."""
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                ERROR_ACCESS_DENIED = 5
                return k32.GetLastError() == ERROR_ACCESS_DENIED  # denied ⇒ alive
            try:
                code = ctypes.c_ulong()
                if k32.GetExitCodeProcess(h, ctypes.byref(code)):
                    return code.value == STILL_ACTIVE
                return True     # unknown → assume alive (never reap on ambiguity)
            finally:
                k32.CloseHandle(h)
        except Exception:  # noqa: BLE001
            return True         # unknown → assume alive (never reap on ambiguity)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True    # unknown → assume alive (never reap on ambiguity)
    return True


@contextmanager
def _stage_lock(out_dir: str):
    """SF1 — EXCLUSIVE whole-stage lock on `<out_dir>.lock`, serializing every
    concurrent stager of the SAME out_dir (the unit of contention: the node derives
    out_dir from the source fingerprint, and an explicit output_dir is one shared
    target). Instance B BLOCKS until A finishes, then sees A's completed marker →
    cache-hit — B can never reap A's live tmp nor read a half-swapped dir. Uses
    flock (POSIX) / msvcrt.locking (Windows); the OS releases the lock automatically
    when the holder dies, so a crashed holder never wedges the next run. The tiny
    lockfile is left in place (unlinking it would race a waiter)."""
    # realpath-normalize so two literal spellings of the same (possibly symlinked)
    # out_dir contend on ONE lock; O_NOFOLLOW (where supported) refuses a pre-planted
    # symlink at the predictable lock path (shared-tmp hardening).
    lock_path = os.path.realpath(out_dir) + _STAGE_LOCK_SUFFIX
    parent = os.path.dirname(lock_path) or "."
    os.makedirs(parent, exist_ok=True)
    fd = os.open(lock_path,
                 os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o644)
    locked_msvcrt = False
    try:
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)     # blocks indefinitely; freed on death
        except ImportError:                    # Windows
            import msvcrt
            # msvcrt LK_LOCK is NOT indefinite (≈10 retries over ~10 s, then OSError)
            # while a real stage holds the lock for MINUTES — loop to emulate flock's
            # indefinite block (deadlock-free: the holder always releases or dies,
            # and the OS drops a dead holder's region locks).
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
            locked_msvcrt = True
        yield
    finally:
        try:
            if locked_msvcrt:
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                try:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except ImportError:
                    pass
        finally:
            os.close(fd)


def _cleanup_stale_dirs(out_dir: str) -> None:
    """Reap leftovers of CRASHED runs — called ONLY while holding `_stage_lock`, so a
    live concurrent stager (which holds the lock for its whole tmp lifetime) can never
    be racing us; the pid-liveness gate is defense-in-depth for lock-less API callers
    + pid reuse (a live/ambiguous pid ⇒ SKIP: leak-safe, never deletes live work).
      * `<out_dir>.tmp-<pid>`   — ownership proof: `_TMP_SENTINEL` OR our completion
        marker (just before the swap the marker goes IN first, then the sentinel comes
        OFF — the tmp carries >=1 proof at every instant; a crash mid-transition leaves
        a marker-carrying — or briefly both-proof — tmp: still OURS, still reapable).
      * `<out_dir>.trash-<pid>` — ownership proof: our completion marker
        (it IS a previous complete stage renamed aside mid-swap, SF2).
    A non-numeric suffix or a missing ownership proof ⇒ untouched (foreign dir)."""
    parent = os.path.dirname(out_dir) or "."
    base = os.path.basename(out_dir)
    if not os.path.isdir(parent):
        return
    for kind, proofs in ((".tmp-", (_TMP_SENTINEL, ".qf_stage_complete")),
                         (".trash-", (".qf_stage_complete",))):
        prefix = base + kind
        for name in os.listdir(parent):
            if not name.startswith(prefix):
                continue
            cand = os.path.join(parent, name)
            if not (os.path.isdir(cand)
                    and any(os.path.isfile(os.path.join(cand, pf)) for pf in proofs)):
                continue
            try:
                pid = int(name[len(prefix):])
            except ValueError:
                continue                        # not our naming — never touch
            if pid != os.getpid() and _pid_alive(pid):
                continue                        # possibly live → skip (leak-safe)
            logger.info("[comfyui_wan_remap] removing stale staging dir %s", cand)
            shutil.rmtree(cand)


def _expert_basename_hint_check(high_expert: str, low_expert: str) -> None:
    """C3 — swapped high/low detection. The checkpoints carry NO __metadata__ and the
    two experts are structurally identical, so the only available signal is the
    filename convention every official release uses (`*high*` / `*low*`). Both hints
    present but REVERSED → raise (a swap silently degrades quality: the wrong expert
    denoises the wrong noise regime). Hints absent → warn once and proceed."""
    def _hint_tokens(path):
        # word-ish tokens (split on _ - . and spaces) so "flow"/"highway" can never
        # false-match "low"/"high" — only a genuine high/low token counts (N3).
        return set(re.split(r"[^a-z0-9]+", os.path.basename(path).lower()))
    hi_name = os.path.basename(high_expert).lower()
    lo_name = os.path.basename(low_expert).lower()
    hi_toks, lo_toks = _hint_tokens(high_expert), _hint_tokens(low_expert)
    hi_says_low = "low" in hi_toks and "high" not in hi_toks
    lo_says_high = "high" in lo_toks and "low" not in lo_toks
    if hi_says_low and lo_says_high:
        raise RuntimeError(
            f"high/low experts look SWAPPED by filename: high_noise_expert={hi_name!r} "
            f"(says 'low') and low_noise_expert={lo_name!r} (says 'high'). A swap "
            f"silently degrades quality — pass the *high_noise* checkpoint as "
            f"high_noise_expert and the *low_noise* one as low_noise_expert.")
    if ("high" not in hi_name and "low" not in hi_name)             or ("high" not in lo_name and "low" not in lo_name):
        logger.warning(
            "[comfyui_wan_remap] expert filenames carry no high/low hint (%s / %s) — "
            "cannot verify the order; make sure high_noise_expert really is the "
            "HIGH-noise checkpoint (the checkpoints are structurally identical).",
            hi_name, lo_name)



def _run_staged_build(out_dir: str, fp: str, need_bytes: int, build_fn,
                      label: str, force: bool = False) -> str:
    """Cache-aware, locked, CRASH-SAFE staged build — the shared core extracted
    from `stage_two_expert` (behaviour-identical OUTCOMES; the one ordering
    delta is that callers now compute need_bytes BEFORE the lock, so a
    cache-hit pays a few extra header reads — read-only, side-effect-free):
    under `<out_dir>.lock`,
    (1) fingerprint cache-hit check, (2) free-disk pre-check with the caller's
    exact planned bytes, (3) build into `<out_dir>.tmp-<pid>` via `build_fn(tmp)`,
    (4) marker-first atomic swap with rollback (SF1/SF2/SF3/S3/S4 semantics
    unchanged; `label` only decorates messages)."""
    marker_name = ".qf_stage_complete"
    marker = os.path.join(out_dir, marker_name)
    with _stage_lock(out_dir):
        if not force and os.path.isfile(marker):
            with open(marker, "r", encoding="utf-8") as f:
                if f.read().strip() == fp:
                    logger.info("[comfyui_wan_remap] staged dir cache hit -> %s", out_dir)
                    return out_dir

        # S4 — free-disk pre-check with the EXACT planned output bytes (fail with an
        # actionable message instead of ENOSPC halfway through a multi-GB write).
        space_probe = out_dir
        while not os.path.isdir(space_probe):
            parent = os.path.dirname(space_probe)
            if parent == space_probe:
                break
            space_probe = parent
        free = shutil.disk_usage(space_probe).free
        if free < need_bytes:
            raise RuntimeError(
                f"not enough free disk for the staged {label}: need "
                f"~{need_bytes / 1024**3:.1f} GiB (dequantized weights + margin) but only "
                f"{free / 1024**3:.1f} GiB free at {space_probe!r}. Point output_dir "
                f"(or QUANTFUNC_CACHE_DIR) at a roomier disk.")

        # S3 — ATOMIC staging: build in <out_dir>.tmp-<pid>, then swap into place.
        _cleanup_stale_dirs(out_dir)
        tmp_dir = f"{out_dir}.tmp-{os.getpid()}"
        if os.path.isdir(tmp_dir):
            if os.path.isfile(os.path.join(tmp_dir, _TMP_SENTINEL)):
                shutil.rmtree(tmp_dir)
            else:
                raise RuntimeError(f"staging tmp path {tmp_dir!r} exists and is not ours")
        os.makedirs(tmp_dir)
        with open(os.path.join(tmp_dir, _TMP_SENTINEL), "w", encoding="utf-8") as f:
            f.write(fp)

        # SF3 — reap own tmp on a BUILD-phase exception; gated OFF once the swap
        # starts (tmp may then hold the ONLY completed build).
        swap_started = False
        try:
            build_fn(tmp_dir)

            # SF2 — CRASH-SAFE swap (marker FIRST, sentinel off SECOND — the tmp
            # carries >=1 ownership proof at every instant).
            with open(os.path.join(tmp_dir, marker_name), "w", encoding="utf-8") as f:
                f.write(fp)
            swap_started = True                  # from here on, NEVER reap tmp blindly
            try:
                os.unlink(os.path.join(tmp_dir, _TMP_SENTINEL))
            except OSError as exc:
                logger.warning(
                    "[comfyui_wan_remap] could not remove the staging sentinel "
                    "(%s) — proceeding with the swap; a stray %s file may remain "
                    "inside the staged dir (harmless).", exc, _TMP_SENTINEL)
            trash_dir = f"{out_dir}.trash-{os.getpid()}"
            old_moved = False
            if os.path.isdir(out_dir):
                os.replace(out_dir, trash_dir)   # old stage aside, atomically
                old_moved = True
            try:
                os.replace(tmp_dir, out_dir)     # new stage live, atomically
            except BaseException:
                if old_moved and not os.path.isdir(out_dir):
                    for _attempt in range(_ROLLBACK_RETRIES):
                        try:
                            os.replace(trash_dir, out_dir)
                            break
                        except OSError:
                            time.sleep(_ROLLBACK_RETRY_DELAY_S)
                    else:
                        logger.error(
                            "[comfyui_wan_remap] COMPOUND fault: the swap-in AND the "
                            "rollback both failed — %s is ABSENT. Nothing is lost: the "
                            "OLD complete stage is at %s and the NEW complete stage is "
                            "at %s (both marker-carrying). Re-running the stage "
                            "self-heals (reaps + rebuilds), or move either dir into "
                            "place manually.", out_dir, trash_dir, tmp_dir)
                raise
            if old_moved and os.path.isdir(trash_dir):
                shutil.rmtree(trash_dir, ignore_errors=True)
        finally:
            if not swap_started and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_dir

def stage_two_expert(high_expert: str, low_expert: str, shared_dir: str,
                     out_dir: str, *, boundary_ratio: float | None = None,
                     force: bool = False) -> str:
    """Stage two single-file Wan experts into a two-expert diffusers model_dir.

    Layout produced:
        <out_dir>/transformer/{config.json, diffusion_pytorch_model.safetensors}
        <out_dir>/transformer_2/{config.json, diffusion_pytorch_model.safetensors}
        <out_dir>/{text_encoder,tokenizer,scheduler}/   (symlink -> shared_dir)
        <out_dir>/vae/{*.safetensors (link), config.json (absent-key fixed)}
        <out_dir>/model_index.json  (_class_name + boundary_ratio)

    `shared_dir` is a Wan diffusers dir supplying vae/text_encoder/tokenizer/
    scheduler (+ a transformer/config.json used as the config base).
    `boundary_ratio=None` = auto (inherit the shared model_index's published value,
    else the modality's published default); an explicit value must be in (0, 1].
    Returns the staged model_dir path. Cache-aware (identical sources skip), and
    ATOMIC: everything is staged into `<out_dir>.tmp-<pid>` then `os.replace`d into
    place, so a concurrent second instance or an aborted run can never leave a
    partial dir that carries a valid completion marker.
    """
    high_expert = os.path.abspath(high_expert)
    low_expert = os.path.abspath(low_expert)
    shared_dir = os.path.abspath(shared_dir)
    out_dir = os.path.abspath(out_dir)

    for p, label in ((high_expert, "high_noise_expert"),
                     (low_expert, "low_noise_expert")):
        if not os.path.isfile(p):
            raise RuntimeError(f"{label} not found: {p}")
        if not is_comfyui_wan_single_file(p):
            raise RuntimeError(
                f"{label} does not look like an original-Wan single-file "
                f"transformer (no blocks.N.self_attn / ffn.0 / patch_embedding): {p}")
    if not os.path.isdir(shared_dir):
        raise RuntimeError(f"shared_components dir not found: {shared_dir}")
    # S1 — the same file staged as BOTH experts silently degrades quality (the
    # boundary switch becomes a no-op): fail loud.
    if os.path.samefile(high_expert, low_expert):
        raise RuntimeError(
            f"high_noise_expert and low_noise_expert are the SAME file ({high_expert}) "
            f"— the A14B two-expert model needs the two DIFFERENT checkpoints.")
    # C3 — filename-convention swap detection (no structural signal exists).
    _expert_basename_hint_check(high_expert, low_expert)
    # G1 — fail at stage time (actionable) instead of fail-late at engine load.
    missing_shared = [sub for sub in ("vae", "text_encoder")
                      if not os.path.isdir(os.path.join(shared_dir, sub))]
    if missing_shared:
        raise RuntimeError(
            f"shared_components dir {shared_dir!r} is missing {missing_shared} — point "
            f"it at a Wan diffusers dir containing vae/ + text_encoder/ (+ tokenizer/, "
            f"scheduler/).")

    in_ch, out_ch = detect_wan_modality(high_expert)
    lo_in, lo_out = detect_wan_modality(low_expert)
    if (in_ch, out_ch) != (lo_in, lo_out):
        raise RuntimeError(
            f"expert modality mismatch: high=({in_ch},{out_ch}) "
            f"low=({lo_in},{lo_out}) — the two experts must be the same variant")
    hi_layers = _count_layers(_read_header(high_expert))
    lo_layers = _count_layers(_read_header(low_expert))
    if hi_layers != lo_layers:
        raise RuntimeError(
            f"expert depth mismatch: high={hi_layers} vs low={lo_layers} blocks — "
            f"the two experts must be the same Wan variant")
    class_name = _ENGINE_WAN_PIPELINE_CLASS   # loadable for BOTH modalities (see above)

    # C1 — resolve the boundary BEFORE fingerprinting (it is baked into the stage).
    shared_mi = os.path.join(shared_dir, "model_index.json")
    base_mi = _load_json(shared_mi) if os.path.isfile(shared_mi) else None
    # The shared dir's OWN modality (its transformer config channels) gates whether
    # its published boundary_ratio may be inherited (modality-specific value).
    base_xfm_cfg = {}
    shared_xfm_cfg = os.path.join(shared_dir, "transformer", "config.json")
    if os.path.isfile(shared_xfm_cfg):
        base_xfm_cfg = _load_json(shared_xfm_cfg)
    shared_modality = None
    if isinstance(base_xfm_cfg.get("in_channels"), int) \
            and isinstance(base_xfm_cfg.get("out_channels"), int):
        shared_modality = (base_xfm_cfg["in_channels"], base_xfm_cfg["out_channels"])
    eff_boundary, boundary_src = resolve_boundary_ratio(
        boundary_ratio, base_mi, in_ch, out_ch, shared_modality)

    # Safety: never let a mis-pointed / workflow-supplied output_dir clobber the
    # experts, the shared model, or a directory holding the user's own content.
    _assert_safe_out_dir(out_dir, [high_expert, low_expert, shared_dir,
                                   os.path.dirname(high_expert),
                                   os.path.dirname(low_expert)])

    # Fingerprint the experts + the shared config files that get baked into the stage
    # (so an edit to a shared config invalidates the cache, not just a new dir mtime).
    # "dequant" = a FIXED literal identifying the staging format in the fingerprint
    # (kept stable across releases so existing staged caches keep hitting).
    fp_inputs = [high_expert, low_expert, shared_dir]
    for rel in ("model_index.json", "vae/config.json", "transformer/config.json"):
        fp_inputs.append(os.path.join(shared_dir, rel))
    fp = _fingerprint(fp_inputs, extra=f"{eff_boundary}|dequant|{in_ch}|{out_ch}")

    # SF1 — EXCLUSIVE whole-stage lock: everything from the cache-hit check to the
    # final swap runs under `<out_dir>.lock`. A concurrent instance staging the same
    # out_dir BLOCKS here until the holder finishes, then sees the completed marker
    # and cache-hits — it can never reap the holder's live tmp (the old defect) nor
    # observe a half-swapped dir. The lock dies with a crashed holder (flock).
    def _build(tmp_dir: str) -> None:
        # -- transformers (base_xfm_cfg loaded early, before boundary resolution) --
        for sub, src in (("transformer", high_expert), ("transformer_2", low_expert)):
            d = os.path.join(tmp_dir, sub)
            os.makedirs(d, exist_ok=True)
            n_layers = _count_layers(_read_header(src))
            cfg = synthesize_transformer_config(base_xfm_cfg, in_ch, out_ch, n_layers)
            _dump_json(cfg, os.path.join(d, "config.json"))
            weight_dst = os.path.join(d, "diffusion_pytorch_model.safetensors")
            nk = remap_dequant_file(src, weight_dst)
            logger.info("[comfyui_wan_remap] %s <- %s (remap+dequant, %d keys)",
                        sub, os.path.basename(src), nk)

        # -- shared components (dir symlink / copy) ----------------------------
        for sub in _SHARED_SUBDIRS:
            s2 = os.path.join(shared_dir, sub)
            if os.path.isdir(s2):
                _stage_shared_dir(s2, os.path.join(tmp_dir, sub))

        # -- vae (link weights via the shared Windows-safe helper + fixed cfg) --
        vae_src = os.path.join(shared_dir, "vae")
        if os.path.isdir(vae_src):
            vae_dst = os.path.join(tmp_dir, "vae")
            os.makedirs(vae_dst, exist_ok=True)
            for wf in os.listdir(vae_src):
                if wf.endswith(".safetensors"):
                    link_or_copy(os.path.join(vae_src, wf), os.path.join(vae_dst, wf))
            vae_cfg_path = os.path.join(vae_src, "config.json")
            base_vae_cfg = _load_json(vae_cfg_path) if os.path.isfile(vae_cfg_path) else {}
            _dump_json(synthesize_vae_config(base_vae_cfg),
                       os.path.join(vae_dst, "config.json"))

        # -- model_index --------------------------------------------------------
        _dump_json(synthesize_model_index(base_mi, class_name, eff_boundary),
                   os.path.join(tmp_dir, "model_index.json"))

    # S4 need-bytes: the two dequantized experts + margin (shared components are
    # links/small configs — covered by the margin, as before).
    need = (estimate_dequant_output_bytes(high_expert)
            + estimate_dequant_output_bytes(low_expert) + _FREE_SPACE_MARGIN_BYTES)
    _run_staged_build(out_dir, fp, need, _build, "A14B", force=force)

    logger.info("[comfyui_wan_remap] staged two-expert %s dir (dequant-fp16, "
                "boundary=%.3f [%s]) -> %s",
                class_name, eff_boundary, boundary_src, out_dir)
    return out_dir


# ============================================================================
# Scan (frontend for the QuantFunc Wan Combine Experts (Auto) node's dropdowns)
#
# Pure filesystem scan that lists a machine's Wan2.2-A14B single-file experts +
# Wan diffusers dirs (vae/text_encoder source) as {label: abs_path} maps for the
# node's high/low/shared dropdowns. The node then delegates the picked triple to
# `stage_two_expert` (above). This scan writes NO weights and touches NO expert
# bytes beyond the cheap header reads `is_comfyui_wan_single_file` /
# `detect_wan_modality` already perform.
# ============================================================================

# Wan2.2-A14B naming: a single-file expert set carries a `high`/`low` expert token
# (the real A14B signal — only A14B ships a high+low pair). The standalone TI2V-5B
# (a single, non-A14B model) carries `5b` and no high/low — excluded both by the
# high/low requirement and by rejecting the `5b` size token.
_NON_A14B_SIZE_TOKENS = ("5b",)


def _basename_tokens(path: str) -> set[str]:
    """Word-ish tokens of a basename (split on non-alphanumerics), lowercased —
    so `t2v`/`i2v`/`high`/`low`/`14b` match as whole tokens and e.g. `highway`
    can never false-match `high` (same tokenizer rule as `_expert_basename_hint_check`,
    kept separate to leave that shipped function byte-unchanged, N3)."""
    return set(re.split(r"[^a-z0-9]+", os.path.basename(path).lower()))


def _looks_like_a14b_single_file(path: str) -> bool:
    """Filename PRE-FILTER (cheap, no header read): a candidate A14B single-file
    expert carries a high/low expert token (the real A14B signal — only A14B ships
    a high+low pair; a single Wan2.1/TI2V-5B model has neither) and NOT a non-A14B
    size token (`5b`). A `14b`/`a14b` size token is a bonus signal, NOT required, so
    a RENAMED expert (e.g. `wan_high_noise.safetensors`) still detects. Header
    confirmation (`is_comfyui_wan_single_file`) + weight-derived modality happen
    after, and same-modality high+low pairing is what ultimately forms a set."""
    if not path.lower().endswith(".safetensors"):
        return False
    toks = _basename_tokens(path)
    if any(t in toks for t in _NON_A14B_SIZE_TOKENS):
        return False
    return ("high" in toks) or ("low" in toks)


def _is_diffusers_a14b_dir(d: str) -> bool:
    """A diffusers A14B dir is dual-expert: BOTH transformer/ and transformer_2/
    carry a config.json. The TI2V-5B diffusers dir has a single transformer/ (no
    transformer_2/) -> correctly excluded."""
    return (os.path.isfile(os.path.join(d, "transformer", "config.json"))
            and os.path.isfile(os.path.join(d, "transformer_2", "config.json")))


def _dir_has_shared_components(d: str) -> bool:
    """True if `d` can supply the shared vae/ + text_encoder/ a single-file pair
    needs (the G1 minimum `stage_two_expert` enforces)."""
    return (os.path.isdir(os.path.join(d, "vae"))
            and os.path.isdir(os.path.join(d, "text_encoder")))


def _is_wan_diffusers_dir(d: str) -> bool:
    """True if `d` is a Wan-family diffusers dir (so its vae/text_encoder are the
    RIGHT shared components for a Wan expert — a Qwen/ZImage/Klein dir also has
    vae+text_encoder but a different VAE). Signals, any one decisive: A14B dual
    transformer_2/; model_index _class_name starts with "Wan"; transformer config
    _class_name == "WanTransformer3DModel"; or vae config _class_name is a Wan VAE."""
    if _is_diffusers_a14b_dir(d):
        return True
    try:
        mi = os.path.join(d, "model_index.json")
        if os.path.isfile(mi):
            cls = _load_json(mi).get("_class_name")
            if isinstance(cls, str) and cls.startswith("Wan"):
                return True
        xc = os.path.join(d, "transformer", "config.json")
        if os.path.isfile(xc) and _load_json(xc).get("_class_name") == "WanTransformer3DModel":
            return True
        vc = os.path.join(d, "vae", "config.json")
        if os.path.isfile(vc):
            vcls = _load_json(vc).get("_class_name") or ""
            if "Wan" in vcls:
                return True
    except Exception as e:  # noqa: BLE001 — unreadable config, treat as non-Wan
        logger.debug("_is_wan_diffusers_dir: config read failed %s: %s", d, e)
    return False


def _labeled(paths: list[str]) -> dict:
    """{display_label: absolute_path} for a dropdown — basename labels,
    disambiguated with the parent-dir name on a basename collision."""
    from collections import Counter
    bases = [os.path.basename(p.rstrip("/")) for p in paths]
    cnt = Counter(bases)
    out: dict = {}
    for p, b in zip(paths, bases):
        label = b if cnt[b] == 1 else os.path.join(
            os.path.basename(os.path.dirname(p.rstrip("/"))), b)
        while label in out:            # defensive: never collapse two distinct paths
            label += " "
        out[label] = os.path.abspath(p)
    return out


def list_wan_a14b_choices(roots: list[str]) -> tuple[dict, dict]:
    """Scan `roots` for the Wan A14B loader node's two dropdowns and return
    (experts, shared_dirs) as ORDERED {display_label: absolute_path} maps:

      * experts    — every Wan single-file transformer that looks like an A14B
                     expert (a high/low filename token, NOT a `5b` token, header-
                     confirmed by `is_comfyui_wan_single_file`); the user picks
                     which is high and which is low.
      * shared_dirs — Wan diffusers dirs that can supply the shared vae/
                     text_encoder/tokenizer/scheduler, A14B-family (dual
                     transformer_2/) listed FIRST so a 5B dir's different VAE
                     can't shadow the correct 14B shared components.

    RECURSIVE (`os.walk`, matching the plugin's other dropdown scanners
    `_get_diffusers_model_options` / `_get_local_transformer_file_options`) so
    per-model subfolders (`models/diffusion_models/wan2.2-a14b/high.safetensors`,
    `models/diffusers/wan/Wan2.2-T2V-A14B-Diffusers/`) are found. A diffusers model
    dir is PRUNED once identified — added as a shared choice if it's a Wan dir, and
    never descended into (its internal sharded transformer weights are diffusers-key,
    not single-file experts). Pure filesystem + cheap header reads. NO staging.
    """
    expert_paths: list[str] = []
    a14b_shared: list[str] = []
    other_shared: list[str] = []
    seen_f: set = set()
    seen_d: set = set()
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, subdirs, filenames in os.walk(root):
            real_d = os.path.realpath(dirpath)
            # A Wan diffusers dir that can supply the shared components (vae/ +
            # text_encoder/) is a SHARED choice and a leaf — PRUNE. This is checked on
            # EVERY visited dir, INDEPENDENT of whether it also looks like a full model
            # dir: it covers both a full A14B dir AND a stripped vae/text_encoder-only
            # shared source (which has neither transformer/ nor model_index.json but is
            # still a valid Wan shared source per `_is_wan_diffusers_dir`'s vae signal).
            if real_d not in seen_d and _dir_has_shared_components(dirpath) \
                    and _is_wan_diffusers_dir(dirpath):
                seen_d.add(real_d)
                (a14b_shared if _is_diffusers_a14b_dir(dirpath)
                 else other_shared).append(dirpath)
                subdirs[:] = []            # a shared dir is a leaf — don't descend
                continue
            # A non-Wan diffusers MODEL dir (Qwen/ZImage/… — model_index.json, or a
            # transformer/config.json) is a leaf too: PRUNE (don't descend into its
            # internal sharded weights — they are diffusers-key, not single-file
            # experts — and don't credit it as a Wan shared source).
            if "model_index.json" in filenames \
                    or os.path.isfile(os.path.join(dirpath, "transformer", "config.json")):
                subdirs[:] = []
                continue
            # Otherwise a plain dir: pick up any loose single-file Wan A14B experts.
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                if not _looks_like_a14b_single_file(full):
                    continue
                real = os.path.realpath(full)
                if real in seen_f:
                    continue
                try:
                    if not is_comfyui_wan_single_file(full):
                        continue
                except Exception as e:  # noqa: BLE001 — corrupt/partial, skip
                    logger.debug("list_wan_a14b_choices: header read failed %s: %s", full, e)
                    continue
                seen_f.add(real)
                expert_paths.append(full)
    return _labeled(expert_paths), _labeled(a14b_shared + other_shared)


# ============================================================================
# Single-file TI2V-5B trio → diffusers staging + self-registering adapter
#
# The user-facing gap this closes: UNETLoader(wan2.2_ti2v_5B_*.safetensors,
# original-Wan keys) + CLIPLoader(umt5_xxl_*) + VAELoader(wan2.2_vae) into
# Build Pipeline previously matched NO adapter ("No format adapter matched"):
# the A14B Combine node only accepts high/low expert PAIRS, and the generic
# single-file adapter has no Wan arch fingerprint. TI2V-5B is a SINGLE
# transformer, so it stages exactly like one A14B expert minus the pairing —
# reusing remap_dequant_file (key remap + optional fp8 dequant) for all three
# legs and the same crash-safe _run_staged_build core.
# ============================================================================

# TI2V-5B modality: t2v single transformer over the Wan2.2 48-ch VAE (in==out).
_TI2V5B_LATENT_CHANNELS = 48
# Bundled engine-verified assets (fetched verbatim from the official
# Wan2.2-TI2V-5B-Diffusers release the engine was verified against).
_WAN_ASSET_TRANSFORMER_CFG = "transformer_configs/Wan22TI2V5B.json"
_WAN_ASSET_VAE_CFG = "vae_configs/Wan22TI2V5B.json"
_WAN_ASSET_TE_CFG = "text_encoder_configs/Wan.json"
_WAN_ASSET_SCHEDULER_CFG = "scheduler_configs/Wan22TI2V5B.json"
_WAN_ASSET_TOKENIZER_DIR = "tokenizers/Wan"
# ComfyUI's umt5 single-file embeds the sentencepiece model as a raw U8 tensor
# (`spiece_model`) — not a weight. The engine's UMT5 tokenizer reads
# tokenizer/tokenizer.json (bundled) instead, so the blob is dropped.
_UMT5_EXTRA_DROP = ("spiece_model",)

# model_index for the staged single-transformer TI2V-5B — field-for-field from
# the official Wan2.2-TI2V-5B-Diffusers model_index.json (boundary_ratio null =
# single transformer, no expert switch; expand_timesteps true is 5B-specific).
_TI2V5B_MODEL_INDEX = {
    "_class_name": _ENGINE_WAN_PIPELINE_CLASS,
    "boundary_ratio": None,
    "expand_timesteps": True,
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}


def _wan_bundled_asset(rel: str) -> str:
    """Absolute path of a bundled Wan asset under <plugin>/bin/ — fail-LOUD when
    absent (a broken plugin install must not stage a half-configured model)."""
    plugin_root = Path(__file__).resolve().parent.parent
    p = plugin_root / "bin" / rel
    if not p.exists():
        raise RuntimeError(
            f"plugin bundled asset missing: {p} — reinstall/update the "
            f"ComfyUI-QuantFunc plugin (bin/{rel} ships with it).")
    return str(p)


def _looks_like_umt5_single_file(path: str | Path) -> bool:
    """Cheap header check: HF-T5 naming (`encoder.block.N...`) = the ComfyUI
    umt5_xxl single-file (fp16 or fp8_scaled)."""
    try:
        hdr = _read_header(path)
    except Exception:  # noqa: BLE001
        return False
    return any(k.startswith("encoder.block.") for k in hdr)


def _looks_like_wan_vae_single_file(path: str | Path) -> bool:
    """Cheap header check: original-Wan VAE naming (`decoder.head.2.*`) OR the
    already-diffusers naming (`decoder.conv_out.*` — remap passes through)."""
    try:
        hdr = _read_header(path)
    except Exception:  # noqa: BLE001
        return False
    return any(k.startswith(("decoder.head.2.", "decoder.conv_out."))
               for k in hdr)


def _looks_like_taew_tiny_vae(path: str | Path) -> bool:
    """taehv TINY-VAE acceleration files (taew2_1/taew2_2): TinyVAEDecoder layout
    keys `decoder.<idx>.*` (e.g. decoder.1.weight ... decoder.22.weight — a bare
    numeric segment right after `decoder.`, unlike full-VAE namings), or a
    'taew' basename. Used ONLY to make the trio's VAE-signature refusal
    actionable when a user wires the tiny-VAE file into VAELoader by mistake."""
    if "taew" in os.path.basename(str(path)).lower():
        return True
    try:
        hdr = _read_header(path)
    except Exception:  # noqa: BLE001
        return False
    return any(re.match(r"^decoder\.\d+\.", k) for k in hdr)


def default_wan_5b_stage_dir() -> str:
    """VOLATILE staging root for the 5B trio (ComfyUI temp — cleared on restart;
    tempfile.gettempdir() outside ComfyUI). Deliberately NEVER a persistent
    cache: model-weight copies must not survive reboots (metadata-only files
    may). Session-cached via _run_staged_build's fingerprint marker."""
    try:
        import folder_paths
        root = folder_paths.get_temp_directory()
    except Exception:  # noqa: BLE001 — outside ComfyUI (tests)
        import tempfile as _tf
        root = _tf.gettempdir()
    return os.path.join(root, "qf_wan_ti2v5b")


def stage_ti2v_5b_trio(xfm_path: str | Path, te_path: str | Path,
                       vae_path: str | Path, out_dir: str | Path,
                       force: bool = False) -> str:
    """Stage a ComfyUI single-file TI2V-5B trio into the engine's diffusers
    layout. All three legs run through remap_dequant_file (streamed, memory-
    bounded; fp8_scaled variants dequant to fp16 inline):

        transformer/  original-Wan keys → diffusers (remap_key; VERIFIED
                      key-exact 825/825 vs the official 5B diffusers release)
        text_encoder/ HF-T5 keys pass through unchanged; fp8 scale siblings +
                      the embedded `spiece_model` blob dropped (VERIFIED the
                      remaining 242 keys == the official release exactly)
        vae/          original-Wan VAE keys → diffusers AutoencoderKLWan
                      (remap_wan_vae_key; VERIFIED key+shape-exact 196/196)
        tokenizer/ scheduler/ model_index.json + per-component config.json
                      from the bundled engine-verified assets.

    Cache-aware + crash-safe via _run_staged_build (same core as the A14B
    two-expert stage). Returns the staged model_dir.
    """
    xfm_path, te_path, vae_path = str(xfm_path), str(te_path), str(vae_path)
    out_dir = os.path.abspath(str(out_dir))

    if not os.path.isfile(xfm_path) or not is_comfyui_wan_single_file(xfm_path):
        raise RuntimeError(
            f"not an original-Wan single-file transformer (no blocks.N.self_attn"
            f"/ffn.0/patch_embedding): {xfm_path}")
    in_ch, out_ch = detect_wan_modality(xfm_path)
    if (in_ch, out_ch) != (_TI2V5B_LATENT_CHANNELS, _TI2V5B_LATENT_CHANNELS):
        raise RuntimeError(
            f"unsupported Wan single-file modality ({in_ch},{out_ch}) — this "
            f"path stages the TI2V-5B (48,48) only. A 16-channel file is a "
            f"Wan2.1/A14B-family transformer: an A14B high/low expert PAIR "
            f"loads via the 'QuantFunc Wan Combine Experts' node; a standalone "
            f"Wan2.1 single-file is not yet supported here.")
    if not _looks_like_umt5_single_file(te_path):
        raise RuntimeError(
            f"the wired CLIP file does not look like the Wan umt5_xxl text "
            f"encoder (no encoder.block.* keys): {te_path}")
    if not _looks_like_wan_vae_single_file(vae_path):
        hint = ""
        if _looks_like_taew_tiny_vae(vae_path):
            hint = (
                " This is a taew TINY-VAE acceleration file, NOT the full Wan "
                "VAE. In VAELoader pick wan2.2_vae.safetensors; for tiny-VAE "
                "acceleration enable the `tiny_vae` toggle on Build Pipeline "
                "instead (its weights already live under "
                "models/QuantFunc/taew/ — no manual selection needed). "
                "你接入的是 taew tiny-VAE 加速文件，不是完整 VAE。VAELoader 请选 "
                "wan2.2_vae.safetensors；tiny-VAE 加速请打开 Build Pipeline 的 "
                "tiny_vae 开关（权重已在 models/QuantFunc/taew/，无需手选）。")
        raise RuntimeError(
            f"the wired VAE file does not look like a Wan VAE (no decoder.head"
            f".2/decoder.conv_out keys): {vae_path}." + hint)

    # Resolve every bundled asset UP FRONT (fail-loud before any GB write).
    xfm_cfg_asset = _wan_bundled_asset(_WAN_ASSET_TRANSFORMER_CFG)
    vae_cfg_asset = _wan_bundled_asset(_WAN_ASSET_VAE_CFG)
    te_cfg_asset = _wan_bundled_asset(_WAN_ASSET_TE_CFG)
    sched_asset = _wan_bundled_asset(_WAN_ASSET_SCHEDULER_CFG)
    tok_dir_asset = _wan_bundled_asset(_WAN_ASSET_TOKENIZER_DIR)

    _assert_safe_out_dir(out_dir, [xfm_path, te_path, vae_path,
                                   os.path.dirname(xfm_path),
                                   os.path.dirname(te_path),
                                   os.path.dirname(vae_path)])

    fp = _fingerprint([xfm_path, te_path, vae_path,
                       xfm_cfg_asset, vae_cfg_asset, te_cfg_asset, sched_asset],
                      extra=f"ti2v5b|dequant|{in_ch}|{out_ch}")

    def _build(tmp_dir: str) -> None:
        # -- transformer (config: bundled base + measured dims, mirroring the
        #    A14B path's synthesize_transformer_config precedence) -------------
        d = os.path.join(tmp_dir, "transformer")
        os.makedirs(d, exist_ok=True)
        n_layers = _count_layers(_read_header(xfm_path))
        cfg = synthesize_transformer_config(_load_json(xfm_cfg_asset),
                                            in_ch, out_ch, n_layers)
        _dump_json(cfg, os.path.join(d, "config.json"))
        nk = remap_dequant_file(
            xfm_path, os.path.join(d, "diffusion_pytorch_model.safetensors"))
        logger.info("[comfyui_wan_remap] transformer <- %s (remap+dequant, %d keys)",
                    os.path.basename(xfm_path), nk)

        # -- text encoder (keys pass through; fp8 dequant; spiece blob dropped) --
        d = os.path.join(tmp_dir, "text_encoder")
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(te_cfg_asset, os.path.join(d, "config.json"))
        nk = remap_dequant_file(te_path, os.path.join(d, "model.safetensors"),
                                key_fn=lambda k: k,
                                extra_drop=_UMT5_EXTRA_DROP)
        logger.info("[comfyui_wan_remap] text_encoder <- %s (dequant, %d keys)",
                    os.path.basename(te_path), nk)

        # -- vae (original-Wan naming → diffusers AutoencoderKLWan) -------------
        d = os.path.join(tmp_dir, "vae")
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(vae_cfg_asset, os.path.join(d, "config.json"))
        nk = remap_dequant_file(
            vae_path, os.path.join(d, "diffusion_pytorch_model.safetensors"),
            key_fn=remap_wan_vae_key)
        logger.info("[comfyui_wan_remap] vae <- %s (remap, %d keys)",
                    os.path.basename(vae_path), nk)

        # -- tokenizer / scheduler / model_index --------------------------------
        d = os.path.join(tmp_dir, "tokenizer")
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(tok_dir_asset):
            src_f = os.path.join(tok_dir_asset, f)
            if os.path.isfile(src_f):
                shutil.copyfile(src_f, os.path.join(d, f))
        d = os.path.join(tmp_dir, "scheduler")
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(sched_asset, os.path.join(d, "scheduler_config.json"))
        _dump_json(_TI2V5B_MODEL_INDEX, os.path.join(tmp_dir, "model_index.json"))

    need = (estimate_dequant_output_bytes(xfm_path)
            + estimate_dequant_output_bytes(te_path, extra_drop=_UMT5_EXTRA_DROP)
            + estimate_dequant_output_bytes(vae_path)
            + _FREE_SPACE_MARGIN_BYTES)
    _run_staged_build(out_dir, fp, need, _build, "single-file TI2V-5B",
                      force=force)
    logger.info("[comfyui_wan_remap] staged single-file TI2V-5B -> %s", out_dir)
    return out_dir


# ---- self-registering Build Pipeline adapter --------------------------------
from .base import BuildContext, FormatAdapter, SourceBundle, StagingResult  # noqa: E402
from .factory import adapter  # noqa: E402


@adapter(priority=60)
class ComfyUIWanSingleFileAdapter(FormatAdapter):
    """Single-file original-Wan transformer (TI2V-5B) + bare umt5 CLIP + Wan VAE.

    Sits ABOVE the generic ComfyUIDiffusionModelAdapter (50) — original-Wan
    keys carry no ComfyUI prefix and no fingerprintable arch, so without this
    adapter the trio matched nothing. Files living inside a diffusers
    model_dir never reach us (HFLayoutAdapter, priority 100, wins first).
    """

    @classmethod
    def detect(cls, sources: SourceBundle) -> bool:
        if sources.checkpoint is not None or sources.transformer is None:
            return False
        try:
            return is_comfyui_wan_single_file(sources.transformer.path)
        except Exception:  # noqa: BLE001
            return False

    def adapt(self, sources: SourceBundle, staging_dir: Path,
              context: BuildContext) -> StagingResult:
        xfm_path = sources.transformer.path
        if sources.text_encoder is None or sources.vae is None:
            raise RuntimeError(
                "Wan single-file needs all three inputs wired into Build "
                "Pipeline: the transformer (UNETLoader), the umt5_xxl text "
                "encoder (CLIPLoader) and the Wan VAE (VAELoader). Missing: "
                + ", ".join(n for n, v in (("clip", sources.text_encoder),
                                           ("vae", sources.vae)) if v is None))
        # stage_ti2v_5b_trio re-validates modality/te/vae and fails LOUD with
        # actionable guidance (A14B pairs -> the Combine Experts node).
        staged = stage_ti2v_5b_trio(xfm_path, sources.text_encoder.path,
                                    sources.vae.path,
                                    default_wan_5b_stage_dir())
        # arch/method mirror what the verified A14B flow yields downstream
        # (hf_native on a staged Wan dir: _class_name "WanPipeline" -> "Wan").
        return StagingResult(
            model_dir=staged,
            arch="Wan",
            method_hint="online_quant",
            cleanup_dir=None,   # volatile session cache — reused across builds
        )
