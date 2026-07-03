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


def remap_dequant_file(src: str | Path, dst: str | Path) -> int:
    """Remap keys AND dequant fp8 (F8_E4M3 × scale_weight → fp16). Needs torch.

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
        if _is_droppable(k):
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
        out_key = remap_key(k)
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


def estimate_dequant_output_bytes(src: str | Path) -> int:
    """Exact DATA bytes `remap_dequant_file` will write for `src` (kept tensors,
    fp8→F16 widening applied) — the same plan math, header-only, no torch. Used for
    the free-disk pre-check; the output json header (~100 KB) rides in the caller's
    safety margin."""
    hdr = _read_header(src)   # DoS-guarded
    total = 0
    for k, info in hdr.items():
        if _is_droppable(k):
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
    marker_name = ".qf_stage_complete"
    marker = os.path.join(out_dir, marker_name)

    # SF1 — EXCLUSIVE whole-stage lock: everything from the cache-hit check to the
    # final swap runs under `<out_dir>.lock`. A concurrent instance staging the same
    # out_dir BLOCKS here until the holder finishes, then sees the completed marker
    # and cache-hits — it can never reap the holder's live tmp (the old defect) nor
    # observe a half-swapped dir. The lock dies with a crashed holder (flock).
    with _stage_lock(out_dir):
        if not force and os.path.isfile(marker):
            with open(marker, "r", encoding="utf-8") as f:
                if f.read().strip() == fp:
                    logger.info("[comfyui_wan_remap] staged dir cache hit -> %s", out_dir)
                    return out_dir

        # S4 — free-disk pre-check with the EXACT planned output bytes (fail with an
        # actionable message instead of ENOSPC halfway through a 56 GB write).
        need = (estimate_dequant_output_bytes(high_expert)
                + estimate_dequant_output_bytes(low_expert) + _FREE_SPACE_MARGIN_BYTES)
        space_probe = out_dir
        while not os.path.isdir(space_probe):
            parent = os.path.dirname(space_probe)
            if parent == space_probe:
                break
            space_probe = parent
        free = shutil.disk_usage(space_probe).free
        if free < need:
            raise RuntimeError(
                f"not enough free disk for the staged A14B: need ~{need / 1024**3:.1f} GiB "
                f"(dequantized experts + margin) but only {free / 1024**3:.1f} GiB free at "
                f"{space_probe!r}. Point output_dir (or QUANTFUNC_CACHE_DIR) at a roomier disk.")

        # S3 — ATOMIC staging: build in <out_dir>.tmp-<pid>, then swap into place.
        # Under the lock, any leftover tmp/trash is from a CRASHED run (a live stager
        # holds the lock for its tmp's whole lifetime) — reap the dead ones.
        _cleanup_stale_dirs(out_dir)
        tmp_dir = f"{out_dir}.tmp-{os.getpid()}"
        if os.path.isdir(tmp_dir):
            # same-pid leftover (previous exception in this process) — ours by sentinel.
            if os.path.isfile(os.path.join(tmp_dir, _TMP_SENTINEL)):
                shutil.rmtree(tmp_dir)
            else:
                raise RuntimeError(f"staging tmp path {tmp_dir!r} exists and is not ours")
        os.makedirs(tmp_dir)
        with open(os.path.join(tmp_dir, _TMP_SENTINEL), "w", encoding="utf-8") as f:
            f.write(fp)

        # SF3 — this process's own tmp is reaped on a BUILD-phase exception (no 56 GB
        # orphan per crash). Once the SWAP starts, tmp may hold the ONLY completed
        # build → the reap is gated OFF (swap_started) and swap failures roll back
        # instead (a marker-carrying leftover tmp is reaped by the next run's
        # cleanup — marker counts as ownership proof).
        swap_started = False
        try:
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

            # SF2 — CRASH-SAFE swap. Marker into tmp FIRST, sentinel off SECOND, then:
            #   old out_dir --atomic rename--> .trash-<pid>
            #   tmp         --atomic rename--> out_dir     (failure ⇒ ROLLBACK trash→out)
            #   rmtree(.trash)
            # Invariant after ANY crash/exception: at least one COMPLETE marker-carrying
            # stage exists ON DISK — at out_dir (old rolled back, or new landed) or in a
            # leftover tmp/trash that the next run's cleanup reaps (marker = ownership
            # proof). Never a markerless partial. Residual (accepted + LOUD): a COMPOUND
            # persistent fault that fails the swap-in AND the retried rollback leaves
            # out_dir absent — both complete stages survive as recovery dirs, an ERROR
            # names them, and the next run self-heals (reap + rebuild).
            # Marker FIRST, sentinel off SECOND — the tmp carries >=1 ownership
            # proof at every instant, so even a SIGKILL between the two operations
            # leaves a reapable dir (never a neither-proof unreapable orphan).
            with open(os.path.join(tmp_dir, marker_name), "w", encoding="utf-8") as f:
                f.write(fp)
            # The marker IS the "completed build" proof — gate the finally-reap OFF
            # from this exact instant (NOT after the sentinel unlink: a transient
            # unlink failure must never let the finally blind-reap the only
            # completed build).
            swap_started = True                  # from here on, NEVER reap tmp blindly
            try:
                os.unlink(os.path.join(tmp_dir, _TMP_SENTINEL))
            except OSError as exc:
                # Best-effort: the build is complete and marker-carrying; a stray
                # sentinel file riding along into the staged dir is harmless (no
                # consumer reads it inside out_dir), whereas failing here would
                # discard a multi-GB completed build over a transient fault.
                logger.warning(
                    "[comfyui_wan_remap] could not remove the staging sentinel "
                    "(%s) — proceeding with the swap; a stray %s file may remain "
                    "inside the staged dir (harmless).", exc, _TMP_SENTINEL)
            trash_dir = f"{out_dir}.trash-{os.getpid()}"
            old_moved = False
            if os.path.isdir(out_dir):
                os.replace(out_dir, trash_dir)   # old stage aside, atomically, marker intact
                old_moved = True
            try:
                os.replace(tmp_dir, out_dir)     # new stage live, atomically
            except BaseException:
                # ROLLBACK: restore the old stage to the canonical path (bounded
                # retry — the rollback rename is the same syscall shape as the
                # failed swap-in, so a PERSISTENT fault can hit it too; retries
                # only rescue transient classes). The completed new build stays in
                # tmp for the next run's cleanup/rebuild either way.
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
                # Best-effort: the swap already SUCCEEDED (out_dir is the valid new
                # stage) — a trash-cleanup failure must not raise over that success;
                # a surviving dead-pid trash is reaped by the next run's cleanup.
                shutil.rmtree(trash_dir, ignore_errors=True)
        finally:
            if not swap_started and os.path.isdir(tmp_dir):
                # BUILD-phase exception only (success renamed tmp away; swap-phase
                # failures keep tmp — it may be the only completed build).
                shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("[comfyui_wan_remap] staged two-expert %s dir (dequant-fp16, "
                "boundary=%.3f [%s]) -> %s",
                class_name, eff_boundary, boundary_src, out_dir)
    return out_dir


# ============================================================================
# Auto-detection (frontend for the QuantFunc Wan Combine Experts (Auto) node)
#
# Pure filesystem scan that groups a machine's Wan2.2-A14B assets into named
# "sets", each resolvable to an engine-loadable model_dir by DELEGATING to the
# staging above (single-file pairs -> stage_two_expert; a diffusers A14B dir ->
# pass-through when already loadable, else a minimal model_index normalization).
# It writes NO weights and touches NO expert bytes beyond the cheap header reads
# `is_comfyui_wan_single_file` / `detect_wan_modality` already perform.
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


def _modality_str(in_ch: int, out_ch: int) -> str:
    """Weight-derived modality label — the same signal the engine dispatches on
    (i2v channel-concats the reference latents so in>out; t2v has in==out)."""
    return "i2v" if in_ch > out_ch else "t2v"


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


def _diffusers_dir_is_loadable(model_index: dict | None) -> bool:
    """A diffusers A14B dir is directly engine-loadable iff its model_index carries
    (a) a `_class_name` in the Wan family AND (b) a positive `boundary_ratio`.

    Matches the ENGINE'S ACTUAL gate (verified against live src/WanVideoPipeline.cpp
    2026-07-02): `wan_detect` accepts ANY `Wan…`-prefixed pipeline class
    (`in.pipeline_class.rfind("Wan", 0) == 0`) — not just `WanPipeline` — so the
    published `WanImageToVideoPipeline` A14B checkpoint IS detected as Wan; and the
    two-expert path engages purely on `boundary_ratio > 0` + a real
    `transformer_2/config.json` (never re-reading `_class_name`). This dir's
    transformer_2/ is already guaranteed by `_is_diffusers_a14b_dir`, so a real
    downloaded A14B-Diffusers dir (Wan-prefixed class + published boundary) loads
    AS-IS — no normalization. A dir with a non-Wan class OR a missing/zero
    boundary_ratio is NOT loadable and is routed through `stage_a14b_diffusers`.
    (Edge: `wan_detect`'s OTHER arm accepts a model_index with an EMPTY `_class_name`
    when transformer_class=="WanTransformer3DModel"; a real HF diffusers model_index
    always carries a non-empty `_class_name`, so this returns False for the empty case
    → routed through the — still correct — normalization rather than pass-through.)"""
    if not model_index:
        return False
    cls = model_index.get("_class_name")
    if not isinstance(cls, str) or not cls.startswith("Wan"):
        return False
    br = model_index.get("boundary_ratio")
    return isinstance(br, (int, float)) and br > 0


def detect_wan_a14b_sets(roots: list[str]) -> list[dict]:
    """Scan `roots` (model dirs) for Wan2.2-A14B sets. Returns an ordered, unique
    list of descriptors, each resolvable by `resolve_wan_a14b_set`:

      single-file pair:
        {"name","kind":"single_file_pair","modality","high","low","shared"}
      diffusers A14B dir:
        {"name","kind":"diffusers_dir","modality","dir","loadable"}

    Detection is pure filesystem + cheap header reads. A single-file pair with no
    resolvable shared Wan diffusers dir (no vae/text_encoder source) is DROPPED
    with a log line (it can't be staged) rather than surfaced as unusable.
    """
    single_files: list[dict] = []      # {path, modality, expert('high'/'low')}
    diffusers_dirs: list[dict] = []     # {name, dir, modality, loadable, shared}
    shared_candidates: list[dict] = []  # {dir, is_a14b}
    seen_files: set[str] = set()
    seen_dirs: set[str] = set()

    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        try:
            entries = sorted(os.listdir(root))
        except OSError as e:  # noqa: BLE001 — unreadable root, skip
            logger.debug("detect_wan_a14b_sets: cannot list %s: %s", root, e)
            continue
        for name in entries:
            full = os.path.join(root, name)
            # -- single-file experts --
            if os.path.isfile(full) and _looks_like_a14b_single_file(full):
                real = os.path.realpath(full)
                if real in seen_files:
                    continue
                try:
                    if not is_comfyui_wan_single_file(full):
                        continue
                    in_ch, out_ch = detect_wan_modality(full)
                except Exception as e:  # noqa: BLE001 — corrupt/partial, skip
                    logger.debug("detect_wan_a14b_sets: header read failed %s: %s", full, e)
                    continue
                toks = _basename_tokens(full)
                expert = "high" if "high" in toks and "low" not in toks else \
                         "low" if "low" in toks and "high" not in toks else None
                if expert is None:
                    continue
                seen_files.add(real)
                single_files.append({"path": full, "modality": _modality_str(in_ch, out_ch),
                                     "expert": expert})
            # -- diffusers dirs (A14B set and/or shared source) --
            elif os.path.isdir(full):
                real = os.path.realpath(full)
                if real in seen_dirs:
                    continue
                seen_dirs.add(real)
                is_a14b = _is_diffusers_a14b_dir(full)
                if _dir_has_shared_components(full):
                    shared_candidates.append({"dir": full, "is_a14b": is_a14b})
                if not is_a14b:
                    continue
                try:
                    xcfg = _load_json(os.path.join(full, "transformer", "config.json"))
                    in_ch = xcfg.get("in_channels"); out_ch = xcfg.get("out_channels")
                    mod = _modality_str(in_ch, out_ch) if isinstance(in_ch, int) \
                        and isinstance(out_ch, int) else "t2v"
                    mi_path = os.path.join(full, "model_index.json")
                    mi = _load_json(mi_path) if os.path.isfile(mi_path) else None
                except Exception as e:  # noqa: BLE001
                    logger.debug("detect_wan_a14b_sets: diffusers config read failed %s: %s", full, e)
                    continue
                diffusers_dirs.append({"name": os.path.basename(full.rstrip("/")),
                                       "kind": "diffusers_dir", "modality": mod,
                                       "dir": full,
                                       "loadable": _diffusers_dir_is_loadable(mi)})

    # -- resolve a shared dir for the single-file pairs: prefer an A14B-family
    #    diffusers dir (guarantees the 14B vae/text_encoder — a TI2V-5B dir has a
    #    DIFFERENT vae and would silently mis-decode a 14B expert). --
    def _pick_shared() -> str | None:
        a14b = sorted(c["dir"] for c in shared_candidates if c["is_a14b"])
        pool = a14b or sorted(c["dir"] for c in shared_candidates)
        if not pool:
            return None
        if len(pool) > 1:
            logger.info("[comfyui_wan_remap] %d shared Wan diffusers dirs found; "
                        "using %s (A14B-family preferred)", len(pool), pool[0])
        return pool[0]

    shared_dir = _pick_shared()

    # -- pair single files by modality (one high + one low each) --
    sets: list[dict] = []
    by_mod: dict[str, dict[str, list[str]]] = {}
    for sf in single_files:
        by_mod.setdefault(sf["modality"], {"high": [], "low": []})[sf["expert"]].append(sf["path"])
    for mod in sorted(by_mod):
        highs, lows = sorted(by_mod[mod]["high"]), sorted(by_mod[mod]["low"])
        if len(highs) > 1 or len(lows) > 1:
            logger.info("[comfyui_wan_remap] %s A14B: multiple experts "
                        "(high=%d low=%d) — pairing %s + %s", mod, len(highs),
                        len(lows), os.path.basename(highs[0]) if highs else "-",
                        os.path.basename(lows[0]) if lows else "-")
        if not highs or not lows:
            logger.info("[comfyui_wan_remap] %s A14B experts incomplete "
                        "(high=%d low=%d) — skipping", mod, len(highs), len(lows))
            continue
        if shared_dir is None:
            logger.info("[comfyui_wan_remap] %s A14B single-file pair found but no "
                        "shared Wan diffusers dir (vae/text_encoder) to stage it — "
                        "skipping", mod)
            continue
        sets.append({"name": f"wan2.2-{mod}-A14B", "kind": "single_file_pair",
                     "modality": mod, "high": highs[0], "low": lows[0],
                     "shared": shared_dir})

    # -- append diffusers A14B dirs (unique display names) --
    used = {s["name"] for s in sets}
    for d in diffusers_dirs:
        nm = d["name"]
        if nm in used:
            nm = f"{nm} ({d['dir']})"
        d = dict(d); d["name"] = nm
        used.add(nm)
        sets.append(d)
    return sets


def stage_a14b_diffusers(src_dir: str, out_dir: str, *,
                         boundary_ratio: float | None = None,
                         force: bool = False) -> str:
    """Normalize a diffusers A14B dir into an engine-loadable model_dir.

    A genuine diffusers A14B dir already carries the engine's native two-expert
    layout (transformer/ + transformer_2/ sharded diffusers weights + shared
    vae/text_encoder/tokenizer/scheduler), read directly via `ShardedSafeTensors`.
    A real downloaded A14B-Diffusers dir (a `Wan…`-prefixed `_class_name` + a
    published `boundary_ratio > 0`) is ALREADY loadable and is passed through
    WITHOUT reaching here (`wan_detect` accepts any `Wan…` prefix; the two-expert
    gate needs only `boundary_ratio > 0` + a real `transformer_2/`). This
    normalization is the FALLBACK for a malformed dir — a NON-Wan `_class_name`, or
    a missing/zero `boundary_ratio` — producing a staged dir that SYMLINKS the
    source's expert + shared subdirs (weights byte-identical, zero copy) and writes
    a corrected `model_index.json` (a Wan-prefixed `_class_name` — the source's own
    when already Wan-prefixed, else `WanPipeline` — plus a resolved positive
    `boundary_ratio`). NO weights are touched (no dequant/remap) — it is a metadata
    + symlink normalization.

    Concurrency/crash-safe via the SAME primitives as `stage_two_expert`
    (`_stage_lock` / `_cleanup_stale_dirs` / marker+sentinel ownership), with the
    same marker-first / sentinel-off / swap_started ordering — but WITHOUT the
    heavy path's rollback-retry (there is no multi-GB build to protect: the source
    is intact and a rebuild is millisecond-scale, so a swap-in failure just leaves
    the old dir or an absent one that the next run instantly re-normalizes).
    """
    src_dir = os.path.abspath(src_dir)
    out_dir = os.path.abspath(out_dir)
    if not _is_diffusers_a14b_dir(src_dir):
        raise RuntimeError(
            f"not a diffusers A14B dir (need transformer/ + transformer_2/ with "
            f"config.json): {src_dir}")
    missing = [sub for sub in ("vae", "text_encoder")
               if not os.path.isdir(os.path.join(src_dir, sub))]
    if missing:
        raise RuntimeError(
            f"diffusers A14B dir {src_dir!r} is missing {missing} — it cannot "
            f"supply the shared components the engine needs.")

    base_mi_path = os.path.join(src_dir, "model_index.json")
    base_mi = _load_json(base_mi_path) if os.path.isfile(base_mi_path) else None
    xcfg = _load_json(os.path.join(src_dir, "transformer", "config.json"))
    in_ch, out_ch = xcfg.get("in_channels"), xcfg.get("out_channels")
    if not isinstance(in_ch, int) or not isinstance(out_ch, int):
        raise RuntimeError(f"{src_dir}/transformer/config.json lacks integer "
                           f"in_channels/out_channels")
    # The dir's OWN modality IS the experts' modality here (single source), so the
    # shared model_index's boundary is inherit-eligible (shared_modality matches).
    eff_boundary, boundary_src = resolve_boundary_ratio(
        boundary_ratio, base_mi, in_ch, out_ch, (in_ch, out_ch))

    # Protect the SOURCE model dir only (not its parent): unlike single-file experts
    # — which sit as loose files in a shared pool dir whose parent is worth guarding —
    # a diffusers `src_dir` IS a self-contained model dir, and its parent is typically
    # a models-root of INDEPENDENT model dirs; guarding that whole root would foreclose
    # a legitimate "stage as a sibling of the source" out_dir.
    _assert_safe_out_dir(out_dir, [src_dir])
    fp_inputs = [src_dir]
    for rel in ("model_index.json", "transformer/config.json",
                "transformer_2/config.json", "vae/config.json"):
        fp_inputs.append(os.path.join(src_dir, rel))
    fp = _fingerprint(fp_inputs, extra=f"{eff_boundary}|normalize|{in_ch}|{out_ch}")
    marker_name = ".qf_stage_complete"
    marker = os.path.join(out_dir, marker_name)

    # Normalize the whole set of subdirs the engine reads (experts + shared).
    _norm_subdirs = ("transformer", "transformer_2", "vae",
                     "text_encoder", "tokenizer", "scheduler")

    with _stage_lock(out_dir):
        if not force and os.path.isfile(marker):
            with open(marker, "r", encoding="utf-8") as f:
                if f.read().strip() == fp:
                    logger.info("[comfyui_wan_remap] normalized dir cache hit -> %s", out_dir)
                    return out_dir
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
        swap_started = False
        try:
            for sub in _norm_subdirs:
                s2 = os.path.join(src_dir, sub)
                if os.path.isdir(s2):
                    _stage_shared_dir(s2, os.path.join(tmp_dir, sub))
            # corrected model_index: a Wan-prefixed class (the engine detects any
            # `Wan…` prefix — keep the source's own published class when it is
            # already Wan-prefixed, else rewrite to the canonical WanPipeline) + a
            # resolved positive boundary_ratio; every other published key preserved.
            mi = dict(base_mi) if base_mi else {}
            src_cls = mi.get("_class_name")
            if not (isinstance(src_cls, str) and src_cls.startswith("Wan")):
                mi["_class_name"] = _ENGINE_WAN_PIPELINE_CLASS
            mi["boundary_ratio"] = float(eff_boundary)
            _dump_json(mi, os.path.join(tmp_dir, "model_index.json"))

            # Marker FIRST, sentinel off SECOND, swap_started gated at the marker
            # write (identical ownership-proof ordering to stage_two_expert's swap).
            with open(os.path.join(tmp_dir, marker_name), "w", encoding="utf-8") as f:
                f.write(fp)
            swap_started = True
            try:
                os.unlink(os.path.join(tmp_dir, _TMP_SENTINEL))
            except OSError as exc:
                logger.warning(
                    "[comfyui_wan_remap] could not remove the staging sentinel "
                    "(%s) — proceeding; a stray %s file may remain (harmless).",
                    exc, _TMP_SENTINEL)
            trash_dir = f"{out_dir}.trash-{os.getpid()}"
            old_moved = False
            if os.path.isdir(out_dir):
                os.replace(out_dir, trash_dir)
                old_moved = True
            try:
                os.replace(tmp_dir, out_dir)
            except BaseException:
                # Single rollback attempt (metadata rename) — no retry loop: unlike
                # the 56 GB dequant path there is nothing expensive to preserve, so
                # a compound fault simply self-heals on the next (ms) re-normalize.
                if old_moved and not os.path.isdir(out_dir):
                    try:
                        os.replace(trash_dir, out_dir)
                    except OSError:
                        logger.error(
                            "[comfyui_wan_remap] normalize swap AND rollback failed "
                            "— %s is absent; re-run to self-heal (the source %s is "
                            "intact).", out_dir, src_dir)
                raise
            if old_moved and os.path.isdir(trash_dir):
                shutil.rmtree(trash_dir, ignore_errors=True)
        finally:
            if not swap_started and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("[comfyui_wan_remap] normalized diffusers A14B dir (%s, "
                "boundary=%.3f [%s]) -> %s",
                _ENGINE_WAN_PIPELINE_CLASS, eff_boundary, boundary_src, out_dir)
    return out_dir


def resolve_wan_a14b_set(descriptor: dict, out_dir: str, *,
                         boundary_ratio: float | None = None,
                         force: bool = False) -> str:
    """Resolve a `detect_wan_a14b_sets` descriptor to an engine-loadable model_dir
    by DELEGATING to the staging above — the auto node adds NO staging of its own.

      single_file_pair -> stage_two_expert(high, low, shared, out_dir, ...)
                          (byte-identical to the manual combine node's call)
      diffusers_dir, already loadable -> the dir itself (zero staging)
      diffusers_dir, not loadable    -> stage_a14b_diffusers(dir, out_dir, ...)
    """
    kind = descriptor.get("kind")
    if kind == "single_file_pair":
        return stage_two_expert(descriptor["high"], descriptor["low"],
                                descriptor["shared"], out_dir,
                                boundary_ratio=boundary_ratio, force=force)
    if kind == "diffusers_dir":
        if descriptor.get("loadable"):
            return os.path.abspath(descriptor["dir"])
        return stage_a14b_diffusers(descriptor["dir"], out_dir,
                                    boundary_ratio=boundary_ratio, force=force)
    raise RuntimeError(f"unknown Wan A14B set kind: {kind!r}")
