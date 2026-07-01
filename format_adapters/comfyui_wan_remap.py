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
     engine wrap lands, we dequant to fp16 on the plugin side (`dequant_fp8=True`,
     the default). Once the engine wires DequantFP8 + a KeyAliasingProvider for
     the Wan factory, `dequant_fp8=False` enables the zero-copy `key_remap.json`
     manifest path (`build_wan_xfm_manifest`) — no data copy, fp8 preserved.

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
import struct
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
# safetensors header IO  (stdlib only — no torch needed for detect/raw-remap)
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
    in_ch = 0
    for k, info in hdr.items():
        if k.startswith("patch_embedding") and k.endswith(".weight"):
            shp = info["shape"]
            if len(shp) == 5:
                in_ch = shp[1]
            break
    if not in_ch:
        raise RuntimeError(f"{path}: no patch_embedding.weight — not a Wan expert?")
    # Derive the patch volume from the patch_embedding kernel (kt*kh*kw).
    pt = ph = pw = 1
    for k, info in hdr.items():
        if k.startswith("patch_embedding") and k.endswith(".weight"):
            shp = info["shape"]
            if len(shp) == 5:
                pt, ph, pw = shp[2], shp[3], shp[4]
            break
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

def remap_file_raw(src: str | Path, dst: str | Path) -> tuple[int, list[str]]:
    """Rename keys and REPACK the data region, preserving each kept tensor's dtype
    (fp8 preserved, no torch). Dropped keys (fp8 `.scale_weight`/`.scale_input` + the
    `scaled_fp8` marker) are excised and the surviving tensors' `data_offsets` are
    renumbered to a gap-free `[0, N)` partition — a valid safetensors file (dropping a
    header entry while copying the blob verbatim would corrupt the offset table).

    Fastest path (byte copy, no dequant) but only LOADABLE by an engine that consumes
    fp8 for the Wan transformer factory — not the current one. Kept for the future
    zero-copy engine path + CPU structure checks; the node's default is the dequant path.
    """
    src, dst = str(src), str(dst)
    data_len = os.path.getsize(src)
    with open(src, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        if hlen > _MAX_HEADER_BYTES or hlen > data_len - 8:
            raise RuntimeError(
                f"{src}: safetensors header length {hlen} is implausible "
                f"(cap {_MAX_HEADER_BYTES}, file {data_len}) — refusing to parse")
        hdr = json.loads(fh.read(hlen))
        meta = hdr.pop("__metadata__", None)
        data_start = 8 + hlen
        data_region = data_len - data_start      # bytes available for tensor data
        new: dict = {}
        copies: list[tuple[int, int]] = []   # (old_start, old_end) in header order
        dropped: list[str] = []
        cursor = 0
        for k, info in hdr.items():
            if _is_droppable(k):
                dropped.append(k)
                continue
            off = info.get("data_offsets")
            if (not isinstance(off, (list, tuple)) or len(off) != 2):
                raise RuntimeError(f"{src}: tensor {k!r} has malformed data_offsets {off!r}")
            old_s, old_e = off
            # Validate the declared slice lies within the data region + is ordered
            # (an adversarial / corrupt header must fail LOUD, never silently emit a
            # truncated/corrupt output file).
            if not (isinstance(old_s, int) and isinstance(old_e, int)
                    and 0 <= old_s <= old_e <= data_region):
                raise RuntimeError(
                    f"{src}: tensor {k!r} data_offsets {off!r} out of range "
                    f"[0, {data_region}] — refusing to repack a corrupt file")
            size = old_e - old_s
            new[remap_key(k)] = {"dtype": info["dtype"], "shape": info["shape"],
                                 "data_offsets": [cursor, cursor + size]}
            copies.append((old_s, old_e))
            cursor += size
        n_keys = len(new)
        if meta is not None:
            new = {"__metadata__": meta, **new}
        nh = json.dumps(new, separators=(",", ":")).encode("utf-8")
        pad = (8 - (len(nh) % 8)) % 8
        nh += b" " * pad
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "wb") as out:
            out.write(struct.pack("<Q", len(nh)))
            out.write(nh)
            for old_s, old_e in copies:
                fh.seek(data_start + old_s)
                remaining = old_e - old_s
                while remaining > 0:
                    buf = fh.read(min(remaining, 64 * 1024 * 1024))
                    if not buf:
                        break
                    out.write(buf)
                    remaining -= len(buf)
    return n_keys, dropped


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
                # Scale applied ONLY when the sibling is present (a real read error on
                # it must surface, not silently emit unscaled garbage).
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


# ============================================================================
# Config synthesis
# ============================================================================

# The synthesized model_index ALWAYS carries _class_name="WanPipeline" — for BOTH
# t2v and i2v experts. Rationale (engine-verified):
#   * The engine's family detect (wan_detect, WanVideoPipeline.cpp) EXACT-matches
#     pipeline_class=="WanPipeline"; its transformer_class fallback only fires when
#     model_index has NO _class_name. "WanImageToVideoPipeline" is registered NOWHERE
#     in the engine — writing it would make the staged dir throw at load.
#   * t2v-vs-i2v behavior is CHANNEL-driven in the engine (is_i2v = in_channels >
#     latent channels; the VAE encoder loads when xfm_in > xfm_out), so the modality
#     information lives in the transformer config's in/out_channels we synthesize —
#     the pipeline-level class string plays no role in it.
# We own this synthesized file, so we write the value that is loadable everywhere.
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
        import shutil
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


def stage_two_expert(high_expert: str, low_expert: str, shared_dir: str,
                     out_dir: str, *, boundary_ratio: float = 0.9,
                     dequant_fp8: bool = True, force: bool = False) -> str:
    """Stage two single-file Wan experts into a two-expert diffusers model_dir.

    Layout produced:
        <out_dir>/transformer/{config.json, diffusion_pytorch_model.safetensors}
        <out_dir>/transformer_2/{config.json, diffusion_pytorch_model.safetensors}
        <out_dir>/{text_encoder,tokenizer,scheduler}/   (symlink → shared_dir)
        <out_dir>/vae/{*.safetensors (symlink), config.json (absent-key fixed)}
        <out_dir>/model_index.json  (_class_name + boundary_ratio)

    `shared_dir` is a Wan diffusers dir supplying vae/text_encoder/tokenizer/
    scheduler (+ a transformer/config.json used as the config base). Returns the
    staged model_dir path. Cache-aware: re-runs with identical sources skip.
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

    # Safety: never let a mis-pointed / workflow-supplied output_dir clobber the
    # experts, the shared model, or a directory holding the user's own content.
    _assert_safe_out_dir(out_dir, [high_expert, low_expert, shared_dir,
                                   os.path.dirname(high_expert),
                                   os.path.dirname(low_expert)])

    mode = "dequant" if dequant_fp8 else "raw"
    # Fingerprint the experts + the shared config files that get baked into the stage
    # (so an edit to a shared config invalidates the cache, not just a new dir mtime).
    fp_inputs = [high_expert, low_expert, shared_dir]
    for rel in ("model_index.json", "vae/config.json", "transformer/config.json"):
        fp_inputs.append(os.path.join(shared_dir, rel))
    fp = _fingerprint(fp_inputs, extra=f"{boundary_ratio}|{mode}|{in_ch}|{out_ch}")
    marker = os.path.join(out_dir, ".qf_stage_complete")
    if not force and os.path.isfile(marker):
        with open(marker, "r", encoding="utf-8") as f:
            if f.read().strip() == fp:
                logger.info("[comfyui_wan_remap] staged dir cache hit → %s", out_dir)
                return out_dir

    os.makedirs(out_dir, exist_ok=True)

    # ── transformers ─────────────────────────────────────────────────────
    base_xfm_cfg = {}
    shared_xfm_cfg = os.path.join(shared_dir, "transformer", "config.json")
    if os.path.isfile(shared_xfm_cfg):
        base_xfm_cfg = _load_json(shared_xfm_cfg)
    for sub, src in (("transformer", high_expert), ("transformer_2", low_expert)):
        d = os.path.join(out_dir, sub)
        os.makedirs(d, exist_ok=True)
        n_layers = _count_layers(_read_header(src))
        cfg = synthesize_transformer_config(base_xfm_cfg, in_ch, out_ch, n_layers)
        _dump_json(cfg, os.path.join(d, "config.json"))
        weight_dst = os.path.join(d, "diffusion_pytorch_model.safetensors")
        if dequant_fp8:
            nk = remap_dequant_file(src, weight_dst)
            logger.info("[comfyui_wan_remap] %s ← %s (remap+dequant, %d keys)",
                        sub, os.path.basename(src), nk)
        else:
            nk, dropped = remap_file_raw(src, weight_dst)
            logger.info("[comfyui_wan_remap] %s ← %s (raw remap fp8-preserved, "
                        "%d keys, dropped %d scale/marker)",
                        sub, os.path.basename(src), nk, len(dropped))

    # ── shared components (dir symlink / copy) ───────────────────────────
    for sub in _SHARED_SUBDIRS:
        s = os.path.join(shared_dir, sub)
        if os.path.isdir(s):
            _stage_shared_dir(s, os.path.join(out_dir, sub))

    # ── vae (link weights via the shared Windows-safe helper + fixed cfg) ─
    vae_src = os.path.join(shared_dir, "vae")
    if os.path.isdir(vae_src):
        vae_dst = os.path.join(out_dir, "vae")
        os.makedirs(vae_dst, exist_ok=True)
        for wf in os.listdir(vae_src):
            if wf.endswith(".safetensors"):
                # out_dir is guaranteed ours by _assert_safe_out_dir → link_or_copy
                # (hardlink → symlink → copy) is safe and cross-platform.
                link_or_copy(os.path.join(vae_src, wf), os.path.join(vae_dst, wf))
        vae_cfg_path = os.path.join(vae_src, "config.json")
        base_vae_cfg = _load_json(vae_cfg_path) if os.path.isfile(vae_cfg_path) else {}
        _dump_json(synthesize_vae_config(base_vae_cfg),
                   os.path.join(vae_dst, "config.json"))

    # ── model_index ──────────────────────────────────────────────────────
    shared_mi = os.path.join(shared_dir, "model_index.json")
    base_mi = _load_json(shared_mi) if os.path.isfile(shared_mi) else None
    _dump_json(synthesize_model_index(base_mi, class_name, boundary_ratio),
               os.path.join(out_dir, "model_index.json"))

    with open(marker, "w", encoding="utf-8") as f:
        f.write(fp)
    logger.info("[comfyui_wan_remap] staged two-expert %s dir (%s, boundary=%.3f) → %s",
                class_name, "dequant-fp16" if dequant_fp8 else "raw-fp8",
                boundary_ratio, out_dir)
    return out_dir
