"""Krea-2 fp8 text-encoder acceptance guard — SHARED by every adapter that can
hand a Krea-2 TE to the engine.

WHY THIS IS A SHARED MODULE AND NOT A PRIVATE HELPER:
the Krea-2 arch fingerprint makes `arch == "Krea2"` reachable from MORE than one
adapter (the ComfyUI single-file path, the bundled-checkpoint path, and the
HF-native synthesised-index path). A guard wired into only ONE of them is not a
guard — the others would stage an fp8 Krea-2 TE with zero validation and walk
straight into the engine blind spots below. Every Krea-2 TE staging route MUST
call `guard_krea2_te_fp8()`; `tests/test_krea2_te_fp8_guard.py` carries a
sweep-lock that fails the suite if a new `add_text_encoder(` site appears without
it.

ENGINE CONTRACT — *of the engine build that carries the krea2 TE fp8 dequant*
(branch `fix/krea2-te-fp8-dequant`, d44f01b8; see ENGINE DEPENDENCY in
format_adapters/comfyui_unet.py). On that build, `DequantFP8Provider`
(src/Serialization.h) dequantizes an fp8 weight ONLY when it has a sibling scale
(`<base>.scale_weight` OR `<base>.weight_scale`) that is F32 with numel == 1, and
THROWS on a block-FP8 `<base>.weight_scale_inv`, a per-channel [out] scale, a
non-F32 scale, and a numel-0 scale.

NOT TRUE OF ENGINE MAIN. On shipping main (9e00b0fe) none of that holds: there is
no `kBlockScaleInvSuffix`, no default-deny scale guard, and `krea2_te_factory`
(src/ComponentImpl.cpp) hardcodes `skip_fp8_dequant=true`, so the krea2 TE never
reaches DequantFP8Provider at all — its fp8 bytes are reinterpreted into the BF16
container (silent garbage). That is precisely why this plugin change MUST NOT be
released ahead of that engine build. Consequently this guard is, against MAIN, the
ONLY thing standing between an unsupported fp8 TE and silent garbage — every
refusal below is load-bearing there, not merely "stricter".

ENGINE BLIND SPOTS — even on the FIXED build we are deliberately STRICTER, never
looser (these two are NOT covered by the engine's own throws):
  * an fp8 tensor whose key is NOT `.weight`-suffixed: `scaleKeyFor` returns ""
    for it, so the scale block is SKIPPED and the tensor is dequantized with an
    IMPLICIT scale of 1.0 — silently wrong, with NO engine error.
  * an fp8 `.weight` with NO scale sibling at all: same implicit-1.0 blind spot.
We REFUSE both rather than mirror them.

FP8 PRESENCE IS READ FROM THE TENSOR DTYPES, NEVER FROM `__metadata__`:
`fingerprint_kind_from_metadata` returns "prequant_lighting_separate" on the BARE
PRESENCE of an untrusted metadata key (`method` / `precision_config` / …, no value
or signature check), so gating the fp8 validation on it would let ANY file skip
this guard by declaring one key — while the engine would STILL run
DequantFP8Provider over it. Read the DATA, not the label.
"""

from __future__ import annotations

import logging
import os
import struct
from pathlib import Path

from .safetensors_io import read_safetensors_header

logger = logging.getLogger(__name__)

_FP8_DTYPES = ("F8_E4M3", "F8_E5M2")
_FP8_WEIGHT_SUFFIX = ".weight"                             # engine kWeightSuffix
FP8_SCALE_SUFFIXES = (".scale_weight", ".weight_scale")    # engine kWeightScaleSuffixes
_FP8_SCALE_SUFFIXES = FP8_SCALE_SUFFIXES                    # in-module alias
_FP8_BLOCK_SCALE_SUFFIX = ".weight_scale_inv"              # engine kBlockScaleInvSuffix


def krea2_te_fp8_support(te_path: str, prefix: str = "") -> tuple[str, str]:
    """(verdict, reason); verdict ∈ {"no-fp8", "ok", "refuse"}.

    `prefix` scopes the scan to one component's key-slice (a BUNDLED checkpoint
    holds the TE under e.g. `text_encoder.` in the same file as the transformer —
    scanning the whole file would false-refuse on the transformer's own fp8).
    Empty prefix = the file IS the text encoder.

    "ok" ONLY when EVERY fp8 tensor in the slice is a `<base>.weight` carrying a
    per-tensor F32 SCALAR scale sibling and no block-FP8 sibling. Header-only.
    """
    # FAIL-CLOSED on a malformed container: the 8-byte length prefix is UNTRUSTED.
    # If it declares more header bytes than the file holds, the reader parses the
    # short buffer — and if that fragment happens to be valid JSON we would see a
    # TRUNCATED tensor list, miss the fp8 tensors that really are there, conclude
    # "no fp8" and ALLOW. A malformed file must never produce an ALLOW.
    try:
        with open(te_path, "rb") as f:
            declared = struct.unpack("<Q", f.read(8))[0]
        avail = os.path.getsize(te_path) - 8
    except Exception as e:                     # unreadable / shorter than 8 bytes
        return ("refuse", f"not a readable safetensors container ({e})")
    if avail < declared:
        return ("refuse", f"truncated safetensors: the header declares {declared} "
                          f"bytes but only {avail} follow — refusing to judge fp8 "
                          f"from a partial tensor list")

    try:
        hdr = read_safetensors_header(te_path)
    except Exception as e:                     # corrupt/oversized/unparseable header
        return ("refuse", f"unreadable safetensors header ({e})")
    if not isinstance(hdr, dict):
        return ("refuse", "safetensors header is not an object")
    hdr.pop("__metadata__", None)
    keys = {k for k in hdr if k.startswith(prefix)} if prefix else set(hdr)
    fp8_tensors = sorted(
        k for k in keys
        if isinstance(hdr.get(k), dict) and hdr[k].get("dtype") in _FP8_DTYPES)
    if not fp8_tensors:
        return ("no-fp8", "no fp8 tensors (BF16-native TE)")

    for wk in fp8_tensors:
        if not wk.endswith(_FP8_WEIGHT_SUFFIX):
            return ("refuse", f"fp8 tensor '{wk}' is not a '{_FP8_WEIGHT_SUFFIX}' key — "
                              f"the engine resolves a scale only for '.weight' keys and "
                              f"would dequantize this with an implicit scale of 1.0 "
                              f"(silently wrong)")
        base = wk[: -len(_FP8_WEIGHT_SUFFIX)]
        if (base + _FP8_BLOCK_SCALE_SUFFIX) in hdr:
            return ("refuse", f"'{base}{_FP8_BLOCK_SCALE_SUFFIX}' is a block-FP8 scale "
                              f"(DeepSeek/vLLM) — needs the block-dequant path, not this "
                              f"per-tensor provider")
        scale_key = next((base + s for s in _FP8_SCALE_SUFFIXES
                          if (base + s) in hdr), None)
        if scale_key is None:
            return ("refuse", f"fp8 weight '{wk}' has no per-tensor scale sibling "
                              f"(.scale_weight/.weight_scale) — not the fp8_scaled layout")
        sv = hdr[scale_key]
        # UNTRUSTED header: a non-dict entry / null-or-non-list shape / non-int dim
        # must produce this guard's clean refusal, not an AttributeError/TypeError.
        if not isinstance(sv, dict):
            return ("refuse", f"scale '{scale_key}' has a malformed header entry "
                              f"(not an object)")
        shape = sv.get("shape")
        if not isinstance(shape, list) or not all(isinstance(d, int) for d in shape):
            return ("refuse", f"scale '{scale_key}' has a malformed shape {shape!r}")
        numel = 1
        for d in shape:
            numel *= d
        if sv.get("dtype") != "F32" or numel != 1:
            return ("refuse", f"scale '{scale_key}' is not a per-tensor F32 scalar "
                              f"(dtype={sv.get('dtype')}, shape={shape}) — a "
                              f"per-channel [out] / block / non-F32 fp8 scale")
    return ("ok", f"{len(fp8_tensors)} fp8 weights, every one a per-tensor F32 "
                  f"scalar scale")


def guard_krea2_te_fp8(te_path: str, prefix: str = "",
                       prequantized_hint: bool = False) -> None:
    """Fail LOUD unless a Krea-2 fp8 TE is in the ONE layout the engine dequantizes.

    `prequantized_hint`: the staging is about to tell the engine this TE is
    PREQUANTIZED (`text_encoder_prequantized`). The engine takes that as
    `skip_fp8_dequant=true` and byte-reinterprets the fp8 bytes into the BF16
    container => SILENT GARBAGE. A bundled checkpoint can carry that flag in its
    own UNTRUSTED metadata, so if it is set while the TE genuinely holds fp8
    tensors, REFUSE — the two are mutually exclusive.

    A non-fp8 (BF16-native) TE is unaffected.

    ORDERING NOTE: the prequantized refusal fires only on an "ok" verdict, so a TE
    that is BOTH prequantized AND in an unsupported fp8 layout reports the generic
    "cannot dequantize" reason rather than the prequant-specific one. That is only
    a message-quality difference (both REFUSE), and it is unreachable today — the
    engine's export path rejects a non-fp16 Krea-2 TE outright — but if that export
    is ever relaxed, report the prequant reason first.
    """
    verdict, reason = krea2_te_fp8_support(te_path, prefix)
    if verdict == "no-fp8":
        return                              # BF16-native TE — nothing to guard
    name = Path(te_path).name
    if verdict == "ok" and prequantized_hint:
        raise RuntimeError(
            f"Krea-2 text encoder '{name}' holds fp8 weights AND is marked "
            f"PREQUANTIZED. The engine reads that marker as skip_fp8_dequant=true "
            f"and byte-reinterprets the fp8 bytes into its BF16 container — silent "
            f"garbage. Refusing: an fp8 Krea-2 TE must be dequantized, not skipped. "
            f"(If this really is a pre-quantized TE, it must not carry raw fp8 "
            f"tensors.)")
    if verdict == "ok":
        logger.info("[krea2_fp8_te] Krea-2 fp8 Qwen3-VL TE (%s): the engine's "
                    "DequantFP8Provider dequantizes fp8→bf16 — routing to staging. "
                    "REQUIRES an engine build with the krea2 TE fp8 dequant.", reason)
        return                              # engine-supported per-tensor F32 fp8
    raise RuntimeError(
        f"Krea-2 text encoder '{name}' is FP8 in a layout the QuantFunc engine "
        f"cannot dequantize: {reason}. The engine dequantizes ONLY the "
        f"per-tensor-scalar fp8_scaled layout (each fp8 `.weight` + an F32 scalar "
        f"'.weight_scale'/'.scale_weight'). Wire a BF16 Qwen3-VL 4B text encoder "
        f"(hidden 2560), or a per-tensor fp8_scaled one.")
