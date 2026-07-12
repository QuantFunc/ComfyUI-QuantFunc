"""Tests for the Krea-2 fp8 text-encoder guard (PRECISE relaxation).

The engine build carrying the krea2 TE fp8 dequant (branch fix/krea2-te-fp8-dequant,
d44f01b8 — NOT engine main, which hardcodes skip_fp8_dequant=true) dequantizes the
PER-TENSOR F32 SCALAR fp8_scaled Qwen3-VL text tower (fp8 -> bf16 container -> int4). So the plugin guard in
format_adapters/comfyui_unet.py was NARROWED from "block ALL fp8" to
"ALLOW only the per-tensor-F32 fp8_scaled layout the engine supports; keep
per-channel / block-FP8 / non-F32 / missing-scale fp8 FAIL-LOUD".

This locks that boundary 1:1 with the engine's DEFAULT-DENY scale guard — the
plugin mirror of the engine ctest tests/cpp/test_fp8_scale_guards.cpp. If the
guard is ever loosened to blanket-allow fp8 (silent-garbage risk on a scale
layout the engine cannot dequantize), these tests go red.

GPU-free / torch-free: header-only safetensors + the guard's own header reads.

Run:  python3 tests/test_krea2_te_fp8_guard.py        (also pytest-compatible)
"""
import os
import sys
import types
import json
import struct
import tempfile
import importlib
import contextlib
from pathlib import Path as pathlib_Path

_PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PLUGIN)
_PKG = os.path.basename(_PLUGIN)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
# Stub heavy optional deps so a logic-only import doesn't drag ComfyUI/torch in.
for _n in ("comfy", "torch", "folder_paths", "comfy.model_management"):
    sys.modules.setdefault(_n, types.ModuleType(_n))

cu = importlib.import_module(f"{_PKG}.format_adapters.comfyui_unet")
_g = importlib.import_module(f"{_PKG}.format_adapters.tools.krea2_fp8_te")

_DSZ = {"F8_E4M3": 1, "F8_E5M2": 1, "F32": 4, "BF16": 2, "F16": 2, "U8": 1}
W = ".weight"


def _write_st(tensors, metadata=None):
    """tensors: list of (key, dtype, shape) -> path to a valid header + zero data."""
    hdr, off = {}, 0
    for k, dt, shp in tensors:
        n = 1
        for d in shp:
            n *= d
        nb = n * _DSZ[dt]
        hdr[k] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + nb]}
        off += nb
    if metadata:
        hdr["__metadata__"] = metadata
    hj = json.dumps(hdr).encode()
    fd, p = tempfile.mkstemp(suffix=".safetensors", prefix="krea2te_")
    with os.fdopen(fd, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(b"\0" * off)
    return p


def _cap_lib(supported=True):
    """A tiny fake engine .so that DOES / DOESN'T carry the krea2 fp8-TE
    capability sentinel — drives the guard's engine-capability probe.
    Returns a path the caller must os.remove()."""
    fd, p = tempfile.mkstemp(suffix=".so", prefix="krea2cap_")
    with os.fdopen(fd, "wb") as f:
        f.write(b"\x7fELF" + b"\0" * 64)          # plausible ELF preamble
        if supported:
            f.write(_g._KREA2_FP8_TE_CAP_TOKEN)    # the .rodata sentinel
        f.write(b"\0" * 4096)                      # exercise the streaming scan
    return p


@contextlib.contextmanager
def _capable_engine_env(supported=True):
    """Point the guard's SELF-resolution (QUANTFUNC_LIB) at a fake engine .so that
    DOES/DOESN'T advertise the capability — for tests that drive the guard through
    an adapter (which calls it internally, so no lib_path can be injected)."""
    lib = _cap_lib(supported)
    prev = os.environ.get("QUANTFUNC_LIB")
    os.environ["QUANTFUNC_LIB"] = lib
    _g._ENGINE_CAP_PROBE_CACHE.clear()
    try:
        yield lib
    finally:
        if prev is None:
            os.environ.pop("QUANTFUNC_LIB", None)
        else:
            os.environ["QUANTFUNC_LIB"] = prev
        _g._ENGINE_CAP_PROBE_CACHE.clear()
        os.remove(lib)


def _allows(tensors, metadata=None):
    """(guard_allowed, message). The guard RAISES to block, returns to allow.

    Layout-acceptance helper: runs against a fake engine that DOES advertise the
    krea2 fp8-TE capability, so it isolates the LAYOUT decision from the
    engine-capability gate (which has its own dedicated tests below). A BF16 TE
    short-circuits before the gate, so the fake lib is a harmless no-op there."""
    p = _write_st(tensors, metadata)
    lib = _cap_lib(True)
    try:
        _g.guard_krea2_te_fp8(p, lib_path=lib)
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        os.remove(p)
        os.remove(lib)


# ---- ALLOW: the engine-supported per-tensor F32 fp8_scaled layout ----
def test_per_tensor_f32_scalar_allowed():
    ok, _ = _allows([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                     ("m.l0.q.weight_scale", "F32", [])])
    assert ok


def test_per_tensor_f32_shape1_allowed():
    ok, _ = _allows([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                     ("m.l0.q.weight_scale", "F32", [1])])
    assert ok


def test_comfyui_alt_scale_weight_name_allowed():
    # engine kWeightScaleSuffixes accepts `.scale_weight` too (ComfyUI order)
    ok, _ = _allows([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                     ("m.l0.q.scale_weight", "F32", [])])
    assert ok


def test_bf16_native_te_passes_through():
    # not fp8 -> the guard early-returns (allow); a BF16 diffusers Krea-2 TE
    # (the pre-existing supported path) must stay unaffected.
    ok, _ = _allows([("m.l0.q" + W, "BF16", [4, 2]),
                     ("m.embed_tokens" + W, "BF16", [8, 4])])
    assert ok


# ---- BLOCK (fail loud): fp8 layouts the engine CANNOT dequantize ----
def test_per_channel_scale_blocked():
    # llm-compressor/compressed-tensors [out] scale. The engine's own DEFAULT-DENY
    # guard also rejects this; the plugin refuses first so the user gets an
    # actionable error without an engine round-trip (defense in depth).
    ok, msg = _allows([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                       ("m.l0.q.weight_scale", "F32", [4])])
    assert not ok and "per-tensor-scalar fp8_scaled" in msg


def test_block_fp8_weight_scale_inv_blocked():
    # DeepSeek/vLLM block-FP8 — engine needs the block-dequant path
    ok, msg = _allows([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                       ("m.l0.q.weight_scale_inv", "F32", [1, 1])])
    assert not ok and "block-FP8" in msg


def test_non_f32_scale_blocked():
    ok, msg = _allows([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                       ("m.l0.q.weight_scale", "BF16", [])])
    assert not ok and "per-tensor-scalar" in msg


def test_missing_scale_blocked():
    ok, msg = _allows([("m.l0.q" + W, "F8_E4M3", [4, 2])])
    assert not ok and "no per-tensor scale sibling" in msg


def test_one_bad_weight_among_good_blocked():
    # the engine throws on the FIRST bad weight it loads — the plugin scans all
    ok, msg = _allows([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                       ("m.l0.q.weight_scale", "F32", []),
                       ("m.l1.q" + W, "F8_E4M3", [4, 2]),
                       ("m.l1.q.weight_scale", "F32", [4])])
    assert not ok


def test_fp8_tensor_not_weight_suffixed_blocked():
    """An fp8 tensor under a NON-`.weight` key must be REFUSED.

    The engine's `scaleKeyFor` resolves a scale ONLY for a `.weight`-suffixed key
    — for any other key its scale block is skipped and the tensor dequantizes with
    an IMPLICIT scale of 1.0, silently wrong and with NO engine error. The plugin
    is deliberately STRICTER than the engine here rather than mirroring that blind
    spot: every `.weight` below is a perfectly valid per-tensor-F32 fp8, and the
    file must STILL be refused because of the one stray non-`.weight` fp8 tensor.
    """
    ok, msg = _allows([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                       ("m.l0.q.weight_scale", "F32", []),
                       ("m.l0.q.bias", "F8_E4M3", [4])])   # fp8, not `.weight`
    assert not ok and "implicit scale of 1.0" in msg


# ---- support() helper: verdict + reason ----
def test_support_helper_verdict_and_reason():
    v, reason = _g.krea2_te_fp8_support(
        _write_st([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                   ("m.l0.q.weight_scale", "F32", [])]))
    assert v == "ok" and "per-tensor F32 scalar" in reason
    v2, reason2 = _g.krea2_te_fp8_support(
        _write_st([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                   ("m.l0.q.weight_scale", "F32", [4])]))
    assert v2 == "refuse" and "per-channel" in reason2
    v3, _ = _g.krea2_te_fp8_support(
        _write_st([("m.l0.q" + W, "BF16", [4, 2])]))
    assert v3 == "no-fp8"


# ---- UNTRUSTED METADATA must not be able to skip the guard (spoof bypass) ----
def test_metadata_marker_cannot_bypass_guard():
    """A file's own `__metadata__` must NEVER decide whether fp8 is validated.

    `fingerprint_kind_from_metadata` returns "prequant_lighting_separate" on the
    BARE PRESENCE of an untrusted key (`method`, `precision_config`, ...), with no
    value or signature check. If the guard gated on that, ANY broken-layout fp8 TE
    could skip validation just by declaring one key — and the engine would STILL
    run DequantFP8Provider over it (it takes skip_fp8_dequant from opts/obfuscation,
    never from the file's metadata), hitting the implicit-1.0-scale blind spot =>
    silent garbage. So presence is read from the DTYPES. These are the exact
    bypasses: a broken fp8 layout + a spoofed marker must STILL be refused.
    """
    for marker in ({"method": "lighting"}, {"precision_config": "{}"},
                   {"quantfunc_obfuscated": "true"}, {"text_rotation_block_size": "256"}):
        # (a) fp8 weight with NO scale sibling (engine would apply implicit 1.0)
        ok, msg = _allows([("m.l0.q" + W, "F8_E4M3", [4, 2])], metadata=marker)
        assert not ok, f"spoofed {marker} bypassed the guard (missing-scale fp8)"
        # (b) fp8 tensor under a non-`.weight` key (same engine blind spot)
        ok, msg = _allows([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                           ("m.l0.q.weight_scale", "F32", []),
                           ("m.l0.q.bias", "F8_E4M3", [4])], metadata=marker)
        assert not ok, f"spoofed {marker} bypassed the guard (non-.weight fp8)"
        # (c) per-channel scale
        ok, msg = _allows([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                           ("m.l0.q.weight_scale", "F32", [4])], metadata=marker)
        assert not ok, f"spoofed {marker} bypassed the guard (per-channel scale)"
    # and a GOOD fp8 file with a marker is still allowed (no false-block)
    ok, _ = _allows([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                     ("m.l0.q.weight_scale", "F32", [])], metadata={"method": "lighting"})
    assert ok


# ---- a MALFORMED header must refuse cleanly, never crash / fail-open ----
def test_malformed_scale_header_refused_cleanly():
    v, reason = _g.krea2_te_fp8_support(
        _write_st([("m.l0.q" + W, "F8_E4M3", [4, 2])]))
    assert v == "refuse"          # no scale sibling
    # a scale whose shape is not a list of ints must refuse, not TypeError
    import json as _j, struct as _s, tempfile as _t, os as _o
    hdr = {"m.l0.q.weight": {"dtype": "F8_E4M3", "shape": [4, 2], "data_offsets": [0, 8]},
           "m.l0.q.weight_scale": {"dtype": "F32", "shape": None, "data_offsets": [8, 12]}}
    hj = _j.dumps(hdr).encode()
    fd, p = _t.mkstemp(suffix=".safetensors", prefix="krea2te_bad_")
    with _o.fdopen(fd, "wb") as f:
        f.write(_s.pack("<Q", len(hj))); f.write(hj); f.write(b"\0" * 12)
    try:
        v, reason = _g.krea2_te_fp8_support(p)
        assert v == "refuse" and "malformed shape" in reason
    finally:
        _o.remove(p)


# ---- is_distilled must come from a real signal, never a silent assumption ----
_hf = importlib.import_module(f"{_PKG}.format_adapters.tools.hf_layout")


def test_distilled_marker_positive_from_turbo_name():
    assert _hf.krea2_is_distilled("Krea2", "/x/krea2_turbo_fp8_scaled.safetensors") is True
    assert _hf.krea2_is_distilled("Krea2", "/x/Krea2-Turbo-BF16.safetensors") is True


def test_distilled_marker_none_when_ambiguous():
    """No turbo/distill signal => None (omit the key) => the engine uses its
    reference-default COMPUTED (base) shift and logs it. Must NOT silently
    assume turbo: forcing the FIXED shift onto a base checkpoint generates
    off-schedule with no error."""
    assert _hf.krea2_is_distilled("Krea2", "/x/krea2_base.safetensors") is None
    assert _hf.krea2_is_distilled("Krea2", "/x/mymodel.safetensors") is None


def test_distilled_marker_rejects_negated_names():
    """A name that EXPLICITLY negates distillation must NOT be read as turbo.

    A naive `"distill" in name` substring test reads `krea2_undistilled`,
    `krea2_non_distilled_base` and `krea2_distill-free` as POSITIVE and stamps the
    turbo FIXED timestep shift onto a checkpoint whose own name says the opposite —
    silently off-schedule, and via the "positive signal" branch, so it would not
    even WARN. Negated names must fall to the ambiguous/base path.
    """
    for n in ("krea2_undistilled.safetensors",
              "krea2_non_distilled_base.safetensors",
              "krea2_distill-free.safetensors",
              "krea2_no_distill.safetensors",
              "krea2_without_distill.safetensors",
              # the negator need not be ADJACENT, nor even BEFORE the marker:
              # any negation token anywhere makes the name AMBIGUOUS, never positive
              "krea2_not_a_turbo_model.safetensors",
              "krea2_no_longer_distilled.safetensors",
              "krea2-not-really-distilled.safetensors",
              "krea2-distilled-not-anymore.safetensors",
              # DIRECTIONAL: a negation pointing AT the distillation word disclaims
              # it even when a cfg word is also present. `cfg-no-turbo` = "no turbo".
              "krea2_cfg_no_turbo.safetensors",
              "krea2-guidance-non-distilled.safetensors",
              "krea2_cfg_non_distilled.safetensors"):
        assert _hf.krea2_is_distilled("Krea2", "/x/" + n) is None, n
    # ...while genuine positives still resolve
    for n in ("krea2_turbo_fp8_scaled.safetensors",
              "Krea2-Turbo-BF16.safetensors",
              "krea2_distilled_v2.safetensors",
              # `cfg-free` / `no-cfg` / `guidance-free` negate CFG, NOT distillation
              # — they are standard naming for a guidance-distilled (turbo) model.
              "krea2_turbo_no_cfg.safetensors",
              "krea2_turbo_cfg_free.safetensors",
              "krea2_turbo_guidance_free_fp8.safetensors",
              # a negation-LOOKING token pointing at a NON-distillation word must
              # not false-negate a real turbo name (denoise / anti-aliased / etc.)
              "krea2_turbo_de_noise.safetensors",
              "krea2_turbo_anti_aliased.safetensors",
              "krea2_turbo_no_artifact.safetensors",
              "krea2_turbo_non_square.safetensors"):
        assert _hf.krea2_is_distilled("Krea2", "/x/" + n) is True, n


def test_distilled_marker_inert_for_other_archs():
    # a `turbo`-named NON-Krea2 checkpoint must never get the key
    assert _hf.krea2_is_distilled("QwenImage", "/x/qwen_turbo.safetensors") is None


def test_every_synthesising_adapter_derives_is_distilled():
    """SWEEP LOCK (mechanism, not memory).

    `write_model_index` SYNTHESISES model_index.json, so EVERY caller must derive
    the Krea-2 distillation marker from a real signal. Applying the fix to only
    one adapter is how a base/turbo checkpoint silently lands on the wrong
    timestep schedule (the engine picks FIXED vs computed from this key, with no
    error either way). If a new adapter calls write_model_index without passing
    `is_distilled`, this test goes red.
    """
    import re
    import glob as _glob
    adapters = _glob.glob(os.path.join(_PLUGIN, "format_adapters", "**", "*.py"),
                          recursive=True)
    sites, offenders = 0, []
    for f in adapters:
        src = open(f).read()
        for m in re.finditer(r"\.write_model_index\(", src):
            sites += 1
            tail = src[m.end():m.end() + 240]
            # must DERIVE it, not hardcode: require the shared helper call
            if "krea2_is_distilled(" not in tail:
                offenders.append(f"{os.path.basename(f)}:{src[:m.start()].count(chr(10))+1}")
    assert sites >= 4, f"sweep is vacuous — found only {sites} write_model_index sites"
    assert not offenders, (
        f"write_model_index() called without deriving is_distilled at: {offenders} "
        f"— every synthesising adapter must call tools.hf_layout.krea2_is_distilled(), "
        f"never hardcode or silently default the Krea-2 timestep marker.")


def test_every_te_staging_route_runs_the_fp8_guard():
    """SWEEP LOCK #2 (the one that matters most).

    `arch == "Krea2"` is reachable from MORE than one adapter, so EVERY route that
    stages a Krea-2 text encoder must call the shared fp8 guard. A guard wired into
    only one adapter is not a guard: the others would hand an unsupported fp8 TE to
    the engine, which dequantizes it with an implicit 1.0 scale — silent garbage,
    no error anywhere. If a new `add_text_encoder(` site appears without the guard
    reachable in its enclosing function, this test goes red.
    """
    import re
    import glob as _glob
    adapters = _glob.glob(os.path.join(_PLUGIN, "format_adapters", "**", "*.py"),
                          recursive=True)
    sites, offenders = 0, []
    for f in adapters:
        src = open(f).read()
        if "def add_text_encoder" in src:      # the HFLayout definition itself
            continue
        for m in re.finditer(r"\.add_text_encoder\(", src):
            sites += 1
            # the guard must appear somewhere in the same enclosing function: scan
            # back to the previous `def ` boundary.
            head = src[:m.start()]
            fn_start = head.rfind("\n    def ")
            body = head[fn_start:] if fn_start != -1 else head
            if "guard_krea2_te_fp8(" not in body:
                offenders.append(f"{os.path.basename(f)}:{head.count(chr(10))+1}")
    assert sites >= 4, f"sweep is vacuous — found only {sites} add_text_encoder sites"
    assert not offenders, (
        f"add_text_encoder() staged WITHOUT the Krea-2 fp8 guard at: {offenders} — "
        f"call tools.krea2_fp8_te.guard_krea2_te_fp8() (gated on arch=='Krea2') "
        f"before staging a text encoder, or an unsupported fp8 Krea-2 TE reaches the "
        f"engine's implicit-1.0-scale blind spot with no error.")


def test_write_model_index_omits_is_distilled_without_signal(tmp_path=None):
    """HFLayout must not stamp is_distilled unless the caller determined it."""
    import tempfile as _tf
    from pathlib import Path as _P
    hf = importlib.import_module(f"{_PKG}.format_adapters.tools.hf_layout")
    root = _P(_tf.mkdtemp(prefix="krea2_mi_"))
    lay = hf.HFLayout(root)
    lay.write_model_index("Krea2")                      # no signal
    mi = json.loads((root / "model_index.json").read_text())
    assert "is_distilled" not in mi, mi
    assert mi["_class_name"] == "Krea2Pipeline"
    lay.write_model_index("Krea2", is_distilled=True)   # positive signal
    mi = json.loads((root / "model_index.json").read_text())
    assert mi["is_distilled"] is True


# ---- a MALFORMED CONTAINER must never fail OPEN (allow) ----
def test_malformed_container_never_fails_open():
    """corrupt / truncated / oversized / empty headers must NOT be ALLOWED.

    A truncated file is the subtle one: the 8-byte length prefix is untrusted, so
    if it declares MORE header bytes than the file holds, the reader parses the
    short buffer — and if that fragment happens to be valid JSON we would see a
    TRUNCATED tensor list, miss the fp8 tensors that really are there, conclude
    "no fp8" and ALLOW the file. Fail-closed instead.
    """
    import struct as _s
    cases = {
        "corrupt-json":  _s.pack("<Q", 20) + b"{not valid json!!!!!",
        "oversized":     _s.pack("<Q", 1 << 40) + b"{}",
        "truncated":     _s.pack("<Q", 500) + b'{"a":1}',   # declares 500, has 7
        "empty":         b"",
    }
    for label, blob in cases.items():
        fd, p = tempfile.mkstemp(suffix=".safetensors", prefix="krea2te_bad_")
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        try:
            allowed = True
            try:
                _g.guard_krea2_te_fp8(p)   # returning = ALLOW
            except Exception:
                allowed = False                          # raising = refuse (loud)
            assert not allowed, f"malformed container '{label}' FAILED OPEN (allowed)"
        finally:
            os.remove(p)


# ---- the bundled-checkpoint route: prefix-scoped scan + prequantized refusal ----


def test_prequantized_plus_fp8_is_refused():
    """fp8 + `text_encoder_prequantized` is the OTHER silent-garbage path.

    A bundle carries that flag in its OWN untrusted metadata; the engine reads it
    as skip_fp8_dequant=true and byte-reinterprets the fp8 bytes into its BF16
    container. The two are mutually exclusive — an fp8 TE must be DEQUANTIZED, not
    skipped — so the guard refuses the combination.
    """
    good_fp8 = _write_st([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                          ("m.l0.q.weight_scale", "F32", [])])
    cap = _cap_lib(True)                                  # capability-advertising engine
    _g.guard_krea2_te_fp8(good_fp8, lib_path=cap)         # fine on its own
    try:
        _g.guard_krea2_te_fp8(good_fp8, prequantized_hint=True, lib_path=cap)
        raised = False
    except RuntimeError as e:
        raised = "PREQUANTIZED" in str(e) or "prequantized" in str(e).lower()
    os.remove(good_fp8); os.remove(cap)
    assert raised, "fp8 + prequantized must be REFUSED (engine would skip dequant)"
    # a BF16 TE marked prequantized is legitimate — must NOT be refused
    bf16 = _write_st([("m.l0.q" + W, "BF16", [4, 2])])
    _g.guard_krea2_te_fp8(bf16, prequantized_hint=True)   # no raise
    os.remove(bf16)


def test_bundle_prefix_scoped_scan():
    """A bundled checkpoint holds the TE as a key PREFIX SLICE of the same file as
    the transformer. The guard must judge ONLY the TE slice — scanning the whole
    file would false-refuse on the transformer's own (irrelevant) fp8 — while the
    scale-sibling lookups still resolve against the full header."""
    # TE slice is a clean per-tensor-F32 fp8; the transformer slice has fp8 with a
    # per-channel scale (unsupported) — which must NOT taint the TE verdict.
    f = _write_st([
        ("text_encoder.l0.q" + W, "F8_E4M3", [4, 2]),
        ("text_encoder.l0.q.weight_scale", "F32", []),
        ("model.diffusion_model.b0" + W, "F8_E4M3", [4, 2]),
        ("model.diffusion_model.b0.weight_scale", "F32", [4]),   # per-channel
    ])
    v, reason = _g.krea2_te_fp8_support(f, prefix="text_encoder.")
    assert v == "ok", f"TE slice wrongly refused: {reason}"
    cap = _cap_lib(True)
    _g.guard_krea2_te_fp8(f, prefix="text_encoder.", lib_path=cap)   # no raise
    os.remove(cap)
    # ...and a BAD TE slice IS refused even though the transformer slice is fine
    f2 = _write_st([
        ("text_encoder.l0.q" + W, "F8_E4M3", [4, 2]),            # no scale sibling
        ("model.diffusion_model.b0" + W, "F8_E4M3", [4, 2]),
        ("model.diffusion_model.b0.weight_scale", "F32", []),
    ])
    v2, _ = _g.krea2_te_fp8_support(f2, prefix="text_encoder.")
    assert v2 == "refuse"
    os.remove(f); os.remove(f2)


def test_same_length_invalid_json_header_refused_cleanly():
    """A header whose declared length MATCHES the file but whose bytes are not
    valid JSON must yield the guard's clean refusal, not a raw JSONDecodeError."""
    import struct as _s
    blob = b"{not-json-at-all}"
    fd, p = tempfile.mkstemp(suffix=".safetensors", prefix="krea2te_bad_")
    with os.fdopen(fd, "wb") as f:
        f.write(_s.pack("<Q", len(blob)))
        f.write(blob)
    try:
        v, reason = _g.krea2_te_fp8_support(p)
        assert v == "refuse" and "unreadable" in reason
    finally:
        os.remove(p)


# ---- engine-capability gate: default-REFUSE a well-formed fp8 TE on a skewed
#      engine (a version floor cannot see the skew; silent garbage otherwise) ----
_GOOD_FP8 = [("m.l0.q" + W, "F8_E4M3", [4, 2]), ("m.l0.q.weight_scale", "F32", [])]


@contextlib.contextmanager
def _clean_cap_env():
    """Neutralise the two ambient inputs to the capability gate (QUANTFUNC_LIB +
    the opt-in env) so a test drives it deterministically, and restore after."""
    saved = {k: os.environ.get(k)
             for k in ("QUANTFUNC_LIB", _g._KREA2_FP8_TE_OPT_IN_ENV)}
    for k in saved:
        os.environ.pop(k, None)
    _g._ENGINE_CAP_PROBE_CACHE.clear()
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _g._ENGINE_CAP_PROBE_CACHE.clear()


def test_wellformed_fp8_refused_when_engine_lacks_capability():
    """The VULN backstop: a PERFECT per-tensor-F32 fp8 TE staged onto an engine
    that does NOT advertise the krea2 fp8-TE dequant (shipping main) must be
    REFUSED — otherwise the engine byte-reinterprets fp8 -> BF16 = silent garbage."""
    with _clean_cap_env():
        p = _write_st(_GOOD_FP8)
        lib = _cap_lib(supported=False)          # engine WITHOUT the sentinel
        try:
            raised = False
            try:
                _g.guard_krea2_te_fp8(p, lib_path=lib)
            except RuntimeError as e:
                raised = "does not advertise" in str(e)
            assert raised, "well-formed fp8 TE was ALLOWED on a non-capable engine"
        finally:
            os.remove(p); os.remove(lib)


def test_wellformed_fp8_allowed_when_engine_advertises_capability():
    """On an engine that DOES carry the capability sentinel, the same well-formed
    fp8 TE is allowed (option (a): no UX cost once the fixed engine ships)."""
    with _clean_cap_env():
        p = _write_st(_GOOD_FP8)
        lib = _cap_lib(supported=True)
        try:
            _g.guard_krea2_te_fp8(p, lib_path=lib)   # must not raise
        finally:
            os.remove(p); os.remove(lib)


def test_wellformed_fp8_refused_when_engine_unresolvable():
    """If the installed engine cannot be located to verify (probe -> None), the
    guard fails CLOSED — 'cannot verify' is not 'safe'."""
    with _clean_cap_env():
        p = _write_st(_GOOD_FP8)
        try:
            raised = False
            try:
                _g.guard_krea2_te_fp8(p, lib_path="/nonexistent/libquantfunc.so")
            except RuntimeError as e:
                raised = "could not be located" in str(e)
            assert raised, "unverifiable engine did not fail closed on a fp8 TE"
        finally:
            os.remove(p)


def test_opt_in_env_overrides_capability_gate():
    """A user who KNOWS their engine supports it can opt in — even with no capable
    lib resolvable — and the fp8 TE is allowed (with a loud override warning)."""
    with _clean_cap_env():
        os.environ[_g._KREA2_FP8_TE_OPT_IN_ENV] = "1"
        p = _write_st(_GOOD_FP8)
        try:
            _g.guard_krea2_te_fp8(p, lib_path="/nonexistent/libquantfunc.so")
        finally:
            os.remove(p)


def test_opt_in_env_falsey_values_do_not_override():
    """An explicit falsey opt-in ('0'/'false'/'no'/'off'/'') must NOT lift the
    gate — only a truthy value does."""
    with _clean_cap_env():
        p = _write_st(_GOOD_FP8)
        try:
            for falsey in ("0", "false", "no", "off", "", "  "):
                os.environ[_g._KREA2_FP8_TE_OPT_IN_ENV] = falsey
                _g._ENGINE_CAP_PROBE_CACHE.clear()
                raised = False
                try:
                    _g.guard_krea2_te_fp8(p, lib_path="/nonexistent/libquantfunc.so")
                except RuntimeError:
                    raised = True
                assert raised, f"falsey opt-in {falsey!r} wrongly lifted the gate"
        finally:
            os.remove(p)


def test_bf16_te_unaffected_by_capability_gate():
    """A BF16-native TE short-circuits BEFORE the capability gate — it is allowed
    even against a non-capable engine (nothing to dequantize, nothing to garble)."""
    with _clean_cap_env():
        p = _write_st([("m.l0.q" + W, "BF16", [4, 2])])
        lib = _cap_lib(supported=False)
        try:
            _g.guard_krea2_te_fp8(p, lib_path=lib)   # must not raise
        finally:
            os.remove(p); os.remove(lib)


def test_capability_probe_scans_across_chunk_boundary():
    """The streaming byte-scan must find a sentinel that STRADDLES a read-chunk
    boundary (the .rodata token can sit anywhere in a ~300MB .so). The scan's
    invariant is chunk >> token (production chunk = 8MB, token = 33B); the test
    uses the smallest realistic chunk (> token) that still forces a straddle."""
    with _clean_cap_env():
        tok = _g._KREA2_FP8_TE_CAP_TOKEN
        chunk = len(tok) + 8                     # > token, so a straddle fits the window
        fd, lib = tempfile.mkstemp(suffix=".so", prefix="krea2capX_")
        with os.fdopen(fd, "wb") as f:
            # token spans the first chunk edge: 5 bytes in chunk 1, the rest in chunk 2.
            f.write(b"\0" * (chunk - 5) + tok + b"\0" * 40)
        saved_chunk = _g._CAP_PROBE_CHUNK_BYTES
        _g._CAP_PROBE_CHUNK_BYTES = chunk
        try:
            assert _g.engine_supports_krea2_te_fp8_dequant(lib_path=lib) is True, \
                "sentinel straddling a chunk boundary was missed"
        finally:
            _g._CAP_PROBE_CHUNK_BYTES = saved_chunk
            os.remove(lib)


def test_capability_probe_none_when_no_lib_resolves():
    """No explicit lib, no QUANTFUNC_LIB, nodes not importable -> probe returns
    None (not a false True/False)."""
    with _clean_cap_env():
        # lib_path pointing at a missing file resolves to None deterministically.
        assert _g.engine_supports_krea2_te_fp8_dequant(
            lib_path="/nonexistent/libquantfunc.so") is None


# Adapters that can resolve arch=="Krea2" and hand a TE to the engine. Any route
# in these files that returns a StagingResult must have the guard reachable — this
# is the companion to the add_text_encoder sweep, which is structurally blind to a
# route that never calls add_text_encoder() (exactly how the hf_native AS-IS
# model_dir route hid: it returns the source dir untouched).
def test_every_staging_result_route_in_krea2_adapters_is_guarded():
    """SWEEP LOCK #3 — closes the blind spot the add_text_encoder sweep has.

    A route can hand a Krea-2 TE to the engine WITHOUT calling add_text_encoder():
    hf_native's AS-IS branch returns the source model_dir untouched and the engine
    loads its text_encoder/ directly. So every `return StagingResult(` in a
    Krea2-capable adapter must have the fp8 guard reachable in its enclosing
    function. A new route that forgets it fails the suite.
    """
    import re
    import glob as _glob
    # glob EVERY adapter (mechanism, not a hand-kept list): any file that mentions
    # "Krea2" is a candidate. For each `return StagingResult(` whose ENCLOSING
    # FUNCTION handles Krea2 (its body mentions "Krea2"), the guard CALL must be
    # reachable — the `(` call form, NOT the bare name, so a lingering import line
    # can't satisfy it. This closes both the hardcoded-tuple gap and the
    # import-substring gap two prior reviewers found.
    adapters = [f for f in _glob.glob(
        os.path.join(_PLUGIN, "format_adapters", "**", "*.py"), recursive=True)
        if "Krea2" in open(f).read()]
    sites, offenders = 0, []
    for f in adapters:
        src = open(f).read()
        for m in re.finditer(r"return StagingResult\(", src):
            head = src[:m.start()]
            fn_start = head.rfind("\n    def ")
            body = head[fn_start:] if fn_start != -1 else head
            if "Krea2" not in body:
                continue                      # this return's function never handles Krea2
            sites += 1
            if "guard_krea2_te_fp8(" not in body:
                offenders.append(f"{os.path.basename(f)}:{head.count(chr(10)) + 1}")
    assert sites >= 1, "sweep is vacuous — no Krea2 StagingResult site found"
    assert not offenders, (
        f"a Krea2-capable adapter returns a StagingResult without the fp8 guard "
        f"CALL reachable at: {offenders} — every route that can hand a Krea-2 TE to "
        f"the engine must call tools.krea2_fp8_te.guard_krea2_te_fp8() (gated on "
        f"arch=='Krea2'), including routes that never call add_text_encoder().")


def test_prequantized_sidecar_cannot_skip_dequant_on_an_fp8_te():
    """An UNTRUSTED sidecar `quantfunc_config.json` must not be able to turn OFF
    the engine's fp8 dequant for a Krea-2 TE that genuinely holds fp8 weights.

    The engine reads `text_encoder.prequantized` as skip_fp8_dequant=true and
    byte-reinterprets the fp8 bytes into its BF16 container => silent garbage. The
    guard must therefore see the SAME prequantized state the engine will: a VALID
    per-tensor-F32 fp8 TE + a prequantized sidecar is a REFUSAL, not a pass. (The
    early guard alone was insufficient: hf_native merges that sidecar AFTER it.)
    """
    import tempfile as _tf
    from pathlib import Path as _P
    hn = importlib.import_module(f"{_PKG}.format_adapters.hf_native")
    base = importlib.import_module(f"{_PKG}.format_adapters.base")

    def _mk(preq):
        r = _P(_tf.mkdtemp(prefix="krea2_preq_"))
        (r / "transformer").mkdir(); (r / "text_encoder").mkdir()
        x = _write_st([("txtfusion.projector.weight", "BF16", [2]),
                       ("blocks.0.attn.wq.weight", "BF16", [2]),
                       ("blocks.0.mod.lin.weight", "BF16", [2])])
        os.replace(x, r / "transformer" / "diffusion_pytorch_model.safetensors")
        (r / "transformer" / "config.json").write_text(
            json.dumps({"_class_name": "Krea2Transformer2DModel"}))
        (r / "model_index.json").write_text(
            json.dumps({"_class_name": "Krea2Pipeline", "is_distilled": True}))
        # a PERFECTLY VALID per-tensor-F32 fp8 TE — the guard would say "ok"
        t = _write_st([("m.l0.q" + W, "F8_E4M3", [4, 2]),
                       ("m.l0.q.weight_scale", "F32", [])])
        os.replace(t, r / "text_encoder" / "model.safetensors")
        (r / "text_encoder" / "config.json").write_text(
            json.dumps({"_class_name": "Qwen3VLForConditionalGeneration"}))
        if preq:
            (r / "quantfunc_config.json").write_text(
                json.dumps({"text_encoder": {"prequantized": True}}))
        return r

    def _adapt(preq):
        r = _mk(preq)
        ref = base.FileRef(
            path=str(r / "transformer" / "diffusion_pytorch_model.safetensors"),
            arch="", kind="", mtime=0.0)
        hn.HFLayoutAdapter().adapt(base.SourceBundle(transformer=ref),
                                   _P(_tf.mkdtemp()), base.BuildContext())

    # A capability-advertising engine, so the allow path isolates the PREQUANTIZED
    # decision from the engine-capability gate (which has its own tests).
    with _capable_engine_env(True):
        _adapt(False)                   # valid fp8 TE, no sidecar -> allowed
        try:
            _adapt(True)
            refused = False
        except RuntimeError as e:
            refused = "PREQUANTIZED" in str(e) or "prequantized" in str(e).lower()
    assert refused, ("an untrusted prequantized sidecar silently disabled the fp8 "
                     "dequant on a genuine fp8 Krea-2 TE (engine => silent garbage)")


def test_asis_model_dir_route_is_also_guarded():
    """The HF-native AS-IS route hands an existing model_dir to the engine with no
    add_text_encoder() call — so the staging sweep-lock does not see it — yet the
    engine still loads that dir's text_encoder/. A Krea-2 diffusers dir holding a
    broken-layout fp8 TE would otherwise reach the implicit-1.0-scale blind spot
    unguarded. It must be REFUSED; a good fp8 / BF16 TE must still pass."""
    import tempfile as _tf
    from pathlib import Path as _P
    hn = importlib.import_module(f"{_PKG}.format_adapters.hf_native")
    base = importlib.import_module(f"{_PKG}.format_adapters.base")

    def _mk(te_tensors):
        r = _P(_tf.mkdtemp(prefix="krea2_asis_"))
        (r / "transformer").mkdir(); (r / "text_encoder").mkdir()
        x = _write_st([("txtfusion.projector.weight", "BF16", [2]),
                       ("blocks.0.attn.wq.weight", "BF16", [2]),
                       ("blocks.0.mod.lin.weight", "BF16", [2])])
        os.replace(x, r / "transformer" / "diffusion_pytorch_model.safetensors")
        (r / "transformer" / "config.json").write_text(
            json.dumps({"_class_name": "Krea2Transformer2DModel"}))
        (r / "model_index.json").write_text(
            json.dumps({"_class_name": "Krea2Pipeline", "is_distilled": True}))
        t = _write_st(te_tensors)
        os.replace(t, r / "text_encoder" / "model.safetensors")
        (r / "text_encoder" / "config.json").write_text(
            json.dumps({"_class_name": "Qwen3VLForConditionalGeneration"}))
        return r

    def _adapt(te_tensors):
        r = _mk(te_tensors)
        ref = base.FileRef(
            path=str(r / "transformer" / "diffusion_pytorch_model.safetensors"),
            arch="", kind="", mtime=0.0)
        hn.HFLayoutAdapter().adapt(base.SourceBundle(transformer=ref),
                                   _P(_tf.mkdtemp()), base.BuildContext())

    # BROKEN fp8 TE (no scale sibling) -> refused regardless of engine capability
    # (the layout refusal precedes the capability gate).
    try:
        _adapt([("m.l0.q" + W, "F8_E4M3", [4, 2])])
        refused = False
    except RuntimeError:
        refused = True
    assert refused, "AS-IS model_dir route staged a BROKEN fp8 Krea-2 TE unguarded"
    # GOOD fp8 (on a capability-advertising engine) + BF16 TEs must still pass
    with _capable_engine_env(True):
        _adapt([("m.l0.q" + W, "F8_E4M3", [4, 2]), ("m.l0.q.weight_scale", "F32", [])])
    _adapt([("m.l0.q" + W, "BF16", [4, 2])])


# ---- claims that previously rested on prose, now locked by tests ----
_af = importlib.import_module(f"{_PKG}.format_adapters.tools.arch_fingerprint")
_rm = importlib.import_module(f"{_PKG}.format_adapters.comfyui_krea2_remap")


def _keys_file(keys):
    return _write_st([(k, "BF16", [2]) for k in keys])


def test_arch_fingerprint_krea2_is_collision_free():
    """The Krea2 double-signal (txtfusion.* AND blocks.N.attn.wq|mod.lin) must fire
    on Krea-2 and on NOTHING else — a mis-fingerprint reroutes another arch's
    staging into the Krea-2 remap."""
    krea2 = _keys_file(["txtfusion.projector.weight",
                        "blocks.0.attn.wq.weight", "blocks.0.mod.lin.weight"])
    assert _af.fingerprint_arch_from_keys(krea2) == "Krea2"
    # each of these must NOT be Krea2
    foreign = {
        "wan":   ["blocks.0.self_attn.q.weight", "blocks.0.ffn.0.weight"],
        "klein": ["double_blocks.0.img_attn.qkv.weight", "single_blocks.0.linear1.weight"],
        "qwen":  ["transformer_blocks.0.attn.to_q.weight"],
        "zimage": ["cap_embedder.0.weight", "layers.0.attention.qkv.weight"],
        # txtfusion alone, WITHOUT the krea2 block signal, must not claim Krea2
        "txtfusion_only": ["txtfusion.projector.weight"],
        # krea2-ish blocks WITHOUT txtfusion must not claim Krea2
        "blocks_only": ["blocks.0.attn.wq.weight", "blocks.0.mod.lin.weight"],
    }
    for name, keys in foreign.items():
        got = _af.fingerprint_arch_from_keys(_keys_file(keys))
        assert got != "Krea2", f"{name} mis-fingerprinted as Krea2 (got {got!r})"


def test_krea2_remap_header_reader_is_dos_capped():
    """The transformer-remap path must NOT re-inline a raw safetensors header read.

    The 8-byte length prefix is UNTRUSTED: a crafted file declaring an exabyte
    header makes a naive `f.read(n)` attempt a multi-GB allocation (OOM/hang).
    The shared reader (tools/safetensors_io) carries a _MAX_HEADER_BYTES cap, and
    comfyui_wan_remap already reuses it — comfyui_krea2_remap must too.
    """
    import struct as _s
    fd, p = tempfile.mkstemp(suffix=".safetensors", prefix="krea2_dos_")
    with os.fdopen(fd, "wb") as f:
        f.write(_s.pack("<Q", 2 ** 62))      # declares ~4.6 exabytes
        f.write(b"{}")
    try:
        # must not raise MemoryError / hang; either a clean refusal or False
        try:
            _rm.is_krea2_bfl(pathlib_Path(p))
        except MemoryError:
            raise AssertionError("is_krea2_bfl OOM'd on a crafted header (no DoS cap)")
        except Exception:
            pass                              # a clean refusal is fine
        try:
            _rm.build_krea2_xfm_manifest(pathlib_Path(p))
        except MemoryError:
            raise AssertionError("build_krea2_xfm_manifest OOM'd (no DoS cap)")
        except Exception:
            pass                              # clean refusal (ValueError from the cap)
    finally:
        os.remove(p)


def test_krea2_remap_rejects_int8_convrot_fail_loud():
    """The ComfyUI ConvRot INT8 Krea-2 format is NOT loadable (the engine has no
    int8 dequant on the fresh-quant path) — it must fail LOUD, never stage."""
    import pytest as _pt
    convrot = _keys_file(["txtfusion.projector.weight",
                          "blocks.0.attn.wq.weight", "blocks.0.mod.lin.weight",
                          "blocks.0.attn.wq.comfy_quant"])
    with _pt.raises(RuntimeError, match="ConvRot"):
        _rm.build_krea2_xfm_manifest(pathlib_Path(convrot))


def test_krea2_remap_unmapped_key_fails_loud():
    """The remap is completeness-guarded: an unknown source key must RAISE (a
    silently-dropped weight is a wrong-output bug), never be skipped."""
    import pytest as _pt
    bad = _keys_file(["txtfusion.projector.weight",
                      "blocks.0.attn.wq.weight", "blocks.0.mod.lin.weight",
                      "blocks.0.totally_unknown_thing.weight"])
    with _pt.raises(RuntimeError, match="UNMAPPED"):
        _rm.build_krea2_xfm_manifest(pathlib_Path(bad))


def test_krea2_te_prefix_layouts():
    """Every real Qwen3-VL TE layout must land on an engine-recognised row."""
    src = _keys_file(["language_model.embed_tokens.weight"])
    assert cu._detect_krea2_te_prefix(src) == ""          # full source: engine handles
    flat = _keys_file(["model.embed_tokens.weight"])
    assert cu._detect_krea2_te_prefix(flat) == "model."   # ComfyUI flat: strip to bare
    bare = _keys_file(["embed_tokens.weight"])
    assert cu._detect_krea2_te_prefix(bare) == ""         # already bare
    # engine row 2: the HF-NESTED ComfyUI single-file `model.language_model.*`.
    # Stripping "model." leaves `language_model.*`, which is exactly the engine's
    # full-source row — so the nested shape lands on a recognised layout too.
    nested = _keys_file(["model.language_model.embed_tokens.weight"])
    assert cu._detect_krea2_te_prefix(nested) == "model."


if __name__ == "__main__":
    import traceback
    tests = {k: v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)}
    fails = 0
    for name, fn in tests.items():
        try:
            fn()
            print(f"ok:   {name}")
        except Exception:
            fails += 1
            print(f"FAIL: {name}")
            traceback.print_exc()
    print("\n" + (f"FAILED({fails})" if fails else "PASSED"))
    sys.exit(1 if fails else 0)
