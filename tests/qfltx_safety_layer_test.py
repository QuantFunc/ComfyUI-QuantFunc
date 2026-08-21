#!/usr/bin/env python3
"""EXECUTING behavioral test for QFLTXModel's engine-conditioning safety layer (the A' CR NO-GO fix).

WHY THIS EXISTS (the self-CR regression/correctness NEEDS-EVIDENCE): reject_list_completeness.py proves the
reject-LIST is COMPLETE (a static textual scan of comfy's consumed keys vs the tuple), but nothing EXECUTED
the ported methods — extra_conds / scale_latent_inpaint / _check_geometry and the Interrupt session-clearing
guard. "4/4 pass" read as if the safety layer was exercised when it was not. This file closes that: it drives
the REAL on-disk method bodies and asserts raise / no-raise on both directions.

HOW (mirrors connector_arch_derivation_test.py's rigor): the plugin uses relative imports + comfy, so a plain
import fails on this box (comfy's torchvision/torchaudio ABI). So we AST-EXTRACT each real method body from
qf_ltx_modelpatcher.py and exec it with a mock `self` + a tiny comfy stub (real torch — it imports fine here).
No comfy, no engine, no GPU. QF_LTXSAFETY_TEST_SRC lets a reviewer point this at a MUTATED copy to prove the
test is able-to-FAIL (a green test that can't go red proves nothing).

Run: python3 tests/qfltx_safety_layer_test.py     # exit 0 = pass, non-zero = fail (run_plugin_tests picks it up)
     QF_LTXSAFETY_TEST_SRC=<mutant> python3 tests/qfltx_safety_layer_test.py   # prove able-to-fail
"""
import ast
import os
import sys
import textwrap
import types

import torch  # available on this box (torchvision/torchaudio are ABI-broken, but torch itself imports)

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.environ.get("QF_LTXSAFETY_TEST_SRC") or os.path.join(_HERE, "..", "qf_ltx_modelpatcher.py")


def _src_text():
    return open(_SRC, errors="replace").read()


def _extract_method(src_text, cls, meth):
    """Return the DEDENTED source of cls.meth (a class method) so it can be exec'd as a standalone function."""
    for node in ast.parse(src_text).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name == meth:
                    return textwrap.dedent(ast.get_source_segment(src_text, m))
    raise AssertionError(f"method {cls}.{meth} not found in {_SRC}")


_WAN_SRC = os.path.join(_HERE, "..", "qf_modelpatcher.py")


def _wan_src_text():
    return open(_WAN_SRC, errors="replace").read()


def _extract_module_fn(src_text, name):
    """DEDENTED source of a MODULE-LEVEL def (the shared interrupt helper lives at module scope)."""
    for node in ast.parse(src_text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return textwrap.dedent(ast.get_source_segment(src_text, node))
    raise AssertionError(f"module fn {name} not found in {_WAN_SRC}")


def _bind_wan_helper(comfy):
    """Exec the REAL shared interrupt helper from qf_modelpatcher.py with the given comfy stub."""
    ns = {"comfy": comfy}
    exec(_extract_module_fn(_wan_src_text(), "_interrupt_poll_end_session_on_raise"), ns)  # noqa: S102
    return ns["_interrupt_poll_end_session_on_raise"]


def _extract_const(src_text, name):
    for node in ast.parse(src_text).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"const {name} not found in {_SRC}")


def _extract_class_attr(src_text, cls, name):
    for node in ast.parse(src_text).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for m in node.body:
                if isinstance(m, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == name for t in m.targets):
                    return ast.literal_eval(m.value)
    raise AssertionError(f"class attr {cls}.{name} not found")


# ── a tiny comfy stub: only what the extracted method bodies touch ────────────────────────────────────────
class _CONDRegular:
    def __init__(self, x):
        self.cond = x


class _InterruptExc(BaseException):
    """Faithful mimic of comfy.model_management.InterruptProcessingException, which subclasses BaseException
    (model_management.py:2003) — NOT Exception. This is load-bearing: it forces the guard to use
    `except BaseException`; a guard written `except Exception` would NOT catch this → the session would strand,
    which this test then flags."""


def _make_comfy(interrupt_raises=False):
    comfy = types.ModuleType("comfy")
    comfy.conds = types.SimpleNamespace(CONDRegular=_CONDRegular)

    def _throw():
        if interrupt_raises:
            raise _InterruptExc("comfy InterruptProcessingException (test)")
    comfy.model_management = types.SimpleNamespace(throw_exception_if_processing_interrupted=_throw)
    return comfy


class _MockEngine:
    """Mock QFEngineHandle: records end_session_if_open + clears current_session (mirrors qf_engine.py)."""
    def __init__(self, open_session=None):
        self.current_session = open_session
        self.unloaded = False
        self.step_count = 0
        self.sampler_step_count = 0
        self.end_calls = 0

    def end_session_if_open(self):
        self.end_calls += 1
        was_open = self.current_session is not None
        self.current_session = None
        return (was_open, True)


def _bind(src, meth, extra_globals=None):
    """Exec a class method's source into a fresh namespace; return the callable."""
    ns = {}
    if extra_globals:
        ns.update(extra_globals)
    exec(_extract_method(src, "QFLTXModel", meth), ns)  # noqa: S102 — trusted repo source
    return ns[meth], ns


def _mock_self(**attrs):
    m = types.SimpleNamespace()
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


# ── tests ─────────────────────────────────────────────────────────────────────────────────────────────────
def _t_extra_conds(src):
    comfy = _make_comfy()
    fn, _ = _bind(src, "extra_conds", {"comfy": comfy})
    keys = _extract_class_attr(src, "QFLTXModel", "_ENGINE_IGNORED_COND_KEYS")
    assert len(keys) >= 7, f"reject-list shrank unexpectedly: {keys}"
    bad = 0
    # (1) EACH reject-listed key, wired individually, must RAISE.
    for k in keys:
        eng = _MockEngine()
        me = _mock_self(_qf=eng, _ENGINE_IGNORED_COND_KEYS=keys, _max_ctx_seq=0)
        try:
            fn(me, **{k: object()})
            print(f"  [FAIL] extra_conds({k}=..) did NOT raise"); bad += 1
        except RuntimeError:
            pass
    # (2) a non-reject key (cross_attn only) must NOT raise and must emit c_crossattn.
    eng = _MockEngine()
    me = _mock_self(_qf=eng, _ENGINE_IGNORED_COND_KEYS=keys, _max_ctx_seq=0,
                    _post_connector_seq=lambda s: int(s))
    out = fn(me, cross_attn=torch.zeros(1, 3, 8))
    if "c_crossattn" not in out or not isinstance(out["c_crossattn"], _CONDRegular):
        print(f"  [FAIL] extra_conds(cross_attn) did not emit c_crossattn: {out}"); bad += 1
    # (3) documented-accepted keys (frame_rate / attention_mask / latent_image) must NOT raise.
    for k in ("frame_rate", "attention_mask", "latent_image"):
        eng = _MockEngine()
        me = _mock_self(_qf=eng, _ENGINE_IGNORED_COND_KEYS=keys, _max_ctx_seq=0,
                        _post_connector_seq=lambda s: int(s))
        try:
            fn(me, **{k: object(), "cross_attn": torch.zeros(1, 3, 8)})
        except RuntimeError:
            print(f"  [FAIL] extra_conds({k}=..) wrongly raised (accepted infra)"); bad += 1
    # (4) run-start session close is invoked (idempotent no-op when nothing open).
    eng = _MockEngine(open_session=object())
    me = _mock_self(_qf=eng, _ENGINE_IGNORED_COND_KEYS=keys, _max_ctx_seq=0,
                    _post_connector_seq=lambda s: int(s))
    fn(me, cross_attn=torch.zeros(1, 3, 8))
    if eng.end_calls != 1 or eng.current_session is not None:
        print(f"  [FAIL] extra_conds did not close a stale session (calls={eng.end_calls})"); bad += 1
    print(f"  extra_conds: {'OK' if bad == 0 else 'FAIL'} ({len(keys)} reject keys raise; cross_attn emits; "
          "frame_rate/attention_mask/latent_image accepted; stale session closed)")
    return bad


def _t_max_ctx_seq(src):
    """CONFORMANCE (_max_ctx_seq port): extra_conds must ACCUMULATE the MAX seq len across a run's cond
    groups (pos+neg BOTH call extra_conds before sampling), never last-wins — else _begin (called on the
    first group only) under-sizes the engine context maxima when a longer group reaches the SAME session as
    a separate step (the WAN gap this ports). Order-independent; a cross_attn-less call must not touch it.
    These arms test the ACCUMULATION ARITHMETIC with an IDENTITY probe (post==raw); the §6.5 POST-connector
    length semantics (the quantity accumulated) are proven against the REAL comfy connector in
    _t_post_connector_seq."""
    comfy = _make_comfy()
    fn, _ = _bind(src, "extra_conds", {"comfy": comfy})
    keys = _extract_class_attr(src, "QFLTXModel", "_ENGINE_IGNORED_COND_KEYS")
    _ident = lambda s: int(s)   # identity probe: isolates the max/order/untouched arithmetic  # noqa: E731
    bad = 0
    # (a) pos S=10 then neg S=20 → accumulator == 20 (max, not the last value).
    eng = _MockEngine(); me = _mock_self(_qf=eng, _ENGINE_IGNORED_COND_KEYS=keys, _max_ctx_seq=0,
                                         _post_connector_seq=_ident)
    fn(me, cross_attn=torch.zeros(1, 10, 8)); fn(me, cross_attn=torch.zeros(1, 20, 8))
    if me._max_ctx_seq != 20:
        print(f"  [FAIL] pos(10)+neg(20) → _max_ctx_seq={me._max_ctx_seq}, expected 20"); bad += 1
    # (b) reverse order neg S=20 then pos S=10 → still 20 (a smaller later must NOT shrink it).
    eng = _MockEngine(); me = _mock_self(_qf=eng, _ENGINE_IGNORED_COND_KEYS=keys, _max_ctx_seq=0,
                                         _post_connector_seq=_ident)
    fn(me, cross_attn=torch.zeros(1, 20, 8)); fn(me, cross_attn=torch.zeros(1, 10, 8))
    if me._max_ctx_seq != 20:
        print(f"  [FAIL] neg(20)+pos(10) → _max_ctx_seq={me._max_ctx_seq}, expected 20 (order-independent)"); bad += 1
    # (c) a cross_attn-less call must leave the accumulator UNTOUCHED (the _begin safe-fallback path).
    eng = _MockEngine(); me = _mock_self(_qf=eng, _ENGINE_IGNORED_COND_KEYS=keys, _max_ctx_seq=7,
                                         _post_connector_seq=_ident)
    fn(me)
    if me._max_ctx_seq != 7:
        print(f"  [FAIL] cross_attn-less call changed _max_ctx_seq to {me._max_ctx_seq}, expected 7"); bad += 1
    print(f"  max_ctx_seq: {'OK' if bad == 0 else 'FAIL'} (accumulates max across pos+neg, order-independent; "
          "cross_attn-less untouched)")
    return bad


def _t_post_connector_seq(src):
    """§6.5 correctness NO-GO closure — REAL-CONNECTOR arms (not mock arithmetic): comfy's
    Embeddings1DConnector does NOT preserve the seq dim (its forward tail-pads with tiled
    learnable_registers to n_reg*ceil(max(1024, S)/n_reg)), so the accumulator must carry the
    POST-connector length. Proves: (1) _post_connector_seq == the REAL module's actual output seq across
    the boundary (≤1024 / non-multiple >1024 / exact multiple), for TWO register configs — n_reg=64 kills a
    hardcoded-128 transcription; (2) the reviewer's concrete failure: a raw-1200 group accumulates 1280
    (its true post length), so a later short group's _begin ceiling covers it — on the PRE-FIX raw
    accumulation this arm reads 1200 and FAILS (able-to-fail direction); (3) registers-off connector →
    identity; (4) the per-length cache holds the measured value. ENV-GATED per the suite convention: needs
    the real comfy package (QF_NATIVE_COMFY_PATH or the box default); unimportable → a DISCLOSED [SKIP]
    (run_plugin_tests counts it; --strict fails it — run in the full comfy env to execute)."""
    comfy_path = os.environ.get("QF_NATIVE_COMFY_PATH", "/media/jonathan/Data/ComfyUI")
    try:
        if comfy_path not in sys.path:
            sys.path.insert(0, comfy_path)
        from comfy.ldm.lightricks.embeddings_connector import Embeddings1DConnector
    except Exception as e:   # noqa: BLE001 — env-gated arm, suite [SKIP] convention
        print(f"  [SKIP] post_connector_seq real-connector arms (comfy unimportable from {comfy_path}: "
              f"{type(e).__name__}: {e}; set QF_NATIVE_COMFY_PATH to a ComfyUI checkout)")
        return 0
    def build(n_reg, inner=32, heads=2, layers=1):
        return Embeddings1DConnector(
            in_channels=inner, cross_attention_dim=64, attention_head_dim=inner // heads,
            num_attention_heads=heads, num_layers=layers, num_learnable_registers=n_reg,
            split_rope=True, double_precision_rope=True, apply_gated_attention=True,
            dtype=torch.float32, device="cpu", operations=torch.nn).eval()
    pfn, _ = _bind(src, "_post_connector_seq", {"torch": torch})
    bad = 0
    # (1) probe == the REAL module's output seq, two register configs, boundary grid.
    for n_reg in (128, 64):
        conn = build(n_reg)
        me = _mock_self(_connector=conn, _post_seq_cache={})
        for S in (100, 1024, 1200, 2000):
            with torch.no_grad():
                real = int(conn(torch.zeros(1, S, 32))[0].shape[1])
            got = pfn(me, S)
            if got != real:
                print(f"  [FAIL] n_reg={n_reg} S={S}: _post_connector_seq={got} but the REAL connector "
                      f"outputs {real}"); bad += 1
    # hard numbers (kill a transcription drift): reviewer's case + the n_reg-rounding discriminator.
    me128 = _mock_self(_connector=build(128), _post_seq_cache={})
    if pfn(me128, 1200) != 1280:
        print(f"  [FAIL] n_reg=128 raw 1200 → {pfn(me128, 1200)}, expected 1280 (reviewer's case)"); bad += 1
    me64 = _mock_self(_connector=build(64), _post_seq_cache={})
    if pfn(me64, 1200) != 1216:
        print(f"  [FAIL] n_reg=64 raw 1200 → {pfn(me64, 1200)}, expected 1216 (rounds by n_reg, not 128)"); bad += 1
    # (2) INTEGRATION — the reviewer's failure scenario through the REAL extra_conds + REAL probe:
    # long neg raw=1200 accumulates its POST length 1280; a later short pos must not shrink it. PRE-FIX
    # (raw accumulation) reads 1200 here → FAIL (the able-to-fail direction of this death rule).
    comfy_stub = _make_comfy()
    efn, _ = _bind(src, "extra_conds", {"comfy": comfy_stub})
    keys = _extract_class_attr(src, "QFLTXModel", "_ENGINE_IGNORED_COND_KEYS")
    me2 = _mock_self(_qf=_MockEngine(), _ENGINE_IGNORED_COND_KEYS=keys, _max_ctx_seq=0,
                     _connector=build(128), _post_seq_cache={})
    me2._post_connector_seq = lambda s: pfn(me2, s)
    efn(me2, cross_attn=torch.zeros(1, 1200, 8)); efn(me2, cross_attn=torch.zeros(1, 20, 8))
    if me2._max_ctx_seq != 1280:
        print(f"  [FAIL] neg(raw 1200)+pos(raw 20) → _max_ctx_seq={me2._max_ctx_seq}, expected 1280 "
              "(the POST-connector length; raw accumulation under-sizes the _begin ceiling)"); bad += 1
    # (3) registers-off connector (num_learnable_registers=0) → comfy keeps S unchanged → identity probe.
    me0 = _mock_self(_connector=build(0), _post_seq_cache={})
    with torch.no_grad():
        real0 = int(build(0)(torch.zeros(1, 37, 32))[0].shape[1])
    if pfn(me0, 37) != real0 or pfn(me0, 37) != 37:
        print(f"  [FAIL] registers-off raw 37 → {pfn(me0, 37)} (real {real0}), expected identity 37"); bad += 1
    # (4) cache: the measured value is stored + a repeat returns it.
    if me128._post_seq_cache.get(1200) != 1280 or pfn(me128, 1200) != 1280:
        print(f"  [FAIL] cache miss/mismatch: {me128._post_seq_cache}"); bad += 1
    print(f"  post_connector_seq: {'OK' if bad == 0 else 'FAIL'} (probe==real module across n_reg 128/64 + "
          "boundary grid; raw-1200 accumulates 1280; registers-off identity; cached)")
    return bad


def _t_wan_interrupt(src):
    """WAN-side analog of _t_interrupt (§6.5 round-3 residual: the shared-helper change made QFWanModel
    ALSO end its session on interrupt — a WAN behavior change that previously had only the structural
    grep-arm, no functional coverage). Execs the REAL QFWanModel._apply_model + the REAL shared helper:
    an interrupt fired at the per-cond-group poll must END the open session (not strand it) and re-raise.
    `src` (the LTX source) is unused — this arm reads qf_modelpatcher.py; it lives here because this file
    owns the seam-pair safety-layer harness (_bind/_mock_self/_make_comfy)."""
    del src
    wan_src = _wan_src_text()
    comfy = _make_comfy(interrupt_raises=True)
    helper = _bind_wan_helper(comfy)
    ns = {"comfy": comfy, "torch": torch,
          "_interrupt_poll_end_session_on_raise": helper}
    exec(_extract_method(wan_src, "QFWanModel", "_apply_model"), ns)  # noqa: S102 — trusted repo source
    fn = ns["_apply_model"]
    eng = _MockEngine(open_session=object())   # session OPEN → _begin/_check_geometry are skipped
    me = _mock_self(_qf=eng, _max_batch=0, _out=None, _step_i=0)
    x = torch.zeros(1, 16, 2, 2, 2)            # [B,C,T,Hl,Wl] wan latent shape (values irrelevant)
    bad = 0
    try:
        fn(me, x, 0.5, c_crossattn=torch.zeros(1, 3, 4096), transformer_options={})
        print("  [FAIL] WAN _apply_model did NOT propagate the interrupt"); bad += 1
    except _InterruptExc:
        if eng.end_calls < 1:
            print("  [FAIL] WAN interrupt did not call end_session_if_open (session change unproven)"); bad += 1
        if eng.current_session is not None:
            print("  [FAIL] WAN interrupt left current_session non-None (stranded)"); bad += 1
    except BaseException as e:   # noqa: BLE001
        print(f"  [FAIL] WAN _apply_model raised the wrong type on interrupt: {type(e).__name__}: {e}"); bad += 1
    print(f"  wan interrupt guard: {'OK' if bad == 0 else 'FAIL'} (interrupt re-raised + session cleared "
          "via the shared helper at the REAL WAN call site)")
    return bad


def _t_shared_interrupt_helper(src):
    """§6.5 simplicity: ONE shared interrupt guard for BOTH model classes. STRUCTURAL: the raw
    comfy.model_management.throw_exception_if_processing_interrupted() call appears on exactly ONE
    non-comment line across qf_modelpatcher.py + qf_ltx_modelpatcher.py — inside the shared helper — and
    BOTH classes call the helper. FUNCTIONAL: the REAL helper source ends the open session and re-raises on
    a BaseException-derived interrupt (an `except Exception` rewrite MISSES it → end_calls==0 → FAIL), and
    is a no-op without an interrupt."""
    wan_src = _wan_src_text()
    bad = 0
    def _code_lines(text):
        return [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    raw_call = "comfy.model_management.throw_exception_if_processing_interrupted()"
    wan_raw = sum(raw_call in ln for ln in _code_lines(wan_src))
    ltx_raw = sum(raw_call in ln for ln in _code_lines(src))
    if wan_raw != 1:
        print(f"  [FAIL] qf_modelpatcher.py has {wan_raw} raw interrupt-poll call lines, expected exactly 1 "
              "(inside _interrupt_poll_end_session_on_raise)"); bad += 1
    if ltx_raw != 0:
        print(f"  [FAIL] qf_ltx_modelpatcher.py has {ltx_raw} raw interrupt-poll call lines, expected 0 "
              "(must route through the shared helper)"); bad += 1
    helper_call = "_interrupt_poll_end_session_on_raise(self._qf)"
    if sum(helper_call in ln for ln in _code_lines(wan_src)) < 1:
        print("  [FAIL] QFWanModel does not call the shared interrupt helper"); bad += 1
    if sum(helper_call in ln for ln in _code_lines(src)) < 1:
        print("  [FAIL] QFLTXModel does not call the shared interrupt helper"); bad += 1
    # FUNCTIONAL — the real helper body: interrupt → end + re-raise; no interrupt → no-op.
    helper = _bind_wan_helper(_make_comfy(interrupt_raises=True))
    eng = _MockEngine(open_session=object())
    try:
        helper(eng)
        print("  [FAIL] helper did not re-raise the interrupt"); bad += 1
    except _InterruptExc:
        if eng.end_calls != 1 or eng.current_session is not None:
            print(f"  [FAIL] helper interrupt path: end_calls={eng.end_calls} "
                  f"session={eng.current_session} (expected 1 / None)"); bad += 1
    except BaseException as e:   # noqa: BLE001
        print(f"  [FAIL] helper raised the wrong type: {type(e).__name__}"); bad += 1
    helper2 = _bind_wan_helper(_make_comfy(interrupt_raises=False))
    eng2 = _MockEngine(open_session=object())
    helper2(eng2)
    if eng2.end_calls != 0:
        print(f"  [FAIL] helper no-interrupt path called end_session ({eng2.end_calls} times)"); bad += 1
    print(f"  shared_interrupt_helper: {'OK' if bad == 0 else 'FAIL'} (one raw poll site; both classes via "
          "helper; ends+re-raises on BaseException interrupt; no-op otherwise)")
    return bad


def _t_scale_latent_inpaint(src):
    fn, _ = _bind(src, "scale_latent_inpaint")
    try:
        fn(_mock_self(), object(), object(), object())
        print("  [FAIL] scale_latent_inpaint did NOT raise"); return 1
    except RuntimeError:
        print("  scale_latent_inpaint: OK (a wired denoise mask fails loud)"); return 0


def _t_check_geometry(src):
    kT = _extract_const(src, "_LTX_TEMPORAL")
    kS = _extract_const(src, "_LTX_SPATIAL")
    fn, _ = _bind(src, "_check_geometry", {"_LTX_TEMPORAL": kT, "_LTX_SPATIAL": kS})
    # a MATCHING geometry: num_frames=9 -> F_lat=(9-1)//8+1=2; 64x64 px -> 2x2 latent (kS=32); 4 steps.
    nf, W, H, steps = 9, 64, 64, 4
    Flat = (nf - 1) // kT + 1
    Wl, Hl = W // kS, H // kS
    me = _mock_self(_num_frames=nf, _width=W, _height=H, _num_steps=steps)
    good = torch.zeros(1, 128, Flat, Hl, Wl)
    to_ok = {"sample_sigmas": list(range(steps + 1))}   # len-1 == steps
    bad = 0
    try:
        fn(me, good, to_ok)   # must NOT raise
    except RuntimeError as e:
        print(f"  [FAIL] _check_geometry raised on a MATCHING geometry: {e}"); bad += 1
    # each mismatch axis must raise INDIVIDUALLY:
    axes = {
        "frames": torch.zeros(1, 128, Flat + 1, Hl, Wl),
        "width":  torch.zeros(1, 128, Flat, Hl, Wl + 1),
        "height": torch.zeros(1, 128, Flat, Hl + 1, Wl),
    }
    for name, x in axes.items():
        try:
            fn(me, x, to_ok)
            print(f"  [FAIL] _check_geometry did NOT raise on {name} mismatch"); bad += 1
        except RuntimeError:
            pass
    # steps mismatch (sigmas len-1 != num_steps) must raise:
    try:
        fn(me, good, {"sample_sigmas": list(range(steps + 3))})
        print("  [FAIL] _check_geometry did NOT raise on steps mismatch"); bad += 1
    except RuntimeError:
        pass
    print(f"  _check_geometry: {'OK' if bad == 0 else 'FAIL'} (matching passes; frames/width/height/steps "
          "each raise individually)")
    return bad


def _t_interrupt(src):
    """The Interrupt poll must be INSIDE a guard that ends the session + re-raises (CR #2). Since the §6.5
    simplicity fix the guard mechanics live in the SHARED module-level helper in qf_modelpatcher.py
    (_interrupt_poll_end_session_on_raise) — so this arm execs the REAL helper source and injects it into
    _apply_model's namespace: it now exercises the real call site AND the real shared guard together."""
    comfy = _make_comfy(interrupt_raises=True)
    helper = _bind_wan_helper(comfy)
    fn, _ = _bind(src, "_apply_model", {"comfy": comfy, "torch": torch,
                                        "_interrupt_poll_end_session_on_raise": helper})
    eng = _MockEngine(open_session=object())   # a session is OPEN when the interrupt fires
    me = _mock_self(
        _qf=eng, _num_frames=9, _width=64, _height=64, _num_steps=4,
        _check_geometry=lambda *a, **k: None,                      # geometry not under test here
        _run_connector=lambda ca: torch.zeros(1, 3, 4096),        # skip the real connector
        _out=None, _step_i=0,
    )
    x = torch.zeros(1, 128, 2, 2, 2)
    bad = 0
    try:
        fn(me, x, 0.5, c_crossattn=torch.zeros(1, 3, 6144), transformer_options={})
        print("  [FAIL] _apply_model did NOT propagate the interrupt"); bad += 1
    except _InterruptExc:
        # the guard must have ENDED the session and re-raised (this only works if it caught BaseException —
        # a guard written `except Exception` would let _InterruptExc pass THROUGH uncaught, end_calls==0).
        if eng.end_calls < 1:
            print("  [FAIL] interrupt did not call end_session_if_open (guard likely `except Exception`, "
                  "which misses comfy's BaseException-derived InterruptProcessingException)"); bad += 1
        if eng.current_session is not None:
            print("  [FAIL] interrupt left current_session non-None (stranded)"); bad += 1
    except BaseException as e:   # noqa: BLE001
        print(f"  [FAIL] _apply_model raised the wrong type on interrupt: {type(e).__name__}: {e}"); bad += 1
    print(f"  interrupt guard: {'OK' if bad == 0 else 'FAIL'} (interrupt re-raised + session cleared, not stranded)")
    return bad


def main():
    src = _src_text()
    print(f"=== QFLTXModel safety-layer behavioral test (src={os.path.relpath(_SRC, _HERE)}) ===")
    bad = 0
    for t in (_t_extra_conds, _t_max_ctx_seq, _t_post_connector_seq, _t_scale_latent_inpaint,
              _t_check_geometry, _t_interrupt, _t_wan_interrupt, _t_shared_interrupt_helper):
        try:
            bad += t(src)
        except Exception as e:   # noqa: BLE001 — a harness error is a FAIL, not a crash-through
            print(f"  [FAIL] {t.__name__} errored: {type(e).__name__}: {e}"); bad += 1
    print("QFLTX_SAFETY_LAYER:", "PASS" if bad == 0 else f"FAIL ({bad} wrong)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
