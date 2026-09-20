#!/usr/bin/env python3
"""Behavioural test for the comfy VRAM-ledger interface (qf_modelpatcher.py):
  QFSessionModelMixin.memory_required(shape, cond_shapes)  — "how much MORE does one forward need"
  QFModelPatcher.model_size() / loaded_size()               — "how much do you hold"
Every branch, BOTH directions, against fake engine handles (no GPU, no comfy models):

  1. primary + engine knows its need        -> comfy-side bytes + engine need (engine asked with the LATENT shape)
  2. primary + engine UNKNOWN (0)           -> comfy-side only (never "needs nothing", never a broken sampler)
  3. shadow patcher's model                 -> comfy-side only, the engine is NOT asked
  4. lazy proxy                             -> a CACHED handle answers (peek → materialize without create); an
                                               uncreated engine is NEVER created here (comfy evicts AFTER this call)
  5. engine raises                          -> comfy-side only (swallowed)
  6. ledger: hold < weights                 -> model_size = weights, loaded_size = hold (comfy sees the deficit)
     ledger: hold >= weights                -> both = hold
     ledger: unloaded flag with residual    -> loaded_size = measured residual
     ledger: measured zero                  -> loaded_size = 0, never weights estimate
     ledger: shadow                         -> the small constant, both
  7. ONE source of truth: no family class overrides the ledger methods; QFH3Model (packed AV) == base exactly

Run:  python tests/vram_ledger_test.py   (needs comfy importable — the ComfyUI env; SKIPs (77) without it)
"""
import io
import os
import sys
import contextlib
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.dirname(_HERE)
if os.environ.get("COMFY_ROOT"):
    sys.path.insert(0, os.environ["COMFY_ROOT"])


def _skip(reason):
    print(f"VRAM_LEDGER: SKIP — {reason}")
    sys.exit(77)


try:
    import comfy.model_management  # noqa: F401,E402
    import comfy.model_patcher  # noqa: F401,E402
except Exception as e:  # noqa: BLE001
    _skip(f"comfy not importable here ({e!r}); run inside the ComfyUI env")

_pkg = os.path.basename(_PLUGIN)
sys.path.insert(0, os.path.dirname(_PLUGIN))
try:
    qfmp = __import__(f"{_pkg}.qf_modelpatcher", fromlist=["QFSessionModelMixin", "QFModelPatcher"])
except Exception as e:  # noqa: BLE001
    _skip(f"plugin package not importable as {_pkg} ({e!r})")

Mixin, Patcher = qfmp.QFSessionModelMixin, qfmp.QFModelPatcher
MB = 1 << 20
SHAPE = [2, 16, 31, 48, 50]          # comfy's [B*2, C, T, H, W] at estimate time
COND = {"c_crossattn": [(2, 616, 5120)]}

fails = 0
def check(cond, msg):
    global fails
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        fails += 1


class _Model(Mixin):
    pass


def _engine(need_mb=None, hold_mb=0, footprint_mb=13694, unloaded=False, raise_need=False):
    def need(shape):
        if raise_need:
            raise RuntimeError("boom")
        need.asked.append(list(shape))
        return (need_mb or 0) * MB
    need.asked = []
    return SimpleNamespace(vram_need_bytes=need, resident_vram_bytes=lambda: hold_mb * MB,
                           footprint_bytes=footprint_mb * MB, unloaded=unloaded, current_session=None)


def _quiet(fn, *a, **k):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        r = fn(*a, **k)
    return r, out.getvalue()


class _LedgerOnlyPatcher(Patcher):
    def __del__(self):   # comfy's ModelPatcher.__del__ unpins/detaches ctor state this bare instance never had
        pass


def _patcher(model):
    p = _LedgerOnlyPatcher.__new__(_LedgerOnlyPatcher)   # ledger methods only read self.model — no comfy ctor needed
    p.model = model
    return p


m = _Model(); m._qf = _engine(need_mb=12000)
side = m._qf_comfy_side_bytes(SHAPE, COND)
check(side > 64 * MB, "comfy-side bytes are geometry-proportional (> the 64 MB floor for this shape)")

# ---- arm 1: primary + engine knows ---------------------------------------------------------------------------
r, log = _quiet(m.memory_required, SHAPE, cond_shapes=COND)
check(r == side + 12000 * MB, "arm1: memory_required = comfy-side + engine need")
check(m._qf.vram_need_bytes.asked == [SHAPE], "arm1: the engine was asked with the LATENT shape as given")
check("engine need 12000 MB" in log, "arm1: ledger line names the engine need")
r2, log2 = _quiet(m.memory_required, SHAPE, cond_shapes=COND)
check(r2 == r and log2 == "", "arm1: same question again → same answer, no second log line (change-only)")

# ---- arm 1b: a PACKED AV latent [B,1,N] (H3 / LTX-AV) is unpacked to the VIDEO stream's engine geometry ------------
PACKED = [2, 1, 24 * 31 * 48 * 50 + 32 * 2 * 207]          # comfy.utils.pack_latents(video, audio) → [B,1,N]
m = _Model(); m._qf = _engine(need_mb=12000); m.latent_shapes = [(1, 24, 31, 48, 50), (1, 32, 2, 207)]
_quiet(m.memory_required, PACKED, cond_shapes=COND)
check(m._qf.vram_need_bytes.asked == [[2, 24, 31, 48, 50]],
      "arm1b: packed [B,1,N] + comfy latent_shapes → engine asked with [B] + video stream dims")
m = _Model(); m._qf = _engine(need_mb=12000); m.latent_shapes = [(1, 24, 31, 48, 51), (1, 32, 2, 207)]  # stale/other geometry
_quiet(m.memory_required, PACKED, cond_shapes=COND)
check(m._qf.vram_need_bytes.asked == [PACKED], "arm1b: latent_shapes whose numel ≠ N (stale geometry) → as given")
m = _Model(); m._qf = _engine(need_mb=12000)                                                  # first run: unset
_quiet(m.memory_required, PACKED, cond_shapes=COND)
check(m._qf.vram_need_bytes.asked == [PACKED], "arm1b: no latent_shapes yet (first run) → as given")

# ---- arm 2: engine UNKNOWN -----------------------------------------------------------------------------------
m = _Model(); m._qf = _engine(need_mb=0)
r, log = _quiet(m.memory_required, SHAPE, cond_shapes=COND)
check(r == side and "nothing measured yet" in log, "arm2: engine 0 → comfy-side only, logged as covered-or-unmeasured")

# ---- arm 3: shadow model → engine not asked --------------------------------------------------------------------
m = _Model(); m._qf = _engine(need_mb=12000); m._qf_shadow = True
r, _ = _quiet(m.memory_required, SHAPE, cond_shapes=COND)
check(r == side and m._qf.vram_need_bytes.asked == [], "arm3: shadow → comfy-side only, engine never asked")

# ---- arm 4: lazy proxy — a CACHED handle is used; an uncreated engine is NEVER created here ---------------------
# (self-CR P-2: comfy calls memory_required BEFORE its own eviction pass, so a create here would run ahead of the
#  room being made. The proxy materializes only over a cache HIT — QFLazyEngine.ensure_if_cached / factory.peek.)
real = _engine(need_mb=9000)
proxy = SimpleNamespace(vram_need_bytes=lambda s: 0, footprint_bytes=1, ensure_if_cached=lambda: real, current_session=None)
m = _Model(); m._qf = proxy
r, _ = _quiet(m.memory_required, SHAPE, cond_shapes=COND)
check(r == side + 9000 * MB and real.vram_need_bytes.asked == [SHAPE], "arm4a: cache HIT → the REAL handle's need is used")
created = []
proxy = SimpleNamespace(vram_need_bytes=lambda s: 0, footprint_bytes=1, ensure_if_cached=lambda: None,
                        ensure=lambda: created.append(1), current_session=None)
m = _Model(); m._qf = proxy
r, log = _quiet(m.memory_required, SHAPE, cond_shapes=COND)
check(r == side and created == [] and "nothing measured yet" in log,
      "arm4b: cache MISS → comfy-side only, ensure() NOT called (no create ahead of comfy's eviction)")
# the real QFLazyEngine: peek None → stays unmaterialized; peek hit → the factory (a cache hit) materializes it
Lazy = qfmp.QFLazyEngine
calls = []
def _fac():
    calls.append(1)
    return SimpleNamespace(pipeline=1, footprint_bytes=14380 * MB, step_count=0, sampler_step_count=0,
                           unloaded=False, current_session=None, vram_need_bytes=lambda s: 7 * MB), "ckey"
lz = Lazy(_fac, 13694 * MB)                       # no factory.peek → a ledger read can never create
check(lz.ensure_if_cached() is None and calls == [] and not lz.materialized and lz.vram_need_bytes(SHAPE) == 0,
      "arm4c: QFLazyEngine without a peek → ensure_if_cached None, factory NOT called, need 0")
_fac.peek = lambda: None
check(lz.ensure_if_cached() is None and calls == [], "arm4d: peek MISS → factory NOT called")
_fac.peek = lambda: object()
check(lz.ensure_if_cached() is not None and calls == [1] and lz.materialized and lz.vram_need_bytes(SHAPE) == 7 * MB,
      "arm4e: peek HIT → materialized through the (cache-hitting) factory once; need now comes from the real handle")
check(lz.ensure_if_cached() is not None and calls == [1], "arm4f: already materialized → no second factory call")

# ---- arm 5: engine raises → comfy-side ------------------------------------------------------------------------
m = _Model(); m._qf = _engine(raise_need=True)
r, _ = _quiet(m.memory_required, SHAPE, cond_shapes=COND)
check(r == side, "arm5: an engine exception is swallowed → comfy-side only")

# ---- arm 6: the patcher ledger --------------------------------------------------------------------------------
m = _Model(); m._qf = _engine(hold_mb=10073, footprint_mb=14380); p = _patcher(m)
check(p.model_size() == 14380 * MB and p.loaded_size() == 10073 * MB,
      "arm6a: hold < weights → model_size = weights, loaded_size = hold (deficit visible to comfy)")
m._qf = _engine(hold_mb=23000, footprint_mb=14380)
check(p.model_size() == 23000 * MB and p.loaded_size() == 23000 * MB, "arm6b: hold ≥ weights → both = hold")
m._qf = _engine(hold_mb=23000, footprint_mb=14380, unloaded=True)
check(p.loaded_size() == 23000 * MB and p.model_size() == 23000 * MB,
      "arm6c: unloaded flag cannot hide measured residual residency")
m._qf = _engine(hold_mb=0, footprint_mb=14380)
check(p.loaded_size() == 0 and p.model_size() == 14380 * MB,
      "arm6d: a measured zero is zero, not the weights estimate")
m._qf_shadow = True
check(p.loaded_size() == Patcher._QF_SHADOW_LEDGER_BYTES and p.model_size() == Patcher._QF_SHADOW_LEDGER_BYTES,
      "arm6e: shadow → the small constant, both")
m._qf = _engine(hold_mb=0, footprint_mb=14380, unloaded=True)
check(p.loaded_size() == 0 and p.model_size() == Patcher._QF_SHADOW_LEDGER_BYTES,
      "arm6f: shadow over an UNLOADED engine → loaded_size 0 (the pre-change order), model_size the constant")

# ---- arm 7: ONE source of truth — no family overrides the ledger interface; H3 (packed AV) gets exactly base ---------
# (self-CR P-1 on 1d51182 found QFH3Model.memory_required stacking a 2026-08-24 heuristic ON TOP of the base's real
#  engine need → double-count. A family that needs different accounting must change the base, not shadow it.)
_LEDGER_METHODS = ("memory_required", "model_size", "loaded_size")
_families = {}
for _mod in ("qf_h3_modelpatcher", "qf_krea2_modelpatcher", "qf_wan_modelpatcher", "qf_ltx_modelpatcher"):
    try:
        _families[_mod] = __import__(f"{_pkg}.{_mod}", fromlist=["*"])
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP arm7 import {_mod}: {e!r}")
_overriders = []
for _mod, _m in _families.items():
    for _name in dir(_m):
        _cls = getattr(_m, _name)
        if isinstance(_cls, type) and issubclass(_cls, Mixin) and _cls is not Mixin:
            for _meth in _LEDGER_METHODS:
                if _meth in vars(_cls):
                    _overriders.append(f"{_mod}.{_name}.{_meth}")
check(_families and not _overriders, "arm7: no family class overrides memory_required/model_size/loaded_size (%s)"
      % (_overriders or "clean"))
_h3 = _families.get("qf_h3_modelpatcher")
if _h3 is not None:
    H3 = _h3.QFH3Model
    h = H3.__new__(H3)                    # memory_required reads only mixin attrs — no comfy ctor needed
    h._qf = _engine(need_mb=12000); h.latent_shapes = [(1, 24, 31, 48, 50), (1, 32, 2, 207)]
    side_p = h._qf_comfy_side_bytes(PACKED, COND)
    r, _ = _quiet(h.memory_required, PACKED, cond_shapes=COND)
    check(r == side_p + 12000 * MB and h._qf.vram_need_bytes.asked == [[2, 24, 31, 48, 50]],
          "arm7: QFH3Model.memory_required == comfy-side + engine need, engine asked with the video stream dims (no double-count)")

print("VRAM_LEDGER: %s (%d failing checks)" % ("PASS" if fails == 0 else "FAIL", fails))
sys.exit(0 if fails == 0 else 1)
