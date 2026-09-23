#!/usr/bin/env python3
"""Death-rule for the lazy-detach window's ANCHOR (2026-09-23, QI2.1 native self-test).

Two loader outputs that differ only in a per-session knob (quality_enhance) are two QFLazyEngine wrappers over
ONE real handle. Keyed on the WRAPPER, the window armed when comfy displaced one of them could never be
cancelled by the other's sampler steps, and its expiry unloaded the SHARED engine under the other's LIVE
session — measured: a quality_enhance-OFF batch-2 run died with 'denoise begin refused: pipeline busy'
exactly 120 s after the quality_enhance-ON model was displaced.

Drives the PRODUCTION _qf_detach_anchor / _qf_arm_lazy_detach / _qf_cancel_pending_detach / QFLazyEngine,
AST-extracted from qf_modelpatcher.py (pure Python: no comfy / torch / GPU), with the window shortened:
  A1 siblings over one real handle: armed through A, a step through B cancels it -> no unload
  A2 control, different real handles: a step through B leaves A's window alone -> A's handle IS unloaded
  A3 mutation: with the anchor reverted to the wrapper itself (the pre-fix keying), A1's scenario DOES
     unload the shared handle — proves A1 can fail
  A4 an unmaterialized wrapper anchors to itself (nothing created, nothing to unload)
  A5 a FRESH sibling (a new loader output, materialized by its first begin through the real ensure() onto the cached
     shared handle) reclaims the displaced sibling's pending window — its begin's cancel ran before materialization
  A6 mutation: with ensure()'s reclaim line removed, A5's scenario DOES unload (proves A5 can fail)
  A7 arming through an UNMATERIALIZED wrapper creates no window at all (it holds nothing; a wrapper-keyed window would be
     missed by every later cancel, which anchor to the real handle once it materializes)
  A8 a cancel that lands while an expiry is already inside unload_vram WAITS for that unload (a begin must not race it)
  A9 control: the previous cancel (pending check before the lock) returns while the unload is still running (A8 can fail)
  A10 comfy's memory-pressure unload (detach unpatch_all=True, primary) frees the engine NOW and clears a pending window
  A11 a clone swap / dropped patcher (unpatch_all=False) stays lazy: no unload until the window expires
  A12 a shadow's unpatch_all=True detach only arms the window (a shadow never drives the shared engine's eviction)
  A13 mutation: with the pre-fix always-lazy detach, A10's scenario leaves the engine resident (A10 can fail)
  A14 nothing held (no engine / already unloaded): no unload, no window
  A15 the engine REFUSES the pressure unload (unload_vram -> 0, still resident): "refused", the window is armed and its
      expiry retries the unload — never a silent 'freed'
  A16 a pressure detach through an UNMATERIALIZED wrapper holds nothing: no materialization, no unload, no window
  A17 mutation: without the refusal check, A15's scenario reports "unload" and arms no retry (A15 can fail)

Run: python3 tests/lazy_detach_shared_engine_test.py   (exit 0 = pass, 1 = the contract is broken)
"""
import ast
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "qf_modelpatcher.py")
_WINDOW = 0.05
_WANT = ("_qf_detach_anchor", "_qf_arm_lazy_detach", "_qf_cancel_pending_detach", "_qf_detach_engine", "QFLazyEngine")
FAILS = []


class _Qfe:   # the one qf_engine member the timer body touches
    @staticmethod
    def _dbg_prof(msg):
        pass


_RECLAIM_LINE = "            _qf_cancel_pending_detach(self._real)\n"


_EAGER_LINE = "    if unpatch_all and not shadow:\n"


_REFUSE_LINE = "        if not getattr(eng, \"unloaded\", False):   # refused: VRAM still held -> retry at the window's expiry\n"


def _load(anchor_mutant=False, ensure_mutant=False, detach_mutant=False, refuse_mutant=False):
    src = open(_SRC, encoding="utf-8").read()
    if refuse_mutant:
        assert src.count(_REFUSE_LINE) == 1, "_qf_detach_engine's refusal check not found exactly once"
        src = src.replace(_REFUSE_LINE, "        if False:\n")
    if detach_mutant:
        assert src.count(_EAGER_LINE) == 1, "_qf_detach_engine's eager branch not found exactly once"
        src = src.replace(_EAGER_LINE, "    if False:\n")
    if ensure_mutant:
        assert src.count(_RECLAIM_LINE) == 1, "ensure()'s reclaim line not found exactly once"
        src = src.replace(_RECLAIM_LINE, "")
    ns = {"os": os, "threading": threading, "qfe": _Qfe, "json": __import__("json")}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _WANT:
            exec(compile(ast.Module([node], []), f"<{node.name}>", "exec"), ns)   # noqa: S102
    missing = [n for n in _WANT if n not in ns]
    if missing:
        print(f"FAIL production symbols not found in qf_modelpatcher.py: {missing}")
        sys.exit(1)
    ns["_QF_LAZY_DETACH_SECONDS"] = _WINDOW
    if anchor_mutant:
        ns["_qf_detach_anchor"] = lambda eng: eng
    return ns


class _Real:
    """The real-handle surface the window touches."""
    def __init__(self):
        self.unloaded = False
        self.unload_calls = 0
        self.pipeline = object()
        self.current_session = None
        self.footprint_bytes = 0

    def unload_vram(self):
        self.unload_calls += 1
        self.unloaded = True
        return 1


def _wrapper(ns, real):
    w = ns["QFLazyEngine"](factory=lambda: (real, "ckey"), footprint_bytes=0)
    w._real = real          # materialized (bypasses ensure(): no factory / LoRA reconcile needed here)
    return w


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  ({detail})"))
    if not cond:
        FAILS.append(name)


def _settle():
    time.sleep(_WINDOW * 6)


# A1 — siblings over one shared real handle
ns = _load()
shared = _Real()
a, b = _wrapper(ns, shared), _wrapper(ns, shared)
ns["_qf_arm_lazy_detach"](a)
check("A1 arming lands on the shared real handle", getattr(shared, "pending_detach", False) is True)
ns["_qf_cancel_pending_detach"](b)
_settle()
check("A1 a sibling's step cancels the window (no unload of the shared engine)",
      shared.unload_calls == 0 and not shared.unloaded, f"unload_calls={shared.unload_calls}")
check("A1 window state cleared", getattr(shared, "pending_detach", True) is False and
      getattr(shared, "_qf_detach_timer", "x") is None)

# A2 — control: two DIFFERENT real handles
ra, rb = _Real(), _Real()
ns["_qf_arm_lazy_detach"](_wrapper(ns, ra))
ns["_qf_cancel_pending_detach"](_wrapper(ns, rb))
_settle()
check("A2 an unrelated handle's step leaves the window alone (expiry unloads)", ra.unload_calls == 1,
      f"unload_calls={ra.unload_calls}")
check("A2 the unrelated handle is untouched", rb.unload_calls == 0)

# A3 — mutation: the pre-fix wrapper keying reproduces the measured defect
mns = _load(anchor_mutant=True)
m_shared = _Real()
ma, mb = _wrapper(mns, m_shared), _wrapper(mns, m_shared)
mns["_qf_arm_lazy_detach"](ma)
mns["_qf_cancel_pending_detach"](mb)
_settle()
check("A3 mutant (anchor = wrapper) unloads the shared engine under the sibling (A1 can fail)",
      m_shared.unload_calls == 1, f"unload_calls={m_shared.unload_calls}")

# A4 — unmaterialized wrapper
u = ns["QFLazyEngine"](factory=lambda: (_Real(), "ckey"), footprint_bytes=0)
check("A4 an unmaterialized wrapper anchors to itself", ns["_qf_detach_anchor"](u) is u)
ns["_qf_cancel_pending_detach"](u)   # no pending state: a no-op, never raises

# A5 — the FRESH sibling (a new loader output: `_real is None` until its first begin materializes it through the REAL
# ensure(), onto the cached shared handle). Its begin's cancel runs on the unmaterialized wrapper (anchors to itself), so the
# reclaim must come from the materialization itself — else the displaced sibling's window expires inside that begin.
fresh_shared = _Real()
old_sib = _wrapper(ns, fresh_shared)
ns["_qf_arm_lazy_detach"](old_sib)
fresh = ns["QFLazyEngine"](factory=lambda: (fresh_shared, "ckey"), footprint_bytes=0)
ns["_qf_cancel_pending_detach"](fresh)          # the begin's cancel, BEFORE materialization: touches nothing
fresh.ensure()                                   # the begin's `lib = self._qf.lib` materialization
_settle()
check("A5 materializing a fresh sibling onto the shared handle reclaims it (no unload)", fresh_shared.unload_calls == 0,
      f"unload_calls={fresh_shared.unload_calls}")

# A6 — mutation: ensure() without the reclaim reproduces the fresh-sibling gap
ens = _load(ensure_mutant=True)
m6 = _Real()
ens["_qf_arm_lazy_detach"](_wrapper(ens, m6))
f6 = ens["QFLazyEngine"](factory=lambda: (m6, "ckey"), footprint_bytes=0)
ens["_qf_cancel_pending_detach"](f6)
f6.ensure()
_settle()
check("A6 mutant (no reclaim in ensure) unloads the shared engine under the fresh sibling (A5 can fail)", m6.unload_calls == 1,
      f"unload_calls={m6.unload_calls}")

# A7 — no window on an unmaterialized wrapper
r7 = _Real()
w7 = ns["QFLazyEngine"](factory=lambda: (r7, "ckey"), footprint_bytes=0)
w7._unloaded = False                             # what partially_load() sets on a not-yet-created wrapper
ns["_qf_arm_lazy_detach"](w7)
check("A7 arming an unmaterialized wrapper creates no window", not getattr(w7, "pending_detach", False)
      and getattr(w7, "_qf_detach_timer", None) is None)
w7.ensure()
_settle()
check("A7 ...and nothing fires into the handle it later materializes onto", r7.unload_calls == 0, f"unload_calls={r7.unload_calls}")


class _SlowReal(_Real):
    def unload_vram(self):
        self.started = True
        time.sleep(0.3)
        self.finished = True
        return super().unload_vram()


def _in_flight(cancel):
    r = _SlowReal(); r.started = r.finished = False
    ns["_qf_arm_lazy_detach"](_wrapper(ns, r))
    t0 = time.time()
    while not r.started and time.time() - t0 < 2.0:
        time.sleep(0.005)
    cancel(_wrapper(ns, r))                      # a sibling's begin/step arrives while the expiry is unloading
    return r.started, r.finished


st, fin = _in_flight(ns["_qf_cancel_pending_detach"])
check("A8 a cancel during an in-flight expiry waits for the unload to finish", st and fin, f"started={st} finished={fin}")


def _old_cancel(eng):                            # the b07302f body: pending checked BEFORE taking the lock
    eng = ns["_qf_detach_anchor"](eng)
    if eng is None or not getattr(eng, "pending_detach", False):
        return
    with eng._qf_detach_lock:
        eng.pending_detach = False


st, fin = _in_flight(_old_cancel)
_settle(); time.sleep(0.3)
check("A9 control: the previous cancel returns mid-unload (A8 can fail)", st and not fin, f"started={st} finished={fin}")


def _pressure_unload(nsx):
    """A primary detach(unpatch_all=True) over a materialized wrapper whose shared handle carries a displaced sibling's window."""
    r = _Real()
    nsx["_qf_arm_lazy_detach"](_wrapper(nsx, r))
    how = nsx["_qf_detach_engine"](_wrapper(nsx, r), True, False)
    return r, how, r.unload_calls


r10, how10, now10 = _pressure_unload(ns)
check("A10 memory-pressure detach unloads immediately", how10 == "unload" and now10 == 1, f"how={how10} unload_calls={now10}")
check("A10 ...and clears the pending window", getattr(r10, "pending_detach", True) is False and
      getattr(r10, "_qf_detach_timer", "x") is None)
_settle()
check("A10 ...so nothing fires later", r10.unload_calls == 1, f"unload_calls={r10.unload_calls}")

r11 = _Real()
how11 = ns["_qf_detach_engine"](_wrapper(ns, r11), False, False)
check("A11 a clone swap stays lazy (no unload yet, window armed)", how11 == "lazy" and r11.unload_calls == 0 and
      getattr(r11, "pending_detach", False) is True, f"how={how11} unload_calls={r11.unload_calls}")
_settle()
check("A11 ...and the window's expiry runs the real unload", r11.unload_calls == 1, f"unload_calls={r11.unload_calls}")

r12 = _Real()
how12 = ns["_qf_detach_engine"](_wrapper(ns, r12), True, True)
check("A12 a shadow's pressure detach only arms the window", how12 == "lazy" and r12.unload_calls == 0,
      f"how={how12} unload_calls={r12.unload_calls}")
ns["_qf_cancel_pending_detach"](r12)

mns13 = _load(detach_mutant=True)
r13, how13, now13 = _pressure_unload(mns13)
check("A13 mutant (always lazy) leaves the engine resident under pressure (A10 can fail)", how13 == "lazy" and now13 == 0,
      f"how={how13} unload_calls={now13}")
mns13["_qf_cancel_pending_detach"](r13)

r14 = _Real(); r14.unloaded = True
check("A14 nothing held: no unload, no window", ns["_qf_detach_engine"](None, True, False) is None and
      ns["_qf_detach_engine"](r14, True, False) is None and r14.unload_calls == 0 and not getattr(r14, "pending_detach", False))


class _RefusingReal(_Real):
    """quantfunc_unload_sync refused (busy / error): unload_vram returns 0 and the engine stays resident."""
    def unload_vram(self):
        self.unload_calls += 1
        return 0


def _refused(nsx):
    r = _RefusingReal()
    how = nsx["_qf_detach_engine"](_wrapper(nsx, r), True, False)
    return r, how, r.unload_calls, getattr(r, "pending_detach", False)


r15, how15, now15, pend15 = _refused(ns)
check("A15 a refused pressure unload reports 'refused' and arms the retry window",
      how15 == "refused" and now15 == 1 and pend15 is True, f"how={how15} unload_calls={now15} pending={pend15}")
_settle()
check("A15 ...whose expiry retries the unload", r15.unload_calls == 2, f"unload_calls={r15.unload_calls}")

made16 = []
w16 = ns["QFLazyEngine"](factory=lambda: made16.append(1) or (_Real(), "ckey"), footprint_bytes=0)
w16._unloaded = False                            # what partially_load() sets on a not-yet-created wrapper
how16 = ns["_qf_detach_engine"](w16, True, False)
check("A16 an unmaterialized wrapper under pressure: nothing held, nothing created, no window",
      how16 is None and not made16 and not getattr(w16, "pending_detach", False), f"how={how16} created={len(made16)}")

mns17 = _load(refuse_mutant=True)
r17, how17, now17, pend17 = _refused(mns17)
check("A17 mutant (no refusal check) reports 'unload' with no retry window (A15 can fail)",
      how17 == "unload" and pend17 is False, f"how={how17} pending={pend17}")

print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
