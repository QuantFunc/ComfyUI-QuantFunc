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
_WANT = ("_qf_detach_anchor", "_qf_arm_lazy_detach", "_qf_cancel_pending_detach", "QFLazyEngine")
FAILS = []


class _Qfe:   # the one qf_engine member the timer body touches
    @staticmethod
    def _dbg_prof(msg):
        pass


_RECLAIM_LINE = "            _qf_cancel_pending_detach(self._real)\n"


def _load(anchor_mutant=False, ensure_mutant=False):
    src = open(_SRC, encoding="utf-8").read()
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

print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
