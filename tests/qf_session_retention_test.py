#!/usr/bin/env python3
"""Death-rule test for the 2026-08-24 session-retention mechanism (busy-wedge leg-2).

WHY (delta-CR R7 finding #6): every prior mock of end_session_if_open hardcoded the OLD
"clear regardless" semantics, so REVERTING the retention fix turned nothing red. This file
drives the REAL QFEngineHandle.end_session_if_open body (qf_engine.py
imports standalone — no comfy, no GPU) with a stub lib, asserting BOTH directions:

  T1  refused end  -> pointer RETAINED + (True, False)      [mutation: revert the `if ok:`
                                                             clear-guard -> T1 goes red]
  T2  ok end       -> pointer CLEARED  + (True, True)
  T3  raising end  -> pointer RETAINED (best-effort branch)
  T5  static: every family file arms `_qf_needs_begin` in extra_conds AND consumes it in the
      lazy-begin gate (the silent-session-REUSE hole fix; textual death-rule — deleting
      either half of the f5b75e2 edit goes red)

Run: python3 tests/qf_session_retention_test.py   (run_plugin_tests.py picks it up)
     QF_RETENTION_TEST_SRC=<mutant qf_engine.py> to prove able-to-fail.
"""
import ctypes
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.environ.get("QF_RETENTION_TEST_SRC") or os.path.join(_HERE, "..", "qf_engine.py")

spec = importlib.util.spec_from_file_location("qfe_under_test", _SRC)
qfe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qfe)

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class _StubLib:
    def __init__(self, end_rc=0, raise_on_end=False):
        self.end_rc = end_rc
        self.raise_on_end = raise_on_end
        self.end_calls = 0

    def quantfunc_denoise_end(self, sess):
        self.end_calls += 1
        if self.raise_on_end:
            raise OSError("stub ctypes failure")
        return self.end_rc

    def quantfunc_last_error(self):
        return b"denoise end refused: session is not in an endable state (stub)"



def _handle(lib):
    h = object.__new__(qfe.QFEngineHandle)
    h.lib = lib
    h.current_session = ctypes.c_void_p(0xDEAD)
    h.pipeline = ctypes.c_void_p(0xBEEF)
    return h


# T1: refused end -> RETAINED (the core retention invariant)
lib = _StubLib(end_rc=1)
h = _handle(lib)
was_open, ok = h.end_session_if_open()
check("T1a refused end reports (True, False)", (was_open, ok) == (True, False), f"{(was_open, ok)}")
check("T1b refused end RETAINS the pointer", h.current_session is not None,
      "pointer cleared on refusal = the pre-fix orphaning bug")
check("T1c engine end attempted exactly once", lib.end_calls == 1, f"{lib.end_calls}")

# T2: ok end -> CLEARED
lib = _StubLib(end_rc=0)
h = _handle(lib)
was_open, ok = h.end_session_if_open()
check("T2a ok end reports (True, True)", (was_open, ok) == (True, True), f"{(was_open, ok)}")
check("T2b ok end clears the pointer", h.current_session is None)

# T3: raising end -> RETAINED (best-effort branch keeps the pointer too)
lib = _StubLib(raise_on_end=True)
h = _handle(lib)
was_open, ok = h.end_session_if_open()
check("T3a raising end reports (True, False)", (was_open, ok) == (True, False), f"{(was_open, ok)}")
check("T3b raising end RETAINS the pointer", h.current_session is not None)

# T5: static death-rule for the needs-begin flag (silent session-REUSE hole)
for fam in ("qf_h3_modelpatcher.py", "qf_ltx_modelpatcher.py"):
    src = open(os.path.join(_HERE, "..", fam), encoding="utf-8", errors="replace").read()
    check(f"T5 {fam} arms _qf_needs_begin at run start", "self._qf_needs_begin = True" in src)
    check(f"T5 {fam} lazy-begin gate consumes the flag",
          'or getattr(self, "_qf_needs_begin", False)' in src)

print()
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("RESULT: all retention death-rules PASS")
