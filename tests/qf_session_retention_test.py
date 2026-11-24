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
import ast
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

    def qf_xfm_session_end(self, sess):
        self.end_calls += 1
        if self.raise_on_end:
            raise OSError("stub ctypes failure")
        return self.end_rc

    def qf_last_error(self):
        return b"session end refused: session is not in an endable state (stub)"



def _handle(lib):
    h = object.__new__(qfe.QFEngineHandle)
    h.lib = lib
    h.current_session = ctypes.c_void_p(0xDEAD)
    h.xfm = ctypes.c_void_p(0xBEEF)
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

# T5: static death-rule for the needs-begin flag (silent session-REUSE hole) across shared bases and all 4 families
def _parse(rel):
    p = os.path.join(_HERE, "..", rel)
    with open(p, encoding="utf-8", errors="replace") as f:
        return ast.parse(f.read(), filename=p)


def _cls(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def _meth(cls_node, name):
    return next((n for n in cls_node.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def _has_attr_store(fn_node, attr, val):
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and n.value.value is val:
            if any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                   and t.value.id == "self" and t.attr == attr for t in n.targets):
                return True
    return False


def _has_attr_read(fn_node, attr):
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Attribute) and n.attr == attr:
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr":
            if len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and n.args[1].value == attr:
                return True
    return False


def _calls_self(fn_node, attr):
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "self" and n.func.attr == attr
        for n in ast.walk(fn_node)
    )


def _calls_super(fn_node, attr):
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr
        and isinstance(n.func.value, ast.Call) and isinstance(n.func.value.func, ast.Name)
        and n.func.value.func.id == "super"
        for n in ast.walk(fn_node)
    )


def _base_names(cls_node):
    return {ast.unparse(b) for b in cls_node.bases}


_mp_tree = _parse("qf_modelpatcher.py")
_mixin = _cls(_mp_tree, "QFSessionModelMixin")
_img = _cls(_mp_tree, "QFImageSessionModel")
_av = _cls(_mp_tree, "QFAVSessionModel")

check("T5 qf_modelpatcher._begin_extra_conds_guard arms _qf_needs_begin = True",
      _has_attr_store(_meth(_mixin, "_begin_extra_conds_guard"), "_qf_needs_begin", True))
check("T5 QFImageSessionModel.extra_conds calls _begin_extra_conds_guard",
      _calls_self(_meth(_img, "extra_conds"), "_begin_extra_conds_guard"))
check("T5 QFImageSessionModel._apply_model consumes and clears _qf_needs_begin",
      _has_attr_read(_meth(_img, "_apply_model"), "_qf_needs_begin")
      and _has_attr_store(_meth(_img, "_apply_model"), "_qf_needs_begin", False))
check("T5 QFAVSessionModel._apply_model consumes _qf_needs_begin",
      _has_attr_read(_meth(_av, "_apply_model"), "_qf_needs_begin"))

_h3_tree = _parse("qf_h3_modelpatcher.py")
_h3_cls = _cls(_h3_tree, "QFH3Model")
check("T5 qf_h3_modelpatcher inherits QFAVSessionModel, arms at run start, clears on begin",
      "qfmp.QFAVSessionModel" in _base_names(_h3_cls)
      and _meth(_h3_cls, "_apply_model") is None
      and _calls_self(_meth(_h3_cls, "extra_conds"), "_begin_extra_conds_guard")
      and _has_attr_store(_meth(_h3_cls, "_av_prepare_session"), "_qf_needs_begin", False))

_ltx_tree = _parse("qf_ltx_modelpatcher.py")
_ltx_cls = _cls(_ltx_tree, "QFLTXModel")
_ltx_av_cls = _cls(_ltx_tree, "QFLTXAVModel")
check("T5 qf_ltx_modelpatcher inherits QFSessionModelMixin/QFAVSessionModel, arms at run start, clears on begin",
      "QFSessionModelMixin" in _base_names(_ltx_cls)
      and "qfmp.QFAVSessionModel" in _base_names(_ltx_av_cls)
      and _meth(_ltx_cls, "_apply_model") is None
      and _meth(_ltx_av_cls, "_apply_model") is None
      and _calls_self(_meth(_ltx_cls, "extra_conds"), "_begin_extra_conds_guard")
      and _calls_super(_meth(_ltx_av_cls, "extra_conds"), "extra_conds")
      and _has_attr_store(_meth(_ltx_av_cls, "_av_prepare_session"), "_qf_needs_begin", False))

for fam_file, fam_cls_name in (("qf_krea2_modelpatcher.py", "QFKrea2Model"),
                               ("qf_qwenimage21_modelpatcher.py", "QFQwenImage21Model")):
    _ftree = _parse(fam_file)
    _fcls = _cls(_ftree, fam_cls_name)
    _ec = _meth(_fcls, "extra_conds")
    _am = _meth(_fcls, "_apply_model")
    check(f"T5 {fam_file} inherits QFImageSessionModel without bypassing extra_conds/_apply_model",
          "qfmp.QFImageSessionModel" in _base_names(_fcls)
          and (_ec is None or _calls_super(_ec, "extra_conds"))
          and (_am is None or _calls_super(_am, "_apply_model")))

print()
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("RESULT: all retention death-rules PASS")
