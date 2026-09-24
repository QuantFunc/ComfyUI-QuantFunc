#!/usr/bin/env python3
"""The loaders' hidden `log_level` input (qf_log_level.add_log_level_input), without ComfyUI:

  L1  the input is declared HIDDEN, in ComfyUI's (type, options) form, and is never a required or optional
      (visible) input: users do not choose the log level; only "warning" and "info" are accepted;
  L2  running the node applies the chosen level BEFORE the node's own function runs, and the
      loader itself never receives `log_level`;
  L3  a workflow that does not set it (every existing workflow) applies warning (engine level 3);
  L4  the node's other inputs, and its return value, are untouched;
  L5  __init__.py attaches the input AFTER every node registration (a loader registered later -- the cloud-TE
      loader once was -- would silently get no input);
  L6  the node function keeps its own name and parameters (plus a keyword-only `log_level`), also when it
      takes **kwargs, so anything that inspects it sees the true signature;
  L7  qf_engine.set_log_level never loads the engine library: with no library loaded it only records the
      level, load_lib() applies it right after loading, and later requests apply at once. (Loading the
      library for the level broke every loader run in a test environment without one.)
  L8  ComfyUI does not validate hidden inputs, so the loader refuses any other value (or a non-string) with a
      ValueError naming the accepted ones, before the level is set or the loader runs;
  L9  a node that already declares hidden inputs (the quality loaders' retired quality_enhance) keeps them, and
      the node's own spec dicts are never modified.
MUTATION: make _run skip set_level -> L2/L3 go RED; forward log_level to the loader -> L2 goes RED; move the
attach loop above the cloud-TE registration -> L5 goes RED; drop the __signature__ -> L6 goes RED; make
set_log_level call load_lib(), or drop the pending-level apply in load_lib -> L7 goes RED; declare the input
optional again -> L1 goes RED; drop the value check in _run -> L8 goes RED; write log_level into the node's
own hidden dict -> L9 goes RED.

Run:  python tests/log_level_input_test.py   (pure Python; no ComfyUI, torch or engine library)
"""
import ast
import importlib.util
import inspect
import os
import sys
import tempfile
import types

_PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("qf_log_level", os.path.join(_PLUGIN, "qf_log_level.py"))
qf_log_level = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qf_log_level)

events = []


class _Loader:
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"transformer": (["a.safetensors"],)},
                "optional": {"attention_backend": (["auto", "sage"],)}}

    def load(self, transformer, attention_backend="auto"):
        """Load the model."""
        events.append(("load", transformer, attention_backend))
        return ("MODEL",)


class _KwLoader:
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"transformer": (["a.safetensors"],)}}

    def load(self, transformer, **extra):
        return ("MODEL",)


_SHARED_HIDDEN = {"quality_enhance": ("BOOLEAN", {})}   # module-level, like the quality loaders' own


class _HiddenLoader:
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"transformer": (["a.safetensors"],)}, "hidden": _SHARED_HIDDEN}

    def load(self, transformer, quality_enhance=None):
        return ("MODEL",)


def _load_qf_engine():
    """qf_engine by file path (it imports only the standard library)."""
    spec = importlib.util.spec_from_file_location("qf_engine_under_test", os.path.join(_PLUGIN, "qf_engine.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    failures = []

    def check(label, ok):
        print(f"{'PASS' if ok else 'FAIL'} {label}")
        if not ok:
            failures.append(label)

    qf_log_level.add_log_level_input(_Loader, lambda level: events.append(("set_level", level)))

    spec = _Loader.INPUT_TYPES()
    check("L1 declared hidden, as ComfyUI's (type, options) input form",
          spec.get("hidden", {}).get("log_level") == ("STRING", {}))
    check("L1 never a visible input (not required, not optional)",
          "log_level" not in spec["required"] and "log_level" not in spec.get("optional", {}))
    check("L1 only warning and info are accepted (no debug)", list(qf_log_level.LOG_LEVELS) == ["warning", "info"])
    check("L4 other inputs untouched", spec["required"] == {"transformer": (["a.safetensors"],)}
          and spec["optional"] == {"attention_backend": (["auto", "sage"],)})

    events.clear()
    out = getattr(_Loader(), _Loader.FUNCTION)(transformer="a.safetensors", attention_backend="sage",
                                               log_level="info")
    check("L2 level applied before the loader runs",
          events == [("set_level", 2), ("load", "a.safetensors", "sage")])
    check("L4 return value untouched", out == ("MODEL",))

    events.clear()
    getattr(_Loader(), _Loader.FUNCTION)(transformer="a.safetensors")
    check("L3 unset input applies warning (3)",
          events == [("set_level", 3), ("load", "a.safetensors", "auto")])

    # L8: an unknown value is refused before anything runs (ComfyUI does not validate hidden inputs).
    for bad in ("debug", "INFO", "", None, 2):
        events.clear()
        try:
            getattr(_Loader(), _Loader.FUNCTION)(transformer="a.safetensors", log_level=bad)
            refused = ""
        except Exception as e:  # noqa: BLE001 — any other exception type is a FAIL of this arm, not a crash
            refused = f"{type(e).__name__}: {e}"
        check(f"L8 log_level={bad!r} refused (ValueError naming the accepted values) before anything runs",
              refused.startswith("ValueError") and "['warning', 'info']" in refused and events == [])

    # L9: an existing hidden declaration is kept, and the node's own dicts are not modified.
    qf_log_level.add_log_level_input(_HiddenLoader, lambda level: None)
    hid = _HiddenLoader.INPUT_TYPES()["hidden"]
    check("L9 the node's own hidden inputs are kept next to log_level",
          hid == {"quality_enhance": ("BOOLEAN", {}), "log_level": ("STRING", {})})
    check("L9 the node's own spec dicts are not modified",
          _SHARED_HIDDEN == {"quality_enhance": ("BOOLEAN", {})} and "log_level" not in _SHARED_HIDDEN)

    # L5: the attach must come after every registration (Call NODE_CLASS_MAPPINGS.update / subscript assignment).
    tree = ast.parse(open(os.path.join(_PLUGIN, "__init__.py"), encoding="utf-8").read())
    regs, attach = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update" \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "NODE_CLASS_MAPPINGS":
            regs.append(node.lineno)
        if isinstance(node, ast.Assign) and any(isinstance(tg, ast.Subscript) and isinstance(tg.value, ast.Name)
                                                and tg.value.id == "NODE_CLASS_MAPPINGS" for tg in node.targets):
            regs.append(node.lineno)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_qf_add_log_level":
            attach.append(node.lineno)
    check("L5 exactly one attach call site in __init__.py", len(attach) == 1)
    check("L5 attach runs after every node registration", bool(attach) and bool(regs) and min(attach) > max(regs))

    # L6: the true signature survives the wrap (plus a keyword-only log_level, placed before any **kwargs).
    sig = inspect.signature(getattr(_Loader, _Loader.FUNCTION))
    check("L6 load keeps its name and docstring",
          _Loader.load.__name__ == "load" and _Loader.load.__doc__ == "Load the model.")
    check("L6 load keeps its parameters, plus keyword-only log_level",
          [(p.name, p.kind.name, p.default) for p in sig.parameters.values()] ==
          [("self", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
           ("transformer", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
           ("attention_backend", "POSITIONAL_OR_KEYWORD", "auto"),
           ("log_level", "KEYWORD_ONLY", "warning")])
    qf_log_level.add_log_level_input(_KwLoader, lambda level: None)
    check("L6 a **kwargs loader gets log_level before the **kwargs",
          list(inspect.signature(_KwLoader.load).parameters) == ["self", "transformer", "log_level", "extra"])

    # L7: asking for a level never loads the library; the level is applied when (or once) it is loaded.
    eng = _load_qf_engine()
    applied = []

    def _no_load():
        raise AssertionError("set_log_level tried to load the engine library")
    eng.resolve_so_path = _no_load
    try:
        eng.set_log_level(2)
        tried_to_load = False
    except AssertionError:
        tried_to_load = True
    check("L7 no library loaded: the level is recorded, nothing is loaded",
          not tried_to_load and eng._LIB is None and eng._LOG_LEVEL == 2)
    with tempfile.TemporaryDirectory() as d:          # the first engine call loads the library (simulated)
        eng.resolve_so_path = lambda: os.path.join(d, "engine-under-test")
        eng.assert_toolchain_compatible = lambda so_path: None
        eng.ctypes = types.SimpleNamespace(RTLD_GLOBAL=0, RTLD_LOCAL=0, CDLL=lambda *a, **k: object())
        eng._bind = lambda raw: types.SimpleNamespace(quantfunc_set_log_level=applied.append)
        eng.load_lib()
    check("L7 load_lib applies the recorded level right after loading", applied == [2])
    eng.set_log_level(3)
    check("L7 once loaded, a new level applies at once", applied == [2, 3])

    print(f"LOG_LEVEL_INPUT: {'PASS' if not failures else 'FAIL'} ({len(failures)} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
