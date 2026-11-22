#!/usr/bin/env python3
"""PC Engine attention backend contract across all 4 consumer SMs (SM75/86/89/120).

QuantFunc PC Engine replaces crashing external backends with unified built-in attention
choices ["auto", "int8", "i4xi8", "int4", "fp16"] (default "auto" on every consumer SM and OS,
including SM86 Windows) plus compatibility aliases for saved workflows:
  legacy int8 aliases -> "int8"
  "int4_pv8" / "i4xi8" -> "i4xi8"
  "fp16_native" / "native" / "flash" -> "fp16"
  "auto" / None / "" -> "auto"
Unknown backend names are refused loudly (ValueError).
"""
import ast
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.dirname(_HERE)
_INIT = os.path.join(_PLUGIN, "__init__.py")


def _source():
    with open(_INIT, encoding="utf-8") as f:
        return f.read()


def _extract_fn_src(src, name):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def _extract_assign(src, name):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def _policy_namespace(src):
    ns = {}
    for const in ("_ATTN_BACKENDS", "_ATTN_BACKEND_ALIASES"):
        ns[const] = _extract_assign(src, const)
        if ns[const] is None:
            raise AssertionError(f"{const} not found in __init__.py")
    for fn in ("_attn_backend_default", "_attn_backend_choices", "_attn_backend_to_engine", "_validate_attn_backend"):
        code = _extract_fn_src(src, fn)
        if not code:
            raise AssertionError(f"{fn} not found in __init__.py")
        exec(compile(code, _INIT, "exec"), ns)  # noqa: S102
    return ns


def _load_plugin():
    comfy_root = os.environ.get("COMFY_ROOT")
    if not comfy_root:
        guess = os.path.dirname(os.path.dirname(_PLUGIN))
        comfy_root = guess if os.path.isfile(os.path.join(guess, "folder_paths.py")) else None
    if not comfy_root:
        return None, "no COMFY_ROOT"
    for p in (comfy_root, _PLUGIN):
        if p not in sys.path:
            sys.path.insert(0, p)
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import folder_paths  # noqa: F401
    spec = importlib.util.spec_from_file_location("qf_plugin_attn_under_test", _INIT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qf_plugin_attn_under_test"] = mod
    spec.loader.exec_module(mod)
    if not getattr(mod, "_IMPORT_OK", False):
        return None, "_IMPORT_OK is False"
    return mod, None


def main():
    failures, passed = [], [0]

    def check(label, ok):
        print(f"{'PASS' if ok else 'FAIL'} {label}")
        if ok:
            passed[0] += 1
        else:
            failures.append(label)

    src = _source()
    ns = _policy_namespace(src)
    want_choices = ["auto", "int8", "i4xi8", "int4", "fp16"]
    check(f"canonical choices are {want_choices}",
          ns["_ATTN_BACKENDS"] == want_choices)

    for cc in ((7, 5), (8, 6), (8, 9), (12, 0), None):
        for plat in ("win32", "linux"):
            check(f"default auto on cc={cc} plat={plat}", ns["_attn_backend_default"](cc, plat)[0] == "auto")
            check(f"choices unified on cc={cc}", ns["_attn_backend_choices"](cc) == want_choices)

    to_engine = ns["_attn_backend_to_engine"]
    expected = {
        None: "auto", "": "auto", "auto": "auto",
        "int8": "int8", "i4xi8": "i4xi8", "int4_pv8": "i4xi8", "int4": "int4", "fp16": "fp16",
        "I4XI8": "i4xi8", "INT4_PV8": "i4xi8", "  i4xi8  ": "i4xi8", "  INT4_PV8\t": "i4xi8",
        "qfa": "int8", "sage": "int8",
        "fp16_native": "fp16", "native": "fp16", "flash": "fp16",
    }
    for raw, want in expected.items():
        check(f"alias {raw!r} -> {want!r}", to_engine(raw) == want and ns["_validate_attn_backend"](raw) is True)

    for bad in ("xformers", "triton", "sdpa", "bogus", "   ", 123, False):
        try:
            to_engine(bad)
            ok = False
        except ValueError:
            ok = True
        check(f"invalid backend {bad!r} refused loudly", ok and isinstance(ns["_validate_attn_backend"](bad), str))

    # Tooltip carries no banned terms
    import _banned_terms as bt
    input_src = _extract_fn_src(src, "_attn_backend_input") or ""
    strings = "".join(v.value for v in ast.walk(ast.parse(input_src))
                      if isinstance(v, ast.Constant) and isinstance(v.value, str))
    hits = bt.desc_hits(strings) + bt.term_hits(strings) + bt.mode_id_hits(strings)
    check(f"tooltip clean of banned terms (hits={hits})", not hits)

    # All 4 loaders call _attn_backend_input() and validate attention_backend
    tree = ast.parse(src)
    loaders = ("QuantFuncLTXLoader", "QuantFuncKrea2Loader", "QuantFuncQwenImage21Loader", "QuantFuncH3Loader")
    per = {c: 0 for c in loaders}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name in per]:
        per[cls.name] = sum(1 for n in ast.walk(cls) if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Name) and n.func.id == "_attn_backend_input")
    check(f"all 4 loaders use _attn_backend_input() ({per})", all(v == 1 for v in per.values()))

    plugin, reason = _load_plugin()
    if plugin is None:
        print(f"[SKIP] wiring: {reason}")
    else:
        for name in loaders:
            cls = plugin.NODE_CLASS_MAPPINGS[name]
            choices, opts = cls.INPUT_TYPES()["optional"]["attention_backend"]
            check(f"{name} offers {want_choices} default='auto'",
                  choices == want_choices and opts["default"] == "auto")
            check(f"{name}.VALIDATE_INPUTS accepts i4xi8/int4_pv8 (cased/padded) + legacy aliases and refuses invalid",
                  cls.VALIDATE_INPUTS("i4xi8") is True and cls.VALIDATE_INPUTS("int4_pv8") is True
                  and cls.VALIDATE_INPUTS("I4XI8") is True and cls.VALIDATE_INPUTS("INT4_PV8") is True
                  and cls.VALIDATE_INPUTS("  i4xi8  ") is True and cls.VALIDATE_INPUTS("  INT4_PV8\t") is True
                  and cls.VALIDATE_INPUTS("qfa") is True and cls.VALIDATE_INPUTS("sage") is True
                  and cls.VALIDATE_INPUTS("flash") is True and cls.VALIDATE_INPUTS("fp16_native") is True
                  and isinstance(cls.VALIDATE_INPUTS("bogus"), str)
                  and isinstance(cls.VALIDATE_INPUTS("   "), str)
                  and isinstance(cls.VALIDATE_INPUTS(123), str)
                  and isinstance(cls.VALIDATE_INPUTS(False), str))

    print(f"SM86_ATTENTION_DEFAULT: {'PASS' if not failures else 'FAIL'} ({passed[0]} pass, {len(failures)} fail)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
