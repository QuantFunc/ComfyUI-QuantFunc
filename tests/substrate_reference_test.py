#!/usr/bin/env python3
"""DEATH RULE: every name a plugin module takes from the shared substrate must exist there.

WHY (measured, 2026-09-24): the release line's design merge deleted the lazy-detach machinery from
qf_modelpatcher.py, but qf_qwenimage21_modelpatcher.py (a file only one side of the merge had) kept calling
qfmp._qf_cancel_pending_detach at the top of _begin. Every Qwen-Image-2.1 generation would raise
AttributeError, and no CPU test runs _begin. A reference into a module is a promise the module keeps; this
checks it statically (AST, no ComfyUI or CUDA needed): every `qfmp.<name>` / `qfe.<name>` attribute and every
`from .qf_modelpatcher / .qf_engine import <name>` in the plugin package must be defined at module level in its
target. Both directions: a synthetic dangling reference must be reported, the real tree must be clean.
"""
import ast
import os
import sys
import tempfile

_PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ALIASES = {"qfmp": "qf_modelpatcher.py", "qfe": "qf_engine.py"}


def _defined(path):
    """Module-level names of `path`: defs, classes, assignments (incl. tuple targets), imports, and those bound
    inside a module-level try/if (the guarded-import pattern)."""
    out = set()
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        for n in ([node] + list(ast.walk(node)) if isinstance(node, (ast.Try, ast.If)) else [node]):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    for e in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                        if isinstance(e, ast.Name):
                            out.add(e.id)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                out.add(n.target.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                out.update((a.asname or a.name).split(".")[0] for a in n.names)
    return out


def dangling(plugin_dir):
    """Every reference in plugin_dir/*.py to a substrate name that the substrate does not define."""
    subs = {alias: _defined(os.path.join(plugin_dir, f)) for alias, f in _ALIASES.items()}
    by_module = {f[:-3]: alias for alias, f in _ALIASES.items()}
    bad = []
    for f in sorted(os.listdir(plugin_dir)):
        if not f.endswith(".py"):
            continue
        for n in ast.walk(ast.parse(open(os.path.join(plugin_dir, f), encoding="utf-8").read())):
            if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in subs
                    and n.attr not in subs[n.value.id]):
                bad.append(f"{f}:{n.lineno} {n.value.id}.{n.attr}")
            if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module in by_module:
                for a in n.names:
                    if a.name not in subs[by_module[n.module]]:
                        bad.append(f"{f}:{n.lineno} from .{n.module} import {a.name}")
    return bad


def main():
    bad = 0
    real = dangling(_PLUGIN)
    print(f"  [{'OK ' if not real else 'FAIL'}] every qfmp./qfe. reference and relative import resolves in the "
          f"substrate -> {real or 'clean'}")
    bad += bool(real)
    # RED control: the exact defect class (a family calling a deleted substrate helper) must be reported.
    with tempfile.TemporaryDirectory(prefix="qf_subref_") as d:
        for f in _ALIASES.values():
            open(os.path.join(d, f), "w").write("def kept():\n    pass\n")
        open(os.path.join(d, "qf_fam_modelpatcher.py"), "w").write(
            "from . import qf_modelpatcher as qfmp\nfrom .qf_engine import gone\n"
            "def _begin(self):\n    qfmp.kept()\n    qfmp._deleted_helper(self)\n")
        seen = dangling(d)
        ok = (any("qfmp._deleted_helper" in x for x in seen) and any("import gone" in x for x in seen)
              and not any("qfmp.kept" in x for x in seen))
        print(f"  [{'OK ' if ok else 'FAIL'}] the rule reports a deleted helper and a dangling import, not a kept "
              f"one -> {seen}")
        bad += not ok
    print("SUBSTRATE_REFERENCE:", "PASS" if not bad else f"FAIL ({bad} wrong)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
