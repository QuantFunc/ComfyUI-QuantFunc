#!/usr/bin/env python3
"""No written-down model or graph facts in the plugin (user 2026-09-28 「不能有任何写死逻辑」).

A value that describes the model (a latent scale, a channel count, a frame grid, a frame rate, a sampling shift) comes
from where ComfyUI keeps it: the graph's tensors, the model's comfy config (latent_format, model_sampling), ComfyUI's own
stock nodes, or the checkpoint. It is never restated in the plugin.

  R1 every module- or class-level NAMED numeric constant in the plugin's production .py files is listed in ALLOWED
     with the reason it is not a model or graph fact. "Numeric" includes a number wrapped in int()/float(), a
     tuple/list/set of numbers, arithmetic on numbers, and a dict with a numeric key or value. A new one fails:
     derive it, or add it with its reason. An ALLOWED entry that no longer exists fails too, so the list stays true.
  R2 in the family modules (qf_*_modelpatcher.py, the shared base included), inside functions: no arithmetic by a
     written-down number >= 3 (a scale factor, a channel count, a grid step), no local variable or attribute
     assigned such a number (naming it does not make it a model value), no default argument holding one, and no
     comparison against one. Exempt: the C API's fixed array lengths, (ctypes.c_int * N), and tensor-RANK checks
     (x.ndim, x.dim(), getattr(x, "ndim", ...), len(x.shape), len(<...dims/shape name>)), which are the ABI's and
     the latent's structure, not model values.

WHAT IT CANNOT SEE (by design; the value-level tests are the guard there): a number produced by any other call, parsed
from a string, read through getattr defaults, imported from another module, or computed in a helper outside the family
modules; and the function bodies of __init__.py (loader widget ranges are UI policy) and qf_engine.py (binary-format
offsets). The value-level arms that no written-down form can pass: qfltx_safety_layer_test _derive_geometry arm 1b and
h3_partial_denoise_contract_test P10 (a latent_format ComfyUI never uses; the output must follow it).

MUTATION (each goes RED): add `_X_SCALE = 16`, `_X = int(16)` or `_X = {"s": 16}` at module level -> R1; write
`w = x.shape[-1] * 16`, `s = 16` then `w * s`, `def f(x, s=16)` or `if x.shape[-1] == 16:` in a family module -> R2;
delete an ALLOWED constant from the code but not from the list -> R1 (stale).

Run:  python tests/no_hardcoded_values_test.py
"""
import ast
import glob
import os
import sys

PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ABI = "C API value, mirrored from include/quantfunc.h (the engine's ABI, not a model fact)"
ALLOWED = {
    ("qf_api_key.py", "_REF_HEX"): "length of the reference token this process issues (its own format)",
    ("qf_api_key.py", "_MAX_FIELD"): "upper bound on the POSTed field (input validation at a trust boundary)",
    **{("qf_engine.py", n): _ABI for n in (
        "QUANTFUNC_OK", "QUANTFUNC_RESOURCE_ABI_VERSION", "QUANTFUNC_RESOURCE_CREATION_ABI_VERSION",
        "QUANTFUNC_RESOURCE_RESIDENCY_ABI_VERSION", "QUANTFUNC_RESOURCE_DOMAIN_ABI_VERSION",
        "QUANTFUNC_RESOURCE_CAPACITY_ABI_VERSION", "QUANTFUNC_RESOURCE_LIFECYCLE_ABI_VERSION",
        "QUANTFUNC_ERROR_INVALID_ARG", "QUANTFUNC_ERROR_UNSUPPORTED", "QUANTFUNC_RESOURCE_READY",
        "QUANTFUNC_RESOURCE_BUSY", "QUANTFUNC_RESOURCE_UNKNOWN", "QUANTFUNC_RESOURCE_CLOSED",
        "QUANTFUNC_RESOURCE_CAPACITY_UNSUPPORTED", "QUANTFUNC_RESOURCE_CAP_QUERY", "QUANTFUNC_RESOURCE_CAP_RELEASE_ALL",
        "QUANTFUNC_RESOURCE_PHASE_SHARED", "QUANTFUNC_RESOURCE_PHASE_PREPARED", "QUANTFUNC_RESOURCE_PHASE_ATTACHED",
        "QUANTFUNC_RESOURCE_PHASE_CLOSED", "_LOG_INFO")},
    **{("qf_engine.py", n): "ELF / dlinfo format constant (the library-file format, not a model fact)"
       for n in ("_SHT_DYNAMIC", "_DT_NULL", "_DT_NEEDED", "_RTLD_DI_LINKMAP")},
    ("qf_engine.py", "_ENGINE_PLATFORMS"): "the published release layout (CUDA major -> engine library file name)",
    ("qf_engine.py", "_ENGINE_SETS_SCHEMA"): "schema version of the published engine sets.json",
    ("qf_engine.py", "_ENGINE_VERIFY_SCHEMA_MAX"): "newest verify.json schema this plugin reads",
    ("qf_engine.py", "_ENGINE_HTTP_TIMEOUT_S"): "network timeout of the engine download",
    ("qf_engine.py", "_ENGINE_DEVICE"): "placeholder only: start_engine_install sets it to ComfyUI's device",
    ("qf_h3_modelpatcher.py", "_H3_DENOISED_SIGMA"): "float tolerance: a stage whose last sigma is above it stops early",
    ("qf_h3_modelpatcher.py", "_H3_FULL_START"): "float tolerance on model_sampling.sigma_max for a full-range start",
    ("qf_log_level.py", "LOG_LEVELS"): "the engine's log scale for quantfunc_set_log_level (C API), not a model fact",
    ("qf_modelpatcher.py", "_KNO_CTX_KEY"): "sentinel context key (no conditioning), not a model value",
    ("qf_modelpatcher.py", "_NATIVE_BUSY_DEADLINE_S"): "retry policy for a transiently busy engine read",
    ("qf_modelpatcher.py", "_NATIVE_BUSY_FIRST_BACKOFF_S"): "retry policy for a transiently busy engine read",
    ("qf_modelpatcher.py", "_NATIVE_BUSY_MAX_BACKOFF_S"): "retry policy for a transiently busy engine read",
    ("qf_modelpatcher.py", "_MS_PER_S"): "unit conversion (seconds to milliseconds) for message text",
    ("qf_modelpatcher.py", "_step_cache"): "dial default = the loader input's own default (off)",
    ("qf_modelpatcher.py", "_block_cache"): "dial default = the loader input's own default (off)",
    ("qf_modelpatcher.py", "_sparse"): "dial default: full attention (off)",
    ("qf_modelpatcher.py", "_QF_COMFY_SIDE_BASE_BYTES"): "estimate of ComfyUI's own sampler buffers (vram_ledger_test pins it)",
    ("qf_modelpatcher.py", "_QF_COMFY_SIDE_LATENT_COPIES"): "estimate of ComfyUI's own sampler buffers (vram_ledger_test pins it)",
    ("qf_modelpatcher.py", "_QF_COMFY_SIDE_COND_COPIES"): "estimate of ComfyUI's own sampler buffers (vram_ledger_test pins it)",
    ("qf_modelpatcher.py", "_QF_COMFY_SIDE_ITEMSIZE"): "estimate of ComfyUI's own sampler buffers (vram_ledger_test pins it)",
    **{("qf_ltx_modelpatcher.py", n): "refusal only: the LTX-2 audio lane's input layout as comfy's LTXAVModel writes "
       "it; the engine checks the packed width but not the split, data shapes come from the latent"
       for n in ("_LTXAV_AUDIO_CH", "_LTXAV_AUDIO_MEL")},
}


def _num(node):
    """The number a node writes down: a literal, its negation, or a literal wrapped in int()/float()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _num(node.operand)
        return None if v is None else -v
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("int", "float")
            and len(node.args) == 1 and not node.keywords):
        return _num(node.args[0])
    return None


def _numeric(node):
    """True for a number, a tuple/list/set of numbers, arithmetic on numbers, or a dict with a numeric key or value."""
    if _num(node) is not None:
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return bool(node.elts) and all(_numeric(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return any(k is not None and _numeric(k) for k in node.keys) or any(_numeric(v) for v in node.values)
    if isinstance(node, ast.BinOp):
        return _numeric(node.left) and _numeric(node.right)
    return False


def production_files():
    return sorted(os.path.basename(f) for f in glob.glob(os.path.join(PLUGIN, "*.py")))


def named_constants():
    found = set()
    for rel in production_files():
        tree = ast.parse(open(os.path.join(PLUGIN, rel), encoding="utf-8").read())
        bodies = [tree.body] + [c.body for c in ast.walk(tree) if isinstance(c, ast.ClassDef)]
        for body in bodies:
            for n in body:
                if isinstance(n, ast.Assign) and _numeric(n.value):
                    for t in n.targets:
                        name = getattr(t, "id", None) or getattr(t, "attr", None)
                        if name:
                            found.add((rel, name, n.lineno))
                elif isinstance(n, ast.AnnAssign) and n.value is not None and _numeric(n.value):
                    name = getattr(n.target, "id", None)
                    if name:
                        found.add((rel, name, n.lineno))
    return found


def _is_ctypes_array_length(binop):
    side = binop.left if _num(binop.right) is not None else binop.right
    return isinstance(side, ast.Attribute) and isinstance(side.value, ast.Name) and side.value.id == "ctypes"


def _is_rank(node):
    """A tensor-rank expression: x.ndim, x.dim(), getattr(x, "ndim", ...), len(x.shape), len(<...dims/shape name>)."""
    if isinstance(node, ast.Attribute) and node.attr == "ndim":
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "dim" and not node.args:
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) >= 1:
        if node.func.id == "getattr" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
                and node.args[1].value == "ndim":
            return True
        if node.func.id == "len":
            a = node.args[0]
            return (isinstance(a, ast.Attribute) and a.attr == "shape") or \
                (isinstance(a, ast.Name) and ("dims" in a.id or "shape" in a.id))
    return False


def _big(node):
    v = _num(node)
    return v is not None and abs(v) >= 3


def written_down_factors():
    hits = []
    for rel in production_files():
        if not (rel.startswith("qf_") and rel.endswith("_modelpatcher.py")):
            continue
        tree = ast.parse(open(os.path.join(PLUGIN, rel), encoding="utf-8").read())
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            where = f"{rel}:{{}} {fn.name}()"
            for d in [d for d in fn.args.defaults + fn.args.kw_defaults if d is not None and _big(d)]:
                hits.append(f"{where.format(fn.lineno)}: default argument {_num(d)}")
            for n in ast.walk(fn):
                if isinstance(n, ast.BinOp) and isinstance(
                        n.op, (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Add, ast.Sub)):
                    for side in (n.left, n.right):
                        if _big(side) and not _is_ctypes_array_length(n):
                            hits.append(f"{where.format(n.lineno)}: {type(n.op).__name__} {_num(side)}")
                elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and n.value is not None and _big(n.value):
                    hits.append(f"{where.format(n.lineno)}: assigns {_num(n.value)}")
                elif isinstance(n, ast.Compare):
                    ops = [n.left] + n.comparators
                    for i, o in enumerate(ops):
                        if _big(o) and not any(_is_rank(x) for j, x in enumerate(ops) if j != i):
                            hits.append(f"{where.format(n.lineno)}: compares with {_num(o)}")
    return hits


def main():
    bad = 0
    found = named_constants()
    unlisted = sorted((rel, name, line) for rel, name, line in found if (rel, name) not in ALLOWED)
    stale = sorted(k for k in ALLOWED if k not in {(rel, name) for rel, name, _ in found})
    for rel, name, line in unlisted:
        print(f"  FAIL R1 {rel}:{line} {name}: a written-down number. Derive it from the graph, the model's comfy config "
              f"or the checkpoint, or list it in ALLOWED with the reason it is not a model fact")
    for rel, name in stale:
        print(f"  FAIL R1 {rel} {name}: listed in ALLOWED but no longer in the code - remove the entry")
    bad += len(unlisted) + len(stale)
    print(f"  {'PASS' if not unlisted and not stale else 'FAIL'} R1 {len(found)} named numeric constants, each must be "
          f"listed with a reason ({sum(1 for v in ALLOWED.values() if v.startswith('PENDING'))} PENDING)")
    factors = written_down_factors()
    for h in factors:
        print(f"  FAIL R2 {h}: a written-down number in a family module; read it from the model or the tensors")
    bad += len(factors)
    print(f"  {'PASS' if not factors else 'FAIL'} R2 no written-down number in the family modules' functions")
    # the rules can fail: every planted form is caught, and a tensor-rank check is not
    r1_forms = ("_X = 16", "_X = int(16)", "_X = float(24)", "_X = {'s': 16}", "_X = (32, 8)")
    planted_r1 = all(_numeric(ast.parse(f).body[0].value) for f in r1_forms)
    rank_ok = _is_rank(ast.parse("x.ndim").body[0].value) and not _is_rank(ast.parse("x.shape[-1]").body[0].value)
    ok = planted_r1 and rank_ok
    print(f"  {'PASS' if ok else 'FAIL'} the rules see each planted constant form and tell a rank check from a size")
    bad += 0 if ok else 1
    print(f"NO_HARDCODED_VALUES: {'PASS' if bad == 0 else f'FAIL ({bad})'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
