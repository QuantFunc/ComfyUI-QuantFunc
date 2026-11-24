#!/usr/bin/env python3
"""Death rules for the enhance SWITCHES and dual-layer QualityLaw guard (PC Engine Rev 4.2):

  1. Production code carries NO retired option key or helper (tests/_banned_terms.py KEYS/NAMES) and all
     four family loaders wire quality_enhance via _apply_common_loader_dials -> set_video_enhance;
  1b. User-visible text: no "almost the same" claim anywhere; the 5 QI-2.1 workflow notes and README
     describe quality_enhance cleanly;
  2. QFSessionModelMixin.dial_opts() ALWAYS sends the canonical session ratio dial (False -> 0.80, True -> 1.0)
     and NEVER sends retired video_enhance / quality / banned keys; an unconfigured model refuses;
  2b. Dual-layer QualityLaw guard on QFQwenImage21Model:
     - quality_enhance=False + t2i (no refs, batch==1, single cond group) -> 0.85
     - quality_enhance=True  + t2i -> 1.0
     - quality_enhance=False + edit (reference_latents / ref_latents non-empty) -> 1.0
     - quality_enhance=False + CFG > 1 (single_cond_group=False) -> 1.0
     - quality_enhance=False + batch > 1 -> 1.0
  3. QuantFuncH3Loader.audio_enhance is a compatibility-only boolean: set_audio_enhance stores bool(...),
     emits a deprecation warning once when True, and NEVER forwards audio_enhance to the engine.

Run: python3 tests/enhance_switch_test.py
"""
import ast
import glob
import json
import os
import re
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import _banned_terms as bt  # noqa: E402

fails = 0


def check(cond, msg):
    global fails
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        fails += 1


# ---- arm 1: AST scan — retired keys and helpers are unreachable from the plugin path ----
hits, trees = [], {}
for root, _dirs, files in os.walk(_PLUGIN):
    if "/tests" in root or "/.git" in root:
        continue
    for fn in files:
        if not fn.endswith(".py"):
            continue
        path = os.path.join(root, fn)
        try:
            with open(path, encoding="utf-8") as fh:
                tree = trees[os.path.relpath(path, _PLUGIN)] = ast.parse(fh.read(), filename=path)
        except SyntaxError as e:
            hits.append(f"{path}: SyntaxError {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and bt.h(node.value) in bt.KEYS:
                hits.append(f"{path}:{node.lineno}: retired key constant")
            if isinstance(node, ast.Name) and bt.h(node.id) in bt.NAMES:
                hits.append(f"{path}:{node.lineno}: retired name {node.id}")
            if isinstance(node, ast.Attribute) and bt.h(node.attr) in bt.NAMES:
                hits.append(f"{path}:{node.lineno}: retired attribute .{node.attr}")
check(not hits, "arm1: no retired option key or helper reachable from the plugin path (%s)" % (hits or "clean"))

_init_tree = trees["__init__.py"]
_loader_names = ("QuantFuncLTXLoader", "QuantFuncKrea2Loader", "QuantFuncQwenImage21Loader", "QuantFuncH3Loader")
_loader_classes = {c.name: c for c in ast.walk(_init_tree) if isinstance(c, ast.ClassDef) and c.name in _loader_names}
_loader_bases_ok = (len(_loader_classes) == 4 and all(
    any(isinstance(b, ast.Name) and b.id == "_QuantFuncLoaderBase" for b in c.bases)
    for c in _loader_classes.values()))
_load_common_calls = [
    c for cls_node in _loader_classes.values()
    for fn_node in cls_node.body if isinstance(fn_node, ast.FunctionDef) and fn_node.name == "load"
    for c in ast.walk(fn_node)
    if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
    and isinstance(c.func.value, ast.Name) and c.func.value.id == "self" and c.func.attr == "_load_common"
]
_base_cls = next((c for c in ast.walk(_init_tree) if isinstance(c, ast.ClassDef) and c.name == "_QuantFuncLoaderBase"), None)
_base_common_fn = (_base_cls and next(
    (fn for fn in _base_cls.body if isinstance(fn, ast.FunctionDef) and fn.name == "_load_common"), None))
_common_calls = [c for c in ast.walk(_init_tree)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                 and c.func.id == "_apply_common_loader_dials"]
_base_dials_calls = (_base_common_fn and [
    c for c in ast.walk(_base_common_fn)
    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "_apply_common_loader_dials"
]) or []
_sets = [c for c in ast.walk(_init_tree)
         if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
         and c.func.attr == "set_video_enhance"]
check(_loader_bases_ok and len(_load_common_calls) == 4 and len(_common_calls) == 1
      and len(_base_dials_calls) == 1 and len(_sets) == 1,
      f"arm1: all 4 loaders inherit _QuantFuncLoaderBase -> _load_common (x{len(_load_common_calls)}) -> "
      f"_apply_common_loader_dials (x{len(_common_calls)}) wiring set_video_enhance (x{len(_sets)})")

# ---- arm 1b: user-visible text ----
_CLAIM = ("almost the same", "nearly the same", "closer to best", "picture stays the same")
_claims = []
for _root, _dirs, _files in os.walk(_PLUGIN):
    if "/tests" in _root or "/.git" in _root:
        continue
    for _fn in _files:
        _path = os.path.join(_root, _fn)
        if _fn.endswith(".py"):
            with open(_path, encoding="utf-8") as _f:
                for _node in ast.walk(ast.parse(_f.read(), filename=_path)):
                    if isinstance(_node, ast.Constant) and isinstance(_node.value, str) \
                            and any(c in _node.value.lower() for c in _CLAIM):
                        _claims.append(f"{os.path.relpath(_path, _PLUGIN)}:{_node.lineno}")
        elif _fn.endswith((".md", ".json")):
            with open(_path, encoding="utf-8", errors="replace") as _f:
                _t = _f.read().lower()
            _claims += [f"{os.path.relpath(_path, _PLUGIN)}: {c!r}" for c in _CLAIM if c in _t]
check(not _claims, "arm1b: no text claims the faster default gives almost / nearly the same result (%s)" % (_claims or "clean"))

_PLAIN = re.compile(r"(?<![\d.])0\.\d+\b(?!\.)|\d\s*%|\bquality`? option", re.I)
_notes = []
for _wf in sorted(glob.glob(os.path.join(_PLUGIN, "example_workflows", "QuantFunc-QwenImage21-*.json"))):
    with open(_wf, encoding="utf-8") as _f:
        _doc = json.load(_f)
    _txt = " ".join(str(v) for n in _doc.get("nodes", [])
                    for v in (n.get("widgets_values") or []) if isinstance(v, str) and "`quality_enhance`" in v)
    _notes.append((os.path.basename(_wf), _txt))
with open(os.path.join(_PLUGIN, "README.md"), encoding="utf-8") as _f:
    _readme = _f.read()
_readme_qe = [p for p in re.split(r"\n\s*\n", _readme) if "quality_enhance" in p]
_bad = [(w, t[:80]) for w, t in _notes if not t or "subject and scene stay the same" not in t or "highest quality" not in t]
for _w, _t in _notes + [("README.md", p) for p in _readme_qe]:
    _bad += [(_w, m.group(0)) for m in [_PLAIN.search(_t)] if m]
    _bad += [(_w, x) for x in bt.term_hits(_t) + bt.mode_id_hits(_t) + bt.desc_hits(_t, set(), bt.NOTE_PARTS, set())]
_bad += [("README.md", "no quality_enhance paragraph")] if not _readme_qe else []
check(len(_notes) == 5 and not _bad, "arm1b: the 5 QI-2.1 workflow notes and the README describe quality_enhance with the "
      "measured trade and carry no banned term, number or earlier option name (%s)" % (_bad or "clean"))

try:
    sys.argv = [sys.argv[0], "--cpu"]
    if os.environ.get("COMFY_ROOT"):
        sys.path.insert(0, os.environ["COMFY_ROOT"])
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.model_management  # noqa: F401
    import torch
    _pkg = os.path.basename(_PLUGIN)
    sys.path.insert(0, os.path.dirname(_PLUGIN))
    qfmp = __import__(f"{_pkg}.qf_modelpatcher", fromlist=["QFSessionModelMixin"])
    qfq = __import__(f"{_pkg}.qf_qwenimage21_modelpatcher", fromlist=["QFQwenImage21Model"])
    qfh3 = __import__(f"{_pkg}.qf_h3_modelpatcher", fromlist=["QFH3Model"])
except Exception as e:  # noqa: BLE001
    print(f"ENHANCE_SWITCH: arms 2-3 SKIP — comfy/plugin not importable here ({e!r})")
    print("ENHANCE_SWITCH: %s (%d failing checks)" % ("PASS" if fails == 0 else "FAIL", fails))
    sys.exit(77 if fails == 0 else 1)


class _Model(qfmp.QFSessionModelMixin):
    pass


# ---- arm 2: dial_opts sends canonical session ratio dial, never retired video_enhance / quality ----
sent = {}
for on in (False, True):
    m = _Model()
    m._qf = SimpleNamespace(current_session=None)
    m.set_video_enhance(on)
    sent[on] = m.dial_opts()
check(sent[False].get("cat_keep_ratio") == 0.80 and sent[True].get("cat_keep_ratio") == 1.0
      and all("video_enhance" not in o and "quality" not in o
              and not any(bt.h(k) in bt.KEYS for k in o) for o in sent.values()),
      f"arm2: dial_opts emits canonical session ratio (False->0.80, True->1.0), never video_enhance/quality (-> {sent})")

m2 = _Model()
m2._qf = SimpleNamespace(current_session=None)
try:
    m2.dial_opts()
    refused = False
except RuntimeError:
    refused = True
check(refused, "arm2: a model its loader never configured refuses to begin (never a silent default)")

# ---- arm 2b: QI-2.1 plugin-layer QualityLaw guard ----
def _qi21_ratio(quality_enhance, *, has_refs=False, batch=1, cond_or_uncond=(0,),
                uuids=("a",), cond_groups=1, via_extra_conds=False):
    qm = qfq.QFQwenImage21Model.__new__(qfq.QFQwenImage21Model)
    qm._qf = SimpleNamespace(current_session=None)
    qm._run_has_refs = False
    qm._run_batch = 1
    qm._run_single_cond_group = True
    qm._run_cond_groups = cond_groups
    qm.set_video_enhance(quality_enhance)
    x = torch.zeros(batch, 16, 1, 8, 8, dtype=torch.bfloat16)
    ctx = torch.zeros(batch, 4, 16, dtype=torch.bfloat16)
    opts = {"cond_or_uncond": list(cond_or_uncond), "uuids": list(uuids)}
    kw = {}
    if via_extra_conds:
        qm._run_has_refs = True
    elif has_refs:
        kw["ref_latents"] = [torch.zeros(batch, 16, 1, 8, 8, dtype=torch.bfloat16)]
    qm._prepare_begin_context(x, ctx, opts, kw)
    return qm.dial_opts().get("cat_keep_ratio")


check(_qi21_ratio(False) == 0.85,
      f"arm2b: QI-2.1 t2i (no refs, B=1, single cond, quality_enhance=False) -> 0.85 (got {_qi21_ratio(False)})")
check(_qi21_ratio(True) == 1.0,
      "arm2b: QI-2.1 quality_enhance=True -> 1.0")
check(_qi21_ratio(False, has_refs=True) == 1.0,
      "arm2b: QI-2.1 edit (ref_latents) -> 1.0 even when quality_enhance=False")
check(_qi21_ratio(False, via_extra_conds=True) == 1.0,
      "arm2b: QI-2.1 edit (extra_conds reference_latents) -> 1.0 even when quality_enhance=False")
check(_qi21_ratio(False, cond_or_uncond=(0, 1), uuids=("a", "b")) == 1.0,
      "arm2b: QI-2.1 CFG>1 (cond_or_uncond=[0,1]) -> 1.0")
check(_qi21_ratio(False, cond_groups=2) == 1.0,
      "arm2b: QI-2.1 multi-cond-group -> 1.0")
check(_qi21_ratio(False, batch=2) == 1.0,
      "arm2b: QI-2.1 batch>1 (B=2) -> 1.0")

# ---- arm 3: H3 audio_enhance compatibility switch never reaches engine ----
h = qfh3.QFH3Model.__new__(qfh3.QFH3Model)
h._qf = SimpleNamespace(current_session=None)
h.set_video_enhance(False)
h.set_audio_enhance("yes")
h_opts = h.dial_opts()
check(h._audio_enhance is True and "audio_enhance" not in h_opts,
      f"arm3: set_audio_enhance stores bool and never sends audio_enhance to engine (opts={h_opts})")

print("ENHANCE_SWITCH: %s (%d failing checks)" % ("PASS" if fails == 0 else "FAIL", fails))
sys.exit(0 if fails == 0 else 1)
