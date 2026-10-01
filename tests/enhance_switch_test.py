#!/usr/bin/env python3
"""Death rules for the enhance SWITCHES (user rule 2026-09-19: the engine's methods stay inside the engine API, which offers
only switches; the plugin layer carries no implementation detail. The quality switch, user 2026-09-25: ONE quality_enhance
switch; what each state does per model family is engine law):

  1. the plugin's production code carries NO retired option key or helper — no source file (outside tests/) uses one of the
     keys in tests/_banned_terms.py (KEYS, compared as SHA-256) as a CODE constant or one of its retired NAMES as a name or
     attribute (comments/docstrings are covered by tests/shipped_terms_test.py), and each of the four loaders hands its run
     the engine's switch (set_video_enhance);
  1b. user-visible text: no "almost the same" claim anywhere; the switch's texts (the README, the five QI-2.1 workflow notes)
     describe the switch and carry no banned term, number or earlier option name;
  2. QFSessionModelMixin.residency_opts ALWAYS sends `video_enhance` (both states) and never `quality` or a retired key; a
     model its loader never configured refuses to begin;
  3. the audio switch setter is a boolean: set_audio_enhance stores bool(...).

Run:  python tests/enhance_switch_test.py   (arms 1-1b are pure-python; arms 2-3 need comfy importable (COMFY_ROOT) → SKIP (77)
without it)
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


# ---- arm 1: AST scan — retired keys and helpers are unreachable from the plugin path ---------------------------------------
hits, trees = [], {}
for root, _dirs, files in os.walk(_PLUGIN):
    if "/tests" in root or "/.git" in root:
        continue
    for fn in files:
        if not fn.endswith(".py"):
            continue
        path = os.path.join(root, fn)
        try:
            tree = trees[os.path.relpath(path, _PLUGIN)] = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        except SyntaxError as e:
            hits.append(f"{path}: SyntaxError {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and bt.h(node.value) in bt.KEYS:
                # a docstring is an Expr(Constant) statement — prose, not code; an exact-match constant is code
                hits.append(f"{path}:{node.lineno}: retired key constant")
            if isinstance(node, ast.Name) and bt.h(node.id) in bt.NAMES:
                hits.append(f"{path}:{node.lineno}: retired name {node.id}")
            if isinstance(node, ast.Attribute) and bt.h(node.attr) in bt.NAMES:
                hits.append(f"{path}:{node.lineno}: retired attribute .{node.attr}")
check(not hits, "arm1: no retired option key or helper reachable from the plugin path (%s)" % (hits or "clean"))
_sets = [c for c in ast.walk(trees["__init__.py"]) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
         and c.func.attr == "set_video_enhance"]
check(len(_sets) == 4, f"arm1: each of the four loaders hands its run the engine's switch (set_video_enhance x{len(_sets)})")

# ---- arm 1b: user-visible text ------------------------------------------------------------------------------------------
# (tests-07 re-CR rounds 4-5): the faster default changes details on every family vs maximum quality (QI-2.1 PSNR 23-27 dB,
# Krea-2 20.7-22 dB, LTX-2.5 15.8 dB), so no text may claim "almost the same" / "nearly the same" / "closer" / "the picture
# stays the same": not a tooltip or description (every string constant in the plugin's code), not the README, not a note.
_CLAIM = ("almost the same", "nearly the same", "closer to best", "picture stays the same")
_claims = []
for _root, _dirs, _files in os.walk(_PLUGIN):
    if "/tests" in _root or "/.git" in _root:
        continue
    for _fn in _files:
        _path = os.path.join(_root, _fn)
        if _fn.endswith(".py"):
            for _node in ast.walk(ast.parse(open(_path, encoding="utf-8").read(), filename=_path)):
                if isinstance(_node, ast.Constant) and isinstance(_node.value, str) \
                        and any(c in _node.value.lower() for c in _CLAIM):
                    _claims.append(f"{os.path.relpath(_path, _PLUGIN)}:{_node.lineno}")
        elif _fn.endswith((".md", ".json")):
            _t = open(_path, encoding="utf-8", errors="replace").read().lower()
            _claims += [f"{os.path.relpath(_path, _PLUGIN)}: {c!r}" for c in _CLAIM if c in _t]
check(not _claims, "arm1b: no text claims the faster default gives almost / nearly the same result (%s)" % (_claims or "clean"))
# the switch's texts: the five QI-2.1 workflow notes and the README describe it, with the measured trade, and carry no banned
# term and neither NOTE_PARTS word (both hashed), no fraction (a version is not one) or percentage and no earlier option name
_PLAIN = re.compile(r"(?<![\d.])0\.\d+\b(?!\.)|\d\s*%|\bquality`? option", re.I)
_notes = []
for _wf in sorted(glob.glob(os.path.join(_PLUGIN, "example_workflows", "QuantFunc-QwenImage21-*.json"))):
    _txt = " ".join(str(v) for n in json.load(open(_wf, encoding="utf-8")).get("nodes", [])
                    for v in (n.get("widgets_values") or []) if isinstance(v, str) and "`quality_enhance`" in v)
    _notes.append((os.path.basename(_wf), _txt))
_readme = open(os.path.join(_PLUGIN, "README.md"), encoding="utf-8").read()
_readme_qe = [p for p in re.split(r"\n\s*\n", _readme) if "quality_enhance" in p]
_bad = [(w, t[:80]) for w, t in _notes if not t or "subject and scene stay the same" not in t or "highest quality" not in t]
for _w, _t in _notes + [("README.md", p) for p in _readme_qe]:
    _bad += [(_w, m.group(0)) for m in [_PLAIN.search(_t)] if m]
    _bad += [(_w, x) for x in bt.term_hits(_t) + bt.mode_id_hits(_t) + bt.desc_hits(_t, set(), bt.NOTE_PARTS, set())]
_bad += [("README.md", "no quality_enhance paragraph")] if not _readme_qe else []
check(len(_notes) == 5 and not _bad, "arm1b: the 5 QI-2.1 workflow notes and the README describe quality_enhance with the "
      "measured trade and carry no banned term, number or earlier option name (%s)" % (_bad or "clean"))

try:
    # Fake-engine test: run ComfyUI in its own --cpu mode (the contract tests' idiom), so a box with no visible GPU
    # (the CPU suite hides CUDA) imports comfy instead of skipping every arm. Must precede the first comfy import.
    sys.argv = [sys.argv[0], "--cpu"]
    if os.environ.get("COMFY_ROOT"):
        sys.path.insert(0, os.environ["COMFY_ROOT"])
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.model_management  # noqa: F401
    _pkg = os.path.basename(_PLUGIN)
    sys.path.insert(0, os.path.dirname(_PLUGIN))
    qfmp = __import__(f"{_pkg}.qf_modelpatcher", fromlist=["QFSessionModelMixin"])
except Exception as e:  # noqa: BLE001
    print(f"ENHANCE_SWITCH: arms 2-3 SKIP — comfy/plugin not importable here ({e!r})")
    print("ENHANCE_SWITCH: %s (%d failing checks)" % ("PASS" if fails == 0 else "FAIL", fails))
    sys.exit(77 if fails == 0 else 1)

Mixin = qfmp.QFSessionModelMixin


class _Model(Mixin):
    pass


# ---- arm 2: residency_opts sends the engine's switch, never `quality` or a retired key -------------------------------------
sent = {}
for on in (False, True):
    m = _Model(); m._qf = SimpleNamespace(current_session=None)
    m.set_video_enhance(on)
    sent[on] = m.residency_opts()
check(all(o.get("video_enhance") is on and not any(k == "quality" or bt.h(k) in bt.KEYS for k in o) for on, o in sent.items()),
      f"arm2: video_enhance is SENT in both states, never `quality` or a retired key (-> {sent})")
m2 = _Model(); m2._qf = SimpleNamespace(current_session=None)
try:
    m2.residency_opts()
    refused = False
except RuntimeError:
    refused = True
check(refused, "arm2: a model its loader never configured refuses to begin (a wiring error, never a silent default)")

# ---- arm 3: the audio switch setter is a boolean --------------------------------------------------------------------
try:
    h3 = __import__(f"{_pkg}.qf_h3_modelpatcher", fromlist=["QFH3Model"])
    h = h3.QFH3Model.__new__(h3.QFH3Model)
    h.set_audio_enhance("yes")
    check(h._audio_enhance is True, "arm3: set_audio_enhance stores bool")
except Exception as e:  # noqa: BLE001
    print(f"  SKIP arm3 H3: {e!r}")

print("ENHANCE_SWITCH: %s (%d failing checks)" % ("PASS" if fails == 0 else "FAIL", fails))
sys.exit(0 if fails == 0 else 1)
