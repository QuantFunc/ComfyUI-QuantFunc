#!/usr/bin/env python3
"""Death rules for the enhance SWITCHES (user 2026-09-19: 「剪枝以及 extra audio step 隐藏在 api 内部,仅在 api 提供一个
开关,插件层屏蔽所有实现细节」):

  1. the plugin's production code carries NO raw enhance knob — no source file (outside tests/) uses the begin-option
     strings `token_prune_keep_ratio` / `extra_audio_steps` as a CODE constant (comments/docstrings may name them),
     and no numeric keep-ratio / total-step constant survives (0.8 / 16 are engine law now);
  2. QFSessionModelMixin.residency_opts ALWAYS emits `quality` (the one speed/quality switch; the engine speaks it) and
     never the retired `video_enhance` or a raw key; a model its loader never gave a quality refuses to begin;
  3. the audio switch setter is a boolean: set_audio_enhance stores bool(...).

Run:  python tests/enhance_switch_test.py   (arm 1 is pure-python; arms 2-3 need comfy importable → SKIP (77) without it)
"""
import ast
import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.dirname(_HERE)
FORBIDDEN_KEYS = {"token_prune_keep_ratio", "extra_audio_steps", "video_enhance"}   # video_enhance: retired, never sent
FORBIDDEN_NAMES = {"_AUDIO_ENHANCE_TOTAL_STEPS", "_quality_enhance_to_token_prune", "set_token_prune",
                   "set_video_enhance", "_video_enhance"}

fails = 0
def check(cond, msg):
    global fails
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        fails += 1


# ---- arm 1: AST scan — raw knobs are unreachable from the plugin path ---------------------------------------------
hits = []
for root, _dirs, files in os.walk(_PLUGIN):
    if "/tests" in root or "/.git" in root:
        continue
    for fn in files:
        if not fn.endswith(".py"):
            continue
        path = os.path.join(root, fn)
        try:
            tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        except SyntaxError as e:
            hits.append(f"{path}: SyntaxError {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in FORBIDDEN_KEYS:
                # a docstring is an Expr(Constant) statement — those are prose, not code
                hits.append(f"{path}:{node.lineno}: code constant {node.value!r}")
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                hits.append(f"{path}:{node.lineno}: name {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
                hits.append(f"{path}:{node.lineno}: attribute .{node.attr}")
# (a docstring/comment that merely MENTIONS a key is not an exact-match Constant, so prose never trips this)
check(not hits, "arm1: no raw enhance knob reachable from the plugin path (%s)" % (hits or "clean"))

# ---- arm 1b: every user-visible text states the MEASURED quality behaviour ----------------------------------------------
# (tests-07 re-CR rounds 4-5): vs best_quality the faster options change details on every family (QI-2.1 PSNR 23-27 dB,
# Krea-2 balance 20.7-22 dB, LTX-2.5 balance 15.8 dB), so no text may claim "almost the same" / "nearly the same" /
# "closer" / "the picture stays the same" (pose or composition does change): not a tooltip or description (every string constant in the plugin's code, f-strings included), not the README,
# not a workflow note. And the QI-2.1 notes carry the measured wording plus the no-choice rule (a GPU without the fast mode
# shows no choice and always runs best_quality).
_CLAIM = ("almost the same", "nearly the same", "closer to best", "picture stays the same")
sys.path.insert(0, _HERE)
from quality_tier_rules import tier_problems  # noqa: E402  (the per-tier rule, shared with loader_dispatch_test)
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
check(not _claims, "arm1b: no text claims a faster option gives almost / nearly the same result (%s)" % (_claims or "clean"))
import glob as _glob
import json as _json
_qi_notes = []
for _wf in sorted(_glob.glob(os.path.join(_PLUGIN, "example_workflows", "QuantFunc-QwenImage21-*.json"))):
    _txt = " ".join(str(v) for n in _json.load(open(_wf, encoding="utf-8")).get("nodes", [])
                    for v in (n.get("widgets_values") or []) if isinstance(v, str) and "`quality`" in v)
    _qi_notes.append((os.path.basename(_wf), _txt))
_bad = [(w, t[:80]) for w, t in _qi_notes if not t
        or "details such as poses, faces or small objects can differ" not in t or "always uses the highest quality" not in t]
_bad += [(w, p) for w, t in _qi_notes for p in tier_problems(t)]
_bad += [("README.md", p) for p in tier_problems(open(os.path.join(_PLUGIN, "README.md"), encoding="utf-8").read())]
check(len(_qi_notes) == 5 and not _bad, "arm1b: the 5 QI-2.1 workflow notes and the README carry the measured per-tier "
      "wording: balance keeps the subject and scene, fast / super_fast can give another variation (%s)" % (_bad or "clean"))

try:
    # Fake-engine test: run ComfyUI in its own --cpu mode (the contract tests' idiom), so a box with no visible GPU
    # (the CPU suite hides CUDA) imports comfy instead of skipping every arm. Must precede the first comfy import.
    sys.argv = [sys.argv[0], "--cpu"]
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


# ---- arm 2: residency_opts sends quality, every option, never the retired switch or a raw key -----------------------
m = _Model(); m._qf = SimpleNamespace(current_session=None)
sent = []
for q in ("super_fast", "fast", "balance", "best_quality"):
    m.set_quality(q)
    sent.append(m.residency_opts())
check([o.get("quality") for o in sent] == ["super_fast", "fast", "balance", "best_quality"]
      and not any(k in o for o in sent for k in FORBIDDEN_KEYS), "arm2: every quality is SENT, never video_enhance / a raw key")
m2 = _Model(); m2._qf = SimpleNamespace(current_session=None)
try:
    m2.residency_opts()
    refused = False
except RuntimeError:
    refused = True
check(refused, "arm2: a model its loader never gave a quality refuses to begin (a wiring error, never a silent default)")

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
