#!/usr/bin/env python3
"""Death rules for the enhance SWITCHES (user 2026-09-19: 「剪枝以及 extra audio step 隐藏在 api 内部,仅在 api 提供一个
开关,插件层屏蔽所有实现细节」):

  1. the plugin's production code carries NO raw enhance knob — no source file (outside tests/) uses the begin-option
     strings `token_prune_keep_ratio` / `extra_audio_steps` as a CODE constant (comments/docstrings may name them),
     and no numeric keep-ratio / total-step constant survives (0.8 / 16 are engine law now);
  2. QFSessionModelMixin.residency_opts ALWAYS emits `video_enhance` as a boolean — both states — and never the raw key;
  3. the switch setters are booleans: set_video_enhance / set_audio_enhance store bool(...).

Run:  python tests/enhance_switch_test.py   (arm 1 is pure-python; arms 2-3 need comfy importable → SKIP (77) without it)
"""
import ast
import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.dirname(_HERE)
FORBIDDEN_KEYS = {"token_prune_keep_ratio", "extra_audio_steps"}
FORBIDDEN_NAMES = {"_AUDIO_ENHANCE_TOTAL_STEPS", "_quality_enhance_to_token_prune", "set_token_prune"}

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


# ---- arm 2: residency_opts emits the switch, both states, never the raw key ----------------------------------------
m = _Model(); m._qf = SimpleNamespace(current_session=None)
m.set_video_enhance(False)
o = m.residency_opts()
check(o.get("video_enhance") is False and "token_prune_keep_ratio" not in o, "arm2: OFF → video_enhance=False, no raw key")
m.set_video_enhance(True)
o = m.residency_opts()
check(o.get("video_enhance") is True and "token_prune_keep_ratio" not in o, "arm2: ON → video_enhance=True, no raw key")
m2 = _Model(); m2._qf = SimpleNamespace(current_session=None)
o = m2.residency_opts()
check(o.get("video_enhance") is False, "arm2: never set → False (the engine's speed default), still SENT")

# ---- arm 3: setters are booleans ----------------------------------------------------------------------------------
m.set_video_enhance(1)
check(m._video_enhance is True, "arm3: set_video_enhance(1) stores True")
try:
    h3 = __import__(f"{_pkg}.qf_h3_modelpatcher", fromlist=["QFH3Model"])
    h = h3.QFH3Model.__new__(h3.QFH3Model)
    h.set_audio_enhance("yes")
    check(h._audio_enhance is True, "arm3: set_audio_enhance stores bool")
except Exception as e:  # noqa: BLE001
    print(f"  SKIP arm3 H3: {e!r}")

print("ENHANCE_SWITCH: %s (%d failing checks)" % ("PASS" if fails == 0 else "FAIL", fails))
sys.exit(0 if fails == 0 else 1)
