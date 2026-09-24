#!/usr/bin/env python3
"""Every official preset whose model_index.json declares a scheduler ships scheduler/scheduler_config.json.

The engine reads <model_dir>/scheduler/scheduler_config.json when it builds these pipelines. A preset
without the file makes every native load print "Scheduler config not found ... using pipeline-arch
default", and a warning is the one thing the engine's default console level still shows.
MUTATION: delete configs/qwen-image-2.1-int4/scheduler/scheduler_config.json -> this test goes RED.

Run:  python tests/preset_scheduler_completeness.py   (pure Python; no ComfyUI, torch or engine library)
"""
import glob
import json
import os
import sys

_CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def _declares_scheduler(model_index_path):
    with open(model_index_path, encoding="utf-8") as fh:
        return "scheduler" in json.load(fh)


def _readable_scheduler(path):
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return False
    return isinstance(cfg, dict) and isinstance(cfg.get("_class_name"), str)


def main():
    presets = [os.path.dirname(p) for p in sorted(glob.glob(os.path.join(_CONFIGS, "*", "model_index.json")))
               if _declares_scheduler(p)]
    missing = [os.path.relpath(os.path.join(d, "scheduler", "scheduler_config.json"), _CONFIGS)
               for d in presets if not _readable_scheduler(os.path.join(d, "scheduler", "scheduler_config.json"))]
    for m in missing:
        print(f"FAIL missing or unreadable: configs/{m}")
    ok = bool(presets) and not missing   # no presets found at all is a broken path, not a pass
    print(f"PRESET_SCHEDULER: {'PASS' if ok else 'FAIL'} ({len(presets)} preset(s) declare a scheduler, "
          f"{len(missing)} without a readable scheduler_config.json)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
