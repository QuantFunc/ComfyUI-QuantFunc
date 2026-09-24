#!/usr/bin/env python3
"""#713 death rule: every transformer file QuantFunc PUBLISHES on ModelScope passes its own preset's file_hints, and
no other preset / arm accepts it.

The Qwen-Image-2.1 hints were written for the internal export names; the files were published later under another
name and every user downloading them was refused by our own loader. Nothing ran the published names through the
check. This test does, with the PRODUCTION predicate (qf_file_hints.name_matches_hints, the one _run_family_load
calls), over the frozen list tests/published_names.json. Refresh that list after every publish:
    python3 tests/refresh_published_names.py            (--check: exit 1 when ModelScope drifted from the list)
A new repo (UNCLASSIFIED) or a new file without a preset assignment turns this test RED until a person assigns it.
A repo classified 'retiring: <why>' belongs to a family being removed from the release: each of its files is checked as
native while its assigned preset is still in this tree, and as none (refused by every preset) once the preset is gone, so
the test holds on both sides of the removal and never leaves a file unchecked.

RED control (the accept check must be able to fail): the pre-#713 QI-2.1 hints refuse the published QI names.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
sys.path.insert(0, PLUGIN)
from qf_file_hints import name_matches_hints  # noqa: E402  (production predicate)

# The QI-2.1 hints as shipped before #713 (d36dc49) — frozen here only as the RED control.
PRE_713_QI21_HINTS = ["*qwen-image-2.1*quantfunc*int4*", "*qwen_image_2.1*quantfunc*int4*", "*qwen-image-2.1*quantfunc*"]

bad = 0


def check(name, ok, detail=""):
    global bad
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {'' if ok else detail}")
    bad += 0 if ok else 1


def main():
    presets = {os.path.basename(os.path.dirname(f)): json.load(open(f)).get("file_hints") or {}
               for f in glob.glob(os.path.join(PLUGIN, "configs", "*", "qf_native.json"))}
    slots = [(p, a, pats) for p, h in presets.items() for a, pats in h.items()]
    listing = json.load(open(os.path.join(HERE, "published_names.json")))
    check("frozen list has repos", bool(listing.get("repos")))
    n_native = 0
    for repo, r in sorted(listing["repos"].items()):
        loader = r.get("loader", "")
        check(f"{repo}: classified (native | none: <why> | retiring: <why>)",
              loader == "native" or loader.startswith(("none:", "retiring:")), f"-> {loader!r}")
        for path, a in sorted(r.get("files", {}).items()):
            name = os.path.basename(path)
            accepted_by = [f"{p}:{arm}" for p, arm, pats in slots if name_matches_hints(name, pats)]
            mode = loader
            if loader.startswith("retiring:"):   # the TREE decides: preset still here -> native, removed -> none
                mode = "native" if a.get("preset") in presets else "none: preset removed"
                print(f"[RETIRING] {repo}/{name}: checked as {mode.split(':')[0]}")
            if mode != "native":
                check(f"{repo}/{name}: refused by every native preset", not accepted_by, f"-> accepted by {accepted_by}")
                continue
            n_native += 1
            want = f"{a.get('preset')}:{a.get('arm')}"
            if not (a.get("preset") and a.get("arm")):
                check(f"{repo}/{name}: assigned to a preset + arm", False, "-> unassigned; set it in published_names.json")
                continue
            check(f"{repo}/{name}: preset {want} exists with hints", bool(presets.get(a["preset"], {}).get(a["arm"])),
                  f"-> presets {sorted(presets)}")
            check(f"{repo}/{name}: accepted by {want} and by nothing else", accepted_by == [want], f"-> accepted by {accepted_by}")
    check("at least one native published file", n_native > 0)

    qi = [os.path.basename(p) for p in listing["repos"].get("Qwen-Image-2.1-4bit", {}).get("files", {})]
    check("RED control: QI-2.1 files are in the list", bool(qi))
    for name in qi:
        check(f"RED control: pre-#713 QI-2.1 hints refuse {name}", not name_matches_hints(name, PRE_713_QI21_HINTS))

    print("PUBLISHED_NAMES:", "PASS" if bad == 0 else f"FAIL ({bad} wrong)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
