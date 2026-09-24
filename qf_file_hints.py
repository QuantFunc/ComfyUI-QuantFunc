"""The ONE predicate deciding whether a transformer file's NAME fits a preset's `file_hints`.

Shared by the loader core (`__init__._run_family_load`) and `tests/published_names_test.py`, so the test that pins
every file we PUBLISH on ModelScope to its preset exercises exactly the production match — never a copy of it.
#713: the Qwen-Image-2.1 hints were written for the internal export names, the published files were named later,
and nothing ever ran the published names through this check, so every user downloading them was refused.
Dependency-free on purpose (the test imports it without ComfyUI)."""
import fnmatch
import os


def name_matches_hints(path, patterns):
    """True when basename(path) matches one of `patterns` (fnmatch, case-insensitive)."""
    base = os.path.basename(path).lower()
    return any(fnmatch.fnmatch(base, str(p).lower()) for p in patterns)
