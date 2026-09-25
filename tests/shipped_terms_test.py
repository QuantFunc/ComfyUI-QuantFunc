#!/usr/bin/env python3
"""Death rule (user rule 2026-09-19, reaffirmed 2026-09-25 「日志没关系 不能通过插件代码看到就好」): no shipped file of the plugin —
code, comments, docstrings, tests, README, example workflows, configs — carries a term that would tell a reader how the engine
gets faster. The terms live in tests/_banned_terms.py as SHA-256 only, so neither that file nor this one names them.
Scanned: every tracked file (git ls-files; a plain directory walk where there is no git) except the vendored tokenizer
vocabularies under bin/tokenizers/ (a model's word list, not the plugin's wording); model file names inside a file are skipped.
Each file is checked whole (a phrase broken across two lines still counts) and line by line (for the location).
Run:  python tests/shipped_terms_test.py   (the matcher's own both-ways check runs first)
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from _banned_terms import selftest, term_hits  # noqa: E402

_VENDORED = ("bin/tokenizers/",)


def shipped_files(root=_PLUGIN):
    try:
        out = subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True, check=True, timeout=60).stdout
        files = [f for f in out.decode("utf-8", "replace").split("\0") if f]
    except (OSError, subprocess.SubprocessError):
        files = [os.path.relpath(os.path.join(r, f), root) for r, _d, fs in os.walk(root)
                 if ".git" not in os.path.relpath(r, root).split(os.sep) for f in fs]
    return sorted(f for f in files if not f.startswith(_VENDORED))


def scan(root=_PLUGIN):
    """(files scanned, [(file, line or None, term)])"""
    n, found = 0, []
    for rel in shipped_files(root):
        try:
            data = open(os.path.join(root, rel), "rb").read()
        except OSError:
            continue
        if b"\0" in data:   # a binary file
            continue
        n += 1
        text = data.decode("utf-8", "replace")
        whole = term_hits(text)
        if not whole:
            continue
        per_line = [(rel, i, t) for i, line in enumerate(text.splitlines(), 1) for t in term_hits(line)]
        found += per_line
        # a phrase broken across two lines is found only on the whole text
        spanning = list(whole)
        for _r, _i, t in per_line:
            if t in spanning:
                spanning.remove(t)
        found += [(rel, None, t) for t in spanning]
    return n, found


def main():
    fails = 0
    selftest()
    print("  PASS the matcher finds each kind of planted canary (word, phrase, CJK) and passes clean text")
    n, found = scan()
    for rel, line, term in found[:40]:
        print(f"  HIT {rel}:{line if line else '(spans lines)'}: {term}")
    ok = n > 0 and not found
    print(("  PASS " if ok else "  FAIL ") + f"no shipped file carries a banned term ({n} text files scanned, {len(found)} hits)")
    fails += not ok
    print("SHIPPED_TERMS: %s (%d failing checks)" % ("PASS" if fails == 0 else "FAIL", fails))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
