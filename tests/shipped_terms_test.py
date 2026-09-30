#!/usr/bin/env python3
"""Death rule (user rule 2026-09-19, reaffirmed 2026-09-25 「日志没关系 不能通过插件代码看到就好」): no shipped file of the plugin —
code, comments, docstrings, tests, README, example workflows, configs — carries a term that would tell a reader how the engine
gets faster. The terms live in tests/_banned_terms.py as SHA-256 only, so neither that file nor this one names them.
Scanned: every tracked file (git ls-files; a plain directory walk where there is no git) except the vendored tokenizer
vocabularies under bin/tokenizers/ (a model's word list, not the plugin's wording); model file names inside a file are skipped.
Each file is checked whole (a phrase broken across two lines still counts) and line by line (for the location).
Run:  python tests/shipped_terms_test.py   (the matcher's own both-ways check runs first)
"""
import io
import os
import subprocess
import sys
import tempfile
import tokenize

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from _banned_terms import h, selftest, term_hits  # noqa: E402

_VENDORED = ("bin/tokenizers/",)
# User 2026-09-29 explicitly restored one formerly-hidden backend as a dropdown label. The exception is that exact option
# VALUE only - a whole string constant "<word>", lowercase, nothing else in it - in the two files that offer and test it:
# a comment, a docstring, a tooltip or message, an identifier or any other file stays under the rule. The word is stored
# as a digest, like every banned term.
_EXPLICIT_BACKEND_HASH = "6a14c3ef2763b29b1d98a553f1b21c4243079103a93b4869be6d199940526a0f"
_EXPLICIT_BACKEND_FILES = {"__init__.py", "qf_modelpatcher.py", "tests/loader_dispatch_test.py"}
_STRINGISH = {tokenize.STRING, getattr(tokenize, "FSTRING_START", -1), getattr(tokenize, "FSTRING_END", -1)}


def _blank_option_value(text, allowed=_EXPLICIT_BACKEND_HASH):
    """Python source `text` with each string constant that IS the option value blanked out (Python's own tokenizer
    decides what a whole string constant is; one next to another string is part of an implicitly concatenated text, so
    it is not). Source that does not tokenize is returned unchanged, so it is scanned whole."""
    lines = text.splitlines(keepends=True)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError):
        return text
    toks = [t for t in toks if t.type not in (tokenize.NL, tokenize.COMMENT)]
    for i, t in enumerate(toks):
        s = t.string
        alone = all(n.type not in _STRINGISH for n in toks[max(i - 1, 0):i] + toks[i + 1:i + 2])
        if (t.type == tokenize.STRING and alone and t.start[0] == t.end[0] and len(s) > 2 and s[0] == s[-1] == '"'
                and s[1] != '"' and s[1:-1] == s[1:-1].lower() and h(s[1:-1]) == allowed):
            (row, c0), c1 = t.start, t.end[1]
            lines[row - 1] = lines[row - 1][:c0] + " " * (c1 - c0) + lines[row - 1][c1:]
    return "".join(lines)



def exemption_selftest():
    """The option-value exemption, both ways, on a canary: only a whole "<word>" string constant is blanked; the word in a
    comment, a docstring, a longer or single-quoted string, an identifier or another case is still found."""
    cw = {h("zqvcanary")}

    def hits(src):
        return term_hits(_blank_option_value(src, allowed=h("zqvcanary")), cw, set(), set())
    kept = ['X = ["auto", "zqvcanary", "flash"]\n', 'if v == "zqvcanary":\n    pass\n', 'f(("zqvcanary", 1))\n']
    found = ['# the "zqvcanary" option\n', "x = 'use \"zqvcanary\" here'\n", 'x = "zqvcanary for you"\n',
             '"""the "zqvcanary" doc"""\n', "_ZQVCANARY_SMS = {75}\n", 'X = ["ZQVCANARY"]\n', "x = 'zqvcanary'\n",
             'X = ["zqvcanary"]  # zqvcanary\n', 'raise E("pick " "zqvcanary" " x")\n',
             'm = ("pick "\n     "zqvcanary")\n']
    ok = (all(not hits(s) for s in kept) and all(hits(s) for s in found)
          and all(f.endswith(".py") for f in _EXPLICIT_BACKEND_FILES))   # markdown/json/js never get the blanking
    return ok, ([hits(s) for s in kept], [hits(s) for s in found])

def listed_files_selftest():
    """The blanking applies to the listed files only: the real option value (read from __init__.py, never spelled here),
    written as that same whole string constant into files that are not listed, is a hit there and nowhere else."""
    src = open(os.path.join(_PLUGIN, "__init__.py"), encoding="utf-8").read()
    word = next((t.string[1:-1] for t in tokenize.generate_tokens(io.StringIO(src).readline)
                 if t.type == tokenize.STRING and len(t.string) > 2 and h(t.string[1:-1]) == _EXPLICIT_BACKEND_HASH), None)
    if word is None:
        return False, "no option value in __init__.py"
    with tempfile.TemporaryDirectory() as d:
        for rel in ("__init__.py", "README.md", "other.py"):
            with open(os.path.join(d, rel), "w", encoding="utf-8") as f:
                f.write(f'X = ["auto", "{word}"]\n')
        _n, found = scan(d)
    hit = sorted({r for r, _l, _t in found})
    return hit == ["README.md", "other.py"], hit


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
        if rel in _EXPLICIT_BACKEND_FILES:
            text = _blank_option_value(text)
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
    ok, detail = exemption_selftest()
    print(("  PASS " if ok else "  FAIL ") + "the option-value exemption blanks only a whole-string option value, only in its Python files"
          + ("" if ok else f" -> {detail}"))
    fails += not ok
    ok, detail = listed_files_selftest()
    print(("  PASS " if ok else "  FAIL ") + "a file that is not listed gets no blanking: the real option value is a hit there"
          + ("" if ok else f" -> {detail}"))
    fails += not ok
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
