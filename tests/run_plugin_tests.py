#!/usr/bin/env python3
"""Single run entry point for the qf_native plugin's death-rule tests.

WHY THIS EXISTS (task tests-written-but-never-registered-as-ctest): the tests under tests/
(cfg_context_key_collision_test.py, connector_arch_derivation_test.py, reject_list_completeness.py,
toolchain_guard_test.py) had NO run entry point — this is a Python ComfyUI plugin, NOT the engine, so there
is no ctest; and there was no pytest.ini / conftest / Makefile / tox / CI workflow either. So the tests
EXISTED but never RAN → e.g. the M5 out-of-scope scope-declaration that cfg_context_key_collision guards
could silently expire with nothing ringing (two dormancies stacked: tests-never-run × declaration-expires).
This runner closes that: it is the ONE command that runs them.

CONTRACT:
  * ENUMERATES the tests (glob tests/*_test.py + *_completeness.py — NOT a hardcoded list, so a NEW test
    file is picked up automatically; this runner + __init__ excluded);
  * per-child exit convention: 0=pass, 1(or any other)=fail, 77=skip (automake SKIP; reject_list uses it);
  * ANY child failing -> the runner exits NON-ZERO (1);
  * counts + REPORTS the in-test '[SKIP]' arm count AND skipped files. A DISCLOSED SKIP IS STILL AN
    UNEXECUTED arm (env-gated on torch / comfy / a fully-initialised ComfyUI) — so it is surfaced, never
    hidden; run in a full ComfyUI+torch env to execute them. `--strict` makes any SKIP/skipped-arm a FAIL
    (use in the full-env CI where nothing should skip);
  * `--selftest` PROVES the runner goes RED on a known-red test AND green on a green-only tree (both
    directions) — not just "green is green".

USAGE:
  python3 tests/run_plugin_tests.py            # run all; non-zero on any FAIL; SKIPs reported (not failed)
  python3 tests/run_plugin_tests.py --strict   # additionally FAIL on any SKIP / skipped arm (full-env CI)
  python3 tests/run_plugin_tests.py --selftest # prove the runner catches a known-red test (both directions)
ENV: PY (python for the child tests; default this interpreter) · COMFY_ROOT (comfy-gated tests read it).
"""
import glob
import os
import subprocess
import sys

# QF_NATIVE_TEST_ROOT lets --selftest point the runner at a synthetic tree to prove RED/GREEN behaviour.
_HERE = os.environ.get("QF_NATIVE_TEST_ROOT") or os.path.dirname(os.path.abspath(__file__))
_PY = os.environ.get("PY", sys.executable)
_SKIP_RC = 77  # automake SKIP convention — reject_list_completeness.py emits it when comfy is absent
_SELF = os.path.basename(os.path.abspath(__file__))


def _discover(root):
    pats = ("*_test.py", "*_completeness.py")
    files = sorted({f for p in pats for f in glob.glob(os.path.join(root, p))})
    return [f for f in files if os.path.basename(f) != _SELF]


def _run_one(path):
    r = subprocess.run([_PY, path], cwd=os.path.dirname(path), capture_output=True, text=True)
    text = (r.stdout or "") + (r.stderr or "")
    return r.returncode, text, text.count("[SKIP]")


def run(root, strict=False):
    tests = _discover(root)
    if not tests:
        print(f"QF_NATIVE TESTS: NO test files under {root} — refusing (fail-closed)")
        return 1
    npass = nfail = nskip = arms = 0
    failed = []
    for t in tests:
        name = os.path.basename(t)
        print(f"=== {name} ===")
        rc, text, sk = _run_one(t)
        if text.strip():
            print(text.rstrip())
        arms += sk
        if rc == 0:
            npass += 1
            print(f"  -> PASS (rc=0){f' [{sk} in-test SKIP arm(s)]' if sk else ''}")
        elif rc == _SKIP_RC:
            nskip += 1
            print(f"  -> SKIP (rc={_SKIP_RC})")
        else:
            nfail += 1
            failed.append(f"{name}(rc={rc})")
            print(f"  -> FAIL (rc={rc})")
    print("=" * 70)
    print(f"QF_NATIVE TESTS: {len(tests)} files | pass={npass} fail={nfail} skip={nskip} | in-test SKIP arms={arms}")
    if arms or nskip:
        print(f"  ⚠ {arms} in-test SKIP arm(s) + {nskip} skipped file(s) NOT executed (env-gated: "
              "torch/comfy/full-ComfyUI). A disclosed SKIP is still unexecuted — run in a full ComfyUI+torch "
              "env (or pass COMFY_ROOT) to execute them.")
    if nfail:
        print(f"  ✗ FAILED: {', '.join(failed)}")
        return 1
    if strict and (arms or nskip):
        print("  ✗ STRICT: a SKIP / skipped arm is treated as FAILURE (--strict full-env mode)")
        return 1
    print("  ✓ all runnable tests passed")
    return 0


def _selftest():
    """Both-directions proof: the runner returns NON-ZERO when a child is red, and ZERO on a green-only tree."""
    import tempfile
    bad = 0

    def _mk(d, name, body):
        open(os.path.join(d, name), "w").write(body)

    def _invoke(root):
        env = dict(os.environ, QF_NATIVE_TEST_ROOT=root)
        return subprocess.run([_PY, os.path.abspath(__file__)], env=env, capture_output=True, text=True).returncode

    # (A) RED: a known-red test file must drive the runner to non-zero.
    with tempfile.TemporaryDirectory() as d:
        _mk(d, "known_green_test.py", "import sys\nsys.exit(0)\n")
        _mk(d, "known_red_test.py", "import sys\nprint('[FAIL] deliberate known-red arm')\nsys.exit(1)\n")
        _mk(d, "known_skip_test.py", "import sys\nprint('[SKIP] env-gated')\nsys.exit(77)\n")
        rc = _invoke(d)
        ok = rc != 0
        print(f"[{'OK ' if ok else 'FAIL'}] selftest RED: red-child tree -> runner rc={rc} (must be non-zero)")
        bad += 0 if ok else 1

    # (B) GREEN: a green-only tree must return zero (no false-red).
    with tempfile.TemporaryDirectory() as d:
        _mk(d, "known_green_test.py", "import sys\nsys.exit(0)\n")
        rc = _invoke(d)
        ok = rc == 0
        print(f"[{'OK ' if ok else 'FAIL'}] selftest GREEN: green-only tree -> runner rc={rc} (must be 0)")
        bad += 0 if ok else 1

    # (C) STRICT: a skip under --strict must fail; without --strict it must pass.
    with tempfile.TemporaryDirectory() as d:
        _mk(d, "known_skip_test.py", "import sys\nprint('[SKIP] env-gated')\nsys.exit(77)\n")
        env = dict(os.environ, QF_NATIVE_TEST_ROOT=d)
        rc_default = subprocess.run([_PY, os.path.abspath(__file__)], env=env, capture_output=True, text=True).returncode
        rc_strict = subprocess.run([_PY, os.path.abspath(__file__), "--strict"], env=env, capture_output=True, text=True).returncode
        ok = rc_default == 0 and rc_strict != 0
        print(f"[{'OK ' if ok else 'FAIL'}] selftest STRICT: skip-only tree -> default rc={rc_default} (0), "
              f"--strict rc={rc_strict} (non-zero)")
        bad += 0 if ok else 1

    # (D) EMPTY: no test files -> fail-closed (never a vacuous green).
    with tempfile.TemporaryDirectory() as d:
        rc = _invoke(d)
        ok = rc != 0
        print(f"[{'OK ' if ok else 'FAIL'}] selftest EMPTY: no-tests tree -> runner rc={rc} (must be non-zero, fail-closed)")
        bad += 0 if ok else 1

    print("RUN_TESTS_SELFTEST: " + ("PASS" if bad == 0 else f"FAIL ({bad} wrong)"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    sys.exit(run(_HERE, strict="--strict" in sys.argv))
