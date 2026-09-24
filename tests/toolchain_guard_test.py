#!/usr/bin/env python3
"""Death-rule for FORK-2's fail-closed CUDA-toolchain guard (qf_engine.assert_toolchain_compatible).

The engine .so loads IN-PROCESS and shares ComfyUI's torch CUDA context; a torch-CUDA / .so-CUDA major
mismatch is an unverified combination that can SILENTLY corrupt output (never a clean fault). The guard
must REFUSE such a load, not warn. This test asserts:
  (A) each of the 6 toolchain states gives its OWN verdict (load vs refuse) — a guard that refuses in
      EVERY state (or loads in every state) is decoration, not a guard;
  (B) load_lib() actually CALLS the guard BEFORE ctypes.CDLL (the decision-logic cases below monkeypatch
      assert_toolchain_compatible directly, so a silent removal of the CALL from load_lib would be
      invisible to them — this arm goes through the PRODUCTION entry point, per the "a suite that bypasses
      the production entry point won't notice a de-wiring" lesson);
  (C) the real ELF DT_NEEDED parser extracts NEEDED from a genuine binary;
  (D) Windows: the real PE import reader and the real guard on built PE32+ images (cudart64_N / cuBLAS-only /
      delay-loaded / no CUDA / truncated / mismatch).
(A) monkeypatches _is_elf / _so_cuda_major; every arm installs a fake torch, so it runs anywhere (no CUDA / no real
torch / no engine library needed).

Run: python3 toolchain_guard_test.py   (exit 0 = all correct; exit 1 = a check is wrong)
"""
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import qf_engine as qfe  # noqa: E402


def _set_fake_torch(cuda):
    m = types.ModuleType("torch")
    m.version = types.SimpleNamespace(cuda=cuda)
    sys.modules["torch"] = m


def _refuses(torch_cuda, so_major, override=False, is_elf=True):
    """True iff assert_toolchain_compatible RAISES for (torch cuda, .so cudart major, ELF-ness)."""
    if override:
        os.environ[qfe._ENV_ALLOW_UNVERIFIED_TOOLCHAIN] = "1"
    else:
        os.environ.pop(qfe._ENV_ALLOW_UNVERIFIED_TOOLCHAIN, None)
    _set_fake_torch(torch_cuda)
    o_is_elf, o_major = qfe._is_elf, qfe._so_cuda_major
    qfe._is_elf = lambda _p: is_elf
    qfe._so_cuda_major = lambda _p: so_major
    try:
        qfe.assert_toolchain_compatible("/nonexistent.so")
        return False
    except RuntimeError:
        return True
    finally:
        qfe._is_elf, qfe._so_cuda_major = o_is_elf, o_major
        os.environ.pop(qfe._ENV_ALLOW_UNVERIFIED_TOOLCHAIN, None)


# (label, torch_cuda, so_cudart_major, override, is_elf, EXPECT_REFUSE)
_CASES = [
    ("ELF match torch13/.so13 -> LOAD",       "13.0", 13,   False, True,  False),
    ("ELF mismatch torch12/.so13 -> REFUSE",  "12.4", 13,   False, True,  True),
    ("ELF static cudart -> REFUSE",           "13.0", None, False, True,  True),
    ("no torch CUDA (CPU torch) -> REFUSE",   None,   13,   False, True,  True),
    ("override on mismatch -> LOAD",          "12.4", 13,   True,  True,  False),
    ("neither ELF nor PE (macOS) -> REFUSE",  "13.0", 13,   False, False, True),
]


def mk_pe(imports, delay=()):
    """A minimal PE32+ image (x64): DOS + PE headers and ONE section holding the import table, the delay-load table and
    the DLL names — what an import reader needs, nothing more."""
    import struct
    rva0, off0 = 0x1000, 0x200
    imp = 20 * (len(imports) + 1)                       # IMAGE_IMPORT_DESCRIPTOR x n + the zero terminator
    dly = 32 * (len(delay) + 1) if delay else 0          # IMAGE_DELAYLOAD_DESCRIPTOR x n + the zero terminator
    strings, at = b"", {}
    for n in (*imports, *delay):
        at[n] = rva0 + imp + dly + len(strings)
        strings += n.encode() + b"\0"
    sec = bytearray(imp + dly) + strings
    for i, n in enumerate(imports):
        struct.pack_into("<I", sec, 20 * i + 12, at[n])                   # Name RVA
    for i, n in enumerate(delay):
        struct.pack_into("<II", sec, imp + 32 * i, 1, at[n])              # Attributes (RVA-based), DllNameRVA
    head = bytearray(off0)
    head[0:2] = b"MZ"
    struct.pack_into("<I", head, 0x3C, 0x40)
    head[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", head, 0x44, 0x8664, 1, 0, 0, 0, 240, 0x2022)   # one section, PE32+ optional header
    opt = 0x44 + 20
    struct.pack_into("<H", head, opt, 0x20B)
    struct.pack_into("<I", head, opt + 108, 16)                          # NumberOfRvaAndSizes
    struct.pack_into("<II", head, opt + 112 + 8 * 1, rva0, imp)          # data directory 1: imports
    if delay:
        struct.pack_into("<II", head, opt + 112 + 8 * 13, rva0 + imp, dly)   # data directory 13: delay-load imports
    sh = opt + 240
    head[sh:sh + 8] = b".idata\0\0"
    struct.pack_into("<IIII", head, sh + 8, len(sec), rva0, len(sec), off0)
    return bytes(head) + bytes(sec)


def _pe_arms():
    """(D) Windows DLLs: the REAL PE import reader and the REAL guard (no monkeypatching): the CUDA major comes from the
    imported cudart64_N.dll, else cuBLAS's DLL name (a static-cudart DLL), from the import or the delay-load table."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="qf_pe_")
    bad = 0

    def pe(name, imports, delay=()):
        p = os.path.join(tmp, name)
        open(p, "wb").write(mk_pe(imports, delay))
        return p

    majors = {
        "cudart 12": (pe("a.dll", ["KERNEL32.dll", "cudart64_12.dll", "cublas64_12.dll"]), 12),
        "static cudart, cuBLAS 13": (pe("b.dll", ["cublas64_13.dll", "cublasLt64_13.dll", "cudnn64_9.dll"]), 13),
        "cudart decides over cuBLAS": (pe("c.dll", ["cudart64_12.dll", "cublas64_13.dll"]), 12),
        "two cudart majors": (pe("d.dll", ["cudart64_12.dll", "cudart64_13.dll"]), None),
        "no CUDA import": (pe("e.dll", ["KERNEL32.dll", "MSVCP140.dll"]), None),
        "delay-loaded cudart 13": (pe("f.dll", ["KERNEL32.dll"], delay=["cudart64_13.dll"]), 13),
    }
    for label, (p, want) in majors.items():
        got = qfe._so_cuda_major(p)
        ok = got == want and qfe._is_pe(p) and not qfe._is_elf(p)
        print(f"  [{'OK ' if ok else 'FAIL'}] PE {label}: CUDA major {got} (expected {want})")
        bad += not ok
    imports = qfe._pe_imports(majors["delay-loaded cudart 13"][0])
    ok = imports == ["kernel32.dll", "cudart64_13.dll"]
    print(f"  [{'OK ' if ok else 'FAIL'}] PE import reader: import + delay-load names, lowercased -> {imports}")
    bad += not ok
    trunc = os.path.join(tmp, "trunc.dll")
    open(trunc, "wb").write(mk_pe(["cudart64_12.dll"])[:0x150])        # headers cut: the reader must not guess
    ok = qfe._pe_imports(trunc) == [] and qfe._so_cuda_major(trunc) is None
    print(f"  [{'OK ' if ok else 'FAIL'}] PE truncated image: no imports, CUDA major undeterminable")
    bad += not ok
    os.environ.pop(qfe._ENV_ALLOW_UNVERIFIED_TOOLCHAIN, None)
    verdicts = {}
    for label, torch_cuda, path in (("torch 12 + cudart64_12 DLL -> LOAD", "12.8", majors["cudart 12"][0]),
                                    ("torch 13 + static-cudart cuBLAS-13 DLL -> LOAD", "13.0",
                                     majors["static cudart, cuBLAS 13"][0]),
                                    ("torch 13 + cudart64_12 DLL -> REFUSE (mismatch)", "13.0", majors["cudart 12"][0]),
                                    ("torch 12 + DLL without CUDA imports -> REFUSE", "12.8", majors["no CUDA import"][0]),
                                    ("torch 12 + truncated DLL -> REFUSE", "12.8", trunc)):
        _set_fake_torch(torch_cuda)
        try:
            qfe.assert_toolchain_compatible(path)
            verdicts[label] = "LOAD"
        except RuntimeError as e:
            verdicts[label] = "REFUSE (mismatch)" if "MISMATCH" in str(e) else "REFUSE"
        ok = label.endswith("-> " + verdicts[label])
        print(f"  [{'OK ' if ok else 'FAIL'}] guard on a real PE: {label}: got {verdicts[label]}")
        bad += not ok
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return bad


def _wiring_ok():
    """(B) Prove load_lib() calls the guard BEFORE ctypes.CDLL — through the PRODUCTION entry point. A
    de-wiring that removes the assert_toolchain_compatible CALL is invisible to the decision-logic cases
    above (they call the guard directly). Monkeypatch load_lib's deps and record the call order."""
    calls = []
    o_assert, o_cdll, o_resolve, o_bind, o_lib = (qfe.assert_toolchain_compatible, qfe.ctypes.CDLL,
                                                  qfe.resolve_so_path, qfe._bind, qfe._LIB)
    qfe._LIB = None
    qfe.resolve_so_path = lambda: "/fake/engine.so"
    qfe.assert_toolchain_compatible = lambda _p: calls.append("guard")
    qfe.ctypes.CDLL = lambda _p, **_k: (calls.append("dlopen"), object())[1]
    qfe._bind = lambda lib: lib
    try:
        qfe.load_lib()
    except Exception:  # noqa: BLE001
        pass
    finally:
        qfe.assert_toolchain_compatible, qfe.ctypes.CDLL = o_assert, o_cdll
        qfe.resolve_so_path, qfe._bind, qfe._LIB = o_resolve, o_bind, None
    if calls[:2] != ["guard", "dlopen"]:
        print(f"  [FAIL] wiring: load_lib() call order was {calls} (guard MUST precede dlopen)")
        return False
    print("  [OK ] wiring: load_lib() calls the toolchain guard BEFORE ctypes.CDLL")
    return True


def _parser_smoke():
    """(C) The decision cases monkeypatch _so_cuda_major, so they never exercise the REAL ELF parser.
    Prove the parser extracts DT_NEEDED from a genuine ELF (a stdlib extension .so). SKIP (not fail) if no
    ELF is locatable (non-ELF platform)."""
    import _ctypes  # a CPython extension module -> a real ELF .so on Linux
    p = getattr(_ctypes, "__file__", "")
    if not p or not os.path.isfile(p):
        print("  [SKIP] parser smoke: no locatable extension .so")
        return True
    needed = qfe._elf_needed(p)
    with open(p, "rb") as f:
        is_elf = f.read(4) == b"\x7fELF"
    if not isinstance(needed, list) or (is_elf and not needed):
        print(f"  [FAIL] parser smoke: real ELF {p} yielded {needed!r}")
        return False
    if is_elf and not qfe._is_elf(p):
        print(f"  [FAIL] parser smoke: _is_elf() wrongly rejected a real ELF {p}")
        return False
    print(f"  [OK ] parser smoke: _elf_needed({os.path.basename(p)}) -> {len(needed)} NEEDED; _is_elf ok")
    return True


def main():
    bad = 0
    verdicts = []
    for label, tc, sm, ov, elf, expect in _CASES:
        got = _refuses(tc, sm, ov, elf)
        ok = got == expect
        verdicts.append(got)
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}: refused={got} (expected {expect})")
        bad += 0 if ok else 1
    if len(set(verdicts)) < 2:  # a guard whose verdict never varies is decoration
        print("  [FAIL] guard verdict is CONSTANT across all states -> not discriminating")
        bad += 1
    if not _wiring_ok():
        bad += 1
    if not _parser_smoke():
        bad += 1
    bad += _pe_arms()
    print("TOOLCHAIN_GUARD:", "PASS" if bad == 0 else f"FAIL ({bad} wrong)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
