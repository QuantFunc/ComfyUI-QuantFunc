#!/usr/bin/env python3
"""Death-rule for the fail-closed CUDA-toolchain guard (qf_engine.assert_toolchain_compatible).

QuantFunc PC Engine O1 invariant:
  - Host libquantfunc.so has ZERO CUDA DT_NEEDED imports (no libcudart / libcublas / libcublasLt).
  - CUDA lives exclusively in the selected per-SM kernel plugin
    (libquantfunc_kernels_cuda_sm{120,89,86,75}.so), whose linked CUDA major MUST match torch.

This test asserts:
  (A) each toolchain state produces its OWN verdict (load vs refuse), including zero-CUDA host check,
      per-SM plugin discovery, CUDA major match/mismatch, missing plugin, static cudart, CPU torch,
      override, and non-ELF/PE;
  (B) load_lib() actually CALLS the guard BEFORE ctypes.CDLL (production entry point);
  (C) the real ELF DT_NEEDED parser extracts NEEDED from a genuine binary, and _select_kernel_plugin
      discovers a real per-SM kernel plugin beside libquantfunc.so;
  (D) Windows: the real PE import reader and the real guard on built PE32+ images.

Run: python3 toolchain_guard_test.py   (exit 0 = all correct; exit 1 = a check is wrong)
"""
import os
import shutil
import sys
import tempfile
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import qf_engine as qfe  # noqa: E402


def _set_fake_torch(cuda):
    m = types.ModuleType("torch")
    m.version = types.SimpleNamespace(cuda=cuda)
    sys.modules["torch"] = m


def _refuses(torch_cuda, so_major, override=False, is_elf=True, host_cuda=(), has_plugin=True):
    """True iff assert_toolchain_compatible RAISES for the given toolchain state."""
    if override:
        os.environ[qfe._ENV_ALLOW_UNVERIFIED_TOOLCHAIN] = "1"
    else:
        os.environ.pop(qfe._ENV_ALLOW_UNVERIFIED_TOOLCHAIN, None)
    _set_fake_torch(torch_cuda)
    saved = (qfe._is_elf, qfe._is_pe, qfe._host_forbidden_cuda, qfe._select_kernel_plugin, qfe._so_cuda_major)
    qfe._is_elf = lambda _p: is_elf
    qfe._is_pe = lambda _p: False
    qfe._host_forbidden_cuda = lambda _p: list(host_cuda)
    qfe._select_kernel_plugin = (lambda _p: "/fake/libquantfunc_kernels_cuda_sm89.so") if has_plugin else (lambda _p: None)
    qfe._so_cuda_major = lambda _p: so_major
    try:
        qfe.assert_toolchain_compatible("/nonexistent/libquantfunc.so")
        return False
    except RuntimeError:
        return True
    finally:
        qfe._is_elf, qfe._is_pe, qfe._host_forbidden_cuda, qfe._select_kernel_plugin, qfe._so_cuda_major = saved
        os.environ.pop(qfe._ENV_ALLOW_UNVERIFIED_TOOLCHAIN, None)


# (label, torch_cuda, plugin_cuda_major, override, is_elf, host_cuda, has_plugin, EXPECT_REFUSE)
_CASES = [
    ("ELF zero-CUDA host + plugin match torch13/sm89-13 -> LOAD",    "13.0", 13,   False, True,  (),                 True,  False),
    ("ELF zero-CUDA host + plugin mismatch torch12/sm89-13 -> REFUSE", "12.4", 13, False, True,  (),                 True,  True),
    ("ELF zero-CUDA host + plugin static cudart -> REFUSE",          "13.0", None, False, True,  (),                 True,  True),
    ("ELF host leaks libcudart -> REFUSE",                           "13.0", 13,   False, True,  ("libcudart.so.13",), True, True),
    ("ELF host leaks libcublas -> REFUSE",                           "13.0", 13,   False, True,  ("libcublas.so.13",), True, True),
    ("ELF zero-CUDA host + missing per-SM plugin -> REFUSE",         "13.0", 13,   False, True,  (),                 False, True),
    ("no torch CUDA (CPU torch) -> REFUSE",                          None,   13,   False, True,  (),                 True,  True),
    ("override on mismatch -> LOAD",                                 "12.4", 13,   True,  True,  (),                 True,  False),
    ("neither ELF nor PE (macOS) -> REFUSE",                         "13.0", 13,   False, False, (),                 True,  True),
]


def mk_pe(imports, delay=()):
    """Minimal PE32+ image (x64) with import and optional delay-load tables."""
    import struct
    rva0, off0 = 0x1000, 0x200
    imp = 20 * (len(imports) + 1)
    dly = 32 * (len(delay) + 1) if delay else 0
    strings, at = b"", {}
    for n in (*imports, *delay):
        at[n] = rva0 + imp + dly + len(strings)
        strings += n.encode() + b"\0"
    sec = bytearray(imp + dly) + strings
    for i, n in enumerate(imports):
        struct.pack_into("<I", sec, 20 * i + 12, at[n])
    for i, n in enumerate(delay):
        struct.pack_into("<II", sec, imp + 32 * i, 1, at[n])
    head = bytearray(off0)
    head[0:2] = b"MZ"
    struct.pack_into("<I", head, 0x3C, 0x40)
    head[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", head, 0x44, 0x8664, 1, 0, 0, 0, 240, 0x2022)
    opt = 0x44 + 20
    struct.pack_into("<H", head, opt, 0x20B)
    struct.pack_into("<I", head, opt + 108, 16)
    struct.pack_into("<II", head, opt + 112 + 8 * 1, rva0, imp)
    if delay:
        struct.pack_into("<II", head, opt + 112 + 8 * 13, rva0 + imp, dly)
    sh = opt + 240
    head[sh:sh + 8] = b".idata\0\0"
    struct.pack_into("<IIII", head, sh + 8, len(sec), rva0, len(sec), off0)
    return bytes(head) + bytes(sec)


def _pe_arms():
    """(D) Windows DLLs: real PE import reader + zero-CUDA host quantfunc.dll + per-SM kernel DLL guard."""
    tmp = tempfile.mkdtemp(prefix="qf_pe_")
    bad = 0

    def pe(sub, name, imports, delay=()):
        d = os.path.join(tmp, sub)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        with open(p, "wb") as f:
            f.write(mk_pe(imports, delay))
        return p

    majors = {
        "cudart 12": (pe("parse", "a.dll", ["KERNEL32.dll", "cudart64_12.dll", "cublas64_12.dll"]), 12),
        "static cudart, cuBLAS 13": (pe("parse", "b.dll", ["cublas64_13.dll", "cublasLt64_13.dll", "cudnn64_9.dll"]), 13),
        "cudart decides over cuBLAS": (pe("parse", "c.dll", ["cudart64_12.dll", "cublas64_13.dll"]), 12),
        "two cudart majors": (pe("parse", "d.dll", ["cudart64_12.dll", "cudart64_13.dll"]), None),
        "no CUDA import": (pe("parse", "e.dll", ["KERNEL32.dll", "MSVCP140.dll"]), None),
        "delay-loaded cudart 13 (2nd delay entry)": (
            pe("parse", "f.dll", ["KERNEL32.dll"], delay=["nvcuda.dll", "cudart64_13.dll"]), 13),
    }
    for label, (p, want) in majors.items():
        got = qfe._so_cuda_major(p)
        ok = got == want and qfe._is_pe(p) and not qfe._is_elf(p)
        print(f"  [{'OK ' if ok else 'FAIL'}] PE {label}: CUDA major {got} (expected {want})")
        bad += not ok
    imports = qfe._pe_imports(majors["delay-loaded cudart 13 (2nd delay entry)"][0])
    ok = imports == ["kernel32.dll", "nvcuda.dll", "cudart64_13.dll"]
    print(f"  [{'OK ' if ok else 'FAIL'}] PE import reader: import + delay-load names, lowercased -> {imports}")
    bad += not ok
    ok = (qfe._host_forbidden_cuda_pe(majors["no CUDA import"][0]) == []
          and qfe._host_forbidden_cuda_pe(majors["cudart 12"][0]) == ["cudart64_12.dll", "cublas64_12.dll"])
    print(f"  [{'OK ' if ok else 'FAIL'}] PE zero-CUDA host detector: clean host=[] and CUDA host flagged")
    bad += not ok
    trunc = os.path.join(tmp, "parse", "trunc.dll")
    with open(trunc, "wb") as f:
        f.write(mk_pe(["cudart64_12.dll"])[:0x150])
    ok = qfe._pe_imports(trunc) == [] and qfe._so_cuda_major(trunc) is None
    print(f"  [{'OK ' if ok else 'FAIL'}] PE truncated image: no imports, CUDA major undeterminable")
    bad += not ok

    # Real Windows dual-file layout: zero-CUDA quantfunc.dll + libquantfunc_kernels_cuda_sm89.dll
    clean_host = ["KERNEL32.dll", "MSVCP140.dll"]
    h_cu12 = pe("win_cu12", "quantfunc.dll", clean_host)
    pe("win_cu12", "libquantfunc_kernels_cuda_sm89.dll", ["KERNEL32.dll", "cudart64_12.dll", "cublas64_12.dll"])
    h_cu13 = pe("win_cu13", "quantfunc.dll", clean_host)
    pe("win_cu13", "libquantfunc_kernels_cuda_sm89.dll", ["cublas64_13.dll", "cublasLt64_13.dll", "cudnn64_9.dll"])
    h_leak_rt = pe("win_leak_rt", "quantfunc.dll", ["KERNEL32.dll", "cudart64_12.dll"])
    pe("win_leak_rt", "libquantfunc_kernels_cuda_sm89.dll", ["cudart64_12.dll"])
    h_leak_bl = pe("win_leak_bl", "quantfunc.dll", ["KERNEL32.dll", "cublas64_13.dll"])
    pe("win_leak_bl", "libquantfunc_kernels_cuda_sm89.dll", ["cublas64_13.dll"])
    h_no_plug = pe("win_no_plug", "quantfunc.dll", clean_host)
    h_wrong_sm = pe("win_wrong_sm", "quantfunc.dll", clean_host)
    pe("win_wrong_sm", "libquantfunc_kernels_cuda_sm75.dll", ["cudart64_12.dll"])
    h_nocuda_plug = pe("win_nocuda_plug", "quantfunc.dll", clean_host)
    pe("win_nocuda_plug", "libquantfunc_kernels_cuda_sm89.dll", clean_host)
    h_trunc_plug = pe("win_trunc_plug", "quantfunc.dll", clean_host)
    with open(os.path.join(tmp, "win_trunc_plug", "libquantfunc_kernels_cuda_sm89.dll"), "wb") as f:
        f.write(mk_pe(["cudart64_12.dll"])[:0x150])

    os.environ.pop(qfe._ENV_ALLOW_UNVERIFIED_TOOLCHAIN, None)
    o_sm = qfe._gpu_sm
    qfe._gpu_sm = lambda _i=0: 89
    try:
        cases = (
            ("Win zero-CUDA host + sm89 cudart64_12 DLL -> LOAD", "12.8", h_cu12),
            ("Win zero-CUDA host + sm89 static-cudart cuBLAS-13 DLL -> LOAD", "13.0", h_cu13),
            ("Win zero-CUDA host + sm89 cudart64_12 DLL vs torch 13 -> REFUSE (mismatch)", "13.0", h_cu12),
            ("Win host leaks cudart64_12 -> REFUSE", "12.8", h_leak_rt),
            ("Win host leaks cublas64_13 -> REFUSE", "13.0", h_leak_bl),
            ("Win zero-CUDA host + missing per-SM DLL -> REFUSE", "12.8", h_no_plug),
            ("Win zero-CUDA host + only sm75 DLL when GPU is sm89 -> REFUSE", "12.8", h_wrong_sm),
            ("Win zero-CUDA host + plugin without CUDA imports -> REFUSE", "12.8", h_nocuda_plug),
            ("Win zero-CUDA host + truncated plugin DLL -> REFUSE", "12.8", h_trunc_plug),
        )
        for label, torch_cuda, path in cases:
            _set_fake_torch(torch_cuda)
            try:
                qfe.assert_toolchain_compatible(path)
                got_v = "LOAD"
            except RuntimeError as e:
                got_v = "REFUSE (mismatch)" if "MISMATCH" in str(e) else "REFUSE"
            ok = label.endswith("-> " + got_v)
            print(f"  [{'OK ' if ok else 'FAIL'}] guard on real PE: {label}: got {got_v}")
            bad += not ok
    finally:
        qfe._gpu_sm = o_sm
        shutil.rmtree(tmp, ignore_errors=True)
    return bad


def _wiring_ok():
    """(B) Prove load_lib() calls the guard BEFORE ctypes.CDLL."""
    calls = []
    o_assert, o_cdll, o_resolve, o_bind, o_lib = (qfe.assert_toolchain_compatible, qfe.ctypes.CDLL,
                                                  qfe.resolve_so_path, qfe._bind, qfe._LIB)
    qfe._LIB = None
    qfe.resolve_so_path = lambda: "/fake/libquantfunc.so"
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
    """(C) Real ELF parser + per-SM kernel plugin selection beside a zero-CUDA host .so."""
    import _ctypes
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
    if is_elf and qfe._host_forbidden_cuda(p):
        print(f"  [FAIL] parser smoke: non-CUDA ELF {p} flagged forbidden CUDA")
        return False
    tmp = tempfile.mkdtemp(prefix="qf_plug_")
    try:
        d_multi = os.path.join(tmp, "multi")
        d_only75 = os.path.join(tmp, "only75")
        os.makedirs(d_multi)
        os.makedirs(d_only75)
        host = os.path.join(d_multi, "libquantfunc.so")
        sm89 = os.path.join(d_multi, "libquantfunc_kernels_cuda_sm89.so")
        sm120 = os.path.join(d_multi, "libquantfunc_kernels_cuda_sm120.so")
        for fp in (host, sm89, sm120):
            open(fp, "wb").close()
        host75 = os.path.join(d_only75, "libquantfunc.so")
        sm75 = os.path.join(d_only75, "libquantfunc_kernels_cuda_sm75.so")
        for fp in (host75, sm75):
            open(fp, "wb").close()
        o_sm, o_maj = qfe._gpu_sm, qfe._torch_cuda_major
        try:
            qfe._gpu_sm, qfe._torch_cuda_major = (lambda _i=0: 89), (lambda: 13)
            picked = qfe._select_kernel_plugin(host)
            wrong_sm_picked = qfe._select_kernel_plugin(host75)
            qfe._gpu_sm = lambda _i=0: None
            single_unknown_sm = qfe._select_kernel_plugin(host75)
            multi_unknown_sm = qfe._select_kernel_plugin(host)
        finally:
            qfe._gpu_sm, qfe._torch_cuda_major = o_sm, o_maj
        if picked != sm89:
            print(f"  [FAIL] plugin select: expected {sm89}, got {picked!r}")
            return False
        if wrong_sm_picked is not None:
            print(f"  [FAIL] plugin select: known sm=89 wrongly fell back to {wrong_sm_picked!r}")
            return False
        if single_unknown_sm != sm75 or multi_unknown_sm is not None:
            print(f"  [FAIL] plugin select (sm=None): single={single_unknown_sm!r} multi={multi_unknown_sm!r}")
            return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"  [OK ] parser smoke: _elf_needed({os.path.basename(p)}) -> {len(needed)} NEEDED; per-SM plugin select & mismatch refusal ok")
    return True


def main():
    bad = 0
    verdicts = []
    for label, tc, sm, ov, elf, host_cuda, has_plug, expect in _CASES:
        got = _refuses(tc, sm, ov, elf, host_cuda, has_plug)
        ok = got == expect
        verdicts.append(got)
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}: refused={got} (expected {expect})")
        bad += 0 if ok else 1
    if len(set(verdicts)) < 2:
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
