#!/usr/bin/env python3
"""Death-rule for the fail-closed CUDA-toolchain guard (qf_engine.assert_toolchain_compatible).

QuantFunc PC Engine O1 invariant:
  - Host libquantfunc.so has ZERO CUDA DT_NEEDED imports (no libcudart / libcublas / libcublasLt).
  - CUDA lives exclusively in the selected per-SM kernel plugin
    (libquantfunc_kernels.so / libquantfunc_kernels-12.so), whose linked CUDA major MUST match torch.

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
    qfe._select_kernel_plugin = (lambda _p: "/fake/libquantfunc_kernels.so") if has_plugin else (lambda _p: None)
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

    # Real Windows dual-file layout: zero-CUDA quantfunc(-12).dll + libquantfunc_kernels(-12).dll
    clean_host = ["KERNEL32.dll", "MSVCP140.dll"]
    h_cu12 = pe("win_cu12", "quantfunc-12.dll", clean_host)
    pe("win_cu12", "libquantfunc_kernels-12.dll", ["KERNEL32.dll", "cudart64_12.dll", "cublas64_12.dll"])
    h_cu13 = pe("win_cu13", "quantfunc.dll", clean_host)
    pe("win_cu13", "libquantfunc_kernels.dll", ["cublas64_13.dll", "cublasLt64_13.dll", "cudnn64_9.dll"])
    h_cu13_bad_kern = pe("win_cu13_bad_kern", "quantfunc.dll", clean_host)
    pe("win_cu13_bad_kern", "libquantfunc_kernels.dll", ["KERNEL32.dll", "cudart64_12.dll", "cublas64_12.dll"])
    h_splitbrain = pe("win_splitbrain", "quantfunc.dll", clean_host)
    pe("win_splitbrain", "libquantfunc_kernels.dll", ["cublas64_13.dll"])
    pe("win_splitbrain", "libquantfunc_kernels-12.dll", ["cudart64_12.dll"])
    h_only_12_for_13 = pe("win_only_12_for_13", "quantfunc.dll", clean_host)
    pe("win_only_12_for_13", "libquantfunc_kernels-12.dll", ["cudart64_12.dll"])
    h_leak_rt = pe("win_leak_rt", "quantfunc-12.dll", ["KERNEL32.dll", "cudart64_12.dll"])
    pe("win_leak_rt", "libquantfunc_kernels-12.dll", ["cudart64_12.dll"])
    h_leak_bl = pe("win_leak_bl", "quantfunc.dll", ["KERNEL32.dll", "cublas64_13.dll"])
    pe("win_leak_bl", "libquantfunc_kernels.dll", ["cublas64_13.dll"])
    h_no_plug = pe("win_no_plug", "quantfunc-12.dll", clean_host)
    h_wrong_cu = pe("win_wrong_cu", "quantfunc-12.dll", clean_host)
    pe("win_wrong_cu", "libquantfunc_kernels.dll", ["cudart64_13.dll"])
    h_legacy_name = pe("win_legacy_name", "quantfunc.dll", clean_host)
    pe("win_legacy_name", "libquantfunc_kernels_cuda_sm89.dll", ["cudart64_13.dll"])
    h_nocuda_plug = pe("win_nocuda_plug", "quantfunc-12.dll", clean_host)
    pe("win_nocuda_plug", "libquantfunc_kernels-12.dll", clean_host)
    h_trunc_plug = pe("win_trunc_plug", "quantfunc-12.dll", clean_host)
    with open(os.path.join(tmp, "win_trunc_plug", "libquantfunc_kernels-12.dll"), "wb") as f:
        f.write(mk_pe(["cudart64_12.dll"])[:0x150])

    os.environ.pop(qfe._ENV_ALLOW_UNVERIFIED_TOOLCHAIN, None)
    o_sm = qfe._gpu_sm
    qfe._gpu_sm = lambda _i=0: 89
    try:
        cases = (
            ("Win zero-CUDA host + libquantfunc_kernels-12.dll (cudart64_12) -> LOAD", "12.8", h_cu12),
            ("Win zero-CUDA host + libquantfunc_kernels.dll (static-cudart cuBLAS-13) -> LOAD", "13.0", h_cu13),
            ("Win zero-CUDA host + libquantfunc_kernels.dll (cudart64_12) vs torch 13 -> REFUSE (mismatch)", "13.0", h_cu13_bad_kern),
            ("Win quantfunc-12.dll + libquantfunc_kernels-12.dll vs torch 13 -> REFUSE (mismatch)", "13.0", h_cu12),
            ("Win quantfunc.dll + both DLLs vs torch 12 (no split-brain) -> REFUSE (mismatch)", "12.8", h_splitbrain),
            ("Win quantfunc.dll + only -12 DLL vs torch 12 -> REFUSE", "12.8", h_only_12_for_13),
            ("Win host leaks cudart64_12 -> REFUSE", "12.8", h_leak_rt),
            ("Win host leaks cublas64_13 -> REFUSE", "13.0", h_leak_bl),
            ("Win zero-CUDA host + missing kernel DLL -> REFUSE", "12.8", h_no_plug),
            ("Win quantfunc-12.dll + only cu13 DLL vs torch 13 -> REFUSE", "13.0", h_wrong_cu),
            ("Win zero-CUDA host + legacy _cuda_sm89 DLL name -> REFUSE", "13.0", h_legacy_name),
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
        d_only12 = os.path.join(tmp, "only12")
        d_only13 = os.path.join(tmp, "only13")
        d_legacy = os.path.join(tmp, "legacy")
        d_env = os.path.join(tmp, "envdir")
        for d in (d_multi, d_only12, d_only13, d_legacy, d_env):
            os.makedirs(d)
        host = os.path.join(d_multi, "libquantfunc.so")
        host12_m = os.path.join(d_multi, "libquantfunc-12.so")
        k13 = os.path.join(d_multi, "libquantfunc_kernels.so")
        k12 = os.path.join(d_multi, "libquantfunc_kernels-12.so")
        for fp in (host, host12_m, k13, k12):
            open(fp, "wb").close()
        host13_on_12 = os.path.join(d_only12, "libquantfunc.so")
        host12 = os.path.join(d_only12, "libquantfunc-12.so")
        only12 = os.path.join(d_only12, "libquantfunc_kernels-12.so")
        for fp in (host13_on_12, host12, only12):
            open(fp, "wb").close()
        host12_on_13 = os.path.join(d_only13, "libquantfunc-12.so")
        only13 = os.path.join(d_only13, "libquantfunc_kernels.so")
        for fp in (host12_on_13, only13):
            open(fp, "wb").close()
        host_leg = os.path.join(d_legacy, "libquantfunc.so")
        open(host_leg, "wb").close()
        open(os.path.join(d_legacy, "libquantfunc_kernels_cuda_sm89.so"), "wb").close()
        env_k13 = os.path.join(d_env, "libquantfunc_kernels.so")
        env_k12 = os.path.join(d_env, "libquantfunc_kernels-12.so")
        for fp in (env_k13, env_k12):
            open(fp, "wb").close()
        o_sm, o_maj, o_env = qfe._gpu_sm, qfe._torch_cuda_major, os.environ.pop(qfe._ENV_KERNEL_PLUGIN_DIR, None)
        try:
            qfe._gpu_sm = lambda _i=0: 89
            # Host flavor strictly selects matching kernel regardless of torch CUDA major (no split-brain)
            for maj in (13, 12, None):
                qfe._torch_cuda_major = (lambda m=maj: m)
                if qfe._select_kernel_plugin(host) != k13 or qfe._select_kernel_plugin(host12_m) != k12:
                    print(f"  [FAIL] plugin select (torch={maj}): strict host pairing broke")
                    return False
            if (qfe._select_kernel_plugin(host13_on_12) is not None
                    or qfe._select_kernel_plugin(host12_on_13) is not None
                    or qfe._select_kernel_plugin(host_leg) is not None):
                print("  [FAIL] plugin select: cross-flavor or legacy kernel was wrongly admitted")
                return False
            os.environ[qfe._ENV_KERNEL_PLUGIN_DIR] = d_env
            if qfe._select_kernel_plugin(host) != env_k13 or qfe._select_kernel_plugin(host12_m) != env_k12:
                print("  [FAIL] plugin select: QF_KERNEL_PLUGIN_DIR did not take precedence with strict pairing")
                return False
        finally:
            qfe._gpu_sm, qfe._torch_cuda_major = o_sm, o_maj
            os.environ.pop(qfe._ENV_KERNEL_PLUGIN_DIR, None)
            if o_env is not None:
                os.environ[qfe._ENV_KERNEL_PLUGIN_DIR] = o_env
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"  [OK ] parser smoke: _elf_needed({os.path.basename(p)}) -> {len(needed)} NEEDED; strict host-kernel pairing & QF_KERNEL_PLUGIN_DIR precedence ok")
    return True


def mk_elf_needed(needed):
    """Minimal ELF64 shared object carrying real .dynstr and .dynamic DT_NEEDED entries."""
    import struct
    dynstr = b"\0" + b"".join(s.encode("ascii") + b"\0" for s in needed)
    offs, at = [], 1
    for s in needed:
        offs.append(at)
        at += len(s) + 1
    dynamic = b"".join(struct.pack("<qQ", 1, off) for off in offs) + struct.pack("<qQ", 0, 0)
    shstrtab = b"\0.dynstr\0.dynamic\0.shstrtab\0"
    secs = [
        (0, 0, 0, 0, 0),
        (1, 3, 64, len(dynstr), 0),
        (9, 6, 64 + len(dynstr), len(dynamic), 1),
        (18, 3, 64 + len(dynstr) + len(dynamic), len(shstrtab), 0),
    ]
    head = bytearray(64)
    head[0:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<Q", head, 0x28, 64 + len(dynstr) + len(dynamic) + len(shstrtab))
    struct.pack_into("<HHH", head, 0x3a, 64, len(secs), 3)
    shdrs = b""
    for name_off, sh_type, sh_off, sh_sz, sh_link in secs:
        sh = bytearray(64)
        struct.pack_into("<II", sh, 0, name_off, sh_type)
        struct.pack_into("<QQI", sh, 0x18, sh_off, sh_sz, sh_link)
        shdrs += bytes(sh)
    return bytes(head) + dynstr + dynamic + shstrtab + shdrs


def _elf_arms():
    """Real ELF64 DT_NEEDED pairs through assert_toolchain_compatible, QF_KERNEL_PLUGIN_DIR override & .dev_lib_lock."""
    import hashlib
    import json
    tmp = tempfile.mkdtemp(prefix="qf_elf_")
    bad = 0

    def elf(sub, name, needed):
        d = os.path.join(tmp, sub)
        os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, name)
        with open(fp, "wb") as f:
            f.write(mk_elf_needed(needed))
        return fp

    clean = ["libstdc++.so.6", "libc.so.6"]
    h13 = elf("cu13", "libquantfunc.so", clean)
    elf("cu13", "libquantfunc_kernels.so", ["libcudart.so.13"])
    h12 = elf("cu12", "libquantfunc-12.so", clean)
    elf("cu12", "libquantfunc_kernels-12.so", ["libcudart.so.12"])
    h_split13 = elf("split13", "libquantfunc.so", clean)
    elf("split13", "libquantfunc_kernels.so", ["libcudart.so.13"])
    elf("split13", "libquantfunc_kernels-12.so", ["libcudart.so.12"])
    h_split12 = elf("split12", "libquantfunc-12.so", clean)
    elf("split12", "libquantfunc_kernels.so", ["libcudart.so.13"])
    elf("split12", "libquantfunc_kernels-12.so", ["libcudart.so.12"])
    h_cross12 = elf("cross12", "libquantfunc-12.so", clean)
    elf("cross12", "libquantfunc_kernels.so", ["libcudart.so.13"])
    h_cross13 = elf("cross13", "libquantfunc.so", clean)
    elf("cross13", "libquantfunc_kernels-12.so", ["libcudart.so.12"])
    h_leak13 = elf("leak13", "libquantfunc.so", ["libcudart.so.13"])
    elf("leak13", "libquantfunc_kernels.so", ["libcudart.so.13"])
    h_leak12 = elf("leak12", "libquantfunc-12.so", ["libcudart.so.12"])
    elf("leak12", "libquantfunc_kernels-12.so", ["libcudart.so.12"])

    os.environ.pop(qfe._ENV_ALLOW_UNVERIFIED_TOOLCHAIN, None)
    o_env = os.environ.pop(qfe._ENV_KERNEL_PLUGIN_DIR, None)
    try:
        cases = (
            ("ELF real cu13 pair -> LOAD", "13.0", h13),
            ("ELF real cu12 pair -> LOAD", "12.8", h12),
            ("ELF libquantfunc.so + both kernels vs torch 12 (no split-brain) -> REFUSE (mismatch)", "12.8", h_split13),
            ("ELF libquantfunc-12.so + both kernels vs torch 13 (no split-brain) -> REFUSE (mismatch)", "13.0", h_split12),
            ("ELF libquantfunc-12.so + only libquantfunc_kernels.so vs torch 13 -> REFUSE", "13.0", h_cross12),
            ("ELF libquantfunc.so + only libquantfunc_kernels-12.so vs torch 12 -> REFUSE", "12.8", h_cross13),
            ("ELF libquantfunc.so leaks libcudart.so.13 -> REFUSE", "13.0", h_leak13),
            ("ELF libquantfunc-12.so leaks libcudart.so.12 -> REFUSE", "12.8", h_leak12),
        )
        for label, torch_cuda, path in cases:
            _set_fake_torch(torch_cuda)
            try:
                qfe.assert_toolchain_compatible(path)
                got_v = "LOAD"
            except RuntimeError as e:
                got_v = "REFUSE (mismatch)" if "MISMATCH" in str(e) else "REFUSE"
            ok = label.endswith("-> " + got_v)
            print(f"  [{'OK ' if ok else 'FAIL'}] {label}: got {got_v}")
            bad += not ok

        # QF_KERNEL_PLUGIN_DIR must take precedence even over an installed pair marker
        bin_dir = os.path.join(tmp, "marked_bin")
        pair_dir = os.path.join(bin_dir, "0.0.18-sm89-cu13")
        os.makedirs(pair_dir)
        mh = elf(os.path.join("marked_bin", "0.0.18-sm89-cu13"), "libquantfunc.so", clean)
        mk = elf(os.path.join("marked_bin", "0.0.18-sm89-cu13"), "libquantfunc_kernels.so", ["libcudart.so.13"])
        marker = {
            "version": "0.0.18", "set": "sm89", "cuda": 13, "sms": [89],
            "host": "libquantfunc.so", "kernel": "libquantfunc_kernels.so",
            "sha256": {
                "libquantfunc.so": hashlib.sha256(open(mh, "rb").read()).hexdigest(),
                "libquantfunc_kernels.so": hashlib.sha256(open(mk, "rb").read()).hexdigest(),
            },
        }
        with open(os.path.join(bin_dir, ".engine-sm89-cu13.json"), "w", encoding="utf-8") as f:
            json.dump(marker, f)
        env_override = os.path.join(tmp, "override_kernels")
        elf("override_kernels", "libquantfunc_kernels.so", ["libcudart.so.12"])
        o_bin = qfe._engine_bin_dir
        qfe._engine_bin_dir = lambda: bin_dir
        try:
            _set_fake_torch("13.0")
            qfe.assert_toolchain_compatible(mh)
            os.environ[qfe._ENV_KERNEL_PLUGIN_DIR] = env_override
            try:
                qfe.assert_toolchain_compatible(mh)
                got_ov = "LOAD"
            except RuntimeError as e:
                got_ov = "MISMATCH" if "MISMATCH" in str(e) else str(e)
            ok = got_ov == "MISMATCH"
            print(f"  [{'OK ' if ok else 'FAIL'}] QF_KERNEL_PLUGIN_DIR overrides installed marker and is checked -> {got_ov}")
            bad += not ok
        finally:
            os.environ.pop(qfe._ENV_KERNEL_PLUGIN_DIR, None)
            qfe._engine_bin_dir = o_bin

        # .dev_lib_lock CUDA 12 end-to-end: resolve_so_path -> assert_toolchain_compatible
        local_bin = os.path.join(tmp, "local_cu12")
        lh = elf("local_cu12", "libquantfunc-12.so", clean)
        elf("local_cu12", "libquantfunc_kernels-12.so", ["libcudart.so.12"])
        open(os.path.join(local_bin, qfe._ENGINE_LOCAL_BUILD_LOCK), "w", encoding="utf-8").close()
        o_bin, o_ov = qfe._engine_bin_dir, os.environ.pop(qfe._ENV_SO_OVERRIDE, None)
        qfe._engine_bin_dir = lambda: local_bin
        try:
            _set_fake_torch("12.8")
            resolved = qfe.resolve_so_path()
            qfe.assert_toolchain_compatible(resolved)
            ok = resolved == os.path.realpath(lh)
            print(f"  [{'OK ' if ok else 'FAIL'}] .dev_lib_lock CUDA 12 resolves libquantfunc-12.so + passes real ELF guard")
            bad += not ok
        finally:
            qfe._engine_bin_dir = o_bin
            if o_ov is not None:
                os.environ[qfe._ENV_SO_OVERRIDE] = o_ov
    finally:
        os.environ.pop(qfe._ENV_KERNEL_PLUGIN_DIR, None)
        if o_env is not None:
            os.environ[qfe._ENV_KERNEL_PLUGIN_DIR] = o_env
        shutil.rmtree(tmp, ignore_errors=True)
    return bad


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
    bad += _elf_arms()
    bad += _pe_arms()
    print("TOOLCHAIN_GUARD:", "PASS" if bad == 0 else f"FAIL ({bad} wrong)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
