#!/usr/bin/env python3
"""Death-rule for the engine's CUDA-library resolution against torch (qf_engine._torch_cuda_plan,
_assert_torch_cuda_family, _engine_cuda_needs, load_lib).

Measured failure (ship-deps-e cu12clash/REPORT.md): in a torch-cu128 ComfyUI the cu12 engine did not dlopen —
"libcusolver.so.11: undefined symbol: cublasSetEnvironmentMode, version libcublas.so.12". torch maps its
cudart / cublas / cublasLt / cudnn at import but its cuSOLVER only after a linalg call, so the engine's
libcusolver.so.11 resolved through the build host's RPATH (or ld.so.cache) to a CUDA 12.9 copy that needs a newer
cuBLAS than torch's. One process must hold one cuBLAS/cuSOLVER set, torch's. This test asserts:
  (A) pip layout: torch's cuSOLVER is found in the sibling nvidia/<pkg>/lib folder and planned for preload;
  (B) conda layout (one lib folder): found there, through the soname symlink; a SIBLING env's copy is never taken;
  (C) a CUDA library the engine maps from outside torch's set -> loud refusal naming the file (also when the maps
      line shows the versioned file a soname symlink resolves to); all from torch's set -> no refusal;
  (D) no CUDA torch in the process -> nothing planned, nothing refused; a soname torch's set does not ship (the
      cu13 engine's cuDNN-compat libcublasLt.so.12) is left to the loader and not refused;
  (E) the closure: host + kernel + sidecar DT_NEEDED, CUDA toolkit sonames only, never the driver;
  (F) load_lib, the production entry point: torch's copy is dlopened BEFORE the engine, a CUDA sidecar that
      torch's set provides is not preloaded from the engine's folder, and a foreign mapping after the load refuses
      with the library left unbound.
Runs anywhere: temp folders stand in for torch's libraries and a text for /proc/self/maps.

Run: python3 cuda_family_test.py   (exit 0 = all correct; exit 1 = a check is wrong)
"""
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import qf_engine as qfe  # noqa: E402


def _touch(*parts):
    p = os.path.join(*parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").close()
    return p


def _link(target, *parts):
    p = os.path.join(*parts)
    os.symlink(os.path.basename(target), p)
    return p


def _maps(*paths):
    """A /proc/self/maps text mapping each path (one r-xp line each, the shape the kernel prints)."""
    return "".join(f"7f{i:010x}000-7f{i:010x}fff r-xp 00000000 103:02 {1000 + i}    {p}\n"
                   for i, p in enumerate(paths, 1))


def _pip(root):
    """torch's pip set: one nvidia/<pkg>/lib folder per package; cuSOLVER present but not mapped yet."""
    nv = os.path.join(root, "site-packages", "nvidia")
    return {
        "cublas": _touch(nv, "cublas", "lib", "libcublas.so.12"),
        "cublasLt": _touch(nv, "cublas", "lib", "libcublasLt.so.12"),
        "cudart": _touch(nv, "cuda_runtime", "lib", "libcudart.so.12"),
        "cudnn": _touch(nv, "cudnn", "lib", "libcudnn.so.9"),
        "cusolver": _touch(nv, "cusolver", "lib", "libcusolver.so.11"),
    }


_ENGINE_CU12 = {"libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcusolver.so.11", "libcudnn.so.9"}


def _check(label, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail and not ok else ""))
    return 0 if ok else 1


def _refused(provided, dirs, maps):
    try:
        qfe._assert_torch_cuda_family(provided, dirs, qfe._mapped_files(maps))
        return None
    except RuntimeError as e:
        return str(e)


def arm_pip(tmp):
    t = _pip(tmp)
    mapped = qfe._mapped_files(_maps(t["cublas"], t["cublasLt"], t["cudart"], t["cudnn"], "/usr/lib/libc.so.6"))
    dirs, provided, preloads = qfe._torch_cuda_plan(_ENGINE_CU12, mapped)
    bad = _check("(A) pip layout: torch's cuSOLVER in the sibling nvidia/cusolver/lib is planned",
                 preloads == [t["cusolver"]], f"preloads={preloads}")
    bad += _check("(A) pip layout: every engine CUDA soname counts as torch-provided",
                  provided == _ENGINE_CU12, f"provided={sorted(provided)}")
    return bad


def arm_conda(tmp):
    lib = os.path.join(tmp, "envs", "a", "lib")
    blas = _touch(lib, "libcublas.so.12.8.4.1")
    rt = _touch(lib, "libcudart.so.12.8.90")
    _link(blas, lib, "libcublas.so.12")
    _link(rt, lib, "libcudart.so.12")
    solver = _link(_touch(lib, "libcusolver.so.11.7.3.90"), lib, "libcusolver.so.11")
    mapped = qfe._mapped_files(_maps(blas, rt))        # maps shows the versioned files the symlinks resolve to
    _, provided, preloads = qfe._torch_cuda_plan({"libcusolver.so.11", "libcublas.so.12"}, mapped)
    bad = _check("(B) conda layout: cuSOLVER planned from torch's one lib folder",
                 preloads == [solver], f"preloads={preloads}")
    bad += _check("(B) conda layout: the mapped versioned cuBLAS counts as torch-provided",
                  "libcublas.so.12" in provided, f"provided={sorted(provided)}")
    # a sibling env holds a cuSOLVER; torch's own env (c) has none -> it must NOT be borrowed from env b
    other = _touch(tmp, "envs", "b", "lib", "libcusolver.so.11")
    lc = os.path.join(tmp, "envs", "c", "lib")
    mapped_c = qfe._mapped_files(_maps(_touch(lc, "libcublas.so.12"), _touch(lc, "libcudart.so.12")))
    _, provided_c, preloads_c = qfe._torch_cuda_plan({"libcusolver.so.11"}, mapped_c)
    bad += _check("(B) conda layout: a sibling env's cuSOLVER is never taken",
                  other not in preloads_c and not preloads_c and not provided_c,
                  f"preloads={preloads_c} provided={sorted(provided_c)}")
    return bad


def arm_foreign(tmp):
    t = _pip(tmp)
    mapped = qfe._mapped_files(_maps(t["cublas"], t["cublasLt"], t["cudart"], t["cudnn"]))
    dirs, provided, _ = qfe._torch_cuda_plan(_ENGINE_CU12, mapped)
    torch_set = (t["cublas"], t["cublasLt"], t["cudart"], t["cudnn"])
    ok = _refused(provided, dirs, _maps(*torch_set, t["cusolver"]))
    bad = _check("(C) every CUDA library from torch's set -> no refusal", ok is None, ok or "")
    build_host = "/root/cuda-12.9.2/lib/libcusolver.so.11"
    msg = _refused(provided, dirs, _maps(*torch_set, build_host))
    bad += _check("(C) cuSOLVER from the build host's toolkit -> refusal naming it",
                  msg is not None and build_host in msg, f"message={msg!r}")
    system = "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcusolver.so.11.6.4.69"
    msg = _refused(provided, dirs, _maps(*torch_set, system))
    bad += _check("(C) a versioned system cuSOLVER file -> refusal naming it",
                  msg is not None and system in msg, f"message={msg!r}")
    dup = "/opt/other/lib/libcublas.so.12"
    msg = _refused(provided, dirs, _maps(*torch_set, t["cusolver"], dup))
    bad += _check("(C) a second cuBLAS beside torch's -> refusal naming it",
                  msg is not None and dup in msg, f"message={msg!r}")
    # a foreign cuSOLVER already mapped before the load: the engine binds to it by soname, so preloading torch's
    # file would only map a second copy; the soname stays torch-provided and the check refuses the foreign one
    _, pre_provided, pre_preloads = qfe._torch_cuda_plan(_ENGINE_CU12,
                                                         qfe._mapped_files(_maps(*torch_set, build_host)))
    bad += _check("(C) a cuSOLVER already mapped from elsewhere is not preloaded over, and stays checked",
                  not pre_preloads and "libcusolver.so.11" in pre_provided,
                  f"preloads={pre_preloads} provided={sorted(pre_provided)}")
    return bad


def arm_no_torch_and_unshipped(tmp):
    dirs, provided, preloads = qfe._torch_cuda_plan(_ENGINE_CU12, qfe._mapped_files(_maps("/usr/lib/libc.so.6")))
    bad = _check("(D) no CUDA torch mapped -> nothing planned", not dirs and not provided and not preloads,
                 f"dirs={dirs} provided={provided} preloads={preloads}")
    bad += _check("(D) no CUDA torch mapped -> nothing refused",
                  _refused(provided, dirs, _maps("/usr/local/cuda/lib64/libcusolver.so.11")) is None)
    # cu13: one nvidia/cu13/lib folder; the engine also asks for the cu12 cublasLt that set does not ship
    lib = os.path.join(tmp, "site-packages", "nvidia", "cu13", "lib")
    blas, rt = _touch(lib, "libcublas.so.13"), _touch(lib, "libcudart.so.13")
    solver = _touch(lib, "libcusolver.so.12")
    mapped = qfe._mapped_files(_maps(blas, rt))
    dirs, provided, preloads = qfe._torch_cuda_plan({"libcusolver.so.12", "libcublasLt.so.12"}, mapped)
    bad += _check("(D) cu13 set: libcusolver.so.12 planned from torch's folder, libcublasLt.so.12 not provided",
                  preloads == [solver] and provided == {"libcusolver.so.12"},
                  f"preloads={preloads} provided={sorted(provided)}")
    msg = _refused(provided, dirs, _maps(blas, rt, solver, "/usr/local/cuda-12/lib64/libcublasLt.so.12"))
    bad += _check("(D) a soname torch's set does not ship is not refused", msg is None, msg or "")
    return bad


def arm_closure(tmp):
    d = os.path.join(tmp, "engine")
    host = _touch(d, "libquantfunc-12.so")
    _touch(d, "libquantfunc_kernels-12.so")
    _touch(d, "libquantfunc_attention.so")
    needed = {
        "libquantfunc-12.so": ["libquantfunc_kernels-12.so", "libquantfunc_attention.so", "libcudart.so.12",
                               "libcusolver.so.11", "libcuda.so.1", "libstdc++.so.6"],
        "libquantfunc_kernels-12.so": ["libcudart.so.12", "libcublasLt.so.12", "libcudnn.so.9", "libcuda.so.1"],
        "libquantfunc_attention.so": ["libcublas.so.12", "libquantfunc-12.so"],
    }
    orig = qfe._elf_needed
    qfe._elf_needed = lambda p: needed.get(os.path.basename(p), [])
    try:
        got = qfe._engine_cuda_needs(host)
    finally:
        qfe._elf_needed = orig
    return _check("(E) closure = host + kernel + sidecar CUDA sonames, never the driver", got == _ENGINE_CU12,
                  f"got={sorted(got)}")


def _load(tmp, after_maps):
    """Run load_lib with every outside effect faked. Returns (dlopen calls, exception or None, library bound)."""
    t = _pip(tmp)
    d = os.path.join(tmp, "engine")
    so = _touch(d, "libquantfunc-12.so")
    before = _maps(t["cublas"], t["cublasLt"], t["cudart"], t["cudnn"])
    maps = [before, after_maps(t)]
    calls = []
    saved = {n: getattr(qfe, n) for n in ("resolve_so_path", "assert_toolchain_compatible", "_engine_cuda_needs",
                                          "_mapped_files", "_sidecar_preloads", "_engine_load_ok", "_bind",
                                          "_emit_fingerprint", "_LIB")}
    cdll = qfe.ctypes.CDLL
    real_mapped = qfe._mapped_files
    qfe.resolve_so_path = lambda: so
    qfe.assert_toolchain_compatible = lambda _p: calls.append("guard")
    qfe._engine_cuda_needs = lambda _p: set(_ENGINE_CU12)
    qfe._mapped_files = lambda maps_text=None: real_mapped(maps.pop(0))
    qfe._sidecar_preloads = lambda _p: ["libcusolver.so.11", "libopencv_core.so.406"]
    qfe._engine_load_ok = lambda _p: None
    qfe._bind = lambda lib: lib
    qfe._emit_fingerprint = lambda: None
    qfe._LIB = None
    qfe.ctypes.CDLL = lambda p, **_k: (calls.append(p), object())[1]
    err = None
    try:
        qfe.load_lib()
    except Exception as e:  # noqa: BLE001
        err = e
    finally:
        bound = qfe._LIB is not None
        qfe.ctypes.CDLL = cdll
        for n, v in saved.items():
            setattr(qfe, n, v)
    return t, so, calls, err, bound


def arm_load_lib(tmp):
    t, so, calls, err, bound = _load(os.path.join(tmp, "clean"),
                                     lambda t: _maps(t["cublas"], t["cublasLt"], t["cudart"], t["cudnn"],
                                                     t["cusolver"]))
    order_ok = (err is None and bound and t["cusolver"] in calls and so in calls
                and calls.index(t["cusolver"]) < calls.index(so) and calls[0] == "guard")
    bad = _check("(F) load_lib: guard, then torch's cuSOLVER, then the engine; library bound", order_ok,
                 f"calls={calls} err={err!r}")
    engine_dir_copy = os.path.join(os.path.dirname(so), "libcusolver.so.11")
    bad += _check("(F) load_lib: a CUDA sidecar torch provides is not preloaded from the engine folder",
                  engine_dir_copy not in calls and os.path.join(os.path.dirname(so), "libopencv_core.so.406") in calls,
                  f"calls={calls}")
    build_host = "/root/cuda-12.9.2/lib/libcusolver.so.11"
    _, _, calls, err, bound = _load(os.path.join(tmp, "foreign"),
                                    lambda t: _maps(t["cublas"], t["cublasLt"], t["cudart"], t["cudnn"], build_host))
    bad += _check("(F) load_lib: a foreign cuSOLVER after the load refuses and leaves the library unbound",
                  isinstance(err, RuntimeError) and build_host in str(err) and not bound,
                  f"err={err!r} bound={bound}")
    return bad


def main():
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = os.path.realpath(tmp)   # /proc/self/maps shows resolved paths (macOS /tmp is a symlink, for one)
        for i, arm in enumerate((arm_pip, arm_conda, arm_foreign, arm_no_torch_and_unshipped, arm_closure,
                                 arm_load_lib)):
            try:
                bad += arm(os.path.join(tmp, str(i)))
            except Exception as e:  # noqa: BLE001 — a missing function is a FAIL of that arm, not a crash
                bad += _check(arm.__name__, False, repr(e))
    print("CUDA_FAMILY:", "PASS" if bad == 0 else f"FAIL ({bad} wrong)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
