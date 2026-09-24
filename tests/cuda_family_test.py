#!/usr/bin/env python3
"""Death-rule for the engine's CUDA-library resolution against torch (qf_engine._torch_cuda_plan,
_assert_torch_cuda_family, _engine_cuda_needs, _linker_path, load_lib).

Measured failure (ship-deps-e cu12clash/REPORT.md): in a torch-cu128 ComfyUI the cu12 engine did not dlopen —
"libcusolver.so.11: undefined symbol: cublasSetEnvironmentMode, version libcublas.so.12". torch maps its
cudart / cublas / cublasLt / cudnn at import but its cuSOLVER only after a linalg call, so the engine's
libcusolver.so.11 resolved through the build host's RPATH (or ld.so.cache) to a CUDA 12.9 copy that needs a newer
cuBLAS than torch's. One process must hold one cuBLAS/cuSOLVER set, torch's.

"torch's copy" of a soname is what the dynamic linker hands out for it (the first loaded object carrying it — the
same lookup the engine's NEEDED entry does), so a second, vendored copy some other package loaded can neither pose
as torch's set nor be picked for the engine. This test asserts:
  (A) pip layout: torch's cuSOLVER is found in the sibling nvidia/<pkg>/lib folder and planned for preload;
  (B) conda layout (one lib folder): found there, through the soname symlink and an unnormalised linker path; a
      SIBLING env's copy is never taken;
  (P) another package's vendored cuBLAS + cuSOLVER (loaded after torch's) never widen torch's set: torch's
      cuSOLVER is the one planned, and a copy nobody binds to is not refused;
  (C) what the linker hands out for an engine CUDA soname after the load must be torch's copy, else a loud refusal
      naming the file (build host, a versioned system file, one already bound before the load);
  (D) no CUDA torch in the process -> nothing planned, nothing refused; a soname torch's set does not ship (the
      cu13 engine's cuDNN-compat libcublasLt.so.12) is left to the loader and not refused;
  (E) the closure: host + kernel + sidecar DT_NEEDED, CUDA toolkit sonames only, never the driver;
  (L) _linker_path against this process's real dynamic linker: a loaded soname -> its file, an unloaded one -> None;
  (F) load_lib, the production entry point: torch's copy is dlopened BEFORE the engine, a CUDA sidecar that
      torch's set provides is not preloaded from the engine's folder, and a foreign binding after the load refuses
      with the library left unbound.
Runs anywhere Linux: temp folders stand in for torch's libraries and a dict for the linker's soname table.

Run: python3 cuda_family_test.py   (exit 0 = all correct; exit 1 = a check is wrong)
"""
import os
import sys
import tempfile
import types

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


def _pip(root):
    """torch's pip set: one nvidia/<pkg>/lib folder per package; cuSOLVER present but not loaded yet."""
    nv = os.path.join(root, "site-packages", "nvidia")
    return {
        "cublas": _touch(nv, "cublas", "lib", "libcublas.so.12"),
        "cublasLt": _touch(nv, "cublas", "lib", "libcublasLt.so.12"),
        "cudart": _touch(nv, "cuda_runtime", "lib", "libcudart.so.12"),
        "cudnn": _touch(nv, "cudnn", "lib", "libcudnn.so.9"),
        "cusolver": _touch(nv, "cusolver", "lib", "libcusolver.so.11"),
    }


def _loaded(t, **extra):
    """The linker's soname table after `import torch` (cuSOLVER not loaded), plus `extra` soname -> file."""
    table = {"libcublas.so.12": t["cublas"], "libcublasLt.so.12": t["cublasLt"],
             "libcudart.so.12": t["cudart"], "libcudnn.so.9": t["cudnn"]}
    table.update({k.replace("_", "."): v for k, v in extra.items()})
    return table.get


_ENGINE_CU12 = {"libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcusolver.so.11", "libcudnn.so.9"}


def _check(label, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail and not ok else ""))
    return 0 if ok else 1


def _refused(provided, dirs, linker):
    try:
        qfe._assert_torch_cuda_family(provided, dirs, linker)
        return None
    except RuntimeError as e:
        return str(e)


def arm_pip(tmp):
    t = _pip(tmp)
    _, provided, preloads = qfe._torch_cuda_plan(_ENGINE_CU12, _loaded(t), 12)
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
    # the linker reports the name it found, unnormalised (an RPATH like $ORIGIN/../../lib); maps show the target
    via_rpath = os.path.join(tmp, "envs", "a", "lib", "python3.12", "site-packages", "torch", "lib", "..", "..", "..",
                             "..", "libcublas.so.12")
    os.makedirs(os.path.join(tmp, "envs", "a", "lib", "python3.12", "site-packages", "torch", "lib"), exist_ok=True)
    linker = {"libcublas.so.12": via_rpath, "libcudart.so.12": via_rpath.replace("libcublas", "libcudart")}.get
    _, provided, preloads = qfe._torch_cuda_plan({"libcusolver.so.11", "libcublas.so.12"}, linker, 12)
    bad = _check("(B) conda layout: cuSOLVER planned from torch's one lib folder",
                 preloads == [solver], f"preloads={preloads}")
    bad += _check("(B) conda layout: the unnormalised linker path of cuBLAS counts as torch-provided",
                  "libcublas.so.12" in provided, f"provided={sorted(provided)}")
    # a sibling env holds a cuSOLVER; torch's own env (c) has none -> it must NOT be borrowed from env b
    other = _touch(tmp, "envs", "b", "lib", "libcusolver.so.11")
    lc = os.path.join(tmp, "envs", "c", "lib")
    linker_c = {"libcublas.so.12": _touch(lc, "libcublas.so.12"), "libcudart.so.12": _touch(lc, "libcudart.so.12")}.get
    _, provided_c, preloads_c = qfe._torch_cuda_plan({"libcusolver.so.11"}, linker_c, 12)
    bad += _check("(B) conda layout: a sibling env's cuSOLVER is never taken",
                  other not in preloads_c and not preloads_c and not provided_c,
                  f"preloads={preloads_c} provided={sorted(provided_c)}")
    return bad


def arm_poison(tmp):
    """Another package vendors its own cuBLAS + cuSOLVER (a cupy-style lib folder), loaded after torch's."""
    t = _pip(tmp)
    vend = os.path.join(tmp, "site-packages", "cupy_backends", "cuda", "lib")
    v_blas, v_solver = _touch(vend, "libcublas.so.12"), _touch(vend, "libcusolver.so.11")
    # the linker still hands out torch's cuBLAS for the soname: torch's copy was loaded first
    dirs, provided, preloads = qfe._torch_cuda_plan(_ENGINE_CU12, _loaded(t), 12)
    bad = _check("(P) a vendored cuBLAS/cuSOLVER folder never joins torch's set; torch's cuSOLVER is planned",
                 preloads == [t["cusolver"]] and vend not in dirs and v_solver not in preloads,
                 f"preloads={preloads} vendored-in-dirs={vend in dirs}")
    after = _loaded(t, libcusolver_so_11=t["cusolver"])
    bad += _check("(P) the vendored cuBLAS nobody binds to (the linker hands out torch's) is not refused",
                  _refused(provided, dirs, after) is None and os.path.exists(v_blas))
    return bad


def arm_foreign(tmp):
    t = _pip(tmp)
    dirs, provided, _ = qfe._torch_cuda_plan(_ENGINE_CU12, _loaded(t), 12)
    ok = _refused(provided, dirs, _loaded(t, libcusolver_so_11=t["cusolver"]))
    bad = _check("(C) every CUDA library the linker hands out is torch's -> no refusal", ok is None, ok or "")
    build_host = _touch(tmp, "root", "cuda-12.9.2", "lib", "libcusolver.so.11")
    msg = _refused(provided, dirs, _loaded(t, libcusolver_so_11=build_host))
    bad += _check("(C) cuSOLVER from the build host's toolkit -> refusal naming it",
                  msg is not None and build_host in msg, f"message={msg!r}")
    system = _touch(tmp, "usr", "local", "cuda-12.6", "targets", "x86_64-linux", "lib", "libcusolver.so.11.6.4.69")
    msg = _refused(provided, dirs, _loaded(t, libcusolver_so_11=system))
    bad += _check("(C) a versioned system cuSOLVER file -> refusal naming it",
                  msg is not None and system in msg, f"message={msg!r}")
    # a foreign cuSOLVER bound BEFORE the load (torch never loaded its own): the engine binds to it by soname, so
    # preloading torch's file would only add a second copy; the soname stays checked and the check refuses it
    early = _touch(tmp, "opt", "other", "lib", "libcusolver.so.11")
    pre_dirs, pre_provided, pre_preloads = qfe._torch_cuda_plan(
        _ENGINE_CU12, _loaded(t, libcusolver_so_11=early), 12)
    msg = _refused(pre_provided, pre_dirs, _loaded(t, libcusolver_so_11=early))
    bad += _check("(C) a cuSOLVER bound from elsewhere before the load is not preloaded over, and is refused",
                  not pre_preloads and "libcusolver.so.11" in pre_provided and msg is not None and early in msg,
                  f"preloads={pre_preloads} provided={sorted(pre_provided)} message={msg!r}")
    return bad


def arm_no_torch_and_unshipped(tmp):
    t = _pip(tmp)
    dirs, provided, preloads = qfe._torch_cuda_plan(_ENGINE_CU12, _loaded(t), None)
    bad = _check("(D) no CUDA torch -> nothing planned", not dirs and not provided and not preloads,
                 f"dirs={dirs} provided={provided} preloads={preloads}")
    dirs, provided, preloads = qfe._torch_cuda_plan(_ENGINE_CU12, {}.get, 12)
    bad += _check("(D) torch's cudart/cublas not loaded -> nothing planned", not dirs and not provided and not preloads,
                  f"dirs={dirs} provided={provided} preloads={preloads}")
    bad += _check("(D) nothing planned -> nothing refused",
                  _refused(provided, dirs, {"libcusolver.so.11": "/usr/local/cuda/lib64/libcusolver.so.11"}.get) is None)
    # cu13: one nvidia/cu13/lib folder; the engine also asks for the cu12 cublasLt that set does not ship
    lib = os.path.join(tmp, "cu13", "site-packages", "nvidia", "cu13", "lib")
    blas, rt = _touch(lib, "libcublas.so.13"), _touch(lib, "libcudart.so.13")
    solver = _touch(lib, "libcusolver.so.12")
    linker = {"libcublas.so.13": blas, "libcudart.so.13": rt}.get
    dirs, provided, preloads = qfe._torch_cuda_plan({"libcusolver.so.12", "libcublasLt.so.12"}, linker, 13)
    bad += _check("(D) cu13 set: libcusolver.so.12 planned from torch's folder, libcublasLt.so.12 not provided",
                  preloads == [solver] and provided == {"libcusolver.so.12"},
                  f"preloads={preloads} provided={sorted(provided)}")
    after = {"libcublas.so.13": blas, "libcudart.so.13": rt, "libcusolver.so.12": solver,
             "libcublasLt.so.12": _touch(tmp, "usr", "local", "cuda-12", "lib64", "libcublasLt.so.12")}.get
    msg = _refused(provided, dirs, after)
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


def arm_real_linker(_tmp):
    """(L) the one piece the dict-driven arms stand in for: this process's real dynamic linker."""
    if not sys.platform.startswith("linux"):
        print("  [SKIP] (L) real linker: not Linux")
        return 0
    libc = qfe._linker_path("libc.so.6")
    bad = _check("(L) a loaded soname -> the file the linker hands out",
                 bool(libc) and os.path.isfile(libc) and os.path.basename(os.path.realpath(libc)).startswith("libc"),
                 f"libc.so.6 -> {libc!r}")
    bad += _check("(L) a soname nothing loaded -> None", qfe._linker_path("libqf-never-loaded.so.99") is None)
    # a real library next to libc that this process has not loaded: the probe must neither report nor LOAD it
    libdir = os.path.dirname(os.path.realpath(libc))
    with open("/proc/self/maps") as f:
        maps = f.read()
    idle = next((n for n in ("libBrokenLocale.so.1", "libanl.so.1", "libutil.so.1")
                 if os.path.exists(os.path.join(libdir, n)) and n not in maps), None)
    if idle is None:
        print("  [SKIP] (L) no unloaded glibc companion library to probe")
        return bad
    got = qfe._linker_path(idle)
    with open("/proc/self/maps") as f:
        loaded_now = idle in f.read()
    bad += _check(f"(L) probing {idle} (present, not loaded) neither reports nor loads it",
                  got is None and not loaded_now, f"got={got!r} loaded_now={loaded_now}")
    return bad


def _load(tmp, after, torch_imported=True):
    """Run load_lib with every outside effect faked; the linker's table flips from torch-at-import to `after(t)`
    when the engine itself is dlopened. Returns (torch set, engine path, dlopen calls, exception or None, bound).
    `torch_imported=False` runs it in a process with no torch module: load_lib must then not ask for torch's major."""
    t = _pip(tmp)
    d = os.path.join(tmp, "engine")
    so = _touch(d, "libquantfunc-12.so")
    state = {"table": _loaded(t)}
    calls = []

    def fake_cdll(p, **_k):
        calls.append(p)
        if p == so:
            state["table"] = after(t)
        return object()

    saved = {n: getattr(qfe, n) for n in ("resolve_so_path", "assert_toolchain_compatible", "_engine_cuda_needs",
                                          "_linker_path", "_torch_cuda_major", "_sidecar_preloads",
                                          "_engine_load_ok", "_bind", "_emit_fingerprint", "_LIB", "_LIB_PATH",
                                          "_FINGERPRINT_PENDING")}
    cdll = qfe.ctypes.CDLL
    qfe.resolve_so_path = lambda: so
    qfe.assert_toolchain_compatible = lambda _p: calls.append("guard")
    qfe._engine_cuda_needs = lambda _p: set(_ENGINE_CU12)
    qfe._linker_path = lambda soname: state["table"](soname)
    qfe._torch_cuda_major = lambda: (calls.append("torch_major"), 12)[1]
    qfe._sidecar_preloads = lambda _p: ["libcusolver.so.11", "libopencv_core.so.406"]
    qfe._engine_load_ok = lambda _p: None
    qfe._bind = lambda lib: lib
    qfe._emit_fingerprint = lambda: None
    qfe._LIB = None
    qfe.ctypes.CDLL = fake_cdll
    had_torch = sys.modules.get("torch")
    if torch_imported:
        sys.modules["torch"] = had_torch or types.ModuleType("torch")
    else:
        sys.modules.pop("torch", None)
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
        if had_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = had_torch
    return t, so, calls, err, bound


def arm_load_lib(tmp):
    t, so, calls, err, bound = _load(os.path.join(tmp, "clean"),
                                     lambda t: _loaded(t, libcusolver_so_11=t["cusolver"]))
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
                                    lambda t: _loaded(t, libcusolver_so_11=build_host))
    bad += _check("(F) load_lib: a foreign cuSOLVER after the load refuses and leaves the library unbound",
                  isinstance(err, RuntimeError) and build_host in str(err) and not bound,
                  f"err={err!r} bound={bound}")
    t, so, calls, err, bound = _load(os.path.join(tmp, "notorch"), lambda t: _loaded(t), torch_imported=False)
    bad += _check("(F) load_lib: with no torch imported it never asks torch's major (no import) and preloads nothing",
                  err is None and bound and "torch_major" not in calls and t["cusolver"] not in calls,
                  f"calls={calls} err={err!r}")
    return bad


def main():
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = os.path.realpath(tmp)   # the linker reports resolved paths (macOS /tmp is a symlink, for one)
        for i, arm in enumerate((arm_pip, arm_conda, arm_poison, arm_foreign, arm_no_torch_and_unshipped, arm_closure,
                                 arm_real_linker, arm_load_lib)):
            try:
                bad += arm(os.path.join(tmp, str(i)))
            except Exception as e:  # noqa: BLE001 — a missing function is a FAIL of that arm, not a crash
                bad += _check(arm.__name__, False, repr(e))
    print("CUDA_FAMILY:", "PASS" if bad == 0 else f"FAIL ({bad} wrong)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
