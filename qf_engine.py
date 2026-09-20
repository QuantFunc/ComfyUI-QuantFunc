"""qf_native.qf_engine — self-contained ctypes bridge to libquantfunc.so for the
native ComfyUI loader. Structs mirror include/quantfunc.h (session structs copied
verbatim from the PROVEN tests/scripts/native_session_t1.py). No tests/lib dependency.
"""
import ctypes
import re as _re_soname
_SONAME_RE = _re_soname.compile(r"lib[^/]*\.so(?:\.\d+[a-z]?)*")   # lib*.so, lib*.so.5, libopencv_core.so.4.5d
import json
import os
import time
import platform
import re
import struct

QUANTFUNC_OK = 0
# quantfunc_dtype_t: FP32=0, FP16=1, BF16=2 (matches QF_DTYPE in the harness)
QF_FP32, QF_FP16, QF_BF16 = 0, 1, 2
# fp8_e4m3 — a valid OUTPUT dtype request for the cloud/standalone TE encode only
# (the result tensor handle reports its own dtype code 1 with per-token scales).
QF_FP8_E4M3 = 3

# Cloud TE progress/cancel callback: int(int done, int total, void* user) — a non-zero
# return CANCELS the in-flight encode (mirrors TECloudParams.progress_callback).
QF_TE_CLOUD_PROGRESS_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)


def _dbg_prof(msg):
    """QF_NATIVE_PROF-gated diagnostic line (the [qf_prof] channel the perf probes use)."""
    import os as _os
    if _os.environ.get("QF_NATIVE_PROF") == "1":
        print(f"[qf_prof] {msg}", flush=True)

# Platform dispatch — the bundled native library lives in bin/<subdir>/<basename>, so a Windows install
# finds bin/windows/quantfunc.dll and a Linux install finds bin/linux/libquantfunc.so. The Windows/Linux
# split follows the production plugin's lib_setup.py (_IS_WINDOWS / bin/<subdir>/); the Darwin branch
# below is an ADDITION here (production ships no macOS build) — a harmless forward-looking default,
# never yet exercised.
_IS_WINDOWS = platform.system() == "Windows"
_IS_DARWIN = platform.system() == "Darwin"
_BIN_SUBDIR = "windows" if _IS_WINDOWS else ("darwin" if _IS_DARWIN else "linux")
_LIB_BASENAME = ("quantfunc.dll" if _IS_WINDOWS else
                 ("libquantfunc.dylib" if _IS_DARWIN else "libquantfunc.so"))
_KEYFILE_BASENAME = "config.json"
# DEV-ONLY override env vars. These are read from the PROCESS ENVIRONMENT, never from a node
# input — a shared workflow.json CANNOT set an environment variable, so it cannot influence which
# native library is dlopened (that would be an arbitrary-code-execution primitive; see resolve_so_path).
_ENV_SO_OVERRIDE = "QF_NATIVE_SO_PATH"
_ENV_KEYFILE_OVERRIDE = "QF_NATIVE_KEYFILE"
# FORK-2 escape hatch: proceed on a torch-CUDA / .so-CUDA combination the guard cannot verify (e.g. a
# static-cudart .so). Read from the PROCESS ENVIRONMENT only — a shared workflow.json cannot set it.
_ENV_ALLOW_UNVERIFIED_TOOLCHAIN = "QF_NATIVE_ALLOW_UNVERIFIED_TOOLCHAIN"
# ELF constants (System V ABI) used by the pure-stdlib DT_NEEDED reader below.
_SHT_DYNAMIC = 6    # section header type: the .dynamic array
_DT_NULL = 0        # dynamic tag: end of the .dynamic array
_DT_NEEDED = 1      # dynamic tag: dynstr offset of a required shared-library name


class InitParams(ctypes.Structure):
    _fields_ = [
        ("model_dir", ctypes.c_char_p),
        ("transformer_path", ctypes.c_char_p),
        ("vae_path", ctypes.c_char_p),
        ("text_encoder_path", ctypes.c_char_p),
        ("tokenizer_path", ctypes.c_char_p),
        ("scheduler_config", ctypes.c_char_p),
        ("model_backend", ctypes.c_char_p),
        ("device_idx", ctypes.c_int),
        ("config_json", ctypes.c_char_p),
    ]


class DenoiseBeginParams(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("num_steps", ctypes.c_int),
        ("max_context_dims", ctypes.c_int * 3),
        ("cond_dtype", ctypes.c_int),
        ("max_pooled_dims", ctypes.c_int * 2),
        ("producer_stream", ctypes.c_size_t),
        ("options_json", ctypes.c_char_p),
    ]


class DenoiseStepParams(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("latent_in", ctypes.c_void_p),
        ("velocity_out", ctypes.c_void_p),
        ("velocity_out_capacity", ctypes.c_size_t),
        ("dims", ctypes.c_int * 5),
        ("dtype", ctypes.c_int),
        ("sigma", ctypes.c_float),
        ("step_index", ctypes.c_int),
        ("total_steps", ctypes.c_int),
        ("context", ctypes.c_void_p),
        ("context_dims", ctypes.c_int * 3),
        ("context_dtype", ctypes.c_int),
        ("pooled", ctypes.c_void_p),
        ("pooled_dims", ctypes.c_int * 2),
        ("pooled_dtype", ctypes.c_int),
        ("txt_seq_lens", ctypes.POINTER(ctypes.c_int)),
        ("cfg_context_key", ctypes.c_uint64),
        ("producer_stream", ctypes.c_size_t),
        ("options_json", ctypes.c_char_p),
    ]


class DenoiseBeginEditParams(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("base", DenoiseBeginParams),
        ("ref_image_paths", ctypes.POINTER(ctypes.c_char_p)),
        ("num_ref_images", ctypes.c_int),
        ("ref_img_resize", ctypes.c_int),
    ]


class DenoiseBeginEditCondParams(ctypes.Structure):
    # i2v cond-latent ABI (design GO 2026-08-21): begin with the WORKFLOW-supplied fixed
    # conditioning tail ([1, in−z, Tlat, Hl, Wl], the model's processed latent space — comfy's
    # WanImageToVideo c_concat verbatim) instead of engine-side ref encode. Field order/types
    # mirror include/quantfunc.h VERBATIM. base = the PLAIN begin params (structural exclusivity
    # with the path-based edit surface). Present only in cond-ABI engine builds — bind
    # defensively (hasattr), exactly like quantfunc_denoise_step_multi.
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("base", DenoiseBeginParams),
        ("cond_tail", ctypes.c_void_p),
        ("cond_tail_dims", ctypes.c_int32 * 5),
        ("cond_tail_dtype", ctypes.c_int),
        ("cond_tail_bytes", ctypes.c_uint64),   # D1: declared allocation length; engine
                                                # refuses any mismatch with dims*itemsize
                                                # AND verifies the real extent covers it
    ]


class DenoiseFinalizeParams(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("latent", ctypes.c_void_p),
        ("latent_capacity", ctypes.c_size_t),
        ("dims", ctypes.c_int * 5),
        ("dtype", ctypes.c_int),
        ("producer_stream", ctypes.c_size_t),
        ("options_json", ctypes.c_char_p),
    ]


class DenoiseStepMultiParams(ctypes.Structure):
    # ONE joint audio+video step (D-class). base = the VIDEO lane (validated identically to a plain
    # quantfunc_denoise_step); the audio lane rides the SAME video sigma. Field order/types mirror
    # include/quantfunc.h quantfunc_denoise_step_multi_params_t VERBATIM. audio_latent_in is the
    # driver's CARRIED audio latent packed as rows [1, K*T, 32] (channel-major pack_audio); the engine
    # undoes audio_scale (= ModelSamplingAV.shift/audio_shift) before the forward and re-converts the
    # returned audio velocity so the driver's update stays correct.
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("base", DenoiseStepParams),
        ("audio_latent_in", ctypes.c_void_p),
        ("audio_velocity_out", ctypes.c_void_p),
        ("audio_velocity_out_capacity", ctypes.c_size_t),
        ("audio_dims", ctypes.c_int * 4),         # [B, C, K, T]  (unpacked audio latent geometry)
        ("audio_dtype", ctypes.c_int),
        ("audio_scale", ctypes.c_float),
    ]


class TECloudParams(ctypes.Structure):
    # Field order/types mirror include/quantfunc.h TECloudParams VERBATIM (natural
    # alignment matches the C compiler, so ctypes reproduces the C layout).
    _fields_ = [
        ("server_url", ctypes.c_char_p),
        ("api_key", ctypes.c_char_p),
        ("device_idx", ctypes.c_int),
        ("model_id", ctypes.c_char_p),
        ("text", ctypes.c_char_p),
        ("ref_paths", ctypes.POINTER(ctypes.c_char_p)),
        ("n_refs", ctypes.c_int),
        ("output_dtype", ctypes.c_int),
        ("resume_task_id", ctypes.c_char_p),
        ("wait_ms", ctypes.c_int),
        ("progress_callback", QF_TE_CLOUD_PROGRESS_CB),
        ("callback_user_data", ctypes.c_void_p),
    ]


def _bind(lib):
    v = ctypes.c_void_p
    lib.quantfunc_create.restype = ctypes.c_int
    lib.quantfunc_create.argtypes = [ctypes.POINTER(InitParams), ctypes.POINTER(v)]
    lib.quantfunc_destroy.restype = None          # header: `void quantfunc_destroy(...)`
    lib.quantfunc_destroy.argtypes = [v]
    lib.quantfunc_last_error.restype = ctypes.c_char_p
    lib.quantfunc_last_error.argtypes = []
    lib.quantfunc_set_log_level.restype = None
    lib.quantfunc_set_log_level.argtypes = [ctypes.c_int]
    # session
    lib.quantfunc_denoise_begin.restype = ctypes.c_int
    lib.quantfunc_denoise_begin.argtypes = [v, ctypes.POINTER(DenoiseBeginParams), ctypes.POINTER(v)]
    lib.quantfunc_denoise_begin_edit.restype = ctypes.c_int
    lib.quantfunc_denoise_begin_edit.argtypes = [v, ctypes.POINTER(DenoiseBeginEditParams), ctypes.POINTER(v)]
    lib.quantfunc_denoise_step.restype = ctypes.c_int
    lib.quantfunc_denoise_step.argtypes = [v, ctypes.POINTER(DenoiseStepParams)]
    # D-class joint audio+video step (MiniMax-H3). Present only in AV-capable engine builds; bind
    # defensively so a non-AV .so still loads (the H3 loader checks hasattr before use).
    if hasattr(lib, "quantfunc_denoise_step_multi"):
        lib.quantfunc_denoise_step_multi.restype = ctypes.c_int
        lib.quantfunc_denoise_step_multi.argtypes = [v, ctypes.POINTER(DenoiseStepMultiParams)]
    # i2v cond-latent begin (cond-ABI builds only; hasattr-gated like step_multi — an old .so
    # simply lacks the symbol and the wan i2v sampling path then refuses with guidance).
    if hasattr(lib, "quantfunc_denoise_cond_tail_supported"):
        # CR A-1 capability query (per-pipeline): 1 = this pipeline consumes a
        # begin_edit_cond cond-latent. Probe THIS (presence + answer), not the wan-era
        # begin_edit_cond symbol.
        lib.quantfunc_denoise_cond_tail_supported.restype = ctypes.c_int
        lib.quantfunc_denoise_cond_tail_supported.argtypes = [v]
    if hasattr(lib, "quantfunc_denoise_begin_edit_cond"):
        lib.quantfunc_denoise_begin_edit_cond.restype = ctypes.c_int
        lib.quantfunc_denoise_begin_edit_cond.argtypes = [
            v, ctypes.POINTER(DenoiseBeginEditCondParams), ctypes.POINTER(v)]
    lib.quantfunc_denoise_finalize.restype = ctypes.c_int
    lib.quantfunc_denoise_finalize.argtypes = [v, ctypes.POINTER(DenoiseFinalizeParams)]
    lib.quantfunc_denoise_end.restype = ctypes.c_int
    lib.quantfunc_denoise_end.argtypes = [v]
    # co-eviction (quantfunc.h:407-424): offload GPU->CPU freeing VRAM; the pipeline auto-reloads
    # from its CPU backup on the next generate. _sync blocks until the VRAM is actually released.
    lib.quantfunc_unload.restype = ctypes.c_int
    lib.quantfunc_unload.argtypes = [v]
    lib.quantfunc_unload_sync.restype = ctypes.c_int
    lib.quantfunc_unload_sync.argtypes = [v]
    # partial VRAM shed (inter-stage eviction fix; absent on older .so -> hasattr-guarded)
    if hasattr(lib, "quantfunc_partial_unload"):
        lib.quantfunc_partial_unload.restype = ctypes.c_int
        lib.quantfunc_partial_unload.argtypes = [v, ctypes.c_uint64,
                                                 ctypes.POINTER(ctypes.c_int64)]
    # comfy-ledger pair (quantfunc.h: what the engine HOLDS / how much MORE it NEEDS for a latent shape); an older
    # .so lacks one or both -> hasattr-guarded, the readers report 0 = UNKNOWN and comfy keeps its own estimates.
    if hasattr(lib, "quantfunc_resident_vram_bytes"):
        lib.quantfunc_resident_vram_bytes.restype = ctypes.c_int
        lib.quantfunc_resident_vram_bytes.argtypes = [v, ctypes.POINTER(ctypes.c_uint64)]
    if hasattr(lib, "quantfunc_vram_need_bytes"):
        lib.quantfunc_vram_need_bytes.restype = ctypes.c_int
        lib.quantfunc_vram_need_bytes.argtypes = [v, ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
                                                  ctypes.POINTER(ctypes.c_uint64)]
    # Cloud TE encode + tensor readout (design v11). hasattr-gated: an older .so
    # simply lacks these and the cloud-TE node then refuses with clear guidance.
    if hasattr(lib, "quantfunc_te_cloud_encode"):
        lib.quantfunc_te_cloud_encode.restype = ctypes.c_int
        lib.quantfunc_te_cloud_encode.argtypes = [
            ctypes.POINTER(TECloudParams),   # in
            ctypes.POINTER(v),               # out_result (tensor handle)
            ctypes.POINTER(ctypes.c_int),    # out_pending
            ctypes.c_char_p,                 # out_task_id (mutable buffer)
            ctypes.c_size_t,                 # out_task_id_cap
        ]
        lib.quantfunc_default_te_cloud_params.restype = TECloudParams
        lib.quantfunc_default_te_cloud_params.argtypes = []
        lib.quantfunc_tensor_info.restype = ctypes.c_int
        lib.quantfunc_tensor_info.argtypes = [
            v, ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32)]
        lib.quantfunc_tensor_read.restype = ctypes.c_int
        lib.quantfunc_tensor_read.argtypes = [v, ctypes.c_void_p, ctypes.c_size_t]
        lib.quantfunc_tensor_read_scales.restype = ctypes.c_int
        lib.quantfunc_tensor_read_scales.argtypes = [v, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        # [token-tags] optional exports (engine 9f3fd77f+): per-row modality tags for the
        # H3 fl2va semantic channel. An OLD .so lacks them — probe, never hard-require.
        try:
            lib.quantfunc_tensor_tags_count.restype = ctypes.c_size_t
            lib.quantfunc_tensor_tags_count.argtypes = [v]
            lib.quantfunc_tensor_read_tags.restype = ctypes.c_int
            lib.quantfunc_tensor_read_tags.argtypes = [v, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
            lib._has_tensor_tags = True
        except AttributeError:
            lib._has_tensor_tags = False
        lib.quantfunc_tensor_destroy.restype = None
        lib.quantfunc_tensor_destroy.argtypes = [v]
    return lib


_LIB = None


def resolve_so_path():
    """Resolve the engine native library WITHOUT any workflow-serializable input.

    THREAT MODEL (this is the #vuln an earlier `so_path` node-widget created): ComfyUI's whole
    distribution model is "open this shared workflow.json", so a node STRING widget in that JSON is
    ATTACKER-CONTROLLED. `ctypes.CDLL` runs the target library's constructors in-process the instant
    it loads, so letting a workflow choose the path is a remote-code-execution primitive (the classic
    "download this companion file, then load my workflow"). This resolver therefore accepts NO path
    from any node input. Both sources it DOES trust cannot be set by a shared workflow:
      1. QF_NATIVE_SO_PATH — a DEV override read from the PROCESS ENVIRONMENT only (a workflow.json
         cannot set an env var). Used as-is on the trusted dev machine; must exist + be a real file.
      2. the package-bundled library bin/<platform>/<basename> (platform-dispatched), or the package
         root as a fallback.
    Returns a realpath; raises loudly if nothing usable exists."""
    pkg = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    override = os.environ.get(_ENV_SO_OVERRIDE, "").strip()
    if override:
        candidates.append(override)
    candidates += [os.path.join(pkg, "bin", _BIN_SUBDIR, _LIB_BASENAME),
                   os.path.join(pkg, _LIB_BASENAME)]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.realpath(c)
    raise RuntimeError(
        f"qf_native: no engine library found — bundle {_LIB_BASENAME} in the package "
        f"bin/{_BIN_SUBDIR}/ , or set {_ENV_SO_OVERRIDE}=<abs path> on the (trusted) dev machine. "
        f"tried={candidates}")


def _elf_needed(path):
    """The DT_NEEDED shared-library names of an ELF file, via a pure-stdlib parse of its .dynamic/.dynstr
    sections. Returns [] on ANY parse failure. Pure-stdlib on purpose: FORK-2's toolchain guard must not
    depend on readelf/ldd being installed on the consumer's box."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        if data[:4] != b"\x7fELF":
            return []
        is64 = data[4] == 2
        end = "<" if data[5] == 1 else ">"          # 1 = little-endian
        if is64:
            e_shoff = struct.unpack_from(end + "Q", data, 0x28)[0]
            e_shentsize = struct.unpack_from(end + "H", data, 0x3a)[0]
            e_shnum = struct.unpack_from(end + "H", data, 0x3c)[0]
        else:
            e_shoff = struct.unpack_from(end + "I", data, 0x20)[0]
            e_shentsize = struct.unpack_from(end + "H", data, 0x2e)[0]
            e_shnum = struct.unpack_from(end + "H", data, 0x30)[0]
        secs = []                                    # (type, offset, size, link) per section header
        for i in range(e_shnum):
            b = e_shoff + i * e_shentsize
            if is64:
                secs.append((struct.unpack_from(end + "I", data, b + 4)[0],
                             struct.unpack_from(end + "Q", data, b + 0x18)[0],
                             struct.unpack_from(end + "Q", data, b + 0x20)[0],
                             struct.unpack_from(end + "I", data, b + 0x28)[0]))
            else:
                secs.append((struct.unpack_from(end + "I", data, b + 4)[0],
                             struct.unpack_from(end + "I", data, b + 0x10)[0],
                             struct.unpack_from(end + "I", data, b + 0x14)[0],
                             struct.unpack_from(end + "I", data, b + 0x18)[0]))
        dyn = next(((o, s, l) for (t, o, s, l) in secs if t == _SHT_DYNAMIC), None)
        if not dyn:
            return []
        dyn_off, dyn_sz, dyn_link = dyn
        if dyn_link >= len(secs):
            return []
        dynstr_off = secs[dyn_link][1]               # .dynamic's linked string table = .dynstr
        needed = []
        entsize = 16 if is64 else 8
        for off in range(dyn_off, dyn_off + dyn_sz, entsize):
            tag, val = struct.unpack_from(end + ("qQ" if is64 else "iI"), data, off)
            if tag == _DT_NULL:
                break
            if tag == _DT_NEEDED:
                z = data.index(b"\x00", dynstr_off + val)
                needed.append(data[dynstr_off + val:z].decode("latin-1"))
        return needed
    except Exception:  # noqa: BLE001 — any malformed ELF → "undeterminable", the caller fails closed
        return []


def _is_elf(path):
    """True iff `path` begins with the ELF magic. Distinguishes a Linux .so (which the DT_NEEDED reader
    CAN inspect) from a non-ELF engine binary — a Windows PE `.dll` ("MZ..") or a macOS Mach-O `.dylib` —
    which it CANNOT, so the toolchain guard must special-case those platforms rather than refuse them all
    as if they were unverifiable Linux binaries."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except Exception:  # noqa: BLE001
        return False


def _so_cuda_major(so_path):
    """The CUDA runtime MAJOR the engine .so DYNAMICALLY links (libcudart.so.<major>), or None if it
    cannot be determined (e.g. a statically-linked cudart, or a non-ELF binary). The regex is NOT
    end-anchored so a versioned SONAME like `libcudart.so.11.0` (CUDA ≤11) still yields major 11."""
    for lib in _elf_needed(so_path):
        m = re.match(r"libcudart\.so\.(\d+)", lib)
        if m:
            return int(m.group(1))
    return None


def assert_toolchain_compatible(so_path):
    """FORK-2 — FAIL-CLOSED CUDA-toolchain guard. The native engine loads IN-PROCESS (ctypes.CDLL) and
    shares ONE CUDA context + device with ComfyUI's torch. A torch built for one CUDA major running an
    engine .so built for a DIFFERENT CUDA major is an UNVERIFIED combination — and this project has been
    bitten by a CUDA concurrency defect that produced SILENT image/video corruption drifting with ambient
    VRAM and NEVER a clean fault (the C>1 non-blocking-lane-stream defect). A toolchain mismatch is that
    same class, so we REFUSE the load (naming both versions) rather than 'warn and proceed'. Refuses on a
    detected MISMATCH and on an UNVERIFIABLE combination (torch has no CUDA, or the .so's cudart major
    cannot be read). Escape hatch for a knowingly-safe combo (e.g. a static-cudart .so):
    QF_NATIVE_ALLOW_UNVERIFIED_TOOLCHAIN=1 (process env only — a workflow.json cannot set it).

    PLATFORM SCOPE (disclosed): the CUDA-version detection is Linux/ELF-only today. A NON-ELF engine
    binary — a Windows PE `.dll` or a macOS Mach-O `.dylib` — cannot be inspected by the DT_NEEDED reader,
    so on those platforms the guard fail-closes with a DISTINCT, disclosed message (naming the platform +
    the Linux-only limitation) that points at the override, rather than a silent generic refuse. Full
    PE/Mach-O toolchain detection is a follow-up; until it lands, Windows/macOS users set the override
    after confirming their torch + engine binary share a CUDA major."""
    if os.environ.get(_ENV_ALLOW_UNVERIFIED_TOOLCHAIN, "").strip().lower() in ("1", "true", "yes"):
        return
    try:
        import torch
        torch_cuda = torch.version.cuda            # e.g. "13.0"; None on a CPU-only torch build
    except Exception:  # noqa: BLE001
        torch_cuda = None
    torch_major = None
    if torch_cuda:
        try:
            torch_major = int(str(torch_cuda).split(".")[0])
        except Exception:  # noqa: BLE001
            torch_major = None
    # Non-ELF engine binary (Windows PE .dll / macOS Mach-O .dylib): the DT_NEEDED reader is ELF-only so
    # the .so's CUDA major cannot be read here. Fail closed (per the FORK-2 constraint) but with a
    # DISTINCT, DISCLOSED message naming the platform + the Linux-only-detection limitation + the override
    # — NOT the silent generic refuse that would deny every Windows/macOS load (matched or not) with no
    # explanation. Full PE/Mach-O toolchain detection is a follow-up (see the loader node's DESCRIPTION).
    if not _is_elf(so_path):
        import platform as _pf
        raise RuntimeError(
            f"qf_native: REFUSING to load — the CUDA-toolchain compatibility check is currently "
            f"Linux/ELF-only, and the engine binary ({so_path}) is a non-ELF {_pf.system() or 'non-Linux'} "
            f"binary whose CUDA version cannot be read here. torch is built for CUDA "
            f"{torch_cuda or 'none / CPU-only'}. Ensure your torch and the engine binary use the SAME CUDA "
            f"major, then set {_ENV_ALLOW_UNVERIFIED_TOOLCHAIN}=1 to proceed. (Full Windows/macOS toolchain "
            f"detection is a follow-up.)")
    so_major = _so_cuda_major(so_path)
    if torch_major is None or so_major is None:
        raise RuntimeError(
            f"qf_native: REFUSING to load — cannot verify the engine's CUDA toolchain matches torch's "
            f"(torch CUDA={torch_cuda or 'none / CPU-only'}, engine .so libcudart major="
            f"{so_major if so_major is not None else 'undeterminable'}). The engine loads in-process and "
            f"shares torch's CUDA context; an unverified toolchain combination can silently corrupt "
            f"output, so this is fail-closed. Install a torch + engine .so built for the SAME CUDA major, "
            f"or set {_ENV_ALLOW_UNVERIFIED_TOOLCHAIN}=1 if you KNOW this combination is safe.")
    if torch_major != so_major:
        raise RuntimeError(
            f"qf_native: REFUSING to load — CUDA toolchain MISMATCH. torch is built for CUDA {torch_cuda} "
            f"(major {torch_major}) but the engine .so ({so_path}) links libcudart.so.{so_major}. Running "
            f"a CUDA-{so_major} engine in-process with a CUDA-{torch_major} torch is an unverified "
            f"combination that can SILENTLY corrupt generated images/video (no crash). Use an engine .so "
            f"built for CUDA {torch_major}, or a torch built for CUDA {so_major}. (Override only if you "
            f"know it is safe: {_ENV_ALLOW_UNVERIFIED_TOOLCHAIN}=1.)")


def load_lib():
    """Load + bind the engine library once. Path comes ONLY from resolve_so_path() (never a workflow
    input). Takes NO argument on purpose — a `so_path` parameter is the attack surface just removed.
    FORK-2: the fail-closed CUDA-toolchain guard runs BEFORE ctypes.CDLL, so a mismatched combination is
    refused rather than dlopen'd into torch's live CUDA context."""
    global _LIB
    if _LIB is None:
        so_path = resolve_so_path()
        assert_toolchain_compatible(so_path)   # FORK-2 fail-closed torch-CUDA / .so-CUDA match check
        # An engine built with QF_WITH_QFA_ATTN=ON links libquantfunc_attention.so (the standalone
        # qfa INT8-QK attention library). Its build-time RUNPATH points at the BUILD box's dir,
        # which doesn't exist on a deployed machine — so when the consumable is shipped NEXT TO
        # the engine .so, preload it (RTLD_GLOBAL) and the dynamic linker resolves the dependency
        # from the already-loaded image, no RPATH surgery / LD_LIBRARY_PATH needed. Absent file =
        # no-op (an OFF-build engine has no such dependency); a PRESENT-but-broken qfa .so fails
        # loud here rather than as an opaque dlopen error on the engine line below.
        # Sidecar dependency preloads: a deployed engine .so may carry NEEDED libs whose
        # build-box versions differ from this machine's (measured: a scratch engine linked the
        # build box's dynamic OpenCV 4.5d; this box ships 4.6 -> dlopen refused — the SS7.5
        # portability class). Every lib*.so* placed NEXT TO the engine .so is preloaded; the
        # dynamic linker then satisfies the engine's NEEDED sonames from the already-loaded
        # images. A generic retry loop discovers dependency order (a lib whose own deps are not
        # loaded yet fails this pass and succeeds on a later one) — no hand-maintained list.
        # Modes: libquantfunc_attention.so keeps RTLD_GLOBAL (qfa symbol export — the proven
        # in-ComfyUI arm); everything else loads RTLD_LOCAL. LOCAL is deliberate: soname-based
        # NEEDED resolution does not require GLOBAL, and a GLOBAL OpenCV would inject cv::*
        # into the process global scope where it can hijack symbol binding of ComfyUI's own
        # bundled cv2 (different OpenCV version -> ABI-mismatch crashes in unrelated code).
        # A sidecar that never loads is skipped silently HERE — the engine dlopen below then
        # fails LOUD with the true unresolved soname, which is the honest error.
        so_dir = os.path.dirname(so_path)
        base = os.path.basename(so_path)
        # The DIRECTORY scan carries the same skip-silently-here discipline as each
        # individual sidecar load below (R1: an unreadable/vanished so_dir — e.g. a probe
        # monkeypatching resolve_so_path to a nonexistent path — must not crash the scan;
        # the engine dlopen below then fails LOUD with the true error, which keeps the
        # "sidecars skip silently, the engine fails loud" statement TRUE for this path too).
        try:
            entries = os.listdir(so_dir)
        except OSError:
            entries = []
        # Only REAL sonames: `lib*.so` or `lib*.so.<digits>[.<digits>...]`. A backup copy of a
        # sidecar (`libquantfunc_attention.so.prod-bak`, `.so.pre-<tag>`) is NOT a sidecar: gdb on
        # 2026-09-12 showed THREE builds of libquantfunc_attention.so mapped into one ComfyUI
        # process (the real one + two backups), and the process aborted at exit with
        # "double free or corruption" from their colliding static destructors.
        # A per-SM kernel .so (QF_SPLIT_KERNEL_SO two-.so build: libquantfunc_kernels_sm<NN>.so)
        # is NOT preloaded as a sidecar. The engine .so DT_NEEDEDs it and finds it via its own
        # $ORIGIN rpath, so it loads as a member of the ENGINE's dlopen group — where the host<->kernel
        # symbols (engine's kernel launchers + the kernel's cachedConvPlanWorkspaceCap) resolve
        # bidirectionally in-group. Eagerly preloading it here (before the engine) would fail: its
        # host symbol is not yet available. Keeping it OUT of a preload also keeps the kernel's many
        # exported symbols out of the process-global scope (no clash with torch's own kernels).
        pending = sorted(
            f for f in entries
            if _SONAME_RE.fullmatch(f) and f != base
            and not f.startswith("libquantfunc_kernels_sm")
        )
        for _ in range(max(1, len(pending))):
            still = []
            for f in pending:
                mode = ctypes.RTLD_GLOBAL if f == "libquantfunc_attention.so" else ctypes.RTLD_LOCAL
                try:
                    ctypes.CDLL(os.path.join(so_dir, f), mode=mode)
                except OSError:
                    still.append(f)
            if not still:
                break
            pending = still
        _LIB = _bind(ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL))
        _log_lib_fingerprint(_LIB, so_path)
    return _LIB


def _log_lib_fingerprint(lib, so_path):
    """[F6, 2026-09-19] ONE line naming the engine library this process actually dlopen'd — path, size, mtime,
    md5, and the engine's own quantfunc_version(). MEASURED need: the 远程-linux 5090 box ran a 4-day-old engine
    for days, and later a deployed library was silently replaced by an older file (found from a backup's mtime,
    not from any log). With this line the ComfyUI log states which binary produced every run. Never raises."""
    try:
        import hashlib
        st = os.stat(so_path)
        h = hashlib.md5()
        with open(so_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        ver = "?"
        try:
            fn = lib.quantfunc_version
            fn.restype = ctypes.c_char_p
            fn.argtypes = []
            ver = (fn() or b"?").decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 — an old .so without the symbol still gets the file fingerprint
            pass
        print("[qf_native] engine lib: %s  size=%d  mtime=%s  md5=%s  quantfunc_version=%s"
              % (so_path, st.st_size, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                 h.hexdigest(), ver), flush=True)
    except Exception as e:  # noqa: BLE001
        print("[qf_native] engine lib: %s (fingerprint unavailable: %r)" % (so_path, e), flush=True)


def last_err(lib):
    e = lib.quantfunc_last_error()
    return e.decode("utf-8", "replace") if e else "(none)"


def _enc(s):
    return s.encode("utf-8") if isinstance(s, str) else s


def _refuse_session_knobs_in_create(config_json):
    """[session-knobs] STRUCTURAL half of the runtime-adjustable session-knob guarantee, sealed
    at the REAL create boundary (every quantfunc_create goes through create_pipeline, so a
    caller cannot bypass it by skipping the package's _get_engine cache wrapper — reviewer-B
    hard-seal). The refused set = EVERY session knob QFSessionModelMixin.residency_opts()
    injects into denoise_begin: the runtime cache/sparse keys (+ the
    loader-widget spellings step_cache/block_cache/sparse). In a create config any of them would enter the
    pipeline cache identity upstream and silently reintroduce a full model rebuild on every
    widget change — refuse loud, both the dict and the pre-serialized-string form."""
    cfg = config_json
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except Exception:  # noqa: BLE001 — unparseable JSON: the engine's own create refuses it loudly
            return

    def _scan(obj):
        # RECURSIVE (reviewer-D LOW): the knob nested anywhere in the config tree would
        # equally enter the json-hashed cache identity — refuse it at any depth.
        if isinstance(obj, dict):
            # session knobs (runtime, re-applied per denoise_begin) — NONE may enter the
            # create config / ckey: the runtime session keys (the same guarantee class; a
            # create-side leak would rebuild the pipeline per widget change).
            if any(k in obj for k in ("cache_mode", "cache_thresh",
                                      "step_cache", "block_cache", "step_cache_thresh",
                                      "block_cache_thresh", "sparse", "sparse_cdf")):
                return True
            return any(_scan(v) for v in obj.values())
        if isinstance(obj, list):
            return any(_scan(v) for v in obj)
        return False
    if _scan(cfg):
        raise RuntimeError(
            "qf_native: a runtime SESSION knob (cache_mode / cache_thresh / "
            "step_cache — they ride every denoise_begin via "
            "QFSessionModelMixin.residency_opts) must never appear anywhere in a create "
            "config — that would bake it into the pipeline cache identity and rebuild "
            "the whole pipeline on every widget change.")


def create_pipeline(lib, *, model_dir, transformer_path=None, model_backend="svdq",
                    device_idx=0, config_json=None):
    _refuse_session_knobs_in_create(config_json)   # [session-knobs] session knob ≠ create key
    # [metadata-KV disk cache — user 2026-09-01 "为啥metadata每次都重新请求后端 不是有缓存吗"]
    # The engine HAS a two-tier keymap/metadata cache (process mem → disk CIPHERTEXT at
    # <_cache_dir>/.quantfunc_keymap_cache/), but the disk tier arms only when create passes
    # `_cache_dir` — which this plugin never did, so every ComfyUI RESTART re-fetched from the
    # backend. Default it to the plugin's own cache/ dir (ciphertext-only on disk; decrypt
    # stays in-memory per use — no security change). An explicit caller _cache_dir still wins.
    try:
        _cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(_cdir, exist_ok=True)
        if config_json is None:
            config_json = {"_cache_dir": _cdir}
        elif isinstance(config_json, dict):
            config_json.setdefault("_cache_dir", _cdir)
        else:
            _cj = json.loads(config_json)
            if isinstance(_cj, dict) and "_cache_dir" not in _cj:
                _cj["_cache_dir"] = _cdir; config_json = json.dumps(_cj)
    except Exception:
        pass  # cache dir is an optimization — never block create on it
    p = InitParams()
    p._keep = [_enc(model_dir), _enc(transformer_path), _enc(model_backend)]
    p.model_dir = p._keep[0]
    p.transformer_path = p._keep[1]
    p.model_backend = p._keep[2]
    p.device_idx = device_idx
    if config_json is not None:
        cj = json.dumps(config_json) if isinstance(config_json, dict) else config_json
        p._keep.append(_enc(cj))
        p.config_json = p._keep[-1]
    handle = ctypes.c_void_p()
    st = lib.quantfunc_create(ctypes.byref(p), ctypes.byref(handle))
    if st != QUANTFUNC_OK:
        raise RuntimeError(f"quantfunc_create failed st={st}: {last_err(lib)}")
    return handle


class ResidentEstimateParams(ctypes.Structure):
    """Mirror of quantfunc_resident_estimate_params_t (engine >= 36f4ed2cb)."""
    _fields_ = [("model_dir", ctypes.c_char_p), ("transformer_weights", ctypes.c_char_p),
                ("server_url", ctypes.c_char_p), ("api_key", ctypes.c_char_p), ("device_idx", ctypes.c_int)]


def estimate_resident_bytes(lib, model_dir, device_idx=0, transformer_path=None, server_url=None, api_key=None):
    """EXACT pre-load resident-VRAM estimate from the ENGINE's own loader law (quantfunc_estimate_resident_bytes:
    safetensors header x the per-slot packed forms of this device's SM tier, arena-page-rounded, + non-block residents;
    header-only, no model load, no VRAM). This is what comfy's ledger should charge BEFORE create — the on-disk size
    under-reports a packed svdq transformer by ~41 % on SM89 (measured; see estimate_footprint_bytes). Returns 0 when
    the .so predates the API, the call fails, or the law does not model the checkpoint (QUANTFUNC_ERROR_UNSUPPORTED —
    never a silently sized guess); callers then fall back to the disk proxy and the live post-load measurement."""
    fn = getattr(lib, "quantfunc_estimate_resident_bytes", None)
    if fn is None or not model_dir:
        return 0
    try:
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.POINTER(ResidentEstimateParams), ctypes.POINTER(ctypes.c_uint64)]
        p = ResidentEstimateParams(model_dir=_enc(model_dir), transformer_weights=_enc(transformer_path) if transformer_path else None,
                                   server_url=_enc(server_url) if server_url else None, api_key=_enc(api_key) if api_key else None,
                                   device_idx=int(device_idx))
        out = ctypes.c_uint64(0)
        st = fn(ctypes.byref(p), ctypes.byref(out))
        if st != 0:
            _dbg_prof(f"resident-estimate unavailable (status {st}): {last_err(lib)}")
            return 0
        return int(out.value)
    except Exception as ex:  # noqa: BLE001 — an estimate must never break loading
        _dbg_prof(f"resident-estimate failed: {ex!r}")
        return 0


def estimate_footprint_bytes(*paths):
    """ESTIMATE (NOT a live-VRAM measurement) of the engine's resident footprint, from the ON-DISK
    packed size (os.path.getsize) of the given weight file(s)/dir(s). qf_modelpatcher's model_size()/
    loaded_size() report this to ComfyUI's memory ledger so a sibling native model is not placed into
    VRAM the engine already holds. HONEST scope, to avoid the "measured the wrong quantity" trap: this
    is a DISK-SIZE PROXY, not comfy's own convention (live state_dict().nbytes) — for a packed quantized
    transformer the on-disk bytes track resident VRAM closely, but call it an estimate, never 'measured'.
    Pass ONLY engine-resident components (the transformer dir); VAE/TE stay native comfy nodes that
    comfy already accounts for, so the caller MUST NOT include them (an over-report evicts fitting
    siblings)."""
    total = 0
    for pth in paths:
        if not pth or not os.path.exists(pth):
            continue
        if os.path.isfile(pth):
            total += os.path.getsize(pth)
        else:
            for root, _dirs, files in os.walk(pth):
                for f in files:
                    if f.endswith((".safetensors", ".bin", ".pt")) or "transformer" in f:
                        try:
                            total += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
    return total


class QFEngineHandle:
    """Owns the .so + created pipeline + the (single, per-pipeline) open denoise session.
    Tracks the session HERE (not only on the model shim) so a stale session from a failed run is
    cleaned up before the next begin, and destroy() always closes it."""
    def __init__(self, lib, pipeline, footprint_bytes=0):
        self.lib = lib
        self.pipeline = pipeline
        self.footprint_bytes = int(footprint_bytes)
        self.current_session = None          # ctypes.c_void_p of the open session, or None
        self.step_count = 0                  # total denoise_step calls (instrument)
        self.sampler_step_count = 0          # distinct sampler steps (instrument)
        self.unloaded = False                # co-eviction: True after unload_vram() freed VRAM (auto-reloads on next generate)

    def end_session_if_open(self):
        """Close the open denoise session if any. Returns (was_open, ok): a NON-OK engine status
        (#2 CR: the int from quantfunc_denoise_end was previously discarded — no ctypes errcheck, so
        it never raised) is surfaced as ok=False so the caller can log/fail loudly rather than clear
        current_session on the false belief that the engine actually closed it."""
        if self.current_session is None:
            return (False, True)
        ok = True
        try:
            st = self.lib.quantfunc_denoise_end(self.current_session)
            if st != QUANTFUNC_OK:
                ok = False
                try:
                    print(f"[qf_native] WARNING quantfunc_denoise_end returned status={st}: "
                          f"{last_err(self.lib)}", flush=True)
                except Exception:  # noqa: BLE001
                    pass
        except Exception as _end_exc:  # noqa: BLE001 — end is best-effort on teardown
            ok = False
            try:  # R7 observability: this branch previously left no trail (leg-1 was
                #   diagnosed FROM logs — a silent branch here would blind the next diagnosis)
                print(f"[qf_native] WARNING quantfunc_denoise_end raised: {_end_exc!r}", flush=True)
            except Exception:  # noqa: BLE001
                pass
        # RETAIN the pointer on a refused end (2026-08-24 busy incident, leg 2). The old
        # "clear regardless" rationale was FALSE: no _begin blocks on a stale pointer (all
        # call this bare and proceed), but clearing it made an engine-side-open session
        # PERMANENTLY unreachable — a step-in-flight refusal left the handle Sessioning
        # with no python pointer, so every later begin was "pipeline busy" until the 5-min
        # engine watchdog. Retention is safe both ways: a step-in-flight session (the
        # step drains in seconds) gets CLOSED by the next end attempt; an already-reaped
        # session's end keeps refusing harmlessly (the by-id session slab makes a dead
        # pointer's end a typed refusal, never a UAF) and the next successful begin
        # overwrites the pointer anyway.
        if ok:
            self.current_session = None
        return (True, ok)

    def partial_unload_vram(self, bytes_requested):
        """Shed ONLY ~bytes_requested of trailing transformer blocks (engine
        quantfunc_partial_unload; family-generic Pipeline base — LTX/H3 single
        transformer, wan main expert). Returns the engine-reported freed bytes
        (0 = unsupported / nothing shed / old .so — caller falls back to
        unload_vram). Never marks the handle `unloaded`: the model stays LIVE
        with a smaller resident prefix; the next session begin restores it."""
        if self.pipeline is None or self.unloaded or bytes_requested <= 0:
            return 0
        if not hasattr(self.lib, "quantfunc_partial_unload"):
            return 0
        # Y vuln fix: comfy's free_memory sweep passes a HUGE "free everything" sentinel
        # (measured 1e32 as float) — int(1e32) overflows c_uint64 (OverflowError swallowed
        # -> silent 0 -> full-unload fallback worked only by coincidence). Clamp into the
        # engine's saturating range explicitly (the engine treats >= total weight bytes as
        # "shed all sheddable").
        bytes_requested = min(int(bytes_requested), (1 << 63) - 1)
        if self.current_session is not None:
            # Retention (2026-08-24): a retained stale pointer must not permanently refuse
            # VRAM reclaim — try the end first; a genuinely open session keeps refusing.
            self.end_session_if_open()
        if self.current_session is not None:
            _dbg_prof("partial_unload refused: session open")
            return 0                        # mid-session: refuse (lease would too)
        freed = ctypes.c_int64(0)
        try:
            st = self.lib.quantfunc_partial_unload(self.pipeline,
                                                   ctypes.c_uint64(int(bytes_requested)),
                                                   ctypes.byref(freed))
        except Exception as ex:  # noqa: BLE001
            _dbg_prof(f"partial_unload EXCEPTION: {ex!r}")
            return 0
        if st != QUANTFUNC_OK:
            _dbg_prof(f"partial_unload engine status={st}: {last_err(self.lib)}")
            return 0
        return int(freed.value)

    def resident_vram_bytes(self):
        """Query the native device-scoped residency counter; failure is not zero.

        A missing pipeline is known to hold nothing. An unload flag is not a
        measurement: native full release may leave live or pinned allocations.
        The current ABI selects a device, not an individual pipeline owner;
        callers must not sum this value once per shared model/engine.
        """
        if self.pipeline is None:
            return 0
        if not hasattr(self.lib, "quantfunc_resident_vram_bytes"):
            raise RuntimeError("QuantFunc library lacks quantfunc_resident_vram_bytes; update the native library")
        out = ctypes.c_uint64(0)
        st = self.lib.quantfunc_resident_vram_bytes(self.pipeline, ctypes.byref(out))
        if st != QUANTFUNC_OK:
            raise RuntimeError(f"QuantFunc residency query failed: {last_err(self.lib)}")
        return int(out.value)

    def vram_need_bytes(self, latent_shape):
        """How much MORE VRAM the engine needs beyond what it holds for ONE forward of `latent_shape`
        ([B,C,H,W] image / [B,C,T,H,W] video — the exact latent the session will step on) via
        quantfunc_vram_need_bytes: the primary transformer's MEASURED working set under that shape (else the same
        spatial shape at another batch scaled, else the model's config estimate) + the engine's plan margin − the
        allocator's cached pool it already holds. 0 = UNKNOWN (old .so / unloaded / no pipeline / nothing measured
        yet) — the caller treats it as "no extra number", never as "needs nothing"."""
        if self.pipeline is None or self.unloaded or not hasattr(self.lib, "quantfunc_vram_need_bytes"):
            return 0
        dims = [int(d) for d in latent_shape]
        if not dims or any(d <= 0 for d in dims):
            return 0
        arr = (ctypes.c_int64 * len(dims))(*dims)
        out = ctypes.c_uint64(0)
        try:
            st = self.lib.quantfunc_vram_need_bytes(self.pipeline, arr, len(dims), ctypes.byref(out))
        except Exception:  # noqa: BLE001
            return 0
        return int(out.value) if st == QUANTFUNC_OK else 0

    def unload_vram(self):
        """Co-eviction (#4): free the engine's VRAM (quantfunc_unload_sync — GPU->CPU, keeps the CPU
        backup; the pipeline auto-reloads on the next generate). Idempotent. Returns the freed byte
        ESTIMATE (footprint) so comfy's ledger can HONESTLY credit it, or 0 if nothing was freed."""
        if self.pipeline is None or self.unloaded:
            return 0
        import os as _os
        if _os.environ.get("QF_NATIVE_PROF") == "1":
            import traceback as _tb
            frames = _tb.extract_stack(limit=5)[:-1]
            chain = " <- ".join(f"{_os.path.basename(f.filename)}:{f.lineno}:{f.name}"
                                for f in reversed(frames))
            print(f"[qf_prof] unload_vram CALLER: {chain}", flush=True)
        self.end_session_if_open()          # a live session on unloaded VRAM would be a UAF on reuse
        if self.current_session is not None:
            raise RuntimeError("QuantFunc cannot unload VRAM while a session is still active")
        st = self.lib.quantfunc_unload_sync(self.pipeline)
        if st != QUANTFUNC_OK:
            raise RuntimeError(f"QuantFunc VRAM unload failed: {last_err(self.lib)}")
        self.unloaded = True
        return int(self.footprint_bytes)

    def destroy(self):
        self.end_session_if_open()
        if self.pipeline is not None:
            try:
                self.lib.quantfunc_destroy(self.pipeline)
            except Exception:  # noqa: BLE001
                pass
            self.pipeline = None
