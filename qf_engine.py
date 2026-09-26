"""qf_native.qf_engine — self-contained ctypes bridge to libquantfunc.so for the
native ComfyUI loader. Structs mirror include/quantfunc.h (session structs copied
verbatim from the PROVEN tests/scripts/native_session_t1.py). No tests/lib dependency.
"""
import builtins
import contextlib
import ctypes
from contextvars import ContextVar
import functools
import glob
import inspect
import json
import logging
import mmap
import os
import time
import platform
import re
import shutil
import struct
import sys
import operator
import threading
import traceback
import types
import weakref
from typing import NamedTuple, Optional

QUANTFUNC_OK = 0
QUANTFUNC_RESOURCE_ABI_VERSION = 1
QUANTFUNC_RESOURCE_CREATION_ABI_VERSION = 1
QUANTFUNC_RESOURCE_RESIDENCY_ABI_VERSION = 2
QUANTFUNC_RESOURCE_DOMAIN_ABI_VERSION = 1
QUANTFUNC_RESOURCE_CAPACITY_ABI_VERSION = 1
QUANTFUNC_RESOURCE_LIFECYCLE_ABI_VERSION = 1
QUANTFUNC_ERROR_INVALID_ARG = 1
QUANTFUNC_ERROR_UNSUPPORTED = 8
QUANTFUNC_RESOURCE_READY = 0
QUANTFUNC_RESOURCE_BUSY = 1
QUANTFUNC_RESOURCE_UNKNOWN = 2
QUANTFUNC_RESOURCE_CLOSED = 3
QUANTFUNC_RESOURCE_CAPACITY_UNSUPPORTED = 4
QUANTFUNC_RESOURCE_CAP_QUERY = 1
QUANTFUNC_RESOURCE_CAP_RELEASE_ALL = 4
QUANTFUNC_RESOURCE_PHASE_SHARED = 1
QUANTFUNC_RESOURCE_PHASE_PREPARED = 2
QUANTFUNC_RESOURCE_PHASE_ATTACHED = 3
QUANTFUNC_RESOURCE_PHASE_CLOSED = 4
# quantfunc_dtype_t: FP32=0, FP16=1, BF16=2 (matches QF_DTYPE in the harness)
QF_FP32, QF_FP16, QF_BF16 = 0, 1, 2


def _dbg_prof(msg):
    """QF_NATIVE_PROF-gated diagnostic line (the [qf_prof] channel the perf probes use)."""
    import os as _os
    if _os.environ.get("QF_NATIVE_PROF") == "1":
        say(f"[qf_prof] {msg}", flush=True)

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


class _ResourceSnapshot(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("state", ctypes.c_uint32), ("capabilities", ctypes.c_uint32),
        ("owner_epoch", ctypes.c_uint64), ("device", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("cca_live", ctypes.c_uint64), ("cca_cached", ctypes.c_uint64),
        ("cca_deferred", ctypes.c_uint64), ("arena_backed", ctypes.c_uint64),
        ("arena_pinned", ctypes.c_uint64),
    ]


class _ResourceRelease(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("state", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
        ("freed_bytes", ctypes.c_uint64),
    ]


class _ResourceResidency(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("state", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
        ("resident_bytes", ctypes.c_uint64),
        ("streamed_blocks", ctypes.c_uint64), ("watermark_drops", ctypes.c_uint64),
        ("recycled_pages", ctypes.c_uint64),
    ]


class _ResourceDomain(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("state", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
        ("resident_bytes", ctypes.c_uint64),
    ]


class _ResourceCapacity(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("state", ctypes.c_uint32), ("component_count", ctypes.c_uint32),
        ("required_persistent_bytes", ctypes.c_uint64),
    ]


class _ResourceLifecycle(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("state", ctypes.c_uint32), ("phase", ctypes.c_uint32),
    ]


class ResourceLifecycle(NamedTuple):
    state: int
    phase: Optional[int]


class ResourceResidency(NamedTuple):
    """Native accounted occupancy; not demand, capacity or complete backend coverage. The three counters are the
    device's streaming counters (process lifetime, every identity on the device): block faults served by streaming,
    watermark ranks lowered, pages recycled between streamed blocks. Non-Ready fields are None."""
    state: int
    resident_bytes: Optional[int]
    streamed_blocks: Optional[int]
    watermark_drops: Optional[int]
    recycled_pages: Optional[int]


class ResourceDomainResidency(NamedTuple):
    state: int
    resident_bytes: Optional[int]


class ResourceCapacity(NamedTuple):
    state: int
    component_count: Optional[int]
    required_persistent_bytes: Optional[int]


class ColdNeed(NamedTuple):
    """#738 The engine's COLD create-time need. `bytes` None = the engine cannot say; `note` says why - None only when
    that is by design (an older library without the entry), so an engine failure never reads like an older library."""
    bytes: Optional[int]
    note: Optional[str]


class ResourceSnapshot(NamedTuple):
    """Native categories, not ModelPatcher capacity or a demand estimate.

    Non-Ready byte fields are None; pinned overlaps backed and is not additive.
    """
    state: int
    device: int
    owner_epoch: int
    capabilities: int
    cca_live: Optional[int]
    cca_cached: Optional[int]
    cca_deferred: Optional[int]
    arena_backed: Optional[int]
    arena_pinned: Optional[int]


class ResourceRelease(NamedTuple):
    state: int
    freed_bytes: Optional[int]


def _bind_resource_api(lib):
    """Require the complete versioned interface only when it is requested."""
    v = ctypes.c_void_p
    signatures = {
        "quantfunc_last_error": (ctypes.c_char_p, []),
        "quantfunc_resource_acquire": (ctypes.c_int, [v, ctypes.c_uint32, ctypes.POINTER(v)]),
        "quantfunc_resource_acquire_shared": (ctypes.c_int, [ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(v)]),
        "quantfunc_resource_destroy": (None, [v]),
        "quantfunc_resource_query": (ctypes.c_int, [v, ctypes.POINTER(_ResourceSnapshot)]),
        "quantfunc_resource_release_eligible": (ctypes.c_int, [v, ctypes.c_uint64, ctypes.POINTER(_ResourceRelease)]),
    }
    for name in signatures:
        if not hasattr(lib, name):
            raise RuntimeError(f"QuantFunc library lacks {name}; update the native library")
    for name, (result, args) in signatures.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = result, args


def _destroy_resource_view(lib, pointer):
    # Adoption rollback and an already-registered finalizer may both run after
    # interruption. Consume the shared holder before crossing the C boundary.
    address = pointer.value
    if address is not None:
        pointer.value = None
        lib.quantfunc_resource_destroy(ctypes.c_void_p(address))


FACTORY_PREPARE_ONLY = ContextVar("quantfunc_factory_prepare_only", default=False)


class NativeContractUnavailable(RuntimeError):
    """A required native authority is absent; never substitute a numeric zero."""


def library_identity(lib):
    """Loaded image identity, not its filename (load_lib retains the CDLL)."""
    handle = getattr(lib, "_handle", None)
    return ("dso", int(handle)) if handle is not None else ("library", lib)


class NativeResource:
    """Retained native view; no scheduler, byte policy or full-unload claim.

    Acquire must be synchronized with model destruction by the model owner.
    Afterwards the native view survives model destruction. This lock serializes
    view operations with close because ctypes may release the GIL.
    """
    def __init__(self, lib, pointer):
        self._lib, self._pointer = lib, pointer
        self._lock = threading.Lock()
        self._cold_need_warned = set()   # each reason the cold need is unknown is warned about once per resource
        self._finalizer = weakref.finalize(self, _destroy_resource_view, lib, pointer)

    @classmethod
    def _adopt(cls, lib, pointer):
        try:
            return cls(lib, pointer)
        except BaseException:
            _destroy_resource_view(lib, pointer)
            raise

    @classmethod
    def acquire(cls, lib, pipeline):
        _bind_resource_api(lib)
        pointer = ctypes.c_void_p()
        status = lib.quantfunc_resource_acquire(pipeline, QUANTFUNC_RESOURCE_ABI_VERSION, ctypes.byref(pointer))
        if status != QUANTFUNC_OK or not pointer:
            raise RuntimeError(f"QuantFunc resource acquisition failed: {last_err(lib)}")
        return cls._adopt(lib, pointer)

    @classmethod
    def shared(cls, lib, device):
        device = operator.index(device)
        if not 0 <= device < (1 << 31):
            raise ValueError("resource device must fit nonnegative int32")
        _bind_resource_api(lib)
        pointer = ctypes.c_void_p()
        status = lib.quantfunc_resource_acquire_shared(device, QUANTFUNC_RESOURCE_ABI_VERSION, ctypes.byref(pointer))
        if status != QUANTFUNC_OK or not pointer:
            raise RuntimeError(f"QuantFunc shared resource acquisition failed: {last_err(lib)}")
        return cls._adopt(lib, pointer)

    @classmethod
    def prepare(cls, lib, device):
        """Create owned identity before loading; no budget is implied."""
        device = operator.index(device)
        if not 0 <= device < (1 << 31):
            raise ValueError("resource device must fit nonnegative int32")
        _bind_resource_api(lib)
        function = getattr(lib, "quantfunc_resource_prepare", None)
        if function is None:
            raise RuntimeError("QuantFunc library lacks quantfunc_resource_prepare; update the native library")
        function.restype = ctypes.c_int
        function.argtypes = [ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        pointer = ctypes.c_void_p()
        status = function(device, QUANTFUNC_RESOURCE_CREATION_ABI_VERSION, ctypes.byref(pointer))
        if status != QUANTFUNC_OK or not pointer:
            raise RuntimeError(f"QuantFunc resource preparation failed: {last_err(lib)}")
        return cls._adopt(lib, pointer)

    def _check_open(self):
        if not self._finalizer.alive:
            raise RuntimeError("QuantFunc resource view is closed")

    def configure(self, params):
        """Bind entry options, not a capacity estimate."""
        if not isinstance(params, InitParams):
            raise TypeError("resource configuration requires InitParams")
        _refuse_session_knobs_in_create(params.config_json)
        with self._lock:
            self._check_open()
            function = getattr(self._lib, "quantfunc_resource_configure", None)
            if function is None:
                raise NativeContractUnavailable("QuantFunc library lacks quantfunc_resource_configure")
            function.restype = ctypes.c_int
            function.argtypes = [ctypes.POINTER(InitParams), ctypes.c_void_p]
            if function(ctypes.byref(params), self._pointer) != QUANTFUNC_OK:
                raise RuntimeError(f"QuantFunc resource configuration failed: {last_err(self._lib)}")

    def query_capacity(self):
        """Configured Prepared persistent capacity; every non-Ready result is nonnumeric."""
        with self._lock:
            self._check_open()
            function = getattr(self._lib, "quantfunc_resource_query_capacity", None)
            if function is None:
                raise NativeContractUnavailable(
                    "QuantFunc library lacks quantfunc_resource_query_capacity")
            function.restype = ctypes.c_int
            function.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ResourceCapacity)]
            out = _ResourceCapacity(ctypes.sizeof(_ResourceCapacity),
                                    QUANTFUNC_RESOURCE_CAPACITY_ABI_VERSION)
            status = function(self._pointer, ctypes.byref(out))
            if status == QUANTFUNC_ERROR_UNSUPPORTED or out.state == QUANTFUNC_RESOURCE_CAPACITY_UNSUPPORTED:
                raise NativeContractUnavailable(
                    f"QuantFunc prepared capacity is unsupported: {last_err(self._lib)}")
            if status != QUANTFUNC_OK:
                raise RuntimeError(f"QuantFunc resource capacity query failed: {last_err(self._lib)}")
            if out.state != QUANTFUNC_RESOURCE_READY:
                return ResourceCapacity(out.state, None, None)  # BUSY is transient: the caller may re-read
            if out.component_count == 0 or out.required_persistent_bytes == 0:
                raise NativeContractUnavailable(
                    "QuantFunc prepared capacity returned no complete persistent components")
            return ResourceCapacity(out.state, int(out.component_count),
                                    int(out.required_persistent_bytes))

    def cold_vram_need_bytes(self):
        """#738 The engine's COLD create-time device need for this configured Prepared resource
        (quantfunc_resource_vram_need_bytes: its CCA-resident persistent bytes + its largest paged block + the ambient
        reserve - the plan side's number, header facts only), as a ColdNeed. When the engine cannot say, the caller keeps
        its own floor and never reads that as 0: an older library is silent (by design); a layout the plan does not model
        or an engine failure is named in the note and warned about once (plugin CR D-V2)."""
        with self._lock:
            self._check_open()
            function = getattr(self._lib, "quantfunc_resource_vram_need_bytes", None)
            if function is None:
                return ColdNeed(None, None)
            function.restype = ctypes.c_int
            function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
            out = ctypes.c_uint64(0)
            status = function(self._pointer, ctypes.byref(out))
            if status == QUANTFUNC_OK and out.value:
                return ColdNeed(int(out.value), None)
            if status == QUANTFUNC_ERROR_INVALID_ARG:
                raise RuntimeError(f"QuantFunc cold VRAM need refused: {last_err(self._lib)}")
            note = (f"not estimable for this layout: {last_err(self._lib)}" if status == QUANTFUNC_ERROR_UNSUPPORTED
                    else f"unknown (engine status {status}: {last_err(self._lib)})")
            if note not in self._cold_need_warned:
                self._cold_need_warned.add(note)
                say(f"[qf_native] WARNING the engine's cold VRAM need is {note}; ComfyUI's own estimate is the floor")
            return ColdNeed(None, note)

    def query(self):
        with self._lock:
            self._check_open()
            out = _ResourceSnapshot(ctypes.sizeof(_ResourceSnapshot), QUANTFUNC_RESOURCE_ABI_VERSION)
            if self._lib.quantfunc_resource_query(self._pointer, ctypes.byref(out)) != QUANTFUNC_OK:
                raise RuntimeError(f"QuantFunc resource query failed: {last_err(self._lib)}")
            counts = (out.cca_live, out.cca_cached, out.cca_deferred, out.arena_backed, out.arena_pinned)
            return ResourceSnapshot(out.state, out.device, out.owner_epoch, out.capabilities,
                                    *(counts if out.state == QUANTFUNC_RESOURCE_READY else (None,) * 5))

    def residency(self):
        with self._lock:
            self._check_open()
            function = getattr(self._lib, "quantfunc_resource_query_residency", None)
            if function is None:
                raise RuntimeError("QuantFunc library lacks quantfunc_resource_query_residency; update the native library")
            function.restype = ctypes.c_int
            function.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ResourceResidency)]
            out = _ResourceResidency(ctypes.sizeof(_ResourceResidency), QUANTFUNC_RESOURCE_RESIDENCY_ABI_VERSION)
            if function(self._pointer, ctypes.byref(out)) != QUANTFUNC_OK:
                raise RuntimeError(f"QuantFunc resource residency query failed: {last_err(self._lib)}")
            if out.state != QUANTFUNC_RESOURCE_READY:
                return ResourceResidency(out.state, None, None, None, None)
            return ResourceResidency(out.state, out.resident_bytes, out.streamed_blocks, out.watermark_drops,
                                     out.recycled_pages)

    def query_domain_residency(self):
        """One coherent Shared-domain actual snapshot; Busy/Unknown/Closed never become zero."""
        with self._lock:
            self._check_open()
            function = getattr(self._lib, "quantfunc_resource_query_domain_residency", None)
            if function is None:
                raise NativeContractUnavailable(
                    "QuantFunc library lacks quantfunc_resource_query_domain_residency")
            function.restype = ctypes.c_int
            function.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ResourceDomain)]
            out = _ResourceDomain(ctypes.sizeof(_ResourceDomain),
                                  QUANTFUNC_RESOURCE_DOMAIN_ABI_VERSION)
            if function(self._pointer, ctypes.byref(out)) != QUANTFUNC_OK:
                raise RuntimeError(
                    f"QuantFunc domain residency query failed: {last_err(self._lib)}")
            if out.state != QUANTFUNC_RESOURCE_READY:
                return ResourceDomainResidency(out.state, None)  # BUSY is transient: the caller may re-read
            return ResourceDomainResidency(out.state, int(out.resident_bytes))

    def lifecycle(self):
        with self._lock:
            self._check_open()
            function = getattr(self._lib, "quantfunc_resource_query_lifecycle", None)
            if function is None:
                raise NativeContractUnavailable("QuantFunc library lacks quantfunc_resource_query_lifecycle")
            function.restype = ctypes.c_int
            function.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ResourceLifecycle)]
            out = _ResourceLifecycle(ctypes.sizeof(_ResourceLifecycle), QUANTFUNC_RESOURCE_LIFECYCLE_ABI_VERSION)
            if function(self._pointer, ctypes.byref(out)) != QUANTFUNC_OK:
                raise RuntimeError(f"QuantFunc resource lifecycle query failed: {last_err(self._lib)}")
            return ResourceLifecycle(out.state, out.phase if out.state == QUANTFUNC_RESOURCE_READY else None)

    def enroll_host(self):
        """Tell the engine a host framework (ComfyUI) shares this view's device (quantfunc_resource_enroll_host): from
        then on another library's allocation that fails for lack of memory takes what the card lacks from the engine's
        reclaimable memory, then retries. It grants and reserves nothing. Idempotent; a repeated call picks up the
        libraries loaded since."""
        with self._lock:
            self._check_open()
            function = getattr(self._lib, "quantfunc_resource_enroll_host", None)
            if function is None:
                raise RuntimeError("QuantFunc library lacks quantfunc_resource_enroll_host; update the native library")
            function.restype = ctypes.c_int
            function.argtypes = [ctypes.c_void_p]
            if function(self._pointer) != QUANTFUNC_OK:
                raise RuntimeError(f"QuantFunc host enrollment failed: {last_err(self._lib)}")

    def release_eligible(self, requested):
        requested = operator.index(requested)
        if not 0 <= requested < (1 << 64):
            raise ValueError("resource release request must fit uint64")
        with self._lock:
            self._check_open()
            out = _ResourceRelease(ctypes.sizeof(_ResourceRelease), QUANTFUNC_RESOURCE_ABI_VERSION)
            if self._lib.quantfunc_resource_release_eligible(self._pointer, requested, ctypes.byref(out)) != QUANTFUNC_OK:
                raise RuntimeError(f"QuantFunc eligible release failed: {last_err(self._lib)}")
            return ResourceRelease(out.state, out.freed_bytes if out.state == QUANTFUNC_RESOURCE_READY else None)

    def release_all(self):
        """Checked exact-resource full release; Ready alone carries freed bytes.

        Does not revive a Closed identity. A retained Closed view is
        deliberately forwarded to native for final old-owner cleanup.
        """
        with self._lock:
            self._check_open()
            function = getattr(self._lib, "quantfunc_resource_release_all", None)
            if function is None:
                raise NativeContractUnavailable("QuantFunc native resource full eviction is unsupported: "
                                                "quantfunc_resource_release_all is unavailable")
            function.restype = ctypes.c_int
            function.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ResourceRelease)]
            out = _ResourceRelease(ctypes.sizeof(_ResourceRelease), QUANTFUNC_RESOURCE_ABI_VERSION)
            if function(self._pointer, ctypes.byref(out)) != QUANTFUNC_OK:
                raise RuntimeError(f"QuantFunc full resource release failed: {last_err(self._lib)}")
            return ResourceRelease(out.state, out.freed_bytes if out.state == QUANTFUNC_RESOURCE_READY else None)

    def close(self):
        with self._lock:
            self._finalizer()

    def __enter__(self):
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, *_):
        self.close()


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


class DenoiseStepRefsParams(ctypes.Structure):
    # ONE denoise step WITH sequence-spliced reference latents (QwenImage-2.1 edit through the native
    # loader). base = the plain step (validated identically to quantfunc_denoise_step); the references are
    # the WORKFLOW's own VAE latents (TextEncodeQwenImage21 -> the model's process_latent_in), each spliced
    # in front of context row image_slots[r]. Field order/types mirror include/quantfunc.h
    # quantfunc_denoise_step_refs_params_t VERBATIM. Present only in refs-capable engine builds (hasattr).
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("base", DenoiseStepParams),
        ("num_refs", ctypes.c_int),
        ("ref_latents", ctypes.POINTER(ctypes.c_void_p)),
        ("ref_dims", ctypes.POINTER(ctypes.c_int32)),     # num_refs x 4: [1, C, h, w]
        ("image_slots", ctypes.POINTER(ctypes.c_int32)),  # num_refs entries
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
    # Reference-latent step (QwenImage-2.1 edit through the native loader). hasattr-gated like step_multi:
    # an older .so lacks the symbol and the qwenimage21 seam then refuses reference images with guidance.
    if hasattr(lib, "quantfunc_denoise_step_refs"):
        lib.quantfunc_denoise_step_refs.restype = ctypes.c_int
        lib.quantfunc_denoise_step_refs.argtypes = [v, ctypes.POINTER(DenoiseStepRefsParams)]
    if hasattr(lib, "quantfunc_denoise_cond_tail_supported"):
        # CR A-1 capability query (per-pipeline): 1 = this pipeline consumes a
        # begin_edit_cond cond-latent. Probe THIS (presence + answer), not the
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
    if hasattr(lib, "quantfunc_unload_sync_ex"):
        lib.quantfunc_unload_sync_ex.restype = ctypes.c_int
        lib.quantfunc_unload_sync_ex.argtypes = [v, ctypes.POINTER(ctypes.c_uint64)]
    # partial VRAM shed (inter-stage eviction fix; absent on older .so -> hasattr-guarded)
    if hasattr(lib, "quantfunc_partial_unload"):
        lib.quantfunc_partial_unload.restype = ctypes.c_int
        lib.quantfunc_partial_unload.argtypes = [v, ctypes.c_uint64,
                                                 ctypes.POINTER(ctypes.c_int64)]
    # Optional symbol binding permits loading older libraries for diagnostics.
    # Residency/reclaim consumers reject missing capabilities explicitly; the
    # separate legacy future-demand reader is not yet a precise request planner.
    if hasattr(lib, "quantfunc_resident_vram_bytes"):
        lib.quantfunc_resident_vram_bytes.restype = ctypes.c_int
        lib.quantfunc_resident_vram_bytes.argtypes = [v, ctypes.POINTER(ctypes.c_uint64)]
    if hasattr(lib, "quantfunc_vram_need_bytes"):
        lib.quantfunc_vram_need_bytes.restype = ctypes.c_int
        lib.quantfunc_vram_need_bytes.argtypes = [v, ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
                                                  ctypes.POINTER(ctypes.c_uint64)]
    return lib


_LIB = None
_LIB_PATH = None             # the engine library load_lib loaded - constant for the process (loaded_so_path)
_FINGERPRINT_PENDING = None  # (lib, path, file identity at load) whose fingerprint waits for the first info-level loader
_LIB_LOCK = threading.Lock()  # one first load per process: two could resolve two different pairs (a marker switched between)
_LOG_LEVEL = None   # the level a loader asked for (qf_log_level); applied when/after the library loads


def set_log_level(level):
    """Engine console detail, process-wide. Applied now if the library is already loaded; otherwise
    load_lib() applies it right after loading, so asking for a level never loads the library by itself
    (a loader run without an engine library behaves exactly as before)."""
    global _LOG_LEVEL
    _LOG_LEVEL = int(level)
    if _LIB is not None:
        _LIB.quantfunc_set_log_level(_LOG_LEVEL)
    _emit_fingerprint()


def loaded_so_path():
    """The engine library this process loaded (load_lib), or None before the load. Constant for the process: the
    installer may mark a newer pair while ComfyUI runs, and that one loads at the next start. A cache keyed on this never
    splits one model across two keys, and nothing re-resolves (or re-hashes the pair) per lookup."""
    return _LIB_PATH


def _emit_fingerprint():
    """Print the loaded library's fingerprint line ONCE, as soon as the level allows it. It is an info line, but the
    library can load before any loader set a level,
    so it waits for the first info-level loader instead of being lost. At warning it is never printed, and its md5 is
    never computed."""
    global _FINGERPRINT_PENDING
    if _FINGERPRINT_PENDING is not None and _LOG_LEVEL is not None and _LOG_LEVEL <= _LOG_INFO:
        lib, path, ident = _FINGERPRINT_PENDING
        _FINGERPRINT_PENDING = None
        _log_lib_fingerprint(lib, path, ident)


_LOG_INFO = 2   # info on the engine's scale (qf_log_level.LOG_LEVELS): the plugin's own detail lines follow the same level


def info(msg, *args, flush=True):
    """The plugin's own detail line — loaded / SESSION OPEN / CLOSED / the VRAM ledger / the library fingerprint and the
    like — printed only while a loader asked for info: the hidden log_level input (users get warning, and before any
    loader ran it is warning too; the harness sends info). A warning or an error never goes through here: it always
    prints. `args` %-format `msg` (the logging idiom the converted lines used)."""
    if _LOG_LEVEL is not None and _LOG_LEVEL <= _LOG_INFO:
        say(msg % args if args else msg, flush=flush)


def console_safe(text):
    """`text` with each character the console cannot encode written as a backslash escape (#738). ComfyUI keeps the OS
    encoding on stdout/stderr with errors='strict' (a redirected Windows console: cp932 / cp949 / cp1252 / cp936 ...),
    so one character the code page cannot hold raises UnicodeEncodeError out of whatever printed it, the loader
    included. Only such characters change, so a code page that holds CJK keeps it readable. The streams themselves are
    never reconfigured: they are ComfyUI's."""
    for stream in (sys.stdout, sys.stderr):
        enc = getattr(stream, "encoding", None)
        if enc:
            try:
                text = text.encode(enc, "backslashreplace").decode(enc)
            except LookupError:   # a codec name Python does not know: plain ASCII is safe on every console
                text = text.encode("ascii", "backslashreplace").decode("ascii")
    return text


def say(msg, flush=True):
    """print() for the plugin's own lines: never raises on the console's code page (console_safe)."""
    print(console_safe(msg), flush=flush)


class _ConsoleSafe(logging.Filter):
    """Makes each record of a plugin logger printable on the console's code page (console_safe). A traceback
    (exc_info, stack_info) is folded into the message first: the handler would format it unescaped. A malformed
    %-format is left to the handler, which reports it as logging always has."""

    def filter(self, record):
        try:
            text = record.getMessage()
        except (TypeError, ValueError):
            return True
        if record.exc_info:
            text += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip("\n")
            record.exc_info = record.exc_text = None
        if record.stack_info:
            text += "\n" + record.stack_info
            record.stack_info = None
        record.msg, record.args = console_safe(text), None
        return True


def logger(name):
    """logging.getLogger(name) with _ConsoleSafe on it: every plugin logger comes from here. A filter works on the
    logger it is attached to, so ComfyUI's own loggers, handlers and streams are untouched."""
    log = logging.getLogger(name)
    if not any(isinstance(f, _ConsoleSafe) for f in log.filters):
        log.addFilter(_ConsoleSafe())
    return log


def _console_safe_one(exc):
    """console_safe on one exception's printed text (its args, OSError/SyntaxError fields, notes), in place; the original
    message stays readable as exc.qf_console_original. Returns whether it now prints safely. One whose text is still
    unsafe (a __str__ computed from other state) is put back as it was: its stand-in carries it (_stand_in)."""
    fields = (("strerror", "filename", "filename2") if isinstance(exc, OSError) else
              ("msg", "filename", "text") if isinstance(exc, SyntaxError) else ())
    args, values, notes = exc.args, [getattr(exc, attr, None) for attr in fields], getattr(exc, "__notes__", None)
    if isinstance(notes, list):
        exc.__notes__ = [console_safe(n) if isinstance(n, str) else n for n in notes]
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 - a broken __str__: a traceback prints a fixed ASCII placeholder for it
        return True
    if console_safe(text) == text:
        return True
    exc.args = tuple(console_safe(a) if isinstance(a, str) else a for a in args)
    for attr, value in zip(fields, values):
        if isinstance(value, str):
            setattr(exc, attr, console_safe(value))
    try:
        now = str(exc)
    except Exception:  # noqa: BLE001
        now = ""
    if console_safe(now) == now:
        try:
            exc.qf_console_original = text
        except (AttributeError, TypeError):
            pass
        return True
    exc.args = args
    for attr, value in zip(fields, values):
        if isinstance(value, str):
            setattr(exc, attr, value)
    if isinstance(notes, list):
        exc.__notes__ = notes
    return False


def _group_members(e):
    """The members a traceback prints for an exception group (Python 3.11+); none for any other exception."""
    return e.exceptions if isinstance(e, getattr(builtins, "BaseExceptionGroup", ())) else ()


def _stand_in(e):
    """A RuntimeError printed in place of `e`, whose text cannot be rewritten (a __str__ computed from other state):
    e's text escaped, e's traceback and chain, and e itself as `qf_console_original`."""
    safe = RuntimeError(console_safe("".join(traceback.format_exception_only(type(e), e)).strip()))
    safe.qf_console_original = e
    # a context-only chain becomes the cause: raised inside another handler, the stand-in gets that handler's exception
    # as its __context__, and a set cause always prints (plugin CR R1/C-L1)
    safe.__cause__ = e.__cause__ if e.__cause__ is not None or e.__suppress_context__ else e.__context__
    safe.__context__ = e.__context__
    safe.__suppress_context__ = e.__suppress_context__   # assigning __cause__ set it
    return safe.with_traceback(e.__traceback__)


def console_safe_exception(exc):
    """Make everything a traceback prints for `exc` hold on the console's code page: its message, its notes and every
    exception chained to it (cause, context, group members). ComfyUI logs an uncaught node exception and its traceback
    to its strict console, and one character the code page lacks makes that logging raise a second exception inside
    ComfyUI's error handling (#738).
    - Text that comes from an exception's args is rewritten in place: the object and its type stay, the original text
      is kept as `qf_console_original`.
    - A CHAINED exception whose text cannot be rewritten (a __str__ computed from other state) is replaced, in the
      links that hold it, by a stand-in RuntimeError carrying its text escaped and its own chain (_stand_in). A group
      with such a member is replaced as a whole (its members cannot be replaced one by one), so its members are
      then on the stand-in's qf_console_original, not in the log.
    Returns the exception to raise: `exc` itself whenever its own text is safe, so ComfyUI still sees its type
    (execution.py tells an OOM and a user cancel by type), else its stand-in, which keeps its chain."""
    order, pending, seen = [], [exc], set()
    while pending:
        e = pending.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        order.append((e, _console_safe_one(e)))
        pending += [e.__cause__, e.__context__, *_group_members(e)]
    unsafe = {id(e) for e, ok in order if not ok}
    grown = True
    while grown:   # a group that prints an unsafe member is unsafe itself
        grown = False
        for e, _ in order:
            if id(e) not in unsafe and any(id(m) in unsafe for m in _group_members(e)):
                unsafe.add(id(e))
                grown = True
    stand = {id(e): _stand_in(e) for e, _ in order if id(e) in unsafe}
    for e in [e for e, _ in order if id(e) not in unsafe] + list(stand.values()):
        for attr in ("__cause__", "__context__"):
            link = getattr(e, attr)
            if link is not None and id(link) in stand:
                setattr(e, attr, stand[id(link)])
    return stand.get(id(exc), exc)


def console_safe_errors(fn):
    """Wrap `fn` (a node FUNCTION, a method ComfyUI calls) so that an exception leaving it is console-safe
    (console_safe_exception). It is re-raised as the same object whenever its own text can be made safe, so its type
    reaches ComfyUI; only one whose own text cannot be rewritten is replaced by its stand-in (chain kept)."""
    if getattr(fn, "__qf_console_safe__", False):
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            safe = console_safe_exception(exc)
            if safe is exc:
                raise
        raise safe   # outside the handler, so Python does not chain the stand-in to the exception it replaces
    wrapper.__qf_console_safe__ = True
    try:   # what a caller introspects stays fn's, for inspect.getfullargspec too (it ignores __wrapped__): ComfyUI
        wrapper.__signature__ = inspect.signature(fn)   # passes VALIDATE_INPUTS only the inputs it names (execution.py)
    except (TypeError, ValueError):   # a callable without one: nothing to keep
        pass
    return wrapper


def _console_safe_attr(cls, name):
    """Wrap one attribute of `cls` with console_safe_errors; a classmethod / staticmethod / property stays one."""
    raw = inspect.getattr_static(cls, name)
    if isinstance(raw, (classmethod, staticmethod)):
        setattr(cls, name, type(raw)(console_safe_errors(raw.__func__)))
    elif isinstance(raw, property):
        fns = (f and console_safe_errors(f) for f in (raw.fget, raw.fset, raw.fdel))
        setattr(cls, name, property(*fns, raw.__doc__))
    elif isinstance(raw, types.FunctionType):
        setattr(cls, name, console_safe_errors(raw))


def console_safe_methods(cls):
    """Class decorator: every method and property the class defines, and its __init__ (ComfyUI constructs model
    patchers itself: ModelPatcher.clone), runs under console_safe_errors; other dunders are Python's. It goes on every
    class ComfyUI calls into (a comfy base, a plugin subclass of one, the mixins combined with one), so the sampler and
    model-management paths, present and future, pass the same boundary as the nodes (tests/text_encoding_test.py)."""
    for name in list(vars(cls)):
        if name == "__init__" or not (name.startswith("__") and name.endswith("__")):
            _console_safe_attr(cls, name)
    return cls


# What ComfyUI calls on a node class besides its FUNCTION, each when present (execution.py, server.py).
_NODE_ENTRY_POINTS = ("INPUT_TYPES", "VALIDATE_INPUTS", "IS_CHANGED", "check_lazy_status")


def console_safe_nodes(mapping):
    """Wrap every registered node's FUNCTION and its other ComfyUI entry points with console_safe_errors. __init__
    calls it once, after the last NODE_CLASS_MAPPINGS registration, so no node skips it (tests/text_encoding_test.py,
    the boundary arm)."""
    for cls in mapping.values():
        for name in (getattr(cls, "FUNCTION", None), *_NODE_ENTRY_POINTS):
            if isinstance(name, str) and hasattr(cls, name):
                _console_safe_attr(cls, name)
    return mapping


# ── Engine library install (option C, user 2026-09-24 「在原生加载器里实现」「根据自己的显卡型号下载对应so」) ──────────
# The plugin installs the engine it needs into bin/<platform>/, SHA-256-verified against the release's published
# verify.json. ONE installer for both published platforms; what differs is one row of _ENGINE_PLATFORMS:
#   Linux: a pair — the host library for torch's CUDA major (the FORK-2 guard refuses any other) and the kernel library of
#     this GPU's class, one build (the same .qf_pair_id in both). Published as {ver}/linux/<host> (one per CUDA major) and
#     {ver}/linux/<set>/<kernel> (one per class and major); the host finds its kernel in its folder through $ORIGIN.
#   Windows: ONE self-contained DLL per class and CUDA major, {ver}/windows/<set>/<dll> (no companion, no pair id).
# Both are keyed by those paths in verify.json (section "linux" / "win32"), with sets.json under {ver}/<folder>/.
# Installed layout, one folder per (release, GPU class, CUDA major):
#   bin/<platform>/<version>-<set>-cu<major>/{host[, kernel]}
#   bin/<platform>/.engine-<set>-cu<major>.json          its marker: names the loadable files, records their SHA-256s
# The marker is written LAST (an atomic rename), so a folder becomes loadable only once its files are verified and in
# place, and a marked folder is never modified (a same-version re-install drops the marker first). A new release always
# goes into a NEW folder: a loaded Windows DLL cannot be replaced, and a folder whose DLL another process still has loaded
# cannot be deleted either (it stays, ignored, until the next install removes it). Two ComfyUI instances of different GPU
# classes (or CUDA majors) sharing this folder keep separate folders and markers. resolve_so_path loads only a marked
# folder whose files still hash to its marker (R5: a failed or unverified library never runs). The installer keeps out
# entirely when QF_NATIVE_SO_PATH names the library or bin/<platform>/.dev_lib_lock marks a local build there.
# EVERY name this installer uses lives in this block; nothing it writes or fetches is named by the server.
_ENGINE_BASE_URL = "https://www.modelscope.cn/models/QuantFunc/Plugin/resolve/master"   # HTTPS only (checked per fetch)
# The published platforms, by bin/<platform>/ name — also the release's folder under {ver}/. key: the platform's section
# of version.json and verify.json; arches: platform.machine() of the published engines; hosts: the engine library per CUDA
# major; kernel: True = a host plus one kernel library per GPU class (Linux), False = one self-contained library per class.
# version.json gates a release for this installer with "kernel_so": true on BOTH platforms ("published in the per-arch
# installer layout"; tests-07 ruling 2026-09-24: no second field).
_ENGINE_PLATFORMS = {
    "linux": {"key": "linux", "arches": ("x86_64",), "kernel": True,
              "hosts": {13: "libquantfunc.so", 12: "libquantfunc-12.so"}},
    "windows": {"key": "win32", "arches": ("AMD64",), "kernel": False,
                "hosts": {13: "quantfunc.dll", 12: "quantfunc-12.dll"}},
}
_ENGINE_KERNEL_RE = re.compile(r"libquantfunc_kernels[-A-Za-z0-9_.]*\.so")   # the host's DT_NEEDED names its kernel
_ENGINE_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")    # a release version: a URL path segment and part of a folder name
_ENGINE_SET_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")  # a GPU class from sets.json: a URL path segment, part of a name
_ENGINE_SETS_SCHEMA = 2  # sets.json: one kernel library per GPU architecture ({"sm89": [89], ...}); 1 was consumer/server
_ENGINE_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ENGINE_PAIR_ID_RE = re.compile(r"[0-9a-f]{32}")      # .qf_pair_id: one id per release and CUDA major, in every file
_ENGINE_MARKER_RE = re.compile(r"\.engine-([a-z][a-z0-9_]{0,31})-cu(\d+)\.json")
_ENGINE_PAIR_RE = re.compile(r"(\d+\.\d+\.\d+)-([a-z][a-z0-9_]{0,31})-cu(\d+)")
_ENGINE_LOCAL_BUILD_LOCK = ".dev_lib_lock"            # bin/<platform>/.dev_lib_lock: bin/<platform>/<lib> is a local build
_ENGINE_INSTALL_LOCKFILE = ".engine-install.lock"     # _install_lock: one installer per plugin folder, across instances
_ENGINE_REINSTALLED = ".reinstalled"                  # in a pair folder: re-downloaded once already after a failed load
_ENGINE_VERIFY_SCHEMA_MAX = 1
_ENGINE_HTTP_TIMEOUT_S = 120
_ENGINE_PLUGIN_VERSION_FILE = "version.json"          # bin/<platform>/version.json: {"comfy": "<plugin version>"}

_ENGINE_STATUS = {"state": "idle", "detail": ""}      # what the first loader run reports if no library is there yet
_ENGINE_STATUS_LOCK = threading.Lock()
_ENGINE_DEVICE = 0   # the CUDA device ComfyUI computes on (start_engine_install sets it): its SM picks the GPU class


class EngineNotInstallable(RuntimeError):
    """This machine cannot take a published engine (CPU / GPU class / CUDA / driver); the message says why."""


def _engine_status(state, detail=""):
    with _ENGINE_STATUS_LOCK:
        _ENGINE_STATUS.update(state=state, detail=detail)


def engine_install_status():
    """(state, detail): idle | checking | downloading | installed | offline | unavailable | failed | local."""
    with _ENGINE_STATUS_LOCK:
        return _ENGINE_STATUS["state"], _ENGINE_STATUS["detail"]


def _engine_bin_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", _BIN_SUBDIR)


def _engine_platform():
    """This platform's row of _ENGINE_PLATFORMS, or None where no engine is published (read per call, never cached)."""
    return _ENGINE_PLATFORMS.get(_BIN_SUBDIR)


def _torch_cuda_major():
    """torch's CUDA major (13 for "13.0"), or None for a CPU-only / unreadable torch. The ONE parse of it."""
    try:
        import torch
        return int(str(torch.version.cuda).split(".")[0]) if torch.version.cuda else None
    except Exception:  # noqa: BLE001
        return None


def _driver_cuda_major():
    """The newest CUDA major the installed driver runs (cuDriverGetVersion 13010 -> 13), or None."""
    try:
        cuda = ctypes.CDLL("nvcuda.dll" if _BIN_SUBDIR == "windows" else "libcuda.so.1")
        v = ctypes.c_int(0)
        if cuda.cuDriverGetVersion(ctypes.byref(v)) != 0:
            return None
        return v.value // 1000
    except Exception:  # noqa: BLE001
        return None


def _gpu_sm(device_idx=0):
    try:
        import torch
        major, minor = torch.cuda.get_device_capability(int(device_idx))
        return major * 10 + minor
    except Exception:  # noqa: BLE001
        return None


def _engine_http_open(url):
    """The ONE network entry. HTTPS only, before and after redirects; returns the open response."""
    import urllib.request
    if not url.startswith("https://"):
        raise RuntimeError(f"refusing a non-HTTPS engine URL: {url}")
    resp = urllib.request.urlopen(url, timeout=_ENGINE_HTTP_TIMEOUT_S)
    if not resp.geturl().startswith("https://"):
        resp.close()
        raise RuntimeError(f"refusing a redirect to a non-HTTPS engine URL: {resp.geturl()}")
    return resp


def _engine_http_get(url):
    """A small document (version.json / verify.json / sets.json), whole."""
    with _engine_http_open(url) as resp:
        return resp.read()


def _engine_fetch_to(url, dest, label):
    """Stream `url` into `dest` (fsync'd), reporting "x of y MB" progress; returns the SHA-256 of what was written."""
    import hashlib
    h, done = hashlib.sha256(), 0
    with _engine_http_open(url) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            _engine_status("downloading", f"{label}: {done >> 20} of {total >> 20 if total else '?'} MB")
        f.flush()
        os.fsync(f.fileno())
    return h.hexdigest()


def _sha256_of(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _version_key(v):
    return tuple(int(x) for x in v.split("."))


def engine_choice(device_idx=0):
    """(cuda_major, sm): the host follows torch's CUDA major; the driver is only a CHECK. Raises
    EngineNotInstallable with a one-line reason when this machine cannot take a published engine."""
    torch_major, driver_major = _torch_cuda_major(), _driver_cuda_major()
    if torch_major is None:   # the resolver loads the pair for torch's CUDA major: nothing else would ever load
        raise EngineNotInstallable("PyTorch reports no CUDA version (a CPU-only PyTorch?); the QuantFunc engine needs a "
                                   "CUDA build of PyTorch; no engine was installed")
    major, hosts = torch_major, _engine_platform()["hosts"]
    if major not in hosts:
        raise EngineNotInstallable(
            f"no QuantFunc engine is published for CUDA {major if major is not None else 'unknown'} "
            f"(torch CUDA {torch_major}, driver CUDA {driver_major}); published: CUDA {sorted(hosts)}")
    if driver_major is not None and driver_major < major:
        raise EngineNotInstallable(
            f"your NVIDIA driver runs up to CUDA {driver_major}, but torch uses CUDA {major}: update the driver; "
            f"no engine was installed")
    sm = _gpu_sm(device_idx)
    if sm is None:
        raise EngineNotInstallable("no CUDA GPU is visible to torch; no engine was installed")
    return major, sm


def _engine_pick_version(versions, plugin_version, major):
    """The release this plugin installs: the classic updater's rule (f53e39d _find_best_compatible_version). Among the
    entries published in the per-architecture layout ("kernel_so": true) and whose "comfy" / "comfy-12" requirement (falling back
    to "comfy") is at most this plugin's version, the one with the highest "lib" / "lib-12" (falling back to "lib", then
    to the key). Returns that entry's KEY, which is the release's path segment on the repo."""
    suffix = "-12" if major == 12 else ""
    best = best_lib = None
    for key, info in (versions or {}).items():
        if not (isinstance(key, str) and _ENGINE_VERSION_RE.fullmatch(key) and isinstance(info, dict)
                and info.get("kernel_so")):
            continue
        req = info.get("comfy" + suffix, info.get("comfy"))
        lib = info.get("lib" + suffix, info.get("lib", key))
        if not all(isinstance(x, str) and _ENGINE_VERSION_RE.fullmatch(x) for x in (req, lib)):
            continue
        if _version_key(req) <= _version_key(plugin_version) and (best is None or _version_key(lib) > _version_key(best_lib)):
            best, best_lib = key, lib
    return best


def _engine_local_choice():
    """Why the installer keeps out, or None: the dev override names the library, or bin/<platform>/ holds a local build."""
    if os.environ.get(_ENV_SO_OVERRIDE, "").strip():
        return f"{_ENV_SO_OVERRIDE} names the engine library"
    if os.path.exists(os.path.join(_engine_bin_dir(), _ENGINE_LOCAL_BUILD_LOCK)):
        return f"bin/{_BIN_SUBDIR}/{_ENGINE_LOCAL_BUILD_LOCK} keeps the local build bin/{_BIN_SUBDIR}/{_LIB_BASENAME}"
    return None


def _pair_dir(m):
    return f"{m['version']}-{m['set']}-cu{m['cuda']}"


def _read_marker(path):
    """A pair's marker, or None. Every field is checked before it can name anything: the marker is a local file, but
    its fields become the folder and file names the loader opens."""
    name = _ENGINE_MARKER_RE.fullmatch(os.path.basename(path))
    plat = _engine_platform()
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        files = [m["host"], m["kernel"]] if plat["kernel"] else [m["host"]]
        ok = (name is not None and m["set"] == name.group(1) and type(m["cuda"]) is int and m["cuda"] == int(name.group(2))
              and _ENGINE_VERSION_RE.fullmatch(m["version"]) is not None and m["host"] == plat["hosts"].get(m["cuda"])
              and (_ENGINE_KERNEL_RE.fullmatch(m["kernel"]) is not None if plat["kernel"] else m["kernel"] is None)
              and all(type(s) is int for s in m["sms"])
              and sorted(m["sha256"]) == sorted(files)
              and all(_ENGINE_SHA256_RE.fullmatch(h) for h in m["sha256"].values()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return m if ok else None


def _markers():
    """[(marker path, marker)] of every valid marker in the plugin's bin/<platform>/, in name order."""
    d = _engine_bin_dir()
    try:
        names = sorted(n for n in os.listdir(d) if _ENGINE_MARKER_RE.fullmatch(n))
    except OSError:
        return []
    return [(os.path.join(d, n), m) for n in names for m in (_read_marker(os.path.join(d, n)),) if m]


def _installed_pair():
    """(marker path, marker) of the pair THIS process loads — torch's CUDA major, the SM of ComfyUI's device — or
    (None, None). The installer keeps exactly one marker per CUDA major claiming an SM (_claim): the pair it chose.
    Two claimants exist only in a folder the installer has not run in since (an offline start after an older plugin);
    then the newest release wins, deterministically."""
    major, sm = _torch_cuda_major(), _gpu_sm(_ENGINE_DEVICE)
    mine = [(p, m) for p, m in _markers() if m["cuda"] == major and sm in m["sms"]]
    return max(mine, key=lambda pm: _version_key(pm[1]["version"])) if mine else (None, None)


def _claim(bin_dir, marker, sms, chosen=None):
    """The installer chose `marker` for `sms` (its release's list for that GPU class): make it the ONLY marker of its
    CUDA major that claims them, so the resolver loads the pair the installer chose. A release may move an SM to another
    class, and a pulled release or an older plugin can move it back. Other markers keep their other SMs (another GPU of
    their class may use them); a marker left claiming nothing goes, with its pair. Markers are rewritten atomically.
    ORDER: the other markers give the SMs up FIRST and the chosen marker is written LAST, so a crash in between leaves no
    marker claiming them — the resolver says no engine is installed and the next start installs again — never two
    claimants it would have to choose between. `chosen`: the new marker (an install); None re-claims the kept one.
    A pair removed here may be the one another ComfyUI sharing this folder has just resolved: its load then refuses
    loudly with nothing loaded, and its next prompt resolves again (engine_install_test arm 31)."""
    major = int(_ENGINE_MARKER_RE.fullmatch(os.path.basename(marker)).group(2))
    want = sorted(set(sms))
    for p, m in _markers():
        if m["cuda"] != major or p == marker:
            continue
        left = [s for s in m["sms"] if s not in sms]
        if left == m["sms"]:
            continue
        if left:
            _engine_write_file(p, json.dumps(dict(m, sms=left)).encode())
        else:
            os.remove(p)
            shutil.rmtree(os.path.join(bin_dir, _pair_dir(m)), ignore_errors=True)
    if chosen is None:
        chosen = _read_marker(marker)
        if chosen is None or sorted(chosen["sms"]) == want:
            return
        chosen = dict(chosen, sms=want)
    _engine_write_file(marker, json.dumps(chosen).encode())   # LAST: the chosen pair becomes the claimant


def _marker_of(so_path):
    """(marker path, marker) of the installed pair whose host is so_path, or (None, None). Real paths are compared: the
    plugin folder is often a symlink under custom_nodes."""
    here = os.path.realpath(os.path.dirname(so_path))
    return next(((p, m) for p, m in _markers()
                 if os.path.realpath(os.path.join(_engine_bin_dir(), _pair_dir(m))) == here), (None, None))


def _pair_intact(m):
    """Every file of the marked engine (Linux: host + kernel; Windows: the DLL) still hashes to what its marker recorded."""
    pair = os.path.join(_engine_bin_dir(), _pair_dir(m))
    try:
        return all(_sha256_of(os.path.join(pair, n)) == h for n, h in m["sha256"].items())
    except OSError:
        return False


def _manifest_key(gpu_set, name):
    """A file's key in verify.json, which is also its path under {ver}/<platform>/: a Linux host is one per CUDA major, a
    kernel (Linux) or a whole library (Windows) one per class."""
    plat = _engine_platform()
    return name if plat["kernel"] and name in plat["hosts"].values() else f"{gpu_set}/{name}"


def _engine_sets(version, hashes):
    """A release's GPU classes {set: [sm, ...]}, from its sets.json — the ONLY source: the engine build prints it from
    the arch lists it builds with, and this plugin keeps no SM list of its own. Hash-checked against verify.json."""
    if "sets.json" not in hashes:
        raise RuntimeError(f"the {version} release publishes no sets.json, so its GPU classes are unknown; no engine "
                           f"was installed")
    raw = _engine_http_get(f"{_ENGINE_BASE_URL}/{version}/{_BIN_SUBDIR}/sets.json")
    import hashlib
    if hashlib.sha256(raw).hexdigest() != hashes["sets.json"]:
        raise RuntimeError(f"the {version} sets.json does not match its published SHA-256")
    doc = json.loads(raw)
    schema = doc.get("schema") if isinstance(doc, dict) else None
    if not (type(schema) is int and schema == _ENGINE_SETS_SCHEMA):   # schema 1 = the old consumer/server classes
        raise RuntimeError(f"the {version} sets.json has schema {schema!r}, and this plugin reads schema "
                           f"{_ENGINE_SETS_SCHEMA} (one kernel library per GPU architecture); no engine was installed")
    sets = doc.get("sets")
    if not (isinstance(sets, dict) and sets and all(
            isinstance(k, str) and _ENGINE_SET_RE.fullmatch(k) and isinstance(v, list) and v
            and all(type(s) is int for s in v) for k, v in sets.items())):
        raise RuntimeError(f"the {version} sets.json is not a GPU-class map this plugin understands")
    return sets


def _engine_write_file(path, data):
    """temp file in the SAME dir -> fsync -> atomic rename (a crash never leaves a half-written file)."""
    tmp = f"{path}.part-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def install_engine(device_idx=None):
    """Fetch-verify-install the engine pair for THIS machine (idempotent; start_engine_install runs it on a thread).

    Remote-first: the release's version.json, verify.json and sets.json are read every time, so a newer compatible
    engine replaces an older one. The marked pair of the wanted release is kept while the release still publishes the
    hashes its marker recorded; when it does not, that is a KNOWN mismatch — its marker goes first, so it is never loaded
    again. Otherwise, into the release's own folder:
      - download the host (Windows: the one DLL of this class), verify its SHA-256;
      - Linux: read the kernel's name from the host's DT_NEEDED, download the kernel, verify it, and check both carry
        the same .qf_pair_id (one build), else install nothing; the KERNEL is renamed into place first, then the host;
      - write the marker LAST: only now is the engine loadable. Older folders of the same class and CUDA major go,
        except the one just replaced (a process may be between reading its marker and loading it).
    A failure installs nothing and keeps a verified older engine (never bricks). Returns the marker, or None when the
    installer keeps out (a local build, or the dev override)."""
    why = _engine_local_choice()
    if why:
        _engine_status("local", why)
        say(f"[qf_native] QuantFunc engine install skipped: {why}", flush=True)
        return None
    plat = _engine_platform()
    if plat is None:
        # No engine is published for this platform: the library placed in bin/<platform>/ (or the package root) is what
        # loads here (resolve_so_path); with it in place there is nothing to install or to say at every start.
        if any(os.path.isfile(os.path.join(d, _LIB_BASENAME))
               for d in (_engine_bin_dir(), os.path.dirname(os.path.abspath(__file__)))):
            _engine_status("local", f"bin/{_BIN_SUBDIR}/{_LIB_BASENAME}")
            return None
        raise EngineNotInstallable(f"QuantFunc engines are published for Linux and Windows only; put the engine library "
                                   f"in bin/{_BIN_SUBDIR}/")
    if platform.machine() not in plat["arches"]:
        raise EngineNotInstallable(f"QuantFunc engines are published for {'/'.join(plat['arches'])} only, and this "
                                   f"machine is {platform.machine() or 'unknown'}; no engine was installed")
    bin_dir = _engine_bin_dir()
    os.makedirs(bin_dir, exist_ok=True)
    with _install_lock(os.path.join(bin_dir, _ENGINE_INSTALL_LOCKFILE)):
        return _install_pair(bin_dir, _ENGINE_DEVICE if device_idx is None else int(device_idx))


@contextlib.contextmanager
def _install_lock(path):
    """One installer per plugin folder at a time, across ComfyUI instances: flock on Linux, a byte-range lock on Windows.
    Waits for the other installer like flock does; released when this block ends, also when the process dies."""
    with open(path, "a+b") as f:
        if _BIN_SUBDIR == "windows":
            import errno
            import msvcrt
            f.seek(0)
            while True:
                try:   # LK_NBLCK: one attempt whose errno says WHY (LK_LOCK reports any failure as EDEADLOCK)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as e:
                    if e.errno != errno.EACCES:    # EACCES = held by the other installer; anything else (a filesystem
                        raise                      # without byte-range locks, a bad handle) fails the install loudly
                    time.sleep(1)
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
            yield


def _install_pair(bin_dir, device_idx):
    _engine_status("checking")
    major, sm = engine_choice(device_idx)
    try:
        with open(os.path.join(bin_dir, _ENGINE_PLUGIN_VERSION_FILE), encoding="utf-8") as f:
            plugin_version = str(json.load(f)["comfy"])
    except (OSError, ValueError, KeyError) as e:
        raise RuntimeError(f"cannot read this plugin's version from bin/{_BIN_SUBDIR}/{_ENGINE_PLUGIN_VERSION_FILE}: {e}")
    if not _ENGINE_VERSION_RE.fullmatch(plugin_version):
        raise RuntimeError(f"malformed plugin version {plugin_version!r}")
    plat = _engine_platform()
    versions = json.loads(_engine_http_get(f"{_ENGINE_BASE_URL}/version.json")).get(plat["key"])
    version = _engine_pick_version(versions, plugin_version, major)
    if version is None:
        raise EngineNotInstallable(f"no published engine with the per-architecture layout is compatible with this plugin "
                                   f"({plugin_version}, CUDA {major})")
    manifest = json.loads(_engine_http_get(f"{_ENGINE_BASE_URL}/{version}/verify.json"))
    # verify.json = {"schema": 1, "<platform>": {"<set>/<file>": sha256}} (the engine's verify_manifest.py); its release
    # is its path, so there is no version key to check.
    if not (isinstance(manifest, dict) and isinstance(manifest.get("schema"), int)
            and 1 <= manifest["schema"] <= _ENGINE_VERIFY_SCHEMA_MAX and isinstance(manifest.get(plat["key"]), dict)):
        raise RuntimeError(f"the {version} verify.json is not a manifest this plugin understands")
    hashes = manifest[plat["key"]]
    sets = _engine_sets(version, hashes)
    gpu_set = next((k for k, sms in sets.items() if sm in sms), None)   # EXACT architecture: never a nearest match
    if gpu_set is None:
        published = ", ".join(f"{s // 10}.{s % 10}" for s in sorted({s for v in sets.values() for s in v}))
        raise EngineNotInstallable(f"no QuantFunc engine is published for this GPU's architecture (SM {sm // 10}.{sm % 10}); "
                                   f"the {version} release has kernels for SM {published}; no engine was installed")
    host = plat["hosts"][major]
    marker = os.path.join(bin_dir, f".engine-{gpu_set}-cu{major}.json")
    have = _read_marker(marker)
    known_bad = None
    if have and have["version"] == version:
        if all(hashes.get(_manifest_key(gpu_set, n)) == h for n, h in have["sha256"].items()):
            _claim(bin_dir, marker, sets[gpu_set])
            _engine_status("installed", f"engine {version} ({gpu_set}, CUDA {major})")
            return have
        os.remove(marker)    # KNOWN mismatch: the release no longer publishes these bytes - never loaded again
        known_bad, have = version, None
    pair = os.path.join(bin_dir, f"{version}-{gpu_set}-cu{major}")
    os.makedirs(pair, exist_ok=True)
    got, parts, kernel = {}, [], None
    try:
        for name in ((host, None) if plat["kernel"] else (host,)):   # the host first: its DT_NEEDED names the kernel
            if name is None:
                needed = [n for n in _elf_needed(parts[0]) if _ENGINE_KERNEL_RE.fullmatch(n)]
                if len(needed) != 1:
                    raise RuntimeError(f"the {version} host library names {len(needed)} QuantFunc kernel libraries "
                                       f"(expected exactly one): {needed}")
                name = kernel = needed[0]
            key = _manifest_key(gpu_set, name)
            if not _ENGINE_SHA256_RE.fullmatch(str(hashes.get(key, ""))):
                raise RuntimeError(f"the {version} manifest has no SHA-256 for {key}")
            parts.append(os.path.join(pair, f".{name}.part"))
            got[name] = _engine_fetch_to(f"{_ENGINE_BASE_URL}/{version}/{_BIN_SUBDIR}/{key}", parts[-1],
                                         f"engine {version}: {name}")
            if got[name] != hashes[key]:
                raise RuntimeError(f"{key} does not match its published SHA-256 (download corrupt or tampered)")
        if kernel:
            ids = [_elf_pair_id(p) for p in parts]      # [host, kernel]
            if ids[0] is None or ids[0] != ids[1]:
                raise RuntimeError(f"the {version} host and kernel libraries are not one build (pair ids {ids[0]} / "
                                   f"{ids[1]}); nothing was installed")
        for n in ([kernel] if kernel else []) + [host]:   # the KERNEL first: a host is never in place without its kernel
            os.replace(os.path.join(pair, f".{n}.part"), os.path.join(pair, n))
    except Exception as e:
        if known_bad:
            raise RuntimeError(f"the installed engine {known_bad} no longer matches its published SHA-256 and the "
                               f"re-download failed ({type(e).__name__}: {e}); it is not loaded") from e
        raise
    finally:
        for p in parts:
            try:
                os.remove(p)
            except OSError:
                pass
    m = {"version": version, "set": gpu_set, "cuda": major, "sms": sets[gpu_set], "host": host, "kernel": kernel,
         "sha256": got}
    _claim(bin_dir, marker, sets[gpu_set], chosen=m)       # the other markers give up its SMs, then this marker, LAST
    keep = {_pair_dir(m), have and _pair_dir(have)}
    for d in os.listdir(bin_dir):
        p = _ENGINE_PAIR_RE.fullmatch(d)
        if p and p.group(2) == gpu_set and int(p.group(3)) == major and d not in keep:
            shutil.rmtree(os.path.join(bin_dir, d), ignore_errors=True)
    _engine_status("installed", f"engine {version} ({gpu_set}, CUDA {major})")
    say(f"[qf_native] installed QuantFunc engine {version} for {gpu_set} GPUs, CUDA {major}", flush=True)
    return m


def _engine_load_failed(so_path, err):
    """A pair whose files exist but will not LOAD is re-downloaded at most ONCE per release (the bound from f53e39d's
    classic updater: without it a driver/GPU mismatch re-downloads on every start). <pair>/.reinstalled records it; the
    marker is dropped and a background install starts now. A second failure of the same pair only reports the load
    error. Returns the message for the loader run."""
    detail = str(err)[:400]
    mpath, m = _marker_of(so_path)
    if not m:
        return f"the engine library failed to load: {detail}"
    flag = os.path.join(os.path.dirname(mpath), _pair_dir(m), _ENGINE_REINSTALLED)
    if os.path.exists(flag):
        return (f"the engine library {m['version']} failed to load again after one re-download: {detail}. It is not "
                f"re-downloaded again for this release: check the NVIDIA driver and the GPU.")
    _engine_write_file(flag, m["version"].encode())
    try:
        os.remove(mpath)
    except OSError:
        pass
    start_engine_install()
    return (f"the engine library {m['version']} failed to load: {detail}. It is being re-downloaded once; queue the "
            f"prompt again when the console says the engine is installed.")


def _engine_load_ok(so_path):
    """A successful load clears the reinstall-once flag of the installed pair."""
    mpath, m = _marker_of(so_path)
    if m:
        try:
            os.remove(os.path.join(os.path.dirname(mpath), _pair_dir(m), _ENGINE_REINSTALLED))
        except OSError:
            pass


def start_engine_install(device_idx=None):
    """Plugin import calls this with ComfyUI's device: installs/updates the engine on a daemon thread, so node
    registration never waits. A later call (a failed load's one re-download) keeps the device given first. Never raises:
    the outcome lands in engine_install_status()."""
    global _ENGINE_DEVICE
    if device_idx is not None:
        _ENGINE_DEVICE = int(device_idx)

    def _run():
        try:
            install_engine()
        except EngineNotInstallable as e:
            _engine_status("unavailable", str(e))
            say(f"[qf_native] QuantFunc engine not installed: {e}", flush=True)
        except Exception as e:  # noqa: BLE001 - offline / manifest / hash: a verified installed pair stays in use
            kept = _installed_pair()[1]
            _engine_status("offline" if kept else "failed", f"{type(e).__name__}: {e}")
            say(f"[qf_native] QuantFunc engine update failed ({type(e).__name__}: {e}); "
                f"{'the installed engine ' + kept['version'] + ' stays in use' if kept else 'no engine is installed'}",
                flush=True)

    t = threading.Thread(target=_run, name="qf-engine-install", daemon=True)
    t.start()
    return t


def resolve_so_path():
    """Resolve the engine native library WITHOUT any workflow-serializable input.

    THREAT MODEL (this is the #vuln an earlier `so_path` node-widget created): ComfyUI's whole
    distribution model is "open this shared workflow.json", so a node STRING widget in that JSON is
    ATTACKER-CONTROLLED. `ctypes.CDLL` runs the target library's constructors in-process the instant
    it loads, so letting a workflow choose the path is a remote-code-execution primitive (the classic
    "download this companion file, then load my workflow"). This resolver therefore accepts NO path
    from any node input. Its sources cannot be set by a shared workflow, and each is EXCLUSIVE (a missing
    library is an error, never a fall-through to some other engine):
      1. QF_NATIVE_SO_PATH — a DEV override read from the PROCESS ENVIRONMENT only (a workflow.json
         cannot set an env var). Used as-is on the trusted dev machine; must exist + be a real file.
      2. bin/<platform>/.dev_lib_lock present — the local build bin/<platform>/<basename> (or the package
         root); the installer keeps out of that folder. Without the lock a library there is ignored: on an
         upgraded install it is the previous updater's copy.
      3. Linux and Windows: the installed engine for this process (torch's CUDA major, ComfyUI's device),
         returned only while its files hash to its marker; one that does not is refused, its marker dropped
         and a re-download started. Other platforms: a library placed in bin/<platform>/ (or the package root).
    Returns a realpath; raises loudly if nothing usable exists."""
    pkg = os.path.dirname(os.path.abspath(__file__))
    bin_dir = _engine_bin_dir()
    override = os.environ.get(_ENV_SO_OVERRIDE, "").strip()
    if override:
        if not os.path.isfile(override):
            raise RuntimeError(f"qf_native: {_ENV_SO_OVERRIDE} names {override!r}, which is not a file")
        return os.path.realpath(override)
    local = os.path.exists(os.path.join(bin_dir, _ENGINE_LOCAL_BUILD_LOCK))
    if local or _engine_platform() is None:
        for c in (os.path.join(bin_dir, _LIB_BASENAME), os.path.join(pkg, _LIB_BASENAME)):
            if os.path.isfile(c):
                return os.path.realpath(c)
        raise RuntimeError(
            "qf_native: no engine library found: "
            + (f"bin/{_BIN_SUBDIR}/{_ENGINE_LOCAL_BUILD_LOCK} marks a local build, but bin/{_BIN_SUBDIR}/{_LIB_BASENAME} "
               f"is not there (build it there, or delete the lock to use the installed engine)" if local else
               f"put {_LIB_BASENAME} in the package bin/{_BIN_SUBDIR}/, or set {_ENV_SO_OVERRIDE}=<abs path> on the "
               f"(trusted) dev machine"))
    mpath, m = _installed_pair()
    if m:
        if _pair_intact(m):
            return os.path.realpath(os.path.join(bin_dir, _pair_dir(m), m["host"]))
        try:
            os.remove(mpath)
        except OSError:
            pass
        start_engine_install()
        raise RuntimeError(f"qf_native: the installed QuantFunc engine {m['version']} does not match the SHA-256 "
                           f"recorded when it was installed (a file changed on disk), so it is NOT loaded. It is being "
                           f"re-downloaded: queue the prompt again when the console says the engine is installed.")
    state, detail = engine_install_status()
    if state in ("checking", "downloading"):
        raise RuntimeError(f"qf_native: the QuantFunc engine library is still downloading ({detail}). Queue the prompt "
                           f"again when the console says the engine is installed.")
    raise RuntimeError(
        f"qf_native: no QuantFunc engine is installed for this GPU ({state}: {detail or 'no install attempted'}). The "
        f"plugin installs it automatically at ComfyUI start; the console says why it did not. A local build: put "
        f"{_LIB_BASENAME} in bin/{_BIN_SUBDIR}/ and create bin/{_BIN_SUBDIR}/{_ENGINE_LOCAL_BUILD_LOCK}.")


def _elf_sections(data):
    """[(name, type, offset, size, link)] of an ELF image's section headers (data: bytes or a memory map). Raises on a
    malformed image; the callers turn any failure into "undeterminable"."""
    if data[:4] != b"\x7fELF":
        return []
    is64 = data[4] == 2
    end = "<" if data[5] == 1 else ">"               # 1 = little-endian
    if is64:
        shoff = struct.unpack_from(end + "Q", data, 0x28)[0]
        shentsize, shnum, shstrndx = struct.unpack_from(end + "HHH", data, 0x3a)
    else:
        shoff = struct.unpack_from(end + "I", data, 0x20)[0]
        shentsize, shnum, shstrndx = struct.unpack_from(end + "HHH", data, 0x2e)
    raw = []                                         # (name offset, type, offset, size, link) per section header
    for i in range(shnum):
        b = shoff + i * shentsize
        if is64:
            raw.append(struct.unpack_from(end + "II", data, b) + struct.unpack_from(end + "QQ", data, b + 0x18)
                       + struct.unpack_from(end + "I", data, b + 0x28))
        else:
            raw.append(struct.unpack_from(end + "II", data, b) + struct.unpack_from(end + "III", data, b + 0x10))
    names = raw[shstrndx][2] if shstrndx < len(raw) else None    # the section-name string table's offset

    return [("" if names is None else _cstr(data, names + n), t, o, s, l) for (n, t, o, s, l) in raw]


def _cstr(data, off):
    """The NUL-terminated string at `off` (data: bytes or a memory map — a map has find, not index)."""
    z = data.find(b"\x00", off)
    if z < 0:
        raise ValueError("unterminated ELF string")
    return data[off:z].decode("latin-1")


def _elf_map(path, read):
    """read(mapped image) on a read-only memory map of `path` (only the pages it touches are read); None on any failure."""
    try:
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            return read(data)
    except Exception:  # noqa: BLE001 - any unreadable / malformed ELF -> "undeterminable", the caller fails closed
        return None


def _elf_needed(path):
    """The DT_NEEDED shared-library names of an ELF file, via a pure-stdlib parse of its .dynamic/.dynstr
    sections. Returns [] on ANY parse failure. Pure-stdlib on purpose: FORK-2's toolchain guard must not
    depend on readelf/ldd being installed on the consumer's box."""
    def read(data):
        secs = _elf_sections(data)
        dyn = next(((o, s, l) for (_, t, o, s, l) in secs if t == _SHT_DYNAMIC), None)
        if not dyn or dyn[2] >= len(secs):
            return []
        dyn_off, dyn_sz, dyn_link = dyn
        dynstr_off = secs[dyn_link][2]               # .dynamic's linked string table = .dynstr
        is64 = data[4] == 2
        end = "<" if data[5] == 1 else ">"
        needed = []
        for off in range(dyn_off, dyn_off + dyn_sz, 16 if is64 else 8):
            tag, val = struct.unpack_from(end + ("qQ" if is64 else "iI"), data, off)
            if tag == _DT_NULL:
                break
            if tag == _DT_NEEDED:
                needed.append(_cstr(data, dynstr_off + val))
        return needed
    return _elf_map(path, read) or []


def _elf_pair_id(path):
    """The build's pair id — its `.qf_pair_id` section: 32 lowercase hex characters and a NUL, carried by every file of
    one release's CUDA major (host, CLI, kernel) — or None when absent or malformed. Read from the file, never by
    loading it."""
    def read(data):
        sec = next(((o, s) for (n, _, o, s, _) in _elf_sections(data) if n == ".qf_pair_id"), None)
        pid = data[sec[0]:sec[0] + sec[1]].split(b"\x00")[0].decode("ascii") if sec else ""
        return pid if _ENGINE_PAIR_ID_RE.fullmatch(pid) else None
    return _elf_map(path, read)


def _is_elf(path):
    """True iff `path` begins with the ELF magic: a Linux .so, whose DT_NEEDED the toolchain guard reads. A Windows PE
    `.dll` ("MZ..") is read through _pe_imports instead; anything else (a macOS Mach-O `.dylib`) is refused."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except Exception:  # noqa: BLE001
        return False


def _is_pe(path):
    """True iff `path` begins with the PE/DOS magic ("MZ"): a Windows DLL, whose imports _pe_imports reads."""
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"MZ"
    except Exception:  # noqa: BLE001
        return False


def _pe_imports(path):
    """The DLL names a Windows PE image imports (its import table and its delay-load table), lowercased, via a
    pure-stdlib parse. Returns [] on ANY parse failure (the toolchain guard then fails closed)."""
    def read(data):
        if data[:2] != b"MZ":
            return []
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            return []
        nsec, opt_size = struct.unpack_from("<H", data, pe + 6)[0], struct.unpack_from("<H", data, pe + 20)[0]
        opt = pe + 24
        dirs = opt + (112 if struct.unpack_from("<H", data, opt)[0] == 0x20B else 96)   # PE32+ / PE32 data directories
        secs = [struct.unpack_from("<IIII", data, opt + opt_size + 40 * i + 8) for i in range(nsec)]

        def at(rva):   # file offset of an RVA: through the section that holds it
            return next(ptr + rva - va for vsize, va, raw, ptr in secs if va <= rva < va + max(vsize, raw))

        names = []
        for index, size, name_at in ((1, 20, 12), (13, 32, 4)):   # (directory, descriptor size, name RVA offset)
            rva = struct.unpack_from("<I", data, dirs + 8 * index)[0]
            if not rva:
                continue
            off = at(rva)
            while True:
                name_rva = struct.unpack_from("<I", data, off + name_at)[0]
                if not name_rva:
                    break
                names.append(_cstr(data, at(name_rva)).lower())
                off += size
        return names
    return _elf_map(path, read) or []


def _so_cuda_major(so_path):
    """The CUDA MAJOR the engine binary DYNAMICALLY links, or None if it cannot be determined. ELF (Linux): its
    libcudart.so.<major> (the regex is NOT end-anchored, so a versioned SONAME like `libcudart.so.11.0` still yields 11).
    PE (Windows): cudart64_<major>.dll, else cublas64_<major>.dll / cublasLt64_<major>.dll — a DLL built with nvcc's
    default static cudart imports no cudart, and cuBLAS's DLL name carries the CUDA major; two majors -> None."""
    for lib in _elf_needed(so_path):
        m = re.match(r"libcudart\.so\.(\d+)", lib)
        if m:
            return int(m.group(1))
    imports = _pe_imports(so_path)
    for pat in (r"cudart64_(\d+)\.dll", r"cublas(?:lt)?64_(\d+)\.dll"):
        majors = {int(m.group(1)) for m in (re.fullmatch(pat, n) for n in imports) if m}
        if majors:
            return majors.pop() if len(majors) == 1 else None
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

    PLATFORM SCOPE: the CUDA major is read from a Linux ELF .so (its libcudart NEEDED entry) and from a
    Windows PE .dll (its imported cudart64_N / cublas64_N DLL, _so_cuda_major). Any other binary — a macOS
    Mach-O `.dylib` — cannot be read, so the guard fail-closes with a DISTINCT, disclosed message (naming
    the platform + the limitation) that points at the override, rather than a silent generic refuse."""
    if os.environ.get(_ENV_ALLOW_UNVERIFIED_TOOLCHAIN, "").strip().lower() in ("1", "true", "yes"):
        return
    torch_major = _torch_cuda_major()           # e.g. 13; None on a CPU-only torch build
    # Neither ELF nor PE (a macOS Mach-O .dylib): nothing here reads its CUDA major. Fail closed (per the FORK-2
    # constraint) but with a DISTINCT, DISCLOSED message naming the platform + the limitation + the override.
    if not _is_elf(so_path) and not _is_pe(so_path):
        import platform as _pf
        raise RuntimeError(
            f"qf_native: REFUSING to load - the CUDA-toolchain compatibility check reads Linux (ELF) and "
            f"Windows (PE) engine libraries only, and the engine binary ({so_path}) is a "
            f"{_pf.system() or 'non-Linux'} binary whose CUDA version cannot be read here. torch is built for "
            f"CUDA {torch_major or 'none / CPU-only'}. Ensure your torch and the engine binary use the SAME CUDA "
            f"major, then set {_ENV_ALLOW_UNVERIFIED_TOOLCHAIN}=1 to proceed.")
    so_major = _so_cuda_major(so_path)
    if torch_major is None or so_major is None:
        raise RuntimeError(
            f"qf_native: REFUSING to load - cannot verify the engine's CUDA toolchain matches torch's "
            f"(torch CUDA={torch_major or 'none / CPU-only'}, engine library CUDA major="
            f"{so_major if so_major is not None else 'undeterminable'}). The engine loads in-process and "
            f"shares torch's CUDA context; an unverified toolchain combination can silently corrupt "
            f"output, so this is fail-closed. Install a torch + engine library built for the SAME CUDA major, "
            f"or set {_ENV_ALLOW_UNVERIFIED_TOOLCHAIN}=1 if you KNOW this combination is safe.")
    if torch_major != so_major:
        raise RuntimeError(
            f"qf_native: REFUSING to load - CUDA toolchain MISMATCH. torch is built for CUDA {torch_major} "
            f"but the engine library ({so_path}) links CUDA {so_major}. Running "
            f"a CUDA-{so_major} engine in-process with a CUDA-{torch_major} torch is an unverified "
            f"combination that can SILENTLY corrupt generated images/video (no crash). Use an engine library "
            f"built for CUDA {torch_major}, or a torch built for CUDA {so_major}. (Override only if you "
            f"know it is safe: {_ENV_ALLOW_UNVERIFIED_TOOLCHAIN}=1.)")


def _sidecar_preloads(so_path):
    """The libraries load_lib preloads before the engine, in name order: the files next to it that its DT_NEEDED
    closure names — never a QuantFunc engine image. Not another host (the platform's hosts: after a torch CUDA-major change
    both can sit in one folder, and neither is a dependency of the other), not a kernel (the host's DT_NEEDED and
    $ORIGIN load its own kernel as a member of its dlopen group, where the ~20 host symbols the BIND_NOW kernel imports
    resolve; preloading it first fails on those). Anything else in that folder — another engine build, a backup copy —
    is not a dependency and is never loaded."""
    d = os.path.dirname(so_path)
    try:
        present = set(os.listdir(d))
    except OSError:
        return []
    never = set((_engine_platform() or {}).get("hosts", {}).values()) | {os.path.basename(so_path)}
    want, todo = set(), [os.path.basename(so_path)]
    while todo:
        for n in _elf_needed(os.path.join(d, todo.pop())):
            if n in present and n not in want and n not in never and not _ENGINE_KERNEL_RE.fullmatch(n):
                want.add(n)
                todo.append(n)
    return sorted(want)


# The CUDA toolkit libraries whose copy must be torch's: one process holds one cuBLAS/cuSOLVER set. The driver
# (libcuda.so.1) is the system's and is never matched.
_CUDA_LIB_RE = re.compile(r"lib(?:cudart|cublas|cublasLt|cusolver|cusolverMg|cusparse|cufft|cufftw|curand|nvrtc"
                          r"|nvJitLink|cudnn\w*)\.so\.\d+")
_RTLD_DI_LINKMAP = 2   # dlinfo() request: the object's struct link_map


class _LinkMap(ctypes.Structure):
    _fields_ = [("l_addr", ctypes.c_void_p), ("l_name", ctypes.c_char_p)]   # the head of glibc's struct link_map


def _linker_path(soname):
    """The file the dynamic linker hands out for `soname` right now: the first loaded object carrying it, the same
    lookup the engine's NEEDED entry does. None when nothing loaded carries it, or off glibc Linux."""
    if _BIN_SUBDIR != "linux":
        return None
    try:
        lib = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)
        dlinfo = ctypes.CDLL(None).dlinfo   # in libc since glibc 2.34; in the interpreter's scope before that
    except (OSError, AttributeError):
        return None
    dlinfo.argtypes, dlinfo.restype = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p], ctypes.c_int
    lm = ctypes.POINTER(_LinkMap)()
    if dlinfo(lib._handle, _RTLD_DI_LINKMAP, ctypes.byref(lm)) != 0 or not lm or not lm.contents.l_name:
        return None
    return os.fsdecode(lm.contents.l_name)


def _engine_cuda_needs(so_path):
    """The CUDA toolkit sonames the engine's DT_NEEDED closure asks for: the host's, and those of the files next to it
    that the closure reaches (its kernel .so, its sidecars)."""
    d = os.path.dirname(so_path)
    seen, todo, cuda = {os.path.basename(so_path)}, [os.path.basename(so_path)], set()
    while todo:
        for n in _elf_needed(os.path.join(d, todo.pop())):
            if _CUDA_LIB_RE.fullmatch(n):
                cuda.add(n)
            elif n not in seen and os.path.isfile(os.path.join(d, n)):
                seen.add(n)
                todo.append(n)
    return cuda


def _torch_cuda_plan(needs, linker, major):
    """(dirs, provided, preloads) for the engine's CUDA sonames `needs`. `linker(soname)` is the file the dynamic
    linker hands out for a soname (_linker_path); `major` is torch's CUDA major (None: no CUDA torch, nothing to match).
    dirs: the folders of torch's CUDA set, from the files the linker hands out for libcudart / libcublas of torch's
    major. torch loaded those at import, so they are torch's: a copy another package loads later is never handed out,
    so it never widens the set. pip spreads the set over sibling <package>/lib folders (nvidia/cublas/lib,
    nvidia/cuda_runtime/lib, ...), so when the two sit in different '<x>/lib' folders under one parent, every
    '<parent>/*/lib' is the set too: torch's cuSOLVER waits there until a linalg call loads it. conda and a system CUDA
    keep the set in one folder; nothing beside it is added (a sibling conda env is another set).
    provided: the sonames torch's set has (handed out from dirs, or a file there). preloads: torch's file for each of
    those nothing has loaded yet. Loaded before the engine, it is what the engine's NEEDED soname binds to, instead of
    the build host's RPATH copy or ld.so.cache's (measured: a CUDA 12.9 libcusolver.so.11 beside torch cu128's cuBLAS
    -> "undefined symbol: cublasSetEnvironmentMode")."""
    if major is None:
        return set(), set(), []
    anchors = {os.path.dirname(p) for p in (linker(f"libcudart.so.{major}"), linker(f"libcublas.so.{major}")) if p}
    dirs = set(anchors)
    parents = [os.path.dirname(os.path.dirname(a)) for a in anchors if os.path.basename(a) == "lib"]
    for parent in {p for p in parents if parents.count(p) > 1}:
        dirs.update(os.path.realpath(d) for d in glob.glob(os.path.join(parent, "*", "lib")))
    provided, preloads = set(), []
    for soname in sorted(needs):
        bound = linker(soname)
        if bound and os.path.dirname(os.path.realpath(bound)) in dirs:
            provided.add(soname)
            continue
        path = next((os.path.join(d, soname) for d in sorted(dirs) if os.path.exists(os.path.join(d, soname))), None)
        if path:
            provided.add(soname)
            if not bound:
                preloads.append(path)
            dirs.add(os.path.dirname(os.path.realpath(path)))   # the linker reports the file the soname resolves to
    return dirs, provided, preloads


def _assert_torch_cuda_family(provided, dirs, linker):
    """After the engine load, the file the linker hands out for each CUDA soname torch's set provides (the copy the
    engine is bound to) must be torch's. Anything else (the build host's toolkit through the engine's RPATH, a system
    CUDA through ld.so.cache, a copy another package loaded before torch needed that soname) puts a second
    cuBLAS/cuSOLVER family under the engine in torch's process: REFUSE, naming the file. A copy loaded after torch's
    is never handed out, so it is not the engine's and is left alone."""
    for soname in sorted(provided):
        bound = linker(soname)
        if bound and os.path.dirname(os.path.realpath(bound)) not in dirs:
            raise RuntimeError(
                f"qf_native: REFUSING the engine - it links {soname}, and this process resolves it to "
                f"{os.path.realpath(bound)}, which is not torch's copy (torch's CUDA libraries are in "
                f"{', '.join(sorted(dirs))}). One process must use one cuBLAS/cuSOLVER set: two of them fail to load or "
                f"compute wrong. Find what loads that file (another custom node, LD_PRELOAD, LD_LIBRARY_PATH) and "
                f"remove it.")


def load_lib():
    """Load + bind the engine library once. Path comes ONLY from resolve_so_path() (never a workflow
    input). Takes NO argument on purpose — a `so_path` parameter is the attack surface just removed.
    FORK-2: the fail-closed CUDA-toolchain guard runs BEFORE ctypes.CDLL, so a mismatched combination is
    refused rather than dlopen'd into torch's live CUDA context."""
    global _LIB, _LIB_PATH, _FINGERPRINT_PENDING
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is not None:   # another thread loaded it while this one waited
            return _LIB
        so_path = resolve_so_path()
        assert_toolchain_compatible(so_path)   # FORK-2 fail-closed torch-CUDA / .so-CUDA match check
        # torch's own copy of each CUDA library the engine needs goes in first (_torch_cuda_plan): torch maps its
        # cuSOLVER only after a linalg call, so without it the engine's libcusolver.so.11 comes from the build host's
        # RPATH or ld.so.cache — a copy that need not match torch's cuBLAS. After the load _assert_torch_cuda_family
        # checks every one of them is torch's.
        torch_major = _torch_cuda_major() if "torch" in sys.modules else None   # never imports torch here
        cuda_dirs, cuda_provided, cuda_preloads = _torch_cuda_plan(_engine_cuda_needs(so_path), _linker_path,
                                                                   torch_major)
        for p in cuda_preloads:
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_LOCAL)
            except OSError as e:
                raise RuntimeError(f"qf_native: could not load torch's own {p}: {e}") from e
        # Sidecar preloads: a library the engine NEEDS that sits next to it may differ from this machine's copy
        # (measured: a scratch engine linked the build box's dynamic OpenCV 4.5d; this box ships 4.6 -> dlopen refused
        # — the SS7.5 portability class); preloaded, it satisfies the engine's NEEDED soname from the loaded image.
        # Only the engine's own DT_NEEDED closure over that folder is loaded (_sidecar_preloads): once every lib*.so
        # there was, which mapped 17 engine builds into one ComfyUI, and three builds of libquantfunc_attention.so
        # that aborted it at exit ("double free or corruption", their static destructors colliding). A generic retry
        # loop finds dependency order. Modes: libquantfunc_attention.so RTLD_GLOBAL (qfa symbol export — the proven
        # in-ComfyUI arm); everything else RTLD_LOCAL, so a GLOBAL OpenCV cannot hijack symbol binding of ComfyUI's own
        # bundled cv2. A sidecar that never loads is skipped silently here: the engine dlopen below then fails LOUD with
        # the true unresolved soname.
        so_dir = os.path.dirname(so_path)
        pending = [f for f in _sidecar_preloads(so_path) if f not in cuda_provided]   # torch's copy serves those
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
        ident = _file_identity(so_path)   # BEFORE the load: a file swapped in during it fails the check below, closed
        try:
            lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            # an installed pair that will not load: re-download it once per release, never in a loop
            raise RuntimeError(f"qf_native: {_engine_load_failed(so_path, e)}") from e
        _engine_load_ok(so_path)
        _assert_torch_cuda_family(cuda_provided, cuda_dirs, _linker_path)
        bound = _bind(lib)
        _LIB_PATH = so_path   # before _LIB: a concurrent caller that sees the library must see its path (the cache key)
        _LIB = bound
        if _LOG_LEVEL is not None:   # a loader asked for a level before the library was loaded
            _LIB.quantfunc_set_log_level(_LOG_LEVEL)
        _FINGERPRINT_PENDING = (_LIB, so_path, ident if _file_identity(so_path) == ident else None)
        _emit_fingerprint()
    return _LIB


def _file_identity(path):
    """(device, inode, size, mtime in ns) of the file at path now, or None."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def _log_lib_fingerprint(lib, so_path, ident):
    """[F6, 2026-09-19] ONE line naming the engine library this process actually dlopen'd — path, size, mtime,
    md5, and the engine's own quantfunc_version(). MEASURED need: the 远程-linux 5090 box ran a 4-day-old engine
    for days, and later a deployed library was silently replaced by an older file (found from a backup's mtime,
    not from any log). With this line the ComfyUI log states which binary produced every run. The line may print long
    after the load (it waits for an info-level loader), so it hashes only the file the process LOADED: ident is that
    file's identity at load, and a different file at the path now (replaced or rewritten) gets no md5 — naming it would
    certify a library this process never mapped. Never raises."""
    try:
        import hashlib
        h = hashlib.md5()
        with open(so_path, "rb") as fh:
            st = os.fstat(fh.fileno())      # the file this descriptor reads: no swap between the check and the hash
            if ident is None or (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) != ident:
                raise RuntimeError("the file at this path changed after the engine loaded it (or could not be "
                                   "identified at the load)")
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        ver = "?"
        try:
            fn = lib.quantfunc_version
            fn.restype = ctypes.c_char_p
            fn.argtypes = []
            ver = (fn() or b"?").decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - an old .so without the symbol still gets the file fingerprint
            pass
        info("[qf_native] engine lib: %s  size=%d  mtime=%s  md5=%s  quantfunc_version=%s"
              % (so_path, st.st_size, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                 h.hexdigest(), ver), flush=True)
    except Exception as e:  # noqa: BLE001
        info("[qf_native] engine lib: %s (fingerprint unavailable: %r)" % (so_path, e), flush=True)


def last_err(lib):
    """The engine's last error text, console-safe. Every plugin raise of an engine error goes through here, and ComfyUI
    logs an uncaught node exception to its strict console (execution.py); a character the code page cannot hold made
    that logging raise (#738)."""
    e = lib.quantfunc_last_error()
    return console_safe(e.decode("utf-8", "replace")) if e else "(none)"


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
    if isinstance(cfg, (str, bytes, bytearray)):
        try:
            cfg = json.loads(cfg)
        except Exception:  # noqa: BLE001 - unparseable JSON: the engine's own create refuses it loudly
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
                                      "block_cache_thresh", "sparse", "sparse_cdf", "quality",
                                      "video_enhance")):
                return True
            return any(_scan(v) for v in obj.values())
        if isinstance(obj, list):
            return any(_scan(v) for v in obj)
        return False
    if _scan(cfg):
        raise RuntimeError(
            "qf_native: a runtime SESSION knob (cache_mode / cache_thresh / "
            "step_cache - they ride every denoise_begin via "
            "QFSessionModelMixin.residency_opts) must never appear anywhere in a create "
            "config - that would bake it into the pipeline cache identity and rebuild "
            "the whole pipeline on every widget change.")


def _refuse_second_arch(device_idx):
    """A kernel library is ONE GPU architecture's code (SASS, no PTX), and a process loads ONE engine pair: the one
    installed for ComfyUI's device. A pipeline on a device whose SM that pair does not cover would fail inside a kernel
    launch ("no kernel image is available"), so it is refused here, before anything is created. A local build or the dev
    override is not an installed pair (its architectures are unknown), and neither is an unreadable SM: not checked."""
    if device_idx == _ENGINE_DEVICE or _engine_platform() is None or _engine_local_choice():
        return
    sm, m = _gpu_sm(device_idx), _installed_pair()[1]
    if sm is None or m is None or sm in m["sms"]:
        return
    raise RuntimeError(f"qf_native: GPU {device_idx} is SM {sm // 10}.{sm % 10}, but this ComfyUI runs the QuantFunc engine "
                       f"built for SM {' / '.join(f'{s // 10}.{s % 10}' for s in m['sms'])} (GPU {_ENGINE_DEVICE}). One "
                       f"engine serves one GPU architecture: run one ComfyUI per GPU architecture (start each with "
                       f"CUDA_VISIBLE_DEVICES set to its GPU).")


def make_create_params(*, model_dir, transformer_path=None, model_backend="svdq",
                       device_idx=0, config_json=None):
    """Normalize one retained recipe for both configuration and creation.

    Does not load a library or model. Caller dictionaries are not mutated.
    The returned structure owns its encoded strings; never log it because the
    config may contain authentication credentials.
    """
    device_idx = operator.index(device_idx)
    if not 0 <= device_idx < (1 << 31):
        raise ValueError("resource device must fit nonnegative int32")
    _refuse_second_arch(device_idx)                # one engine pair per process: one GPU architecture
    _refuse_session_knobs_in_create(config_json)   # [session-knobs] session knob != create key
    if isinstance(config_json, dict):
        config_json = dict(config_json)
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
        pass  # cache dir is an optimization - never block create on it
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
    return p


def create_pipeline(lib, *, prepared_resource=None, create_params=None, **create_kwargs):
    """Create from one recipe; serialize the resource view against close.

    Native code remains authoritative for device, lifecycle and owner adoption.
    Explicit params cannot be combined with a second, potentially different
    recipe. Same-thread resource reentry is unsupported.
    """
    if FACTORY_PREPARE_ONLY.get():
        raise RuntimeError("QuantFunc factory attempted model creation during dependency preparation")
    if create_params is not None and create_kwargs:
        raise ValueError("use retained create_params or create keywords, not both")
    p = make_create_params(**create_kwargs) if create_params is None else create_params
    if not isinstance(p, InitParams):
        raise TypeError("creation requires InitParams")
    # Explicit structs must not bypass the common session/create boundary.
    _refuse_session_knobs_in_create(p.config_json)
    if prepared_resource is not None:
        if prepared_resource._lib is not lib:
            raise ValueError("prepared resource belongs to a different native library wrapper")
        create_with_resource = getattr(lib, "quantfunc_create_with_resource", None)
        if create_with_resource is None:
            raise NativeContractUnavailable("QuantFunc library lacks quantfunc_create_with_resource")
        create_with_resource.restype = ctypes.c_int
        create_with_resource.argtypes = [ctypes.POINTER(InitParams), ctypes.c_void_p,
                                         ctypes.POINTER(ctypes.c_void_p)]
    handle = ctypes.c_void_p()
    if prepared_resource is not None:
        with prepared_resource._lock:
            prepared_resource._check_open()
            st = create_with_resource(ctypes.byref(p), prepared_resource._pointer, ctypes.byref(handle))
            if st != QUANTFUNC_OK or not handle:
                raise RuntimeError(f"quantfunc_create_with_resource failed st={st}: {last_err(lib)}")
        return handle
    st = lib.quantfunc_create(ctypes.byref(p), ctypes.byref(handle))
    if st != QUANTFUNC_OK:
        raise RuntimeError(f"quantfunc_create failed st={st}: {last_err(lib)}")
    return handle


class ResidentEstimateParams(ctypes.Structure):
    """Mirror of quantfunc_resident_estimate_params_t (engine >= 36f4ed2cb)."""
    _fields_ = [("model_dir", ctypes.c_char_p), ("transformer_weights", ctypes.c_char_p),
                ("server_url", ctypes.c_char_p), ("api_key", ctypes.c_char_p), ("device_idx", ctypes.c_int)]


def estimate_resident_bytes(lib, model_dir, device_idx=0, transformer_path=None, server_url=None, api_key=None):
    """Query the native pre-load transformer pack law, preserving unsupported/error.

    This is model capacity, not current residency or complete request peak.
    Coverage is defined by the native ABI's checkpoint/tier contract; never
    substitute a file size when that contract cannot model the input.
    """
    if not model_dir and not transformer_path:
        raise ValueError("model_dir or transformer_path is required for the capacity estimate")
    fn = getattr(lib, "quantfunc_estimate_resident_bytes", None)
    if fn is None:
        raise RuntimeError("QuantFunc library lacks quantfunc_estimate_resident_bytes; update the native library")
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.POINTER(ResidentEstimateParams), ctypes.POINTER(ctypes.c_uint64)]
    p = ResidentEstimateParams(model_dir=_enc(model_dir) if model_dir else None,
                               transformer_weights=_enc(transformer_path) if transformer_path else None,
                               server_url=_enc(server_url) if server_url else None,
                               api_key=_enc(api_key) if api_key else None, device_idx=int(device_idx))
    out = ctypes.c_uint64(0)
    st = fn(ctypes.byref(p), ctypes.byref(out))
    if st != QUANTFUNC_OK:
        raise RuntimeError(f"QuantFunc capacity estimate failed (status {st}): {last_err(lib)}")
    return int(out.value)


class QFEngineHandle:
    """Owns the .so + created pipeline + the (single, per-pipeline) open denoise session.
    Tracks the session HERE (not only on the model shim) so a stale session from a failed run is
    cleaned up before the next begin, and destroy() always closes it."""
    @classmethod
    def create(cls, lib, *, capacity_bytes=0, footprint_bytes=0, prepared_resource=None,
               create_params=None, **create_kwargs):
        """Consume a retained cold identity, or prepare one for standalone use.

        A caller-provided view remains caller-owned on failure. Retaining this
        same Python object lets host resource adapters outlive pipeline teardown;
        the existing NativeResource finalizer closes its view at last ownership.
        No host registration is implied by this factory.
        """
        if create_params is not None and create_kwargs:
            raise ValueError("use retained create_params or create keywords, not both")
        if create_params is not None and not isinstance(create_params, InitParams):
            raise TypeError("creation requires InitParams")
        device = create_params.device_idx if create_params is not None else create_kwargs.get("device_idx", 0)
        resource = (NativeResource.prepare(lib, device)
                    if prepared_resource is None else prepared_resource)
        pipeline = None
        try:
            pipeline = create_pipeline(lib, prepared_resource=resource,
                                       create_params=create_params, **create_kwargs)
            return cls(lib, pipeline, footprint_bytes=footprint_bytes, resource=resource,
                       capacity_bytes=capacity_bytes)
        except BaseException:
            try:
                if pipeline is not None:
                    lib.quantfunc_destroy(pipeline)
            finally:
                if prepared_resource is None:
                    resource.close()
            raise

    def __init__(self, lib, pipeline, footprint_bytes=0, resource=None, capacity_bytes=0):
        self.lib = lib
        self.pipeline = pipeline
        self.resource = resource
        self.capacity_bytes = int(capacity_bytes)
        self.footprint_bytes = int(footprint_bytes)
        self.current_session = None          # ctypes.c_void_p of the open session, or None
        self.step_count = 0                  # total denoise_step calls (instrument)
        self.sampler_step_count = 0          # distinct sampler steps (instrument)
        self.unloaded = False                # co-eviction: True after unload_vram() freed VRAM (auto-reloads on next generate)
        # The LoRA set this pipeline currently runs, as QFLazyEngine._lora_sig() spells it. A pipeline is created with
        # NO LoRA (the cache key is the weights only), so it starts at the base; pipeline_update swaps it in place.
        self.applied_lora_sig = "[]"

    def pipeline_update(self, update):
        """ONE quantfunc_pipeline_update on this live pipeline (a runtime mutation between generations: no rebuild,
        no reload). `update` is the JSON object the C API takes (e.g. {"lora": [...]}: the declarative full set, []
        restores the base). A refusal (busy, an unsupported entry, an OOM) raises with the engine's own message."""
        if self.pipeline is None:
            raise RuntimeError("QuantFunc pipeline_update: no live pipeline")
        fn = getattr(self.lib, "quantfunc_pipeline_update", None)
        if fn is None:
            raise RuntimeError("QuantFunc library lacks quantfunc_pipeline_update; update the native library")
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        st = fn(self.pipeline, json.dumps(update).encode())
        if st != QUANTFUNC_OK:
            raise RuntimeError(f"QuantFunc pipeline_update failed (status {st}): {last_err(self.lib)}")

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
                    say(f"[qf_native] WARNING quantfunc_denoise_end returned status={st}: "
                        f"{last_err(self.lib)}", flush=True)
                except Exception:  # noqa: BLE001
                    pass
        except Exception as _end_exc:  # noqa: BLE001 - end is best-effort on teardown
            ok = False
            try:  # R7 observability: this branch previously left no trail (leg-1 was
                #   diagnosed FROM logs — a silent branch here would blind the next diagnosis)
                say(f"[qf_native] WARNING quantfunc_denoise_end raised: {_end_exc!r}", flush=True)
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
        """Return confirmed native release bytes; failure is not a zero result.

        The legacy native primitive is device-scoped. A shortfall is returned
        to the host, which decides whether to request complete eviction.
        """
        if self.pipeline is None or bytes_requested <= 0:
            return 0
        if not hasattr(self.lib, "quantfunc_partial_unload"):
            raise RuntimeError("QuantFunc library lacks quantfunc_partial_unload; update the native library")
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
            raise RuntimeError("QuantFunc cannot unload VRAM while a session is still active")
        freed = ctypes.c_int64(0)
        st = self.lib.quantfunc_partial_unload(self.pipeline,
                                               ctypes.c_uint64(bytes_requested),
                                               ctypes.byref(freed))
        if st != QUANTFUNC_OK:
            raise RuntimeError(f"QuantFunc partial VRAM unload failed: {last_err(self.lib)}")
        if freed.value < 0:
            raise RuntimeError("QuantFunc partial VRAM unload returned negative released bytes")
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
        allocator's cached pool it already holds. This legacy estimate is not a
        complete cold-request peak bound. A successful zero remains ambiguous
        (covered or not measured); an ABI error/missing query is never that zero."""
        if self.pipeline is None or self.unloaded:
            return 0
        if not hasattr(self.lib, "quantfunc_vram_need_bytes"):
            raise RuntimeError("QuantFunc library lacks quantfunc_vram_need_bytes; update the native library")
        dims = [int(d) for d in latent_shape]
        if not dims or any(d <= 0 for d in dims):
            return 0
        arr = (ctypes.c_int64 * len(dims))(*dims)
        out = ctypes.c_uint64(0)
        st = self.lib.quantfunc_vram_need_bytes(self.pipeline, arr, len(dims), ctypes.byref(out))
        if st != QUANTFUNC_OK:
            raise RuntimeError(f"QuantFunc demand query failed: {last_err(self.lib)}")
        return int(out.value)

    def unload_vram(self):
        """Synchronously reclaim and return native confirmed bytes, never file size.

        Repeated calls still ask native: an earlier release may have left live
        allocations or pages that have since become reclaimable. This ABI's
        count is device-scoped and excludes unowned shared CUDA pool backing.
        """
        if self.pipeline is None:
            return 0
        if not hasattr(self.lib, "quantfunc_unload_sync_ex"):
            raise RuntimeError("QuantFunc library lacks quantfunc_unload_sync_ex; update the native library")
        import os as _os
        if _os.environ.get("QF_NATIVE_PROF") == "1":
            import traceback as _tb
            frames = _tb.extract_stack(limit=5)[:-1]
            chain = " <- ".join(f"{_os.path.basename(f.filename)}:{f.lineno}:{f.name}"
                                for f in reversed(frames))
            say(f"[qf_prof] unload_vram CALLER: {chain}", flush=True)
        self.end_session_if_open()          # a live session on unloaded VRAM would be a UAF on reuse
        if self.current_session is not None:
            raise RuntimeError("QuantFunc cannot unload VRAM while a session is still active")
        freed = ctypes.c_uint64(0)
        st = self.lib.quantfunc_unload_sync_ex(self.pipeline, ctypes.byref(freed))
        if st != QUANTFUNC_OK:
            raise RuntimeError(f"QuantFunc VRAM unload failed: {last_err(self.lib)}")
        if _os.environ.get("QF_NATIVE_PROF") == "1" and self.resource is not None:
            try:
                owner = self.resource.query()
                with NativeResource.shared(self.lib, owner.device) as shared:
                    _dbg_prof(f"unload resource owner={owner} shared={shared.query()} confirmed_freed={freed.value}")
            except Exception as diagnostic_error:
                _dbg_prof(f"unload resource snapshot unavailable ({type(diagnostic_error).__name__})")
        self.unloaded = True
        return int(freed.value)

    def destroy(self):
        self.end_session_if_open()
        if self.pipeline is not None:
            try:
                self.lib.quantfunc_destroy(self.pipeline)
            except Exception:  # noqa: BLE001
                pass
            self.pipeline = None
        if self.resource is not None:
            # Drop this handle's reference, not another consumer's native view.
            # NativeResource's existing finalizer closes the last retained view;
            # neither this nor view destruction certifies physical GPU release.
            self.resource = None
