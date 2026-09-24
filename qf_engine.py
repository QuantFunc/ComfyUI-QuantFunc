"""qf_native.qf_engine — self-contained ctypes bridge to libquantfunc.so for the
native ComfyUI loader. Structs mirror include/quantfunc.h (session structs copied
verbatim from the PROVEN tests/scripts/native_session_t1.py). No tests/lib dependency.
"""
import ctypes
from contextvars import ContextVar
import re as _re_soname
_SONAME_RE = _re_soname.compile(r"lib[^/]*\.so(?:\.\d+[a-z]?)*")   # lib*.so, lib*.so.5, libopencv_core.so.4.5d
import json
import os
import time
import platform
import re
import struct
import operator
import threading
import weakref
from typing import NamedTuple, Optional

QUANTFUNC_OK = 0
QUANTFUNC_RESOURCE_ABI_VERSION = 1
QUANTFUNC_RESOURCE_CREATION_ABI_VERSION = 1
QUANTFUNC_RESOURCE_RESIDENCY_ABI_VERSION = 1
QUANTFUNC_RESOURCE_DOMAIN_ABI_VERSION = 1
QUANTFUNC_RESOURCE_CAPACITY_ABI_VERSION = 1
QUANTFUNC_RESOURCE_GRANT_ABI_VERSION = 1
QUANTFUNC_RESOURCE_DOMAIN_GRANTS_ABI_VERSION = 1
QUANTFUNC_RESOURCE_LIFECYCLE_ABI_VERSION = 1
QUANTFUNC_ERROR_UNSUPPORTED = 8
QUANTFUNC_RESOURCE_READY = 0
QUANTFUNC_RESOURCE_BUSY = 1
QUANTFUNC_RESOURCE_UNKNOWN = 2
QUANTFUNC_RESOURCE_CLOSED = 3
QUANTFUNC_RESOURCE_CAPACITY_UNSUPPORTED = 4
QUANTFUNC_RESOURCE_CAP_QUERY = 1
QUANTFUNC_RESOURCE_CAP_RELEASE_ALL = 4
QUANTFUNC_RESOURCE_GRANT_OWNER = 1
QUANTFUNC_RESOURCE_GRANT_SHARED = 2
QUANTFUNC_RESOURCE_GRANT_DEVICE = 4
QUANTFUNC_RESOURCE_PHASE_SHARED = 1
QUANTFUNC_RESOURCE_PHASE_PREPARED = 2
QUANTFUNC_RESOURCE_PHASE_ATTACHED = 3
QUANTFUNC_RESOURCE_PHASE_CLOSED = 4
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


class _ResourceGrant(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("state", ctypes.c_uint32), ("enrolled", ctypes.c_uint32),
        ("limit_bytes", ctypes.c_uint64), ("pending_bytes", ctypes.c_uint64),
    ]


class _ResourceDomainGrants(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("mask", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
        ("owner_limit_bytes", ctypes.c_uint64),
        ("shared_limit_bytes", ctypes.c_uint64),
        ("device_limit_bytes", ctypes.c_uint64),
    ]


class _ResourceDomainGrantsResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("state", ctypes.c_uint32), ("applied_mask", ctypes.c_uint32),
    ]


class ResourceGrant(NamedTuple):
    """Native finite permission; non-Ready or unenrolled bytes are not usable."""
    state: int
    enrolled: Optional[bool]
    limit_bytes: Optional[int]
    pending_bytes: Optional[int]


class ResourceDomainGrantsResult(NamedTuple):
    state: int
    applied_mask: Optional[int]


class _ResourceLifecycle(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("state", ctypes.c_uint32), ("phase", ctypes.c_uint32),
    ]


class ResourceLifecycle(NamedTuple):
    state: int
    phase: Optional[int]


class ResourceResidency(NamedTuple):
    """Native accounted occupancy; not demand, capacity or complete backend coverage."""
    state: int
    resident_bytes: Optional[int]


class ResourceDomainResidency(NamedTuple):
    state: int
    resident_bytes: Optional[int]


class ResourceCapacity(NamedTuple):
    state: int
    component_count: Optional[int]
    required_persistent_bytes: Optional[int]


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
        """Create owned identity before loading; no budget or grant is implied."""
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
        """Bind entry options, not a capacity estimate or load grant."""
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
            return ResourceResidency(out.state, out.resident_bytes if out.state == QUANTFUNC_RESOURCE_READY else None)

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

    def _grant_call(self, name, limit=None):
        with self._lock:
            self._check_open()
            function = getattr(self._lib, name, None)
            if function is None:
                raise RuntimeError(f"QuantFunc library lacks {name}; update the native library")
            function.restype = ctypes.c_int
            function.argtypes = [ctypes.c_void_p] + ([ctypes.c_uint64] if limit is not None else []) + [ctypes.POINTER(_ResourceGrant)]
            out = _ResourceGrant(ctypes.sizeof(_ResourceGrant), QUANTFUNC_RESOURCE_GRANT_ABI_VERSION)
            args = [self._pointer] + ([limit] if limit is not None else []) + [ctypes.byref(out)]
            if function(*args) != QUANTFUNC_OK:
                raise RuntimeError(f"QuantFunc {name} failed: {last_err(self._lib)}")
            if out.state != QUANTFUNC_RESOURCE_READY:
                return ResourceGrant(out.state, None, None, None)
            return ResourceGrant(out.state, bool(out.enrolled),
                                 out.limit_bytes if out.enrolled else None,
                                 out.pending_bytes if out.enrolled else None)

    def enroll_host(self):
        return self._grant_call("quantfunc_resource_enroll_host")

    def query_grant(self):
        return self._grant_call("quantfunc_resource_query_grant")

    def set_grant(self, limit):
        limit = operator.index(limit)
        if not 0 <= limit < (1 << 64):
            raise ValueError("resource grant must fit uint64")
        return self._grant_call("quantfunc_resource_set_grant", limit)

    def query_device_grant(self):
        """Read the device-wide QuantFunc admission ceiling.

        Native accepts only a Shared resource view. The ceiling covers Shared
        plus every Owned identity on that device; it is permission, not
        occupancy or a reservation.
        """
        return self._grant_call("quantfunc_resource_query_device_grant")

    def set_device_grant(self, limit):
        limit = operator.index(limit)
        if not 0 <= limit < (1 << 64):
            raise ValueError("device grant must fit uint64")
        return self._grant_call("quantfunc_resource_set_device_grant", limit)

    def set_domain_grants(self, owned, mask, *, owner_limit_bytes=0,
                          shared_limit_bytes=0, device_limit_bytes=0):
        """Atomically publish selected Owner/Shared/device limits through a Shared view."""
        mask = operator.index(mask)
        known = (QUANTFUNC_RESOURCE_GRANT_OWNER | QUANTFUNC_RESOURCE_GRANT_SHARED |
                 QUANTFUNC_RESOURCE_GRANT_DEVICE)
        if mask == 0 or mask & ~known:
            raise ValueError("domain grant mask must select only Owner/Shared/device")
        if bool(mask & QUANTFUNC_RESOURCE_GRANT_OWNER) != (owned is not None):
            raise ValueError("domain grant Owner selection and owned resource must match")
        if owned is not None:
            if not isinstance(owned, NativeResource):
                raise TypeError("owned domain grant target must be a NativeResource")
            if owned is self:
                raise ValueError("domain grant Owner target cannot be the Shared view")
            if library_identity(owned._lib) != library_identity(self._lib):
                raise ValueError("domain grant views belong to different native libraries")

        limits = [operator.index(value) for value in
                  (owner_limit_bytes, shared_limit_bytes, device_limit_bytes)]
        if any(value < 0 or value >= (1 << 64) for value in limits):
            raise ValueError("domain grant limits must fit uint64")
        selections = (QUANTFUNC_RESOURCE_GRANT_OWNER, QUANTFUNC_RESOURCE_GRANT_SHARED,
                      QUANTFUNC_RESOURCE_GRANT_DEVICE)
        if any(not mask & bit and value != 0 for bit, value in zip(selections, limits)):
            raise ValueError("unselected domain grant limits must be zero")

        resources = [self] if owned is None else sorted((self, owned), key=id)

        def call():
            for resource in resources:
                resource._check_open()
            function = getattr(self._lib, "quantfunc_resource_set_domain_grants", None)
            if function is None:
                raise NativeContractUnavailable(
                    "QuantFunc library lacks quantfunc_resource_set_domain_grants")
            function.restype = ctypes.c_int
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.POINTER(_ResourceDomainGrants),
                                 ctypes.POINTER(_ResourceDomainGrantsResult)]
            command = _ResourceDomainGrants(
                ctypes.sizeof(_ResourceDomainGrants),
                QUANTFUNC_RESOURCE_DOMAIN_GRANTS_ABI_VERSION, mask, 0, *limits)
            out = _ResourceDomainGrantsResult(
                ctypes.sizeof(_ResourceDomainGrantsResult),
                QUANTFUNC_RESOURCE_DOMAIN_GRANTS_ABI_VERSION)
            owned_pointer = ctypes.c_void_p() if owned is None else owned._pointer
            if function(self._pointer, owned_pointer, ctypes.byref(command),
                        ctypes.byref(out)) != QUANTFUNC_OK:
                raise RuntimeError(f"QuantFunc atomic domain grant failed: {last_err(self._lib)}")
            if out.state != QUANTFUNC_RESOURCE_READY:
                return ResourceDomainGrantsResult(out.state, None)  # all-or-none: nothing applied; BUSY may be re-issued
            if out.applied_mask != mask:
                raise RuntimeError(
                    f"QuantFunc atomic domain grant unavailable (state={out.state}, "
                    f"applied_mask={out.applied_mask}, requested_mask={mask})")
            return ResourceDomainGrantsResult(out.state, out.applied_mask)

        if len(resources) == 1:
            with resources[0]._lock:
                return call()
        with resources[0]._lock:
            with resources[1]._lock:
                return call()

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

        Does not change grants or revive a Closed identity. A retained Closed
        view is deliberately forwarded to native for final old-owner cleanup.
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
    # Reference-latent step (QwenImage-2.1 edit through the native loader). hasattr-gated like step_multi:
    # an older .so lacks the symbol and the qwenimage21 seam then refuses reference images with guidance.
    if hasattr(lib, "quantfunc_denoise_step_refs"):
        lib.quantfunc_denoise_step_refs.restype = ctypes.c_int
        lib.quantfunc_denoise_step_refs.argtypes = [v, ctypes.POINTER(DenoiseStepRefsParams)]
    # i2v cond-latent begin (cond-ABI builds only; hasattr-gated like step_multi — an old .so
    # simply lacks the symbol and the wan i2v sampling path then refuses with guidance).
    if hasattr(lib, "quantfunc_quality_fast_available"):
        # [quality] can the loaders' super_fast / fast take effect on CUDA device N — the ENGINE's own arming rule for
        # that GPU (the plugin keeps no GPU list): 1 yes, 0 no (balance / best_quality only), -1 bad device.
        lib.quantfunc_quality_fast_available.restype = ctypes.c_int
        lib.quantfunc_quality_fast_available.argtypes = [ctypes.c_int]
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
    if isinstance(cfg, (str, bytes, bytearray)):
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
                                      "block_cache_thresh", "sparse", "sparse_cdf", "quality")):
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
    _refuse_session_knobs_in_create(config_json)   # [session-knobs] session knob ≠ create key
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


def quality_fast_available_file(lib, model_dir, transformer_path=None, device_idx=0, server_url=None, api_key=None):
    """[quality] Can super_fast / fast take effect for THIS checkpoint on this GPU (quantfunc_quality_fast_available_file:
    the GPU's tier AND the file — a layer the fast mode speeds up, none stored in a form it cannot use)? 1 yes, 0 no, -1
    error; None = the engine predates the query (the caller keeps the GPU-level answer)."""
    fn = getattr(lib, "quantfunc_quality_fast_available_file", None)
    if fn is None:
        return None
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.POINTER(ResidentEstimateParams)]
    p = ResidentEstimateParams(model_dir=_enc(model_dir) if model_dir else None,
                               transformer_weights=_enc(transformer_path) if transformer_path else None,
                               server_url=_enc(server_url) if server_url else None,
                               api_key=_enc(api_key) if api_key else None, device_idx=int(device_idx))
    return int(fn(ctypes.byref(p)))


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
        No host registration or grant is implied by this factory.
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
            print(f"[qf_prof] unload_vram CALLER: {chain}", flush=True)
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
