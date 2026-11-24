"""qf_native.qf_capi -- frozen 25-symbol qf_* C-API ABI declarations and ColdNeed value object.

Pure stdlib (ctypes + typing) with zero side effects at import time.
Structs and function signatures mirror include/qf_capi.h verbatim.
"""
import ctypes
from typing import NamedTuple, Optional

# qf_status_t (include/qf_capi.h)
QF_OK = 0
QF_ERR_INVALID_ARG = 1
QF_ERR_BACKEND = 2
QF_ERR_OOM = 3
QF_ERR_IO = 4
QF_ERR_INTERNAL = 5
QF_ERR_BUSY = 6
QF_ERR_AUTH = 7
QF_ERR_UNSUPPORTED = 8

# qf_dtype_t (include/qf_capi.h)
QF_DTYPE_FP32 = 0
QF_DTYPE_FP16 = 1
QF_DTYPE_BF16 = 2
QF_DTYPE_INT8 = 3
QF_DTYPE_INT32 = 4
QF_DTYPE_FP8_E4M3 = 5
QF_DTYPE_INT4_PACKED = 6
QF_DTYPE_E2M1_PACKED = 7
QF_FP32, QF_FP16, QF_BF16 = QF_DTYPE_FP32, QF_DTYPE_FP16, QF_DTYPE_BF16

# Capability bits & limits (include/qf_capi.h)
QF_ATTN_CAP_FP16 = 1 << 0
QF_ATTN_CAP_INT8 = 1 << 1
QF_ATTN_CAP_INT4 = 1 << 2
QF_MAX_STEP_REFS = 16
QF_UINT64_MAX = (1 << 64) - 1


# -- Frozen 8 C structs (include/qf_capi.h) ----------------------------------

class QFAuthConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("server_url", ctypes.c_char_p),
        ("api_key", ctypes.c_char_p),
        ("cache_dir", ctypes.c_char_p),
        ("device_id", ctypes.c_int32),
    ]


class QFDeviceCaps(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("backend_type", ctypes.c_uint32),
        ("device_id", ctypes.c_int32),
        ("arch_code", ctypes.c_int32),
        ("compute_unit_count", ctypes.c_int32),
        ("total_vram_bytes", ctypes.c_uint64),
        ("free_vram_bytes", ctypes.c_uint64),
        ("attn_caps_mask", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("backend_feature_flags", ctypes.c_uint64),
        ("device_name", ctypes.c_char * 128),
        ("uuid_hex", ctypes.c_char * 65),
        ("plugin_path", ctypes.c_char * 512),
    ]


class QFXfmCreateParams(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("weights_path", ctypes.c_char_p),
        ("device_id", ctypes.c_int32),
        ("config_json", ctypes.c_char_p),
    ]


class QFVramEstimate(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("resident_weight_bytes", ctypes.c_uint64),
        ("min_offload_bytes", ctypes.c_uint64),
        ("working_set_floor_bytes", ctypes.c_uint64),
        ("recommended_offload_mode", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


class QFSessionBeginParams(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("num_steps", ctypes.c_int32),
        ("max_context_dims", ctypes.c_int32 * 3),
        ("cond_dtype", ctypes.c_int32),
        ("producer_stream", ctypes.c_size_t),
        ("options_json", ctypes.c_char_p),
    ]


class QFStepParams(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("latent_in", ctypes.c_void_p),
        ("velocity_out", ctypes.c_void_p),
        ("velocity_out_capacity", ctypes.c_uint64),
        ("dims", ctypes.c_int32 * 5),
        ("dtype", ctypes.c_int32),
        ("sigma", ctypes.c_float),
        ("step_index", ctypes.c_int32),
        ("total_steps", ctypes.c_int32),
        ("context", ctypes.c_void_p),
        ("context_dims", ctypes.c_int32 * 3),
        ("context_dtype", ctypes.c_int32),
        ("cfg_context_key", ctypes.c_uint64),
        ("producer_stream", ctypes.c_size_t),
    ]


class QFStepAvParams(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("base", QFStepParams),
        ("audio_latent_in", ctypes.c_void_p),
        ("audio_velocity_out", ctypes.c_void_p),
        ("audio_velocity_out_capacity", ctypes.c_uint64),
        ("audio_dims", ctypes.c_int32 * 4),
        ("audio_dtype", ctypes.c_int32),
        ("audio_scale", ctypes.c_float),
        ("audio_reference_active", ctypes.c_uint32),
    ]


class QFStepRefsParams(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("base", QFStepParams),
        ("num_refs", ctypes.c_int32),
        ("ref_latents", ctypes.POINTER(ctypes.c_void_p)),
        ("ref_dims", ctypes.POINTER(ctypes.c_int32)),
        ("image_slots", ctypes.POINTER(ctypes.c_int32)),
    ]


class ColdNeed(NamedTuple):
    """Cold create-time VRAM estimate from qf_xfm_estimate_vram."""

    bytes: Optional[int]
    note: Optional[str] = None
    min_offload_bytes: Optional[int] = None
    working_set_floor_bytes: Optional[int] = None
    recommended_offload_mode: Optional[int] = None

    @property
    def resident_weight_bytes(self):
        return self.bytes

    def effective_model_bytes(self, total_vram_bytes=None, vram_budget_bytes=0):
        """Budget- and VRAM-aware effective model bytes for ComfyUI pre-create eviction planning."""
        if self.bytes is None:
            return 0
        resident = int(self.bytes)
        min_off = int(self.min_offload_bytes or 0)
        work = int(self.working_set_floor_bytes or 0)
        phys = int(total_vram_bytes) if (total_vram_bytes is not None and total_vram_bytes > 0) else 0
        budget = int(vram_budget_bytes) if (vram_budget_bytes is not None and vram_budget_bytes > 0) else 0
        eff_total = min(phys, budget) if (phys > 0 and budget > 0) else (phys or budget)
        fits_phys = eff_total > 0 and (resident + work) <= (eff_total * 9) // 10
        is_offload = (self.recommended_offload_mode == 1) or (eff_total > 0 and min_off > 0 and not fits_phys)
        if not is_offload or min_off <= 0:
            return resident
        if eff_total > 0 and not fits_phys:
            target = (eff_total * 17) // 20 - work
            return min(resident, max(min_off, target))
        return min(resident, min_off)


# -- Frozen 25-symbol qf_* C-API binding -------------------------------------

_V = ctypes.c_void_p
_CAPI_SIGNATURES = (
    ("qf_version", ctypes.c_char_p, []),
    ("qf_last_error", ctypes.c_char_p, []),
    ("qf_set_log_level", None, [ctypes.c_int]),
    ("qf_get_log_level", ctypes.c_int, []),
    ("qf_set_log_stderr", None, [ctypes.c_int]),
    ("qf_auth_configure", ctypes.c_int, [ctypes.POINTER(QFAuthConfig)]),
    ("qf_auth_set_api_key", ctypes.c_int, [_V, ctypes.c_char_p]),
    ("qf_device_query_caps", ctypes.c_int, [ctypes.c_int32, ctypes.POINTER(QFDeviceCaps)]),
    ("qf_xfm_estimate_vram", ctypes.c_int, [ctypes.POINTER(QFXfmCreateParams), ctypes.POINTER(QFVramEstimate)]),
    ("qf_default_xfm_create_params", QFXfmCreateParams, []),
    ("qf_xfm_create", ctypes.c_int, [ctypes.POINTER(QFXfmCreateParams), ctypes.POINTER(_V)]),
    ("qf_xfm_destroy", None, [_V]),
    ("qf_xfm_set_loras", ctypes.c_int, [_V, ctypes.c_char_p]),
    ("qf_xfm_unload_vram", ctypes.c_int, [_V, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]),
    ("qf_xfm_resident_vram_bytes", ctypes.c_int, [_V, ctypes.POINTER(ctypes.c_uint64)]),
    ("qf_xfm_vram_need_bytes", ctypes.c_int,
     [_V, ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]),
    ("qf_default_session_begin_params", QFSessionBeginParams, []),
    ("qf_default_step_params", QFStepParams, []),
    ("qf_default_step_av_params", QFStepAvParams, []),
    ("qf_default_step_refs_params", QFStepRefsParams, []),
    ("qf_xfm_session_begin", ctypes.c_int, [_V, ctypes.POINTER(QFSessionBeginParams), ctypes.POINTER(_V)]),
    ("qf_xfm_step", ctypes.c_int, [_V, ctypes.POINTER(QFStepParams)]),
    ("qf_xfm_step_av", ctypes.c_int, [_V, ctypes.POINTER(QFStepAvParams)]),
    ("qf_xfm_step_refs", ctypes.c_int, [_V, ctypes.POINTER(QFStepRefsParams)]),
    ("qf_xfm_session_end", ctypes.c_int, [_V]),
)

__all__ = [
    "QF_OK", "QF_ERR_INVALID_ARG", "QF_ERR_BACKEND", "QF_ERR_OOM", "QF_ERR_IO",
    "QF_ERR_INTERNAL", "QF_ERR_BUSY", "QF_ERR_AUTH", "QF_ERR_UNSUPPORTED",
    "QF_DTYPE_FP32", "QF_DTYPE_FP16", "QF_DTYPE_BF16", "QF_DTYPE_INT8", "QF_DTYPE_INT32",
    "QF_DTYPE_FP8_E4M3", "QF_DTYPE_INT4_PACKED", "QF_DTYPE_E2M1_PACKED",
    "QF_FP32", "QF_FP16", "QF_BF16",
    "QF_ATTN_CAP_FP16", "QF_ATTN_CAP_INT8", "QF_ATTN_CAP_INT4",
    "QF_MAX_STEP_REFS", "QF_UINT64_MAX",
    "QFAuthConfig", "QFDeviceCaps", "QFXfmCreateParams", "QFVramEstimate",
    "QFSessionBeginParams", "QFStepParams", "QFStepAvParams", "QFStepRefsParams",
    "ColdNeed", "_CAPI_SIGNATURES",
]
