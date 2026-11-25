"""qf_native.qf_modelpatcher -- family-agnostic ComfyUI ModelPatcher substrate for QuantFunc PC Engine.

Drives the 25-symbol qf_* C-API directly from QFModelPatcher:
  - loaded_size()      -> super().loaded_size() + qf_xfm_resident_vram_bytes
  - memory_required()  -> comfy-side bytes + qf_xfm_vram_need_bytes (or cold qf_xfm_estimate_vram)
  - partially_unload() -> qf_xfm_unload_vram + super().partially_unload
  - detach(unpatch_all=True) -> qf_xfm_unload_vram(UINT64_MAX) + super().detach
"""
from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
import inspect
import itertools
import json
import math
import sys
import time
from typing import NamedTuple
import weakref

import torch
import comfy.conds
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.nested_tensor
import comfy.patcher_extension

from . import qf_engine as qfe
from . import qf_sol


_QF_SAMPLING_GEOMETRY = ContextVar("quantfunc_sampling_geometry", default=None)
_QF_RUN_COUNTER = itertools.count(1)
_QF_RUN_ID = ContextVar("quantfunc_run_id", default=None)
_QF_ENGINE_CACHE_ACQUISITION = ContextVar("quantfunc_engine_cache_acquisition", default=None)
LOADER_SIG = ContextVar("quantfunc_loader_sig", default=None)
LOADER_API_KEY = ContextVar("quantfunc_loader_api_key", default=None)

# Symbolic kNoCtxKey: 0 disables engine step caches safely when ComfyUI provides no uuid.
_KNO_CTX_KEY = 0

# Plugin-layer QualityLaw default ratio for non-QI21 families (ltx25, minimax_h3, krea2) when quality_enhance=False.
_CAT_KEEP_DEFAULT = 0.80

# ComfyUI CFGGuider.outer_sample positional index for latent_shapes.
_OUTER_SAMPLE_LATENT_SHAPES_ARG = 8

# Bounded wait (seconds) before retrying session_begin when a previous session is draining.
_BUSY_BEGIN_RETRY_S = 2.0

# Bounded wait (seconds) before retrying session_end when swapping LoRAs after an interrupted run.
_LORA_DRAIN_RETRY_S = 2.0

_ABSENT = object()
_log = qfe.logger(__name__)

LORA_HELP_URL = "https://www.quantfunc.com/docs/lora"
NATIVE_LORA_NODE = "QuantFunc Native LoRA"
QF_LORA_STACK_ATTR = "_qf_lora_stack"
QF_LORA_REBUILD_ATTR = "_qf_rebuild"


class _EngineCacheAcquisition:
    """Atomic cache lookup/publication transaction for one lazy consumer."""

    def __init__(self, consumer):
        self.consumer = consumer
        self._pin_rollbacks = []

    def register_pin(self, rollback):
        self._pin_rollbacks.append(rollback)

    def commit(self):
        self._pin_rollbacks.clear()

    def rollback(self):
        callbacks, self._pin_rollbacks = self._pin_rollbacks, []
        for callback in reversed(callbacks):
            try:
                callback()
            except Exception:  # noqa: BLE001
                pass


def _current_engine_cache_acquisition():
    return _QF_ENGINE_CACHE_ACQUISITION.get()


@contextmanager
def _engine_cache_acquisition(consumer):
    current = _QF_ENGINE_CACHE_ACQUISITION.get()
    if current is not None:
        if current.consumer is not consumer:
            raise RuntimeError("nested QuantFunc cache acquisition changed consumer identity")
        yield current
        return
    acquisition = _EngineCacheAcquisition(consumer)
    token = _QF_ENGINE_CACHE_ACQUISITION.set(acquisition)
    try:
        yield acquisition
    except BaseException:
        acquisition.rollback()
        raise
    else:
        acquisition.commit()
    finally:
        _QF_ENGINE_CACHE_ACQUISITION.reset(token)


def _qf_dtype(torch_dtype):
    return {torch.float32: qfe.QF_DTYPE_FP32, torch.float16: qfe.QF_DTYPE_FP16,
            torch.bfloat16: qfe.QF_DTYPE_BF16}[torch_dtype]


def _interrupt_poll_end_session_on_raise(qf):
    """Poll ComfyUI interrupt; close open session on ANY BaseException before re-raising."""
    try:
        comfy.model_management.throw_exception_if_processing_interrupted()
    except BaseException:
        qf.end_session_if_open()
        raise


class _CtxKeyAssigner:
    """Assigns each conditioning group a stable per-generation symbolic key from ComfyUI's uuid."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._uuid_to_key = {}

    def key(self, cond_uuid):
        if cond_uuid is None:
            return _KNO_CTX_KEY
        k = self._uuid_to_key.get(cond_uuid)
        if k is None:
            k = len(self._uuid_to_key) + 1
            self._uuid_to_key[cond_uuid] = k
        return k


class _QFStub(torch.nn.Module):
    """Zero-byte stand-in for BaseModel.diffusion_model; fails loud if ever called directly."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype

    def arm_concat_shape(self, in_channels):
        pe = torch.nn.Module()
        pe.weight = torch.nn.Parameter(
            torch.empty(1, int(in_channels), 0, 0, 0, dtype=self.dtype), requires_grad=False)
        self.patch_embedding = pe

    def forward(self, *a, **k):
        raise RuntimeError(
            "QF stub diffusion_model must never be called - _apply_model is overridden; "
            "a ComfyUI upgrade may have changed the apply-model dispatch")


def current_torch_device():
    """Capture one Comfy device for both the logical model and engine cache key."""
    device = comfy.model_management.get_torch_device()
    return device, int(getattr(device, "index", 0) or 0)


def make_engine_factory(get_engine_fn, bind_pipeline_model):
    """Weakref-backed deferred factory so model<->engine cycles never occur."""
    engine_models = []

    def factory():
        eng, ckey = get_engine_fn()
        for wr in engine_models:
            m = wr()
            if m is not None:
                bind_pipeline_model(ckey, m)
        return eng, ckey

    def register_model(m):
        engine_models.append(weakref.ref(m))

    return factory, register_model


class _AVPackedMeta(NamedTuple):
    nested: bool
    orig_x: object
    orig_video: object
    orig_audio: object
    targets_per_group: int


@qfe.console_safe_methods
class QFSessionModelMixin:
    """Shared session dials, VRAM ledger, and qf_xfm_step* dispatch across all 4 families."""

    _step_cache = 0.0
    _block_cache = 0.0
    _SCRATCH_ATTRS = ("_out", "_out_video", "_out_audio")
    _ENGINE_IGNORED_COND_KEYS = ()
    _TAG = ""
    _REFUSED_HINT = ""

    def _init_session_state(self, engine, *, arm_concat_channels=None):
        self.diffusion_model = _QFStub()
        if arm_concat_channels is not None:
            self.diffusion_model.arm_concat_shape(int(arm_concat_channels))
        self._qf = engine
        self._num_steps = 0
        self._step_i = 0
        self._sess_denoise = 0
        self._ctx_key_assigner = _CtxKeyAssigner()
        self._out = None
        self._qf_conds_done = False
        self._qf_conds_run_id = None
        self._qf_conds_noise_ref = None
        self._reset_run_cond_state()

    def _reset_run_cond_state(self):
        self._max_ctx_seq = 0

    def _begin_extra_conds_guard(self, tag, *, on_close=None, noise=None):
        was_open, ok = self._qf.end_session_if_open()
        self._qf_needs_begin = True
        if on_close is not None:
            on_close(was_open, ok)
        if was_open:
            qfe.info(f"[qf_native] {tag}: closed a pre-existing session at run start "
                     f"(prior run interrupted/uncleaned); end ok={ok}")
        run_id = _QF_RUN_ID.get()
        if run_id is not None:
            fresh = (run_id != getattr(self, "_qf_conds_run_id", None))
            self._qf_conds_run_id = run_id
            if fresh:
                self._qf_conds_noise_ref = None
        elif noise is not None:
            prev_ref = getattr(self, "_qf_conds_noise_ref", None)
            prev = prev_ref() if callable(prev_ref) else None
            fresh = bool(was_open or getattr(self, "_qf_conds_done", False)
                         or (prev_ref is not None and prev is not noise))
            try:
                self._qf_conds_noise_ref = weakref.ref(noise)
            except TypeError:
                self._qf_conds_noise_ref = None
        else:
            fresh = bool(was_open or getattr(self, "_qf_conds_done", False))
        if fresh:
            self._reset_run_cond_state()
        self._qf_conds_done = False
        return fresh

    def _reject_ignored_conds(self, kwargs, tag, hint=""):
        for k in self._ENGINE_IGNORED_COND_KEYS:
            if kwargs.get(k) is not None:
                raise RuntimeError(
                    f"qf_native {tag}: '{k}' conditioning is wired, but the native session "
                    f"cannot consume it - it would be silently ignored, so it is refused. "
                    f"Remove the node feeding '{k}'{hint}.")

    def _ensure_scratch(self, attr, shape, dtype, device):
        buf = getattr(self, attr, None)
        target_shape = tuple(int(d) for d in shape)
        if buf is None or tuple(buf.shape) != target_shape or buf.dtype != dtype or buf.device != device:
            buf = torch.empty(target_shape, dtype=dtype, device=device)
            setattr(self, attr, buf)
        return buf

    def _release_scratch_buffers(self):
        d = vars(self)
        for attr in self._SCRATCH_ATTRS:
            if attr in d:
                setattr(self, attr, None)

    def _open_native_session(self, tag, max_context_dims, cond_dtype, width, height, opts,
                             *, allow_busy_retry=True, before_begin=None, after_begin=None,
                             skip_end_if_open=False):
        lib = self._qf.lib
        if not skip_end_if_open:
            self._qf.end_session_if_open()
        self._ctx_key_assigner.reset()
        bpx = qfe.QFSessionBeginParams()
        ctypes.memset(ctypes.byref(bpx), 0, ctypes.sizeof(bpx))
        bpx.struct_size = ctypes.sizeof(bpx)
        bpx.width = int(width)
        bpx.height = int(height)
        bpx.num_steps = self._num_steps
        bpx.max_context_dims = (ctypes.c_int32 * 3)(*(int(v) for v in max_context_dims))
        self._max_ctx_seq = 0
        bpx.cond_dtype = cond_dtype
        bpx._opts = json.dumps(opts).encode("utf-8")
        bpx.options_json = bpx._opts
        session = ctypes.c_void_p()
        self._begin_keep = bpx
        if before_begin is not None:
            before_begin()
        _torch = globals().get("torch")
        if _torch is not None and _torch.cuda.is_available():
            _dev = getattr(self, "device", None)
            if _dev is None or getattr(_dev, "type", None) != "cuda":
                _did = getattr(getattr(self, "_qf", None), "device_id", None)
                _dev = _did if isinstance(_did, int) and not isinstance(_did, bool) and _did >= 0 else None
            _cs = _torch.cuda.current_stream(_dev) if _dev is not None else _torch.cuda.current_stream()
            if getattr(_cs, "cuda_stream", 0) != 0:
                _cs.synchronize()
        st = lib.qf_xfm_session_begin(self._qf.xfm, ctypes.byref(bpx), ctypes.byref(session))
        if allow_busy_retry and st != qfe.QF_OK and (
                st == qfe.QF_ERR_BUSY or "busy" in (qfe.last_err(lib) or "").lower()):
            time.sleep(_BUSY_BEGIN_RETRY_S)
            self._qf.end_session_if_open()
            st = lib.qf_xfm_session_begin(self._qf.xfm, ctypes.byref(bpx), ctypes.byref(session))
        if st != qfe.QF_OK:
            raise RuntimeError(f"qf_xfm_session_begin ({tag}) failed: {qfe.last_err(lib)}")
        self._qf.current_session = session
        if after_begin is not None:
            after_begin()
        self._step_i = 0
        self._sess_denoise = 0
        self._qf_conds_done = True
        return session

    def _close_session_on_latent_out(self, tag, step_fn_name):
        try:
            if self._qf.current_session is not None:
                try:
                    qfe.info(f"[qf_native] {tag} SESSION CLOSED after {self._step_i} sampler steps, "
                             f"{self._sess_denoise} {step_fn_name} calls", flush=True)
                finally:
                    self._qf.end_session_if_open()
        finally:
            self._release_scratch_buffers()
            self._reset_run_cond_state()
            self._qf_conds_done = True

    def set_step_cache(self, t):
        self._step_cache = float(t)

    def set_block_cache(self, t):
        self._block_cache = float(t)

    def set_attn_backend(self, v):
        self._attn_backend = str(v or "auto")

    def set_sol_tau(self, v):
        self._sol_options = None
        try:
            self._sol_tau = float(v)
        except (TypeError, ValueError):
            self._sol_tau = 1.0

    def set_sol_options(self, options):
        if not isinstance(options, qf_sol.Options):
            raise TypeError("Sol options must be normalized before applying them")
        self._sol_options = options

    def set_video_enhance(self, on):
        self._video_enhance = bool(on)

    _SESSION_DIALS = ("_step_cache", "_block_cache", "_attn_backend", "_sol_tau", "_sol_options", "_video_enhance")

    def _effective_cat_ratio(self, quality_enhance):
        """QualityLaw: True -> 1.0; False -> 0.80 default (QI21 overrides with edit/CFG/batch guard)."""
        return 1.0 if quality_enhance else _CAT_KEEP_DEFAULT

    def _stage_schedule(self, transformer_options, tag):
        sigmas = transformer_options.get("sample_sigmas") if isinstance(transformer_options, dict) else None
        if sigmas is None or len(sigmas) < 2:
            raise RuntimeError(
                f"qf_native {tag}: the sampler did not publish a sigma schedule "
                "(transformer_options['sample_sigmas']) - the engine session needs the step count. "
                "Use a stock KSampler / SamplerCustom on this model.")
        self._num_steps = len(sigmas) - 1
        try:
            first, last = float(sigmas[0]), float(sigmas[-1])
        except Exception:  # noqa: BLE001
            return None
        if first <= last:
            raise RuntimeError(
                f"qf_native {tag}: sigma schedule must be strictly DECREASING (got "
                f"{first:.4f}->{last:.4f}) - a non-decreasing schedule would drive the session backwards.")
        return first, last

    def adopt_session_dials_from(self, src):
        for name in type(self)._SESSION_DIALS:
            if name in vars(src):
                setattr(self, name, vars(src)[name])

    def residency_opts(self):
        """Session options fragment for video models (optional cache thresholds + common dials)."""
        o = {}
        te = float(getattr(self, "_step_cache", 0.0) or 0.0)
        tf = float(getattr(self, "_block_cache", 0.0) or 0.0)
        if te > 0.0:
            o["step_cache_thresh"] = te
        if tf > 0.0:
            o["block_cache_thresh"] = tf
        o.update(self.dial_opts())
        return o

    def dial_opts(self):
        """Common session options emitted for every family: attention_backend, quality ratio, optional sparse."""
        on = getattr(self, "_video_enhance", None)
        if on is None:
            raise RuntimeError(
                "QuantFunc: this model has no quality_enhance setting (the loader sets one on every model it "
                "builds; set_video_enhance was never called)")
        out = {
            "attention_backend": str(getattr(self, "_attn_backend", "auto") or "auto"),
            "cat_keep_ratio": float(self._effective_cat_ratio(on)),
        }
        options = getattr(self, "_sol_options", None)
        if options is None:
            options = qf_sol.from_loader(sol_tau=getattr(self, "_sol_tau", None))
        sparse = options.native(getattr(self, "model_sampling", None))
        if sparse is not None:
            out["sparse"] = sparse
        return out

    _QF_COMFY_SIDE_BASE_BYTES = 64 * 1024 * 1024
    _QF_COMFY_SIDE_LATENT_COPIES = 8
    _QF_COMFY_SIDE_COND_COPIES = 2
    _QF_COMFY_SIDE_ITEMSIZE = 2

    def _qf_comfy_side_bytes(self, input_shape, cond_shapes=None):
        n_lat = 1
        for d in list(input_shape):
            n_lat *= max(1, int(d))
        n_cond = 0
        for shapes in (cond_shapes or {}).values():
            for sh in (shapes or []):
                try:
                    n = 1
                    for d in list(sh):
                        n *= max(1, int(d))
                    n_cond += n
                except (TypeError, ValueError):
                    continue
        return int(self._QF_COMFY_SIDE_BASE_BYTES
                   + self._QF_COMFY_SIDE_ITEMSIZE * (n_lat * self._QF_COMFY_SIDE_LATENT_COPIES
                                                     + n_cond * self._QF_COMFY_SIDE_COND_COPIES))

    def _qf_engine_latent_dims(self, input_shape, *, strict=True):
        dims = [int(d) for d in input_shape]
        if len(dims) != 3 or dims[1] != 1:
            return dims
        request = _QF_SAMPLING_GEOMETRY.get()
        ls = getattr(self, "latent_shapes", None)
        if request is not None and request[0] is self:
            ls = request[1]
        if ls:
            try:
                streams = [[int(d) for d in s] for s in ls]
                if streams and sum(math.prod(s[1:]) for s in streams) == dims[2]:
                    return [dims[0]] + streams[0][1:]
            except (TypeError, ValueError):
                pass
        if strict:
            raise RuntimeError(
                "QuantFunc packed AV demand requires current latent_shapes; "
                "invoke through ComfyUI's OUTER_SAMPLE hook or supply request geometry")
        return dims

    def memory_required(self, input_shape, cond_shapes=None, **_kw):
        """Inference VRAM reserve: comfy-side copies + incremental engine working set (never triggers create)."""
        comfy_side = self._qf_comfy_side_bytes(input_shape, cond_shapes)
        lazy = eng = getattr(self, "_qf", None)
        need = 0
        cold_known = False
        cold_note = None
        if eng is not None:
            peek = getattr(eng, "ensure_if_cached", None)
            if callable(peek):
                eng = peek()
            if eng is not None:
                need = int(eng.vram_need_bytes(self._qf_engine_latent_dims(input_shape)))
            else:
                cold = lazy.cold_vram_need_bytes()
                cold_note = cold.note
                if cold.bytes is not None:
                    # QFModelPatcher.model_size() already reports resident_weight_bytes to ComfyUI's
                    # LoadedModel (which adds model_size() - loaded_size()); report only the incremental
                    # working-set floor here so cold eviction targets ~1x instead of ~2.1x.
                    cold_known = True
                    need = int(cold.working_set_floor_bytes or 0)
        floor = int(super().memory_required(
            self._qf_engine_latent_dims(input_shape, strict=False),
            cond_shapes=cond_shapes or {})) if (eng is None and not cold_known) else 0
        total = int(max(floor, comfy_side + need))
        sig = (tuple(int(d) for d in input_shape), comfy_side, need, floor, eng is None, cold_known)
        if sig != getattr(self, "_qf_ledger_last", None):
            self._qf_ledger_last = sig
            try:
                hold = f"{int(eng.resident_vram_bytes()) >> 20} MB" if eng is not None else "0 MB"
            except Exception as error:  # noqa: BLE001
                hold = f"unknown ({type(error).__name__}: {error})"
            qfe.info(
                "[qf_native] VRAM ledger: memory_required%s = %d MB (comfy-side %d MB, engine need %d MB%s); "
                "engine hold %s"
                % (list(sig[0]), total >> 20, comfy_side >> 20, need >> 20,
                   (" (cold: incremental working set; resident weights reported via model_size)"
                    if cold_known else
                    " (cold: no pipeline yet; the engine's need is %s: comfy's own estimate %d MB is the floor)"
                    % (cold_note, floor >> 20) if cold_note else
                    " (cold: no pipeline yet; comfy's own estimate %d MB is the floor)" % (floor >> 20))
                   if eng is None else
                   ("" if need else " (0: covered by what it holds, or nothing measured yet)"),
                   hold), flush=True)
        return total

    def _sigma_step_index(self, sigma, sig_all, transformer_options):
        step_index = self._step_i
        sched = transformer_options.get("sample_sigmas") if isinstance(transformer_options, dict) else None
        if sched is not None and len(sched) >= 2:
            cur = float(sig_all[0].item()) if sig_all is not None else float(sigma)
            step_index = min(range(len(sched) - 1), key=lambda k: abs(float(sched[k]) - cur))
        return step_index

    def _call_denoise_step(self, p, fail_prefix, fn_name="qf_xfm_step"):
        _torch = globals().get("torch")
        if _torch is not None and _torch.cuda.is_available():
            _dev = getattr(self, "device", None)
            if _dev is None or getattr(_dev, "type", None) != "cuda":
                _did = getattr(getattr(self, "_qf", None), "device_id", None)
                _dev = _did if isinstance(_did, int) and not isinstance(_did, bool) and _did >= 0 else None
            _cs = _torch.cuda.current_stream(_dev) if _dev is not None else _torch.cuda.current_stream()
            if getattr(_cs, "cuda_stream", 0) != 0:
                _cs.synchronize()
        try:
            st = getattr(self._qf.lib, fn_name)(self._qf.current_session, ctypes.byref(p))
        except Exception:
            self._qf.end_session_if_open()
            raise
        if st != qfe.QF_OK:
            err = qfe.last_err(self._qf.lib)
            self._qf.end_session_if_open()
            raise RuntimeError(f"{fail_prefix} failed: {err}")

    def _call_denoise_step_av(self, mp, fail_prefix):
        _torch = globals().get("torch")
        if _torch is not None and _torch.cuda.is_available():
            _dev = getattr(self, "device", None)
            if _dev is None or getattr(_dev, "type", None) != "cuda":
                _did = getattr(getattr(self, "_qf", None), "device_id", None)
                _dev = _did if isinstance(_did, int) and not isinstance(_did, bool) and _did >= 0 else None
            _cs = _torch.cuda.current_stream(_dev) if _dev is not None else _torch.cuda.current_stream()
            if getattr(_cs, "cuda_stream", 0) != 0:
                _cs.synchronize()
        lib = self._qf.lib
        try:
            st = lib.qf_xfm_step_av(self._qf.current_session, ctypes.byref(mp))
        except Exception:
            self._qf.end_session_if_open()
            raise
        if st != qfe.QF_OK:
            err = qfe.last_err(lib)
            self._qf.end_session_if_open()
            raise RuntimeError(f"{fail_prefix} failed: {err}")


@qfe.console_safe_methods
class QFImageSessionModel(QFSessionModelMixin):
    """Shared image-family model wrapper (Krea-2, Qwen-Image-2.1)."""

    _TAG = ""
    _NO_COND_HINT = ""
    _REFUSED_HINT = ""
    _ENGINE_IGNORED_COND_KEYS = ()

    def __init__(self, model_config, engine, device=None):
        super().__init__(model_config, device=device)
        self._init_session_state(engine, arm_concat_channels=int(self.latent_format.latent_channels))

    def _check_conds(self, kwargs):
        """Per-family run-start conditioning validation (default: no-op)."""

    def _prepare_begin_context(self, xin, ctx, transformer_options, kwargs):
        """Hook called immediately before _begin() so subclasses can inspect batch/CFG/refs."""

    def extra_conds(self, **kwargs):
        self._begin_extra_conds_guard(self._TAG, noise=kwargs.get("noise"))
        try:
            self._reject_ignored_conds(kwargs, self._TAG, self._REFUSED_HINT)
            self._check_conds(kwargs)
            out = super().extra_conds(**kwargs)
            cross_attn = kwargs.get("cross_attn", None)
            if cross_attn is not None:
                self._max_ctx_seq = max(self._max_ctx_seq, int(cross_attn.shape[1]))
            return out
        except BaseException:
            self._reset_run_cond_state()
            self._qf_conds_done = True
            raise

    def _begin(self, x_group, ctx_group):
        scale = int(self.latent_format.spacial_downscale_ratio)
        max_seq = max(self._max_ctx_seq, int(ctx_group.shape[1]))
        session = self._open_native_session(
            self._TAG,
            (int(ctx_group.shape[0]), max_seq, int(ctx_group.shape[2])),
            _qf_dtype(ctx_group.dtype),
            int(x_group.shape[-1]) * scale,
            int(x_group.shape[-2]) * scale,
            self.dial_opts(),
        )
        qfe.info(f"[qf_native] {self._TAG.upper()} SESSION OPEN handle={session.value:#x} steps={self._num_steps} "
                 f"latent={tuple(x_group.shape)} cond={tuple(ctx_group.shape)}")

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        self._qf_conds_done = True
        sigma = t
        ctx = c_crossattn
        if ctx is None:
            raise RuntimeError(f"qf_native {self._TAG}: no c_crossattn cond - {self._NO_COND_HINT}")
        if control is not None:
            raise RuntimeError(
                f"qf_native {self._TAG}: a ControlNet is wired, but the native session consumes no "
                "control input - it would be silently ignored, so it is refused.")
        if c_concat is not None:
            raise RuntimeError(
                f"qf_native {self._TAG}: c_concat conditioning is not part of the {self._TAG} seam - remove "
                "the node feeding it.")
        try:
            dev = x.device
            x_bf = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
            if ctx.dtype != torch.bfloat16 or ctx.device != dev:
                ctx = ctx.to(device=dev, dtype=torch.bfloat16)
            sig_all = sigma.reshape(-1) if torch.is_tensor(sigma) else None
            xin = x_bf
            batch = int(xin.shape[0])
            if getattr(self, "_qf_needs_begin", True) or self._qf.current_session is None:
                self._stage_schedule(transformer_options, self._TAG)
                self._prepare_begin_context(xin, ctx, transformer_options, kwargs)
                self._begin(xin[0:1], ctx[0:1])
                self._qf_needs_begin = False
            self._out = self._ensure_scratch("_out", xin.shape, xin.dtype, xin.device)
            cuuids = transformer_options.get("uuids") if isinstance(transformer_options, dict) else None
            step_index = self._sigma_step_index(sigma, sig_all, transformer_options)
            for i in range(batch):
                _interrupt_poll_end_session_on_raise(self._qf)
                xi = xin[i:i + 1].contiguous()
                oi = self._out[i:i + 1]
                if xi.dim() == 5 and xi.shape[2] == 1:
                    xi = xi.squeeze(2).contiguous()
                    oi = oi.squeeze(2)
                ci = ctx[i:i + 1].contiguous()
                cuid = cuuids[i] if (cuuids is not None and i < len(cuuids)) else None
                ctx_key = self._ctx_key_assigner.key(cuid)
                sig_i = (float(sig_all[i].item()) if (sig_all is not None and sig_all.numel() >= batch)
                         else (float(sig_all[0].item()) if sig_all is not None else float(sigma)))
                p = qfe.QFStepParams()
                ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
                p.struct_size = ctypes.sizeof(p)
                p.latent_in = xi.data_ptr()
                p.velocity_out = oi.data_ptr()
                p.velocity_out_capacity = oi.numel() * oi.element_size()
                p.dims[:xi.dim()] = list(xi.shape)
                p.dtype = _qf_dtype(xi.dtype)
                p.sigma = sig_i
                p.step_index = step_index
                p.total_steps = self._num_steps
                p.context = ci.data_ptr()
                p.context_dims = (ctypes.c_int32 * 3)(*ci.shape)
                p.context_dtype = _qf_dtype(ci.dtype)
                p.cfg_context_key = ctx_key
                self._denoise_group(p, i, xi, ci, kwargs, step_index, ctx_key)
                self._qf.step_count += 1
                self._sess_denoise += 1
            self._step_i += 1
            self._qf.sampler_step_count += 1
            return self.model_sampling.calculate_denoised(sigma, self._out.float(), x)
        except BaseException:
            self._qf.end_session_if_open()
            self._qf_needs_begin = True
            raise

    def _denoise_group(self, p, i, xi, ci, kwargs, step_index, ctx_key):
        self._call_denoise_step(p, f"{self._TAG} qf_xfm_step[step={step_index},group={i},key={ctx_key}]")

    def process_latent_out(self, latent):
        self._close_session_on_latent_out(self._TAG.upper(), "qf_xfm_step")
        return super().process_latent_out(latent)


@qfe.console_safe_methods
class QFAVSessionModel(QFSessionModelMixin):
    """Shared joint audio+video session template (MiniMax-H3, LTX-2.5 AV)."""

    def _unpack_av_latent(self, x):
        nested = getattr(x, "is_nested", False)
        if nested:
            streams = x.unbind()
            return True, streams[0], streams[1]
        ls = getattr(self, "latent_shapes", None)
        if not ls or len(ls) < 2:
            raise RuntimeError(
                f"qf_native {self._TAG}: packed latent (shape={tuple(getattr(x, 'shape', ()))}) "
                f"but model.latent_shapes is unset.")
        n = int(math.prod(ls[0][1:]))
        xf = x.reshape(int(x.shape[0]), -1)
        video_shape, audio_shape = list(ls[0]), list(ls[1])
        video_shape[0] = audio_shape[0] = int(x.shape[0])
        return False, xf[:, :n].reshape(video_shape), xf[:, n:].reshape(audio_shape)

    def _execute_av_group_step(self, *, vid_in, vid_dims, vid_out, aud_in, aud_dims, aud_out,
                               vi, sig_i, step_index, ctx_key, audio_scale, ref_active, group_i):
        _interrupt_poll_end_session_on_raise(self._qf)
        p = qfe.QFStepParams()
        ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
        p.struct_size = ctypes.sizeof(p)
        p.latent_in = vid_in.data_ptr()
        p.velocity_out = vid_out.data_ptr()
        p.velocity_out_capacity = vid_out.numel() * vid_out.element_size()
        p.dims[:len(vid_dims)] = list(vid_dims)
        p.dtype = _qf_dtype(vid_in.dtype)
        p.sigma = sig_i
        p.step_index = step_index
        p.total_steps = self._num_steps
        p.context = vi.data_ptr()
        p.context_dims = (ctypes.c_int32 * 3)(*vi.shape)
        p.context_dtype = _qf_dtype(vi.dtype)
        p.cfg_context_key = ctx_key
        mp = qfe.QFStepAvParams()
        ctypes.memset(ctypes.byref(mp), 0, ctypes.sizeof(mp))
        mp.struct_size = ctypes.sizeof(mp)
        mp.base = p
        mp.audio_latent_in = aud_in.data_ptr()
        mp.audio_velocity_out = aud_out.data_ptr()
        mp.audio_velocity_out_capacity = aud_out.numel() * aud_out.element_size()
        mp.audio_dims[:len(aud_dims)] = list(aud_dims)
        mp.audio_dtype = _qf_dtype(aud_in.dtype)
        mp.audio_scale = float(audio_scale)
        mp.audio_reference_active = 1 if ref_active else 0
        self._call_denoise_step_av(mp, f"{self._TAG} qf_xfm_step_av[step={step_index},group={group_i}]")
        self._qf.step_count += 1
        self._sess_denoise += 1

    def _av_unpack_and_validate(self, x, c_crossattn, transformer_options):
        raise NotImplementedError

    def _av_prepare_session(self, x_vid, x_audio, vemb, transformer_options, kwargs, opening):
        raise NotImplementedError

    def _av_step_tensors(self, x_vid, x_audio, vemb, meta, sigma, sig_all, transformer_options, kwargs, step_index):
        raise NotImplementedError

    def _av_repack_output(self, out_vid, out_audio, meta, sigma):
        if meta.nested:
            den_v = self.model_sampling.calculate_denoised(sigma, out_vid.float(), meta.orig_video)
            den_a = self.model_sampling.calculate_denoised(sigma, out_audio.float(), meta.orig_audio)
            return comfy.nested_tensor.NestedTensor((den_v, den_a))
        batch = int(out_vid.shape[0])
        vel_flat = torch.cat([out_vid.reshape(batch, -1), out_audio.reshape(batch, -1)], dim=1).reshape(
            meta.orig_x.shape)
        return self.model_sampling.calculate_denoised(sigma, vel_flat.float(), meta.orig_x)

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        self._qf_conds_done = True
        sigma = t
        if c_crossattn is None:
            raise RuntimeError(f"qf_native {self._TAG}: no c_crossattn cond - {self._NO_COND_HINT}")
        if control is not None:
            raise RuntimeError(
                f"qf_native {self._TAG}: a ControlNet is wired, but this seam does not consume control hints.")
        try:
            x_vid, x_audio, vemb, meta = self._av_unpack_and_validate(x, c_crossattn, transformer_options)
            opening = self._qf.current_session is None or getattr(self, "_qf_needs_begin", False)
            self._av_prepare_session(x_vid, x_audio, vemb, transformer_options, kwargs, opening)
            sig_all = sigma.reshape(-1) if torch.is_tensor(sigma) else None
            step_index = self._sigma_step_index(sigma, sig_all, transformer_options)
            out_vid, out_audio = self._av_step_tensors(
                x_vid, x_audio, vemb, meta, sigma, sig_all, transformer_options, kwargs, step_index)
            self._step_i += 1
            self._qf.sampler_step_count += 1
            return self._av_repack_output(out_vid, out_audio, meta, sigma)
        except BaseException:
            self._qf.end_session_if_open()
            self._qf_needs_begin = True
            raise

    def process_latent_out(self, latent):
        self._close_session_on_latent_out(self._TAG, "qf_xfm_step_av")
        return super().process_latent_out(latent)


class QFLazyEngine:
    """Deferred QFEngineHandle proxy: materializes on first real use and applies API key + LoRAs in place."""

    def __init__(self, factory, api_key=None, read_auth=None, peek_cached=None, estimate_cold=None,
                 vram_budget_bytes=0):
        self._factory = factory
        self._peek_cached = peek_cached
        self._estimate_cold = estimate_cold
        self._real = None
        self._ckey = None
        self._lora = []
        self._api_key = api_key
        self._read_auth = read_auth
        self.capacity_bytes = None
        self.host_weight_bytes = None
        self.vram_budget_bytes = int(vram_budget_bytes or 0)
        self.step_count = 0
        self.sampler_step_count = 0

    def _adopt_real(self, real, ckey):
        if not isinstance(real, qfe.QFEngineHandle) or real.xfm is None:
            raise RuntimeError("QuantFunc factory did not materialize a live transformer handle")
        self._real, self._ckey = real, ckey
        if real.capacity_bytes > 0:
            self.capacity_bytes = real.capacity_bytes
        hwb = getattr(real, "host_weight_bytes", None)
        if hwb is not None and hwb > 0:
            self.host_weight_bytes = hwb
        self._real.step_count = self.step_count
        self._real.sampler_step_count = self.sampler_step_count
        return self._real

    def _drop_dead_real(self):
        if self._real is not None and self._real.xfm is None:
            self._real = None
            self.capacity_bytes = None
            self.host_weight_bytes = None

    def ensure(self):
        self._drop_dead_real()
        if self._real is None:
            with _engine_cache_acquisition(self):
                real, ckey = self._factory()
                self._adopt_real(real, ckey)
        self._apply_runtime_api_key()
        self._apply_runtime_lora()
        return self._real

    def _apply_runtime_api_key(self):
        real, want = self._real, self._api_key
        if getattr(real, "applied_api_key", None) == want:
            return False
        key = want if want is not None else self._read_auth()[0]
        if not key:
            raise RuntimeError(
                "qf_native: this model is signed in with the key from a loader's API key field. This loader's "
                "field is empty and there is no key in config.json or QUANTFUNC_API_KEY to switch to: enter a "
                "key in this loader's API key field, or restore the key in config.json.")
        real.set_api_key(key)
        real.applied_api_key = want
        qfe.info("[qf_native] API key switched in place (no reload): now "
                 + ("this loader's field key" if want is not None else "the key of QUANTFUNC_API_KEY / config.json"),
                 flush=True)
        return True

    def _apply_runtime_lora(self):
        real, want = self._real, self._lora_sig()
        if getattr(real, "applied_lora_sig", "[]") == want:
            return False
        if getattr(real, "current_session", None):
            real.end_session_if_open()
            if getattr(real, "current_session", None):
                time.sleep(_LORA_DRAIN_RETRY_S)
                real.end_session_if_open()
        if getattr(real, "current_session", None):
            raise RuntimeError(
                "qf_native: the LoRA set changed while this model's generation is still running. The engine "
                "applies a LoRA change only between generations; re-queue the prompt.")
        lora = self.lora_set()
        real.applied_lora_sig = None
        try:
            real.set_loras(lora)
        except RuntimeError as e:
            if any(entry.get("path") and qfe.console_safe(entry["path"]) in str(e) for entry in lora):
                warn_lora_not_loaded()
            raise
        real.applied_lora_sig = want
        qfe.info(f"[qf_native] LoRA set applied in place (no reload): {len(lora)} LoRA(s)", flush=True)
        return True

    @property
    def materialized(self):
        return self._real is not None and self._real.xfm is not None

    def ensure_if_cached(self):
        """Return an already-cached QFEngineHandle without ever creating one."""
        if self._real is not None and self._real.xfm is not None:
            return self._real
        self._drop_dead_real()
        if callable(self._peek_cached):
            hit = self._peek_cached()
            if hit is not None:
                eng, ckey = hit
                if isinstance(eng, qfe.QFEngineHandle) and eng.xfm is not None:
                    self._adopt_real(eng, ckey)
                    return self._real
        return None

    def lora_set(self):
        return [dict(e) for e in self._lora]

    def _lora_sig(self):
        return json.dumps(self.lora_set(), sort_keys=True)

    @property
    def lib(self):
        return self.ensure().lib

    @property
    def xfm(self):
        return None if self._real is None else self._real.xfm

    @property
    def current_session(self):
        return None if self._real is None else self._real.current_session

    @current_session.setter
    def current_session(self, v):
        self.ensure().current_session = v

    def retain_reference_audio(self, owner):
        if owner is None:
            if self._real is not None:
                self._real.retain_reference_audio(None)
        else:
            self.ensure().retain_reference_audio(owner)

    def unload_sync(self):
        assert self._real is not None
        return self._real.unload_sync()

    def unload_vram(self, bytes_requested=qfe.QF_UINT64_MAX):
        entry = self._real if self.materialized else self.ensure_if_cached()
        return 0 if entry is None else entry.unload_vram(bytes_requested)

    def reference_audio_pending(self):
        return self._real is not None and self._real.reference_audio_pending()

    def reference_audio_copy_completed(self):
        assert self._real is not None
        self._real.reference_audio_copy_completed()

    def end_session_if_open(self):
        entry = self._real
        if entry is None or getattr(entry, "xfm", None) is None:
            fn = getattr(self, "ensure_if_cached", None)
            entry = fn() if callable(fn) else None
        return (False, True) if entry is None else entry.end_session_if_open()

    def resident_vram_bytes(self):
        entry = self._real if self.materialized else self.ensure_if_cached()
        return 0 if entry is None else entry.resident_vram_bytes()

    def vram_need_bytes(self, latent_shape):
        entry = self._real if self.materialized else self.ensure_if_cached()
        return 0 if entry is None else entry.vram_need_bytes(latent_shape)

    def cold_vram_need_bytes(self):
        if callable(self._estimate_cold):
            return self._estimate_cold()
        return qfe.ColdNeed(None, None)


def ensure_model_config_attrs(model_config):
    for attr, default in (("manual_cast_dtype", None), ("custom_operations", None),
                          ("optimizations", {}), ("scaled_fp8", None)):
        if inspect.getattr_static(model_config, attr, _ABSENT) is _ABSENT:
            setattr(model_config, attr, default)
    return model_config


def refuse_all_zero_initial_latent(xin, tag):
    """Fail loud before any engine step if the initial latent is pure zeros."""
    if float(xin.abs().max()) == 0.0:
        raise RuntimeError(
            f"qf_native {tag}: the initial latent is ALL ZEROS - denoising pure zeros "
            f"cannot produce an image (it renders a BLACK video). "
            f"Set add_noise=enable on the first sampler stage.")


def tag_lora_rebuild(patcher, lora_entries, rebuild):
    m = patcher.model
    if not isinstance(m._qf, QFLazyEngine):
        raise TypeError(f"tag_lora_rebuild: {type(m).__name__}._qf must be a QFLazyEngine, not {type(m._qf).__name__}")
    m._qf._lora = [dict(e) for e in lora_entries]
    setattr(m, QF_LORA_STACK_ATTR, list(lora_entries))
    setattr(m, QF_LORA_REBUILD_ATTR, rebuild)
    return patcher


def family_build(deps, weights_path, create_extra, supported_model, unet_config, make_model, label,
                 pinned_memory=False):
    """Shared builder for all 4 families: single transformer handle per weight recipe, unified in-place LoRA."""
    get_engine = deps["get_engine"]
    bind_pipeline_model = deps["bind_pipeline_model"]
    read_auth = deps["read_auth"]
    peek_engine = deps.get("peek_engine")
    estimate_engine = deps.get("estimate_engine")
    loader_sig = LOADER_SIG.get()
    api_key = LOADER_API_KEY.get()

    def build(lora_entries):
        create_cfg = dict(create_extra or {}, **({"pinned_memory": True} if pinned_memory else {})) or None
        device, device_id = current_torch_device()
        factory, register_model = make_engine_factory(
            lambda: get_engine(weights_path, create_cfg=create_cfg, device_id=device_id, api_key=api_key),
            bind_pipeline_model)
        peek_fn = ((lambda: peek_engine(weights_path, create_cfg=create_cfg, device_id=device_id))
                   if callable(peek_engine) else None)
        est_fn = ((lambda: estimate_engine(weights_path, create_cfg=create_cfg, device_id=device_id, api_key=api_key))
                  if callable(estimate_engine) else None)
        model_config = supported_model(dict(unet_config))
        ensure_model_config_attrs(model_config)
        budget_mb = int((create_cfg or {}).get("vram_budget_mb") or 0)
        engine = QFLazyEngine(factory, api_key=api_key, read_auth=read_auth,
                              peek_cached=peek_fn, estimate_cold=est_fn,
                              vram_budget_bytes=max(0, budget_mb) << 20)
        model = make_model(model_config, engine, device)
        model._qf_loader_sig = loader_sig
        register_model(model)
        patcher = QFModelPatcher(model, load_device=device, offload_device=comfy.model_management.unet_offload_device())
        qfe.info(label)
        return tag_lora_rebuild(patcher, lora_entries, build)

    return build


def lora_stack_of(patcher):
    return list(getattr(getattr(patcher, "model", None), QF_LORA_STACK_ATTR, []) or [])


def rebuild_of(patcher):
    return getattr(getattr(patcher, "model", None), QF_LORA_REBUILD_ATTR, None)


def warn_lora_not_loaded(stock_loader=False):
    if stock_loader:
        why = (f"ComfyUI's Load LoRA / LoraLoaderModelOnly nodes cannot apply a LoRA to it, "
               f"use the \"{NATIVE_LORA_NODE}\" node instead")
    else:
        why = (f"the \"{NATIVE_LORA_NODE}\" node cannot use this file (it is not in the format that node takes, "
               f"or it is a LoRA for another model)")
    _log.warning(f"[qf_native] LoRA could not be loaded for this QuantFunc model: {why}. Please read {LORA_HELP_URL}")


def _in_comfy_lora_load(frame):
    while frame is not None:
        if frame.f_code.co_name == "load_lora_for_models" and frame.f_globals.get("__name__") == "comfy.sd":
            return True
        frame = frame.f_back
    return False


@qfe.console_safe_methods
class QFModelPatcher(comfy.model_patcher.ModelPatcher):
    """ComfyUI ModelPatcher with direct PC Engine VRAM coexistence (no ABI-2 resource broker)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep self.size strictly for Torch module bytes so clone() never compounds native VRAM.
        self.size = 0
        self.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                  "quantfunc.request_geometry", QFModelPatcher._sampling_geometry)

    @staticmethod
    def _sampling_geometry(executor, *args, **kwargs):
        model = executor.class_obj.model_patcher.model
        shapes = kwargs.get("latent_shapes", args[_OUTER_SAMPLE_LATENT_SHAPES_ARG]
                            if len(args) > _OUTER_SAMPLE_LATENT_SHAPES_ARG else None)
        token = _QF_SAMPLING_GEOMETRY.set((model, shapes))
        run_token = _QF_RUN_ID.set(next(_QF_RUN_COUNTER))
        try:
            restore = getattr(model, "_qf_restore_frozen_audio", None)
            if restore is not None and "denoise_mask_function" in executor.class_obj.model_options:
                restore = None
            output = executor(*args, **kwargs)
            if restore is not None and "denoise_mask_function" not in executor.class_obj.model_options:
                return restore(output, *args, **kwargs)
            return output
        except BaseException:
            eng = getattr(model, "_qf", None)
            if eng is not None and callable(getattr(eng, "end_session_if_open", None)):
                try:
                    was_open, _ = eng.end_session_if_open()
                    if was_open and hasattr(model, "_qf_needs_begin"):
                        model._qf_needs_begin = True
                except Exception as exc:  # noqa: BLE001
                    _log.warning("[qf_native] session close after sampling abort returned error: %s", exc)
            raise
        finally:
            _QF_RUN_ID.reset(run_token)
            _QF_SAMPLING_GEOMETRY.reset(token)

    def _engine(self):
        return getattr(getattr(self, "model", None), "_qf", None)

    def _torch_model_size(self):
        return super().model_size()

    def clone(self, *args, **kwargs):
        cloned = super().clone(*args, **kwargs)
        if isinstance(cloned, QFModelPatcher):
            cloned.size = 0
        return cloned

    def add_patches(self, patches, *args, **kwargs):
        strength = args[0] if args else kwargs.get("strength_patch", 1.0)
        if strength and _in_comfy_lora_load(getattr(inspect.currentframe(), "f_back", None)):
            warn_lora_not_loaded(stock_loader=True)
        return super().add_patches(patches, *args, **kwargs)

    def is_dynamic(self):
        return False

    def loaded_size(self):
        base = super().loaded_size()
        eng = self._engine()
        if eng is None:
            return base
        return base + int(eng.resident_vram_bytes())

    def model_size(self):
        base = self._torch_model_size()
        eng = self._engine()
        if eng is None:
            return base
        resident = int(eng.resident_vram_bytes())
        cap = int(getattr(eng, "capacity_bytes", 0) or 0)
        if resident or cap:
            return base + max(resident, cap)
        cold = eng.cold_vram_need_bytes() if callable(getattr(eng, "cold_vram_need_bytes", None)) else None
        if cold is None:
            return base
        total_vram = None
        if getattr(getattr(self, "load_device", None), "type", None) == "cuda":
            try:
                total_vram = int(comfy.model_management.get_total_memory(self.load_device))
            except Exception:  # noqa: BLE001
                total_vram = None
        budget = int(getattr(eng, "vram_budget_bytes", 0) or 0)
        eff_fn = getattr(cold, "effective_model_bytes", None)
        if callable(eff_fn):
            return base + int(eff_fn(total_vram, vram_budget_bytes=budget))
        return base + int(getattr(cold, "bytes", 0) or 0)

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        if device_to != self.load_device:
            raise ValueError("a QuantFunc MODEL cannot migrate its native resources")
        if extra_memory < 0:
            self.partially_unload(self.offload_device, -extra_memory, force_patch_weights=force_patch_weights)
            return 0
        eng = self._engine()
        before_native = int(eng.resident_vram_bytes()) if eng is not None else 0
        if eng is not None and callable(getattr(eng, "ensure", None)):
            eng.ensure()
        before_torch = int(getattr(self.model, "model_loaded_weight_memory", 0))
        super().partially_load(device_to, extra_memory, force_patch_weights=force_patch_weights)
        # ComfyUI's full_load path copies self.model_size() (which includes native bytes) into
        # model_loaded_weight_memory; clamp to Torch-only size so loaded_size() never double-counts native VRAM.
        torch_size = self._torch_model_size()
        if getattr(self.model, "model_loaded_weight_memory", 0) > torch_size:
            self.model.model_loaded_weight_memory = torch_size
        after_torch = int(getattr(self.model, "model_loaded_weight_memory", 0))
        after_native = int(eng.resident_vram_bytes()) if eng is not None else 0
        return max(0, (after_torch - before_torch) + (after_native - before_native))

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        freed_native = 0
        want = max(0, min(int(memory_to_free), qfe.QF_UINT64_MAX))
        eng = self._engine()
        if want > 0 and eng is not None and callable(getattr(eng, "unload_vram", None)):
            try:
                freed_native = int(eng.unload_vram(want))
            except Exception as exc:  # noqa: BLE001 - busy session / OOM callback must not crash unload
                _log.warning("[qf_native] partial VRAM unload skipped (busy or error): %s", exc)
        return freed_native + super().partially_unload(device_to, memory_to_free,
                                                       force_patch_weights=force_patch_weights)

    def adopt_comfy_state_from(self, src):
        override = (self.model, (self.backup, self.backup_buffers,
                                 self.object_patches_backup, self.pinned))
        self.model.adopt_session_dials_from(src.model)
        return src.clone(model_override=override)

    def detach(self, unpatch_all=True):
        if unpatch_all:
            eng = self._engine()
            if eng is not None and callable(getattr(eng, "end_session_if_open", None)):
                try:
                    was_open, _ = eng.end_session_if_open()
                    if was_open and hasattr(self.model, "_qf_needs_begin"):
                        self.model._qf_needs_begin = True
                except Exception as exc:  # noqa: BLE001
                    _log.warning("[qf_native] session close during detach returned error: %s", exc)
            if eng is not None and callable(getattr(eng, "unload_vram", None)):
                try:
                    eng.unload_vram(qfe.QF_UINT64_MAX)
                except Exception as exc:  # noqa: BLE001 - OOM handler / POST /free has no failure channel
                    _log.warning("[qf_native] full VRAM unload during detach returned error: %s", exc)
            rel = getattr(self.model, "_release_scratch_buffers", None)
            if callable(rel):
                rel()
        if sys is None or sys.is_finalizing():
            return self.model
        return super().detach(unpatch_all=unpatch_all)
