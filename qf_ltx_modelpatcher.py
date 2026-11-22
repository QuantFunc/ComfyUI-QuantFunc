"""qf_ltx_modelpatcher -- LTX-2.5 joint audio+video family seam for QuantFunc PC Engine.

Binds frozen 25-symbol qf_* C-API:
  - session begin: frame_t_scales, audio_frame_t_scales, audio_dims, av_unprocessed_ctx, av_conds
  - step: QFStepAvParams with audio_reference_active (0 or 1) via qf_xfm_step_av
  - OUTER_SAMPLE: _qf_restore_frozen_audio preserves zero-mask frozen audio frames
"""
import ctypes
import json
import math
import os
import struct
import time

import torch
import comfy.conds
import comfy.model_base
import comfy.nested_tensor
import comfy.supported_models

from . import qf_engine as qfe
from . import qf_modelpatcher as qfmp
from .qf_modelpatcher import _qf_dtype, _QFStub, _interrupt_poll_end_session_on_raise, QFSessionModelMixin

FAMILY = "ltx2"
_LTXAV_AUDIO_CH = 8
_LTXAV_AUDIO_MEL = 16
_LTXAV_AUDIO_PACK = _LTXAV_AUDIO_CH * _LTXAV_AUDIO_MEL
_VIDEO_LATENT_NDIM = 5
_AUDIO_LATENT_NDIM = 4
_REF_AUDIO_MAX_NUMEL = (1 << 31) - 1
_ST_HEADER_MAX_BYTES = 1 << 31
_BUSY_BEGIN_RETRY_S = 2.0
_CONN_VID_PREFIX = "model.diffusion_model.video_embeddings_connector."
_CONN_AUD_PREFIX = "model.diffusion_model.audio_embeddings_connector."


def _file_has_prefix(path, prefix):
    try:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if n > _ST_HEADER_MAX_BYTES:
                return False
            hdr = json.loads(fh.read(n))
        return any(k.startswith(prefix) for k in hdr)
    except Exception:  # noqa: BLE001
        return False


def matches(pipeline_class, transformer_class=""):
    return str(pipeline_class) == "LTX2Pipeline"


@qfe.console_safe_methods
class QFLTXModel(QFSessionModelMixin, comfy.model_base.LTXV):
    """LTX base wrapper carrying conditioning safety checks, mask-to-scale extraction, and session begin."""

    _ENGINE_IGNORED_COND_KEYS = (
        "concat_mask", "keyframe_idxs", "guide_attention_entries",
        "noise_concat", "cross_attn_controlnet", "concat_latent_image",
        "generated_keyframes",
    )

    def __init__(self, model_config, engine, device=None):
        super().__init__(model_config, device=device)
        self.diffusion_model = _QFStub()
        self._qf = engine
        self._num_steps = 0
        self._num_frames = 0
        self._fps = None
        self._step_i = 0
        self._sess_denoise = 0
        self._out = None
        self._ctx_key_assigner = qfmp._CtxKeyAssigner()
        self._max_ctx_seq = 0

    def _reference_audio_cond(self, payload):
        if payload is not None:
            raise RuntimeError("qf_native LTX: ref_audio needs an AV checkpoint; video-only is unsupported")
        return None

    def _post_connector_seq(self, raw_s):
        return int(raw_s)

    def _begin_extra_opts(self):
        return {}

    def extra_conds(self, **kwargs):
        was_open, ok = self._qf.end_session_if_open()
        self._qf_needs_begin = True
        if hasattr(self, "_clear_audio_reference_after_close"):
            self._clear_audio_reference_after_close(was_open, ok)
        if was_open:
            qfe.info("[qf_native] LTX: closed a pre-existing session at run start; end ok=%s", ok)
        self._pending_frame_t_scale = None
        self._pending_audio_frame_t_scale = None
        dm = kwargs.get("denoise_mask")
        if dm is not None:
            scales, a_scales = self._frame_scales_from_mask(dm)
            scales = torch.tensor(scales, dtype=torch.bfloat16, device="cpu").float().tolist()
            a_scales = torch.tensor(a_scales, dtype=torch.bfloat16, device="cpu").float().tolist()
            self._pending_frame_t_scale = None if all(v >= 0.9999 for v in scales) else scales
            self._pending_audio_frame_t_scale = (
                None if (not a_scales or all(v >= 0.9999 for v in a_scales)) else a_scales)
        for k in self._ENGINE_IGNORED_COND_KEYS:
            if kwargs.get(k) is not None:
                raise RuntimeError(
                    f"qf_native LTX: '{k}' conditioning is wired, but the QuantFunc LTX native session "
                    f"cannot consume '{k}'. Remove the node feeding it. For image-to-video use "
                    f"LTXVImgToVideoInplace on the LATENT path.")
        ref_cond = self._reference_audio_cond(kwargs.get("ref_audio"))
        out = {"frame_rate": self._native_frame_rate(kwargs)}
        if ref_cond is not None:
            out["ref_audio"] = ref_cond
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
            am = kwargs.get("attention_mask", None)
            if am is not None:
                out["attention_mask"] = comfy.conds.CONDRegular(am)
            self._max_ctx_seq = max(self._max_ctx_seq, self._post_connector_seq(int(cross_attn.shape[1])))
        return out

    def _frame_scales_from_mask(self, dm):
        if dm.ndim in (3, _VIDEO_LATENT_NDIM) and int(dm.shape[0]) > 1:
            scales = [self._frame_scales_from_mask(dm[i:i + 1]) for i in range(int(dm.shape[0]))]
            if any(value != scales[0] for value in scales[1:]):
                raise RuntimeError("qf_native LTX: target batches need the same per-frame/audio-row timestep scales")
            return scales[0]
        if not bool(torch.isfinite(dm).all()):
            raise RuntimeError("qf_native LTX: denoise_mask must be finite")
        m = dm
        audio_rows = []
        if m.ndim == 3:
            ls = getattr(self, "latent_shapes", None)
            if not ls or len(ls) < 1:
                raise RuntimeError("qf_native LTX: flat denoise_mask but model carries no latent_shapes.")
            n_video = int(math.prod(ls[0][1:]))
            flat = m.reshape(-1)
            if int(flat.shape[0]) < n_video:
                raise RuntimeError(
                    f"qf_native LTX: flat denoise_mask has {int(flat.shape[0])} elements "
                    f"but video latent needs {n_video} - geometry mismatch.")
            audio_part = flat[n_video:]
            if audio_part.numel() and float(audio_part.min()) < 0.9999:
                if len(ls) < 2:
                    raise RuntimeError(
                        "qf_native LTX: denoise_mask masks AUDIO lane but model carries no audio latent shape.")
                n_audio = int(math.prod(ls[1][1:]))
                if int(audio_part.shape[0]) != n_audio:
                    raise RuntimeError(
                        f"qf_native LTX: flat denoise_mask audio part has {int(audio_part.shape[0])} elements "
                        f"but audio latent needs {n_audio} - geometry mismatch.")
                a = flat[n_video:n_video + n_audio].reshape(list(ls[1][1:]))
                for row in range(int(a.shape[1])):
                    al = a[:, row]
                    lo, hi = float(al.min()), float(al.max())
                    if hi - lo > 1e-4:
                        raise RuntimeError(
                            "qf_native LTX: denoise_mask varies WITHIN an audio row; per-audio-frame masks only.")
                    audio_rows.append(max(0.0, min(1.0, hi)))
            m = flat[:n_video].reshape(list(ls[0][1:])).movedim(1, 0)
        elif m.ndim == _VIDEO_LATENT_NDIM:
            m = m[0, 0]
        else:
            raise RuntimeError(
                f"qf_native LTX: denoise_mask shape {tuple(dm.shape)} is not [B,C,F,H,W] or packed AV flat.")
        per_frame = []
        for f in range(int(m.shape[0])):
            mf = m[f]
            lo, hi = float(mf.min()), float(mf.max())
            if hi - lo > 1e-4:
                raise RuntimeError(
                    "qf_native LTX: denoise_mask varies WITHIN a latent frame; per-frame Inplace masks only.")
            per_frame.append(max(0.0, min(1.0, hi)))
        return per_frame, audio_rows

    def _derive_geometry(self, xin, transformer_options):
        t_lat = int(xin.shape[2])
        self._num_frames = (t_lat - 1) * int(self.latent_format.temporal_downscale_ratio) + 1
        self._stage_schedule(transformer_options, "LTX")

    def _native_frame_rate(self, kwargs):
        try:
            return comfy.model_base.LTXV.extra_conds(
                self, **({"frame_rate": kwargs["frame_rate"]} if "frame_rate" in kwargs else {}))["frame_rate"]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"qf_native LTX: ComfyUI's LTX model code did not resolve a frame_rate ({type(e).__name__}: {e}). "
                f"Wire LTXVConditioning.") from e

    def _call_fps(self, kwargs, opening):
        fr = kwargs.get("frame_rate")
        if fr is None:
            raise RuntimeError("qf_native LTX: model call carries no frame_rate conditioning.")
        fps = float(fr)
        if opening:
            self._fps = fps
        elif fps != self._fps:
            raise RuntimeError(
                f"qf_native LTX: conditionings carry different frame rates ({self._fps:g} vs {fps:g}).")

    def _begin(self, x_group, vemb_group):
        lib = self._qf.lib
        _, close_ok = self._qf.end_session_if_open()
        ref_enabled = getattr(self, "_audio_reference", None) is not None
        if getattr(self._qf, "reference_audio_pending", lambda: False)():
            raise RuntimeError("qf_native LTX-AV: previous reference begin has no proven copy completion")
        if ref_enabled and not close_ok:
            raise RuntimeError("qf_native LTX-AV: previous reference session close refused; retry after drain")
        self._ctx_key_assigner.reset()
        bpx = qfe.QFSessionBeginParams()
        ctypes.memset(ctypes.byref(bpx), 0, ctypes.sizeof(bpx))
        bpx.struct_size = ctypes.sizeof(bpx)
        scale = int(self.latent_format.spacial_downscale_ratio)
        bpx.width = int(x_group.shape[-1]) * scale
        bpx.height = int(x_group.shape[-2]) * scale
        bpx.num_steps = self._num_steps
        max_seq = max(self._max_ctx_seq, int(vemb_group.shape[1]))
        bpx.max_context_dims = (ctypes.c_int32 * 3)(int(vemb_group.shape[0]), max_seq, int(vemb_group.shape[2]))
        self._max_ctx_seq = 0
        bpx.cond_dtype = _qf_dtype(vemb_group.dtype)
        mask_opt = {}
        if getattr(self, "_pending_frame_t_scale", None):
            mask_opt["frame_t_scales"] = self._pending_frame_t_scale
        if getattr(self, "_pending_audio_frame_t_scale", None):
            mask_opt["audio_frame_t_scales"] = self._pending_audio_frame_t_scale
        bpx._opts = json.dumps({
            "num_frames": self._num_frames,
            "fps": float(self._fps),
            **self.residency_opts(),
            **mask_opt,
            **self._begin_extra_opts(),
        }).encode("utf-8")
        bpx.options_json = bpx._opts
        session = ctypes.c_void_p()
        self._begin_keep = bpx
        if ref_enabled:
            self._qf.retain_reference_audio(self._audio_reference)
            self._audio_reference_begin_failed = True
        st = lib.qf_xfm_session_begin(self._qf.xfm, ctypes.byref(bpx), ctypes.byref(session))
        if not ref_enabled and st != qfe.QF_OK and "busy" in (qfe.last_err(lib) or ""):
            time.sleep(_BUSY_BEGIN_RETRY_S)
            self._qf.end_session_if_open()
            st = lib.qf_xfm_session_begin(self._qf.xfm, ctypes.byref(bpx), ctypes.byref(session))
        if st != qfe.QF_OK:
            raise RuntimeError(f"qf_xfm_session_begin (LTX) failed: {qfe.last_err(lib)}")
        self._qf.current_session = session
        if ref_enabled:
            self._qf.reference_audio_copy_completed()
            self._audio_reference_bound = True
            self._audio_reference_begin_failed = False
        self._step_i = 0
        self._sess_denoise = 0
        qfe.info(f"[qf_native] LTX SESSION OPEN handle={session.value:#x} steps={self._num_steps} "
                 f"cond={tuple(vemb_group.shape)} frames={self._num_frames}", flush=True)
        if mask_opt.get("audio_frame_t_scales"):
            qfe.info(f"qf_native LTX-AV audio mask: {len(mask_opt['audio_frame_t_scales'])} rows", flush=True)


@qfe.console_safe_methods
class QFLTXAVModel(QFLTXModel):
    """LTX-2.5 joint audio+video wrapper driving qf_xfm_session_begin + qf_xfm_step_av."""

    def __init__(self, model_config, engine, device=None):
        QFLTXModel.__init__(self, model_config, engine, device=device)
        self._out_audio = None
        self._ctx_unprocessed = False
        self._audio_rows = 0
        self._audio_pack = 0

    @staticmethod
    def _reference_activation_dtype(_device):
        return torch.bfloat16

    @staticmethod
    def _reference_audio_tokens(payload):
        if not isinstance(payload, dict) or set(payload) != {"tokens"}:
            raise RuntimeError("qf_native LTX-AV: ref_audio must contain exactly tokens")
        tokens = payload["tokens"]
        if (not torch.is_tensor(tokens) or tokens.ndim != 3 or tokens.shape[0] != 1
                or tokens.shape[1] <= 0 or tokens.shape[2] != _LTXAV_AUDIO_PACK
                or tokens.numel() > _REF_AUDIO_MAX_NUMEL):
            raise RuntimeError("qf_native LTX-AV: ref_audio tokens must be [1,N,128], N>0 (one clip)")
        if tokens.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError("qf_native LTX-AV: ref_audio needs FP16/BF16/FP32 tokens")
        if not torch.isfinite(tokens).all().item():
            raise RuntimeError("qf_native LTX-AV: ref_audio tokens must be finite")
        return tokens

    def _reference_audio_cond(self, payload):
        if payload is None:
            return None
        self._reference_audio_tokens(payload)
        return comfy.conds.CONDConstant(payload)

    def _clear_audio_reference_after_close(self, was_open, ok):
        if getattr(self, "_audio_reference", None) is None or not ok:
            return
        if getattr(self, "_audio_reference_begin_failed", False) and self._qf.reference_audio_pending():
            return
        self._qf.retain_reference_audio(None)
        self._audio_reference = None
        self._audio_reference_bound = False
        self._audio_reference_begin_failed = False

    def _prepare_audio_reference(self, payload, device, activation_dtype, opening):
        if opening and getattr(self, "_audio_reference", None) is not None:
            was_open, ok = self._qf.end_session_if_open()
            self._clear_audio_reference_after_close(was_open, ok)
            if self._audio_reference is not None:
                raise RuntimeError("qf_native LTX-AV: reference copy/close still pending; wait for safe drain")
        if payload is None:
            return False
        tokens = self._reference_audio_tokens(payload)
        state = getattr(self, "_audio_reference", None)
        if not opening:
            if state is None or not getattr(self, "_audio_reference_bound", False):
                raise RuntimeError("qf_native LTX-AV: ref_audio binding needs a new session")
            source, shape, dtype, snapshot, _ = state
            if (tokens is not source or tuple(tokens.shape) != shape or tokens.dtype != dtype
                    or not torch.equal(tokens.detach().to(device="cpu"), snapshot)):
                raise RuntimeError("qf_native LTX-AV: ref_audio changed; start a new session")
            return True
        shape, dtype = tuple(tokens.shape), tokens.dtype
        snapshot = tokens.detach().to(device="cpu").clone()
        upload = snapshot.to(device=device, dtype=activation_dtype).contiguous()
        if not torch.isfinite(upload).all().item():
            raise RuntimeError("qf_native LTX-AV: normalized ref_audio must be finite (cast overflow)")
        self._audio_reference = (tokens, shape, dtype, snapshot, upload)
        self._audio_reference_bound = False
        self._audio_reference_begin_failed = False
        return True

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        self._ctx_unprocessed = bool(kwargs.get("unprocessed_ltxav_embeds", False))
        return out

    def _begin_extra_opts(self):
        if not self._ctx_unprocessed:
            raise RuntimeError(
                "qf_native LTX-AV: wired text encoder did not mark output unprocessed_ltxav_embeds - "
                "use the LTX-2.5 'with-proj' TE through ComfyUI's LTXAV clip.")
        if self._audio_rows <= 0:
            raise RuntimeError("qf_native LTX-AV: audio latent rows unset at begin - wiring error")
        opts = {"audio_dims": [1, int(self._audio_rows), int(self._audio_pack)], "av_unprocessed_ctx": True}
        reference = getattr(self, "_audio_reference", None)
        if reference is not None:
            upload = reference[-1]
            if not torch.isfinite(upload).all().item():
                raise RuntimeError("qf_native LTX-AV: normalized ref_audio must be finite")
            opts["av_conds"] = [{
                "kind": "audio",
                "audio_latent_ptr": hex(upload.data_ptr()),
                "audio_latent_dims": list(upload.shape),
                "audio_latent_dtype": int(_qf_dtype(upload.dtype)),
            }]
        return opts

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        sigma = t
        if c_crossattn is None:
            raise RuntimeError("qf_native LTX-AV: no c_crossattn cond - wire the LTX-2.5 TE.")
        if control is not None:
            raise RuntimeError("qf_native LTX-AV: ControlNet is wired, but this seam does not consume control hints.")
        nested = getattr(x, "is_nested", False)
        if nested:
            streams = x.unbind()
            x_video, x_audio = streams[0], streams[1]
        else:
            ls = getattr(self, "latent_shapes", None)
            if not ls or len(ls) < 2:
                raise RuntimeError(
                    f"qf_native LTX-AV: packed latent (shape={tuple(getattr(x, 'shape', ()))}) "
                    f"but model.latent_shapes is unset.")
            n = int(math.prod(ls[0][1:]))
            xf = x.reshape(int(x.shape[0]), -1)
            video_shape, audio_shape = list(ls[0]), list(ls[1])
            video_shape[0] = audio_shape[0] = int(x.shape[0])
            x_video = xf[:, :n].reshape(video_shape)
            x_audio = xf[:, n:].reshape(audio_shape)
        if (x_audio.ndim != _AUDIO_LATENT_NDIM or int(x_audio.shape[1]) != _LTXAV_AUDIO_CH
                or int(x_audio.shape[3]) != _LTXAV_AUDIO_MEL):
            raise RuntimeError(
                f"qf_native LTX-AV: audio latent shape {tuple(x_audio.shape)} - "
                f"expected [B,{_LTXAV_AUDIO_CH},L,{_LTXAV_AUDIO_MEL}].")
        dev = x_video.device
        xin = x_video.to(torch.bfloat16).contiguous()
        batch = int(xin.shape[0])
        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        if batch > 1 and (cou is None or len(cou) == 0 or batch % len(cou) != 0):
            raise RuntimeError(
                f"qf_native LTX-AV: engine forward is B==1 per cond group (got batch={batch}, cou={cou})")
        targets_per_group = batch // len(cou) if batch > 1 else 1
        self._derive_geometry(xin, transformer_options)
        _, ca, la, fa = (int(d) for d in x_audio.shape)
        self._audio_rows = la
        self._audio_pack = ca * fa
        vemb = c_crossattn.to(dev, dtype=torch.bfloat16).contiguous()
        if int(vemb.shape[0]) != batch:
            raise RuntimeError("qf_native LTX-AV: context batch must match flattened target batch")
        opening = self._qf.current_session is None or getattr(self, "_qf_needs_begin", False)
        self._call_fps(kwargs, opening)
        ref_active = self._prepare_audio_reference(
            kwargs.get("ref_audio"), dev,
            (self._reference_activation_dtype(dev) if opening and kwargs.get("ref_audio") is not None else None),
            opening)
        if opening:
            self._qf_needs_begin = False
            qfmp.refuse_all_zero_initial_latent(xin, "LTX-AV")
            self._begin(xin[0:1].contiguous(), vemb[0:1].contiguous())
        _, c, f_dim, h, w = xin.shape
        n_tok = f_dim * h * w
        if self._out is None or self._out.shape != (1, n_tok, c):
            self._out = torch.empty((1, n_tok, c), dtype=xin.dtype, device=dev)
        if self._out_audio is None or self._out_audio.shape != (1, la, ca * fa):
            self._out_audio = torch.empty((1, la, ca * fa), dtype=torch.float32, device=dev)
        sig_all = sigma.reshape(-1) if torch.is_tensor(sigma) else None
        step_index = self._sigma_step_index(sigma, sig_all, transformer_options)
        cuuids = transformer_options.get("uuids") if isinstance(transformer_options, dict) else None
        out5d = torch.empty_like(xin)
        out_audio = torch.empty_like(x_audio)
        for i in range(batch):
            _interrupt_poll_end_session_on_raise(self._qf)
            xi = xin[i:i + 1].contiguous()
            tokens = xi.reshape(1, c, n_tok).transpose(1, 2).contiguous()
            arows = x_audio[i:i + 1].float().permute(0, 2, 1, 3).reshape(1, la, ca * fa).contiguous()
            vi = vemb[i:i + 1].contiguous()
            sig_i = (float(sig_all[i].item()) if (sig_all is not None and sig_all.numel() >= batch)
                     else (float(sig_all[0].item()) if sig_all is not None else float(sigma)))
            p = qfe.QFStepParams()
            ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
            p.struct_size = ctypes.sizeof(p)
            p.latent_in = tokens.data_ptr()
            p.velocity_out = self._out.data_ptr()
            p.velocity_out_capacity = self._out.numel() * self._out.element_size()
            p.dims = (ctypes.c_int32 * 5)(1, n_tok, c, 0, 0)
            p.dtype = _qf_dtype(tokens.dtype)
            p.sigma = sig_i
            p.step_index = step_index
            p.total_steps = self._num_steps
            p.context = vi.data_ptr()
            p.context_dims = (ctypes.c_int32 * 3)(*vi.shape)
            p.context_dtype = _qf_dtype(vi.dtype)
            group = i // targets_per_group
            cuid = cuuids[group] if (cuuids is not None and group < len(cuuids)) else None
            if targets_per_group > 1 and cuid is not None:
                cuid = (cuid, i % targets_per_group)
            p.cfg_context_key = self._ctx_key_assigner.key(cuid)
            mp = qfe.QFStepAvParams()
            ctypes.memset(ctypes.byref(mp), 0, ctypes.sizeof(mp))
            mp.struct_size = ctypes.sizeof(mp)
            mp.base = p
            mp.audio_latent_in = arows.data_ptr()
            mp.audio_velocity_out = self._out_audio.data_ptr()
            mp.audio_velocity_out_capacity = self._out_audio.numel() * self._out_audio.element_size()
            mp.audio_dims = (ctypes.c_int32 * 4)(1, la, ca * fa, 0)
            mp.audio_dtype = _qf_dtype(arows.dtype)
            mp.audio_scale = 1.0
            mp.audio_reference_active = 1 if (getattr(self, "_audio_reference_bound", False) and ref_active) else 0
            self._call_denoise_step_av(mp, f"LTX-AV qf_xfm_step_av[step={step_index},group={i}]")
            out5d[i:i + 1] = self._out.transpose(1, 2).reshape(1, c, f_dim, h, w).to(xin.dtype)
            out_audio[i:i + 1] = self._out_audio.reshape(1, la, ca, fa).permute(0, 2, 1, 3).to(out_audio.dtype)
            self._qf.step_count += 1
            self._sess_denoise += 1
        self._step_i += 1
        self._qf.sampler_step_count += 1
        if nested:
            den_v = self.model_sampling.calculate_denoised(sigma, out5d.float(), x_video)
            den_a = self.model_sampling.calculate_denoised(sigma, out_audio.float(), x_audio)
            return comfy.nested_tensor.NestedTensor((den_v, den_a))
        vel_flat = torch.cat([out5d.reshape(batch, -1), out_audio.reshape(batch, -1)], dim=1).reshape(x.shape)
        return self.model_sampling.calculate_denoised(sigma, vel_flat.float(), x)

    def process_latent_out(self, latent):
        if self._qf.current_session is not None:
            try:
                qfe.info(f"[qf_native] LTX-AV SESSION CLOSED after {self._step_i} sampler steps, "
                         f"{self._sess_denoise} qf_xfm_step_av calls", flush=True)
            finally:
                self._qf.end_session_if_open()
        return super(QFLTXModel, self).process_latent_out(latent)

    def _qf_restore_frozen_audio(self, output, noise, latent_image, sampler, sigmas,
                                 denoise_mask=None, callback=None, disable_pbar=False,
                                 seed=None, latent_shapes=None):
        if denoise_mask is None:
            return output
        if (not isinstance(latent_shapes, (list, tuple)) or len(latent_shapes) != 2
                or not all(isinstance(s, (list, tuple)) for s in latent_shapes)
                or len(latent_shapes[0]) != _VIDEO_LATENT_NDIM or len(latent_shapes[1]) != _AUDIO_LATENT_NDIM
                or not all(isinstance(d, int) and not isinstance(d, bool) and d > 0
                           for s in latent_shapes for d in s)
                or latent_shapes[0][0] != latent_shapes[1][0]):
            raise RuntimeError("qf_native LTX-AV: frozen audio requires per-call video/audio shapes with matching batch")
        batch = latent_shapes[0][0]
        video_size = math.prod(latent_shapes[0][1:])
        row_size = video_size + math.prod(latent_shapes[1][1:])
        for label, tensor in (("input", latent_image), ("mask", denoise_mask), ("output", output)):
            if (not isinstance(tensor, torch.Tensor) or tensor.is_nested or tensor.ndim < 2
                    or tensor.shape[0] != batch or tensor.numel() != batch * row_size):
                raise RuntimeError(f"qf_native LTX-AV: frozen audio {label} does not match packed per-call geometry")
        frozen = denoise_mask.reshape(batch, -1)[:, video_size:] == 0
        if not bool(frozen.any()):
            return output
        clean = latent_image.reshape(batch, -1)[:, video_size:].to(device=output.device, dtype=output.dtype)
        restored = output.reshape(batch, -1).clone()
        restored[:, video_size:] = torch.where(frozen.to(device=output.device), clean, restored[:, video_size:])
        return restored.reshape(output.shape)


def _resolve_ltx_connectors(transformer1_path):
    """Locate the joint-AV connectors safetensors source (self-contained or same-dir sibling)."""
    def _qualifies(path):
        return _file_has_prefix(path, _CONN_VID_PREFIX) and _file_has_prefix(path, _CONN_AUD_PREFIX)

    if _qualifies(transformer1_path):
        return transformer1_path
    parent = os.path.dirname(transformer1_path)
    try:
        sibs = sorted(f for f in os.listdir(parent)
                      if f.lower().endswith(".safetensors") and "connector" in f.lower())
    except OSError:
        sibs = []
    qualified = [os.path.join(parent, f) for f in sibs
                 if os.path.join(parent, f) != transformer1_path and _qualifies(os.path.join(parent, f))]
    if qualified:
        if len(qualified) > 1:
            qfe.info(f"[qf_native] ltx2: {len(qualified)} qualifying connectors files in {parent!r}; "
                     f"using {os.path.basename(qualified[0])}", flush=True)
        return qualified[0]
    why = ("packs video_embeddings_connector but NOT audio_embeddings_connector - LTX-2.5 needs BOTH - "
           if _file_has_prefix(transformer1_path, _CONN_VID_PREFIX) else "is a transformer-only export ")
    raise RuntimeError(
        f"qf_native ltx2: {os.path.basename(transformer1_path)} {why}"
        f"and no connectors completion file was found next to it (probed {parent!r} for "
        f"*connector*.safetensors packing BOTH {_CONN_VID_PREFIX}* and {_CONN_AUD_PREFIX}*).")


def register(deps):
    def build(transformer1_path, bundle_dir=None, lora_entries=(), pinned_memory=False):
        config_path = os.path.join(bundle_dir, "transformer", "config.json") if bundle_dir else ""
        if not os.path.isfile(config_path):
            raise RuntimeError(f"qf_native ltx2: transformer config missing: {config_path}")
        conn_src = _resolve_ltx_connectors(transformer1_path)
        create_extra = {
            "model_arch": "ltx25",
            "config_path": config_path,
            "connectors_path": conn_src,
        }
        return qfmp.family_build(
            deps, transformer1_path, create_extra,
            comfy.supported_models.LTXAV,
            {"image_model": "ltxav", "disable_unet_model_creation": True},
            QFLTXAVModel,
            f"[qf_native] loaded QuantFuncLTXLoader package={os.path.basename(transformer1_path)} (create deferred)",
            pinned_memory=pinned_memory,
        )(list(lora_entries))

    return build
