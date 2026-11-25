"""qf_ltx_modelpatcher -- LTX-2.5 joint audio+video family seam for QuantFunc PC Engine.

Binds frozen 25-symbol qf_* C-API:
  - session begin: frame_t_scales, audio_frame_t_scales, audio_dims, av_unprocessed_ctx, av_conds
  - step: QFStepAvParams with audio_reference_active (0 or 1) via qf_xfm_step_av
  - OUTER_SAMPLE: _qf_restore_frozen_audio preserves zero-mask frozen audio frames
"""
import math
import os
import re
import struct

import torch
import comfy.conds
import comfy.model_base
import comfy.supported_models

from . import qf_engine as qfe
from . import qf_modelpatcher as qfmp
from .qf_modelpatcher import _qf_dtype, QFSessionModelMixin

FAMILY = "ltx2"
_LTXAV_AUDIO_CH = 8
_LTXAV_AUDIO_MEL = 16
_LTXAV_AUDIO_PACK = _LTXAV_AUDIO_CH * _LTXAV_AUDIO_MEL
_VIDEO_LATENT_NDIM = 5
_AUDIO_LATENT_NDIM = 4
_REF_AUDIO_MAX_NUMEL = (1 << 31) - 1
_ST_HEADER_MAX_BYTES = 1 << 31
_CONN_VID_PREFIX = "model.diffusion_model.video_embeddings_connector."
_CONN_AUD_PREFIX = "model.diffusion_model.audio_embeddings_connector."
_JSON_STRUCT_RE = re.compile(rb'["{}\[\]]')


def _iter_safetensors_top_keys(raw):
    """Yield depth-1 JSON object keys (excluding __metadata__) from raw safetensors header bytes."""
    n = len(raw)
    ws = b" \t\r\n"
    q = ord('"')
    bs = ord("\\")
    lbrace = ord("{")
    rbrace = ord("}")
    lbrack = ord("[")
    rbrack = ord("]")
    colon = ord(":")
    comma = ord(",")

    def _skip_ws(pos):
        while pos < n and raw[pos] in ws:
            pos += 1
        return pos

    def _read_string(pos):
        pos += 1
        start = pos
        has_esc = False
        while pos < n:
            qi = raw.find(b'"', pos)
            if qi == -1:
                return None, n
            bi = raw.find(b"\\", pos, qi)
            if bi == -1:
                raw_s = raw[start:qi]
                if not has_esc:
                    return raw_s, qi + 1
                out = bytearray()
                j = 0
                m = len(raw_s)
                while j < m:
                    c = raw_s[j]
                    if c == bs and j + 1 < m:
                        nxt = raw_s[j + 1]
                        if nxt in (q, bs, ord("/")):
                            out.append(nxt)
                        else:
                            out.append(c)
                            out.append(nxt)
                        j += 2
                    else:
                        out.append(c)
                        j += 1
                return bytes(out), qi + 1
            has_esc = True
            pos = bi + 2
            if pos > n:
                return None, n
        return None, n

    def _skip_value(pos):
        pos = _skip_ws(pos)
        if pos >= n:
            return n
        b = raw[pos]
        if b == q:
            _, end = _read_string(pos)
            return end
        if b in (lbrace, lbrack):
            depth = 0
            while pos < n:
                m = _JSON_STRUCT_RE.search(raw, pos)
                if m is None:
                    return n
                c = raw[m.start()]
                if c == q:
                    val, pos = _read_string(m.start())
                    if val is None:
                        return n
                    continue
                pos = m.end()
                if c in (lbrace, lbrack):
                    depth += 1
                else:
                    depth -= 1
                    if depth == 0:
                        return pos
            return n
        while pos < n and raw[pos] not in (comma, rbrace, rbrack) and raw[pos] not in ws:
            pos += 1
        return pos

    i = _skip_ws(0)
    if i >= n or raw[i] != lbrace:
        return
    i += 1
    while i < n:
        i = _skip_ws(i)
        if i >= n or raw[i] == rbrace:
            return
        if raw[i] != q:
            return
        key_b, i = _read_string(i)
        if key_b is None:
            return
        i = _skip_ws(i)
        if i >= n or raw[i] != colon:
            return
        i += 1
        if key_b != b"__metadata__":
            yield key_b
        i = _skip_value(i)
        i = _skip_ws(i)
        if i < n and raw[i] == comma:
            i += 1
            continue
        return


def _file_has_prefix(path, prefix):
    try:
        hdr_len_size = struct.calcsize("<Q")
        with open(path, "rb") as fh:
            len_buf = fh.read(hdr_len_size)
            if len(len_buf) != hdr_len_size:
                return False
            n = struct.unpack("<Q", len_buf)[0]
            if n == 0 or n > _ST_HEADER_MAX_BYTES:
                return False
            raw = fh.read(n)
            if len(raw) != n:
                return False
    except Exception:  # noqa: BLE001
        return False
    prefix_b = prefix.encode("utf-8") if isinstance(prefix, str) else bytes(prefix)
    if not prefix_b:
        return False
    wrap_b = b"model.diffusion_model."
    targets = (prefix_b,) if prefix_b.startswith(wrap_b) else (prefix_b, wrap_b + prefix_b)
    for key_b in _iter_safetensors_top_keys(raw):
        if any(key_b.startswith(t) for t in targets):
            return True
    return False


def matches(pipeline_class, transformer_class=""):
    return str(pipeline_class) == "LTX2Pipeline"


@qfe.console_safe_methods
class QFLTXModel(QFSessionModelMixin, comfy.model_base.LTXV):
    """LTX base wrapper carrying conditioning safety checks, mask-to-scale extraction, and session begin."""

    _TAG = "LTX"
    _NO_COND_HINT = "wire the LTX-2.5 TE."
    _REFUSED_HINT = " (for image-to-video use LTXVImgToVideoInplace on the LATENT path)"
    _ENGINE_IGNORED_COND_KEYS = (
        "concat_mask", "keyframe_idxs", "guide_attention_entries",
        "noise_concat", "cross_attn_controlnet", "concat_latent_image",
        "generated_keyframes",
    )

    def __init__(self, model_config, engine, device=None):
        super().__init__(model_config, device=device)
        self._init_session_state(engine)
        self._num_frames = 0
        self._fps = None

    def _reset_run_cond_state(self):
        super()._reset_run_cond_state()
        self._pending_frame_t_scale = None
        self._pending_audio_frame_t_scale = None

    def _reference_audio_cond(self, payload):
        if payload is not None:
            raise RuntimeError("qf_native LTX: ref_audio needs an AV checkpoint; video-only is unsupported")
        return None

    def _post_connector_seq(self, raw_s):
        return int(raw_s)

    def _begin_extra_opts(self):
        return {}

    def extra_conds(self, **kwargs):
        self._begin_extra_conds_guard(
            "LTX",
            on_close=getattr(self, "_clear_audio_reference_after_close", None),
            noise=kwargs.get("noise"),
        )
        try:
            dm = kwargs.get("denoise_mask")
            if dm is not None:
                scales, a_scales = self._frame_scales_from_mask(dm)
                scales = torch.tensor(scales, dtype=torch.bfloat16, device="cpu").float().tolist()
                a_scales = torch.tensor(a_scales, dtype=torch.bfloat16, device="cpu").float().tolist()
                self._pending_frame_t_scale = None if all(v >= 0.9999 for v in scales) else scales
                self._pending_audio_frame_t_scale = (
                    None if (not a_scales or all(v >= 0.9999 for v in a_scales)) else a_scales)
            self._reject_ignored_conds(kwargs, "LTX", self._REFUSED_HINT)
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
        except BaseException:
            self._reset_run_cond_state()
            self._qf_conds_done = True
            raise

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
        _, close_ok = self._qf.end_session_if_open()
        ref_enabled = getattr(self, "_audio_reference", None) is not None
        if getattr(self._qf, "reference_audio_pending", lambda: False)():
            raise RuntimeError("qf_native LTX-AV: previous reference begin has no proven copy completion")
        if ref_enabled and not close_ok:
            raise RuntimeError("qf_native LTX-AV: previous reference session close refused; retry after drain")
        scale = int(self.latent_format.spacial_downscale_ratio)
        max_seq = max(self._max_ctx_seq, int(vemb_group.shape[1]))
        mask_opt = {}
        if getattr(self, "_pending_frame_t_scale", None):
            mask_opt["frame_t_scales"] = self._pending_frame_t_scale
        if getattr(self, "_pending_audio_frame_t_scale", None):
            mask_opt["audio_frame_t_scales"] = self._pending_audio_frame_t_scale
        opts = {
            "num_frames": self._num_frames,
            "fps": float(self._fps),
            **self.residency_opts(),
            **mask_opt,
            **self._begin_extra_opts(),
        }

        def _before_ref():
            if ref_enabled:
                self._qf.retain_reference_audio(self._audio_reference)
                self._audio_reference_begin_failed = True

        def _after_ref():
            if ref_enabled:
                self._qf.reference_audio_copy_completed()
                self._audio_reference_bound = True
                self._audio_reference_begin_failed = False

        session = self._open_native_session(
            "LTX",
            (int(vemb_group.shape[0]), max_seq, int(vemb_group.shape[2])),
            _qf_dtype(vemb_group.dtype),
            int(x_group.shape[-1]) * scale,
            int(x_group.shape[-2]) * scale,
            opts,
            allow_busy_retry=not ref_enabled,
            before_begin=_before_ref,
            after_begin=_after_ref,
            skip_end_if_open=True,
        )
        qfe.info(f"[qf_native] LTX SESSION OPEN handle={session.value:#x} steps={self._num_steps} "
                 f"cond={tuple(vemb_group.shape)} frames={self._num_frames}", flush=True)
        if mask_opt.get("audio_frame_t_scales"):
            qfe.info(f"qf_native LTX-AV audio mask: {len(mask_opt['audio_frame_t_scales'])} rows", flush=True)


@qfe.console_safe_methods
class QFLTXAVModel(qfmp.QFAVSessionModel, QFLTXModel):
    """LTX-2.5 joint audio+video wrapper driving qf_xfm_session_begin + qf_xfm_step_av."""

    _TAG = "LTX-AV"

    def __init__(self, model_config, engine, device=None):
        QFLTXModel.__init__(self, model_config, engine, device=device)
        self._out_audio = None
        self._audio_rows = 0
        self._audio_pack = 0

    def _reset_run_cond_state(self):
        super()._reset_run_cond_state()
        self._ctx_unprocessed = False

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
        try:
            self._ctx_unprocessed = bool(kwargs.get("unprocessed_ltxav_embeds", False))
            return out
        except BaseException:
            self._reset_run_cond_state()
            self._qf_conds_done = True
            raise

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

    def _av_unpack_and_validate(self, x, c_crossattn, transformer_options):
        nested, x_video, x_audio = self._unpack_av_latent(x)
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
        vemb = c_crossattn.to(dev, dtype=torch.bfloat16).contiguous()
        if int(vemb.shape[0]) != batch:
            raise RuntimeError("qf_native LTX-AV: context batch must match flattened target batch")
        meta = qfmp._AVPackedMeta(nested, x, x_video, x_audio, targets_per_group)
        return xin, x_audio, vemb, meta

    def _av_prepare_session(self, xin, x_audio, vemb, transformer_options, kwargs, opening):
        geom = (tuple(int(d) for d in xin.shape[1:]), tuple(int(d) for d in x_audio.shape[1:]))
        if opening or geom != getattr(self, "_qf_av_geom", None):
            self._derive_geometry(xin, transformer_options)
            if not opening:
                self._qf_av_geom = geom
        _, ca, la, fa = (int(d) for d in x_audio.shape)
        self._audio_rows = la
        self._audio_pack = ca * fa
        dev = xin.device
        self._call_fps(kwargs, opening)
        ref_active = self._prepare_audio_reference(
            kwargs.get("ref_audio"), dev,
            (self._reference_activation_dtype(dev) if opening and kwargs.get("ref_audio") is not None else None),
            opening)
        self._qf_ref_active = ref_active
        if opening:
            qfmp.refuse_all_zero_initial_latent(xin, "LTX-AV")
            self._begin(xin[0:1].contiguous(), vemb[0:1].contiguous())
            self._qf_av_geom = geom
            self._qf_needs_begin = False

    def _av_step_tensors(self, xin, x_audio, vemb, meta, sigma, sig_all, transformer_options, kwargs, step_index):
        dev = xin.device
        batch = int(xin.shape[0])
        _, c, f_dim, h, w = (int(d) for d in xin.shape)
        _, ca, la, fa = (int(d) for d in x_audio.shape)
        n_tok = f_dim * h * w
        self._out = self._ensure_scratch("_out", (1, n_tok, c), xin.dtype, dev)
        self._out_audio = self._ensure_scratch("_out_audio", (1, la, ca * fa), torch.float32, dev)
        cuuids = transformer_options.get("uuids") if isinstance(transformer_options, dict) else None
        ref_active = bool(getattr(self, "_audio_reference_bound", False)
                          and getattr(self, "_qf_ref_active", False))
        targets_per_group = meta.targets_per_group
        out5d = torch.empty_like(xin)
        out_audio = torch.empty_like(x_audio)
        for i in range(batch):
            xi = xin[i:i + 1].contiguous()
            tokens = xi.reshape(1, c, n_tok).transpose(1, 2).contiguous()
            arows = x_audio[i:i + 1].float().permute(0, 2, 1, 3).reshape(1, la, ca * fa).contiguous()
            vi = vemb[i:i + 1].contiguous()
            sig_i = (float(sig_all[i].item()) if (sig_all is not None and sig_all.numel() >= batch)
                     else (float(sig_all[0].item()) if sig_all is not None else float(sigma)))
            group = i // targets_per_group
            cuid = cuuids[group] if (cuuids is not None and group < len(cuuids)) else None
            if targets_per_group > 1 and cuid is not None:
                cuid = (cuid, i % targets_per_group)
            ctx_key = self._ctx_key_assigner.key(cuid)
            self._execute_av_group_step(
                vid_in=tokens, vid_dims=(1, n_tok, c), vid_out=self._out,
                aud_in=arows, aud_dims=(1, la, ca * fa), aud_out=self._out_audio,
                vi=vi, sig_i=sig_i, step_index=step_index, ctx_key=ctx_key,
                audio_scale=1.0, ref_active=ref_active, group_i=i,
            )
            out5d[i:i + 1] = self._out.transpose(1, 2).reshape(1, c, f_dim, h, w).to(xin.dtype)
            out_audio[i:i + 1] = self._out_audio.reshape(1, la, ca, fa).permute(0, 2, 1, 3).to(out_audio.dtype)
        return out5d, out_audio

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
