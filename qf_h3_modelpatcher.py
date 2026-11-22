"""qf_h3_modelpatcher -- MiniMax-H3 joint audio+video family seam for QuantFunc PC Engine."""
import ctypes
import inspect
import json
import math
import os
import time

import torch
import comfy.conds
import comfy.model_base
import comfy.nested_tensor
import comfy.supported_models
from comfy.ldm.minimax.model import pack_audio, unpack_audio, patchify_video, unpatchify_video

from . import qf_engine as qfe
from . import qf_modelpatcher as qfmp
from .qf_modelpatcher import _qf_dtype, _QFStub, _interrupt_poll_end_session_on_raise, QFSessionModelMixin

FAMILY = "minimax-h3"
_log = qfe.logger(__name__)
_H3_PATCH = tuple(inspect.signature(patchify_video).parameters["patch_size"].default)
_BUSY_BEGIN_RETRY_S = 2.0
_AUDIO_ENHANCE_DEPRECATION_WARNED = False


def _h3_nodes():
    try:
        from comfy_extras import nodes_minimax_h3
    except ImportError as exc:
        raise RuntimeError(
            "qf_native H3: ComfyUI's comfy_extras.nodes_minimax_h3 is missing - update ComfyUI") from exc
    return nodes_minimax_h3


def _h3_frames_from_latent_t(latent_t):
    n = _h3_nodes()
    lt = int(latent_t)
    fc = n.align_frame_count(1)
    while n.video_latent_t(fc) < lt:
        fc = n.align_frame_count(fc + 1)
    if n.video_latent_t(fc) != lt:
        raise RuntimeError(
            f"qf_native H3: no frame count on ComfyUI's H3 grid has a video latent of length {lt}")
    return fc


def matches(pipeline_class, transformer_class=""):
    return (str(pipeline_class) in ("MiniMaxH3ModularPipeline", "MiniMaxH3Pipeline")
            or str(transformer_class) == "MiniMaxH3Transformer3DModel")


@qfe.console_safe_methods
class QFH3Model(QFSessionModelMixin, comfy.model_base.MiniMaxH3):
    """MiniMax-H3 joint-AV model wrapper driving qf_xfm_session_begin + qf_xfm_step_av."""

    _SESSION_DIALS = QFSessionModelMixin._SESSION_DIALS + ("_audio_enhance",)
    _ENGINE_IGNORED_COND_KEYS = (
        "denoise_mask", "concat_mask", "noise_concat",
        "concat_latent_image", "cross_attn_controlnet",
        "minimax_visual_cond_noise_aug", "minimax_audio_cond_noise_aug",
    )

    def __init__(self, model_config, engine, device=None):
        super().__init__(model_config, device=device)
        self.diffusion_model = _QFStub()
        self._qf = engine
        self._num_steps = 0
        self._num_frames = 0
        self._fps = float(_h3_nodes().FPS)
        self._audio_enhance = False
        self._step_i = 0
        self._sess_denoise = 0
        self._out_video = None
        self._out_audio = None
        self._max_ctx_seq = 0
        self._ctx_key_assigner = qfmp._CtxKeyAssigner()

    def set_audio_enhance(self, on):
        """Positional-compat switch only: warns once when True and NEVER forwarded to PC Engine."""
        global _AUDIO_ENHANCE_DEPRECATION_WARNED
        self._audio_enhance = bool(on)
        if self._audio_enhance and not _AUDIO_ENHANCE_DEPRECATION_WARNED:
            _AUDIO_ENHANCE_DEPRECATION_WARNED = True
            _log.warning(
                "[qf_native] H3: audio_enhance is deprecated in QuantFunc PC Engine and ignored "
                "(kept only for workflow compatibility).")

    def extra_conds(self, **kwargs):
        was_open, ok = self._qf.end_session_if_open()
        self._qf_needs_begin = True
        if was_open:
            qfe.info("[qf_native] H3: closed a pre-existing session at run start; end ok=%s", ok)
        payload = {}
        kf = kwargs.get("minimax_keyframes")
        if kf is not None:
            for e in kf:
                lat = e.get("latent")
                if lat is None or getattr(lat, "ndim", 0) != 5:
                    raise RuntimeError(
                        f"qf_native H3: minimax_keyframes entry lacks a 5-D latent "
                        f"(got {type(lat).__name__}{getattr(lat, 'shape', '')})")
            payload["keyframes"] = kf
        refs = kwargs.get("minimax_refs")
        if refs is not None:
            for r in refs:
                kind = r.get("kind")
                if kind in ("image", "video", "video_audio"):
                    lat = r.get("latent")
                    if lat is None or getattr(lat, "ndim", 0) != 5:
                        raise RuntimeError(
                            f"qf_native H3: minimax_refs {kind} entry lacks a 5-D latent "
                            f"(got {type(lat).__name__}{getattr(lat, 'shape', '')})")
                    if kind == "image" and int(lat.shape[2]) != 1:
                        raise RuntimeError(
                            f"qf_native H3: minimax_refs image entry must be single-frame (got T={int(lat.shape[2])})")
                    if kind in ("video", "video_audio") and int(lat.shape[2]) < 1:
                        raise RuntimeError("qf_native H3: minimax_refs video entry has no frames")
                if kind in ("audio", "video_audio"):
                    alat = r.get("audio_latent")
                    if alat is None or getattr(alat, "ndim", 0) != 4:
                        raise RuntimeError(f"qf_native H3: minimax_refs {kind} entry lacks a 4-D audio latent")
                if kind not in ("image", "audio", "video", "video_audio"):
                    raise RuntimeError(f"qf_native H3: unknown minimax_refs kind '{kind}'")
            payload["refs"] = refs
        tags = kwargs.get("minimax_token_tags")
        if tags is not None:
            payload["text_token_tags"] = tags
        for k in self._ENGINE_IGNORED_COND_KEYS:
            if kwargs.get(k) is not None:
                raise RuntimeError(
                    f"qf_native H3: '{k}' conditioning is wired, but the native session cannot consume it. "
                    f"Remove the node feeding it.")
        out = {}
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
            self._max_ctx_seq = max(self._max_ctx_seq, int(cross_attn.shape[1]))
        if payload:
            out["minimax_payload"] = comfy.conds.CONDConstant(payload)
        return out

    def scale_latent_inpaint(self, *args, **kwargs):
        raise RuntimeError(
            "qf_native H3: a denoise/inpaint mask (SetLatentNoiseMask) is wired, but the QuantFunc H3 "
            "native session does not support masked inpainting. Remove the mask node.")

    def _derive_geometry(self, x_video, transformer_options):
        self._num_frames = _h3_frames_from_latent_t(int(x_video.shape[2]))
        self._stage_schedule(transformer_options, "H3")

    def _begin(self, x_video, x_audio, vemb, av_payload=None):
        lib = self._qf.lib
        self._qf.end_session_if_open()
        self._ctx_key_assigner.reset()
        bpx = qfe.QFSessionBeginParams()
        ctypes.memset(ctypes.byref(bpx), 0, ctypes.sizeof(bpx))
        bpx.struct_size = ctypes.sizeof(bpx)
        scale = int(self.latent_format.spacial_downscale_ratio)
        bpx.width = int(x_video.shape[-1]) * scale
        bpx.height = int(x_video.shape[-2]) * scale
        bpx.num_steps = self._num_steps
        max_seq = max(self._max_ctx_seq, int(vemb.shape[1]))
        bpx.max_context_dims = (ctypes.c_int32 * 3)(int(vemb.shape[0]), max_seq, int(vemb.shape[2]))
        self._max_ctx_seq = 0
        bpx.cond_dtype = _qf_dtype(vemb.dtype)
        audio_dims = [int(d) for d in x_audio.shape]
        ms = self.model_sampling
        if not hasattr(ms, "shift") or not hasattr(ms, "audio_shift"):
            raise RuntimeError(
                f"qf_native H3: model_sampling is {type(ms).__name__} without audio_shift; "
                f"use ModelSamplingMiniMaxH3 or keep checkpoint defaults.")
        audio_shift = ms.shift if ms.audio_shift is None else ms.audio_shift
        if not all(isinstance(v, (int, float)) and v > 0 for v in (ms.shift, audio_shift)):
            raise RuntimeError(
                f"qf_native H3: shift={ms.shift!r}, audio_shift={ms.audio_shift!r} must both be positive.")
        opts = {
            **self.residency_opts(),
            "audio_dims": audio_dims,
            "av_sigma_shift_video": float(ms.shift),
            "av_sigma_shift_audio": float(audio_shift),
            "num_frames": self._num_frames,
            "fps": float(self._fps),
        }
        self._kf_keep = []
        conds = []
        pl = av_payload or {}
        self._bound_av_counts = (len(pl.get("keyframes") or []), len(pl.get("refs") or []))
        for e in (pl.get("keyframes") or []):
            lat = e["latent"]
            if not lat.is_cuda:
                lat = lat.to(x_video.device)
            lat = lat.contiguous()
            self._kf_keep.append(lat)
            fidx = int(e.get("resolved_frame_index", 0))
            conds.append({
                "kind": "keyframe",
                "frame_index": 0 if fidx == 0 else -1,
                "latent_ptr": hex(lat.data_ptr()),
                "latent_dims": [int(d) for d in lat.shape],
                "latent_dtype": int(_qf_dtype(lat.dtype)),
            })
        for r in (pl.get("refs") or []):
            c = {"kind": r["kind"]}
            if r.get("latent") is not None:
                lat = r["latent"]
                if not lat.is_cuda:
                    lat = lat.to(x_video.device)
                lat = lat.contiguous()
                self._kf_keep.append(lat)
                c["latent_ptr"] = hex(lat.data_ptr())
                c["latent_dims"] = [int(d) for d in lat.shape]
                c["latent_dtype"] = int(_qf_dtype(lat.dtype))
            if r.get("audio_latent") is not None:
                alat = r["audio_latent"]
                if not alat.is_cuda:
                    alat = alat.to(x_video.device)
                alat = alat.contiguous()
                self._kf_keep.append(alat)
                c["audio_latent_ptr"] = hex(alat.data_ptr())
                c["audio_latent_dims"] = [int(d) for d in alat.shape]
                c["audio_latent_dtype"] = int(_qf_dtype(alat.dtype))
            conds.append(c)
        if conds:
            opts["av_conds"] = conds
        tags = pl.get("text_token_tags")
        if tags is not None:
            tl = tags.view(-1).tolist() if hasattr(tags, "view") else list(tags)
            ranges, start = [], None
            for i, tv in enumerate(tl):
                if int(tv) == 0 and start is None:
                    start = i
                elif int(tv) != 0 and start is not None:
                    ranges.append([start, i - start])
                    start = None
            if start is not None:
                ranges.append([start, len(tl) - start])
            if ranges:
                opts["text_vision_row_ranges"] = ranges
        bpx._opts = json.dumps(opts).encode("utf-8")
        bpx.options_json = bpx._opts
        session = ctypes.c_void_p()
        st = lib.qf_xfm_session_begin(self._qf.xfm, ctypes.byref(bpx), ctypes.byref(session))
        if st != qfe.QF_OK and "busy" in (qfe.last_err(lib) or ""):
            time.sleep(_BUSY_BEGIN_RETRY_S)
            self._qf.end_session_if_open()
            st = lib.qf_xfm_session_begin(self._qf.xfm, ctypes.byref(bpx), ctypes.byref(session))
        self._begin_keep = bpx
        if st != qfe.QF_OK:
            raise RuntimeError(f"qf_xfm_session_begin (H3) failed: {qfe.last_err(lib)}")
        self._qf.current_session = session
        self._kf_keep = []
        self._step_i = 0
        self._sess_denoise = 0
        qfe.info(f"[qf_native] H3 SESSION OPEN handle={session.value:#x} steps={self._num_steps} "
                 f"cond={tuple(vemb.shape)} audio_dims={audio_dims} frames={self._num_frames}", flush=True)

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        sigma = t
        if c_crossattn is None:
            raise RuntimeError("qf_native H3: no c_crossattn cond - wire the H3 CLIP prompt.")
        if control is not None:
            raise RuntimeError("qf_native H3: a ControlNet is wired but this seam does not consume control hints.")
        nested = getattr(x, "is_nested", False)
        if nested:
            streams = x.unbind()
            x_video, x_audio = streams[0], streams[1]
        else:
            ls = getattr(self, "latent_shapes", None)
            if not ls or len(ls) < 2:
                raise RuntimeError(
                    f"qf_native H3: packed latent (shape={tuple(x.shape)}) but model.latent_shapes is unset.")
            n = int(math.prod(ls[0][1:]))
            xf = x.reshape(int(x.shape[0]), -1)
            video_shape, audio_shape = list(ls[0]), list(ls[1])
            video_shape[0] = audio_shape[0] = int(x.shape[0])
            x_video = xf[:, :n].reshape(video_shape)
            x_audio = xf[:, n:].reshape(audio_shape)
        dev = x_video.device
        batch = int(x_video.shape[0])
        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        cuuids = transformer_options.get("uuids") if isinstance(transformer_options, dict) else None
        if batch > 1 and (cou is None or len(cou) != batch):
            raise RuntimeError(
                f"qf_native H3: engine forward is B==1 per cond group (got batch={batch}, cou={cou})")
        vemb = c_crossattn.to(dev, dtype=torch.bfloat16).contiguous()
        if self._qf.current_session is None or getattr(self, "_qf_needs_begin", False):
            self._qf_needs_begin = False
            self._derive_geometry(x_video, transformer_options)
            qfmp.refuse_all_zero_initial_latent(x_video, "H3")
            self._begin(x_video[0:1].contiguous(), x_audio[0:1].contiguous(), vemb[0:1].contiguous(),
                        av_payload=kwargs.get("minimax_payload"))
        pl_now = kwargs.get("minimax_payload") or {}
        now_counts = (len(pl_now.get("keyframes") or []), len(pl_now.get("refs") or []))
        if now_counts != getattr(self, "_bound_av_counts", now_counts):
            raise RuntimeError(
                f"qf_native H3: cond group carries (keyframes, refs)={now_counts} but session bound "
                f"{self._bound_av_counts} - av_conds bind session-wide.")
        _, cv, tv, hv, wv = x_video.shape
        _, ca, ka, ta = (int(d) for d in x_audio.shape)
        n_rows = ka * ta
        pt, ph, pw = _H3_PATCH
        tt, th, tw = tv // pt, hv // ph, wv // pw
        ntok = tt * th * tw
        pch = cv * pt * ph * pw
        if self._out_video is None or self._out_video.shape != (1, ntok, pch):
            self._out_video = torch.empty((1, ntok, pch), dtype=torch.float32, device=dev)
        if self._out_audio is None or self._out_audio.shape != (1, n_rows, ca):
            self._out_audio = torch.empty((1, n_rows, ca), dtype=torch.float32, device=dev)
        sig_all = sigma.reshape(-1) if torch.is_tensor(sigma) else None
        step_index = self._sigma_step_index(sigma, sig_all, transformer_options)
        audio_scale = float(getattr(self.model_sampling, "audio_scale", 1.0))
        out_video = torch.empty_like(x_video)
        out_audio = torch.empty_like(x_audio)
        for i in range(batch):
            _interrupt_poll_end_session_on_raise(self._qf)
            vid_tok = patchify_video(x_video[i:i + 1]).to(torch.float32).contiguous().reshape(1, ntok, pch)
            xi_audio = pack_audio(x_audio[i:i + 1].float()).unsqueeze(0).contiguous()
            vi = vemb[i:i + 1].contiguous()
            sig_i = (float(sig_all[i].item()) if (sig_all is not None and sig_all.numel() >= batch)
                     else (float(sig_all[0].item()) if sig_all is not None else float(sigma)))
            p = qfe.QFStepParams()
            ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
            p.struct_size = ctypes.sizeof(p)
            p.latent_in = vid_tok.data_ptr()
            p.velocity_out = self._out_video.data_ptr()
            p.velocity_out_capacity = self._out_video.numel() * self._out_video.element_size()
            p.dims = (ctypes.c_int32 * 5)(1, ntok, pch, 0, 0)
            p.dtype = _qf_dtype(vid_tok.dtype)
            p.sigma = sig_i
            p.step_index = step_index
            p.total_steps = self._num_steps
            p.context = vi.data_ptr()
            p.context_dims = (ctypes.c_int32 * 3)(*vi.shape)
            p.context_dtype = _qf_dtype(vi.dtype)
            cuid = cuuids[i] if (cuuids is not None and i < len(cuuids)) else None
            p.cfg_context_key = self._ctx_key_assigner.key(cuid)
            mp = qfe.QFStepAvParams()
            ctypes.memset(ctypes.byref(mp), 0, ctypes.sizeof(mp))
            mp.struct_size = ctypes.sizeof(mp)
            mp.base = p
            mp.audio_latent_in = xi_audio.data_ptr()
            mp.audio_velocity_out = self._out_audio.data_ptr()
            mp.audio_velocity_out_capacity = self._out_audio.numel() * self._out_audio.element_size()
            mp.audio_dims = (ctypes.c_int32 * 4)(1, ca, ka, ta)
            mp.audio_dtype = _qf_dtype(xi_audio.dtype)
            mp.audio_scale = audio_scale
            mp.audio_reference_active = 0
            self._call_denoise_step_av(mp, f"H3 qf_xfm_step_av[step={step_index},group={i}]")
            out_video[i:i + 1] = unpatchify_video(
                self._out_video.reshape(ntok, pch), tt, th, tw, cv, patch_size=_H3_PATCH).to(out_video.dtype)
            out_audio[i:i + 1] = unpack_audio(self._out_audio[0]).to(out_audio.dtype)
            self._qf.step_count += 1
            self._sess_denoise += 1
        self._step_i += 1
        self._qf.sampler_step_count += 1
        if nested:
            den_v = self.model_sampling.calculate_denoised(sigma, out_video.float(), x_video)
            den_a = self.model_sampling.calculate_denoised(sigma, out_audio.float(), x_audio)
            return comfy.nested_tensor.NestedTensor((den_v, den_a))
        vel_flat = torch.cat([out_video.reshape(batch, -1), out_audio.reshape(batch, -1)], dim=1).reshape(x.shape)
        return self.model_sampling.calculate_denoised(sigma, vel_flat.float(), x)

    def process_latent_out(self, latent):
        if self._qf.current_session is not None:
            try:
                qfe.info(f"[qf_native] H3 SESSION CLOSED after {self._step_i} sampler steps, "
                         f"{self._sess_denoise} qf_xfm_step_av calls", flush=True)
            finally:
                self._qf.end_session_if_open()
        return super().process_latent_out(latent)


def register(deps):
    def build(transformer1_path, bundle_dir=None, lora_entries=(), pinned_memory=False):
        config_path = os.path.join(bundle_dir, "transformer", "config.json") if bundle_dir else ""
        if not os.path.isfile(config_path):
            raise RuntimeError(f"qf_native minimax-h3: transformer config missing: {config_path}")
        return qfmp.family_build(
            deps, transformer1_path,
            {"model_arch": "minimax_h3", "config_path": config_path},
            comfy.supported_models.MiniMaxH3,
            {"image_model": "minimax_h3", "disable_unet_model_creation": True},
            QFH3Model,
            f"[qf_native] loaded QuantFuncH3Loader package={os.path.basename(transformer1_path)} (create deferred)",
            pinned_memory=pinned_memory,
        )(list(lora_entries))

    return build
