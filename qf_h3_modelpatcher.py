"""qf_h3_modelpatcher -- MiniMax-H3 joint audio+video family seam for QuantFunc PC Engine."""
import inspect
import os

import torch
import comfy.conds
import comfy.model_base
import comfy.supported_models
from comfy.ldm.minimax.model import pack_audio, unpack_audio, patchify_video, unpatchify_video

from . import qf_engine as qfe
from . import qf_modelpatcher as qfmp
from .qf_modelpatcher import _qf_dtype, QFSessionModelMixin

FAMILY = "minimax-h3"
_log = qfe.logger(__name__)
_H3_PATCH = tuple(inspect.signature(patchify_video).parameters["patch_size"].default)
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
class QFH3Model(qfmp.QFAVSessionModel, comfy.model_base.MiniMaxH3):
    """MiniMax-H3 joint-AV model wrapper driving qf_xfm_session_begin + qf_xfm_step_av."""

    _TAG = "H3"
    _NO_COND_HINT = "wire the H3 CLIP prompt."
    _SESSION_DIALS = QFSessionModelMixin._SESSION_DIALS + ("_audio_enhance",)
    _ENGINE_IGNORED_COND_KEYS = (
        "denoise_mask", "concat_mask", "noise_concat",
        "concat_latent_image", "cross_attn_controlnet",
        "minimax_visual_cond_noise_aug", "minimax_audio_cond_noise_aug",
    )

    def __init__(self, model_config, engine, device=None):
        super().__init__(model_config, device=device)
        self._init_session_state(engine)
        self._num_frames = 0
        self._fps = float(_h3_nodes().FPS)
        self._audio_enhance = False
        self._out_video = None
        self._out_audio = None

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
        self._begin_extra_conds_guard("H3", noise=kwargs.get("noise"))
        try:
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
            self._reject_ignored_conds(kwargs, "H3")
            out = {}
            cross_attn = kwargs.get("cross_attn", None)
            if cross_attn is not None:
                out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
                self._max_ctx_seq = max(self._max_ctx_seq, int(cross_attn.shape[1]))
            if payload:
                out["minimax_payload"] = comfy.conds.CONDConstant(payload)
            return out
        except BaseException:
            self._reset_run_cond_state()
            self._qf_conds_done = True
            raise

    def scale_latent_inpaint(self, *args, **kwargs):
        raise RuntimeError(
            "qf_native H3: a denoise/inpaint mask (SetLatentNoiseMask) is wired, but the QuantFunc H3 "
            "native session does not support masked inpainting. Remove the mask node.")

    def _derive_geometry(self, x_video, transformer_options):
        self._num_frames = _h3_frames_from_latent_t(int(x_video.shape[2]))
        self._stage_schedule(transformer_options, "H3")

    def _begin(self, x_video, x_audio, vemb, av_payload=None):
        scale = int(self.latent_format.spacial_downscale_ratio)
        max_seq = max(self._max_ctx_seq, int(vemb.shape[1]))
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
        session = self._open_native_session(
            "H3",
            (int(vemb.shape[0]), max_seq, int(vemb.shape[2])),
            _qf_dtype(vemb.dtype),
            int(x_video.shape[-1]) * scale,
            int(x_video.shape[-2]) * scale,
            opts,
        )
        self._kf_keep = []
        qfe.info(f"[qf_native] H3 SESSION OPEN handle={session.value:#x} steps={self._num_steps} "
                 f"cond={tuple(vemb.shape)} audio_dims={audio_dims} frames={self._num_frames}", flush=True)

    def _av_unpack_and_validate(self, x, c_crossattn, transformer_options):
        nested, x_video, x_audio = self._unpack_av_latent(x)
        dev = x_video.device
        batch = int(x_video.shape[0])
        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        if batch > 1 and (cou is None or len(cou) != batch):
            raise RuntimeError(
                f"qf_native H3: engine forward is B==1 per cond group (got batch={batch}, cou={cou})")
        vemb = c_crossattn.to(dev, dtype=torch.bfloat16).contiguous()
        meta = qfmp._AVPackedMeta(nested, x, x_video, x_audio, 1)
        return x_video, x_audio, vemb, meta

    def _av_prepare_session(self, x_video, x_audio, vemb, transformer_options, kwargs, opening):
        geom = (tuple(int(d) for d in x_video.shape[1:]), tuple(int(d) for d in x_audio.shape[1:]))
        if opening or geom != getattr(self, "_qf_av_geom", None):
            self._derive_geometry(x_video, transformer_options)
            if not opening:
                self._qf_av_geom = geom
        if opening:
            qfmp.refuse_all_zero_initial_latent(x_video, "H3")
            self._begin(x_video[0:1].contiguous(), x_audio[0:1].contiguous(), vemb[0:1].contiguous(),
                        av_payload=kwargs.get("minimax_payload"))
            self._qf_av_geom = geom
            self._qf_needs_begin = False
        pl_now = kwargs.get("minimax_payload") or {}
        now_counts = (len(pl_now.get("keyframes") or []), len(pl_now.get("refs") or []))
        if now_counts != getattr(self, "_bound_av_counts", now_counts):
            raise RuntimeError(
                f"qf_native H3: cond group carries (keyframes, refs)={now_counts} but session bound "
                f"{self._bound_av_counts} - av_conds bind session-wide.")

    def _av_step_tensors(self, x_video, x_audio, vemb, meta, sigma, sig_all, transformer_options, kwargs, step_index):
        dev = x_video.device
        batch = int(x_video.shape[0])
        _, cv, tv, hv, wv = (int(d) for d in x_video.shape)
        _, ca, ka, ta = (int(d) for d in x_audio.shape)
        n_rows = ka * ta
        pt, ph, pw = _H3_PATCH
        tt, th, tw = tv // pt, hv // ph, wv // pw
        ntok = tt * th * tw
        pch = cv * pt * ph * pw
        self._out_video = self._ensure_scratch("_out_video", (1, ntok, pch), torch.float32, dev)
        self._out_audio = self._ensure_scratch("_out_audio", (1, n_rows, ca), torch.float32, dev)
        cuuids = transformer_options.get("uuids") if isinstance(transformer_options, dict) else None
        audio_scale = float(getattr(self.model_sampling, "audio_scale", 1.0))
        out_video = torch.empty_like(x_video)
        out_audio = torch.empty_like(x_audio)
        for i in range(batch):
            vid_tok = patchify_video(x_video[i:i + 1]).to(torch.float32).contiguous().reshape(1, ntok, pch)
            xi_audio = pack_audio(x_audio[i:i + 1].float()).unsqueeze(0).contiguous()
            vi = vemb[i:i + 1].contiguous()
            sig_i = (float(sig_all[i].item()) if (sig_all is not None and sig_all.numel() >= batch)
                     else (float(sig_all[0].item()) if sig_all is not None else float(sigma)))
            cuid = cuuids[i] if (cuuids is not None and i < len(cuuids)) else None
            ctx_key = self._ctx_key_assigner.key(cuid)
            self._execute_av_group_step(
                vid_in=vid_tok, vid_dims=(1, ntok, pch), vid_out=self._out_video,
                aud_in=xi_audio, aud_dims=(1, n_rows, ca), aud_out=self._out_audio,
                vi=vi, sig_i=sig_i, step_index=step_index, ctx_key=ctx_key,
                audio_scale=audio_scale, ref_active=False, group_i=i,
            )
            out_video[i:i + 1] = unpatchify_video(
                self._out_video.reshape(ntok, pch), tt, th, tw, cv, patch_size=_H3_PATCH).to(out_video.dtype)
            out_audio[i:i + 1] = unpack_audio(self._out_audio[0], ch=ka).to(out_audio.dtype)
        return out_video, out_audio


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
