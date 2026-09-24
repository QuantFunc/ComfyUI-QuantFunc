"""qf_native.qf_h3_modelpatcher — the MiniMax-H3 seam: a comfy ModelPatcher whose `.model` is a
MiniMaxH3-shim (disable_unet) that a STOCK KSampler drives, forwarding through the QuantFunc engine's
H3 joint audio+video external denoise SESSION (quantfunc_denoise_begin + quantfunc_denoise_step_multi
+ finalize/end — the D-class AV step added for H3).

WHY H3 is DIFFERENT from the LTX/WAN seams (fork blueprint + engine seam-v2 grounding):
- JOINT AUDIO+VIDEO. The KSampler latent is a comfy.nested_tensor.NestedTensor (video, audio):
  video = [B,24,T_lat,H//16,W//16] (5D, spatial /16), audio = [B,32,2,audio_t]. Each per-step model
  call steps BOTH lanes via quantfunc_denoise_step_multi (base = the VIDEO lane, validated identically
  to a plain denoise_step; the audio lane rides the SAME video sigma).
- 5D VIDEO LATENT, NOT packed. Unlike LTX ([1,N,128] tokens), the H3 engine denoise_step takes the raw
  5D video latent [B,24,T,H,W] (dims=[B,24,T,H,W]); the engine does the H3 patchify internally.
- PACKED AUDIO ROWS. audio_latent_in is the driver's carried audio latent packed channel-major to
  [1, K*T, 32] (pack_audio); the velocity comes back the same way and is unpacked to [B,32,2,T].
- ModelSamplingAV. The pack is a single-schedule flow whose audio target is scaled by audio_scale
  (= shift/audio_shift). The sampler carries the audio SCALED onto the video schedule; the engine
  undoes that scale before the forward and re-converts the returned audio velocity (C-API contract).
- NO CONNECTOR. H3 conditions DIRECTLY on Qwen3-VL hidden states (c_crossattn) — no LTX-style
  video_embeddings_connector.

SCOPE (2026-08-22, user goal "H3 支持图/视频/音频输入,输出视频带音频"): t2va + fl2va keyframes +
ref2va references ALL ride the begin options av_conds bridge (_begin forwards kind/latent/audio_latent
verbatim; the engine seam binds spec.av_keyframes / spec.av_refs). Reference kinds: image (single-frame
latent), audio (soundtrack, must pair with a visual ref — official rule), video / video_audio
(MULTI-FRAME latent blocks — engine buildRef2VABlocksDenoiseInputs, the official _video_grid cursor
math, parity+official-value death rules in test_minimax_h3_pack.cpp).
"""
import os
import ctypes
import time
import json
import math

import torch

import comfy.model_base
import comfy.conds
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import comfy.nested_tensor

from . import qf_engine as qfe
from . import qf_modelpatcher as qfmp
from .qf_modelpatcher import (_qf_dtype, _QFStub,
                              _interrupt_poll_end_session_on_raise,
                              QFSessionModelMixin)


# ── H3 geometry constants (comfy comfy_extras/nodes_minimax_h3.py + ldm/minimax/model.py) ──
_H3_SPATIAL = 16          # video latent -> pixels (width = W_lat * 16)
_H3_FPS = 24.0            # the H3 frame grid is defined at 24 fps (comfy_extras/nodes_minimax_h3.FPS)


def _h3_frames_from_latent_t(latent_t):
    """INVERT comfy's own video_latent_t (nodes_minimax_h3): latent_t = 2 for fc<=5, else
    ((fc-5)//17)*5 + 2 on the 17k+5 frame grid. Ask comfy's module when importable so an
    upstream grid change cannot silently drift this seam; fall back to the closed form."""
    lt = int(latent_t)
    try:
        from comfy_extras import nodes_minimax_h3 as _h3n
        fc = 5
        while _h3n.video_latent_t(fc) < lt and fc < 100000:
            fc += 17
        if _h3n.video_latent_t(fc) == lt:
            return fc
    except Exception:  # noqa: BLE001 - fall through to the closed form
        pass
    return 5 if lt <= 2 else (lt - 2) // 5 * 17 + 5
_H3_VIDEO_CHANNELS = 24   # video latent channels
_H3_AUDIO_CHANNELS = 32   # audio_latents_dim (AutoencoderKLMiniMaxH3Audio) - kMmh3AudioLatentDim
_H3_AUDIO_STEREO = 2      # K (stereo)

# The official audio pack/unpack + video patchify/unpatchify — import from comfy so a ComfyUI upgrade
# can't drift us (the engine's external-denoise seam mirrors these exact transforms).
from comfy.ldm.minimax.model import pack_audio, unpack_audio, patchify_video, unpatchify_video


class QFH3Model(QFSessionModelMixin, comfy.model_base.MiniMaxH3):
    """MiniMax-H3 svdq joint-AV pipeline exposed as a native comfy MODEL (native-KSampler seam), t2va."""

    # H3's own loader dials, carried across a LoRA rebuild like the mixin's (see QFSessionModelMixin._SESSION_DIALS).
    _SESSION_DIALS = QFSessionModelMixin._SESSION_DIALS + ("_audio_enhance", "_allow_partial_denoise")

    def __init__(self, model_config, engine, device=None):
        super().__init__(model_config, device=device)     # disable_unet honored in BaseModel.__init__
        self.diffusion_model = _QFStub()
        self._qf = engine
        self._num_steps = 0               # DERIVED per run from sample_sigmas (len-1) at _begin
        self._num_frames = 0              # DERIVED per run from the video latent's T (see _derive_geometry)
        self._fps = _H3_FPS               # the H3 grid is defined AT 24 fps (comfy nodes_minimax_h3.FPS)
        self._audio_enhance = False       # [audio_enhance] top up extra audio-only steps to total 16 (see _begin)
        self._allow_partial_denoise = False  # explicit workflow opt-in; full-range remains the safe default
        # The AV flow shifts come from the model_sampling object — the stock
        # ModelSamplingMiniMaxH3 (MiniMaxH3SigmaShift) patches it, and model_config supplies the
        # defaults otherwise. They are read at _begin (getattr(ms, "shift"/"audio_shift")), so this
        # loader carries NO shift widgets (official-workflow shape: that node owns them).
        self._step_i = 0
        self._sess_denoise = 0
        self._out_video = None            # reused velocity_out buffer [1,24,T,H,W]
        self._out_audio = None            # reused audio velocity_out buffer [1, K*T, 32]
        self._max_ctx_seq = 0
        # [step-cache-key] symbolic per-conditioning cfg_context_key (_CtxKeyAssigner, engine lighting CLAUDE.md #B3;
        # was constant 0 = kNoCtxKey — the merged the step cache session gate force-computes at
        # key=0, silently disabling EC on this seam). fl2va is single-branch today; the
        # uuid-derived key stays correct if a second cond branch ever appears.
        self._ctx_key_assigner = qfmp._CtxKeyAssigner()

    def set_audio_enhance(self, on):
        """[enhance switch, user 2026-09-19] the ONE audio-quality switch (H3). ON → the engine refines the audio
        after the video denoise; how many extra sub-steps, over what context, and when it must not run (CFG,
        no audio lane, already-long schedules) is ENGINE law behind the begin option `audio_enhance` — this
        plugin carries no number for it. OFF (default) → byte-identical to no knob. Runtime session knob."""
        self._audio_enhance = bool(on)

    def set_allow_partial_denoise(self, on):
        """Allow split/trimmed sampler schedules for explicit double-sampling workflows.

        The external H3 seam receives the actual sigma, step index, and stage-local total step
        count on every call.  Keep the historical full-range refusal as the default because a
        partial schedule is only meaningful when the workflow deliberately manages both stages.
        """
        self._allow_partial_denoise = bool(on)

    # memory_required: NO override — the base QFSessionModelMixin reports the ENGINE's own measured working set
    # (quantfunc_vram_need_bytes) for every family. The former H3-only heuristic here (2026-08-24 "brim fix":
    # 1 KB/latent-elem + 1.5 GB floor, stacked on top of the base) would now DOUBLE-COUNT the real number
    # (self-CR P-1 on 1d51182: ~5.8 GB of phantom need at 800x768 B=2 on top of the measured 3.6 GB) and push
    # comfy's `need × 1.5 < free` cond-batching rule the wrong way; the measured number covers what it estimated.

    # Conditioning keys this external session does NOT consume — a wired node feeding one would be
    # SILENTLY dropped -> plausible-but-wrong AV. FAIL LOUD (mirror QFLTXModel's defensive superset).
    # Defensive SUPERSET (wan/LTX precedent): every key comfy's MiniMaxH3.extra_conds + the
    # BaseModel callees can consume that this seam does NOT implement. The noise-aug pair and
    # cross_attn_controlnet have NO producer node in this ComfyUI (measured: read-only in
    # model_base), so guarding them cannot break the stock graph — it makes a FUTURE producer
    # fail loud instead of silently changing the conditioning (comfy's own MiniMaxH3 puts the
    # noise-aug values into its payload; this seam builds its own payload and would drop them).
    _ENGINE_IGNORED_COND_KEYS = ("denoise_mask", "concat_mask", "noise_concat",
                                 "concat_latent_image", "cross_attn_controlnet",
                                 "minimax_visual_cond_noise_aug", "minimax_audio_cond_noise_aug")

    def extra_conds(self, **kwargs):
        # RUN-START clean slate FIRST (session lifecycle) — before the loud-fails, so a rejected
        # bad-wiring requeue still closes a session stranded by a prior Interrupt.
        was_open, ok = self._qf.end_session_if_open()
        # Retention (2026-08-24): even a REFUSED close (previous run's step still draining
        # engine-side) must NOT let this run silently REUSE that session — force the first
        # _apply_model through _begin (whose materialize-first close + busy-retry recovers
        # correctly). Cleared at the gate; re-armed every run start.
        self._qf_needs_begin = True
        if was_open:
            qfe.info("[qf_native] H3: closed a pre-existing session at run start (prior run interrupted); "
                  f"end ok={ok}", flush=True)
        # fl2va keyframes / ref2va references / #633 token tags: VALIDATED here, then passed
        # per-cond-group through the conditioning as ONE payload (comfy.conds.CONDConstant —
        # the OFFICIAL model_base.MiniMaxH3 mechanism, key "minimax_payload"). NO cross-call
        # self-stash: a stash written by one cond group was silently clobbered by a sibling
        # group without the key (MEASURED: ref2av refs dropped -> plain t2va with DONE success —
        # the silent-drop class this loader exists to refuse).
        payload = {}
        kf = kwargs.get("minimax_keyframes")
        if kf is not None:
            for e in kf:
                lat = e.get("latent")
                if lat is None or getattr(lat, "ndim", 0) != 5:
                    raise RuntimeError(
                        "qf_native H3: minimax_keyframes entry lacks a 5-D pre-encoded latent "
                        f"(got {type(lat).__name__}{getattr(lat, 'shape', '')}) - wire the vae into "
                        "MiniMaxH3ImageToVideo so the keyframe is VAE-encoded.")
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
                            "qf_native H3: minimax_refs image entry must be single-frame "
                            f"[1,24,1,h,w] (got T={int(lat.shape[2])})")
                    if kind in ("video", "video_audio") and int(lat.shape[2]) < 1:
                        raise RuntimeError("qf_native H3: minimax_refs video entry has no frames")
                if kind in ("audio", "video_audio"):
                    alat = r.get("audio_latent")
                    if alat is None or getattr(alat, "ndim", 0) != 4:
                        raise RuntimeError(
                            f"qf_native H3: minimax_refs {kind} entry lacks a 4-D audio latent")
                if kind not in ("image", "audio", "video", "video_audio"):
                    raise RuntimeError(f"qf_native H3: unknown minimax_refs kind '{kind}'")
            payload["refs"] = refs
        tags = kwargs.get("minimax_token_tags")
        if tags is not None:
            payload["text_token_tags"] = tags
        for _k in self._ENGINE_IGNORED_COND_KEYS:
            if kwargs.get(_k) is not None:
                raise RuntimeError(
                    f"qf_native H3: '{_k}' conditioning is wired, but the QuantFunc H3 native session runs "
                    f"its OWN joint-AV denoise from the loader widgets + the prompt hidden states - it cannot "
                    f"consume comfy's '{_k}', which would be silently ignored. Remove the node feeding it.")
        out = {}
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
            # size the begin context maxima to the LARGEST cond group (pos+neg reach the engine as
            # SEPARATE B==1 steps against ONE session; _begin runs on the first group only).
            self._max_ctx_seq = max(self._max_ctx_seq, int(cross_attn.shape[1]))
        if payload:
            out["minimax_payload"] = comfy.conds.CONDConstant(payload)
        return out

    def scale_latent_inpaint(self, *args, **kwargs):
        raise RuntimeError(
            "qf_native H3: a denoise/inpaint mask (SetLatentNoiseMask) is wired, but the QuantFunc H3 "
            "native session does not support masked inpainting. Remove the mask node.")

    def _derive_geometry(self, x_video, transformer_options):
        """DERIVE the session geometry from the graph (official-loader shape — the loader has no
        geometry/step/shift widgets). x_video=[1,24,T,H,W] from the stock empty-AV-latent /
        MiniMaxH3ImageToVideo; the step count from the sampler's own sigma schedule."""
        self._num_frames = _h3_frames_from_latent_t(int(x_video.shape[2]))
        sigmas = transformer_options.get("sample_sigmas") if isinstance(transformer_options, dict) else None
        if sigmas is None or len(sigmas) < 2:
            raise RuntimeError(
                "qf_native H3: the sampler did not publish a sigma schedule "
                "(transformer_options['sample_sigmas']) - the engine session needs the step count. "
                "Use a stock KSampler / SamplerCustom on this model.")
        self._num_steps = len(sigmas) - 1
        try:
            s_first, s_last = float(sigmas[0]), float(sigmas[-1])
        except Exception:  # noqa: BLE001
            return
        ms = getattr(self, "model_sampling", None)
        s_max = float(getattr(ms, "sigma_max", s_first)) if ms is not None else s_first
        partial = s_last > 1e-3 or (s_max > 0 and s_first < 0.98 * s_max)
        if partial and not self._allow_partial_denoise:
            raise RuntimeError(
                f"qf_native H3: partial / trimmed denoise is disabled (sigmas run "
                f"{s_first:.4f}->{s_last:.4f}, full range would be {s_max:.4f}->0). Enable "
                f"'allow_partial_denoise' on QuantFuncH3Loader only for an intentional split-sigma "
                f"or double-sampling workflow; otherwise use one full-range sampler.")
        if partial:
            qfe.info(f"[qf_native] H3: partial denoise explicitly enabled: "
                  f"{s_first:.4f}->{s_last:.4f}, stage_steps={self._num_steps}", flush=True)

    def _begin(self, x_video, x_audio, vemb, av_payload=None):
        """Open the joint-AV external denoise session. x_video=[1,24,T,H,W], x_audio=[1,32,2,audio_t],
        vemb=[1,S,D] Qwen3-VL hidden states. audio_dims [B,C,K,T] + the AV sigma shifts ride options_json.
        CFG handling for audio_enhance lives in the engine (step-time per-branch key detection, §3.4)."""
        lib = self._qf.lib   # MATERIALIZE FIRST: a fresh deferred wrapper (_real=None) cache-hits
        # the SAME resident engine an interrupted run may have left with an OPEN session; ending
        # before materializing was a no-op on that wrapper -> begin hit "pipeline busy" until the
        # 5-min engine watchdog (measured 2026-08-24). Post-materialize, the close is real.
        self._qf.end_session_if_open()
        self._ctx_key_assigner.reset()   # per-generation uuid->key numbering (no cross-gen leak)
        bpx = qfe.DenoiseBeginParams()
        ctypes.memset(ctypes.byref(bpx), 0, ctypes.sizeof(bpx))
        bpx.struct_size = ctypes.sizeof(bpx)
        bpx.width = int(x_video.shape[-1]) * _H3_SPATIAL
        bpx.height = int(x_video.shape[-2]) * _H3_SPATIAL
        bpx.num_steps = self._num_steps
        _max_seq = max(self._max_ctx_seq, int(vemb.shape[1]))
        bpx.max_context_dims = (ctypes.c_int * 3)(int(vemb.shape[0]), _max_seq, int(vemb.shape[2]))
        self._max_ctx_seq = 0
        bpx.max_pooled_dims = (ctypes.c_int * 2)(0, 0)     # H3 takes no pooled cond
        bpx.cond_dtype = _qf_dtype(vemb.dtype)
        audio_dims = [int(x_audio.shape[0]), int(x_audio.shape[1]),
                      int(x_audio.shape[2]), int(x_audio.shape[3])]   # [B,32,2,audio_t]
        ms = self.model_sampling
        # The AV flow shifts come from the model_sampling object the graph installed — the stock
        # ModelSamplingMiniMaxH3 (MiniMaxH3SigmaShift) add_object_patch'es a ModelSamplingAV, and
        # model_config supplies an AV default when no node is wired. A NON-AV sampling object means
        # a generic model-sampling node (ModelSamplingSD3/Flux/…) replaced it and DROPPED the audio
        # schedule: MEASURED — ModelSamplingSD3 on an H3 model yields ModelSamplingAdvanced with no
        # audio_shift, so a silent getattr default would run the audio branch at 3.0 while the user
        # believes they set the schedule. Refuse LOUD instead of rendering a silently mis-scheduled AV.
        if not hasattr(ms, "shift") or not hasattr(ms, "audio_shift"):
            raise RuntimeError(
                f"qf_native H3: the model_sampling object is {type(ms).__name__}, which carries no "
                f"audio_shift - a GENERIC model-sampling node (ModelSamplingSD3 / ModelSamplingFlux / "
                f"similar) replaced MiniMax-H3's ModelSamplingAV and dropped the AUDIO schedule. Use "
                f"the stock ModelSamplingMiniMaxH3 node (shift_video + shift_audio) for H3, or wire no "
                f"sampling node at all to keep the checkpoint defaults.")
        _opts = {
            **self.residency_opts(),                              # [session-knobs] generic knob
            "audio_dims": audio_dims,
            "av_sigma_shift_video": float(ms.shift),
            "av_sigma_shift_audio": float(ms.audio_shift or 3.0),
            "num_frames": self._num_frames,
            "fps": float(self._fps),
        }
        # [enhance switch] the product switch only; the engine computes the extra audio sub-steps from
        # num_steps and disarms itself under CFG / without an audio lane (MiniMaxH3Pipeline). Sent only
        # when ON (OFF = byte-identical to no knob; the raw extra_audio_steps key is never built here).
        if self._audio_enhance:
            _opts["audio_enhance"] = True
        # [fl2va/ref2va bridge] forward the pre-encoded keyframe/reference latents from the
        # per-group conditioning payload (official minimax_payload mechanism) as begin options
        # av_conds: device pointers as HEX STRINGS (borrowed until begin returns — the engine
        # copy-binds; _kf_keep pins the tensors across the call). The session is begin-bound from
        # the FIRST-invoked cond group; with an external CFG the sibling group shares the binding.
        self._kf_keep = []
        conds = []
        pl = av_payload or {}
        # [C2 ordering guard] record what THIS (first-invoked) group binds; _apply_model
        # compares every group's payload against it and fails loud on a mismatch — a
        # LATER group carrying refs the binding lacks would otherwise be silently dropped.
        self._bound_av_counts = (len(pl.get("keyframes") or []), len(pl.get("refs") or []))
        for e in (pl.get("keyframes") or []):
            lat = e["latent"]
            if not lat.is_cuda:
                lat = lat.to(x_video.device)
            lat = lat.contiguous()
            self._kf_keep.append(lat)
            fidx = int(e.get("resolved_frame_index", 0))
            # comfy anchors in PIXEL frames (0 / frame_count-1); the engine begin-bind wants the
            # LATENT anchor: 0 = first, negative = last (it adds T_lat). Others → engine fail-loud.
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
            _opts["av_conds"] = conds
        tags = pl.get("text_token_tags")
        if tags is not None:
            tl = tags.view(-1).tolist() if hasattr(tags, "view") else list(tags)
            ranges, start = [], None
            for i, tv in enumerate(tl):
                if int(tv) == 0 and start is None:
                    start = i
                elif int(tv) != 0 and start is not None:
                    ranges.append([start, i - start]); start = None
            if start is not None:
                ranges.append([start, len(tl) - start])
            if ranges:
                _opts["text_vision_row_ranges"] = ranges
        bpx._opts = json.dumps(_opts).encode()
        bpx.options_json = bpx._opts
        session = ctypes.c_void_p()
        st = lib.quantfunc_denoise_begin(self._qf.pipeline, ctypes.byref(bpx), ctypes.byref(session))
        if st != qfe.QUANTFUNC_OK and "busy" in (qfe.last_err(lib) or ""):
            # ONE bounded recovery (2026-08-24 busy incident): a just-interrupted run's final
            # step may still be draining engine-side — its refused end RETAINED our pointer,
            # so an end+retry can win once the step lands. Non-busy refusals fall through.
            time.sleep(2.0)
            self._qf.end_session_if_open()
            st = lib.quantfunc_denoise_begin(self._qf.pipeline, ctypes.byref(bpx), ctypes.byref(session))
        self._begin_keep = bpx
        if st != qfe.QUANTFUNC_OK:
            raise RuntimeError(f"denoise_begin (H3 t2va) failed: {qfe.last_err(lib)}")
        self._qf.current_session = session
        self._qf.unloaded = False
        self._kf_keep = []          # engine copy-bound the keyframes/refs at begin; release the pins
        self._step_i = 0
        self._sess_denoise = 0
        qfe.info(f"[qf_native] H3 SESSION OPEN handle={session.value:#x} steps={self._num_steps} "
              f"cond={tuple(vemb.shape)} audio_dims={audio_dims} frames={self._num_frames} "
              f"shift={audio_dims and getattr(ms,'shift',None)}/{getattr(ms,'audio_shift',None)}", flush=True)

    # _call_denoise_step_multi: HOISTED to QFSessionModelMixin (qf_modelpatcher.py) — shared
    # verbatim with QFLTXAVModel (LTX-2.5 joint AV), the second AV consumer.

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        sigma = t
        if c_crossattn is None:
            raise RuntimeError("qf_native H3: no c_crossattn cond - wire the H3 CLIP (MiniMaxH3ImageToVideo "
                               "prompt) into positive/negative.")
        if control is not None:
            raise RuntimeError("qf_native H3: a ControlNet is wired but this seam does not consume comfy "
                               "control hints - remove it.")
        # The AV latent reaches _apply_model in ONE OF TWO forms (comfy.model_base.MiniMaxH3
        # ._scale_audio_slice handles both): (a) a nested view (comfy.nested_tensor.NestedTensor /
        # torch nested, .is_nested + .unbind()), or (b) a FLAT PACKED plain tensor [B,1,N] where
        # N = video_flat | audio_flat (video_flat = prod(latent_shapes[0][1:]), row-major). MEASURED
        # the sampler hands _apply_model the PACKED form. Split via self.latent_shapes (comfy sets it).
        _nested = getattr(x, "is_nested", False)
        if _nested:
            streams = x.unbind()
            x_video, x_audio = streams[0], streams[1]        # [B,24,T,H,W], [B,32,2,audio_t]
        else:
            ls = getattr(self, "latent_shapes", None)
            if not ls or len(ls) < 2:
                raise RuntimeError(f"qf_native H3: packed latent (type={type(x).__name__} "
                                   f"shape={tuple(x.shape)}) but model.latent_shapes is unset - cannot unpack.")
            n = int(math.prod(ls[0][1:]))
            xf = x.reshape(int(x.shape[0]), -1)              # [B, N]
            x_video = xf[:, :n].reshape(list(ls[0]))         # [B,24,T,H,W]
            x_audio = xf[:, n:].reshape(list(ls[1]))         # [B,32,2,audio_t]
        dev = x_video.device
        B = int(x_video.shape[0])
        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        # [step-cache-key] comfy's per-conditioning uuids — symbolic-key source (_CtxKeyAssigner).
        cuuids = transformer_options.get("uuids") if isinstance(transformer_options, dict) else None
        if B > 1 and (cou is None or len(cou) != B):
            raise RuntimeError(f"qf_native H3: engine forward is B==1 per cond group but got batch={B} "
                               f"with cond_or_uncond={cou} - batch_size>1 latents are not supported")
        vemb = c_crossattn.to(dev, dtype=torch.bfloat16).contiguous()
        if self._qf.current_session is None or getattr(self, "_qf_needs_begin", False):
            self._qf_needs_begin = False
            self._derive_geometry(x_video, transformer_options)   # session geometry from the GRAPH
            # shared black-video guard (see qf_modelpatcher.refuse_all_zero_initial_latent);
            # H3 is joint AV — the VIDEO lane's zero-latent is the same int4 NaN factory.
            qfmp.refuse_all_zero_initial_latent(x_video, "H3")
            self._begin(x_video[0:1].contiguous(), x_audio[0:1].contiguous(), vemb[0:1].contiguous(),
                        av_payload=kwargs.get("minimax_payload"))
        # [C2 ordering guard] the engine binds av_conds SESSION-WIDE from the FIRST-invoked cond
        # group (_begin above). ComfyUI does not guarantee pos-before-neg invocation order, so a
        # group whose payload DIFFERS from the bound one would have its keyframes/refs silently
        # ignored — fail loud instead. (ConditioningZeroOut preserves the payload constants, so
        # a stock pos+zeroed-neg CFG pair matches; a genuinely divergent wiring is a user error.)
        _pl_now = kwargs.get("minimax_payload") or {}
        _now_counts = (len(_pl_now.get("keyframes") or []), len(_pl_now.get("refs") or []))
        if _now_counts != getattr(self, "_bound_av_counts", _now_counts):
            raise RuntimeError(
                f"qf_native H3: this cond group carries (keyframes, refs)={_now_counts} but the "
                f"session was begin-bound from the first-invoked group with {self._bound_av_counts} "
                "- av_conds bind session-wide. Make every cond group carry the same reference/"
                "keyframe set (attach them on the positive and derive the negative via "
                "ConditioningZeroOut), or run without CFG.")

        _, Cv, Tv, Hv, Wv = x_video.shape
        Ta = int(x_audio.shape[-1])
        n_rows = _H3_AUDIO_STEREO * Ta                       # pack_audio: ch*T rows
        # the engine session denoises PATCHIFIED video tokens [1, Ntok, C*pt*ph*pw] (2x2 spatial, FP16),
        # NOT the raw 5D latent — patchify_video/unpatchify_video bridge it (mirrors the official model).
        ntok = Tv * (Hv // 2) * (Wv // 2)
        pch = Cv * 4                                          # c*pt*ph*pw = 24*1*2*2 = 96
        if self._out_video is None or self._out_video.shape != (1, ntok, pch):
            self._out_video = torch.empty((1, ntok, pch), dtype=torch.float32, device=dev)  # engine H3 latent = FP32
        if self._out_audio is None or self._out_audio.shape != (1, n_rows, _H3_AUDIO_CHANNELS):
            self._out_audio = torch.empty((1, n_rows, _H3_AUDIO_CHANNELS), dtype=torch.float32, device=dev)

        sig_all = sigma.reshape(-1) if torch.is_tensor(sigma) else None
        step_index = self._sigma_step_index(sigma, sig_all, transformer_options)
        audio_scale = float(getattr(self.model_sampling, "audio_scale", 1.0))
        out_video = torch.empty_like(x_video)
        out_audio = torch.empty_like(x_audio)
        for i in range(B):
            _interrupt_poll_end_session_on_raise(self._qf)
            # video: raw 5D [1,24,T,H,W] -> patchified tokens [1, Ntok, 96] FP32 (the session geometry+dtype)
            vid_tok = patchify_video(x_video[i:i + 1]).to(torch.float32).contiguous().reshape(1, ntok, pch)
            # pack the driver's carried audio [1,32,2,T] -> [1, K*T, 32] channel-major rows (fp32)
            xi_audio = pack_audio(x_audio[i:i + 1].float()).unsqueeze(0).contiguous()   # [1, K*T, 32]
            vi = vemb[i:i + 1].contiguous()
            sig_i = float(sig_all[i].item()) if (sig_all is not None and sig_all.numel() >= B) else \
                (float(sig_all[0].item()) if sig_all is not None else float(sigma))
            # ── video (base) lane ──
            p = qfe.DenoiseStepParams()
            ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
            p.struct_size = ctypes.sizeof(p)
            p.latent_in = vid_tok.data_ptr()
            p.velocity_out = self._out_video.data_ptr()
            p.velocity_out_capacity = self._out_video.numel() * self._out_video.element_size()
            p.dims = (ctypes.c_int * 5)(1, ntok, pch, 0, 0)         # patchified tokens [1, Ntok, 96]
            p.dtype = _qf_dtype(vid_tok.dtype)                      # FP16
            p.sigma = sig_i
            p.step_index = step_index
            p.total_steps = self._num_steps
            p.context = vi.data_ptr()
            p.context_dims = (ctypes.c_int * 3)(*vi.shape)
            p.context_dtype = _qf_dtype(vi.dtype)
            # [step-cache-key] symbolic key from comfy's per-conditioning uuid (was constant 0;
            # the step-cache session gate force-computes at key=0 — see __init__ note).
            cuid = cuuids[i] if (cuuids is not None and i < len(cuuids)) else None
            p.cfg_context_key = self._ctx_key_assigner.key(cuid)
            # ── audio lane (rides the same video sigma) ──
            mp = qfe.DenoiseStepMultiParams()
            ctypes.memset(ctypes.byref(mp), 0, ctypes.sizeof(mp))
            mp.struct_size = ctypes.sizeof(mp)
            mp.base = p
            mp.audio_latent_in = xi_audio.data_ptr()
            mp.audio_velocity_out = self._out_audio.data_ptr()
            mp.audio_velocity_out_capacity = self._out_audio.numel() * self._out_audio.element_size()
            mp.audio_dims = (ctypes.c_int * 4)(int(x_audio.shape[0]), _H3_AUDIO_CHANNELS,
                                               _H3_AUDIO_STEREO, Ta)
            mp.audio_dtype = _qf_dtype(xi_audio.dtype)
            mp.audio_scale = audio_scale
            self._call_denoise_step_multi(mp, f"H3 denoise_step_multi[step={step_index},group={i}]")
            # video velocity tokens [1,Ntok,96] (engine already negated) -> raw 5D [1,24,T,H,W]
            out_video[i:i + 1] = unpatchify_video(self._out_video.reshape(ntok, pch),
                                                  Tv, Hv // 2, Wv // 2, Cv).to(out_video.dtype)
            # audio velocity: unpack [1,K*T,32] -> [1,32,2,T]
            out_audio[i:i + 1] = unpack_audio(self._out_audio[0]).to(out_audio.dtype)
            self._qf.step_count += 1
            self._sess_denoise += 1
        self._step_i += 1
        self._qf.sampler_step_count += 1
        # calculate_denoised on the SAME (video) sigma — the pack is a single-schedule flow; the engine
        # re-converted the audio velocity onto the carried (video-schedule) variable. Return in the SAME
        # form the sampler handed us (packed [B,1,N] or nested), so process_latent_out's audio-unscale fits.
        if _nested:
            den_v = self.model_sampling.calculate_denoised(sigma, out_video.float(), x_video)
            den_a = self.model_sampling.calculate_denoised(sigma, out_audio.float(), x_audio)
            return comfy.nested_tensor.NestedTensor((den_v, den_a))
        B = int(x.shape[0])
        vel_flat = torch.cat([out_video.reshape(B, -1), out_audio.reshape(B, -1)], dim=1).reshape(x.shape)
        return self.model_sampling.calculate_denoised(sigma, vel_flat.float(), x)

    def process_latent_out(self, latent):
        # close the session (t2va: engine finalize early-returns; keep the sampler's latent untouched).
        if self._qf.current_session is not None:
            try:
                qfe.info(f"[qf_native] H3 SESSION CLOSED after {self._step_i} sampler steps, "
                      f"{self._sess_denoise} denoise_step_multi calls", flush=True)
            finally:
                self._qf.end_session_if_open()
        return super().process_latent_out(latent)


FAMILY = "minimax-h3"


def matches(pipeline_class, transformer_class=""):
    """The ENGINE's own detector, BOTH halves (src/MiniMaxH3Pipeline.cpp
    minimax_h3_pipeline_detect):
        pipeline_class in {"MiniMaxH3ModularPipeline", "MiniMaxH3Pipeline"}
        || transformer_class == "MiniMaxH3Transformer3DModel"."""
    return (str(pipeline_class) in ("MiniMaxH3ModularPipeline", "MiniMaxH3Pipeline")
            or str(transformer_class) == "MiniMaxH3Transformer3DModel")


def register(deps):
    """Return the minimax-h3 family BUILDER. `deps` gives the package-level helpers (engine cache,
    liveness registry) without importing __init__."""
    def build(transformer1_path, bundle_dir=None, lora_entries=()):
        """File-based loading for MiniMax-H3 — the shared staging pattern:
        stage the shipped config bundle (configs/minimax-h3-*/, official configs) + symlink
        the single transformer file; engine create runs denoise_only=True (TE + VAE weights
        skipped — comfy's stock MiniMaxH3 nodes own conditioning/refs and comfy decodes;
        the engine reads the staged configs for session geometry only). No extra weight
        links: unlike ltx2 the H3 external session needs no engine-side connector/projection
        weights (refs arrive as av_conds latents from comfy). H3 svdq is PRE-quantized, so the create is
        MINIMAL: the svdquant metadata carries the layout/precision, and anything on top competes and
        mis-resolves (the LTX minimal note)."""
        model_dir = qfmp.stage_denoise_only_package(bundle_dir, transformer1_path)
        return qfmp.family_build(
            deps, model_dir, {"denoise_only": True}, comfy.supported_models.MiniMaxH3,
            {"image_model": "minimax_h3", "disable_unet_model_creation": True}, QFH3Model,
            f"[qf_native] loaded QuantFuncNativeLoader (MiniMax-H3 svdq AV) package={os.path.basename(transformer1_path)} "
            f"capacity=native Prepared query (create deferred)")(list(lora_entries))

    return build
