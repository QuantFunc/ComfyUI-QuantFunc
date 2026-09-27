"""qf_native.qf_ltx_modelpatcher — the LTX-2 seam: a comfy ModelPatcher whose `.model` is an
LTXV-shim (disable_unet) that a STOCK KSampler drives, forwarding through the QuantFunc engine's
LTX-2 external denoise SESSION (quantfunc_denoise_begin/step/finalize/end, the SAME generic ABI the
wan seam uses — reused from qf_engine.py).

WHY this is DIFFERENT from the wan seam (design findings, dossier seq-229..234):
- cfg_context_key: seq-229 established the LTX-2 lighting transformer engages NONE of the
  key-trusting step caches (ctx_cache_/cross_kv_cache_/the engine's block-cache state slot ) — historically every step passed
  cfg_context_key=0 (kNoCtxKey). SINCE the step-cache session gate (merge 516a6d78) the key IS
  consumed (one EcEntry per cond branch; key=0 force-computes), so BOTH step loops (t2v + AV) now
  derive uuid-symbolic keys via the shared _CtxKeyAssigner (wan #B3 pattern) — see [step-cache-key]
  comments at the loops.
- PACKED latent. The engine session denoises a packed token latent [1,N,128] (N=F_lat*H_lat*W_lat),
  NOT comfy's 5D [1,128,F,H,W]. seq-230: the mapping is a plain REVERSIBLE reshape+transpose (row-major
  (F,H,W)), the exact inverse of the engine's pack_video_tokens — nothing inferred.

Landing HELD (build+verify only). Only THIS loader node is swapped into an otherwise-STOCK LTX-2.3 T2V
workflow.
"""
import ctypes
import time
import json
import math
import os

import torch

import comfy.model_base
import comfy.conds
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import comfy.nested_tensor

from . import qf_engine as qfe
# Reuse the shared model-layer helpers. NOTE: the engine cache (_get_engine + the liveness
# tracker) lives in __init__.py (which imports THIS file), so those are threaded into register()
# via deps to avoid a circular import — same seam as every family module.
from . import qf_modelpatcher as qfmp
from .qf_modelpatcher import (_qf_dtype, _QFStub,
                              _interrupt_poll_end_session_on_raise,
                              QFSessionModelMixin)


# LTX VAE scale factors (engine kS=32 spatial, kT=8 temporal, kC=128 channels — LTX2VideoPipeline).
_LTX_SPATIAL = 32
_LTX_TEMPORAL = 8
_LTX_DEFAULT_FPS = 25.0   # informational default; the sampler/graph owns real timing


@qfe.console_safe_methods   # an exception leaving it is console-safe (#738)
class QFLTXModel(QFSessionModelMixin, comfy.model_base.LTXV):
    """LTX-2 svdq pipeline exposed as a native comfy MODEL (native-KSampler seam), t2v + i2v.

    i2v is WAN-ALIGNED (user 2026-08-22 "只关注latent"): the model consumes ONLY latents.
    The workflow's official LTXVImgToVideoInplace node VAE-encodes the image, writes it into
    the latent's leading frame(s) and sets noise_mask = 1-strength — all comfy-side, zero
    cond keys. comfy's KSamplerX0Inpaint then blends x against the clean latent per step
    OUTSIDE this model (via the inherited BaseModel.scale_latent_inpaint — deliberately NOT
    overridden), which is exact for this seam because the engine step is STATELESS in x
    (latents in, velocity out; the sampler owns x). No loader image socket, no plugin-side
    VAE, no engine begin_edit."""

    def __init__(self, model_config, engine, connector, device=None, audio_connector=None):
        super().__init__(model_config, device=device)   # disable_unet honored in BaseModel.__init__
        self.diffusion_model = _QFStub()
        self._qf = engine                    # QFEngineHandle (lib + pipeline + open session)
        self._connector = connector          # comfy Embeddings1DConnector (video), weights loaded
        self._audio_connector = audio_connector  # comfy audio_embeddings_connector (2048); None -> video-only 4096
        self._num_steps = 0                  # DERIVED per run from sample_sigmas (len-1) at _begin
        self._num_frames = 0                 # DERIVED per run from the latent: (Tlat-1)*8 + 1
        self._fps = _LTX_DEFAULT_FPS         # informational (rides options_json); LTX conditions on its own
        self._step_i = 0
        self._sess_denoise = 0
        self._out = None                     # reused packed velocity_out buffer [1,N,128]
        # [step-cache-key] symbolic per-conditioning cfg_context_key (_CtxKeyAssigner; engine lighting CLAUDE.md #B3).
        # Historically LTX passed 0 (kNoCtxKey) because it engaged NO key-trusting cache
        # (dossier seq-229) — the merged the step cache session gate NOW trusts the key (its §6.5
        # no-key guard force-computes at key=0, silently disabling EC). uuid-derived keys give
        # each cond branch its own EcEntry (no cross-branch collapse) at zero cost when EC off.
        self._ctx_key_assigner = qfmp._CtxKeyAssigner()
        # Max POST-CONNECTOR seq len across a run's cond groups (pos+neg), accumulated in extra_conds and used
        # to size the engine begin context MAXIMA (see _begin). pos/neg can have
        # DIFFERENT prompt lengths and reach the engine as SEPARATE B==1 step calls against ONE session, while
        # _begin runs on the first group only — so it must be sized to the LARGEST group, not just the first.
        # ★ POST-connector, NOT raw (§6.5 correctness NO-GO): comfy's Embeddings1DConnector does NOT preserve
        # the seq dim — its forward tail-pads with tiled learnable_registers to
        # n_reg*ceil(max(1024, S)/n_reg) (MEASURED: S=1200→1280, 2000→2048, ≤1024→1024 at n_reg=128), so a
        # raw accumulator UNDER-sizes the ceiling whenever a >1024-raw group is not the group that _begins.
        self._max_ctx_seq = 0
        self._post_seq_cache = {}            # raw S -> measured post-connector S (see _post_connector_seq)

    # ── engine-conditioning safety layer (added for the CR conformance NO-GO) ──
    # LTXV.extra_conds (+ the LTXV.concat_cond / BaseModel.extra_conds callees) can consume mask / keyframe /
    # guide / c_concat / controlnet / i2v-concat channels when a corresponding node is wired. This external-
    # session seam runs the engine's OWN t2v schedule from the loader widgets + the connector video_embeds and
    # consumes NONE of them, so a bare _apply_model(**kwargs) would SILENTLY drop them → a plausible-but-wrong
    # video with no warning. FAIL LOUD instead. This is a DEFENSIVE SUPERSET — every
    # consumable key that is neither emitted (cross_attn) nor documented-accepted (attention_mask/frame_rate);
    # its completeness against THIS comfy is machine-checked by tests/reject_list_completeness.py (LTXV entry).
    # denoise_mask is deliberately NOT in this tuple: the Inplace i2v latent route rides it
    # (comfy's sampler consumes it via KSamplerX0Inpaint + the inherited scale_latent_inpaint;
    # it never needs the engine) — see the class docstring + reject_list_completeness.py.
    _ENGINE_IGNORED_COND_KEYS = ("concat_mask", "keyframe_idxs", "guide_attention_entries",
                                 "noise_concat", "cross_attn_controlnet", "concat_latent_image",
                                 # LTX-AV-era cond keys this comfy's LTXAV.extra_conds exposes —
                                 # from LTXVReferenceAudio / the keyframe generators. The native
                                 # session consumes neither; a wired producer must fail loud, not
                                 # be silently dropped.
                                 # ★ NEVER put a closing bracket in a comment INSIDE this tuple:
                                 # the reject-list audit parses it with a  [^ closing-bracket ]*
                                 # character class, so the first one TRUNCATES the parsed set and
                                 # every key after it is reported UNCOVERED. Measured 2026-08-21:
                                 # an aside in brackets hid ref_audio + generated_keyframes.
                                 "ref_audio", "generated_keyframes")

    def extra_conds(self, **kwargs):
        # RUN-START clean slate FIRST (#2 session lifecycle) — BEFORE the loud-fails below, so a rejected
        # bad-wiring requeue still closes a session STRANDED by a prior Interrupt (else a permanently-erroring
        # graph leaks a GPU-resident stranded session across requeues). extra_conds is comfy's ONLY hook that
        # fires at the start of EVERY run even for an EMPTY (denoise=1.0) latent, and comfy CACHES+REUSES this
        # model instance across a requeue — so closing here forces _apply_model to _begin fresh. Idempotent
        # across the pos+neg extra_conds calls of a run (the 2nd finds nothing open → no-op).
        was_open, ok = self._qf.end_session_if_open()
        # Retention (2026-08-24): even a REFUSED close (previous run's step still draining
        # engine-side) must NOT let this run silently REUSE that session — force the first
        # _apply_model through _begin (whose materialize-first close + busy-retry recovers
        # correctly). Cleared at the gate; re-armed every run start.
        self._qf_needs_begin = True
        if was_open:
            qfe.info("[qf_native] LTX: closed a pre-existing session at run start "
                  f"(prior run interrupted/uncleaned); end ok={ok}", flush=True)
        # masked-latent i2v (comfy LTXVImgToVideoInplace): capture the run's denoise_mask
        # as PER-LATENT-FRAME sigma scales for the engine session. This is the comfy
        # LTXV.process_timestep mechanism (conditioned tokens run at mask*t) — without it
        # the engine reads the outer-blended near-clean conditioned frames at the GLOBAL
        # sigma and emits corrupt velocities (measured: red-speckle degeneration, run
        # b16cbc7f). Cleared per run here; pos+neg extra_conds set identical values.
        self._pending_frame_t_scale = None
        _dm = kwargs.get("denoise_mask")
        if _dm is not None:
            _scales = self._frame_scales_from_mask(_dm)
            # all-1.0 = no conditioning anywhere -> omit (keeps the engine's t2v scalar
            # path byte-identical instead of a mathematically-equal per-token pass).
            self._pending_frame_t_scale = None if all(v >= 0.9999 for v in _scales) else _scales
        # ★ CLOSE THE SILENT-DISCARD CLASS: a wired keyframe / guide-attention / concat node reaches here
        # as one of these keys and would be dropped — fail loud. (denoise_mask is NOT here: the Inplace
        # latent route rides it and comfy's sampler consumes it entirely outside this model.)
        for _k in self._ENGINE_IGNORED_COND_KEYS:
            if kwargs.get(_k) is not None:
                raise RuntimeError(
                    f"qf_native LTX: '{_k}' conditioning is wired, but the QuantFunc LTX native session "
                    f"consumes only latents + the connector video_embeds - it cannot consume comfy's "
                    f"'{_k}', which would be silently ignored. Remove the node feeding it. For "
                    f"image-to-video use LTXVImgToVideoInplace on the LATENT path (its noise_mask is "
                    f"applied by comfy's sampler and IS supported); LTXVAddGuide / keyframe nodes are not.")
        # MIRROR ONLY c_crossattn (the engine's connector consumes it), building `out` BY HAND —
        # we deliberately do NOT call super().extra_conds. Reason: emit EXACTLY the one channel the engine takes,
        # and don't re-enter comfy's cond-building (LTXV.extra_conds + its concat_cond/encode_adm callees), which
        # would re-populate the very keys we reject. (LTXV does not override concat_cond — the effective
        # BaseModel.concat_cond is concat_keys-gated and never touches diffusion_model — so calling super() would
        # not crash here; building by hand keeps the emitted set minimal and explicit.) frame_rate is unused by this seam; attention_mask IS consumed (the connector mask — 19B mask fix).
        out = {}
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
            # [19B mask fix] forward the clip's attention_mask: gemma LEFT-pads, so the connector MUST
            # mask the pad rows (the engine's own internal path builds a content-tail mask before its
            # connector forward — "text mask active=N/S"). The old all-ones simplification aggregated
            # PAD rows into the registers → garbage conditioning → structured-mush output (MEASURED:
            # stock comfy with the SAME TE chain coherent, this seam mush, fixed by this mask).
            am = kwargs.get("attention_mask", None)
            if am is not None:
                out["attention_mask"] = comfy.conds.CONDRegular(am)
            # Accumulate the max POST-CONNECTOR seq len across this run's cond groups (pos+neg BOTH call
            # extra_conds before sampling). ★ POST, not raw (§6.5 correctness NO-GO — an earlier revision
            # claimed "the connector PRESERVES the seq dim"; FALSE: comfy's Embeddings1DConnector tail-pads
            # to n_reg*ceil(max(1024, S)/n_reg), so raw 1200 becomes vemb 1280). _begin sizes the session
            # context maxima from this accumulator vs the FIRST group's actual vemb; a raw accumulator
            # under-sizes it whenever a >1024-raw group (a long negative) is NOT the group that _begins —
            # that group's step then exceeds the begin maxima and the engine LOUD-REJECTS
            # (quantfunc_api.cpp "positive and per-dim <= the begin maxima"; correctness, not silent).
            # _post_connector_seq MEASURES the length from the real loaded connector (cached).
            self._max_ctx_seq = max(self._max_ctx_seq, self._post_connector_seq(int(cross_attn.shape[1])))
        return out

    def _frame_scales_from_mask(self, dm):
        """denoise_mask -> per-LATENT-frame sigma scales (comfy process_timestep parity).

        Two shapes reach this seam (MEASURED):
        - video-only 5D channel-repeated [B,C,F,H,W] (plain latent path; the mask is
          channel-uniform so channel 0 is representative);
        - joint-AV FLAT [B,1,total] — comfy packs the nested (video, audio) latent and
          flattens the mask alike (total = video_elems + audio_elems; measured
          1,818,368 = 128*16*22*40 + 8*126*16). Split via self.latent_shapes.
        Inplace masks are FRAME-uniform; a mask varying within a frame (spatial inpaint)
        or masking the AUDIO lane is refused loud — the engine scales per FRAME only."""
        import math as _m
        m = dm
        if m.ndim == 3:
            ls = getattr(self, "latent_shapes", None)
            if not ls or len(ls) < 1:
                raise RuntimeError(
                    "qf_native LTX: flat denoise_mask but the model carries no "
                    "latent_shapes to split it - unsupported mask source.")
            n_video = int(_m.prod(ls[0][1:]))
            flat = m.reshape(-1)
            if int(flat.shape[0]) < n_video:
                raise RuntimeError(
                    f"qf_native LTX: flat denoise_mask has {int(flat.shape[0])} elements "
                    f"but the video latent needs {n_video} - geometry mismatch.")
            audio_part = flat[n_video:]
            if audio_part.numel() and float(audio_part.min()) < 0.9999:
                raise RuntimeError(
                    "qf_native LTX: the denoise_mask masks the AUDIO lane - audio "
                    "conditioning is not supported on this seam (video Inplace only).")
            m = flat[:n_video].reshape(list(ls[0]))[0]      # [C,F,H,W]
            m = m.movedim(1, 0)                             # [F,C,H,W]
        elif m.ndim == 5:
            m = m[0, 0]                                     # [F,H,W]
        else:
            raise RuntimeError(
                f"qf_native LTX: denoise_mask with shape {tuple(dm.shape)} is not a "
                f"[B,C,F,H,W] latent mask nor the packed AV flat form - unsupported "
                f"mask source.")
        per_frame = []
        for f in range(int(m.shape[0])):
            mf = m[f]
            lo, hi = float(mf.min()), float(mf.max())
            if hi - lo > 1e-4:
                raise RuntimeError(
                    "qf_native LTX: denoise_mask varies WITHIN a latent frame - the "
                    "engine session scales the timestep per FRAME (Inplace-style masks "
                    "only); spatial inpaint masks are not supported on this seam.")
            per_frame.append(max(0.0, min(1.0, hi)))
        return per_frame

    # scale_latent_inpaint is deliberately INHERITED (comfy BaseModel), NOT loud-failed (2026-08-22
    # wan-align pivot): comfy's KSamplerX0Inpaint calls it only when a denoise/noise mask is wired
    # (LTXVImgToVideoInplace's frame-0 conditioning, or SetLatentNoiseMask) and blends x against the
    # clean latent per step OUTSIDE _apply_model. That blend is EXACT for this seam because the
    # engine step is stateless in x (latents in, velocity out — the sampler owns x), so masked
    # conditioning needs zero engine involvement. This is the i2v mechanism now.

    def _derive_geometry(self, xin, transformer_options):
        """DERIVE the session geometry from the graph (official-loader shape — the loader has no
        geometry widgets). xin: [B,128,F_lat,H_lat,W_lat]; LTX VAE scale temporal 8 / spatial 32.
        Step count from the sampler's own sigma schedule. PARTIAL / TRIMMED ranges are
        ACCEPTED (two-stage official workflows: stage-A 1.0->0.975 low-res, stage-B
        0.85->0 refine after the x2 latent upsample): the external session is PURELY
        sigma-driven — the engine's session step consumes the driver's sigma each call
        and its session path has ZERO num_steps / step-index consumption (verified by
        source sweep + c5.8 legC bit-identical under a full external Euler drive). The
        old refusal's rationale ("engine runs its OWN internal schedule") described the
        INTERNAL generate_video loop, not this seam. Only pathological schedules
        (fewer than 2 sigmas / non-decreasing) are refused."""
        Tlat = int(xin.shape[2])
        self._num_frames = (Tlat - 1) * _LTX_TEMPORAL + 1
        sigmas = transformer_options.get("sample_sigmas") if isinstance(transformer_options, dict) else None
        if sigmas is None or len(sigmas) < 2:
            raise RuntimeError(
                "qf_native LTX: the sampler did not publish a sigma schedule "
                "(transformer_options['sample_sigmas']) - the engine session needs the step count. "
                "Use a stock KSampler / SamplerCustom on this model.")
        self._num_steps = len(sigmas) - 1
        try:
            s_first, s_last = float(sigmas[0]), float(sigmas[-1])
        except Exception:  # noqa: BLE001 - non-tensor sigmas: keep the count, skip the range check
            return
        if s_first <= s_last:
            raise RuntimeError(
                f"qf_native LTX: sigma schedule must be strictly DECREASING (got "
                f"{s_first:.4f}->{s_last:.4f}) - a non-decreasing schedule would drive the "
                f"session backwards.")

    def _begin(self, x_group, vemb_group):
        """Open the t2v external denoise session. x_group = [1,128,F,H,W] latent; vemb_group =
        [1,S,4096] POST-connector video_embeds. Geometry: engine derives F_lat/H_lat/W_lat from
        num_frames + width/height (spatial 32, temporal 8)."""
        lib = self._qf.lib   # MATERIALIZE FIRST (see qf_h3_modelpatcher._begin: a deferred wrapper
        # no-ops the close while the cached engine still holds an interrupted run's open session)
        self._qf.end_session_if_open()
        self._ctx_key_assigner.reset()   # per-generation uuid->key numbering (no cross-gen leak)
        bpx = qfe.DenoiseBeginParams()
        ctypes.memset(ctypes.byref(bpx), 0, ctypes.sizeof(bpx))
        bpx.struct_size = ctypes.sizeof(bpx)
        # latent is [1, 128, F_lat, H_lat, W_lat]; target pixels = latent * VAE scale (spatial 32).
        bpx.width = int(x_group.shape[-1]) * _LTX_SPATIAL
        bpx.height = int(x_group.shape[-2]) * _LTX_SPATIAL
        bpx.num_steps = self._num_steps
        # Size the context MAXIMA to the LARGEST POST-CONNECTOR vemb seq across this run's cond groups
        # (accumulated in extra_conds for pos+neg via _post_connector_seq — same quantity as this group's
        # vemb.shape[1], so the max compares like with like), NOT just this first group's — a longer
        # different-length negative reaches the engine as a SEPARATE B==1 step against this SAME session and
        # must fit the begin maxima. max(...) with this group is the safe fallback if the accumulator was
        # never populated (cross_attn-less flow). (POST-length fix per §6.5.)
        _max_seq = max(self._max_ctx_seq, int(vemb_group.shape[1]))
        bpx.max_context_dims = (ctypes.c_int * 3)(int(vemb_group.shape[0]), _max_seq,
                                                  int(vemb_group.shape[2]))
        self._max_ctx_seq = 0             # reset for the next run's extra_conds accumulation
        # LTX takes NO pooled cond.
        bpx.max_pooled_dims = (ctypes.c_int * 2)(0, 0)
        bpx.cond_dtype = _qf_dtype(vemb_group.dtype)
        _mask_opt = ({"video_frame_t_scale": self._pending_frame_t_scale}
                     if getattr(self, "_pending_frame_t_scale", None) else {})
        bpx._opts = json.dumps({**{"num_frames": self._num_frames, "fps": float(self._fps)},
                                **self.residency_opts(),
                                **_mask_opt,
                                **self._begin_extra_opts()}).encode()
        bpx.options_json = bpx._opts
        session = ctypes.c_void_p()
        # ONE begin path (wan-align 2026-08-22): the session is ALWAYS a plain t2v-shape begin.
        # i2v conditioning is entirely comfy-side (LTXVImgToVideoInplace latent + noise_mask via
        # KSamplerX0Inpaint) — the engine never sees an image or a cond latent.
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
            raise RuntimeError(f"denoise_begin (LTX) failed: {qfe.last_err(lib)}")
        self._qf.current_session = session
        self._step_i = 0
        self._sess_denoise = 0
        qfe.info(f"[qf_native] LTX SESSION OPEN handle={session.value:#x} steps={self._num_steps} "
              f"cond={tuple(vemb_group.shape)} frames={self._num_frames}", flush=True)


# ── LTX-2.5 joint audio+video (c5.8b) ────────────────────────────────────────────────
# Packed audio latent row width: comfy audio latent [B,8,L,16] ↔ engine rows [1,L,128]
# with d = c*16 + f (engine diffusers _unpack_audio_latents: packed[b,l,d] →
# unpacked[b, d/16, l, d%16]) — i.e. permute(0,2,1,3).reshape.
_LTXAV_AUDIO_CH = 8
_LTXAV_AUDIO_MEL = 16
_LTXAV_AUDIO_PACK = _LTXAV_AUDIO_CH * _LTXAV_AUDIO_MEL   # 128


@qfe.console_safe_methods   # an exception leaving it is console-safe (#738)
class QFLTXAVModel(QFLTXModel):
    """LTX-2.5 JOINT AUDIO+VIDEO through the native session (t2av).

    Differences vs the video-only QFLTXModel base:
    - The comfy latent is the AV pair (video [B,128,F,H,W], audio [B,8,L,16]) — nested or
      sampler-packed flat (split via self.latent_shapes, the H3-measured dual form).
    - NO plugin-side connector: the LTX-2.5 'with-proj' TE emits the UNPROCESSED dual-proj
      concat (extra unprocessed_ltxav_embeds=True) and the ENGINE runs its own checkpoint-
      loaded connector tail (begin av_unprocessed_ctx=true →
      LTX2TextConnectors::forwardProjected) — one connector implementation, the
      7-tier-verified internal one.
    - Every step drives BOTH lanes via quantfunc_denoise_step_multi at the SAME sigma (the
      LTX co-denoise contract — no H3-style shift/carry; audio_scale is inert at 1.0).
    BASE-CLASS note: deliberately QFLTXModel-only (NOT + comfy.model_base.LTXAV) —
    model_base.{LTXV,LTXAV} are BaseModel SIBLINGS whose __init__ super()-chains pass
    different unet_model kwargs, so a diamond MRO crashes ("unexpected keyword argument
    'unet_model'", MEASURED first run). Nothing needs the LTXAV model class here: the
    AV latent_format + memory factor ride the supported_models.LTXAV CONFIG the loader
    passes; comfy's nested-latent packing is model-agnostic (samplers.py pack_latents +
    inner_model.latent_shapes); the only comfy isinstance on model_base.LTXAV
    (lora.py:371) tests (LTXV, LTXAV) — satisfied via the LTXV base."""

    def __init__(self, model_config, engine, device=None):
        # i2v-AV (wan-align 2026-08-22 "只关注latent"): NO image/vae plumbing here — the
        # workflow's LTXVImgToVideoInplace conditions the VIDEO half of the joint latent
        # (encode + frame-0 write + noise_mask), and comfy's sampler applies the mask
        # outside the model. The session begin is identical to t2av.
        QFLTXModel.__init__(self, model_config, engine, connector=None,
                            device=device, audio_connector=None)
        self._out_audio = None            # reused packed audio velocity buffer [1,L,128] fp32
        self._ctx_unprocessed = False     # set per run by extra_conds from the TE's marker
        self._audio_rows = 0              # L, set from the actual audio latent at _apply_model

    def _post_connector_seq(self, raw_s):
        # The ENGINE connector tail preserves S (registers REPLACE pad rows — no comfy-style
        # tail-pad), and this seam sizes the begin maxima on the RAW ctx it actually sends.
        return int(raw_s)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        # The LTXAV 'with-proj' TE marks its output UNPROCESSED (dual-proj concat, connector
        # NOT run). Required by this seam — checked fail-loud at _begin_extra_opts.
        self._ctx_unprocessed = bool(kwargs.get("unprocessed_ltxav_embeds", False))
        return out

    def _run_connector(self, c_crossattn, attention_mask=None):
        # AV seam: pass the RAW dual-proj concat through — the engine connector tail runs
        # per distinct cond (see the class docstring). attention_mask is unused: the comfy
        # LTXAV TE already TRIMS the left-pad to the active tail before projection
        # (lt.py encode_token_weights `out[:, :, -sum(mask):]`), so every row is valid.
        return c_crossattn

    def _begin_extra_opts(self):
        if not self._ctx_unprocessed:
            raise RuntimeError(
                "qf_native LTX-AV: the wired text encoder did not mark its output "
                "unprocessed_ltxav_embeds - this loader needs the LTX-2.5 'with-proj' TE "
                "(gemma4-12b-with-proj-*.safetensors through the comfy LTXAV clip), whose "
                "connector runs ENGINE-side. A 19B/2.3 connector-file TE chain belongs on "
                "the video-only QuantFuncNativeLoader path.")
        if self._audio_rows <= 0:
            raise RuntimeError("qf_native LTX-AV: audio latent rows unset at begin - wiring error")
        return {"audio_dims": [1, int(self._audio_rows), _LTXAV_AUDIO_PACK],
                "av_unprocessed_ctx": True}

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        return self._apply_model_timed(x, t, c_concat=c_concat, c_crossattn=c_crossattn,
                                       control=control, transformer_options=transformer_options,
                                       **kwargs)

    def _apply_model_timed(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        sigma = t
        if c_crossattn is None:
            raise RuntimeError("qf_native LTX-AV: no c_crossattn cond - wire the LTX-2.5 "
                               "gemma4 'with-proj' TE (CLIPTextEncode -> LTXVConditioning)")
        if control is not None:
            raise RuntimeError("qf_native LTX-AV: a ControlNet is wired, but this seam does "
                               "not consume comfy control hints - remove it.")
        # AV latent arrives nested OR sampler-packed flat (the H3-measured dual form).
        _nested = getattr(x, "is_nested", False)
        if _nested:
            streams = x.unbind()
            x_video, x_audio = streams[0], streams[1]          # [B,128,F,H,W], [B,8,L,16]
        else:
            ls = getattr(self, "latent_shapes", None)
            if not ls or len(ls) < 2:
                raise RuntimeError(f"qf_native LTX-AV: packed latent (type={type(x).__name__} "
                                   f"shape={tuple(getattr(x, 'shape', ()))}) but model.latent_shapes "
                                   "is unset - cannot unpack the AV pair.")
            n = int(math.prod(ls[0][1:]))
            xf = x.reshape(int(x.shape[0]), -1)
            x_video = xf[:, :n].reshape(list(ls[0]))
            x_audio = xf[:, n:].reshape(list(ls[1]))
        if x_audio.ndim != 4 or int(x_audio.shape[1]) != _LTXAV_AUDIO_CH \
                or int(x_audio.shape[3]) != _LTXAV_AUDIO_MEL:
            raise RuntimeError(f"qf_native LTX-AV: audio latent shape {tuple(x_audio.shape)} - "
                               f"expected [B,{_LTXAV_AUDIO_CH},L,{_LTXAV_AUDIO_MEL}] "
                               "(LTXVEmptyLatentAudio / LTXVConcatAVLatent)")
        dev = x_video.device
        xin = x_video.to(torch.bfloat16).contiguous()          # [B,128,F,H,W]
        B = int(xin.shape[0])
        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        if B > 1 and (cou is None or len(cou) != B):
            raise RuntimeError(f"qf_native LTX-AV: engine forward is B==1 per cond group but got "
                               f"batch={B} with cond_or_uncond={cou}")
        self._derive_geometry(xin, transformer_options)
        La = int(x_audio.shape[2])
        self._audio_rows = La
        # RAW dual-proj ctx (no plugin connector — see _run_connector).
        vemb = self._run_connector(c_crossattn, attention_mask=kwargs.get("attention_mask")
                                   ).to(dev, dtype=torch.bfloat16).contiguous()
        if self._qf.current_session is None or getattr(self, "_qf_needs_begin", False):
            self._qf_needs_begin = False
            # shared black-video guard (see qf_modelpatcher.refuse_all_zero_initial_latent).
            qfmp.refuse_all_zero_initial_latent(xin, "LTX-AV")
            self._begin(xin[0:1].contiguous(), vemb[0:1].contiguous())
        _, C, F, H, W = xin.shape
        N = F * H * W
        if self._out is None or self._out.shape != (1, N, C):
            self._out = torch.empty((1, N, C), dtype=xin.dtype, device=dev)
        if self._out_audio is None or self._out_audio.shape != (1, La, _LTXAV_AUDIO_PACK):
            self._out_audio = torch.empty((1, La, _LTXAV_AUDIO_PACK), dtype=torch.float32, device=dev)
        sig_all = sigma.reshape(-1) if torch.is_tensor(sigma) else None
        step_index = self._sigma_step_index(sigma, sig_all, transformer_options)
        # [step-cache-key] the AV class has its OWN step loop (this one), so the base t2v
        # loop's uuid-derived key never runs here — extract uuids for THIS loop too (the
        # e1_w2-measured miss: AV sessions kept passing key=0 → EC silently disabled).
        cuuids = transformer_options.get("uuids") if isinstance(transformer_options, dict) else None
        out5d = torch.empty_like(xin)
        out_audio = torch.empty_like(x_audio)
        for i in range(B):
            _interrupt_poll_end_session_on_raise(self._qf)
            xi = xin[i:i + 1].contiguous()
            tokens = xi.reshape(1, C, N).transpose(1, 2).contiguous()          # [1,N,128]
            # pack audio [1,8,L,16] → rows [1,L,128] (d = c*16 + f — engine unpack order)
            arows = (x_audio[i:i + 1].float().permute(0, 2, 1, 3)
                     .reshape(1, La, _LTXAV_AUDIO_PACK).contiguous())
            vi = vemb[i:i + 1].contiguous()
            sig_i = float(sig_all[i].item()) if (sig_all is not None and sig_all.numel() >= B) else \
                (float(sig_all[0].item()) if sig_all is not None else float(sigma))
            p = qfe.DenoiseStepParams()
            ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
            p.struct_size = ctypes.sizeof(p)
            p.latent_in = tokens.data_ptr()
            p.velocity_out = self._out.data_ptr()
            p.velocity_out_capacity = self._out.numel() * self._out.element_size()
            p.dims = (ctypes.c_int * 5)(1, N, C, 0, 0)
            p.dtype = _qf_dtype(tokens.dtype)
            p.sigma = sig_i
            p.step_index = step_index
            p.total_steps = self._num_steps
            p.context = vi.data_ptr()
            p.context_dims = (ctypes.c_int * 3)(*vi.shape)
            p.context_dtype = _qf_dtype(vi.dtype)
            # [step-cache-key] uuid-symbolic key (was constant 0 = kNoCtxKey; the EC session
            # gate force-computes at 0 — the AV loop was the measured miss, see loop head).
            cuid = cuuids[i] if (cuuids is not None and i < len(cuuids)) else None
            p.cfg_context_key = self._ctx_key_assigner.key(cuid)
            mp = qfe.DenoiseStepMultiParams()
            ctypes.memset(ctypes.byref(mp), 0, ctypes.sizeof(mp))
            mp.struct_size = ctypes.sizeof(mp)
            mp.base = p
            mp.audio_latent_in = arows.data_ptr()
            mp.audio_velocity_out = self._out_audio.data_ptr()
            mp.audio_velocity_out_capacity = self._out_audio.numel() * self._out_audio.element_size()
            mp.audio_dims = (ctypes.c_int * 4)(1, La, _LTXAV_AUDIO_PACK, 0)
            mp.audio_dtype = _qf_dtype(arows.dtype)
            mp.audio_scale = 1.0            # LTX co-denoise: same-sigma, no carried variable
            self._call_denoise_step_multi(mp, f"LTX-AV denoise_step_multi[step={step_index},group={i}]")
            out5d[i:i + 1] = self._out.transpose(1, 2).reshape(1, C, F, H, W).to(xin.dtype)
            # unpack audio velocity rows [1,L,128] → [1,8,L,16] (exact inverse of the pack)
            out_audio[i:i + 1] = (self._out_audio.reshape(1, La, _LTXAV_AUDIO_CH, _LTXAV_AUDIO_MEL)
                                  .permute(0, 2, 1, 3).to(out_audio.dtype))
            self._qf.step_count += 1
            self._sess_denoise += 1
        self._step_i += 1
        self._qf.sampler_step_count += 1
        # calculate_denoised per lane on the SAME sigma (LTX joint schedule), returned in the
        # SAME form the sampler handed us (the H3-measured contract).
        if _nested:
            den_v = self.model_sampling.calculate_denoised(sigma, out5d.float(), x_video)
            den_a = self.model_sampling.calculate_denoised(sigma, out_audio.float(), x_audio)
            return comfy.nested_tensor.NestedTensor((den_v, den_a))
        Bx = int(x.shape[0])
        vel_flat = torch.cat([out5d.reshape(Bx, -1), out_audio.reshape(Bx, -1)], dim=1).reshape(x.shape)
        return self.model_sampling.calculate_denoised(sigma, vel_flat.float(), x)

    def process_latent_out(self, latent):
        # Finalize on the VIDEO half (the session's step pin is the packed video tokens;
        # t2av: engine finalize early-returns — buffer read-not-written), close the session,
        # then hand the ORIGINAL joint latent to comfy's LTXAV post-processing untouched.
        if self._qf.current_session is not None:
            try:
                # The final latent arrives NESTED or sampler-PACKED flat (the same dual form
                # _apply_model handles — MEASURED: process_latent_out runs BEFORE comfy's
                # un-pack, so the packed [B,1,N_total] shape reaches here). Extract the video
                # half either way.
                if getattr(latent, "is_nested", False):
                    lv = latent.unbind()[0]
                elif latent.ndim == 5:
                    lv = latent
                else:
                    ls = getattr(self, "latent_shapes", None)
                    if not ls or len(ls) < 2:
                        raise RuntimeError(f"qf_native LTX-AV: finalize got a non-nested ndim="
                                           f"{latent.ndim} latent and model.latent_shapes is "
                                           "unset - cannot extract the video half")
                    n = int(math.prod(ls[0][1:]))
                    lv = latent.reshape(int(latent.shape[0]), -1)[:, :n].reshape(list(ls[0]))
                if lv.ndim != 5:
                    raise RuntimeError(f"qf_native LTX-AV: finalize expected the 5D video latent, "
                                       f"got ndim={lv.ndim} (nested={getattr(latent, 'is_nested', False)})")
                lat5 = (lv[0:1] if lv.shape[0] > 1 else lv)
                Bc, C, F, H, W = lat5.shape
                N = F * H * W
                tokens = (lat5.reshape(1, C, N).transpose(1, 2)
                          .to(torch.bfloat16).contiguous())
                lib = self._qf.lib
                fp = qfe.DenoiseFinalizeParams()
                ctypes.memset(ctypes.byref(fp), 0, ctypes.sizeof(fp))
                fp.struct_size = ctypes.sizeof(fp)
                fp.latent = tokens.data_ptr()
                fp.latent_capacity = tokens.numel() * tokens.element_size()
                fp.dims = (ctypes.c_int * 5)(1, N, C, 0, 0)
                fp.dtype = _qf_dtype(tokens.dtype)
                fst = lib.quantfunc_denoise_finalize(self._qf.current_session, ctypes.byref(fp))
                if fst != qfe.QUANTFUNC_OK:
                    raise RuntimeError(f"qf_native: LTX-AV denoise_finalize failed: {qfe.last_err(lib)}")
                qfe.info(f"[qf_native] LTX-AV SESSION CLOSED after {self._step_i} sampler steps, "
                      f"{self._sess_denoise} denoise_step_multi calls, finalize=OK", flush=True)
            finally:
                self._qf.end_session_if_open()
        # No comfy-side frame-0 pin here (wan-align 2026-08-22): i2v conditioning is the
        # Inplace latent + noise_mask, and KSamplerX0Inpaint's per-step blend already leaves
        # the final x's conditioned frames at the clean latent — the decode input is right
        # by construction.
        return super(QFLTXModel, self).process_latent_out(latent)


FAMILY = "ltx2"

def _file_has_prefix(path, prefix):
    """Cheap safetensors HEADER probe: does any key start with `prefix`? Unreadable/not a
    safetensors => False. Module-level: used by the AV discriminant (vocoder.*) AND the
    [ltx25-allin] te_file relaxation (text_embedding_projection.*)."""
    import struct as _st
    try:
        with open(path, "rb") as fh:
            n = _st.unpack("<Q", fh.read(8))[0]
            if n > (1 << 31):
                return False
            hdr = json.loads(fh.read(n))
        return any(k.startswith(prefix) for k in hdr)
    except Exception:  # noqa: BLE001 - unreadable = no keys
        return False




def matches(pipeline_class, transformer_class=""):
    """The ENGINE's own detector (src/LTX2VideoPipeline.cpp ltx2_pipeline_detect):
        pipeline_class == "LTX2Pipeline"
    — an EXACT match with no transformer_class half, so this seam has none either."""
    return str(pipeline_class) == "LTX2Pipeline"


def register(deps):
    """Return the ltx2 family BUILDER. `deps` gives the package-level helpers (engine cache,
    liveness registry) without importing __init__."""


    def build(transformer1_path, bundle_dir=None, lora_entries=(), pinned_memory=False):
        """File-based (ComfyUI single-file) loading for LTX-2.5 — the shared staging pattern.
        CONNECTORS-SOURCE contract (user 2026-08-31 — fully support the transformer-only
        export; supersedes the 2026-08-22 one-file-only ruling):
        - transformer/  <- the transformer export (ALL-IN or transformer-only);
        - connectors/   <- resolved in priority order (a QUALIFYING source packs BOTH
          modality blocks — model.diffusion_model.{video,audio}_embeddings_connector.* —
          the joint-AV discriminant; a video-only source is rejected, see below):
            (1) the transformer file itself, when it qualifies (ALL-IN — the
                official-comfy-single-file mirror; byte-identical to the old path);
            (2) a same-dir sibling `*connector*.safetensors` that qualifies —
                the shared, recipe-independent completion file (extracted VERBATIM from
                the official comfy model file's connector keys; one copy serves every
                quantized transformer variant).
          NEITHER found -> fail loud naming the dir probed, the BOTH-modality
          requirement, and both remedies (a video-only-connector source additionally
          gets the joint-AV explanation — it must not fall through to the retired 2.3
          video-only path whose connector_ckpt widget no longer exists). The engine
          #565 comfy25 branch consumes either source identically (prefix detect ->
          rename+strip).
        text_embedding_projection is deliberately NOT required from the model file: the
        workflow's with-proj clip applies it TE-side (comfy lt.py dual_linear), the
        session cond arrives post-projection, and the engine loads it opportunistically
        (single-file first, te-dir fallback, else skip — LTX2VideoPipeline.cpp:376-397).
        AV capability = the packed audio_embeddings_connector in the RESOLVED connectors
        source (the engine's denoise_only has_audio_ probe + the _is_av discriminant
        below both read that source; every resolved source qualifies -> file-mode
        always arms AV); audio decode is the workflow's own audio
        VAELoader, never engine-side.
        Engine create runs denoise_only=True (VAE decode weights skipped; TE lazy)."""
        # CONNECTORS-SOURCE resolution (user 2026-08-31). The engine applies the
        # embeddings-connector tail itself on the unprocessed dual-proj cond — the
        # official comfy contract (lt.py marks TE output `unprocessed_ltxav_embeds`;
        # the model side owns the connector) — so a weight source MUST exist; no
        # workflow module can substitute it (the TE-side compat branch is legacy-shape
        # only and upstream-marked TODO:remove). The probe is CONTENT-based (header
        # prefix via _file_has_prefix), never name-based trust; the name filter merely
        # bounds which siblings get their header read.
        # A QUALIFYING source packs BOTH modality blocks (video AND audio) — the same
        # joint-AV discriminant `_is_av` reads downstream. LTX-2.5 file-mode is
        # joint-AV; a video-only-connector source cannot arm it, and accepting one
        # would fall through to the retired 2.3 video-only path whose connector_ckpt
        # widget no longer exists (an unactionable dead-letter refusal) — so it is
        # rejected HERE with an accurate message instead.
        _CONN_VID = "model.diffusion_model.video_embeddings_connector."
        _CONN_AUD = "model.diffusion_model.audio_embeddings_connector."
        def _qualifies(path):
            return (_file_has_prefix(path, _CONN_VID)
                    and _file_has_prefix(path, _CONN_AUD))
        conn_src = None
        if _qualifies(transformer1_path):
            conn_src = transformer1_path   # ALL-IN: self-link, byte-identical to the old path
        else:
            _dir = os.path.dirname(transformer1_path)
            try:
                _sibs = sorted(f for f in os.listdir(_dir)
                               if f.lower().endswith(".safetensors")
                               and "connector" in f.lower())
            except OSError:  # unreadable dir = no siblings; the raise below names it
                _sibs = []
            _qualified = [os.path.join(_dir, _f) for _f in _sibs
                          if os.path.join(_dir, _f) != transformer1_path
                          and _qualifies(os.path.join(_dir, _f))]
            if _qualified:
                conn_src = _qualified[0]   # deterministic: sorted-first
                if len(_qualified) > 1:
                    qfe.info(f"[qf_native] ltx2: {len(_qualified)} qualifying connectors "
                          f"completion files in {_dir!r}; using the alphabetically-first "
                          f"{os.path.basename(conn_src)} (others: "
                          f"{', '.join(os.path.basename(p) for p in _qualified[1:])})",
                          flush=True)
            if conn_src is None:
                _why = ("packs the video_embeddings_connector but NOT the audio one - "
                        "LTX-2.5 file-mode is joint-AV and needs BOTH - "
                        if _file_has_prefix(transformer1_path, _CONN_VID)
                        else "is a transformer-only export ")
                raise RuntimeError(
                    f"qf_native ltx2: {os.path.basename(transformer1_path)} {_why}"
                    f"and no connectors completion file was found "
                    f"next to it (probed {_dir!r} for *connector*.safetensors packing "
                    f"BOTH {_CONN_VID}* and {_CONN_AUD}*). The engine applies the "
                    f"embeddings-connector tail "
                    f"itself (official comfy contract: TE output is unprocessed_ltxav_embeds), "
                    f"so a source is required. Fix: place the shared ltx-2.5-connectors file "
                    f"in the same folder, or use an ALL-IN export (*-allin-*.safetensors).")
            qfe.info(f"[qf_native] ltx2: transformer-only export - connectors completion file: "
                  f"{os.path.basename(conn_src)}", flush=True)
        extra = {"connectors": conn_src}
        model_dir = qfmp.stage_denoise_only_package(bundle_dir, transformer1_path, extra_links=extra)
        return _build_from_package(model_dir, os.path.basename(transformer1_path),
                                   lora_entries=lora_entries,
                                   # pinned memory is the loaders' pinned_memory switch (user 2026-09-26, default
                                   # OFF), as for every family; LTX-2.5 no longer forces it on (it did since
                                   # 2026-08-23, for the two-stage flow's offload round trips)
                                   create_extra={"denoise_only": True}, pinned_memory=pinned_memory)

    def _build_from_package(model_dir, model_name, lora_entries=(), create_extra=None, pinned_memory=False):
        # NOTE (CR simplicity): the joint audio+video (a2v) path needs an engine external-step
        # split (makeExternalStepFn) this seam does not implement, and since the widget was
        # removed there is no way to request it — so the old join_audio_prompt flag and its two
        # refusal branches are GONE (structurally unreachable code is not a guard). The AV path
        # below is LTX-2.5's own engine-side joint AV, which is a different mechanism.
        # ── LTX-2.5 JOINT-AV auto-detect (c5.8b): the SAME discriminant the engine's own
        # has_audio_ uses — the engine model_dir ships audio_vae/ weights (the video-only
        # 19B staging deliberately omits it) — so plugin and engine agree by construction.
        # AV → QFLTXAVModel (t2av; engine-side connector via av_unprocessed_ctx; NO
        # connector_ckpt needed). A package that is not AV is refused below.
        # CR conf-1: mirror the engine's FULL has_audio_ discriminant (LTX2VideoPipeline
        # :440): the audio DECODE chain staged (audio_vae weights + a vocoder — 2.3
        # separate dir or 2.5 bundled vocoder.* keys), OR — the denoise_only AV-lane
        # probe (user 2026-08-22 "官方工作流都是独立组件"): the staged connectors source
        # packs the audio_embeddings_connector blocks (the all-in single file, exactly
        # like the official comfy single-file; the split-layout connectors file carries
        # them too). The native seam never decodes audio engine-side (comfy's own
        # LTXVAudioVAEDecode does), so audio-VAE staging is NOT required for AV.
        # A mismatch either way would hit the engine's begin-time audio-lane refusal
        # (loud, but confusing); agreeing by construction avoids it. Decided once per staged package: a LoRA rebuild
        # of the same package gets the same answer.
        _avae_dir = os.path.join(model_dir, "audio_vae")
        _avae_files = ([os.path.join(_avae_dir, f) for f in os.listdir(_avae_dir)
                        if f.endswith(".safetensors")]
                       if os.path.isdir(_avae_dir) else [])
        _voc_dir = os.path.join(model_dir, "vocoder")
        _voc_ok = os.path.isdir(_voc_dir) and any(
            f.endswith(".safetensors") for f in os.listdir(_voc_dir))
        _voc_bundled = (not _voc_ok) and any(
            _file_has_prefix(f, "vocoder.") for f in _avae_files)
        _conn_staged = os.path.join(model_dir, "connectors", "model.safetensors")
        _is_av = (bool(_avae_files) and (_voc_ok or _voc_bundled)) or (
            _file_has_prefix(_conn_staged,
                             "model.diffusion_model.audio_embeddings_connector.")
            or _file_has_prefix(_conn_staged, "audio_embeddings_connector."))
        # LTX svdq is PRE-quantized: pass the MINIMUM to create (minimal=True → empty config_json).
        # The svdquant metadata carries the layout/precision (embedded config: cross_attn_mod=True /
        # audio_cross_attn_mod=True / cross_attention_dim 4096 / num_heads 32 → 9-mod). ANYTHING
        # supplied on top (auto_optimize / height / width / a precision map) competes with it and the
        # engine resolves to the 19b default (connector 3840 / transformer 6-mod — the E2E run-1..8
        # create mis-size). The harness's whole svdq entry is {backend:svdq, model_dir}; mirror that.
        # RESIDUAL (tracked, §6.5 wan text_precision round): an EMPTY config also means the engine's
        # resolve-at-entry text-precision falls to its SM-DEFAULT tier (est::smDefaultTextPrecision:
        # fp4 on SM120+, int4 below). That default is what BROKE the wan loader (UMT5 rejects 4-bit);
        # it currently WORKS here because LTX's TE tiers accept the default — but if an LTX TE arch
        # without a wired 4-bit tier ever routes through this minimal create, it hits the same class.
        # No fix now (adding keys back defeats minimal=True's purpose); this note is the tripwire.
        if _is_av:
            return qfmp.family_build(
                deps, model_dir, create_extra, comfy.supported_models.LTXAV,
                {"image_model": "ltxav", "disable_unet_model_creation": True}, QFLTXAVModel,
                f"[qf_native] loaded QuantFuncNativeLoader (LTX-2.5 JOINT-AV svdq) package={model_name} "
                f"capacity=native Prepared query (create deferred)", pinned_memory)(list(lora_entries))
        raise RuntimeError("QuantFuncNativeLoader: connector_ckpt (comfy LTX-2.3 ckpt with the "
                           "video_embeddings_connector) is required")

    return build
