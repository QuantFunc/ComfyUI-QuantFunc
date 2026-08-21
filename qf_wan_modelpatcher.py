"""qf_wan_modelpatcher — the WAN 2.x family seam (model + builder), self-contained.

ARCHITECTURE (one module per model family):
  qf_modelpatcher.py  = FAMILY-AGNOSTIC substrate only (session mixin, patcher, lazy engine,
                        LoRA-rebuild contract, tempfile/interrupt/ctx-key helpers).
  qf_<family>_modelpatcher.py = ONE family: its comfy model subclass(es) + its builder + its
                        detection rule. Nothing here may be imported by another family module.
  __init__.py         = package wiring only: engine cache, the single loader node + LoRA node,
                        and the family REGISTRY it assembles from the modules listed in
                        _FAMILY_MODULES. Adding a family = add a module + one entry there.

Every family module exposes the same three names, which is the whole interface:
  FAMILY   : str                        — the model_type key ("wan")
  matches(pipeline_class) -> bool       — the engine's OWN detector predicate, transcribed
  register(deps) -> builder             — deps carries the shared helpers; returns build(...)
"""
import os
import ctypes
import json
import logging

import torch

import comfy.model_base
import comfy.model_management
import comfy.supported_models
import comfy.conds

from . import qf_engine as qfe
from . import qf_modelpatcher as qfmp
from .qf_modelpatcher import (_qf_dtype, _QFStub, _CtxKeyAssigner, QFModelPatcher,
                              QFSessionModelMixin, save_ref_tempfile, cleanup_ref_tempfile,
                              _interrupt_poll_end_session_on_raise)



FAMILY = "wan"


def matches(pipeline_class, transformer_class=""):
    """The ENGINE's own detector, BOTH halves (src/WanVideoPipeline.cpp wan_detect):
        pipeline_class.rfind("Wan", 0) == 0 || transformer_class == "WanTransformer3DModel"
    The whole Wan family is matched by the "Wan…Pipeline" prefix (WanPipeline for TI2V-5B,
    WanImageToVideoPipeline for A14B i2v, …); the transformer half catches a package whose
    model_index carries no pipeline class."""
    return (str(pipeline_class).startswith("Wan")
            or str(transformer_class) == "WanTransformer3DModel")


class QFWanModel(QFSessionModelMixin, comfy.model_base.WAN21):
    # Wan VAE scale factors (AutoencoderKLWan): temporal 4, spatial 8. The session geometry is
    # DERIVED from the latent the sampler hands us + the sampler's own sigma schedule — the loader
    # carries NO geometry widgets (official-loader shape: length/size come from WanImageToVideo or
    # EmptyHunyuanLatentVideo, steps from KSampler, fps from CreateVideo).
    _VAE_T, _VAE_S = 4, 8
    _DEFAULT_FPS = 16.0                   # wan 2.x training rate; only rides options_json (informational)

    def __init__(self, model_config, engine, start_image, device=None, resident_block_count=999):
        # disable_unet_model_creation is honored inside BaseModel.__init__
        super().__init__(model_config, device=device)
        self.diffusion_model = _QFStub()
        self._resident_block_count = int(resident_block_count)   # [manual-residency] video begin knob
        self._qf = engine                 # QFEngineHandle (owns lib + pipeline + the open session)
        self._start_image = start_image   # comfy IMAGE tensor [B,H,W,C] float 0..1, or None (Option C)
        self._num_steps = 0               # DERIVED per run from sample_sigmas (len-1) at _begin
        self._num_frames = 0              # DERIVED per run from the latent: (Tlat-1)*4 + 1
        self._fps = self._DEFAULT_FPS
        self._step_i = 0                  # SAMPLER step index (one _apply_model call = one sampler step)
        self._ctx_key_assigner = _CtxKeyAssigner()  # symbolic cfg_context_key from comfy uuids (#B3)
        self._sess_denoise = 0            # denoise_step calls THIS session (instrument; cfg>1 = 2/step)
        self._max_batch = 0               # max cond-group batch seen (2 = CFG batch was split)
        self._max_ctx_seq = 0             # max cross_attn seq len across a run's cond groups (accumulated in
        #                                   extra_conds) → sizes the session max_context_dims so a LATER,
        #                                   LONGER different-length CONDRegular group (e.g. a >512-tok negative
        #                                   umt5 pads to 1024) does not exceed the begin maxima. Reset in _begin.
        self._out = None                  # reused velocity_out device buffer, batch-shaped

    # ---- i2v reference (Option C): comfy IMAGE tensor → disposable temp PNG for begin_edit --------
    # SHARED between QFWanModel and QFLTXModel (lifted to module level for the LTX i2v wiring —
    # same §6.5-simplicity rationale as _interrupt_poll_end_session_on_raise: the WAN/LTX copies
    # of session hardening drifted once already; one copy or it happens again).
    def _save_ref_tempfile(self, image):
        return save_ref_tempfile(image)

    @staticmethod
    def _cleanup_tempfile(path):
        cleanup_ref_tempfile(path)

    # The conditioning keys the engine's OWN begin_edit conditioning REPLACES, DERIVED (not from general
    # WAN knowledge — that missed clip_vision_output, then context_latents, then the BaseModel-level pair)
    # from the extra_conds channels this class bypasses: WAN21.extra_conds's own body + the `out =
    # super().extra_conds(**kwargs)` = BaseModel.extra_conds it opens with. tests/reject_list_completeness.py
    # is the MECHANISM that keeps this list honest — it scans BOTH those bodies AND the concat_cond/encode_adm
    # methods BaseModel.extra_conds invokes, and FAILS on any uncovered consumable key (run on comfy upgrades).
    #   • WAN21 body: clip_vision_output, time_dim_concat, reference_latents, context_latents.
    #   • BaseModel super: cross_attn_controlnet, noise_concat — both INERT in stock comfy for WAN
    #     (WanModel.forward has a **kwargs sink; noise_concat has zero producers anywhere; cross_attn_controlnet
    #     is consumed ONLY by comfy's ControlNet-apply plumbing (controlnet.py), which this engine can't honor
    #     — and a wired ControlNet is ALREADY loud-failed in _apply_model). Dropping them is behaviour-IDENTICAL
    #     to stock; guarded defensively so a future producer can't silently regress.
    # NOT in THIS list because they are covered by a DIFFERENT hook, not silently dropped:
    #   • concat_latent_image → its own specific loud-fail below (concat_mask/concat_mask_index only co-occur
    #     with it, so that one guard covers the whole concat path).
    #   • denoise/inpaint mask → the `scale_latent_inpaint` override (the mask reaches the SAMPLER, not
    #     extra_conds — an extra_conds guard fires too late; see that method).
    #   • y/adm → impossible: WAN21 doesn't override BaseModel.encode_adm → it returns None unconditionally.
    # `cross_attn` is the ONLY channel the engine takes (emitted below as c_crossattn); `camera_conditions`
    # is a defensive superset (only WAN21_Camera reads it). A whitelist (call super().extra_conds + reject any
    # non-c_crossattn output) is NOT viable: super()→concat_cond dereferences
    # self.diffusion_model.patch_embedding.weight, which our _QFStub lacks, AND for an i2v model builds a
    # zeros-concat even with no concat_latent_image — so it would crash / always-trip. Hence this reject-list.
    _ENGINE_IGNORED_COND_KEYS = ("clip_vision_output", "time_dim_concat", "reference_latents",
                                 "context_latents", "cross_attn_controlnet", "noise_concat",
                                 "camera_conditions")

    def extra_conds(self, **kwargs):
        # Option C: the engine does its OWN i2v channel-concat internally from THIS loader's start_image
        # (a file-path begin_edit) — it CANNOT consume comfy's VAE-encoded concat_latent_image (no
        # ref-latent field in the denoise ABI), so comfy's concat_cond path is deliberately NOT used.
        # RUN-START clean slate FIRST (#2 session lifecycle) — BEFORE the loud-fails below, so a rejected
        # bad-wiring requeue still closes a session STRANDED by a prior Interrupt (else a permanently-
        # erroring graph would leak a GPU-resident stranded session across requeues — CR vuln nit).
        # extra_conds is comfy's ONLY hook that fires at the start of EVERY run for an EMPTY (denoise=1.0)
        # latent (samplers.py gates process_latent_in on count_nonzero>0), and comfy CACHES+REUSES this
        # QFWanModel instance across a requeue — so closing here forces _apply_model to _begin fresh.
        # Idempotent across the pos+neg extra_conds calls of a run (the 2nd finds nothing open → no-op).
        was_open, ok = self._qf.end_session_if_open()
        if was_open:
            print("[qf_native] closed a pre-existing session at run start "
                  f"(prior run interrupted/uncleaned); end ok={ok}", flush=True)
        # ★ HARD REQUIREMENT (CR): a wired-but-ignored WanImageToVideo.start_image shows up here as
        # concat_latent_image. Silently discarding it is the exact defect the CR NO-GO'd — FAIL LOUD
        # with the TRUE current limitation (the loader's start_image widget was REMOVED in the
        # file-based redesign; i2v is not yet wired on the denoise_only file path).
        if kwargs.get("concat_latent_image") is not None:
            raise RuntimeError(
                "qf_native: WanImageToVideo.start_image is wired, but i2v is NOT yet supported on the "
                "file-based QuantFunc native loader (its denoise_only engine session cannot consume "
                "comfy's encoded concat_latent_image, and the engine-side reference encode is "
                "deliberately not loaded). Use the t2v preset/workflow for now — i2v arrives with its "
                "own wan2.2-a14b-i2v preset. Silently ignoring your reference would be worse; refusing.")
        # ★ CLOSE THE SILENT-DISCARD CLASS: every OTHER image/reference channel WAN21.extra_conds consumes
        # that this bypass would drop (clip_fea / time_dim / reference / context / camera). The engine does
        # its OWN i2v conditioning from the loader's start_image, so any of these wired from a stock node
        # (CLIPVisionEncode→clip_vision_output, BerniniConditioning→context_latents, …) must FAIL LOUD.
        for _k in self._ENGINE_IGNORED_COND_KEYS:
            if kwargs.get(_k) is not None:
                raise RuntimeError(
                    f"qf_native: '{_k}' conditioning is wired, but the QuantFunc wan native session "
                    f"cannot consume comfy's '{_k}' — it would be silently ignored, so it is refused. "
                    f"Remove the node feeding '{_k}'. (Reference/i2v conditioning is not yet supported "
                    f"on the file-based loader; it arrives with the wan2.2-a14b-i2v preset.)")
        # We MIRROR WAN21's CONDRegular for cross_attn (the only channel the engine takes).
        out = {}
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            # Track the MAX cross_attn seq length across this run's cond groups. extra_conds runs for BOTH
            # pos+neg BEFORE _begin, and with CONDRegular (no LCM padding) a DIFFERENT-LENGTH negative (umt5
            # min_length=512 pads a >512-tok prompt to 1024) reaches the engine as a SEPARATE call with its
            # own length — so the session's max_context_dims (set in _begin) must cover the LARGEST group, or
            # the engine rejects "context_dims <= the begin maxima" (measured: short-pos + >512-tok-neg cfg run).
            self._max_ctx_seq = max(self._max_ctx_seq, int(cross_attn.shape[1]))
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
        return out

    def scale_latent_inpaint(self, *args, **kwargs):
        # comfy's KSamplerX0Inpaint calls this (BaseModel.scale_latent_inpaint) ONLY when a
        # denoise/noise mask is wired (SetLatentNoiseMask, samplers.py: `if denoise_mask is not None:`).
        # The mask is applied by the SAMPLER WRAPPER — it per-step blends the latent OUTSIDE
        # extra_conds/_apply_model — so the reject-list can never see it; THIS override is the correct
        # hook (it fires exactly, and only, when a mask is present). The QuantFunc wan native session
        # does its OWN i2v conditioning from the loader's start_image and has no inpaint-mask seam, so a
        # wired mask would silently blend against the noise latent → corrupt i2v. FAIL LOUD instead.
        raise RuntimeError(
            "qf_native: a denoise/inpaint mask (e.g. SetLatentNoiseMask) is wired, but the QuantFunc wan "
            "native session does not support masked inpainting — comfy's sampler would silently blend the "
            "mask against the noise latent and corrupt the result. Remove the mask / SetLatentNoiseMask "
            "node.")

    def _derive_geometry(self, xin, transformer_options):
        """DERIVE the session geometry from the graph (official-loader shape — the loader has no
        geometry widgets). xin: [B,C,Tlat,Hl,Wl] from WanImageToVideo / EmptyHunyuanLatentVideo;
        the step count from the sampler's own sigma schedule. The only refusal left is a sampler
        that publishes NO sigma schedule at all; sub-range schedules are legal (see below)."""
        Tlat = int(xin.shape[2])
        self._num_frames = (Tlat - 1) * self._VAE_T + 1
        sigmas = transformer_options.get("sample_sigmas") if isinstance(transformer_options, dict) else None
        if sigmas is None or len(sigmas) < 2:
            raise RuntimeError(
                "qf_native: the sampler did not publish a sigma schedule (transformer_options["
                "'sample_sigmas']) — the engine session needs the step count. Use a stock KSampler / "
                "SamplerCustom on this model.")
        self._num_steps = len(sigmas) - 1
        # SUB-RANGE sigma schedules (KSamplerAdvanced start/end_at_step, denoise<1) are LEGAL —
        # verified engine-side 2026-08-21: the external session's expert selection is PER-STEP
        # sigma-driven over ABSOLUTE quantities (WanVideoPipeline makeVideoStepFn: ts = sigma *
        # scheduler num_train_timesteps vs boundary_t = boundary_ratio * the SAME constant — no
        # schedule-range dependence), and residency is per-step ENSURE-RESIDENT, not a one-way
        # full-range latch (the c5.7a legC-replay hardening). So the official two-KSamplerAdvanced
        # wan workflow maps 1:1: the high-noise stage's sigmas are all >= the boundary -> engine
        # runs transformer1; the low stage's are below -> transformer2 — even a MIS-wired stage
        # (low MODEL into the high sampler) still computes correctly, because the engine picks by
        # sigma, not by which loader output was used. An earlier refusal here guarded a mis-timing
        # that cannot occur under that mechanism; it blocked the official dual-sampler template
        # and is deliberately removed (sampler/scheduler are ComfyUI-owned — user directive).

    def _begin(self, x_group, ctx_group):
        """Open the edit session. x_group/ctx_group are PER-COND-GROUP (B==1) slices — begin binds
        the geometry MAXIMA with B==1 (the engine + include/quantfunc.h require cond B==1)."""
        self._qf.end_session_if_open()    # clean any stale session from a prior failed run on this pipeline
        # start_image=None = t2v (branch below). The ENGINE refuses an i2v checkpoint
        # (in>out) without a ref fail-loud at begin — no silent t2v on an i2v model.
        lib = self._qf.lib
        bpx = qfe.DenoiseBeginParams()
        ctypes.memset(ctypes.byref(bpx), 0, ctypes.sizeof(bpx))
        bpx.struct_size = ctypes.sizeof(bpx)
        # x is the 5D noise latent [1,C,T,Hl,Wl]; the engine derives its own geometry from num_frames +
        # width/height, so pass the target pixel W/H (latent * vae scale 8).
        bpx.width = int(x_group.shape[-1]) * 8
        bpx.height = int(x_group.shape[-2]) * 8
        bpx.num_steps = self._num_steps
        # Size the context MAXIMA to the LARGEST cross_attn seq across this run's cond groups (accumulated in
        # extra_conds for pos+neg), NOT just this first group's — a longer different-length negative reaches
        # the engine as a separate CONDRegular call and must fit the begin maxima. max(...) with this group is
        # the safe fallback if the accumulator was never populated (e.g. cross_attn-less flow).
        _max_seq = max(self._max_ctx_seq, int(ctx_group.shape[1]))
        bpx.max_context_dims = (ctypes.c_int * 3)(int(ctx_group.shape[0]), _max_seq, int(ctx_group.shape[2]))
        self._max_ctx_seq = 0             # reset for the next run's extra_conds accumulation
        bpx.cond_dtype = _qf_dtype(ctx_group.dtype)
        bpx._opts = json.dumps({"num_frames": self._num_frames, "fps": float(self._fps),
                                "resident_block_count": self._resident_block_count}).encode()
        bpx.options_json = bpx._opts
        session = ctypes.c_void_p()
        if self._start_image is None:
            # t2v — the A14B/Wan2.1 T2V checkpoints take no reference frame. The engine's own
            # E3 gate refuses an i2v checkpoint (in>out) without a ref fail-loud, so a user
            # who forgot the ref on an i2v model still gets a clear error, never a silent t2v.
            st = lib.quantfunc_denoise_begin(self._qf.pipeline, ctypes.byref(bpx), ctypes.byref(session))
            self._begin_keep = bpx
            if st != qfe.QUANTFUNC_OK:
                raise RuntimeError(f"denoise_begin (wan t2v) failed: {qfe.last_err(lib)}")
            self._qf.current_session = session
            self._qf.unloaded = False
            self._step_i = 0
            self._sess_denoise = 0
            self._max_batch = 0
            self._ctx_key_assigner.reset()
            print(f"[qf_native] SESSION OPEN (t2v) handle={session.value:#x} steps={self._num_steps} "
                  f"cond={tuple(ctx_group.shape)}", flush=True)
            return
        # Option C: save the loader's start_image to a disposable temp PNG the engine loads+encodes.
        # Lifetime = the begin_edit call only (the engine reads the file at begin); deleted in finally
        # on BOTH success and exception, so no user images accumulate in the temp dir.
        ref_tmp = self._save_ref_tempfile(self._start_image)
        try:
            epx = qfe.DenoiseBeginEditParams()
            ctypes.memset(ctypes.byref(epx), 0, ctypes.sizeof(epx))
            epx.struct_size = ctypes.sizeof(epx)
            epx.base = bpx
            epx._bp = bpx
            epx._ref = [str(ref_tmp).encode()]
            epx._ref_arr = (ctypes.c_char_p * 1)(*epx._ref)
            epx.ref_image_paths = epx._ref_arr
            epx.num_ref_images = 1
            epx.ref_img_resize = 0
            st = lib.quantfunc_denoise_begin_edit(self._qf.pipeline, ctypes.byref(epx),
                                                  ctypes.byref(session))
            self._begin_keep = epx
        finally:
            self._cleanup_tempfile(ref_tmp)
        if st != qfe.QUANTFUNC_OK:
            raise RuntimeError(f"denoise_begin_edit failed: {qfe.last_err(lib)}")
        self._qf.current_session = session
        self._qf.unloaded = False         # a successful begin implies the pipeline is GPU-resident again
        self._step_i = 0
        self._sess_denoise = 0
        self._max_batch = 0
        self._ctx_key_assigner.reset()    # per-generation: restart uuid→key numbering, no cross-gen leak
        print(f"[qf_native] SESSION OPEN handle={session.value:#x} steps={self._num_steps} "
              f"cond={tuple(ctx_group.shape)}", flush=True)

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        sigma = t
        ctx = c_crossattn
        if ctx is None:
            raise RuntimeError("qf_native: no c_crossattn cond — wire a native CLIPTextEncode")
        # A wired ControlNet reaches _apply_model as a non-None `control` object. The QuantFunc wan
        # native session consumes NO control input (ControlNet is not part of this denoise seam), so
        # accepting it would SILENTLY drop the guidance — the SAME silent-discard class extra_conds
        # fails loud on for WanImageToVideo.start_image. Refuse fail-loud, never render a plausible-
        # but-wrong video.
        if control is not None:
            raise RuntimeError(
                "qf_native: a ControlNet is wired into this sampler, but the QuantFunc wan native "
                "session does not consume comfy control hints — they would be silently ignored. Remove "
                "the ControlNet (native wan ControlNet is not supported through this loader).")
        dev = x.device
        # engine wants cond FP16/BF16 (native umt5 emits FP32) + latent BF16 (comfy samples fp32).
        ctx = ctx.to(device=dev, dtype=torch.bfloat16).contiguous()   # [B, seq, 4096]
        xin = x.to(torch.bfloat16).contiguous()                       # [B, C, T, Hl, Wl]
        B = int(xin.shape[0])
        # cfg_context_key is a SYMBOLIC key from comfy's per-conditioning uuid (self._ctx_key_assigner, in the
        # loop below), NOT a content hash and NOT the 0/1 cond_or_uncond ROLE index. The role index is not
        # unique per content (comfy batches by SHAPE only → a stock ConditioningCombine/SetArea puts two
        # DIFFERENT-content conditionings in one 0/1 bucket → a role key collides them → the engine's L2
        # cross-KV cache emits row-0's text for row-1: the seq-219 wrong-output defect). A content hash is the
        # MIRROR failure — it collides two branches with IDENTICAL content into the STATEFUL fbcache_ slot
        # (src/gemm/lighting/CLAUDE.md #B3). comfy's uuid (aligned with cond_or_uncond) is distinct per
        # conditioning entry, stable across steps, and position-independent (robust to a mid-run composition
        # change like ConditioningSetTimestepRange) — see _CtxKeyAssigner. cond_or_uncond is still read only to
        # REFUSE an unmappable latent batch (batch_size>1).
        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        cuuids = transformer_options.get("uuids") if isinstance(transformer_options, dict) else None
        if B > 1 and (cou is None or len(cou) != B):
            # engine is B==1-per-forward; a latent batch we can't map to per-cond-group B==1 rows
            # (e.g. batch_size>1) is REFUSED LOUDLY rather than surfacing as an opaque engine throw.
            raise RuntimeError(f"qf_native: engine forward is B==1 per cond group but got batch={B} "
                               f"with cond_or_uncond={cou} — batch_size>1 latents are not supported")
        self._max_batch = max(self._max_batch, B)   # 2 = comfy batched cond+uncond → the split fired
        if self._qf.current_session is None:
            self._derive_geometry(xin, transformer_options)  # session geometry from the GRAPH
            self._begin(xin[0:1].contiguous(), ctx[0:1].contiguous())
        if self._out is None or self._out.shape != xin.shape or self._out.dtype != xin.dtype:
            self._out = torch.empty_like(xin)
        sig_all = sigma.reshape(-1) if torch.is_tensor(sigma) else None
        step_index = self._sigma_step_index(sigma, sig_all, transformer_options)  # sigma-schedule-derived (QFSessionModelMixin)
        for i in range(B):
            # Interrupt checkpoint. comfy's run_every_op (comfy/ops.py) is a PER-OP checkpoint this seam's
            # opaque quantfunc_denoise_step never triggers — BUT comfy ALSO interrupts per OUTER sampler
            # step via its progress-bar hook (main.py hijack_progress -> ProgressBar.update_absolute ->
            # throw_exception_if_processing_interrupted), which fires for a progress-reporting sampler. So
            # the denoise is NOT "uninterruptible": MEASURED on 远程-linux (comfy 0.17), WITHOUT this poll
            # the run still stops ~0.81s after Interrupt (vs ~0.50s with it — within noise). This poll is a
            # ROBUSTNESS + granularity refinement, not an enabler: it fires at EACH per-cond-group boundary
            # and does NOT depend on the sampler reporting progress (a custom/non-reporting sampler would
            # otherwise leave THIS seam's denoise uninterruptible). Use comfy's OWN atomic helper — it
            # re-checks under the mutex, RESETS the global flag (no leak; matches the platform idiom), and
            # raises comfy's InterruptProcessingException, which the executor classifies as an interrupt
            # (execution.py "Processing interrupted" + execution_interrupted broadcast, 0.17 & 0.27),
            # identical to a native interrupt. The shared helper additionally ENDS the open session before
            # re-raising (§6.5 simplicity: parity with LTX's hardened poll — previously WAN left the session
            # OPEN and relied only on extra_conds' next-run close; that outer net remains as backstop).
            _interrupt_poll_end_session_on_raise(self._qf)
            xi = xin[i:i + 1].contiguous()
            oi = self._out[i:i + 1]
            ci = ctx[i:i + 1].contiguous()
            cuid = cuuids[i] if (cuuids is not None and i < len(cuuids)) else None
            ctx_key = self._ctx_key_assigner.key(cuid)          # symbolic key from comfy uuid (0=kNoCtxKey if absent)
            if os.environ.get("QF_NATIVE_DEBUG_CTXKEY"):           # off by default; confirms the uuid path is live
                print(f"[qf_native] CTXKEY step={step_index} grp={i} "
                      f"uuid={str(cuid)[:8] if cuid is not None else None} key={ctx_key}", flush=True)
            sig_i = float(sig_all[i].item()) if (sig_all is not None and sig_all.numel() >= B) else \
                (float(sig_all[0].item()) if sig_all is not None else float(sigma))
            p = qfe.DenoiseStepParams()
            ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
            p.struct_size = ctypes.sizeof(p)
            p.latent_in = xi.data_ptr()
            p.velocity_out = oi.data_ptr()
            p.velocity_out_capacity = oi.numel() * oi.element_size()
            dims = list(xi.shape) + [0] * (5 - xi.dim())
            p.dims = (ctypes.c_int * 5)(*dims)
            p.dtype = _qf_dtype(xi.dtype)
            p.sigma = sig_i
            p.step_index = step_index                          # sigma-derived (see above); NOT the per-call counter
            p.total_steps = self._num_steps
            p.context = ci.data_ptr()
            p.context_dims = (ctypes.c_int * 3)(*ci.shape)
            p.context_dtype = _qf_dtype(ci.dtype)
            p.cfg_context_key = ctx_key                        # symbolic key from comfy uuid (see _CtxKeyAssigner); NOT a hash / role index
            self._call_denoise_step(p, f"denoise_step[step={step_index},group={i},key={ctx_key}]")  # QFSessionModelMixin
            self._qf.step_count += 1
            self._sess_denoise += 1
        self._step_i += 1
        self._qf.sampler_step_count += 1
        return self.model_sampling.calculate_denoised(sigma, self._out.float(), x)

    def process_latent_out(self, latent):
        # NORMAL end-of-sampling: finalize (wan masked-blend pins frame-0; NO-OP for channel-concat) + end.
        # An INTERRUPT skips this — that stranded session is closed by extra_conds at the START of the
        # next run (#2; process_latent_in cannot host it — comfy skips that hook for empty latents).
        if self._qf.current_session is not None:
            try:
                lib = self._qf.lib
                fin = qfe.DenoiseFinalizeParams()
                ctypes.memset(ctypes.byref(fin), 0, ctypes.sizeof(fin))
                fin.struct_size = ctypes.sizeof(fin)
                # Match the session's STEP binding (the finalize pin, quantfunc_api.cpp checks
                # latent_dtype_pin == the finalize dtype): _apply_model feeds the engine BF16 latents,
                # so the pin dtype is bf16 — finalize MUST pass bf16 too. comfy's sampler latent is
                # fp32; passing it raw mismatches the pin → finalize REFUSED (silently, before this
                # status check existed). For wan channel-concat i2v finalize is a no-op (the ref lives
                # in the concat channels, not the noise frame-0), so the bf16 view is read-not-written
                # and the fp32 `latent` returned to the VAE below is unchanged.
                lat = (latent[0:1] if latent.shape[0] > 1 else latent).to(torch.bfloat16).contiguous()
                fin.latent = lat.data_ptr()
                fin.latent_capacity = lat.numel() * lat.element_size()
                dims = list(lat.shape) + [0] * (5 - lat.dim())
                fin.dims = (ctypes.c_int * 5)(*dims)
                fin.dtype = _qf_dtype(lat.dtype)
                # Check finalize's status like begin_edit/step/end — the engine can REFUSE it
                # (dims/dtype mismatch, stale session, quantfunc.h). A silently-discarded failure would
                # let a corrupt/un-finalized latent reach the VAE decode as if it succeeded.
                fst = lib.quantfunc_denoise_finalize(self._qf.current_session, ctypes.byref(fin))
                if fst != qfe.QUANTFUNC_OK:
                    raise RuntimeError(f"qf_native: denoise_finalize failed: {qfe.last_err(lib)}")
                print(f"[qf_native] SESSION CLOSED after {self._step_i} sampler steps, "
                      f"{self._sess_denoise} denoise_step calls, max_batch_per_step={self._max_batch} "
                      f"({'CFG-split fired' if self._max_batch > 1 else 'B==1 (cfg=1 or unbatched)'})",
                      flush=True)
            finally:
                self._qf.end_session_if_open()
        return super().process_latent_out(latent)


def register(deps):
    """Return the wan family BUILDER. `deps` gives the package-level helpers (engine cache,
    liveness registry, footprint estimator, lazy-engine class) without importing __init__."""
    get_engine = deps["get_engine"]
    bind_pipeline_model = deps["bind_pipeline_model"]
    may_release_handle = deps["may_release_handle"]
    estimate_footprint = deps["estimate_footprint"]
    apply_checkpoint_flow_shift = deps["apply_checkpoint_flow_shift"]

    def build(transformer1_path, transformer2_path, resident_block_count, bundle_dir,
              lora_entries=()):
        """wan A14B from BARE transformer FILES (INT8-Fast-aligned): transformer1 = HIGH-noise
        expert, transformer2 = LOW-noise expert — BOTH required (A14B is dual-expert; a missing
        low expert would denoise the low-sigma phase with nothing). The engine create is
        denoise_only: TE + VAE WEIGHTS are skipped (comfy's CLIP supplies conditioning, comfy's
        VAEDecode decodes), configs come from the plugin-shipped wan-a14b bundle."""
        if not transformer2_path:
            raise RuntimeError(
                "qf_native wan: this builder currently supports the DUAL-expert A14B shape only "
                "(transformer1 = high-noise + transformer2 = low-noise). A single-expert wan "
                "preset needs its own config bundle + a single-expert model_config manifest.")

        def _build(entries):
            # One staged, config-complete package per (bundle, file-pair): shipped A14B configs +
            # weight symlinks. Weights stay where the user put them (no copy).
            model_dir = qfmp.stage_denoise_only_package(bundle_dir, transformer1_path,
                                                        transformer2_path)
            # denoise_only (engine WanVideoPipeline.cpp): skip the UMT5 TE (~10 GB dead mass here
            # — conditioning comes from comfy's NATIVE CLIP) + the VAE weights (comfy decodes);
            # the engine still reads the staged vae/config.json for session geometry. The old
            # text_precision=int8 pin existed only because create used to BUILD the TE; with
            # denoise_only it is obsolete and deliberately gone.
            cfg = {"denoise_only": True}
            if entries:
                cfg["lora"] = list(entries)   # WanVideoPipeline splits target-tagged entries per expert

            def _factory():
                # comfy's device (0 when CUDA_VISIBLE_DEVICES pins one card; the real index on a
                # multi-visible setup) — the engine must create on the SAME card comfy computes on.
                dev = comfy.model_management.get_torch_device()
                eng, ckey = get_engine(model_dir, create_cfg=cfg,
                                       device_idx=getattr(dev, "index", 0) or 0)
                # Liveness tracker for the host-RAM sweep: BOTH loader outputs (model_high +
                # model_low) are live consumers of the ONE shared handle — bind each, so the
                # sweep keeps the handle while EITHER survives (the registry holds a weakref
                # LIST per ckey; _may_release refuses while any sibling lives).
                for _m in (model_high, model_low):
                    bind_pipeline_model(ckey, _m)
                return eng, ckey

            # DEFERRED create (QFLazyEngine): a chained QuantFuncNativeLoRA rebuilds for its
            # accumulated set, so an eager create here would build ONE PIPELINE PER CHAIN LINK.
            engine = qfmp.QFLazyEngine(_factory, estimate_footprint(model_dir),
                                      may_release=may_release_handle)

            # Wan A14B uses the Wan 2.1 VAE (16-ch AutoencoderKLWan) → WAN21_I2V latent_format
            # (Wan21, 16-ch). NOT WAN22_T2V (48-ch, that is the 5B TI2V VAE). UNet build DISABLED
            # (no 14B torch weights — _apply_model is overridden to drive the engine).
            unet_config = {"image_model": "wan2.1", "model_type": "i2v",
                           "disable_unet_model_creation": True}
            model_config = comfy.supported_models.WAN21_I2V(unet_config)
            qfmp.ensure_model_config_attrs(model_config)

            device = comfy.model_management.get_torch_device()
            offload = comfy.model_management.unet_offload_device()
            # DUAL MODEL outputs over ONE shared engine (user 2026-08-21 pivot: the wan loader
            # mirrors the official two-UNETLoader workflow shape — model_high wires to the
            # high-noise KSamplerAdvanced stage, model_low to the low stage). Expert selection is
            # ENGINE-side per-step sigma (see _derive_geometry) — the outputs exist for workflow
            # shape + per-stage comfy bookkeeping, and even a mis-wired stage still computes
            # correctly. model_low is the SHADOW (tiny ledger share, never drives the shared
            # engine's eviction — see QFModelPatcher._is_shadow); both models bind to the ckey so
            # handle destruction waits for both.
            model_high = QFWanModel(model_config, engine, None, device=device,
                                    resident_block_count=resident_block_count)
            model_low = QFWanModel(model_config, engine, None, device=device,
                                   resident_block_count=resident_block_count)
            model_low._qf_shadow = True
            # Default the sampler to the CHECKPOINT'S OWN flow schedule when the staged package
            # declares one (no-op-safe when absent; a stock ModelSamplingSD3 downstream still wins).
            apply_checkpoint_flow_shift(model_high, model_dir)
            apply_checkpoint_flow_shift(model_low, model_dir)
            patcher_high = QFModelPatcher(model_high, load_device=device, offload_device=offload)
            patcher_low = QFModelPatcher(model_low, load_device=device, offload_device=offload)
            print(f"[qf_native] loaded QuantFunc Wan Loader (svdq, denoise_only, dual MODEL) "
                  f"high={os.path.basename(transformer1_path)} "
                  f"low={os.path.basename(transformer2_path)} "
                  f"resident_blocks={resident_block_count} loras={len(entries)} "
                  f"footprint~{engine.footprint_bytes // (1024*1024)}MB (create deferred)")

            def _lora_rebuild_dual(_entries):
                # v1 refusal, LOUD (never silent): a QuantFuncNativeLoRA chained on ONE of the two
                # outputs would rebuild a NEW engine pair while the OTHER output still references
                # the old engine — two full engines resident (2x VRAM) and a silently split LoRA
                # state. Engine-side per-expert LoRA (target high/low) belongs on the LOADER as a
                # lora input — tracked follow-up; the 4-step production checkpoints ship BAKED.
                raise RuntimeError(
                    "qf_native wan: QuantFuncNativeLoRA cannot chain onto the dual-output wan "
                    "loader yet (it would fork the shared engine). Use checkpoints with the LoRA "
                    "baked in (the shipped 4-step exports), or wait for the loader-level lora "
                    "input.")

            qfmp.tag_lora_rebuild(patcher_high, entries, _lora_rebuild_dual)
            qfmp.tag_lora_rebuild(patcher_low, entries, _lora_rebuild_dual)
            return patcher_high, patcher_low

        return _build(list(lora_entries))

    return build
