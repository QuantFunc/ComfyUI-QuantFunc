"""qf_native.qf_modelpatcher — the seam: a comfy ModelPatcher whose WAN21-shim `.model`
drives quantfunc_denoise_step instead of a torch UNet.

Design (measured from comfy 0.27.0 + include/quantfunc.h + the proven native_session_video_t1.py):
- `.model` = subclass of comfy.model_base.WAN21 built with unet_config["disable_unet_model_creation"]
  =True → NO 14B torch UNet; keeps the real model_sampling (FLOW/ModelSamplingDiscreteFlow) + Wan21
  latent_format + process_latent_in/out.
- `_apply_model`: bypass comfy's transformer; feed the engine RAW x + normalized sigma + cond →
  quantfunc_denoise_step → velocity; return model_sampling.calculate_denoised(sigma, velocity, x).
- ★ CFG BATCHING: comfy `_calc_cond_batch` batches cond+uncond
  into ONE apply_model call (B==2) when cfg!=1, then `.chunk(batch_chunks)`. The engine forward is
  B==1-per-forward. So `_apply_model` UN-BATCHES: one denoise_step per cond group, each provably B==1,
  cfg_context_key = a SYMBOLIC key from comfy's per-conditioning uuid (_CtxKeyAssigner), NOT a content hash
  and NOT the 0/1 cond_or_uncond ROLE index. comfy batches by SHAPE only, so a stock
  ConditioningCombine/SetArea can put DIFFERENT content in ONE role bucket — a role key would collide them
  in the engine's cross-KV cache; a CONTENT hash is equally wrong (it collides two branches with identical
  content into the engine's STATEFUL fbcache_ slot — src/gemm/lighting/CLAUDE.md #B3). comfy's uuid is
  distinct per conditioning entry, content-independent, stable across steps, and position-independent
  (robust to a mid-run composition change) — see _CtxKeyAssigner.
  ★ COST (a real trade-off, NOT free): un-batching runs TWO sequential B==1 engine forwards per sampler
  step where comfy handed ONE batched B==2 — a genuine per-step latency cost on every cfg!=1 workflow. It
  is NOT optional today: the engine throws on B>1 (WanTransformerLighting.cpp:484), so batching the two
  cond groups into one forward is impossible until engine-side B>1 support lands (a separate, larger
  workstream — the standing `batch_size>1 across all models` goal). cfg==1 (distilled / no-CFG) runs ONE
  forward per step and is unaffected. This 2× forward cost is confirmed by the cfg=1-vs-cfg=7 A/B (both
  complete, MSE 4459 → the uncond forward genuinely runs; it is not skipped).
- ★ i2v REFERENCE (Option C): the engine's wan i2v channel-concat takes the reference frame as a FILE
  PATH it VAE-encodes ITSELF (WanVideoPipeline.cpp:754-825; the denoise ABI has no ref-latent field).
  It CANNOT consume comfy's VAE-encoded `concat_latent_image`. So the reference arrives as an IMAGE
  GRAPH input on the loader (fanned from a stock LoadImage), which we save to a disposable temp file
  and hand to denoise_begin_edit — the engine VAE-encodes the ORIGINAL pixels (no dependence on comfy
  having loaded a matching VAE). A user who ALSO wires WanImageToVideo.start_image produces a
  `concat_latent_image` the engine would silently ignore — extra_conds() FAILS LOUDLY on that (the
  silent-discard must be closed, not relocated).
- Session lifecycle: LAZY denoise_begin_edit on the FIRST step of a run; finalize+end at
  process_latent_out (normal end); extra_conds closes at RUN START any session STRANDED by an
  Interrupt (comfy skips process_latent_in for empty/denoise=1.0 latents + caches+reuses this
  instance across requeues), so the next run never continues an aborted one (its KV cache);
  end-on-failure per step.
- Co-eviction: when comfy needs the VRAM for a sibling model, the ModelPatcher frees the engine's VRAM
  (quantfunc_unload_sync) and reports the real freed bytes, so comfy's ledger is honest.
"""
import ctypes
import json
import os
import tempfile

import torch
import comfy.model_base
import comfy.model_patcher
import comfy.model_management
import comfy.conds

from . import qf_engine as qfe


def _qf_dtype(torch_dtype):
    return {torch.float32: qfe.QF_FP32, torch.float16: qfe.QF_FP16,
            torch.bfloat16: qfe.QF_BF16}[torch_dtype]


def _interrupt_poll_end_session_on_raise(qf):
    """SHARED per-cond-group interrupt poll that never strands an open engine session (ONE copy for
    QFWanModel + QFLTXModel — §6.5 simplicity: the session-clearing hardening was added to LTX and had
    drifted from WAN, the exact maintenance-double-cost this helper removes). comfy's
    throw_exception_if_processing_interrupted raises InterruptProcessingException, which subclasses
    BaseException (model_management.py), NOT Exception — so the guard ★ MUST be `except BaseException`
    (`except Exception` MISSES it; a self-CR test caught this on the LTX side). On ANY raise we end the
    open session FIRST (else current_session stays GPU-resident across a cached-node requeue; extra_conds'
    run-start close remains the outer net), then RE-RAISE — KeyboardInterrupt/SystemExit still propagate,
    we only clean up. No interrupt → pure no-op."""
    try:
        comfy.model_management.throw_exception_if_processing_interrupted()
    except BaseException:
        qf.end_session_if_open()
        raise


# The engine's "no trustworthy per-branch identity" sentinel: cfg_context_key == 0 (kNoCtxKey) makes the WAN
# lighting transformer DISABLE all three step caches and recompute BIT-EXACT (WanTransformerLighting.cpp's
# `have_ctx_key = raw_ctx_key != kNoCtxKey`; lighting_step_cache.h `kNoCtxKey`) — the sanctioned safe
# degradation "rather than falling back to a poisonable" reuse. NON-zero keys (>=1) enable caching.
_KNO_CTX_KEY = 0


class _CtxKeyAssigner:
    """Assigns each conditioning group a SYMBOLIC cfg_context_key derived from comfy's per-conditioning UUID
    (`transformer_options["uuids"][i]`, aligned with `cond_or_uncond`) — a stable non-zero int when a uuid is
    present, else 0 (kNoCtxKey, caches off). Reset ONCE per generation (in `_begin`). CONTENT-INDEPENDENT:
    the key labels *which conditioning*, never *what it contains*.

    WHY A UUID, AND WHY NOT A CONTENT HASH / ROLE INDEX / BATCH POSITION
    (src/gemm/lighting/CLAUDE.md #B3 — the hard death-rule this class exists to satisfy):
    The engine keys THREE per-step caches off this ABI value — the L2 memoization caches (`ctx_cache_`,
    `cross_kv_cache_`) AND the STATEFUL L3 First-Block/TeaCache trajectory cache (`fbcache_`; running
    `accum`/`prev_blk0`). The key MUST be:
      (a) DISTINCT per distinct conditioning — else the L2 cross-KV cache emits one branch's projected text
          for another. The 0/1 cond_or_uncond ROLE INDEX is NOT unique: comfy batches by SHAPE only
          (comfy/conds.py CONDRegular.can_concat), so a stock ConditioningCombine/SetArea puts two
          DIFFERENT-content conditionings in ONE role bucket → same key → wrong text reuse (the seq-219
          defect);
      (b) DISTINCT per branch even for IDENTICAL content — else two branches share the STATEFUL `fbcache_`
          slot and their trajectories interleave (why #B3 forbids a CONTENT HASH, and why the engine deleted
          its own content signature `ctx_sig`); and
      (c) STABLE for a given conditioning across a generation's steps INCLUDING under a within-generation
          COMPOSITION CHANGE. `ConditioningSetTimestepRange` makes comfy's `get_area_and_mult` return None
          for a cond outside its timestep window (`_calc_cond_batch` drops it from that step's batch), so a
          POSITION-derived key would renumber the survivors → a later step's key would HIT an earlier step's
          DIFFERENT-branch cross-KV = stale wrong-content reuse (the same defect class as (a), triggered by
          reordering; the L2 caches never re-verify content on a key hit).
    comfy's per-conditioning `uuid` — assigned once per generation in `sampler_helpers.convert_cond` via
    `uuid.uuid4()` (a FRESH uuid PER ENTRY, so even two identical-content conds a ConditioningCombine
    produces get DIFFERENT uuids), then carried each step in `transformer_options["uuids"]` — satisfies ALL
    THREE: DISTINCT per conditioning entry → (a) and (b); STABLE across the run's steps (fixed at setup);
    and INDEPENDENT of batch position, so dropping/adding a cond does not renumber the others → (c). It is
    neither a content hash nor a `data_ptr`/`id()`. Each distinct uuid is mapped to a small non-zero int
    (the ABI wants uint64) in FIRST-SEEN order. The engine uses the key only to LABEL a cache slot, so ANY
    stable distinct bijection yields byte-identical OUTPUT for a fixed conditioning set; on the standard
    cond+uncond path the first-seen order gives {1,2} == the old cou[i]+1 (observed on the box), so the plain
    cfg path is byte-identical to the original role-index. The fix diverges only for the multi-cond bucket
    (distinct keys) and under a composition change (each survivor's key stays stable). The per-generation
    reset keeps the map from leaking across generations and restarts the first-seen numbering.

    FALLBACK (comfy exposed NO uuid — an older/edge build): return 0 = the engine's kNoCtxKey sentinel, which
    makes the WAN transformer DISABLE all three step caches and recompute BIT-EXACT — the engine "cannot
    safely tell cond from uncond" without a trustworthy identity, so it refuses to cache rather than reuse a
    poisonable slot (WanTransformerLighting.cpp `have_ctx_key`). This is SAFE under ANY composition change (no
    caching ⇒ no cross-branch reuse), unlike a position-derived ordinal — which would reintroduce the very
    composition-change stale-reuse defect the uuid path removes (a survivor's position shifts on a drop and
    its new key HITs another branch's slot; the L2 caches never re-verify content on a hit). Both box comfy
    (0.17) and target (0.27) populate `uuids`, so this path is DORMANT; when it fires it only forfeits the
    cache speedup, never mis-reuses. A 0 also can never collide with a live uuid key (>=1) in a mixed call."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Per-GENERATION reset (called in `_begin`): clear the uuid→key map (no leak across generations;
        first-seen numbering restarts at 1)."""
        self._uuid_to_key = {}      # comfy conditioning uuid -> stable per-generation symbolic key

    def key(self, cond_uuid):
        """Symbolic cfg_context_key for this cond group. comfy's per-conditioning uuid → a stable NON-ZERO
        int (caching ON, stable across steps + composition changes); NO uuid → 0 = kNoCtxKey (the engine
        disables its step caches and recomputes bit-exact — the safe degradation, never a position reuse)."""
        if cond_uuid is None:
            return _KNO_CTX_KEY                       # no trustworthy identity -> engine disables caches (safe)
        k = self._uuid_to_key.get(cond_uuid)
        if k is None:
            k = len(self._uuid_to_key) + 1            # first-seen order -> {1,2,…}; non-zero (caching ON)
            self._uuid_to_key[cond_uuid] = k
        return k


class _QFStub(torch.nn.Module):
    """Empty stand-in for BaseModel.diffusion_model (comfy may .to(device) it — ~0 bytes).
    Carries the attrs comfy's sampling introspects (dtype). It is ALSO a deliberate FAIL-LOUD
    tripwire: any comfy path that stops routing through our _apply_model override lands here."""
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype

    def forward(self, *a, **k):
        raise RuntimeError("QF stub diffusion_model must never be called — _apply_model is overridden; "
                           "a ComfyUI upgrade may have changed the apply-model dispatch")


def save_ref_tempfile(image):
    """Write a loader start_image IMAGE (comfy [B,H,W,C] float 0..1) to a disposable temp PNG the
    engine can load_image()+VAE-encode (Option C — the engine encodes the pixels itself). Frame 0
    is the reference first frame. The file lives ONLY across the begin_edit call (caller deletes it
    in a finally) — no user images accumulate. SHARED by the WAN and LTX i2v seams."""
    from PIL import Image
    import numpy as np
    img = image
    if img.dim() == 4:
        img = img[0]                                  # [H,W,C]
    arr = (img.clamp(0, 1).cpu().float().numpy() * 255.0 + 0.5).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = arr[:, :, :3]
    fd, path = tempfile.mkstemp(suffix=".png", prefix="qf_native_ref_")
    os.close(fd)
    try:
        Image.fromarray(arr).save(path)
    except Exception:
        cleanup_ref_tempfile(path)   # a failed .save() must not leak the just-created temp file
        raise
    return path


def cleanup_ref_tempfile(path):
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


class QFSessionModelMixin:
    """Shared base for every QuantFunc native-session model wrapper (Wan / LTX / — coming — H3).

    These wrappers all drive the SAME engine external-denoise seam (begin → per-sampler-step
    quantfunc_denoise_step → finalize) over comfy's stock KSampler, differing only in the
    MODEL-SPECIFIC parts (context source, latent packing, per-step param wiring, output shape).
    This mixin holds the parts that are IDENTICAL across models so they live once, not once per
    class; each model subclasses it alongside its comfy base
    (`class QFWanModel(QFSessionModelMixin, comfy.model_base.WAN21)`).

    INCREMENT 1 (this commit — behavior-PRESERVING, low-risk): the two blocks that are
    character-identical in QFWanModel._apply_model and QFLTXModel._apply_model — the
    sigma-schedule-derived step index and the quantfunc_denoise_step call + session-clearing
    error handling — are extracted here VERBATIM. Each model's _apply_model now calls these
    instead of inlining them; the produced values/strings are unchanged (the only per-model
    difference, the failure-message prefix, is passed in). No control flow is restructured.

    INCREMENT 2 (follow-up, gated on the Wan/LTX GPU regression run): promote the rest of the
    shared _apply_model skeleton (require-ctx / refuse-control / batch-refuse / begin-gate / the
    per-cond-group loop with interrupt-poll + counters / finalize) into this mixin with hooks for
    the divergent parts (_prepare_context, _begin_session, _alloc_out, _fill_and_run_step,
    _final_prediction). That restructure changes the shape of behavior-critical generation code,
    so it must be proven byte-identical by a real Wan+LTX generate regression, not just review —
    deferred until a GPU box is available (same gate as the H3 fl2va proof).
    """

    def _sigma_step_index(self, sigma, sig_all, transformer_options):
        """SAMPLER step index from the sigma's position in the schedule — call-count-independent.

        With CONDRegular a different-length cond/uncond arrive as SEPARATE _apply_model calls (2
        per sampler step, not one batched B==2), so a per-call counter double-counts and overruns
        total_steps. The sigma is identical for cond and uncond of the same step, so its schedule
        index is the true step index. Falls back to the per-call counter (self._step_i) when the
        sampler exposes no schedule. Verbatim-shared by QFWanModel + QFLTXModel."""
        step_index = self._step_i
        _sched = transformer_options.get("sample_sigmas") if isinstance(transformer_options, dict) else None
        if _sched is not None and len(_sched) >= 2:
            _cur = float(sig_all[0].item()) if sig_all is not None else float(sigma)
            step_index = min(range(len(_sched) - 1), key=lambda k: abs(float(_sched[k]) - _cur))
        return step_index

    def _call_denoise_step(self, p, fail_prefix):
        """Run one quantfunc_denoise_step, clearing the (GPU-resident) session on ANY failure so a
        mid-sample error never strands a stale session. `fail_prefix` is the model-specific
        message head (e.g. 'denoise_step[step=..,group=..,key=..]' for WAN, 'LTX denoise_step[..]'
        for LTX). Verbatim-shared by QFWanModel + QFLTXModel."""
        try:
            st = self._qf.lib.quantfunc_denoise_step(self._qf.current_session, ctypes.byref(p))
        except Exception:
            self._qf.end_session_if_open()
            raise
        if st != qfe.QUANTFUNC_OK:
            err = qfe.last_err(self._qf.lib)
            self._qf.end_session_if_open()          # never strand a stale session on a mid-sample failure
            raise RuntimeError(f"{fail_prefix} failed: {err}")

    def _call_denoise_step_multi(self, mp, fail_prefix):
        """Run one quantfunc_denoise_step_multi (joint-AV step), clearing the (GPU-resident)
        session on ANY failure — the AV twin of _call_denoise_step. Shared by QFH3Model (H3
        joint AV) and QFLTXAVModel (LTX-2.5 joint AV); hoisted here from qf_h3_modelpatcher
        so the second AV consumer does not copy it (polymorphism-over-copy mandate)."""
        lib = self._qf.lib
        if not hasattr(lib, "quantfunc_denoise_step_multi"):
            self._qf.end_session_if_open()
            raise RuntimeError(f"{fail_prefix}: the loaded engine .so has no "
                               "quantfunc_denoise_step_multi (D-class AV step) — rebuild/point "
                               "the engine lib at an AV-capable build.")
        try:
            st = lib.quantfunc_denoise_step_multi(self._qf.current_session, ctypes.byref(mp))
        except Exception:
            self._qf.end_session_if_open()
            raise
        if st != qfe.QUANTFUNC_OK:
            err = qfe.last_err(lib)
            self._qf.end_session_if_open()
            raise RuntimeError(f"{fail_prefix} failed: {err}")


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
        # concat_latent_image. Silently discarding it is the exact defect the CR NO-GO'd — FAIL LOUD and
        # point the user at the loader's own start_image input.
        if kwargs.get("concat_latent_image") is not None:
            raise RuntimeError(
                "qf_native: WanImageToVideo.start_image is wired, but the QuantFunc engine takes the "
                "reference frame through the QuantFuncNativeWanLoader's OWN 'start_image' IMAGE input "
                "(it VAE-encodes the pixels itself and cannot use comfy's encoded concat_latent_image). "
                "Wire your LoadImage into the LOADER's start_image and leave WanImageToVideo.start_image "
                "EMPTY — otherwise the reference frame would be silently ignored.")
        # ★ CLOSE THE SILENT-DISCARD CLASS: every OTHER image/reference channel WAN21.extra_conds consumes
        # that this bypass would drop (clip_fea / time_dim / reference / context / camera). The engine does
        # its OWN i2v conditioning from the loader's start_image, so any of these wired from a stock node
        # (CLIPVisionEncode→clip_vision_output, BerniniConditioning→context_latents, …) must FAIL LOUD.
        for _k in self._ENGINE_IGNORED_COND_KEYS:
            if kwargs.get(_k) is not None:
                raise RuntimeError(
                    f"qf_native: '{_k}' conditioning is wired, but the QuantFunc wan native session does "
                    f"its OWN i2v/reference conditioning from the loader's start_image — it cannot consume "
                    f"comfy's '{_k}', which would be silently ignored. Remove the node feeding '{_k}' "
                    f"(the engine derives all i2v conditioning from the loader's start_image reference).")
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
            "mask against the noise latent and corrupt the i2v result. Remove the mask / SetLatentNoiseMask "
            "node (the engine derives the reference from the loader's start_image, not a latent mask).")

    def _derive_geometry(self, xin, transformer_options):
        """DERIVE the session geometry from the graph (official-loader shape — the loader has no
        geometry widgets). xin: [B,C,Tlat,Hl,Wl] from WanImageToVideo / EmptyHunyuanLatentVideo;
        the step count from the sampler's own sigma schedule. The only refusal left is the one
        that is a REAL incompatibility, not a widget disagreement: a TRIMMED sigma range
        (KSamplerAdvanced start_step/last_step) would mis-time the engine's internal dual-expert
        swap, which is keyed to a full-range schedule."""
        Tlat = int(xin.shape[2])
        self._num_frames = (Tlat - 1) * self._VAE_T + 1
        sigmas = transformer_options.get("sample_sigmas") if isinstance(transformer_options, dict) else None
        if sigmas is None or len(sigmas) < 2:
            raise RuntimeError(
                "qf_native: the sampler did not publish a sigma schedule (transformer_options["
                "'sample_sigmas']) — the engine session needs the step count. Use a stock KSampler / "
                "SamplerCustom on this model.")
        self._num_steps = len(sigmas) - 1
        # A trimmed sub-range (denoise<1 / start_step>0 / last_step<steps) does not START at the
        # schedule's first sigma or END at ~0 — the engine's dual-expert swap timing assumes the
        # full range, so refuse LOUD instead of rendering a silently mis-timed result.
        try:
            s_first, s_last = float(sigmas[0]), float(sigmas[-1])
        except Exception:  # noqa: BLE001 — non-tensor sigmas: skip the range check, keep the count
            return
        ms = getattr(self, "model_sampling", None)
        s_max = float(getattr(ms, "sigma_max", s_first)) if ms is not None else s_first
        if s_last > 1e-3 or (s_max > 0 and s_first < 0.98 * s_max):
            raise RuntimeError(
                f"qf_native: partial / trimmed denoise is NOT supported (sigmas run "
                f"{s_first:.4f}→{s_last:.4f}, full range would be {s_max:.4f}→0). The engine session "
                f"runs its OWN internal schedule (including the dual-expert swap timing) keyed to the "
                f"full step range, so a trimmed sub-range would mis-time the swap and corrupt the "
                f"output. Use a single full-range KSampler (denoise=1.0, no start_step/last_step).")

    # ---- session lifecycle -------------------------------------------------
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


class QFLazyEngine:
    """A QFEngineHandle that materializes ON FIRST REAL USE.

    WHY (measured): sidecar LoRA is applied at pipeline CREATE time, so a chained
    QuantFuncNativeLoRA node has to build a pipeline for its accumulated LoRA set. Doing that
    EAGERLY in the node makes an N-node chain create N+1 pipelines — and comfy's node-output
    cache pins every intermediate model, so each intermediate ALSO pins a multi-GB CPU backup.
    Deferring means only the model the sampler actually touches is ever created.

    The proxy answers the CHEAP part of the handle surface locally while unmaterialized (an
    engine that does not exist holds no VRAM and has no session), and materializes only for
    the members that genuinely need a live pipeline (`lib`, `pipeline`). Explicit members, no
    __getattr__ magic: a typo must fail loud, not silently forward.
    """

    def __init__(self, factory, footprint_bytes):
        self._factory = factory          # () -> (QFEngineHandle, ckey)
        self._real = None
        self.footprint_bytes = int(footprint_bytes)   # from the package weights: no create needed
        # Nothing created yet => nothing resident. Reporting "unloaded" keeps comfy's ledger
        # HONEST (loaded_size -> 0) for a chain link the sampler never touches.
        self._unloaded = True
        self.step_count = 0
        self.sampler_step_count = 0

    # ---- materialization ----
    def ensure(self):
        if self._real is None:
            self._real, _ckey = self._factory()
            self._real.step_count = self.step_count
            self._real.sampler_step_count = self.sampler_step_count
            self.footprint_bytes = int(self._real.footprint_bytes)
        return self._real

    @property
    def materialized(self):
        return self._real is not None

    # ---- members that NEED a live pipeline ----
    @property
    def lib(self):
        return self.ensure().lib

    @property
    def pipeline(self):
        # Unmaterialized = "no pipeline yet", which is what every caller's `is not None` guard
        # means. Materializing here would defeat the deferral (comfy probes this while planning).
        return None if self._real is None else self._real.pipeline

    # ---- state that is meaningful WITHOUT a pipeline ----
    @property
    def current_session(self):
        return None if self._real is None else self._real.current_session

    @current_session.setter
    def current_session(self, v):
        self.ensure().current_session = v

    @property
    def unloaded(self):
        # Nothing created => nothing resident. Honest for comfy's ledger (loaded_size -> 0).
        return self._unloaded if self._real is None else self._real.unloaded

    @unloaded.setter
    def unloaded(self, v):
        if self._real is None:
            self._unloaded = bool(v)
        else:
            self._real.unloaded = bool(v)

    # ---- lifecycle: all no-ops while unmaterialized (nothing exists to close/free/destroy) ----
    def end_session_if_open(self):
        return (False, True) if self._real is None else self._real.end_session_if_open()

    def unload_vram(self):
        return 0 if self._real is None else self._real.unload_vram()

    def destroy(self):
        if self._real is not None:
            self._real.destroy()


class QFModelPatcher(comfy.model_patcher.ModelPatcher):
    """ModelPatcher over an engine-managed pipeline. The engine owns + moves its own VRAM, so the
    weight-move overrides are no-ops — but model_size/loaded_size REPORT the engine's real footprint
    so ComfyUI's memory ledger ACCOUNTS FOR the engine's multi-GB VRAM, and co-EVICTION actually frees
    it (quantfunc_unload_sync) rather than letting comfy pop us while we stay resident (falsified
    ledger → sibling OOM)."""
    def _engine(self):
        return getattr(getattr(self, "model", None), "_qf", None)

    def _footprint(self):
        eng = self._engine()
        return max(1, int(eng.footprint_bytes)) if eng is not None else 1

    def model_size(self):
        return self._footprint()

    def loaded_size(self):
        # 0 while unloaded (VRAM genuinely freed) so comfy's ledger is honest across a co-eviction;
        # the footprint while resident.
        eng = self._engine()
        if eng is not None and getattr(eng, "unloaded", False):
            return 0
        return self._footprint()

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        # The engine reloads lazily inside the next denoise_begin (its ~3s-grace auto-reload). Clearing
        # the flag lets model_size/loaded_size report the footprint again; no proactive torch load.
        eng = self._engine()
        if eng is not None:
            eng.unloaded = False
        return 0

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        # Co-eviction (#4): comfy needs VRAM for a sibling → ACTUALLY free the engine's VRAM
        # (unload_sync → GPU->CPU, reloads on next generate) and return the REAL freed bytes so comfy's
        # ledger is honest. The old 0-return let comfy pop us + believe multi-GB was free while the
        # engine stayed fully resident → the next model loaded into that "freed" space OOM'd.
        eng = self._engine()
        if eng is None:
            return 0
        return eng.unload_vram()

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True,
                    force_patch_weights=False):
        # The engine owns its weights (no torch UNet to move) so NEVER load_weights — but the base
        # object_patches loop (CFG-rescale / set_model_* / sampling-schedule patches from stock nodes)
        # MUST still run, else those patches silently no-op (#5). Delegate with load_weights=False.
        return super().patch_model(device_to=device_to, lowvram_model_memory=lowvram_model_memory,
                                   load_weights=False, force_patch_weights=force_patch_weights)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        # Restore object_patches (base handles the no torch-weight case cleanly since backup is empty).
        return super().unpatch_model(device_to=device_to, unpatch_weights=False)
