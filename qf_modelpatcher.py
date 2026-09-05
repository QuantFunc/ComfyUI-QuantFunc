"""qf_native.qf_modelpatcher — the FAMILY-AGNOSTIC seam substrate: a comfy ModelPatcher whose
shim `.model` (built by a family module) drives quantfunc_denoise_step instead of a torch UNet.

Design (measured from comfy 0.27.0 + include/quantfunc.h + the proven native_session_video_t1.py):
- `.model` = a FAMILY subclass of the matching comfy.model_base class, built with
  unet_config["disable_unet_model_creation"] (the family modules own those subclasses)
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
  content into the engine's STATEFUL block-cache state slot — src/gemm/lighting/CLAUDE.md #B3). comfy's uuid is
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
import hashlib
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


# Lazy-detach reclaim window. comfy's unload_model_clones detaches the OLD clone of this
# SAME model milliseconds before loading the NEW clone (measured on the LTX two-stage
# workflow: the reload began 27 ms after the full offload finished) — a bureaucratic clone
# swap, not memory pressure. A full engine unload there costs the whole weight set round-trip
# (measured: ~8.4 s extra in the next stage's first step). So detach() keeps the engine
# RESIDENT and arms this one-shot window: a successor clone's load/step RECLAIMS the engine
# (zero reload); no successor within the window ⇒ the real unload runs, so a workflow switch
# still frees VRAM within seconds. 15 s = 500x the measured 27 ms swap gap, small enough that
# a genuine drop frees before a human notices.
#
# 2026-09-05 (LTX-2.5 acceptance, measured on the user box): 15 s EXPIRED INSIDE EVERY RUN — the
# window armed after the main sampler ran out during the refine sampler / VAE decode, so the next
# run's main node paid the full reload (3.0 s: 12,597 per-parameter cudaMalloc + ~11 GB pageable
# H2D, nsys W10) plus the engine's per-geometry rope rebuild (~1.2 s). Real VRAM pressure does
# NOT depend on this timer: comfy's free_memory sweep reaches partially_unload() (engine
# quantfunc_partial_unload sheds blocks, full unload as the fallback) and cancels the window
# (_qf_cancel_pending_detach) — so the window only bounds how long an IDLE engine keeps VRAM
# nobody asked for. 120 s covers a whole multi-stage run plus the gap to the next queue entry;
# model-agnostic (the policy lives here, not in any loader).
_QF_LAZY_DETACH_SECONDS = 120.0


def _qf_cancel_pending_detach(eng):
    """Reclaim a lazily-detached engine (a successor clone took over): cancel the one-shot
    unload timer. Safe no-op when nothing is pending."""
    if eng is None or not getattr(eng, "pending_detach", False):
        return
    lock = getattr(eng, "_qf_detach_lock", None)
    if lock is None:
        return   # pending without a lock cannot happen (detach creates the lock first)
    with lock:
        eng.pending_detach = False
        t = getattr(eng, "_qf_detach_timer", None)
        if t is not None:
            t.cancel()
            eng._qf_detach_timer = None
    if os.environ.get("QF_NATIVE_PROF") == "1":
        print("[qf_prof] lazy-detach RECLAIMED (engine stayed resident, zero reload)", flush=True)


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
# `have_ctx_key = raw_ctx_key != kNoCtxKey`; the engine's cache layer `kNoCtxKey`) — the sanctioned safe
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
    `cross_kv_cache_`) AND the STATEFUL L3 block-level trajectory cache (`the engine's block-cache state slot `; running
    `accum`/`prev_blk0`). The key MUST be:
      (a) DISTINCT per distinct conditioning — else the L2 cross-KV cache emits one branch's projected text
          for another. The 0/1 cond_or_uncond ROLE INDEX is NOT unique: comfy batches by SHAPE only
          (comfy/conds.py CONDRegular.can_concat), so a stock ConditioningCombine/SetArea puts two
          DIFFERENT-content conditionings in ONE role bucket → same key → wrong text reuse (the seq-219
          defect);
      (b) DISTINCT per branch even for IDENTICAL content — else two branches share the STATEFUL `the engine's block-cache state slot `
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

    def arm_concat_shape(self, in_channels):
        """Give comfy's BaseModel.concat_cond the ONE attribute it introspects
        (diffusion_model.patch_embedding.weight.shape[1] = the transformer's total input
        channels) so the STOCK concat machinery works on this stub: a t2v checkpoint
        (in == latent channels) yields extra_channels==0 -> c_concat None (byte-unchanged);
        a channel-concat i2v checkpoint (in > latent) yields comfy's own [mask|image] tail —
        the SAME layout+semantics the engine's cond-ABI consumes (verified against comfy
        model_base.WAN21.concat_cond: cat((mask, image)) with mask = 1-mask -> frame0=1,
        matching the engine's [noise|mask|cond] with frame0=1). The carrier is a ~0-byte
        empty-data parameter whose SHAPE alone matters."""
        pe = torch.nn.Module()
        pe.weight = torch.nn.Parameter(
            torch.empty(1, int(in_channels), 0, 0, 0, dtype=self.dtype), requires_grad=False)
        self.patch_embedding = pe

    def forward(self, *a, **k):
        raise RuntimeError("QF stub diffusion_model must never be called — _apply_model is overridden; "
                           "a ComfyUI upgrade may have changed the apply-model dispatch")


def make_engine_factory(get_engine_fn, bind_pipeline_model):
    """[P5, independent-CR 2026-08-30] The ONE deferred-create factory shape every
    family builder previously hand-copied 4x (Wan / H3 / LTX-AV / LTX-video — that
    exact drift left 3 of 4 sites unfixed in the leak class): a weakref LIST of
    this build-chain's models (weakrefs so a superseded chain model can still be
    GC'd — the Wan discipline, no model<->engine cycle) + a factory that, at real
    engine create, binds every SURVIVING model to the resolved ckey.

    Returns (factory, register_model): pass `factory` to QFLazyEngine; call
    `register_model(m)` for every model the build creates (Wan calls it twice)."""
    import weakref as _weakref
    engine_models = []
    def factory():
        eng, ckey = get_engine_fn()
        for _wr in engine_models:
            _m = _wr()
            if _m is not None:
                bind_pipeline_model(ckey, _m)
        return eng, ckey
    def register_model(m):
        engine_models.append(_weakref.ref(m))
    return factory, register_model


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
    """Shared base for every QuantFunc native-session model wrapper (wan / LTX-2 / MiniMax-H3).

    These wrappers all drive the SAME engine external-denoise seam (begin → per-sampler-step
    quantfunc_denoise_step → finalize) over comfy's stock KSampler, differing only in the
    MODEL-SPECIFIC parts (context source, latent packing, per-step param wiring, output shape).
    This mixin holds the parts that are IDENTICAL across models so they live once, not once per
    class; each model subclasses it alongside its comfy base
    (e.g. `class QFWanModel(QFSessionModelMixin, comfy.model_base.WAN21)` in the wan family
    module — this substrate is family-AGNOSTIC and defines no model class itself).

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

    # ── [manual-residency] runtime-adjustable resident block count (GENERIC layer) ──
    # The native seam's ONLY residency knob is a SESSION parameter, not a pipeline-identity
    # parameter: every family merges residency_opts() into its denoise_begin options_json, and
    # the engine re-applies it at EVERY session begin (applyManualResidentBlocks →
    # setResidentBlockPrefix — bidirectional: shrink frees blocks [tgt,current), grow reloads
    # [current,tgt) from backups, never-OOM-guarded). A widget change therefore takes effect on
    # the NEXT run with NO pipeline rebuild. MEASURED (3090, wan A14B dual-expert, one resident
    # comfy process, 2026-08-22): rb30→rb10 second run success in 88.6s (pure gen time, no
    # create) with "MANUAL residency — 10 → target 10 of 40 (achieved 10)" on both experts and
    # GPU used 16322 → 11806 MiB (the shed blocks really freed). The STRUCTURAL half of the
    # guarantee — the knob never entering create_cfg/ckey (which would rebuild the pipeline on
    # every widget change) — is enforced fail-loud by _refuse_session_knobs_in_create() at the
    # single engine-create chokepoint (__init__._get_engine).
    _resident_block_count = 999   # class default; families set the widget value in __init__
    # ── [step-cache] runtime step-cache threshold (SAME session-knob class as residency:
    # merged into the denoise_begin options_json below, re-read by the engine at EVERY
    # session begin → a widget change takes effect on the NEXT run with NO rebuild; the
    # knob never enters create_cfg/ckey). 0.0 = OFF: the keys are OMITTED entirely, the
    # engine spec stays cache_mode=0/thresh=0 → the step path is BYTE-IDENTICAL (the
    # engine-side off-path guarantee, the engine's cache layer the engine's step-cache wrapper). >0 arms
    # lighting::CacheMode::the step cache with this mean_abs_diff skip budget. ──
    _step_cache = 0.0             # class default; loaders set the widget value (step_cache)
    _block_cache = 0.0            # class default; loaders set the widget value (block_cache)
    _sparse = 1.0                 # class default; 1.0 = dense (loaders set the sparse widget)

    def set_resident_block_count(self, n):
        self._resident_block_count = int(n)

    def set_step_cache(self, t):
        self._step_cache = float(t)

    def set_block_cache(self, t):
        self._block_cache = float(t)

    def set_sparse(self, s):
        self._sparse = float(s)

    def set_attn_backend(self, v):
        # [runtime dial 2026-08-29] session knob — rides residency_opts() into EVERY
        # denoise_begin; the engine swaps the per-forward dispatch string (no rebuild).
        self._attn_backend = str(v or "auto")

    def set_sol_tau(self, v):
        # [sol-tau dial 2026-08-31] the ONE user-facing Sol-Attn knob (user "就一个
        # 就好"): the qfa=sol route's z-score keep threshold. LOWER = more exact
        # blocks (denser/slower; <= -8 = the engine routes true DENSE qfa), HIGHER =
        # sparser/faster. Same rides-residency_opts session-knob class as
        # set_attn_backend; only meaningful under attention_backend=qfa.
        try:
            self._sol_tau = float(v)
        except (TypeError, ValueError):
            self._sol_tau = 1.0

    def residency_opts(self):
        """The begin-options fragment EVERY family merges into its options_json — the ONE
        injection point for ALL runtime session knobs (residency + the two cache
        thresholds + the sparse dial; same non-ckey, re-applied-at-every-begin class →
        a widget change takes effect on the NEXT run with NO pipeline rebuild). Cache
        keys are OMITTED at thresh<=0 and the sparse key at >=1.0, so every OFF path
        is byte-identical. step_cache (STEP-level) and block_cache (BLOCK-level) are
        ORTHOGONAL and COMPOSABLE (user 2026-08-24): the step cache decides whether a
        step runs at all; the block cache decides, inside a computed step, whether the
        remaining blocks run — a step-skipped step never touches the transformer, so
        block-cache state simply carries over. An OLDER engine refuses an unknown key
        LOUD — honest, never silent."""
        o = {"resident_block_count": int(self._resident_block_count)}
        te = float(getattr(self, "_step_cache", 0.0) or 0.0)
        tf = float(getattr(self, "_block_cache", 0.0) or 0.0)
        sp = float(getattr(self, "_sparse", 1.0) or 1.0)
        if te > 0.0:
            o["step_cache_thresh"] = te
        if tf > 0.0:
            o["block_cache_thresh"] = tf
        # sparse_cdf is sent UNCONDITIONALLY (including 1.0): the visible panel
        # value must ALWAYS override whatever the pipeline currently runs —
        # 1.0 explicitly resolves to dense in the engine, so a cached/created
        # pipeline carrying an old sparse state can never ghost a stale dial
        # onto a user who dialed it back to 1.0 (the measured 0.8-ghost class).
        # Engine-side: every session-capable video pipeline accepts the key
        # (begin capability gate), and this mixin is video-family-only.
        o["sparse_cdf"] = sp
        ab = str(getattr(self, "_attn_backend", "auto") or "auto")
        if ab != "auto":
            o["attention_backend"] = ab   # [runtime dial] auto = engine default, key omitted
        # [sol-tau dial] omitted at the 1.0 default (old-engine compatible: an older
        # .so refuses unknown keys LOUD, and default users never send it). Ghost-proof
        # despite the omission: the ENGINE resets an absent sol_tau to 1.0 at every
        # begin (absent = reset-to-default, not keep-current) — so dialing back to
        # 1.0 truly restores the default even on a reused engine-resident pipeline.
        st = float(getattr(self, "_sol_tau", 1.0) or 1.0)
        if abs(st - 1.0) > 1e-6:
            o["sol_tau"] = st
        return o

    def _assert_wire_lora(self):
        """[wiring-lora] RUN-START side assert (reviewer-C correctness fix): THIS model's wire
        is the AUTHORITY for its side — write its own QF_LORA_STACK_ATTR (the wire's stack;
        [] on a raw loader output) into the shared engine's side registry at every run, so a
        registry entry can never OUTLIVE the node that authored it. Deleting a LoRA node and
        rewiring the sampler to the raw output therefore resets that side to [] on the next
        run (no silent LoRA pollution — the defect reviewer C executed). Idempotent for
        unchanged wiring; models without a high/low wire tag (single-expert families) no-op."""
        side = getattr(self, QF_EXPERT_ATTR, "all")
        if side not in ("high", "low"):
            return
        eng = getattr(self, "_qf", None)
        if eng is None or not hasattr(eng, "set_lora_side"):
            return
        eng.set_lora_side(side, list(getattr(self, QF_LORA_STACK_ATTR, []) or []))

    # ── comfy inference-memory estimate override (inter-stage thrash fix, 2026-08-22) ──
    # comfy's load_models_gpu decides evictions from BaseModel.memory_required(input_shape),
    # which for a torch WAN at seq_len 33600 estimates GBs of activation memory. Our engine
    # manages its OWN working VRAM inside the session (the estimate is not just wrong — acting
    # on it is harmful): measured on the user's 4090 log, loading the SECOND stage's 64MB
    # shadow patcher triggered "All models offloaded to CPU" on the primary (a ~15s/prompt
    # engine offload+reload round-trip between the two KSamplerAdvanced stages). The comfy-side
    # truth for this wrapper is just the latent/cond tensors + conversion scratch — a fixed
    # small cushion. Engine working memory is the ENGINE's ledger, reported via the patcher's
    # model_size/loaded_size (already wired), not via inference estimates.
    # D3/D4 (delta CR): the estimate is GEOMETRY-PROPORTIONAL and the signature matches the
    # REAL call site — comfy/sampler_helpers.py estimate_memory() calls
    # `memory_required(shape, cond_shapes=cond_shapes)` (keyword!) on EVERY standard
    # KSampler run; the earlier positional-only `(self, input_shape)` override raised
    # TypeError there (D4 NO-GO — the 6/6 suite arm called positionally and proved only the
    # return value, not the calling convention). `**_kw` tolerates future comfy drift.
    # Honest accounting (D3): comfy-side bytes for this wrapper are the latent in/out copies
    # + cond tensors + conversion scratch — proportional to geometry, NOT the torch-WAN
    # activation estimate (2.36 GB at 33.6k tok; acting on it evicted the engine = the
    # measured 15 s/prompt inter-stage thrash), and NOT a flat constant that under-reports
    # huge geometries to OTHER tenants' eviction math. K factors: input + noise + output
    # velocity + c_concat tail + ~4x conversion/temporary copies ≈ 8 latent-sized tensors;
    # cond tensors counted 2x (borrowed + converted). Floor keeps small runs honest.
    _QF_COMFY_SIDE_BASE_BYTES = 64 * 1024 * 1024
    _QF_COMFY_SIDE_LATENT_COPIES = 8
    _QF_COMFY_SIDE_COND_COPIES = 2
    _QF_COMFY_SIDE_ITEMSIZE = 2  # fp16/bf16 latents+conds

    def memory_required(self, input_shape, cond_shapes=None, **_kw):
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
        return int(self._QF_COMFY_SIDE_BASE_BYTES +
                   self._QF_COMFY_SIDE_ITEMSIZE * (n_lat * self._QF_COMFY_SIDE_LATENT_COPIES +
                                                   n_cond * self._QF_COMFY_SIDE_COND_COPIES))

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
        _qf_cancel_pending_detach(self._qf)   # an active step supersedes any lazy-detach window
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
        _qf_cancel_pending_detach(self._qf)   # an active step supersedes any lazy-detach window
        lib = self._qf.lib
        if not hasattr(lib, "quantfunc_denoise_step_multi"):
            self._qf.end_session_if_open()
            raise RuntimeError(f"{fail_prefix}: the loaded engine .so has no "
                               "quantfunc_denoise_step_multi (D-class AV step) — rebuild/point "
                               "the engine lib at an AV-capable build.")
        import os as _os
        _prof = _os.environ.get("QF_NATIVE_PROF") == "1"
        if _prof:
            import time as _time
            _t0 = _time.perf_counter()
        try:
            st = lib.quantfunc_denoise_step_multi(self._qf.current_session, ctypes.byref(mp))
        except Exception:
            self._qf.end_session_if_open()
            raise
        if _prof:
            print(f"[qf_prof] engine_call {(_time.perf_counter()-_t0)*1000:.0f} ms", flush=True)
        if st != qfe.QUANTFUNC_OK:
            err = qfe.last_err(lib)
            self._qf.end_session_if_open()
            raise RuntimeError(f"{fail_prefix} failed: {err}")


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

    def __init__(self, factory, footprint_bytes, retire=None):
        self._factory = factory          # () -> (QFEngineHandle, ckey)
        self._real = None
        self._ckey = None                # the materialized handle's cache key (for retire)
        # [wiring-lora] per-SIDE declarative LoRA registry (dual-expert, ONE shared engine).
        # Each chained LoRA node REPLACES its wire-side's whole cumulative stack, and EVERY
        # wire re-writes its own truth at run start via _assert_wire_lora (a raw loader
        # output writes []) — so re-execution converges by construction and a DELETED node's
        # side is reset by the raw wire's next run. Loader-cached state is never appended to.
        # The factory reads lora_union() at CREATE; reconcile_lora() retires a handle whose
        # created union no longer matches.
        self._lora_sides = {}
        self._created_lora_sig = None
        # retire(ckey, eng, requester, *, keep_binding=False, reason="") -> bool: the cache
        # layer's SINGLE liveness-gated retire chokepoint (the only .destroy() site in the
        # plugin). None => this wrapper can NEVER destroy (fail-safe: a wrapper that cannot
        # prove exclusivity must not free a handle a sibling may hold).
        self._retire = retire
        self.footprint_bytes = int(footprint_bytes)   # from the package weights: no create needed
        # Nothing created yet => nothing resident. Reporting "unloaded" keeps comfy's ledger
        # HONEST (loaded_size -> 0) for a chain link the sampler never touches.
        self._unloaded = True
        self.step_count = 0
        self.sampler_step_count = 0

    # ---- materialization ----
    def ensure(self):
        if self._real is not None and self._real.pipeline is None:
            # The real handle was DESTROYED under us (a sibling's lifecycle path, or any future
            # one). Never hand a NULL pipeline to denoise_begin — drop it and re-create from disk.
            self._real = None
        # [wiring-lora] drift-retire lives HERE, at the ONE materialization chokepoint (CR
        # generality fix): a family cannot forget a per-begin hook that does not exist. Cheap
        # (a small json sig compare); sides only mutate during node execution (sequential,
        # pre-sampling), so an open mid-generation session never sees a drift.
        self.reconcile_lora()
        if self._real is None:
            self._real, self._ckey = self._factory()
            self._real.step_count = self.step_count
            self._real.sampler_step_count = self.sampler_step_count
            self.footprint_bytes = int(self._real.footprint_bytes)
            self._created_lora_sig = self._lora_sig()   # [wiring-lora] union this create embeds
        return self._real

    @property
    def materialized(self):
        return self._real is not None

    # ---- [wiring-lora] per-side LoRA registry (see __init__ docblock) ----
    # DUAL-OUTPUT FAMILY ONBOARDING (wan = the reference implementation): a new multi-output
    # family needs exactly FOUR wirings — (1) tag each output model with QF_EXPERT_ATTR at
    # build AND tag the raw pair's wire stacks EMPTY (loader-level entries live only in the
    # "loader" side), (2) construct the shared QFLazyEngine with
    # retire=deps["retire_handle"] (the single liveness-gated destroy chokepoint),
    # (3) route its LoRA rebuild through set_lora_side + return a fresh same-side patcher
    # (see wan _lora_rebuild_dual), (4) call self._assert_wire_lora() at the model's
    # RUN-START hook (wan: extra_conds) so a side entry can never outlive its author node.
    # DELETION SEMANTICS (the honest claim, reviewers C+E): deleting a LoRA node on EITHER
    # wire converges at the next run — both wires re-assert their truth at their run hooks
    # before the next session opens (reviewer-E measured: under comfy's sequential
    # single-worker execution the mid-generation drift guard in reconcile_lora is NOT
    # reachable — it is DEFENSIVE, kept for any future concurrent/out-of-band mutation
    # path). Never a silent stale-LoRA output.
    # Single-expert families deliberately keep the OTHER architecture — a fresh _build per
    # LoRA set (one consumer, no shared engine to keep coherent) — do not migrate them
    # without need. Half-adoption fails LOUD: set_lora_side refuses without a retire chokepoint
    # (else a superseded handle would silently leak its host backup), and the drift-retire
    # runs inside ensure() itself (no per-family hook to forget).
    def set_lora_side(self, side, entries):
        """REPLACE one wire-side's cumulative LoRA stack ('high'/'low'/'all')."""
        if self._retire is None:
            raise RuntimeError(
                "QFLazyEngine.set_lora_side: this engine was constructed WITHOUT retire — "
                "the per-side LoRA registry would silently orphan superseded handles (host-RAM "
                "leak). Pass retire=deps['retire_handle'] at the family's QFLazyEngine(...) "
                "construction (see the onboarding note above).")
        self._lora_sides[str(side)] = [dict(e) for e in (entries or [])]

    def lora_union(self):
        """The engine-create 'lora' list: every side's entries, side-sorted for determinism
        ('high' < 'loader' < 'low'). The SORT is for signature stability only — sidecar LoRA
        entries are per-layer ADDITIVE low-rank columns applied per their own target, so the
        relative order of sides is mathematically inert (within one wire the user's chain
        order is preserved). A future NON-additive merge semantic would need a real order
        contract here."""
        out = []
        for side in sorted(self._lora_sides):
            out.extend(dict(e) for e in self._lora_sides[side])
        return out

    def _lora_sig(self):
        return json.dumps(self.lora_union(), sort_keys=True)

    def reconcile_lora(self):
        """Session-BEGIN hook: LoRA merges at pipeline CREATE, so a materialized handle whose
        created union no longer matches the current side registry is RETIRED here (session
        closed, retired via the retire chokepoint) and the next ensure() re-creates
        with the current union. Unmaterialized (the common deferred path) or unchanged union
        => no-op. Returns True when a retire happened (observable for tests/logs)."""
        if self._real is None or self._created_lora_sig == self._lora_sig():
            return False
        if getattr(self._real, "current_session", None):
            # Retention (2026-08-24): a REFUSED end now keeps the pointer, so first try to
            # close it — a stale/abandoned session ends here and the retire proceeds; only a
            # session that STILL refuses to end (genuinely running) takes the raise below.
            self._real.end_session_if_open()
        if getattr(self._real, "current_session", None):
            # A wire changed its LoRA truth MID-GENERATION (e.g. the OTHER expert's wire
            # asserted a different stack at its stage — a LoRA node deleted there). The
            # running session was created under the old union and cannot be retargeted
            # mid-run; fail LOUD instead of silently finishing with the wrong weights. The
            # next queue re-creates with the current wiring (the side registry is already
            # correct). The stranded session is closed by the run-start end_session_if_open.
            raise RuntimeError(
                "qf_native: LoRA wiring changed MID-GENERATION (a wire's LoRA set no longer "
                "matches the running session's union — e.g. a LoRA node was added/removed on "
                "the other expert's wire). Re-queue the prompt: the next run re-creates the "
                "engine with the current wiring.")
        self.end_session_if_open()
        old, old_ck = self._real, self._ckey
        self._real, self._ckey = None, None
        self._created_lora_sig = None
        if old is not None and self._retire is not None:
            # identity-gated single chokepoint: refuses if a FOREIGN wrapper still shares the
            # entry (the new union creates under a NEW ckey, so both handles coexist then)
            self._retire(old_ck, old, self, reason="lora reconcile (union drift)")
        return True

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

    def partial_unload_vram(self, bytes_requested):
        # MEASURED (2026-09-05, LTX-2.5 acceptance box): comfy asked for 300–2700 MB at every VAE load, but the
        # patcher's `hasattr(eng, "partial_unload_vram")` saw THIS wrapper (no forwarder) → False → the full
        # unload ran every time (vram::releaseAll evicted 16.9 GB, re-paged at the next run: +2.9 s). Forward
        # it; unmaterialized = nothing to shed (0 → the patcher's full-unload fallback is a no-op too).
        return 0 if self._real is None else self._real.partial_unload_vram(bytes_requested)

    # NOTE: deliberately NO destroy() on the wrapper. The raw ungated destroy was dead code with
    # zero callers, and any future caller reaching for it would reproduce the shared-handle UAF the
    # liveness gate exists to prevent — release(requester=...) is the one sanctioned teardown (the
    # cache's _sweep_dead_pipelines destroys REAL handles, never wrappers). Fail-loud by absence.
    def release(self, requester=None):
        """Free the created pipeline (VRAM + the CPU backup) and go back to UNMATERIALIZED — the
        factory re-creates it from disk on the next use. Used by comfy's RAM manager
        (partially_unload_ram); refuses while a session is open.

        SHARED-HANDLE SAFETY (measured defect this closes): the real handle may be SHARED by
        other live models (two loader nodes on the same package hit one cache entry), so destroy
        is gated on the cache layer's retire chokepoint (liveness registry) — destroying under a live
        sibling left it a NULL pipeline for its next denoise while its ledger kept reporting the
        destroyed backup. No callback wired => refuse (fail-safe; better an unreclaimed backup
        than a use-after-destroy). Returns True iff the handle was actually destroyed."""
        if self._real is None:
            return False
        if self._real.pipeline is None:
            # already destroyed elsewhere — nothing to free; just fall back to lazy re-create
            self._real = None
            self._unloaded = True
            return False
        if getattr(self._real, "current_session", None) is not None:
            # Retention (2026-08-24): try to close a stale/abandoned session first — only a
            # session that still refuses to end (genuinely running) blocks the release.
            self._real.end_session_if_open()
        if getattr(self._real, "current_session", None) is not None:
            return False
        # keep_binding=True: the SAME ckey's next handle re-binds this entry's surviving
        # consumers (release does not change the create recipe). requester rides through so
        # the retire chokepoint applies the pair-mate discriminator (a bound MODEL is
        # accepted — its _qf wrapper is used).
        if self._retire is None or not self._retire(self._ckey, self._real,
                                                    requester if requester is not None else self,
                                                    keep_binding=True, reason="comfy RAM release"):
            return False
        self._real = None
        self._unloaded = True
        return True


# ── shared sidecar-LoRA rebuild contract (CR simplicity: ONE definition, not one per family) ──
# The engine merges sidecar LoRA at pipeline CREATE time, so a downstream QuantFuncNativeLoRA node
# re-creates the pipeline for the accumulated set. Every family's builder tags its model with these
# two attributes through tag_lora_rebuild(); the LoRA node reads them through lora_stack_of()/
# rebuild_of(). Names live here so the three seams cannot drift apart.
QF_LORA_STACK_ATTR = "_qf_lora_stack"
QF_LORA_REBUILD_ATTR = "_qf_rebuild"
QF_EXPERT_ATTR = "_qf_expert"   # "high"/"low" on the wan dual pair; absent => "all" (single-expert)


def expert_of(patcher):
    """[wiring-lora] WIRING-derived LoRA target: which loader output this MODEL came from
    ('high'/'low' — the wan builder tags its pair), 'all' for single-expert families. User
    directive 2026-08-22: the LoRA node carries NO target widget — chaining it on the loader's
    high output MEANS it acts on the high expert. Reads the MODEL (clones share .model), so a
    comfy patcher clone keeps its wire identity."""
    return getattr(getattr(patcher, "model", None), QF_EXPERT_ATTR, "all")


def ensure_model_config_attrs(model_config):
    """comfy's model_config classes grow attributes over releases; a seam that builds one directly
    must tolerate an older/newer comfy. ONE definition (was copy-pasted in every family builder)."""
    for attr, default in (("manual_cast_dtype", None), ("custom_operations", None),
                          ("optimizations", {}), ("scaled_fp8", None)):
        if not hasattr(model_config, attr):
            setattr(model_config, attr, default)
    return model_config


def stage_denoise_only_package(bundle_dir, transformer1_path, transformer2_path=None,
                               extra_links=None):
    """Build a config-complete PACKAGE dir the engine's `denoise_only` create can read WITHOUT
    copying the multi-GB weights.

    The file-based loader hands us bare transformer .safetensors FILE(s) (INT8-Fast shape), but the
    engine still needs the arch + VAE CONFIGS for session geometry (denoise_only skips the TE+VAE
    WEIGHTS, not the configs). So we stage: the family's shipped CONFIG bundle (model_index.json +
    transformer/ transformer_2/ vae/ config.json — tiny JSON, no weights) COPIED in, and the user's
    picked weight file(s) SYMLINKED as transformer/model.safetensors (+ transformer_2/ for wan's
    low-noise expert). The engine (denoise_only=True) reads the configs, loads the transformer
    weights via the symlinks, and never touches TE/VAE weights (comfy owns CLIP + VAE).

    Staged under ComfyUI's OWN temp dir (folder_paths.get_temp_directory(), never system /tmp),
    keyed deterministically by (bundle, realpath(files)) so repeated loads reuse one dir; rebuilt
    fresh each call (configs are tiny, symlinks are free) so a re-pick can't leave a stale link."""
    import shutil
    import folder_paths
    if not os.path.isdir(bundle_dir):
        raise RuntimeError(
            f"qf_native: config bundle missing: {bundle_dir} — this family's arch/VAE configs are "
            f"not shipped in the plugin (configs/<family>/). Cannot stage a denoise_only package.")
    real1 = os.path.realpath(transformer1_path)
    real2 = os.path.realpath(transformer2_path) if transformer2_path else None
    # extra_links: {subdir: target_path} — single-expert AV families link MORE weight files
    # into the staged package (ltx2: the SAME single xfm file into connectors/ [#565 comfy25
    # prefix branch], the gemma with-proj TE into text_encoder/ [connector aggregate_embed],
    # the audio_vae file [engine has_audio_ discriminant = weights presence]). Deterministic
    # key covers them so a re-pick restages.
    extra_links = {k: os.path.realpath(v) for k, v in (extra_links or {}).items() if v}
    key_src = "|".join([bundle_dir, real1, real2 or ""] +
                       [f"{k}={v}" for k, v in sorted(extra_links.items())])
    key = hashlib.sha1(key_src.encode()).hexdigest()[:16]
    root = os.path.join(folder_paths.get_temp_directory(), "qf_native_stage")
    os.makedirs(root, exist_ok=True)
    stage = os.path.join(root, key)
    if os.path.lexists(stage):
        if os.path.isdir(stage) and not os.path.islink(stage):
            shutil.rmtree(stage)
        else:
            os.remove(stage)
    shutil.copytree(bundle_dir, stage)   # tiny config JSONs only — no weights
    def _link_expert(sub, target):
        d = os.path.join(stage, sub)
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, "model.safetensors")
        if os.path.lexists(link):
            os.remove(link)
        try:
            os.symlink(target, link)
        except OSError:
            # Windows without symlink privilege: hardlink works on NTFS same-volume with no admin.
            # A multi-GB COPY fallback is deliberately refused (silent disk eating).
            try:
                os.link(target, link)
            except OSError as exc:
                raise RuntimeError(
                    f"qf_native: cannot link {target} into the staging dir ({exc}) — enable "
                    f"symlinks (Windows: Developer Mode) or keep the weights on the same volume "
                    f"as ComfyUI's temp directory.") from exc
    _link_expert("transformer", real1)
    for sub, target in sorted(extra_links.items()):
        _link_expert(sub, target)
    if real2:
        _link_expert("transformer_2", real2)
    else:
        # single-expert: drop the bundle's transformer_2/ config so the engine's two-expert
        # detection (boundary_ratio>0 AND transformer_2/config.json) resolves to single-expert.
        shutil.rmtree(os.path.join(stage, "transformer_2"), ignore_errors=True)
    return stage


def refuse_all_zero_initial_latent(xin, tag):
    """ALL-ZERO INITIAL LATENT GUARD — SHARED across every family seam (wan/LTX/LTX-AV/H3;
    one mechanism, N users). An Empty latent whose first sampler stage has add_noise=disable
    hands the engine a pure-zero tensor at sigma_max: int4 per-token quantization divides by
    amax=0 (0/0 = NaN), the NaN propagates silently through every step, and VAEDecode writes
    an EXACT-BLACK video with zero errors anywhere — the worst failure shape (measured on the
    user's wan run 2026-08-21; the class is family-independent, any int4 transformer NaNs the
    same way). Denoising pure zeros is never meaningful, so refuse LOUD at session begin,
    BEFORE any engine call, with the actual fix named. A noised latent / any i2v or
    latent-input flow is nonzero and never trips this."""
    if float(xin.abs().max()) == 0.0:
        raise RuntimeError(
            f"qf_native {tag}: the initial latent is ALL ZEROS — denoising pure zeros "
            f"produces NaN through int4 quantization (amax=0) and renders a BLACK video. "
            f"Almost always this means the FIRST sampler stage has add_noise=disable on an "
            f"Empty latent: set add_noise=enable on the first stage (official templates ship "
            f"it enabled; later stages keep disable — they receive the leftover-noise latent).")


def tag_lora_rebuild(patcher, lora_entries, rebuild):
    """Mark a freshly built patcher's model with its LoRA set + how to re-create with a new one."""
    m = patcher.model
    setattr(m, QF_LORA_STACK_ATTR, list(lora_entries))
    setattr(m, QF_LORA_REBUILD_ATTR, rebuild)
    return patcher


def lora_stack_of(patcher):
    return list(getattr(getattr(patcher, "model", None), QF_LORA_STACK_ATTR, []) or [])


def rebuild_of(patcher):
    return getattr(getattr(patcher, "model", None), QF_LORA_REBUILD_ATTR, None)


class QFModelPatcher(comfy.model_patcher.ModelPatcher):
    """ModelPatcher over an engine-managed pipeline. The engine owns + moves its own VRAM, so the
    weight-move overrides are no-ops — but model_size/loaded_size REPORT the engine's real footprint
    so ComfyUI's memory ledger ACCOUNTS FOR the engine's multi-GB VRAM, and co-EVICTION actually frees
    it (quantfunc_unload_sync) rather than letting comfy pop us while we stay resident (falsified
    ledger → sibling OOM)."""
    def _engine(self):
        return getattr(getattr(self, "model", None), "_qf", None)

    # A SHADOW patcher is the second (and any further) MODEL output sharing ONE engine (the wan
    # dual-expert loader emits model_high + model_low over a single engine handle). Only the
    # PRIMARY output carries the engine's footprint in comfy's ledger and may drive co-eviction;
    # a shadow reports a tiny constant and never unloads the shared engine — otherwise the ledger
    # would double-count the engine (2x ~14GB > the card) and an eviction aimed at the shadow
    # would silently rip the engine out from under the primary mid-workflow. The flag lives on
    # the MODEL (not the patcher) because comfy clone()s patchers freely; the model object is
    # shared by reference so the marker survives every clone. Handle DESTRUCTION stays governed
    # by the liveness registry (both models bind to the ckey), so a shadow still keeps the
    # handle alive — this flag only affects ledger REPORTING + co-EVICTION drive.
    _QF_SHADOW_LEDGER_BYTES = 64 * 1024 * 1024

    def _is_shadow(self):
        return bool(getattr(getattr(self, "model", None), "_qf_shadow", False))

    def _footprint(self):
        if self._is_shadow():
            return self._QF_SHADOW_LEDGER_BYTES
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
            _qf_cancel_pending_detach(eng)   # successor clone loading — reclaim a lazy detach
            eng.unloaded = False
        return 0

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        # Co-eviction (#4) + PARTIAL shed (2026-08-23 inter-stage eviction fix, ALL families —
        # this patcher is shared by wan/ltx2/h3): comfy asks to free `memory_to_free` (a few GB
        # for a sibling VAE/upsampler between a two-stage workflow's samplers). The old
        # all-or-nothing path offloaded the ENTIRE weight set for that small ask (measured LTX:
        # 18.3 GB round-tripped over pageable copies = ~10-15 s/prompt; nsys cudaMemcpyAsync =
        # 86% of API time). Now: shed only enough trailing transformer blocks
        # (engine quantfunc_partial_unload; backups already exist so the shed itself copies
        # NOTHING, and the next session begin reloads just those blocks). SAFETY LADDER: if the
        # partial shed cannot cover ~the request (old .so / monolith / refused), fall back to
        # the full unload so comfy's ledger never over-credits.
        if self._is_shadow():
            # A shadow never drives the SHARED engine's eviction (the primary output owns it);
            # its ledger share is the tiny constant, so comfy loses nothing by this 0.
            return 0
        eng = self._engine()
        if eng is None:
            return 0
        want = int(memory_to_free or 0)
        _qf_cancel_pending_detach(eng)   # real pressure supersedes a lazy-detach window
        qfe._dbg_prof(f"partially_unload asked={want // (1024*1024)} MB "
                      f"(loaded={self.loaded_size() // (1024*1024)} MB)")
        if want > 0 and hasattr(eng, "partial_unload_vram"):
            freed = eng.partial_unload_vram(want)
            if freed >= want:
                print(f"[qf_native] partial VRAM shed: {freed // (1024*1024)} MB freed for a "
                      f"{want // (1024*1024)} MB request (weights stay live)", flush=True)
                return freed
            # partial insufficient — full fallback keeps the honest-ledger guarantee
            qfe._dbg_prof(f"partial shed INSUFFICIENT: freed={freed // (1024*1024)} MB < "
                          f"want={want // (1024*1024)} MB -> full-unload fallback")
        return eng.unload_vram()


    # ---- comfy lifecycle / VRAM+RAM manager integration ----------------------------------
    def adopt_comfy_state_from(self, src):
        """Return a patcher that has THIS patcher's model but SRC's comfy-level state.

        WHY (CR regression, measured class): a downstream QuantFuncNativeLoRA re-creates the
        pipeline, so it hands back a DIFFERENT patcher+model. Anything an upstream node had
        applied with add_object_patch — most importantly ModelSamplingSD3 /
        ModelSamplingMiniMaxH3's shift patch — lives on the OLD patcher and would be silently
        dropped, leaving the checkpoint default with no error (the exact silent-shift class the
        H3 guard was added for). Transplanting via comfy's OWN clone(model_override=...) carries
        object_patches, model_options, callbacks, wrappers, attachments, injections and hooks —
        one authoritative list, so it cannot drift as comfy's clone() grows fields.
        """
        override = (self.model, (self.backup, self.backup_buffers,
                                 self.object_patches_backup, self.pinned))
        return src.clone(model_override=override)

    def detach(self, unpatch_all=True):
        """comfy is dropping this model. LAZY: keep the engine resident and arm a short one-shot
        unload window (see _QF_LAZY_DETACH_SECONDS — the measured common caller is comfy's
        unload_model_clones swapping clones of this SAME model between two samplers, where an
        eager full unload costs a pointless whole-weight-set round-trip). A successor clone's
        load/step reclaims the engine with zero reload; window expiry runs the REAL unload
        (unload_vram keeps the CPU backup, so a later run reloads lazily; no destroy = no
        use-after-free against a model comfy may still hold)."""
        eng = self._engine()
        if eng is not None and not getattr(eng, "unloaded", False):
            import threading
            if getattr(eng, "_qf_detach_lock", None) is None:
                eng._qf_detach_lock = threading.Lock()
            with eng._qf_detach_lock:
                eng.pending_detach = True
                old = getattr(eng, "_qf_detach_timer", None)
                if old is not None:
                    old.cancel()

                def _materialize():
                    try:
                        with eng._qf_detach_lock:
                            if not getattr(eng, "pending_detach", False) or \
                                    getattr(eng, "unloaded", False):
                                return
                            eng.pending_detach = False
                            eng._qf_detach_timer = None
                            qfe._dbg_prof("lazy-detach window expired -> real engine unload")
                            eng.unload_vram()
                    except Exception:  # noqa: BLE001 — a timer thread must never raise
                        pass

                t = threading.Timer(_QF_LAZY_DETACH_SECONDS, _materialize)
                t.daemon = True
                eng._qf_detach_timer = t
                t.start()
            print(f"[qf_prof] detach -> LAZY (engine resident; {_QF_LAZY_DETACH_SECONDS:.0f}s "
                  f"reclaim window)", flush=True)
        return super().detach(unpatch_all=unpatch_all)

    # ── HOST-RAM reporting ────────────────────────────────────────────────────────────────
    # REACHABILITY, stated honestly (measured in this ComfyUI): comfy calls loaded_ram_size() and
    # partially_unload_ram() ONLY behind `model.is_dynamic()` (model_management.py:665/1006/1451;
    # base ModelPatcher.is_dynamic() returns False and comfy's own comment says "Loaded RAM
    # pressure tracking is only implemented for DynamicVram loading"). This patcher is NOT dynamic
    # — opting in would mean implementing comfy's whole DynamicVram surface (weight pinning, VBAR,
    # backup restore) for an engine that owns its memory outside torch. So these two are correct
    # but currently INERT: the live integration for this plugin is detach() + model_size() /
    # loaded_size(), which comfy calls unconditionally. They are kept (not deleted) because they
    # are the right answers if this patcher ever goes dynamic, and they carry comfy's dynamic-path
    # signature so that switch cannot crash on an unexpected kwarg.

    def _engine_holds_cpu_backup(self):
        """True only for a pipeline that WAS created and then co-evicted (weights now in host RAM).
        A never-materialized lazy handle holds NOTHING — `unloaded` alone cannot tell those apart
        (it is True in both states), and reporting the footprint for a handle that never allocated
        would be claiming memory we do not hold."""
        eng = self._engine()
        if eng is None or not getattr(eng, "unloaded", False):
            return False, None
        # A lazy handle exposes `materialized`; a plain handle always has a real pipeline.
        if not getattr(eng, "materialized", True):
            return False, eng
        # A DESTROYED handle (pipeline gone — e.g. a sibling wrapper legitimately released the
        # last reference) holds no backup either; unload_vram keeps the pipeline valid, destroy
        # nulls it, so this is exactly the "backup actually exists" discriminator.
        if getattr(eng, "pipeline", None) is None:
            return False, eng
        return True, eng

    def loaded_ram_size(self):
        """HOST-RAM this model is actually responsible for: the engine's CPU backup after a
        co-eviction, else 0 (including for a handle that was never created)."""
        if self._is_shadow():
            return 0   # the PRIMARY output reports the shared engine's backup — no double-count
        holds, eng = self._engine_holds_cpu_backup()
        return max(0, int(getattr(eng, "footprint_bytes", 0))) if holds else 0

    def partially_unload_ram(self, ram_to_unload, subsets=None):
        """comfy's RAM manager asking for host memory back. Releasing the engine handle frees the
        CPU backup; the lazy handle re-creates from disk on the next use, so this is a real
        reclaim, not a leak — but only when we genuinely hold one and nothing is in flight (an
        open session must not have its pipeline pulled out from under it). `subsets` is comfy's
        dynamic-path kwarg (it names weight subsets to drop); this engine's backup is all-or-
        nothing, so it is accepted and ignored rather than crashing on an unexpected argument."""
        holds, eng = self._engine_holds_cpu_backup()
        if not holds:
            return 0
        if getattr(eng, "current_session", None) is not None:
            # Retention (2026-08-24, delta-CR R8): this pre-check is the 4th truthiness
            # consumer — short-circuiting here defeated release()'s own self-heal for a
            # retained-but-dead pointer. Attempt the end first; only a session that STILL
            # refuses to end (genuinely running) declines the RAM reclaim.
            end = getattr(eng, "end_session_if_open", None)
            if end is not None:
                end()
        if getattr(eng, "current_session", None) is not None:
            return 0
        release = getattr(eng, "release", None)
        if release is None:            # a plain (non-lazy) handle cannot be re-created: keep it
            return 0
        freed = max(0, int(getattr(eng, "footprint_bytes", 0)))
        try:
            # requester=self.model: the cache layer refuses the destroy while any OTHER live
            # model shares the handle (a sibling loader node on the same package) — report freed
            # bytes ONLY when the destroy actually happened, else the ledger gets credited for
            # memory a sibling still holds.
            if not release(requester=self.model):
                return 0
        except Exception:  # noqa: BLE001
            return 0
        return freed

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
