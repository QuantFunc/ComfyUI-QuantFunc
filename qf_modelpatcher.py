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
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import weakref

import math
import torch
import comfy.model_base
import comfy.model_patcher
import comfy.model_management
import comfy.conds
import comfy.patcher_extension

from . import qf_engine as qfe


# Request metadata, not a memory ledger. Context-local so nested calls/clones
# cannot leave another request's shape behind after success or cancellation.
_QF_SAMPLING_GEOMETRY = ContextVar("quantfunc_sampling_geometry", default=None)


class _EngineCacheAcquisition:
    """One lazy consumer's atomic cache lookup/publication transaction.

    The package cache installs a weak consumer pin while holding its identity
    lock, then registers the exact inverse here.  Validation/adoption failures
    roll back only pins created by this acquisition; success retains the weak
    pin for the lazy wrapper's lifetime.
    """
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
            except Exception:
                pass


_QF_ENGINE_CACHE_ACQUISITION = ContextVar("quantfunc_engine_cache_acquisition", default=None)


def _current_engine_cache_acquisition():
    return _QF_ENGINE_CACHE_ACQUISITION.get()


@contextmanager
def _engine_cache_acquisition(consumer):
    """Scope one factory call so cache pins survive success and unwind on error."""
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


def current_torch_device():
    """Capture one Comfy device for both the logical model and native cache key."""
    device = comfy.model_management.get_torch_device()
    return device, int(getattr(device, "index", 0) or 0)


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
        # 就好"): the sol route's z-score keep threshold. LOWER = more exact blocks
        # (denser/slower; <= -8 = true DENSE), HIGHER = sparser/faster. Same
        # rides-residency_opts session-knob class as set_attn_backend.
        try:
            self._sol_tau = float(v)
        except (TypeError, ValueError):
            self._sol_tau = 1.0

    def set_video_enhance(self, on):
        # [enhance switch, user 2026-09-19] the ONE video-quality switch: OFF (default) = the engine's speed
        # policy, ON = full quality. What that means (which tokens are recomputed, how much, which step stays
        # full, what yields to a step-cache) is ENGINE law behind the begin option `video_enhance` — this
        # plugin carries no number for it. Same rides-residency_opts class as set_sol_tau.
        self._video_enhance = bool(on)

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
        o = {}
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
        # [enhance switch] ALWAYS sent (both states are meaningful: OFF = the engine's speed policy, ON = full
        # quality; an absent key would leave the engine on its raw-key default, which is neither). Requires the
        # engine that ships with this plugin (an older .so refuses unknown begin keys LOUD — by design, never
        # silently ignored). The raw token_prune_keep_ratio key is the harness/expert surface and is never
        # built here; the engine refuses a begin that carries both.
        o["video_enhance"] = bool(getattr(self, "_video_enhance", False))
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

    # ── comfy's per-model VRAM interface (2026-09-19, user: 「与 comfyui 打通,让它知道我们需要多少显存、当前占了多少」) ──
    # comfy asks a model TWO numbers and does the rest itself: `memory_required(shape)` — how much MORE VRAM one
    # forward of `shape` needs (sampler_helpers.estimate_memory → load_models_gpu(memory_required=…) → free_memory
    # unloads comfy's IDLE models, largest first, keeping the sampler's declared set; samplers.calc_cond_batch →
    # `need × 1.5 < free` decides cond batching) — and `loaded_size()`/`model_size()` — what it holds (the patcher,
    # below). Cold demand keeps Comfy's ordinary request estimate as its floor; a hot pipeline refines it with
    # quantfunc_vram_need_bytes, while actual residency comes from quantfunc_resident_vram_bytes. Comfy's own
    # allocator makes the room before the denoise starts; nothing here reaches into comfy's state (user: 「不要 hack」).
    # What this REPLACES (git: 52ec260 / 9b6f7d3 / a6ec609): this function returned the comfy-side bytes ONLY — a
    # deliberate under-report so comfy would never evict the engine for a stage-2 shadow load (2026-08-22: the
    # torch-WAN activation estimate, 2.36 GB at 33.6k tok, evicted the ENGINE when the 64 MB shadow patcher loaded
    # → 15 s/prompt inter-stage thrash) — with the side effect that on a card where comfy's idle TE/VAEs FIT, comfy
    # kept them resident through the denoise and the engine shed its own weight pages every step (MEASURED 5090/H3
    # 800x768: 26–50 blocks re-copied per step). H3 additionally stacked a static per-element heuristic on top
    # (deleted with this change — it would double-count the real number). The 2026-08-22 constraint still holds for
    # the SHADOW (a second MODEL output over ONE shared engine, wan dual-expert): it reports comfy-side bytes only —
    # the primary is a separate LoadedModel that is NOT currently_used at the shadow's load, so a real need there
    # would evict it. Follow-up: declare the primary via get_additional_models() so comfy keeps it, then the shadow
    # can report the real need too.
    # Comfy-side bytes for this wrapper (D3): the latent in/out copies + cond tensors + conversion scratch — geometry-
    # proportional; K factors: input + noise + output velocity + c_concat tail + ~4x conversion/temporary copies ≈ 8
    # latent-sized tensors; cond tensors counted 2x (borrowed + converted). Floor keeps small runs honest. Signature
    # matches the REAL call site: `memory_required(shape, cond_shapes=cond_shapes)` (keyword!); `**_kw` tolerates drift.
    _QF_COMFY_SIDE_BASE_BYTES = 64 * 1024 * 1024
    _QF_COMFY_SIDE_LATENT_COPIES = 8
    _QF_COMFY_SIDE_COND_COPIES = 2
    _QF_COMFY_SIDE_ITEMSIZE = 2  # fp16/bf16 latents+conds

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
        return int(self._QF_COMFY_SIDE_BASE_BYTES +
                   self._QF_COMFY_SIDE_ITEMSIZE * (n_lat * self._QF_COMFY_SIDE_LATENT_COPIES +
                                                   n_cond * self._QF_COMFY_SIDE_COND_COPIES))

    def _qf_engine_latent_dims(self, input_shape):
        """Use current host request metadata before the admission pass.

        AV's flattened element count is not a spatial extent. The common
        OUTER_SAMPLE hook supplies shapes before inner_sample publishes them;
        a direct caller may supply latent_shapes itself. This selects only the
        legacy primary-video key, not a complete audio/reference peak forecast.
        """
        dims = [int(d) for d in input_shape]
        if len(dims) != 3 or dims[1] != 1:
            return dims
        request = _QF_SAMPLING_GEOMETRY.get()
        ls = getattr(self, "latent_shapes", None)
        if request is not None and request[0] is self:
            ls = request[1]  # even None must not fall back to a prior request
        if ls:
            try:
                streams = [[int(d) for d in s] for s in ls]
                if streams and sum(math.prod(s[1:]) for s in streams) == dims[2]:
                    return [dims[0]] + streams[0][1:]
            except (TypeError, ValueError):
                pass
        raise RuntimeError("QuantFunc packed AV demand requires current latent_shapes; "
                           "invoke through ComfyUI's OUTER_SAMPLE hook or supply request geometry")

    def memory_required(self, input_shape, cond_shapes=None, **_kw):
        """Official cold request floor, refined by hot native demand when available.

        The ordinary Comfy BaseModel estimate is the only request-shaped cold
        host estimate available before a native pipeline exists.  Keep it as a
        floor exactly as quantfunc_vram_need_bytes documents; Prepared
        persistent capacity is deliberately not used as transient demand.
        Once a cached pipeline exists, combine its exact additional need with
        the plugin-owned tensor copies, without creating during this query.

        comfy-side bytes + (PRIMARY only) the ENGINE's own "how much MORE do I need" for a forward of `input_shape`
        — quantfunc_vram_need_bytes(latent [B,C,(T,)H,W]): the primary transformer's working set under that shape
        (MEASURED by an earlier forward; else the same spatial shape at another batch scaled; else the config
        estimate) + the plan margin − the allocator's cached pool it already holds. The engine is asked through the
        CACHED real handle when one exists (a fresh lazy proxy answers 0 on its own while the cached engine already
        holds the arena) and is NEVER created here — comfy's eviction pass runs after this call, so a create
        here would run ahead of the room being made. Engine 0 = nothing to ask for (its cached pool already covers the
        working set — counted in loaded_size — OR nothing measured yet on a first forward of a new shape) →
        comfy-side only. This legacy estimate is not a complete cold-request
        peak bound; prior measurements do not make a later request exact.
        Missing ABI support and failed native queries propagate to the host.
        Accepted by the user (2026-09-19 「comfyui 路径不合并 CFG 就好」): with a real need comfy runs cond/uncond
        un-batched on a card that cannot hold 1.5× the B=2 working set — comfy's own rule, correct for us too (a B=2
        forward the card cannot hold pages inside the engine); our own full-pipeline path keeps CFG batched."""
        comfy_side = self._qf_comfy_side_bytes(input_shape, cond_shapes)
        try:
            ordinary_request = int(super().memory_required(
                input_shape, cond_shapes=cond_shapes or {}))
        except AttributeError:
            # Test/minimal subclasses without a Comfy BaseModel parent retain
            # the explicit plugin-side request floor.
            ordinary_request = 0
        cold_floor = max(comfy_side, ordinary_request)
        # Shadow/clone wrappers use the same physical dependency, but a sampler
        # driving only a shadow still needs the engine's inference demand.
        eng = getattr(self, "_qf", None)
        need = 0
        if eng is not None:
            # Lookup may bind an existing cached handle, but must never create
            # ahead of host admission. Query errors must reach the host instead
            # of authorizing execution with a fabricated zero demand.
            peek = getattr(eng, "ensure_if_cached", None)
            if callable(peek):
                eng = peek()
            if eng is not None:
                need = int(eng.vram_need_bytes(self._qf_engine_latent_dims(input_shape)))
        total = int(max(cold_floor, comfy_side + need))
        # One line per CHANGE of the answer (comfy asks per estimate + per cond-batch decision): the numbers comfy
        # will act on, so a ledger question is answerable from the log, not a guess.
        sig = (tuple(int(d) for d in input_shape), cold_floor, comfy_side, need)
        if sig != getattr(self, "_qf_ledger_last", None):
            self._qf_ledger_last = sig
            try:
                hold = f"{int(eng.resident_vram_bytes()) >> 20} MB" if eng is not None else "0 MB"
            except Exception as error:  # diagnostic only; never invent a measured zero
                hold = f"unknown ({type(error).__name__}: {error})"
            print("[qf_native] VRAM ledger: memory_required%s = %d MB "
                  "(host cold floor %d MB, comfy-side %d MB, engine need %d MB%s); engine hold %s"
                  % (list(sig[0]), total >> 20, cold_floor >> 20, comfy_side >> 20,
                     need >> 20,
                     "" if need else " (0: covered by what it holds, or nothing measured yet)",
                     hold), flush=True)
        return total

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

    def __init__(self, factory, retire=None):
        self._factory = factory          # () -> (QFEngineHandle, ckey)
        self._real = None
        self._prepared_entry = None
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
        # Persistent VRAM capacity comes only from the configured Prepared
        # resource. CPU-backup bytes are a separate, currently unknown ledger.
        self.capacity_bytes = None
        self.footprint_bytes = 0
        # Nothing created yet => nothing resident. Reporting "unloaded" keeps comfy's ledger
        # HONEST (loaded_size -> 0) for a chain link the sampler never touches.
        self._unloaded = True
        self.step_count = 0
        self.sampler_step_count = 0

    # ---- materialization ----
    def prepare_resource(self):
        """Resolve the actual cache recipe without calling model creation.

        Runs every factory, including the custom dual-output factory, through
        the shared _get_engine prepare-only branch. No lazy .lib access here.
        """
        self.reconcile_lora()
        token = qfe.FACTORY_PREPARE_ONLY.set(True)
        try:
            with _engine_cache_acquisition(self):
                entry, ckey = self._factory()
                if not isinstance(entry, (QFPreparedEntry, qfe.QFEngineHandle)):
                    raise RuntimeError("QuantFunc factory lacks the common prepared-resource contract")
                if getattr(entry, "resource", None) is None:
                    raise RuntimeError("QuantFunc cached engine has no retained native owner identity")
                self._prepared_entry = entry
                if isinstance(entry, QFPreparedEntry):
                    entry.bind_materializer(self, ckey)
                else:
                    self._bind_capacity(entry.capacity_bytes)
        finally:
            qfe.FACTORY_PREPARE_ONLY.reset(token)
        return entry, ckey

    def _bind_capacity(self, capacity_bytes):
        capacity = int(capacity_bytes)
        if capacity <= 0:
            raise qfe.NativeContractUnavailable(
                "QuantFunc cached engine has no native Prepared capacity")
        if self.capacity_bytes is not None and self.capacity_bytes != capacity:
            raise RuntimeError("QuantFunc native capacity changed for one lazy engine identity")
        self.capacity_bytes = capacity

    def _adopt_real(self, real, ckey):
        if not isinstance(real, qfe.QFEngineHandle) or real.pipeline is None:
            raise RuntimeError("QuantFunc factory did not materialize a live engine handle")
        _require_engine_host_grants(real)
        self._bind_capacity(real.capacity_bytes)
        self._real, self._ckey = real, ckey
        self._real.step_count = self.step_count
        self._real.sampler_step_count = self.sampler_step_count
        self._created_lora_sig = self._lora_sig()
        return self._real

    def _materialize_prepared(self, entry):
        """The one host-admitted cold-create path used by every family."""
        if self._real is not None and self._real.pipeline is not None:
            return self._real
        if entry is not self._prepared_entry:
            raise RuntimeError("QuantFunc prepared identity is not bound to this lazy engine")
        _require_engine_host_grants(entry)
        with _engine_cache_acquisition(self):
            real, ckey = self._factory()
            if isinstance(real, QFPreparedEntry):
                raise RuntimeError("QuantFunc factory remained in prepare-only mode during host materialization")
            return self._adopt_real(real, ckey)

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
            entry, ckey = self.prepare_resource()
            if isinstance(entry, QFPreparedEntry):
                entry.materialize(preferred=self)
            else:
                self._adopt_real(entry, ckey)
        # An existing handle may have survived a full-release attempt, including
        # Busy/Unknown after the native grant was fenced. Cache hits are not a
        # new host admission and must not resume it implicitly.
        _require_engine_host_grants(self._real)
        return self._real

    @property
    def materialized(self):
        return self._real is not None

    def ensure_if_cached(self):
        """Materialize ONLY if the real handle already exists in the pipeline cache (a cache HIT: no create) —
        the ledger reads (comfy's memory_required / loaded_size run BEFORE comfy's own eviction pass) must never
        run a create ahead of comfy making room (self-CR P-2). Returns the real handle or None."""
        if self._real is not None and self._real.pipeline is not None:
            return self._real
        entry, _ = self.prepare_resource()
        return None if isinstance(entry, QFPreparedEntry) else entry

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

    def resident_vram_bytes(self):
        return 0 if self._real is None else self._real.resident_vram_bytes()

    def vram_need_bytes(self, latent_shape):
        entry = self._real or self.ensure_if_cached()
        if entry is None:
            # The native ABI defines a cold/unestimable result as zero and
            # requires the host to retain its own request estimate as a floor.
            # QFSessionModelMixin.memory_required does so via BaseModel.
            return 0
        return entry.vram_need_bytes(latent_shape)

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


_CANONICAL_LOCK = threading.RLock()
# One strong Shared root per loaded image/device, not per model. Owned lookups
# are weak; actual entries/handles/consuming patchers retain their adapters.
_RESOURCE_DOMAINS = {}


class _CanonicalResourceDomain:
    """Per-device host ledger state; serializes host transitions, not native forwards."""
    def __init__(self, shared):
        self.shared = shared
        self.owners = weakref.WeakValueDictionary()
        self.shared_growth_fenced = False
        # Serializes one host admission/measurement transaction.  Native load
        # and release do not call back into this Python scheduler. Native
        # inference may still run concurrently and is constrained by grants.
        self.transaction_lock = threading.RLock()


_log = logging.getLogger(__name__)

# Native queries never wait (quantfunc.h). The engine answers QUANTFUNC_RESOURCE_BUSY whenever another thread holds
# the allocator's or the target's lock at that instant, which is ordinary while native work runs. MEASURED (issue
# #704, 远程-linux-c): an LTX-2.5 run died before step 2 on one such answer, 1 run in 4 under card contention.
# Identity survives BUSY: the engine fills device, owner_epoch and capabilities before its reader runs. Values
# (grants, lifecycle phase, bytes) do not.
_IDENTITY_STATES = (qfe.QUANTFUNC_RESOURCE_READY, qfe.QUANTFUNC_RESOURCE_BUSY, qfe.QUANTFUNC_RESOURCE_UNKNOWN)
# A value read that answers BUSY is re-read until this deadline, then refused. BUSY lasts one native critical
# section; the bound is far above that and only stops a lock that never frees. The re-reads run inside the caller's
# _domain_transaction, so another thread on the same device domain (e.g. Comfy's /free -> detach) waits for them: at
# most this deadline per read, never a deadlock (native code never takes that Python lock).
_NATIVE_BUSY_DEADLINE_S = 2.0
_NATIVE_BUSY_FIRST_BACKOFF_S = 0.001
_NATIVE_BUSY_MAX_BACKOFF_S = 0.05


def _query_identity(resource, *, owner_epoch=None, device=None, allow_closed=False):
    """Identity (state, device, owner_epoch, capabilities) of one native view; byte fields are None unless READY.

    NativeResource.query() raises on any non-OK status, and an answer the engine never populated carries no
    CAP_QUERY. owner_epoch and device, when given, must match exactly: an Owned view's own epoch, 0 for the
    device's Shared view. CLOSED is refused unless allow_closed (full eviction treats a Closed target as done).
    """
    snapshot = resource.query()
    states = _IDENTITY_STATES + ((qfe.QUANTFUNC_RESOURCE_CLOSED,) if allow_closed else ())
    if snapshot.state not in states or not snapshot.capabilities & qfe.QUANTFUNC_RESOURCE_CAP_QUERY:
        raise RuntimeError(f"QuantFunc resource identity unavailable (state={snapshot.state})")
    if ((owner_epoch is not None and snapshot.owner_epoch != owner_epoch) or
            (device is not None and snapshot.device != device)):
        raise RuntimeError(f"QuantFunc resource identity changed: epoch {snapshot.owner_epoch} on device "
                           f"{snapshot.device}, expected epoch {owner_epoch} on device {device}")
    if snapshot.state != qfe.QUANTFUNC_RESOURCE_READY:
        _log.debug("[qf_native] resource identity answered state=%d; identity used, bytes not", snapshot.state)
    return snapshot


class _NativeStillBusy(RuntimeError):
    """A native call answered BUSY until _NATIVE_BUSY_DEADLINE_S."""


def _read_past_busy(read, what):
    """One native call, re-issued while it answers BUSY, until _NATIVE_BUSY_DEADLINE_S: a read, or a command whose
    re-issue is safe (see _set_domain_grants and partially_unload).

    Only BUSY is retried. Any other state (READY, UNKNOWN, CLOSED) returns at once for the caller to judge, and an
    API error raises from `read` unchanged.
    """
    started = time.monotonic()
    busy_reads = 0
    backoff = _NATIVE_BUSY_FIRST_BACKOFF_S
    while True:
        result = read()
        waited = time.monotonic() - started
        if result.state != qfe.QUANTFUNC_RESOURCE_BUSY:
            if busy_reads:
                _log.debug("[qf_native] %s answered BUSY %d time(s) for %.1f ms, then state=%d (deadline %.0f ms)",
                           what, busy_reads, waited * 1e3, result.state, _NATIVE_BUSY_DEADLINE_S * 1e3)
            return result
        busy_reads += 1
        if waited >= _NATIVE_BUSY_DEADLINE_S:
            raise _NativeStillBusy(f"QuantFunc {what} stayed BUSY for {waited * 1e3:.0f} ms ({busy_reads} reads, "
                                   f"deadline {_NATIVE_BUSY_DEADLINE_S * 1e3:.0f} ms): another native thread held "
                                   "the lock the whole time")
        time.sleep(min(backoff, _NATIVE_BUSY_DEADLINE_S - waited))
        backoff = min(backoff * 2, _NATIVE_BUSY_MAX_BACKOFF_S)


def _ready_read(read, what):
    """A native call whose READY answer is used: BUSY is re-issued (bounded); any other non-READY is refused."""
    result = _read_past_busy(read, what)
    if result.state != qfe.QUANTFUNC_RESOURCE_READY:
        raise RuntimeError(f"QuantFunc {what} unavailable (state={result.state})")
    return result


def _set_domain_grants(shared, owned, mask, **limits):
    """One atomic domain-grant command, re-issued while it answers BUSY (bounded, like a read).

    Re-issuing is exact: a BUSY answer applied NOTHING and the command carries absolute limits. The C API answers BUSY
    on an owned-target lock or phase miss before it reaches the allocator (quantfunc_api.cpp:2071-2076), and the
    allocator's setHostDomainGrants returns every non-Ready before its first assignment (Tensor.h:3098-3134). An
    engine change that breaks either premise must revisit this retry.
    """
    return _ready_read(lambda: shared._resource.set_domain_grants(owned, mask, **limits), "atomic domain grant")


def _ready_grant(resource, *, enroll=False):
    result = resource.enroll_host() if enroll else _read_past_busy(resource.query_grant, "host grant")
    if result.state == qfe.QUANTFUNC_RESOURCE_BUSY:
        # Only enrollment reaches here: it is a native transaction, so a BUSY answer is refused, never re-issued.
        raise RuntimeError("QuantFunc host enrollment answered BUSY: another native thread held the lock")
    if result.state != qfe.QUANTFUNC_RESOURCE_READY or not result.enrolled:
        raise RuntimeError(f"QuantFunc host grant unavailable (state={result.state}, "
                           f"enrolled={result.enrolled}); an Attached engine cannot be migrated")
    return result


def _ready_device_grant(resource):
    result = _read_past_busy(resource.query_device_grant, "device grant")
    if result.state != qfe.QUANTFUNC_RESOURCE_READY or not result.enrolled:
        raise RuntimeError(f"QuantFunc device grant unavailable (state={result.state}, "
                           f"enrolled={result.enrolled})")
    return result


def _resource_domain(lib, device, *, prepare=False):
    key = (qfe.library_identity(lib), int(device))
    with _CANONICAL_LOCK:
        domain = _RESOURCE_DOMAINS.get(key)
    if domain is None:
        # Native view creation may enter the driver.  Do it outside the global
        # registry lock, then publish one winner; a racing spare view is closed
        # only after the registry lock has been released.
        shared = qfe.NativeResource.shared(lib, device)
        adapter = QFNativeResourcePatcher(shared)
        candidate = _CanonicalResourceDomain(adapter)
        adapter._domain_key = key
        adapter._domain = candidate
        adapter._shared_adapter = adapter
        with _CANONICAL_LOCK:
            domain = _RESOURCE_DOMAINS.get(key)
            if domain is None:
                _RESOURCE_DOMAINS[key] = candidate
                domain = candidate
                candidate = None
        if candidate is not None:
            candidate.shared._resource.close()
    if prepare:
        with domain.transaction_lock:
            # Public native authority validates empty-device enrollment; never
            # migrate a live Attached engine or reset an already-issued limit.
            _ready_grant(domain.shared._resource, enroll=True)
            domain.shared._host_managed = True
    return domain.shared, domain.owners, domain.transaction_lock


_QF_FULL_LOAD_SENTINEL = 10 ** 30
_QF_UINT64_MAX = (1 << 64) - 1


def _host_allowance(extra_memory):
    """Normalize Comfy allowance; None is full-load and negatives request shrink."""
    if isinstance(extra_memory, float) and not math.isfinite(extra_memory):
        raise ValueError("host allowance must be finite")
    try:
        value = int(extra_memory)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("host allowance must be a finite byte count") from exc
    return None if value >= _QF_FULL_LOAD_SENTINEL else max(-_QF_UINT64_MAX, min(value, _QF_UINT64_MAX))


def _domain_members(adapter):
    domain = getattr(adapter, "_domain", None)
    if domain is None:
        raise RuntimeError("QuantFunc canonical resource domain was retired")
    with domain.transaction_lock:
        return domain.shared, list(domain.owners.values())


def _domain_transaction(adapter):
    domain = getattr(adapter, "_domain", None)
    if domain is None:
        raise RuntimeError("QuantFunc canonical resource domain was retired")
    return domain.transaction_lock


def _domain_loaded_size(adapter):
    """One coherent native snapshot of all Shared + Owned backing in the domain."""
    with _domain_transaction(adapter):
        shared, _ = _domain_members(adapter)
        return _ready_read(shared._resource.query_domain_residency, "domain residency").resident_bytes


def _publish_domain_grants(adapter, *, publish_owner=False, growth_allowance=0):
    """Publish one formal Comfy admission from exact native residency.

    Owner and Shared each use their own current residency plus the one official
    growth (Comfy's allowance, or what Comfy left free when that is more: see
    below).  Only the aggregate Device ceiling uses coherent domain residency.
    The Device ceiling bounds both resource ceilings.
    """
    with _domain_transaction(adapter):
        domain = adapter._domain
        shared, owners = _domain_members(adapter)
        total = int(comfy.model_management.get_total_memory(shared.load_device))
        if total <= 0:
            raise RuntimeError("ComfyUI returned no physical capacity for the QuantFunc device")
        total = min(total, _QF_UINT64_MAX)
        domain_actual = min(_ready_read(shared._resource.query_domain_residency, "domain residency").resident_bytes,
                            total)
        if growth_allowance is None:
            allowance = None
            device_limit = total
        else:
            allowance = int(growth_allowance)
            if allowance < 0:
                raise ValueError("domain grant growth allowance must be nonnegative")
            # Comfy's extra_memory is its WEIGHTS budget: load_models_gpu computes it as
            # free - minimum_memory_required, i.e. with the inference reserve it holds for THIS model's
            # sampling taken out. A Torch model spends that reserve on activations outside any ceiling;
            # native weights, activations and cache all live under this one, so publishing the budget
            # alone refuses the engine the room Comfy freed for it. MEASURED (H3, 32 GB): 24017 MB free,
            # 20951 MB budget, 14270 MB of weights -> step 0 refused with ~10 GB physically free.
            # The ceiling is what Comfy left free at this admission, never less than its own budget.
            free = max(0, int(comfy.model_management.get_free_memory(shared.load_device)))
            allowance = min(max(allowance, free), total)
            device_limit = min(total, domain_actual + allowance)

        def resource_limit(resource_adapter):
            actual = min(int(resource_adapter.loaded_size()), total)
            limit = total if allowance is None else min(total, actual + allowance)
            return min(device_limit, limit)

        target_owner = adapter if publish_owner else None
        if target_owner is not None and target_owner is shared:
            raise ValueError("Shared has no Owner grant")
        if target_owner is not None and target_owner not in owners:
            raise RuntimeError("QuantFunc Owner is not retained by its canonical domain")
        shared_target = resource_limit(shared)
        if target_owner is None:
            mask = qfe.QUANTFUNC_RESOURCE_GRANT_SHARED | qfe.QUANTFUNC_RESOURCE_GRANT_DEVICE
            _set_domain_grants(shared, None, mask, shared_limit_bytes=shared_target, device_limit_bytes=device_limit)
        else:
            owner_target = resource_limit(target_owner)
            mask = (qfe.QUANTFUNC_RESOURCE_GRANT_OWNER |
                    qfe.QUANTFUNC_RESOURCE_GRANT_SHARED |
                    qfe.QUANTFUNC_RESOURCE_GRANT_DEVICE)
            _set_domain_grants(shared, target_owner._resource, mask, owner_limit_bytes=owner_target,
                               shared_limit_bytes=shared_target, device_limit_bytes=device_limit)
        domain.shared_growth_fenced = False
        return device_limit


def _revoke_domain_growth(adapter):
    """Passively revoke growth without claiming that existing occupancy vanished."""
    with _domain_transaction(adapter):
        domain = adapter._domain
        shared, owners = _domain_members(adapter)
        domain.shared_growth_fenced = True
        target_owner = None if adapter is shared else adapter
        if target_owner is not None and target_owner not in owners:
            raise RuntimeError("QuantFunc Owner is not retained by its canonical domain")
        if target_owner is None:
            mask = qfe.QUANTFUNC_RESOURCE_GRANT_SHARED | qfe.QUANTFUNC_RESOURCE_GRANT_DEVICE
            _set_domain_grants(shared, None, mask, shared_limit_bytes=0, device_limit_bytes=0)
        else:
            mask = (qfe.QUANTFUNC_RESOURCE_GRANT_OWNER |
                    qfe.QUANTFUNC_RESOURCE_GRANT_SHARED |
                    qfe.QUANTFUNC_RESOURCE_GRANT_DEVICE)
            _set_domain_grants(shared, target_owner._resource, mask, owner_limit_bytes=0,
                               shared_limit_bytes=0, device_limit_bytes=0)


def _capture_domain_grants(adapter, identity):
    """Capture exact pre-admission authorization; grants are not occupancy."""
    shared = adapter._domain.shared
    owner_limit = (int(_ready_grant(adapter._resource).limit_bytes)
                   if identity.owner_epoch else None)
    shared_fenced = bool(adapter._domain.shared_growth_fenced)
    shared_limit = int(_ready_grant(shared._resource).limit_bytes)
    device_limit = int(_ready_device_grant(shared._resource).limit_bytes)
    return owner_limit, shared_limit, device_limit, shared_fenced


def _restore_domain_grants(adapter, snapshot):
    """Attempt one atomic restoration; native non-Ready remains all-or-none."""
    owner_limit, shared_limit, device_limit, shared_fenced = snapshot
    shared = adapter._domain.shared
    if owner_limit is None:
        _set_domain_grants(shared, None, qfe.QUANTFUNC_RESOURCE_GRANT_SHARED | qfe.QUANTFUNC_RESOURCE_GRANT_DEVICE,
                           shared_limit_bytes=shared_limit, device_limit_bytes=device_limit)
    else:
        _set_domain_grants(shared, adapter._resource,
                           qfe.QUANTFUNC_RESOURCE_GRANT_OWNER | qfe.QUANTFUNC_RESOURCE_GRANT_SHARED |
                           qfe.QUANTFUNC_RESOURCE_GRANT_DEVICE,
                           owner_limit_bytes=owner_limit, shared_limit_bytes=shared_limit,
                           device_limit_bytes=device_limit)
    adapter._domain.shared_growth_fenced = shared_fenced


def canonical_resource_adapters(engine):
    resource = getattr(engine, "resource", None)
    if resource is None:
        raise RuntimeError("QuantFunc engine has no retained native owner identity")
    snapshot = _query_identity(resource)
    if snapshot.owner_epoch == 0:
        raise RuntimeError("QuantFunc model requires an Owned resource, not Shared")
    shared, owners, transaction_lock = _resource_domain(resource._lib, snapshot.device)
    with transaction_lock:
        owner = owners.get(snapshot.owner_epoch)
        if owner is None:
            owner = QFNativeResourcePatcher(resource)
            owners[snapshot.owner_epoch] = owner
            owner._domain_key = (qfe.library_identity(resource._lib), int(snapshot.device))
            owner._domain = shared._domain
            owner._shared_adapter = shared
        # Production canonical adapters may only execute with actual native
        # enrollment. Querying the graph itself remains non-mutating; Attached
        # identities are never enrolled here as a migration shortcut.
        owner._host_managed = shared._host_managed = True
        # Retained by actual identity, not a primary/shadow wrapper or host weakref.
        engine._qf_resource_adapters = (owner, shared)
        return engine._qf_resource_adapters


def _require_engine_host_grants(engine):
    adapters = canonical_resource_adapters(engine)
    with _domain_transaction(adapters[0]):
        shared = None
        shared_grant = None
        for adapter in adapters:
            if (adapter._admission_failed and
                    adapter._admitting_thread != threading.get_ident()):
                raise qfe.NativeContractUnavailable(
                    "QuantFunc execution is fenced after a failed host admission; "
                    "retry through ComfyUI model loading")
            adapter.require_load_contract()
            grant = _ready_grant(adapter._resource)
            if adapter is adapter._shared_adapter:
                shared = adapter
                shared_grant = grant
            elif int(grant.limit_bytes) == 0:
                raise qfe.NativeContractUnavailable(
                    "QuantFunc engine execution requires a positive Owned host grant")
        if shared is None:
            raise RuntimeError("QuantFunc canonical resource domain has no Shared adapter")
        if shared._domain.shared_growth_fenced or int(shared_grant.limit_bytes) == 0:
            raise qfe.NativeContractUnavailable(
                "QuantFunc engine execution requires a positive Shared host grant")
        if int(_ready_device_grant(shared._resource).limit_bytes) == 0:
            raise qfe.NativeContractUnavailable(
                "QuantFunc engine execution requires a positive aggregate device grant")


class QFPreparedEntry:
    """Configured cold identity plus a host-bound materialization chokepoint."""
    def __init__(self, lib, device, *, create_params=None):
        self.lib = lib
        self.create_params = create_params
        self.capacity_bytes = None
        self.component_count = None
        self.cache_key = None
        self._cache_usable = True
        self._materializers = weakref.WeakSet()
        self._materialize_lock = threading.RLock()
        _resource_domain(lib, device, prepare=True)  # Shared BEFORE Owned/backing.
        self.resource = qfe.NativeResource.prepare(lib, device)
        try:
            if create_params is None:
                raise qfe.NativeContractUnavailable(
                    "QuantFunc Prepared capacity requires the complete create descriptor")
            self.resource.configure(create_params)
            # A target still Creating answers BUSY for its whole create; past the deadline that is a refusal.
            capacity = _ready_read(self.resource.query_capacity, "resource capacity")
            self.capacity_bytes = int(capacity.required_persistent_bytes)
            self.component_count = int(capacity.component_count)
            _ready_grant(self.resource, enroll=True)
            owner, shared = canonical_resource_adapters(self)
            self._owner_adapter, self._shared_adapter = owner, shared
            owner._prepared = True
            owner._host_managed = True
            owner.bind_prepared(self, self.capacity_bytes)
        except BaseException:
            if hasattr(self, "_owner_adapter"):
                self.retire_unpublished()
            else:
                self._cache_usable = False
                self.resource.close()
            raise

    def retire_unpublished(self):
        """Make a losing prepared candidate invisible, then close its native view.

        Candidate construction registers the Owned adapter in the canonical
        domain before cache publication.  A loser therefore must be removed by
        its exact owner epoch while holding the domain transaction; closing it
        first would leave refresh traversals able to observe a dead adapter.
        """
        owner = self._owner_adapter
        with _domain_transaction(owner):
            if not self._cache_usable:
                return
            self._cache_usable = False
            domain = owner._domain
            epoch = owner._owner_epoch
            if domain.owners.get(epoch) is owner:
                domain.owners.pop(epoch, None)
            owner._closed_identity = True
            owner._host_managed = False
            self._materializers.clear()
            self.resource.close()

    def retire_materialized(self):
        """Remove an old Attached/Closed identity from future create lookup.

        Keep its native view and canonical Owner adapter alive: retained
        aliases may still report/reclaim residual backing after pipeline
        teardown.  A same-recipe retry must prepare a new owner epoch.
        """
        with _domain_transaction(self._owner_adapter):
            with self._materialize_lock:
                self._cache_usable = False
                self._materializers.clear()

    def bind_materializer(self, engine, cache_key):
        with _domain_transaction(self._owner_adapter):
            with self._materialize_lock:
                if not self._cache_usable:
                    raise qfe.NativeContractUnavailable(
                        "QuantFunc prepared cache candidate was retired before publication")
                if self.cache_key is not None and self.cache_key != cache_key:
                    raise RuntimeError("QuantFunc prepared identity was rebound to a different cache recipe")
                if self.capacity_bytes is None or self.capacity_bytes <= 0:
                    raise qfe.NativeContractUnavailable(
                        "QuantFunc prepared identity has no native persistent capacity")
                self.cache_key = cache_key
                self._materializers.add(engine)
                engine._bind_capacity(self.capacity_bytes)

    def has_materializer(self, engine):
        with _domain_transaction(self._owner_adapter):
            with self._materialize_lock:
                return self._cache_usable and engine in self._materializers

    def materialize(self, preferred=None):
        # Fixed order: domain -> prepared entry. The factory may take the global
        # engine cache lock only for short lookup/publication sections; no cache
        # holder may enter this pair in the opposite direction.
        with _domain_transaction(self._owner_adapter):
            with self._materialize_lock:
                if not self._cache_usable:
                    raise qfe.NativeContractUnavailable(
                        "QuantFunc prepared cache candidate is no longer materializable")
                candidates = ([preferred] if preferred is not None else []) + list(self._materializers)
                for engine in candidates:
                    if engine is not None and engine in self._materializers:
                        return engine._materialize_prepared(self)
        raise qfe.NativeContractUnavailable(
            "QuantFunc prepared resource has no live common lazy-engine materializer")


class QFNativeResourcePatcher(comfy.model_patcher.ModelPatcher):
    """Host adapter for an existing native resource, not a model-load estimate.

    The caller owns one canonical adapter per resource and must retain it while
    registered: ComfyUI holds patchers weakly. No native model is created here.
    Current backing is the entire size of this *residency-only* dependency;
    model capacity, inference demand and load grants belong to their distinct
    native contracts. Do not substitute this adapter for a lazy model loader.
    Full detach requires CAP_RELEASE_ALL and a growth-denying grant (or an
    already Closed target). A plain zero snapshot never authorizes deregistration.
    """
    def __init__(self, resource):
        # The adapter's own identity, fixed for the life of its native view; every later check must see exactly it.
        identity = _query_identity(resource)
        load_device = torch.device("cuda", identity.device)
        model = torch.nn.Module()
        model.device = load_device
        super().__init__(model, load_device, torch.device("cpu"))
        self._resource = resource
        self._prepared = False
        self._host_managed = False
        self._needs_readmission = False
        self._closed_identity = False
        self._owner_epoch = identity.owner_epoch  # 0 for the device's Shared view
        self._domain_key = None
        self._domain = None
        self._shared_adapter = None
        self._prepared_entry = None
        self._capacity_bytes = None
        self._admission_failed = False
        self._admitting_thread = None

    def _identity(self, *, allow_closed=False):
        return _query_identity(self._resource, owner_epoch=self._owner_epoch,
                               device=self.load_device.index, allow_closed=allow_closed)

    def preflight_host_load(self):
        """Read-only fail-before-pop validation used during dependency expansion.

        Zero grants on a Prepared resource are valid: this is not admission.
        The check deliberately performs no grant mutation, release, create, or
        engine-cache operation.
        """
        with _domain_transaction(self):
            identity = self._identity()
            lifecycle = _ready_read(self._resource.lifecycle, "host preflight lifecycle")
            if lifecycle.phase == qfe.QUANTFUNC_RESOURCE_PHASE_CLOSED:
                raise RuntimeError("QuantFunc host preflight rejected a Closed resource identity")
            grant = _ready_grant(self._resource)
            if self._prepared_entry is not None and not self._prepared_entry._cache_usable:
                raise qfe.NativeContractUnavailable(
                    "QuantFunc host preflight rejected a retired create identity")
            if identity.owner_epoch and lifecycle.phase == qfe.QUANTFUNC_RESOURCE_PHASE_PREPARED:
                if (self._prepared_entry is None or self._capacity_bytes is None or
                        self._capacity_bytes <= 0 or not self._prepared_entry._cache_usable):
                    raise qfe.NativeContractUnavailable(
                        "QuantFunc host preflight lacks a usable Prepared capacity contract")
            shared = self._domain.shared
            _ready_grant(shared._resource)
            _ready_device_grant(shared._resource)
            _ready_read(shared._resource.query_domain_residency, "domain residency")
            return grant

    def model_patches_models(self):
        # Direct load_models_gpu([resource_adapter]) must preflight too.  Do not
        # return self as an additional model or alter the dependency graph.
        self.preflight_host_load()
        return super().model_patches_models()

    def bind_prepared(self, entry, capacity_bytes):
        with _domain_transaction(self):
            capacity = int(capacity_bytes)
            if self._prepared_entry is not None and self._prepared_entry is not entry:
                raise RuntimeError("QuantFunc owner adapter was rebound to another prepared identity")
            if self._capacity_bytes is not None and self._capacity_bytes != capacity:
                raise RuntimeError("QuantFunc owner capacity changed for one native identity")
            self._prepared_entry, self._capacity_bytes = entry, capacity

    def loaded_size(self):
        with _domain_transaction(self):
            return _ready_read(self._resource.residency, "resource residency").resident_bytes

    def require_load_contract(self):
        with _domain_transaction(self):
            lifecycle = _ready_read(self._resource.lifecycle, "resource lifecycle")
            self._closed_identity = lifecycle.phase == qfe.QUANTFUNC_RESOURCE_PHASE_CLOSED
            if self._closed_identity:
                raise RuntimeError("QuantFunc Closed resource identity is not reloadable")
            if lifecycle.phase == qfe.QUANTFUNC_RESOURCE_PHASE_PREPARED:
                if self._prepared_entry is None or self._capacity_bytes is None:
                    raise qfe.NativeContractUnavailable(
                        "QuantFunc Prepared resource is not bound to the common cold-load adapter")
            if self._needs_readmission and self._capacity_bytes is None:
                raise qfe.NativeContractUnavailable(
                    "QuantFunc evicted resource has no retained cold-capacity contract")

    def model_size(self):
        with _domain_transaction(self):
            self.require_load_contract()
            if self._capacity_bytes is not None:
                return int(self._capacity_bytes)
            # Shared has no Prepared model capacity. Its already-resident bytes
            # remain a zero-deficit dependency in Comfy's ledger.
            return self.loaded_size()

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        if device_to != self.load_device:
            raise ValueError("a native resource cannot migrate to another device")
        allowance = _host_allowance(extra_memory)
        with _domain_transaction(self):
            # ONE critical section for the whole Comfy operation, like every other method here: the shrink revokes
            # growth domain-wide, so releasing the lock before the re-admission would show a peer the fenced state.
            if allowance is not None and allowance < 0:
                # Comfy's negative budget means "shrink by this much, THEN RUN" (load_models_gpu computes
                # max(0, free - minimum_memory_required, ...) - loaded_memory; a stock patcher unloads that much and
                # samples in low-VRAM mode). The release revokes growth, so returning here left the engine fenced
                # with nothing to re-open it before the sampler starts. MEASURED (host-vram-h3 measure-10): stage 2
                # of the double-sample asks 17 GB, Comfy shrank the engine 15.7 -> 11.7 GB, and seven runs in a
                # row died at session begin on "requires a positive Owned host grant". The shrink is now followed
                # by the same formal admission as any load, with no weights budget of its own: the ceiling is
                # what Comfy left free (its inference reserve), which is exactly the room it made for this run.
                print(f"[qf_native] Comfy's budget is negative: shrinking {-allowance >> 20} MB, then admitting "
                      "the run without a weights budget", flush=True)
                self.partially_unload(device_to, -allowance,
                                      force_patch_weights=force_patch_weights)
                allowance = 0
            self.require_load_contract()
            identity = self._identity()
            before_domain = _domain_loaded_size(self)
            prior = (_capture_domain_grants(self, identity)
                     if self._host_managed else None)
            published = False
            self._admitting_thread = threading.get_ident()
            try:
                if self._host_managed:
                    _publish_domain_grants(
                        self, publish_owner=bool(identity.owner_epoch),
                        growth_allowance=allowance)
                    published = True
                if self._prepared_entry is not None:
                    # A non-READY answer carries no phase; reading None as "not Prepared" skipped materialize().
                    lifecycle = _ready_read(self._resource.lifecycle, "admission lifecycle")
                    if lifecycle.phase == qfe.QUANTFUNC_RESOURCE_PHASE_PREPARED:
                        self._prepared_entry.materialize()
                self._prepared = False
                self._needs_readmission = False
                after_domain = _domain_loaded_size(self)
                self._admission_failed = False
                return max(0, after_domain - before_domain)
            except BaseException as error:
                if published:
                    self._admission_failed = True
                    self._domain.shared_growth_fenced = True
                    try:
                        _restore_domain_grants(self, prior)
                    except BaseException as rollback_error:
                        try:
                            error.add_note(
                                "QuantFunc grant rollback was unavailable; Python admission remains fenced: "
                                f"{rollback_error}")
                        except Exception:
                            pass
                raise
            finally:
                self._admitting_thread = None

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        want = max(0, min(int(memory_to_free), (1 << 64) - 1))
        if not want:
            return 0
        with _domain_transaction(self):
            try:
                if self._host_managed:
                    if self is not self._domain.shared:
                        self._needs_readmission = True
                    # Native inference does not take the Python transaction lock.
                    # Revoke growth before release so freed capacity cannot be
                    # reclaimed concurrently; release failure never restores it.
                    _revoke_domain_growth(self)
                # A BUSY release may already have freed eligible backing it does not report (engine releaseEligible,
                # vram_manager/HostMemory.cpp:145-181), so the retry can free more than `want`: a reload-time cost only.
                # `freed` is only what a READY answer reports, never an estimate; it errs low and Comfy then falls back
                # to its own full detach.
                result = _ready_read(lambda: self._resource.release_eligible(want), "resource release")
            except _NativeStillBusy as busy:
                # Comfy's unload path has no failure channel: vouch for no freed bytes instead of raising.
                _log.warning("[qf_native] %s; reporting 0 freed, Comfy falls back to a full detach", busy)
                return 0
            return result.freed_bytes

    def clone(self, disable_dynamic=False, model_override=None, force_deepcopy=False):
        if model_override is not None or force_deepcopy:
            raise ValueError("a native resource identity cannot be copied or replaced")
        return self

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        raise ValueError("a native resource has no patchable model weights")

    def add_object_patch(self, name, obj):
        raise ValueError("a native resource identity cannot be patched")

    def detach(self, unpatch_all=True):
        if unpatch_all:
            with _domain_transaction(self):
                identity = self._identity(allow_closed=True)
                if not identity.capabilities & qfe.QUANTFUNC_RESOURCE_CAP_RELEASE_ALL:
                    raise qfe.NativeContractUnavailable("QuantFunc native resource full eviction is unsupported "
                                                        "without CAP_RELEASE_ALL; keep the host record")
                try:
                    lifecycle = _ready_read(self._resource.lifecycle, "full eviction lifecycle")
                    closed = lifecycle.phase == qfe.QUANTFUNC_RESOURCE_PHASE_CLOSED
                    self._closed_identity = closed
                    if not closed:
                        _ready_grant(self._resource)
                        # Even a failing foreign call may already have changed the
                        # grant. Do not allow a cache hit to undo this admission fence.
                        if identity.owner_epoch:
                            self._needs_readmission = True
                        _revoke_domain_growth(self)
                        self._host_managed = True
                    # Closed targets cannot grow; native permits final exact-old-owner
                    # cleanup but this never makes their identity reusable.
                    result = self._resource.release_all()
                    if result.state != qfe.QUANTFUNC_RESOURCE_READY:
                        # Comfy's unload hook has no failure channel: unload_all_models() runs unguarded on the
                        # prompt worker (its OOM handler and POST /free), so an error escaping here ends the
                        # server's only worker and every later prompt hangs. Native Busy is also the ORDINARY
                        # answer for a live pipeline: release_all certifies ZERO backing, and the non-paged
                        # floor never reaches zero. Everything eligible is already released; growth was revoked
                        # above and is revoked again below, so nothing regrows before the next formal admission,
                        # which re-reads native residency. What remains is physically visible to Comfy.
                        print(f"[qf_native] full eviction left native backing (state={result.state}); "
                              "growth stays fenced until the next formal admission", flush=True)
                    if self._host_managed and not closed:
                        _revoke_domain_growth(self)
                except _NativeStillBusy as busy:
                    # The unload hook has no failure channel (see the release_all comment above): a lock held past the
                    # deadline stops the eviction where it is - nothing more is released - instead of raising, the same
                    # outcome as release_all's non-READY branch. Growth stays fenced and an Owned adapter is
                    # re-admitted formally. Every other refusal (API error, UNKNOWN, not enrolled) still raises.
                    self._domain.shared_growth_fenced = True
                    if identity.owner_epoch:
                        self._needs_readmission = True
                    _log.warning("[qf_native] %s; full eviction stopped, growth stays fenced until the next formal "
                                 "admission", busy)
        # Keep the view and canonical object alive across host deregistration.
        return super().detach(unpatch_all=unpatch_all)


class QFModelPatcher(comfy.model_patcher.ModelPatcher):
    """Logical MODEL; canonical dependencies own native bytes, this patcher owns Torch bytes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                  "quantfunc.request_geometry", QFModelPatcher._sampling_geometry)

    @staticmethod
    def _sampling_geometry(executor, *args, **kwargs):
        model = executor.class_obj.model_patcher.model
        token = _QF_SAMPLING_GEOMETRY.set((model, kwargs.get("latent_shapes")))
        try:
            return executor(*args, **kwargs)
        finally:
            _QF_SAMPLING_GEOMETRY.reset(token)

    def _engine(self):
        return getattr(getattr(self, "model", None), "_qf", None)

    def _native_dependencies(self):
        engine = self._engine()
        if engine is None:
            raise RuntimeError("QuantFunc MODEL has no engine identity")
        entry = engine.prepare_resource()[0] if isinstance(engine, QFLazyEngine) else engine
        dependencies = list(canonical_resource_adapters(entry))
        for dependency in dependencies:
            dependency.preflight_host_load()
        self.set_additional_models("quantfunc.native_resources", dependencies)
        return dependencies

    def model_patches_models(self):
        # Direct load_models_gpu uses this path, not nested additional_models.
        return list(dict.fromkeys(super().model_patches_models() + self._native_dependencies()))

    def get_additional_models(self):
        # Also refresh before official nested traversal and after clone/LoRA.
        self._native_dependencies()
        return super().get_additional_models()

    # Shadow remains relevant to legacy CPU-backup ownership only. Physical GPU
    # accounting and release belong to canonical dependencies for EVERY output.

    def _is_shadow(self):
        return bool(getattr(getattr(self, "model", None), "_qf_shadow", False))

    def model_size(self):
        return super().model_size()

    def loaded_size(self):
        return super().loaded_size()

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        if device_to != self.load_device:
            raise ValueError("a QuantFunc MODEL cannot migrate its native resources")
        for adapter in self._native_dependencies():
            adapter.require_load_contract()
            _ready_grant(adapter._resource)
        # Native bytes are loaded by the canonical dependencies. The official
        # base implementation owns only this model's ordinary Torch parameters.
        return super().partially_load(device_to, extra_memory,
                                      force_patch_weights=force_patch_weights)

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        return super().partially_unload(device_to, memory_to_free,
                                        force_patch_weights=force_patch_weights)


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
        """Detach logical patches only; canonical dependencies own native eviction."""
        # Only logical patches belong here. Native adapters retain their own
        # full-detach refusal; detaching a clone never releases their views.
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
