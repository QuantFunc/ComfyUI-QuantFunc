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
  in the engine's per-step caches; a CONTENT hash is equally wrong (it collides two branches with identical
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
- Session lifecycle: LAZY denoise_begin on the FIRST step of a run; finalize+end at
  process_latent_out (normal end); extra_conds closes at RUN START any session STRANDED by an
  Interrupt (comfy skips process_latent_in for empty/denoise=1.0 latents + caches+reuses this
  instance across requeues), so the next run never continues an aborted one (its cached state);
  end-on-failure per step.
- Co-eviction: the native resource adapters (QFNativeResourcePatcher) are ordinary ComfyUI models. When comfy needs
  the VRAM for a sibling model it asks them back (release_eligible / release_all: the engine's own walk) and they report
  the real freed bytes, so comfy's ledger is honest.
"""
import ctypes
from contextlib import contextmanager
from contextvars import ContextVar
import functools
import hashlib
import inspect
import json
import os
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
# The create inputs of the loader running now (set by the package's loader core). family_build stamps them on every model
# its build makes, LoRA rebuilds included, so the pipeline those models bind records which loader inputs make it.
LOADER_SIG = ContextVar("quantfunc_loader_sig", default=None)
# The API key of the running loader's own field (None = QUANTFUNC_API_KEY / config.json), published while the loader
# node runs (qf_api_key.add_api_key_input): family_build hands it to every lazy engine its build makes, LoRA rebuilds
# included.
LOADER_API_KEY = ContextVar("quantfunc_loader_api_key", default=None)


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
    """SHARED per-cond-group interrupt poll that never strands an open engine session (ONE copy for every
    family's per-cond-group loop). comfy's
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
    The engine keys its per-step caches off this ABI value, including a STATEFUL one that carries state from
    step to step. The key MUST be:
      (a) DISTINCT per distinct conditioning — else a cache emits one branch's projected text
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
          DIFFERENT-branch cached state = stale wrong-content reuse (the same defect class as (a), triggered by
          reordering; the caches never re-verify content on a key hit).
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
            k = len(self._uuid_to_key) + 1            # first-seen order -> {1,2,...}; non-zero (caching ON)
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
        raise RuntimeError("QF stub diffusion_model must never be called - _apply_model is overridden; "
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
    GC'd — no model<->engine cycle) + a factory that, at real
    engine create, binds every SURVIVING model to the resolved ckey.

    Returns (factory, register_model): pass `factory` to QFLazyEngine; call
    `register_model(m)` for every model the build creates."""
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


@qfe.console_safe_methods   # an exception leaving it is console-safe (#738)
class QFSessionModelMixin:
    """Shared base for every QuantFunc native-session model wrapper (LTX-2.5 / MiniMax-H3 / Krea-2 / Qwen-Image-2.1).

    These wrappers all drive the SAME engine external-denoise seam (begin → per-sampler-step
    quantfunc_denoise_step → finalize) over comfy's stock KSampler, differing only in the
    MODEL-SPECIFIC parts (context source, latent packing, per-step param wiring, output shape).
    This mixin holds the parts that are IDENTICAL across models so they live once, not once per
    class; each model subclasses it alongside its comfy base
    (e.g. `class QFKrea2Model(QFSessionModelMixin, comfy.model_base.Krea2)` in the Krea-2 family
    module — this substrate is family-AGNOSTIC and defines no model class itself).

    It carries the pieces every family's _apply_model shares verbatim: the sigma-schedule-derived step index
    and the quantfunc_denoise_step call with its session-clearing error handling (the per-family failure-message
    prefix is passed in).
    """

    # ── [step-cache] runtime step-cache threshold (SAME session-knob class as residency:
    # merged into the denoise_begin options_json below, re-read by the engine at EVERY
    # session begin → a widget change takes effect on the NEXT run with NO rebuild; the
    # knob never enters create_cfg/ckey). 0.0 = OFF: the keys are OMITTED entirely, the
    # engine spec stays cache_mode=0/thresh=0 → the step path is BYTE-IDENTICAL (the
    # engine-side off-path guarantee, the engine's cache layer the engine's step-cache wrapper). >0 arms
    # the step cache with this threshold. ──
    _step_cache = 0.0             # class default; loaders set the widget value (step_cache)
    _block_cache = 0.0            # class default; loaders set the widget value (block_cache)
    _sparse = 1.0                 # class default; 1.0 = dense (loaders set the sparse widget)

    def set_step_cache(self, t):
        self._step_cache = float(t)

    def set_block_cache(self, t):
        self._block_cache = float(t)

    def set_attn_backend(self, v):
        # [runtime dial 2026-08-29] session knob — rides residency_opts() into EVERY
        # denoise_begin; the engine swaps the per-forward dispatch string (no rebuild).
        # Every loader calls this on the model it builds, so the add-on attention choice is refused here on an engine
        # without it: at the loader, before any model loads (an older engine does not run it as chosen).
        v = str(v or "auto")
        if v == "qfa" and not engine_routes_attention():
            raise RuntimeError(f"QuantFunc: attention_backend '{v}' needs a newer QuantFunc engine; the installed one does "
                               "not have it. Pick auto, or update the QuantFunc engine.")
        self._attn_backend = v

    def set_sol_tau(self, v):
        # [sol-tau dial 2026-08-31] the ONE user-facing attention dial (user "就一个
        # 就好"). LOWER = closer to exact (<= -8 = exact), HIGHER = faster. Same
        # rides-residency_opts session-knob class as set_attn_backend.
        try:
            self._sol_tau = float(v)
        except (TypeError, ValueError):
            self._sol_tau = 1.0

    def set_video_enhance(self, on):
        # [quality_enhance 2026-09-25] the loaders' switch → the engine's `video_enhance` (OFF = the engine's faster default for
        # the model family; ON = maximum quality). Rides residency_opts like the other dials; `quality` is never sent.
        self._video_enhance = bool(on)

    # Every loader-set session dial: each set_* on this mixin or a family model writes exactly one of these (a death
    # rule in loader_dispatch_test checks the setters, subclasses included). A QuantFuncNativeLoRA rebuild hands back a
    # NEW model, so QFModelPatcher.adopt_comfy_state_from carries these; a dial missing here would make the LoRA'd model
    # run that dial's default without a word.
    _SESSION_DIALS = ("_step_cache", "_block_cache", "_sparse", "_attn_backend", "_sol_tau", "_video_enhance")

    def _stage_schedule(self, transformer_options, tag):
        """The ONE sampler-schedule rule of the video families (LTX, H3). Their external session is driven one sampler
        stage at a time, by the sigma the sampler passes each step: its step count is the stage's own (len - 1), and a
        partial or trimmed range is accepted, so two-stage and double-sampling workflows run with no switch. Refused:
        no schedule, fewer than 2 sigmas, or a schedule that is not strictly decreasing (it would drive the session
        backwards). Returns the (first, last) sigma, or None when they are not numbers (the count is still set)."""
        sigmas = transformer_options.get("sample_sigmas") if isinstance(transformer_options, dict) else None
        if sigmas is None or len(sigmas) < 2:
            raise RuntimeError(
                f"qf_native {tag}: the sampler did not publish a sigma schedule "
                "(transformer_options['sample_sigmas']) - the engine session needs the step count. "
                "Use a stock KSampler / SamplerCustom on this model.")
        self._num_steps = len(sigmas) - 1
        try:
            first, last = float(sigmas[0]), float(sigmas[-1])
        except Exception:  # noqa: BLE001 - non-tensor sigmas: keep the count, skip the range check
            return None
        if first <= last:
            raise RuntimeError(
                f"qf_native {tag}: sigma schedule must be strictly DECREASING (got "
                f"{first:.4f}->{last:.4f}) - a non-decreasing schedule would drive the "
                f"session backwards.")
        return first, last

    def adopt_session_dials_from(self, src):
        """Copy the dials the loader set on SRC (the model a LoRA rebuild replaces) onto this model."""
        for name in type(self)._SESSION_DIALS:
            if name in vars(src):
                setattr(self, name, vars(src)[name])

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
        # [sol-tau dial] omitted at the 1.0 default (old-engine compatible: an older
        # .so refuses unknown keys LOUD, and default users never send it). Ghost-proof
        # despite the omission: the ENGINE resets an absent sol_tau to 1.0 at every
        # begin (absent = reset-to-default, not keep-current) — so dialing back to
        # 1.0 truly restores the default even on a reused engine-resident pipeline.
        st = float(getattr(self, "_sol_tau", 1.0) or 1.0)
        if abs(st - 1.0) > 1e-6:
            o["sol_tau"] = st
        o.update(self.dial_opts())
        return o

    def dial_opts(self):
        """The begin keys EVERY family's session takes, image and video alike (the image families send only these;
        the video residency keys are engine-refused there).
        - attention_backend: ALWAYS sent, "auto" included. The engine KEEPS the current backend when the key is
          absent, so omitting "auto" left the previous run's explicit choice in force (MEASURED: a run set back to
          auto rendered byte-for-byte as the flash run before it). "auto" is every family's create-time default.
        - video_enhance: ALWAYS sent, both states (the quality_enhance switch; what OFF does per model family is engine
          law). Every loader sets it on the model it builds, so a model without one is a wiring error: refused here,
          never a silent default. `quality` is never sent."""
        on = getattr(self, "_video_enhance", None)
        if on is None:
            raise RuntimeError("QuantFunc: this model has no quality_enhance setting (the loader sets one on every model it "
                               "builds; set_video_enhance was never called)")
        return {"attention_backend": str(getattr(self, "_attn_backend", "auto") or "auto"), "video_enhance": on}

    # ── comfy's per-model VRAM interface (2026-09-19, user: 「与 comfyui 打通,让它知道我们需要多少显存、当前占了多少」) ──
    # comfy asks a model TWO numbers and does the rest itself: `memory_required(shape)` — how much MORE VRAM one
    # forward of `shape` needs (sampler_helpers.estimate_memory → load_models_gpu(memory_required=…) → free_memory
    # unloads comfy's IDLE models, largest first, keeping the sampler's declared set; samplers.calc_cond_batch →
    # `need × 1.5 < free` decides cond batching) — and `loaded_size()`/`model_size()` — what it holds (the patcher,
    # below). A cold request (no pipeline yet) takes comfy's own estimate as the floor (#716: no native pre-create
    # working-set estimate exists yet); a hot pipeline asks native numbers only (D3: comfy-side + quantfunc_vram_need_bytes),
    # while actual residency comes from quantfunc_resident_vram_bytes. Comfy's own
    # allocator makes the room before the denoise starts; nothing here reaches into comfy's state (user: 「不要 hack」).
    # What this REPLACES (git: 52ec260 / 9b6f7d3 / a6ec609): this function returned the comfy-side bytes ONLY — a
    # deliberate under-report so comfy would never evict the engine for a stage-2 shadow load (2026-08-22: the
    # torch-WAN activation estimate, 2.36 GB at 33.6k tok, evicted the ENGINE when the 64 MB shadow patcher loaded
    # → 15 s/prompt inter-stage thrash) — with the side effect that on a card where comfy's idle TE/VAEs FIT, comfy
    # kept them resident through the denoise and the engine shed its own weight pages every step (MEASURED 5090/H3
    # 800x768: 26–50 blocks re-copied per step). H3 additionally stacked a static per-element heuristic on top
    # (deleted with this change — it would double-count the real number).
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
        """ComfyUI's inference reserve for one forward of `input_shape`: native numbers once a pipeline exists (D3,
        tests-07 ruling 2026-09-24: "the plan side estimates; the upper layer doesn't invent numbers"); comfy's own
        estimate as the floor while none exists (#716, below).

        = comfy-side bytes (the plugin's own tensor copies) + the ENGINE's working set for that shape:
        quantfunc_vram_need_bytes(latent [B,C,(T,)H,W]) — the primary transformer's working set (MEASURED by an
        earlier forward; else the same spatial shape at another batch scaled; else the model's config estimate) + the
        plan margin − the allocator's cached pool it already holds. The engine is asked through the CACHED real
        handle when one exists and is NEVER created here (comfy's eviction pass runs after this call).

        COLD (no pipeline exists for these weights yet — #716, tests-07 ruling (a), interim): ComfyUI's own
        BaseModel.memory_required for the shape is the FLOOR, so comfy makes room BEFORE the engine's first forward.
        The engine has no pre-create working-set estimate yet (G3). MEASURED without the floor: LTX-2.5 1920x1088
        cold asked 127 MB, comfy kept its 11 GB text encoder, and step 0 hit the ceiling (release-ltx25-allin-1); with
        it, 10/10 (host-vram-ltx25-accept-26). The G3 fix deletes this floor.
        HOT (a pipeline exists; a stage-2 model shares it): native numbers only (D3). comfy's torch estimate is never
        a floor there: at every stage it made comfy evict every other model (38 GB for a Wan 80x80x21 latent), the
        inter-stage thrash D3 removed.
        The engine WEIGHTS are never counted here: the canonical resource adapters carry them as their own
        LoadedModel (model_size = the Prepared capacity). Missing ABI support and failed native queries propagate.
        Accepted by the user (2026-09-19 「comfyui 路径不合并 CFG 就好」): with a real need comfy runs cond/uncond
        un-batched on a card that cannot hold 1.5x the B=2 working set; our own full-pipeline path keeps CFG batched."""
        comfy_side = self._qf_comfy_side_bytes(input_shape, cond_shapes)
        lazy = eng = getattr(self, "_qf", None)
        need = 0
        cold_need = cold_note = None
        if eng is not None:
            # Lookup may bind an existing cached handle, but must never create ahead of host admission. Query errors
            # must reach the host instead of authorizing execution with a fabricated zero demand.
            peek = getattr(eng, "ensure_if_cached", None)
            if callable(peek):
                eng = peek()
            if eng is not None:
                need = int(eng.vram_need_bytes(self._qf_engine_latent_dims(input_shape)))
            else:
                # #738 COLD: the engine's own create-time need (the plan side's number, header facts only) — so comfy
                # evicts its idle models before our create (MEASURED: "need 0 MB cold" left a 4.4 GB idle TE resident
                # and Krea-2's create failed physically on a 6 GB card). It stays UNDER comfy's own estimate as the
                # floor: the create-time need has no forward working set, and the first forward follows this one ask
                # (#716: LTX-2.5 1920x1088 cold needed comfy to evict an 11 GB TE for it). Unknown -> the floor alone.
                cold_need, cold_note = lazy.cold_vram_need_bytes()
        if cold_need is not None:
            need = int(cold_need)
        floor = int(super().memory_required(input_shape, cond_shapes=cond_shapes or {})) if eng is None else 0
        total = int(max(floor, comfy_side + need))
        # One line per CHANGE of the answer (comfy asks per estimate + per cond-batch decision): the numbers comfy
        # will act on, so a ledger question is answerable from the log, not a guess.
        sig = (tuple(int(d) for d in input_shape), comfy_side, need, floor, eng is None)
        if sig != getattr(self, "_qf_ledger_last", None):
            self._qf_ledger_last = sig
            try:
                hold = f"{int(eng.resident_vram_bytes()) >> 20} MB" if eng is not None else "0 MB"
            except Exception as error:  # diagnostic only; never invent a measured zero
                hold = f"unknown ({type(error).__name__}: {error})"
            qfe.info("[qf_native] VRAM ledger: memory_required%s = %d MB (comfy-side %d MB, engine need %d MB%s); "
                  "engine hold %s"
                  % (list(sig[0]), total >> 20, comfy_side >> 20, need >> 20,
                     (" (cold: no pipeline yet; the engine's create-time need, floored by comfy's own estimate %d MB)"
                      % (floor >> 20) if cold_need is not None else
                      " (cold: no pipeline yet; the engine's need is %s: comfy's own estimate %d MB is the floor)"
                      % (cold_note, floor >> 20) if cold_note else
                      " (cold: no pipeline yet; comfy's own estimate %d MB is the floor)" % (floor >> 20))
                     if eng is None else
                     "" if need else " (0: covered by what it holds, or nothing measured yet)",
                     hold), flush=True)
        return total

    def _sigma_step_index(self, sigma, sig_all, transformer_options):
        """SAMPLER step index from the sigma's position in the schedule — call-count-independent.

        With CONDRegular a different-length cond/uncond arrive as SEPARATE _apply_model calls (2
        per sampler step, not one batched B==2), so a per-call counter double-counts and overruns
        total_steps. The sigma is identical for cond and uncond of the same step, so its schedule
        index is the true step index. Falls back to the per-call counter (self._step_i) when the
        sampler exposes no schedule. Shared by every family."""
        step_index = self._step_i
        _sched = transformer_options.get("sample_sigmas") if isinstance(transformer_options, dict) else None
        if _sched is not None and len(_sched) >= 2:
            _cur = float(sig_all[0].item()) if sig_all is not None else float(sigma)
            step_index = min(range(len(_sched) - 1), key=lambda k: abs(float(_sched[k]) - _cur))
        return step_index

    def _call_denoise_step(self, p, fail_prefix, fn_name="quantfunc_denoise_step"):
        """Run one quantfunc_denoise_step (or the same-contract `fn_name` export, e.g.
        quantfunc_denoise_step_refs), clearing the (GPU-resident) session on ANY failure so a
        mid-sample error never strands a stale session. `fail_prefix` is the model-specific
        message head (e.g. 'LTX denoise_step[..]'). Shared by every family."""
        try:
            st = getattr(self._qf.lib, fn_name)(self._qf.current_session, ctypes.byref(p))
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
                               "quantfunc_denoise_step_multi (D-class AV step) - rebuild/point "
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


@qfe.console_safe_methods   # an exception leaving it is console-safe (#738)
class QFImageSessionModel(QFSessionModelMixin):
    """The image families' seam (Krea-2, Qwen-Image-2.1), written once: comfy owns the text encoder, the VAE, the sampler
    and CFG; the engine owns only the denoise, through the generic external session (quantfunc_denoise_begin, then one
    quantfunc_denoise_step per cond group per sampler step). A family sets _TAG (its console / error tag), _NO_COND_HINT
    and _ENGINE_IGNORED_COND_KEYS — the comfy consumables it refuses
    LOUD, never drops, audited per family by tests/reject_list_completeness.py — and may override _check_conds (more
    run-start refusals) and _denoise_group (how one cond group's step is called)."""
    _TAG = ""
    _NO_COND_HINT = ""
    _REFUSED_HINT = ""          # appended to a refused-conditioning message
    _ENGINE_IGNORED_COND_KEYS = ()

    def __init__(self, model_config, engine, device=None):
        super().__init__(model_config, device=device)
        self.diffusion_model = _QFStub()
        # in == the latent channels of the model's own latent_format -> extra 0 -> no concat
        self.diffusion_model.arm_concat_shape(int(self.latent_format.latent_channels))
        self._qf = engine
        self._num_steps = 0
        self._step_i = 0
        self._ctx_key_assigner = _CtxKeyAssigner()
        self._sess_denoise = 0
        self._max_ctx_seq = 0
        self._out = None

    def _check_conds(self, kwargs):
        """A family's own run-start refusals, after the ignored-keys check (default: none)."""

    def extra_conds(self, **kwargs):
        was_open, ok = self._qf.end_session_if_open()   # run-start clean slate (#2 lifecycle)
        self._qf_needs_begin = True
        if was_open:
            qfe.info(f"[qf_native] {self._TAG}: closed a pre-existing session at run start "
                     f"(prior run interrupted/uncleaned); end ok={ok}")
        for _k in self._ENGINE_IGNORED_COND_KEYS:
            if kwargs.get(_k) is not None:
                raise RuntimeError(
                    f"qf_native {self._TAG}: '{_k}' conditioning is wired, but the native session "
                    f"cannot consume it - it would be silently ignored, so it is refused. "
                    f"Remove the node feeding '{_k}'{self._REFUSED_HINT}.")
        self._check_conds(kwargs)
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            self._max_ctx_seq = max(self._max_ctx_seq, int(cross_attn.shape[1]))
        return out

    def _begin(self, x_group, ctx_group):
        lib = self._qf.lib                       # MATERIALIZE FIRST (deferred-wrapper no-op close)
        self._qf.end_session_if_open()
        bpx = qfe.DenoiseBeginParams()
        ctypes.memset(ctypes.byref(bpx), 0, ctypes.sizeof(bpx))
        bpx.struct_size = ctypes.sizeof(bpx)
        _s = int(self.latent_format.spacial_downscale_ratio)   # latent -> pixels, the model's latent_format
        bpx.width = int(x_group.shape[-1]) * _s
        bpx.height = int(x_group.shape[-2]) * _s
        bpx.num_steps = self._num_steps
        _max_seq = max(self._max_ctx_seq, int(ctx_group.shape[1]))
        bpx.max_context_dims = (ctypes.c_int * 3)(
            int(ctx_group.shape[0]), _max_seq, int(ctx_group.shape[2]))
        self._max_ctx_seq = 0
        bpx.cond_dtype = _qf_dtype(ctx_group.dtype)
        # IMAGE session: only the dials every family takes (the video residency_opts keys are engine-refused here).
        bpx._opts = json.dumps(self.dial_opts()).encode()
        bpx.options_json = bpx._opts
        session = ctypes.c_void_p()
        st = lib.quantfunc_denoise_begin(self._qf.pipeline, ctypes.byref(bpx), ctypes.byref(session))
        self._begin_keep = bpx
        if st != qfe.QUANTFUNC_OK:
            raise RuntimeError(f"denoise_begin ({self._TAG}) failed: {qfe.last_err(lib)}")
        self._qf.current_session = session
        self._step_i = 0
        self._sess_denoise = 0
        self._ctx_key_assigner.reset()
        qfe.info(f"[qf_native] {self._TAG.upper()} SESSION OPEN handle={session.value:#x} steps={self._num_steps} "
                 f"latent={tuple(x_group.shape)} cond={tuple(ctx_group.shape)}")

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        sigma = t
        ctx = c_crossattn
        if ctx is None:
            raise RuntimeError(f"qf_native {self._TAG}: no c_crossattn cond - {self._NO_COND_HINT}")
        if control is not None:
            raise RuntimeError(
                f"qf_native {self._TAG}: a ControlNet is wired, but the native session consumes no "
                "control input - it would be silently ignored, so it is refused.")
        if c_concat is not None:
            raise RuntimeError(
                f"qf_native {self._TAG}: c_concat conditioning is not part of the {self._TAG} seam - remove "
                "the node feeding it.")
        # BF16-latent/BF16-cond families (the engine's activation dtype); comfy may hand FP32 and may keep the cond
        # host-side (measured on cu12/3090, 2026-08-30: a CPU cond data_ptr reached the engine's cond copy as cudaMemcpy
        # 'invalid argument'), so cast to the LATENT's device + bf16 in one .to() (a no-op when already there). The
        # step ABI takes DEVICE pointers.
        _dev = x.device
        x_bf = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        if ctx.dtype != torch.bfloat16 or ctx.device != _dev:
            ctx = ctx.to(device=_dev, dtype=torch.bfloat16)
        sig_all = sigma.reshape(-1) if torch.is_tensor(sigma) else None
        sched = transformer_options.get("sample_sigmas", None)
        if sched is not None:
            self._num_steps = max(1, int(sched.numel()) - 1)
        elif self._num_steps <= 0:
            self._num_steps = 1
        xin = x_bf
        B = int(xin.shape[0])
        if getattr(self, "_qf_needs_begin", True) or self._qf.current_session is None:
            self._begin(xin[0:1], ctx[0:1])
            self._qf_needs_begin = False
        if self._out is None or self._out.shape != xin.shape or self._out.dtype != xin.dtype \
                or self._out.device != xin.device:
            self._out = torch.empty_like(xin)
        # comfy carries the per-conditioning uuids as transformer_options["uuids"] (samplers.py:324/511). A wrong key
        # silently yields ctx key 0 = the engine's step caches OFF: Krea-2 read "cond_uuids" until the Qwen-Image-2.1
        # seam, cloned from it, logged cfg_context_key=0 (2026-09-22) — the reason this loop now exists once.
        cuuids = transformer_options.get("uuids") if isinstance(transformer_options, dict) else None
        step_index = self._sigma_step_index(sigma, sig_all, transformer_options)
        for i in range(B):
            _interrupt_poll_end_session_on_raise(self._qf)
            xi = xin[i:i + 1].contiguous()
            oi = self._out[i:i + 1]
            # The engine session is IMAGE 4D. A latent format that rides a singleton T axis (Krea-2 on comfy's Wan21
            # format, [B,C,T=1,H,W]) is squeezed for the ABI; both views share storage, so the velocity lands in _out.
            if xi.dim() == 5 and xi.shape[2] == 1:
                xi = xi.squeeze(2).contiguous()
                oi = oi.squeeze(2)
            ci = ctx[i:i + 1].contiguous()
            cuid = cuuids[i] if (cuuids is not None and i < len(cuuids)) else None
            ctx_key = self._ctx_key_assigner.key(cuid)
            sig_i = float(sig_all[i].item()) if (sig_all is not None and sig_all.numel() >= B) else \
                (float(sig_all[0].item()) if sig_all is not None else float(sigma))
            p = qfe.DenoiseStepParams()
            ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
            p.struct_size = ctypes.sizeof(p)
            p.latent_in = xi.data_ptr()
            p.velocity_out = oi.data_ptr()
            p.velocity_out_capacity = oi.numel() * oi.element_size()
            p.dims[:xi.dim()] = list(xi.shape)          # unused tail stays 0 (memset above); length = the ABI struct's
            p.dtype = _qf_dtype(xi.dtype)
            p.sigma = sig_i
            p.step_index = step_index
            p.total_steps = self._num_steps
            p.context = ci.data_ptr()
            p.context_dims = (ctypes.c_int * 3)(*ci.shape)
            p.context_dtype = _qf_dtype(ci.dtype)
            p.cfg_context_key = ctx_key
            self._denoise_group(p, i, xi, ci, kwargs, step_index, ctx_key)
            self._qf.step_count += 1
            self._sess_denoise += 1
        self._step_i += 1
        self._qf.sampler_step_count += 1
        return self.model_sampling.calculate_denoised(sigma, self._out.float(), x)

    def _denoise_group(self, p, i, xi, ci, kwargs, step_index, ctx_key):
        """One cond group's step (default: the plain step; Qwen-Image-2.1 adds its reference latents)."""
        self._call_denoise_step(p, f"{self._TAG} denoise_step[step={step_index},group={i},key={ctx_key}]")

    def process_latent_out(self, latent):
        # NORMAL end-of-sampling: close the session (no finalize — the image seam has no masked-blend); an INTERRUPT
        # skips this and extra_conds closes it at the next run start.
        if self._qf.current_session is not None:
            self._qf.end_session_if_open()
            qfe.info(f"[qf_native] {self._TAG.upper()} SESSION CLOSED after {self._step_i} sampler steps, "
                     f"{self._sess_denoise} denoise_step calls")
        return super().process_latent_out(latent)


class QFLazyEngine:
    """A QFEngineHandle that materializes ON FIRST REAL USE.

    WHY: comfy's node-output cache pins every intermediate model of a chain (loader → LoRA → LoRA …). The
    pipeline is created only when the sampler first touches a model, the create carries the weights only, and
    every LoRA set of one model shares that one pipeline: ensure() applies THIS consumer's set in place
    (_apply_runtime_lora; user rule 2026-09-24 「换 LoRA 也不重建」).

    The proxy answers the CHEAP part of the handle surface locally while unmaterialized (an
    engine that does not exist holds no VRAM and has no session), and materializes only for
    the members that genuinely need a live pipeline (`lib`, `pipeline`). Explicit members, no
    __getattr__ magic: a typo must fail loud, not silently forward.
    """

    def __init__(self, factory, api_key=None, read_auth=None):
        self._factory = factory          # () -> (QFEngineHandle, ckey)
        self._real = None
        self._prepared_entry = None
        self._ckey = None                # the materialized handle's cache key
        # This consumer's declarative LoRA set, set by tag_lora_rebuild: each chained LoRA node builds a fresh lazy
        # engine with its whole cumulative stack, so a raw loader output holds [] and deleting a LoRA node needs no cleanup.
        self._lora = []
        # This consumer's API key: its loader field's key, or None for the key of QUANTFUNC_API_KEY / config.json, which
        # read_auth() (the package's _read_auth) reads only when a pipeline signed in with a field key must switch back.
        self._api_key = api_key
        self._read_auth = read_auth
        # Persistent VRAM capacity comes only from the configured Prepared
        # resource. CPU-backup bytes are a separate, currently unknown ledger.
        self.capacity_bytes = None
        self.footprint_bytes = 0
        self.step_count = 0
        self.sampler_step_count = 0

    # ---- materialization ----
    def prepare_resource(self):
        """Resolve the actual cache recipe without calling model creation.

        Runs the family's factory through the shared _get_engine prepare-only
        branch. No lazy .lib access here.
        """
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
        self._bind_capacity(real.capacity_bytes)
        self._real, self._ckey = real, ckey
        self._real.step_count = self.step_count
        self._real.sampler_step_count = self.sampler_step_count
        return self._real

    def _materialize_prepared(self, entry):
        """The one cold-create path used by every family."""
        if self._real is not None and self._real.pipeline is not None:
            return self._real
        if entry is not self._prepared_entry:
            raise RuntimeError("QuantFunc prepared identity is not bound to this lazy engine")
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
        if self._real is None:
            entry, ckey = self.prepare_resource()
            if isinstance(entry, QFPreparedEntry):
                entry.materialize(preferred=self)
            else:
                self._adopt_real(entry, ckey)
        # The ONE materialization chokepoint also puts this consumer's API key and LoRA set on the shared pipeline, so
        # no family needs a per-begin hook it could forget. The key first: a LoRA update may fetch with it.
        self._apply_runtime_api_key()
        self._apply_runtime_lora()
        return self._real

    def _apply_runtime_api_key(self):
        """Sign the shared pipeline in with THIS consumer's API key before its run: one quantfunc_set_api_key, only
        when it differs from the key the pipeline runs on (a setting that is not the weights never rebuilds); the engine
        checks the key when the run begins. An emptied field goes back to the key of QUANTFUNC_API_KEY / config.json.
        With no key there the run is refused: an empty key would turn the engine's key checks off. Returns True when a
        switch ran."""
        real, want = self._real, self._api_key
        if getattr(real, "applied_api_key", None) == want:
            return False
        key = want if want is not None else self._read_auth()[0]
        if not key:
            raise RuntimeError(
                "qf_native: this model is signed in with the key from a loader's API key field. This loader's field is "
                "empty and there is no key in config.json or QUANTFUNC_API_KEY to switch to: enter a key in this "
                "loader's API key field, or restore the key in config.json.")
        real.set_api_key(key)      # a refusal leaves the pipeline on its key, so the next run retries
        real.applied_api_key = want
        qfe.info("[qf_native] API key switched in place (no reload): now "
                 + ("this loader's field key" if want is not None else "the key of QUANTFUNC_API_KEY / config.json"),
                 flush=True)
        return True

    def _apply_runtime_lora(self):
        """Put THIS consumer's LoRA set on the shared pipeline before its run: ONE declarative
        quantfunc_pipeline_update {"lora": [...]} (the full set; [] restores the base), only when the pipeline's
        applied set differs. The applied set lives on the REAL handle, shared by every lazy engine of the same
        weights, so an unchanged set costs one string compare and never touches the engine. Returns True when an
        update ran."""
        real, want = self._real, self._lora_sig()
        if getattr(real, "applied_lora_sig", "[]") == want:
            return False
        if getattr(real, "current_session", None):
            real.end_session_if_open()   # a stale session from an interrupted run blocks the mutation lease
            if getattr(real, "current_session", None):
                # the begin path's ONE bounded recovery: an interrupted run's last step may still be draining, and
                # its refused end RETAINED the pointer, so one more end can win once the step lands
                time.sleep(2.0)
                real.end_session_if_open()
        if getattr(real, "current_session", None):
            raise RuntimeError(
                "qf_native: the LoRA set changed while this model's generation is still running. The engine applies "
                "a LoRA change only between generations; re-queue the prompt.")
        lora = self.lora_set()
        # UNKNOWN until the engine confirms: a refused update may have left the previous set OR rolled the weights
        # back to the base mid-apply, so after any failure the next run must re-send its set, never trust a mark.
        real.applied_lora_sig = None
        try:
            real.pipeline_update({"lora": lora})
        except RuntimeError as e:
            # the engine names the LoRA it cannot load ("'<path>' matched 0 target modules ..."); a busy engine does not.
            # Its message reached us through console_safe, so the path is matched as that console writes it.
            if any(entry.get("path") and qfe.console_safe(entry["path"]) in str(e) for entry in lora):
                warn_lora_not_loaded()
            raise
        real.applied_lora_sig = want
        qfe.info(f"[qf_native] LoRA set applied in place (no reload): {len(lora)} LoRA(s)", flush=True)
        return True

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

    # ---- LoRA: the consumer's declarative set (tag_lora_rebuild sets it), applied in place at ensure() ----
    def lora_set(self):
        """The declarative 'lora' list pipeline_update carries, in the user's chain order."""
        return [dict(e) for e in self._lora]

    def _lora_sig(self):
        return json.dumps(self.lora_set(), sort_keys=True)

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

    # ---- lifecycle: all no-ops while unmaterialized (nothing exists to close/free/destroy) ----
    def end_session_if_open(self):
        return (False, True) if self._real is None else self._real.end_session_if_open()

    def resident_vram_bytes(self):
        return 0 if self._real is None else self._real.resident_vram_bytes()

    def vram_need_bytes(self, latent_shape):
        entry = self._real or self.ensure_if_cached()
        if entry is None:
            # A cold/unestimable result is zero: QFSessionModelMixin.memory_required then asks cold_vram_need_bytes
            # (#738), and only an engine that cannot say falls back to comfy's own estimate as the floor (#716).
            return 0
        return entry.vram_need_bytes(latent_shape)

    def cold_vram_need_bytes(self):
        """#738 No pipeline yet: the engine's create-time need for these weights, from the Prepared resource the cold
        path already configured (never a create): a qf_engine.ColdNeed (bytes None = the engine cannot say; its note
        says why, None when that is by design - an older library)."""
        return self._prepared_entry.resource.cold_vram_need_bytes()

    # NOTE: deliberately NO destroy() and NO release() on the wrapper: native backing is evicted by the canonical
    # resource adapters, and the cache's _sweep_dead_pipelines destroys REAL handles, never wrappers. A caller
    # reaching for a wrapper teardown would reproduce the shared-handle UAF the liveness gate prevents.


# ── shared sidecar-LoRA rebuild contract (CR simplicity: ONE definition, not one per family) ──
# A downstream QuantFuncNativeLoRA node rebuilds the PATCHER for the accumulated set (the pipeline is shared and
# the set goes on in place, see QFLazyEngine). Every family's builder tags its model with these
# two attributes through tag_lora_rebuild(); the LoRA node reads them through lora_stack_of()/
# rebuild_of(). Names live here so the three seams cannot drift apart.
QF_LORA_STACK_ATTR = "_qf_lora_stack"
QF_LORA_REBUILD_ATTR = "_qf_rebuild"


_ABSENT = object()


def ensure_model_config_attrs(model_config):
    """comfy's model_config classes grow attributes over releases; a seam that builds one directly
    must tolerate an older/newer comfy. ONE definition (was copy-pasted in every family builder).

    Looked up STATICALLY: comfy's BASE answers any missing attribute through a __getattr__ that logs
    "you accessed scaled_fp8 from the model config object which doesn't exist. Please fix your code"
    and returns None (0.37 dropped scaled_fp8). hasattr() therefore warned on every load AND saw the
    attribute as present, so it was never set and every later read warned again."""
    for attr, default in (("manual_cast_dtype", None), ("custom_operations", None),
                          ("optimizations", {}), ("scaled_fp8", None)):
        if inspect.getattr_static(model_config, attr, _ABSENT) is _ABSENT:
            setattr(model_config, attr, default)
    return model_config


def stage_config_package(bundle_dir, transformer1_path, extra_links=None):
    """A config-complete PACKAGE dir for the engine's `denoise_only` create, holding NO weights.

    The file-based loader hands us a bare transformer .safetensors FILE, but the engine still needs the arch + VAE
    CONFIGS for session geometry (denoise_only skips the TE+VAE WEIGHTS, not the configs). So the family's shipped
    CONFIG bundle (model_index.json + transformer/ vae/ config.json - tiny JSON) is COPIED into ComfyUI's OWN temp dir
    (folder_paths.get_temp_directory(), never system /tmp), keyed deterministically by (bundle, realpath(files)): one
    dir per weight file. Rebuilt fresh each call (configs are tiny), so a re-pick can't leave a stale file. The weights
    reach the engine as its transformer_path create input (Krea-2 and Qwen-Image-2.1 always; LTX-2.5 and MiniMax-H3 on an
    engine with the _WEIGHT_PATHS_SYMBOL export), or as the links stage_denoise_only_package adds (LTX-2.5 and
    MiniMax-H3 on an older engine)."""
    import shutil
    import folder_paths
    if not os.path.isdir(bundle_dir):
        raise RuntimeError(
            f"qf_native: config bundle missing: {bundle_dir} - this family's arch/VAE configs are "
            f"not shipped in the plugin (configs/<family>/). Cannot stage a denoise_only package.")
    real1 = os.path.realpath(transformer1_path)
    # extra_links: {subdir: target_path} (LTX-2.5) - the deterministic key covers them so a re-pick restages.
    extra_links = {k: os.path.realpath(v) for k, v in (extra_links or {}).items() if v}
    key_src = "|".join([bundle_dir, real1] +
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
    shutil.copytree(bundle_dir, stage)   # tiny config JSONs only - no weights
    return stage


# The export engines carry (0.0.17+, #777) when every family loads its weights from paths - the create's
# transformer_path, plus LTX-2.5's connectors_path - with model_dir a config-only package. Without it LTX-2.5 reads its
# connectors only from the package and MiniMax-H3 its folded checkpoint's fold count only beside the weights, so those two
# keep the link.
_WEIGHT_PATHS_SYMBOL = "quantfunc_weight_paths"
# The export engines carry (0.0.17+) when they route attention per GPU and load the add-on attention library; the
# loaders' add-on choice needs it (set_attn_backend -> engine_routes_attention). MiniMax-H3 reads the same export for
# its auto.
_ATTN_ROUTE_SYMBOL = "quantfunc_attention_auto_route"


def engine_reads_weight_paths():
    """True when the loaded QuantFunc engine loads LTX-2.5 and MiniMax-H3 weights from paths (no link needed)."""
    return hasattr(qfe.load_lib(), _WEIGHT_PATHS_SYMBOL)


def engine_routes_attention():
    """True when the loaded QuantFunc engine carries the add-on attention (its per-GPU attention route)."""
    return hasattr(qfe.load_lib(), _ATTN_ROUTE_SYMBOL)


def stage_denoise_only_package(bundle_dir, transformer1_path, extra_links=None):
    """The package of LTX-2.5 and MiniMax-H3: the config package plus the weight files SYMLINKED in, the transformer as
    transformer/model.safetensors. For an engine without the _WEIGHT_PATHS_SYMBOL export (before 0.0.17): it reads
    LTX-2.5's connectors only from the package, and MiniMax-H3's folded-checkpoint qf_adaln_fold_dropped_resq only from
    the config.json beside the weights (#777). extra_links: {subdir: target_path} - single-expert AV families link MORE weight files
    (ltx2: the SAME single xfm file into connectors/ [#565 comfy25 prefix branch], the gemma with-proj TE into
    text_encoder/ [connector aggregate_embed], the audio_vae file [engine has_audio_ discriminant = weights presence])."""
    stage = stage_config_package(bundle_dir, transformer1_path, extra_links)
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
                    f"qf_native: cannot link {target} into ComfyUI's temp directory ({exc}). LTX-2.5 and MiniMax-H3 "
                    f"need that link with this QuantFunc engine when the model is on another drive or a network share "
                    f"and Windows Developer Mode is off; a QuantFunc engine update removes this. For now enable Windows "
                    f"Developer Mode, or move the model to the drive ComfyUI's temp directory is on.") from exc
    _link_expert("transformer", os.path.realpath(transformer1_path))
    for sub, target in sorted((extra_links or {}).items()):
        if target:
            _link_expert(sub, os.path.realpath(target))
    return stage


def refuse_all_zero_initial_latent(xin, tag):
    """ALL-ZERO INITIAL LATENT GUARD — SHARED across every family seam (wan/LTX/LTX-AV/H3;
    one mechanism, N users). An Empty latent whose first sampler stage has add_noise=disable
    hands the engine a pure-zero tensor at sigma_max: the engine's arithmetic divides zero by
    zero (0/0 = NaN), the NaN propagates silently through every step, and VAEDecode writes
    an EXACT-BLACK video with zero errors anywhere — the worst failure shape (measured on the
    user's wan run 2026-08-21; the class is family-independent, any of our transformers NaNs the
    same way). Denoising pure zeros is never meaningful, so refuse LOUD at session begin,
    BEFORE any engine call, with the actual fix named. A noised latent / any i2v or
    latent-input flow is nonzero and never trips this."""
    if float(xin.abs().max()) == 0.0:
        raise RuntimeError(
            f"qf_native {tag}: the initial latent is ALL ZEROS - denoising pure zeros "
            f"cannot produce an image (it renders a BLACK video). "
            f"Almost always this means the FIRST sampler stage has add_noise=disable on an "
            f"Empty latent: set add_noise=enable on the first stage (official templates ship "
            f"it enabled; later stages keep disable - they receive the leftover-noise latent).")


def tag_lora_rebuild(patcher, lora_entries, rebuild):
    """Mark a freshly built patcher's model with its LoRA set + how to re-create with a new one, and hand that set to
    its lazy engine (applied in place at ensure()). Every family's build ends here, so this is the ONE wiring point."""
    m = patcher.model
    if not isinstance(m._qf, QFLazyEngine):   # the set lives on the lazy engine; any other holder would drop it silently
        raise TypeError(f"tag_lora_rebuild: {type(m).__name__}._qf must be a QFLazyEngine, not {type(m._qf).__name__}")
    m._qf._lora = [dict(e) for e in lora_entries]
    setattr(m, QF_LORA_STACK_ATTR, list(lora_entries))
    setattr(m, QF_LORA_REBUILD_ATTR, rebuild)
    return patcher


def family_build(deps, model_dir, create_extra, supported_model, unet_config, make_model, label, pinned_memory=False,
                 transformer_path=None):
    """The ONE builder every family returns through (it was a closure copied into each family module). build(lora_entries)
    makes a lazy engine for model_dir — the pipeline is created only when a sampler first needs it, and there is ONE per
    model file whatever the LoRA set: the weights are the cache key, and QFLazyEngine applies this patcher's set in place
    at run start — the family's comfy model around it, and its patcher, tagged so that QuantFuncNativeLoRA can rebuild the
    patcher for another set. One device capture per build drives both the logical patcher and the engine identity; the
    engine factory holds the model only weakly (make_engine_factory: a strong capture was a model -> engine -> factory ->
    model cycle, comfy's "Potential memory leak" warning). make_model(model_config, engine, device) returns the family's
    model (a model class fits). pinned_memory (the loaders' switch, default OFF) adds the engine's use_pinned_memory create
    key, so its two states are two pipelines: changing it reloads the model; OFF leaves the create config as it was. The
    loader field's API key (LOADER_API_KEY) goes to the create and to each lazy engine, never into the create config: every
    key shares the one pipeline, and the lazy engine switches its key in place. transformer_path: the weight file the
    engine loads the transformer from (a create input, so part of the pipeline's identity); None = from model_dir."""
    get_engine, bind_pipeline_model, read_auth = deps["get_engine"], deps["bind_pipeline_model"], deps["read_auth"]
    loader_sig = LOADER_SIG.get()
    api_key = LOADER_API_KEY.get()

    def build(lora_entries):
        create_cfg = dict(create_extra or {}, **({"use_pinned_memory": True} if pinned_memory else {})) or None
        device, device_idx = current_torch_device()
        factory, register_model = make_engine_factory(
            lambda: get_engine(model_dir, create_cfg=create_cfg, device_idx=device_idx, api_key=api_key,
                               transformer_path=transformer_path),
            bind_pipeline_model)
        model_config = supported_model(dict(unet_config))
        ensure_model_config_attrs(model_config)
        model = make_model(model_config, QFLazyEngine(factory, api_key=api_key, read_auth=read_auth), device)
        model._qf_loader_sig = loader_sig
        register_model(model)
        patcher = QFModelPatcher(model, load_device=device, offload_device=comfy.model_management.unet_offload_device())
        qfe.info(label)
        return tag_lora_rebuild(patcher, lora_entries, build)
    return build


def lora_stack_of(patcher):
    return list(getattr(getattr(patcher, "model", None), QF_LORA_STACK_ATTR, []) or [])


def rebuild_of(patcher):
    return getattr(getattr(patcher, "model", None), QF_LORA_REBUILD_ATTR, None)


_CANONICAL_LOCK = threading.RLock()
# One strong Shared adapter per loaded image/device, not per model. It keeps its device's Owned adapters by owner
# epoch, weakly (`_owners`); actual entries/handles/consuming patchers retain their adapters.
_RESOURCE_DOMAINS = {}


_log = qfe.logger(__name__)   # console-safe (#738)

LORA_HELP_URL = "https://www.quantfunc.com/docs/lora-convert"


def warn_lora_not_loaded():
    """The ONE line a LoRA that could not be loaded for a QuantFunc model prints, whichever family and path (user
    2026-09-30 「lora无法加载的时候让用户自己看 …」): a pointer to the page, instead of leaving the user with per-key lines."""
    _log.warning(f"[qf_native] LoRA could not be loaded for this QuantFunc model. Please read {LORA_HELP_URL}")


def _in_comfy_lora_load(frame):
    """True when ComfyUI's own LoRA load (comfy.sd.load_lora_for_models, which LoraLoader, LoraLoaderModelOnly and other
    LoRA nodes call) is `frame` or one of its callers (a method wrapper can sit between it and add_patches)."""
    while frame is not None:
        if frame.f_code.co_name == "load_lora_for_models" and frame.f_globals.get("__name__") == "comfy.sd":
            return True
        frame = frame.f_back
    return False

# Native queries never wait (quantfunc.h). The engine answers QUANTFUNC_RESOURCE_BUSY whenever another thread holds
# the allocator's or the target's lock at that instant, which is ordinary while native work runs. MEASURED (issue
# #704, 远程-linux-c): an LTX-2.5 run died before step 2 on one such answer, 1 run in 4 under card contention.
# Identity survives BUSY: the engine fills device, owner_epoch and capabilities before its reader runs. Values
# (lifecycle phase, bytes) do not.
_IDENTITY_STATES = (qfe.QUANTFUNC_RESOURCE_READY, qfe.QUANTFUNC_RESOURCE_BUSY, qfe.QUANTFUNC_RESOURCE_UNKNOWN)
# A value read that answers BUSY is re-read until this deadline, then refused. BUSY lasts one native critical
# section; the bound is far above that and only stops a lock that never frees.
_NATIVE_BUSY_DEADLINE_S = 2.0
_NATIVE_BUSY_FIRST_BACKOFF_S = 0.001
_NATIVE_BUSY_MAX_BACKOFF_S = 0.05
_MS_PER_S = 1e3                   # seconds -> milliseconds, for the BUSY messages (their ms wording is tested)


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
    re-issue is safe (see partially_unload).

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
                           what, busy_reads, waited * _MS_PER_S, result.state, _NATIVE_BUSY_DEADLINE_S * _MS_PER_S)
            return result
        busy_reads += 1
        if waited >= _NATIVE_BUSY_DEADLINE_S:
            raise _NativeStillBusy(f"QuantFunc {what} stayed BUSY for {waited * _MS_PER_S:.0f} ms ({busy_reads} reads, "
                                   f"deadline {_NATIVE_BUSY_DEADLINE_S * _MS_PER_S:.0f} ms): another native thread held "
                                   "the lock the whole time")
        time.sleep(min(backoff, _NATIVE_BUSY_DEADLINE_S - waited))
        backoff = min(backoff * 2, _NATIVE_BUSY_MAX_BACKOFF_S)


def _ready_read(read, what):
    """A native call whose READY answer is used: BUSY is re-issued (bounded); any other non-READY is refused."""
    result = _read_past_busy(read, what)
    if result.state != qfe.QUANTFUNC_RESOURCE_READY:
        raise RuntimeError(f"QuantFunc {what} unavailable (state={result.state})")
    return result


# ── #704: ONE rule for every method ComfyUI's memory manager calls on a loaded model ───────────────────────────────
# Enumerated from the installed ComfyUI 0.37.0 comfy/model_management.py (md5 f626009567972a4872f36f18a1671485);
# test_704_every_comfy_facing_override_declares_a_busy_policy re-derives the called set from the installed source:
#   model_size            LoadedModel.model_memory :798, LoadedModel.model_offloaded_memory :804
#   loaded_size           LoadedModel.model_loaded_memory :801, .model_offloaded_memory :804, .model_unload :837
#   partially_unload      LoadedModel.model_unload :838
#   detach                LoadedModel.model_unload :841, load_models_gpu :988 (unpatch_all=False)
#   partially_load        LoadedModel.model_use_more_vram :848 <- model_load :820 <- load_models_gpu :1033
#   model_patches_models  load_models_gpu :954
#   is_dynamic            free_memory :914, load_models_gpu :964 :996 :1036 (no resource read: always False)
#   get_additional_models ModelPatcher.get_nested_additional_models (model_patcher.py:1405) <- unload_model_and_clones
#                         :2130 and sampler_helpers._prepare_sampling :193, both for a prompt's own model
#   loaded_ram_size       load_models_gpu :1036, only when model.is_dynamic()
#   partially_unload_ram  free_model_pins :682 (models_for_pin_eviction :667) and reset_cast_buffers :1483, only when
#                         model.is_dynamic(); no QuantFunc patcher is dynamic or overrides them (ComfyUI's base answers 0:
#                         native backing is the canonical resource adapters', never the logical patcher's host RAM)
# free_memory (:893) sizes EVERY loaded model (:902-907) before it decides anything. It runs on every load_models_gpu
# (:1001, :1010), including prompts that never touch a QuantFunc model, and from unload_all_models (:2121), which
# main.py:390 (POST /free), execution.py:644 (the OOM handler) and execution.py:837 call with no failure channel. So a
# sizing or unload method never lets a persistent BUSY escape: it answers by its policy. A load method runs only for
# the model a prompt asked for, and that prompt reports the failure, so it refuses honestly. A method that makes no
# resource read cannot meet a BUSY; it is declared "no resource read" so the completeness test still accounts for it.
_COMFY_BUSY_POLICY = {}


def _comfy_facing(policy):
    """Declare a ComfyUI-facing method and its persistent-BUSY policy: a function that answers in its place (called
    with the adapter, the _NativeStillBusy, then the method's arguments), or "refuses" / "no resource read"."""
    def declare(method):
        _COMFY_BUSY_POLICY[method.__qualname__] = getattr(policy, "__name__", policy)
        if not callable(policy):
            return method

        @functools.wraps(method)
        def guarded(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except _NativeStillBusy as busy:
                return policy(self, busy, *args, **kwargs)
        return guarded
    return declare


def _busy_loaded_size(adapter, busy, *_args, **_kwargs):
    """The last READY residency, stale by whatever the engine paged since; ComfyUI still reads the device's real free
    memory (get_free_memory) for every decision that matters. None ever observed means no READY residency was ever
    read for this view (ComfyUI reads it right after every load, load_models_gpu :1035), so it was never loaded: 0.
    Under-counting is also ComfyUI 0.37.0's evict-MORE side, never its OOM side: model_unload (:837) then skips
    the partial path and fully detaches, free_memory (:905) orders this model first, and model_memory_required (:808)
    frees its full size when it is requested; over-counting would make ComfyUI free nothing for it. Do not flip this."""
    seen = adapter._last_ready_resident
    if seen is None:
        _log.warning("[qf_native] %s; loaded_size answers 0: no READY residency was ever observed, so this view was "
                     "never loaded", busy)
        return 0
    _log.warning("[qf_native] %s; loaded_size answers the last READY residency, %d B (stale)", busy, seen)
    return seen


def _busy_model_size(adapter, busy, *_args, **_kwargs):
    """An Owned view's size is its Prepared capacity, fixed at prepare time; a Shared view's is its residency."""
    if adapter._capacity_bytes is None:
        return _busy_loaded_size(adapter, busy)
    _log.warning("[qf_native] %s; model_size answers the Prepared capacity, %d B (fixed at prepare time)", busy,
                 adapter._capacity_bytes)
    return int(adapter._capacity_bytes)


def _busy_zero_freed(adapter, busy, *_args, **_kwargs):
    """A release vouches only for what a READY answer reports: 0, and ComfyUI falls back to its own full detach."""
    _log.warning("[qf_native] %s; reporting 0 freed, Comfy falls back to a full detach", busy)
    return 0


def _busy_stop_eviction(adapter, busy, unpatch_all=True):
    """The eviction stops where it is (nothing more is released), like release_all's non-READY branch, and ComfyUI's
    record is dropped as usual: the next run faults back what it needs."""
    _log.warning("[qf_native] %s; full eviction stopped", busy)
    return comfy.model_patcher.ModelPatcher.detach(adapter, unpatch_all=unpatch_all)


def _resource_domain(lib, device, *, prepare=False):
    """The device's canonical Shared adapter, created on first use. prepare=True also tells the engine a host framework
    shares the device (enroll_host: another library's failed allocation takes the engine's reclaimable memory; nothing
    is granted), before an Owned identity or any backing exists."""
    key = (qfe.library_identity(lib), int(device))
    with _CANONICAL_LOCK:
        shared = _RESOURCE_DOMAINS.get(key)
    if shared is None:
        # Native view creation may enter the driver.  Do it outside the global
        # registry lock, then publish one winner; a racing spare view is closed
        # only after the registry lock has been released.
        candidate = QFNativeResourcePatcher(qfe.NativeResource.shared(lib, device))
        candidate._shared_adapter = candidate
        candidate._owners = weakref.WeakValueDictionary()
        with _CANONICAL_LOCK:
            shared = _RESOURCE_DOMAINS.setdefault(key, candidate)
        if shared is not candidate:
            candidate._resource.close()
    if prepare:
        shared._resource.enroll_host()
    return shared


def _domain_loaded_size(adapter):
    """One coherent native snapshot of all Shared + Owned backing on the adapter's device."""
    return _ready_read(adapter._shared_adapter._resource.query_domain_residency, "domain residency").resident_bytes


def canonical_resource_adapters(engine):
    resource = getattr(engine, "resource", None)
    if resource is None:
        raise RuntimeError("QuantFunc engine has no retained native owner identity")
    snapshot = _query_identity(resource)
    if snapshot.owner_epoch == 0:
        raise RuntimeError("QuantFunc model requires an Owned resource, not Shared")
    shared = _resource_domain(resource._lib, snapshot.device)
    with _CANONICAL_LOCK:
        owner = shared._owners.get(snapshot.owner_epoch)
        if owner is None:
            owner = QFNativeResourcePatcher(resource)
            owner._shared_adapter = shared
            shared._owners[snapshot.owner_epoch] = owner
        # Retained by actual identity, not a primary/shadow wrapper or host weakref.
        engine._qf_resource_adapters = (owner, shared)
        return engine._qf_resource_adapters


class QFPreparedEntry:
    """Configured cold identity plus a host-bound materialization chokepoint."""
    def __init__(self, lib, device, *, create_params=None, api_key=None):
        self.lib = lib
        self.create_params = create_params
        # The loader-field API key create_params carry (None: the key of QUANTFUNC_API_KEY / config.json). The engine
        # creates with exactly the configured inputs, so the pipeline starts on this key (QFEngineHandle.applied_api_key).
        self.api_key = api_key
        self.capacity_bytes = None
        self.cache_key = None
        self._cache_usable = True
        self._materializers = weakref.WeakSet()
        self._materialize_lock = threading.RLock()
        _resource_domain(lib, device, prepare=True)  # host enrollment of the device, BEFORE Owned/backing.
        self.resource = qfe.NativeResource.prepare(lib, device)
        try:
            if create_params is None:
                raise qfe.NativeContractUnavailable(
                    "QuantFunc Prepared capacity requires the complete create descriptor")
            self.resource.configure(create_params)
            # A target still Creating answers BUSY for its whole create; past the deadline that is a refusal.
            capacity = _ready_read(self.resource.query_capacity, "resource capacity")
            self.capacity_bytes = int(capacity.required_persistent_bytes)
            owner, shared = canonical_resource_adapters(self)
            self._owner_adapter, self._shared_adapter = owner, shared
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

        Candidate construction registers the Owned adapter under its device's
        Shared adapter before cache publication. A loser is removed by its
        exact owner epoch before its view closes, so no lookup can hand out a
        closed adapter.
        """
        owner = self._owner_adapter
        with self._materialize_lock:
            if not self._cache_usable:
                return
            self._cache_usable = False
            self._materializers.clear()
        with _CANONICAL_LOCK:
            owners = owner._shared_adapter._owners
            if owners.get(owner._owner_epoch) is owner:
                owners.pop(owner._owner_epoch, None)
        owner._closed_identity = True
        self.resource.close()

    def retire_materialized(self):
        """Remove an old Attached/Closed identity from future create lookup.

        Keep its native view and canonical Owner adapter alive: retained
        aliases may still report/reclaim residual backing after pipeline
        teardown.  A same-recipe retry must prepare a new owner epoch.
        """
        with self._materialize_lock:
            self._cache_usable = False
            self._materializers.clear()

    def bind_materializer(self, engine, cache_key):
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
        with self._materialize_lock:
            return self._cache_usable and engine in self._materializers

    def materialize(self, preferred=None):
        # Single flight per identity. Fixed order: prepared entry -> engine cache. The factory may take the global
        # engine cache lock only for short lookup/publication sections; no cache holder may take this entry's lock.
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


@qfe.console_safe_methods   # an exception leaving it is console-safe (#738)
class QFNativeResourcePatcher(comfy.model_patcher.ModelPatcher):
    """ComfyUI's ModelPatcher for one native resource view: an Owned view per model identity, a Shared view per
    device (design #751 §6.1). The caller owns one canonical adapter per resource and must retain it while
    registered: ComfyUI holds patchers weakly. It creates no native model except a Prepared identity's cold create.
    - is_dynamic: False. Our pages are not aimdo VBARs, so load_models_gpu frees memory for our load.
    - model_size: the Prepared capacity (a Shared view: its residency). loaded_size: the engine's resident bytes.
    - partially_load: materializes a Prepared identity; it sets no budget.
    - partially_unload(n): release_eligible(n), the engine's walk over this owner's reclaimable classes; returns freed.
    - detach: release_all (requires CAP_RELEASE_ALL). A plain zero snapshot never authorizes deregistration.
    """
    def __init__(self, resource):
        # The adapter's own identity, fixed for the life of its native view; every later check must see exactly it.
        identity = _query_identity(resource)
        load_device = torch.device("cuda", identity.device)
        model = torch.nn.Module()
        model.device = load_device
        super().__init__(model, load_device, torch.device("cpu"))
        self._resource = resource
        self._closed_identity = False
        self._owner_epoch = identity.owner_epoch  # 0 for the device's Shared view
        self._shared_adapter = None
        self._prepared_entry = None
        self._capacity_bytes = None
        self._last_ready_resident = None  # ComfyUI-facing sizing answers it on a persistent BUSY

    def _identity(self, *, allow_closed=False):
        return _query_identity(self._resource, owner_epoch=self._owner_epoch,
                               device=self.load_device.index, allow_closed=allow_closed)

    @_comfy_facing("no resource read")
    def is_dynamic(self):
        # Our pages are not aimdo VBARs: ComfyUI must free memory for our load (load_models_gpu, for_dynamic=False).
        return False

    def preflight_host_load(self):
        """Read-only fail-before-pop validation used during dependency expansion.

        The check deliberately performs no release, create, or engine-cache operation.
        """
        identity = self._identity()
        lifecycle = _ready_read(self._resource.lifecycle, "host preflight lifecycle")
        if lifecycle.phase == qfe.QUANTFUNC_RESOURCE_PHASE_CLOSED:
            raise RuntimeError("QuantFunc host preflight rejected a Closed resource identity")
        if self._prepared_entry is not None and not self._prepared_entry._cache_usable:
            raise qfe.NativeContractUnavailable(
                "QuantFunc host preflight rejected a retired create identity")
        if identity.owner_epoch and lifecycle.phase == qfe.QUANTFUNC_RESOURCE_PHASE_PREPARED:
            if (self._prepared_entry is None or self._capacity_bytes is None or
                    self._capacity_bytes <= 0 or not self._prepared_entry._cache_usable):
                raise qfe.NativeContractUnavailable(
                    "QuantFunc host preflight lacks a usable Prepared capacity contract")
        _domain_loaded_size(self)

    @_comfy_facing("refuses")
    def model_patches_models(self):
        # Direct load_models_gpu([resource_adapter]) must preflight too.  Do not
        # return self as an additional model or alter the dependency graph.
        self.preflight_host_load()
        return super().model_patches_models()

    def bind_prepared(self, entry, capacity_bytes):
        capacity = int(capacity_bytes)
        if self._prepared_entry is not None and self._prepared_entry is not entry:
            raise RuntimeError("QuantFunc owner adapter was rebound to another prepared identity")
        if self._capacity_bytes is not None and self._capacity_bytes != capacity:
            raise RuntimeError("QuantFunc owner capacity changed for one native identity")
        self._prepared_entry, self._capacity_bytes = entry, capacity

    def _resident_bytes(self):
        """This view's READY residency (strict: past the BUSY deadline it raises), remembered for ComfyUI's sizing."""
        self._last_ready_resident = _ready_read(self._resource.residency, "resource residency").resident_bytes
        return self._last_ready_resident

    @_comfy_facing(_busy_loaded_size)
    def loaded_size(self):
        return self._resident_bytes()

    def require_load_contract(self):
        lifecycle = _ready_read(self._resource.lifecycle, "resource lifecycle")
        self._closed_identity = lifecycle.phase == qfe.QUANTFUNC_RESOURCE_PHASE_CLOSED
        if self._closed_identity:
            raise RuntimeError("QuantFunc Closed resource identity is not reloadable")
        if lifecycle.phase == qfe.QUANTFUNC_RESOURCE_PHASE_PREPARED:
            if self._prepared_entry is None or self._capacity_bytes is None:
                raise qfe.NativeContractUnavailable(
                    "QuantFunc Prepared resource is not bound to the common cold-load adapter")

    @_comfy_facing(_busy_model_size)
    def model_size(self):
        # Sizing only, never a load check: free_memory sizes EVERY listed model on every load_models_gpu, for prompts
        # that never touch this one, and a retired owner stays listed until gc collects it (the host-RAM sweep Closes it
        # while Comfy still lists it), so refusing here killed the next model's load. A LOAD still refuses a Closed
        # identity (partially_load, preflight_host_load). A Closed owner keeps its Prepared capacity as its size: in
        # ComfyUI 0.37 that only orders free_memory's evictions (largest offloaded first), which puts the dead owner first.
        if self._capacity_bytes is not None:
            return int(self._capacity_bytes)
        # Shared has no Prepared model capacity. Its already-resident bytes
        # remain a zero-deficit dependency in Comfy's ledger.
        return self._resident_bytes()

    @_comfy_facing("refuses")
    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        """Materialize a Prepared identity (its one cold create); a loaded one has nothing to load. The engine places
        and streams its own pages, so ComfyUI's weights budget sets nothing here. Returns the domain residency added."""
        if device_to != self.load_device:
            raise ValueError("a native resource cannot migrate to another device")
        if extra_memory < 0:
            # Comfy's negative budget means "shrink by this much, then run" (load_models_gpu computes
            # max(0, free - minimum_memory_required, ...) - loaded_memory); a stock patcher unloads that much too.
            qfe.info(f"[qf_native] Comfy's budget is negative: shrinking {int(-extra_memory) >> 20} MB", flush=True)
            self.partially_unload(device_to, -extra_memory, force_patch_weights=force_patch_weights)
        self.require_load_contract()
        before_domain = _domain_loaded_size(self)
        if self._prepared_entry is not None:
            # A non-READY answer carries no phase; reading None as "not Prepared" skipped materialize().
            lifecycle = _ready_read(self._resource.lifecycle, "admission lifecycle")
            if lifecycle.phase == qfe.QUANTFUNC_RESOURCE_PHASE_PREPARED:
                self._prepared_entry.materialize()
        return max(0, _domain_loaded_size(self) - before_domain)

    def _closed_now(self):
        """Is this identity Closed? Closed is final, so a True answer is kept: the retire marks what it Closes, which
        needs no read. Otherwise ONE lifecycle read (BUSY re-issued to the deadline): only a READY Closed answer counts.
        A BUSY past the deadline or an UNKNOWN answer is not a Closed answer, so the caller's release still runs."""
        if self._closed_identity or not self._owner_epoch:
            return self._closed_identity
        try:
            lifecycle = _read_past_busy(self._resource.lifecycle, "resource lifecycle")
        except _NativeStillBusy:
            return False
        self._closed_identity = (lifecycle.state == qfe.QUANTFUNC_RESOURCE_READY and
                                 lifecycle.phase == qfe.QUANTFUNC_RESOURCE_PHASE_CLOSED)
        return self._closed_identity

    @_comfy_facing(_busy_zero_freed)
    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        want = max(0, min(int(memory_to_free), (1 << 64) - 1))
        if not want:
            return 0
        if self._closed_now():
            # A Closed identity has nothing eligible (native answers CLOSED): free nothing, and Comfy falls back to the
            # full detach, whose release_all is the old owner's final cleanup.
            return 0
        # The engine walks this owner's reclaimable classes (quantfunc_resource_release_eligible). A BUSY release may
        # already have freed eligible backing it does not report, so the retry can free more than `want`: a
        # reload-time cost only. `freed` is only what a READY answer reports, never an estimate; it errs low and
        # Comfy then falls back to its own full detach.
        return _ready_read(lambda: self._resource.release_eligible(want), "resource release").freed_bytes

    def clone(self, disable_dynamic=False, model_override=None, force_deepcopy=False):
        if model_override is not None or force_deepcopy:
            raise ValueError("a native resource identity cannot be copied or replaced")
        return self

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        raise ValueError("a native resource has no patchable model weights")

    def add_object_patch(self, name, obj):
        raise ValueError("a native resource identity cannot be patched")

    @_comfy_facing(_busy_stop_eviction)
    def detach(self, unpatch_all=True):
        if unpatch_all:
            identity = self._identity(allow_closed=True)
            if not identity.capabilities & qfe.QUANTFUNC_RESOURCE_CAP_RELEASE_ALL:
                raise qfe.NativeContractUnavailable("QuantFunc native resource full eviction is unsupported "
                                                    "without CAP_RELEASE_ALL; keep the host record")
            # The engine walks everything this owner holds (quantfunc_resource_release_all). A Closed target is
            # forwarded too: native permits final exact-old-owner cleanup, which never makes its identity reusable.
            result = self._resource.release_all()
            if result.state != qfe.QUANTFUNC_RESOURCE_READY:
                # Comfy's unload hook has no failure channel: unload_all_models() runs unguarded on the
                # prompt worker (its OOM handler and POST /free), so an error escaping here ends the
                # server's only worker and every later prompt hangs. Native Busy is also the ORDINARY
                # answer for a live pipeline: release_all certifies ZERO backing, and the non-paged
                # floor never reaches zero. Everything eligible is already released; what remains is
                # physically visible to Comfy, and the next run faults back what it needs.
                qfe.say(f"[qf_native] full eviction left native backing (state={result.state})", flush=True)
        # Keep the view and canonical object alive across host deregistration.
        return super().detach(unpatch_all=unpatch_all)


@qfe.console_safe_methods   # an exception leaving it is console-safe (#738)
class QFModelPatcher(comfy.model_patcher.ModelPatcher):
    """Logical MODEL; canonical dependencies own native bytes, this patcher owns Torch bytes."""
    def add_patches(self, patches, *args, **kwargs):
        # ComfyUI's own LoRA load patches torch weights, which a QuantFunc model never computes with, so such a LoRA
        # never reaches it; ComfyUI prints its per-key lines, and this says what to do, once per load. At model strength 0
        # the LoRA is meant for the text encoder only: nothing to say.
        strength = args[0] if args else kwargs.get("strength_patch", 1.0)
        if strength and _in_comfy_lora_load(getattr(inspect.currentframe(), "f_back", None)):
            warn_lora_not_loaded()
        return super().add_patches(patches, *args, **kwargs)

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

    @_comfy_facing("refuses")
    def model_patches_models(self):
        # Direct load_models_gpu uses this path, not nested additional_models.
        return list(dict.fromkeys(super().model_patches_models() + self._native_dependencies()))

    @_comfy_facing("refuses")
    def get_additional_models(self):
        # Also refresh before official nested traversal and after clone/LoRA.
        self._native_dependencies()
        return super().get_additional_models()

    @_comfy_facing("no resource read")
    def model_size(self):
        return super().model_size()

    @_comfy_facing("no resource read")
    def loaded_size(self):
        return super().loaded_size()

    @_comfy_facing("refuses")
    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        if device_to != self.load_device:
            raise ValueError("a QuantFunc MODEL cannot migrate its native resources")
        for adapter in self._native_dependencies():
            adapter.require_load_contract()
        # Native bytes are loaded by the canonical dependencies. The official
        # base implementation owns only this model's ordinary Torch parameters.
        return super().partially_load(device_to, extra_memory,
                                      force_patch_weights=force_patch_weights)

    @_comfy_facing("no resource read")
    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        return super().partially_unload(device_to, memory_to_free,
                                        force_patch_weights=force_patch_weights)


    # ---- comfy lifecycle / VRAM+RAM manager integration ----------------------------------
    def adopt_comfy_state_from(self, src):
        """Return a patcher that has THIS patcher's model but SRC's comfy-level state.

        WHY (CR regression, measured class): a downstream QuantFuncNativeLoRA re-creates the
        PATCHER and MODEL (the pipeline is shared: its set goes on in place), so it hands back a
        DIFFERENT patcher+model. Anything an upstream node had
        applied with add_object_patch — most importantly ModelSamplingSD3 /
        ModelSamplingMiniMaxH3's shift patch — lives on the OLD patcher and would be silently
        dropped, leaving the checkpoint default with no error (the exact silent-shift class the
        H3 guard was added for). Transplanting via comfy's OWN clone(model_override=...) carries
        object_patches, model_options, callbacks, wrappers, attachments, injections and hooks —
        one authoritative list, so it cannot drift as comfy's clone() grows fields.
        """
        override = (self.model, (self.backup, self.backup_buffers,
                                 self.object_patches_backup, self.pinned))
        # The loader's session dials (attention backend, quality, caches, H3's audio / partial-denoise opt-ins) live on
        # the MODEL the rebuild replaced, not in comfy's state: carry them, or the LoRA'd model silently runs defaults.
        self.model.adopt_session_dials_from(src.model)
        return src.clone(model_override=override)

    @_comfy_facing("no resource read")
    def detach(self, unpatch_all=True):
        """Detach logical patches only; canonical dependencies own native eviction."""
        # Only logical patches belong here. Native adapters retain their own
        # full-detach refusal; detaching a clone never releases their views.
        return super().detach(unpatch_all=unpatch_all)
