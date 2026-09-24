"""qf_native — native ComfyUI loader for the QuantFunc engine (wan svdq).

ONE loader node creates a QuantFunc pipeline and returns a native comfy MODEL (a QFModelPatcher)
that native KSampler / KSamplerAdvanced drive via quantfunc_denoise_step. CLIP + VAE stay NATIVE
comfy nodes (maximize comfy-ecosystem compatibility).
"""
import os
import json
import logging
import weakref
import threading

# GUARDED imports (mirror the real plugin __init__.py) — a broken comfy-internals import must NOT
# take down node registration on a ComfyUI upgrade; degrade to zero nodes + a loud warning.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
try:
    import comfy.model_management
    import comfy.supported_models
    from . import qf_engine as qfe
    from . import qf_modelpatcher as qfmp
    from .qf_modelpatcher import QFModelPatcher
    _IMPORT_OK = True
except Exception as _exc:  # noqa: BLE001 — never break registration; report loudly
    logging.warning("[qf_native] disabled — a required import failed (ComfyUI API drift?): %r", _exc)
    _IMPORT_OK = False


# folder_paths gives the model-file listing surface (INT8-Fast-aligned: FILES, not dirs).
try:
    import folder_paths as _folder_paths
except Exception as _fp_exc:  # noqa: BLE001 — never break registration
    _folder_paths = None
    logging.warning("[qf_native] folder_paths unavailable: %r", _fp_exc)

_NO_LORA_HINT = "(no LoRA in models/loras)"


_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
_NO_CFG_HINT = "(no official model config shipped)"


def _model_config_choices(family=None):
    """The OFFICIAL model-config presets shipped with the plugin — one subdir of configs/ per
    model, each carrying a qf_native.json manifest (family routing + shape) beside the arch/VAE
    config JSONs. The dropdown lists the DIRECTORY NAMES, so adding a preset FOR AN EXISTING
    family = drop in a config dir, data-only. A preset for a NEW family is NOT reachable by data
    alone (R6, mutation-proven): it additionally needs its family seam module in _FAMILY_MODULES
    AND a loader node class exposing that family's dropdown — the per-family node design trades
    the old any-family node for family-scoped UX, so the code surface for a new family is the
    node class + module entry, stated here so nobody trusts the old data-only claim. `family`
    filters the list for the
    PER-FAMILY loader nodes (user 2026-08-21 pivot: one loader node per model family), so a wan
    preset can never appear in the LTX node's dropdown; a manifest whose family key is unreadable
    is simply not listed for a filtered call (the unfiltered call still shows it, and load()
    fail-louds on it)."""
    try:
        out = []
        for d in sorted(os.listdir(_CONFIGS_DIR)):
            mp = os.path.join(_CONFIGS_DIR, d, "qf_native.json")
            if not os.path.isfile(mp):
                continue
            if family is not None:
                try:
                    with open(mp, "r", encoding="utf-8") as fh:
                        if str(json.load(fh).get("family")) != family:
                            continue
                except Exception:  # noqa: BLE001 — unreadable manifest: hide from filtered lists
                    continue
            out.append(d)
        return out or [_NO_CFG_HINT]
    except Exception:  # noqa: BLE001
        return [_NO_CFG_HINT]


def _load_model_config(name):
    """Resolve + read a preset's manifest. The name is a widget value (workflow-serializable =
    untrusted): it must be exactly one of the listed preset dirs — no separators, no traversal."""
    if name == _NO_CFG_HINT or os.sep in name or "/" in name or "\\" in name or name in ("", ".", ".."):
        raise RuntimeError(f"qf_native: invalid model_config {name!r} — pick one of the shipped "
                           f"presets ({_model_config_choices()}).")
    bundle = os.path.join(_CONFIGS_DIR, name)
    mf = os.path.join(bundle, "qf_native.json")
    if not os.path.isfile(mf):
        raise RuntimeError(f"qf_native: model_config {name!r} has no qf_native.json manifest "
                           f"(shipped presets: {_model_config_choices()}).")
    try:
        manifest = json.load(open(mf))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"qf_native: model_config {name!r} manifest unreadable: {exc}") from exc
    if not isinstance(manifest, dict) or not manifest.get("family"):
        raise RuntimeError(f"qf_native: model_config {name!r} manifest must declare a family.")
    return bundle, manifest


_NO_XFM_HINT = "(no .safetensors in models/diffusion_models)"
_XFM_NONE = "(none)"


def _transformer_choices():
    """The .safetensors FILES under comfy's models/diffusion_models — the SAME surface the
    reference INT8-Fast UNetLoaderINTW8A8 lists (folder_paths.get_filename_list). The user picks a
    transformer weight FILE directly (transformer1 = the / high-noise expert; transformer2 = the
    optional low-noise expert for wan A14B). NOT a package DIRECTORY — the engine's denoise-only
    create builds ONLY the transformer from this file (CLIP + VAE stay native comfy nodes)."""
    if _folder_paths is None:
        return [_NO_XFM_HINT]
    try:
        files = [f for f in _folder_paths.get_filename_list("diffusion_models")
                 if f.lower().endswith(".safetensors")]
    except Exception:  # noqa: BLE001
        files = []
    # Exclude files that live INSIDE a model-PACKAGE directory (any ancestor dir carrying
    # model_index.json): comfy's recursive listing otherwise surfaces package INTERNALS —
    # k9b/vae/diffusion_pytorch_model.safetensors and friends — which are never valid
    # transformer picks and drowned the dropdown (user-reported).
    try:
        roots = [r for r in _folder_paths.get_folder_paths("diffusion_models") if os.path.isdir(r)]
    except Exception:  # noqa: BLE001
        roots = []

    def _inside_package(rel):
        parts = rel.replace("\\", "/").split("/")[:-1]
        for root in roots:
            cur = root
            for seg in parts:
                cur = os.path.join(cur, seg)
                if os.path.isfile(os.path.join(cur, "model_index.json")):
                    return True
        return False

    files = [f for f in files if "/" not in f.replace("\\", "/") or not _inside_package(f)]
    return sorted(files) or [_NO_XFM_HINT]


def _resolve_transformer(name):
    """Resolve a listed transformer filename to its full path via comfy's own containment
    (get_full_path_or_raise confines it to the diffusion_models roots — the untrusted-widget
    #vuln guard, same as _resolve_lora)."""
    if _folder_paths is None or name in ("", _NO_XFM_HINT, _XFM_NONE):
        raise RuntimeError("qf_native: no transformer weight selected — put the svdq .safetensors "
                           "under ComfyUI/models/diffusion_models/ and pick it in transformer1.")
    return _folder_paths.get_full_path_or_raise("diffusion_models", name)


# (the te_file/audio_vae dropdown helpers died with the aux widgets — the comfy seam
#  owns ONLY the denoise stage; listing workflow-stage surfaces here was full-pipeline
#  thinking. RED LINE, user 2026-08-22: ctypes full-pipeline logic must NEVER enter the
#  comfy seam.)


def _lora_choices():
    if _folder_paths is None:
        return [_NO_LORA_HINT]
    try:
        return list(_folder_paths.get_filename_list("loras")) or [_NO_LORA_HINT]
    except Exception:  # noqa: BLE001
        return [_NO_LORA_HINT]


def _resolve_lora(name):
    if _folder_paths is None or name == _NO_LORA_HINT:
        raise RuntimeError("qf_native: no LoRA available — put .safetensors files in models/loras/")
    return _folder_paths.get_full_path_or_raise("loras", name)


# Auth resolves from the PROCESS ENVIRONMENT + the package-bundled keyfile ONLY — never a node
# widget. `keyfile` was previously a workflow STRING widget: same class as `so_path` (a shared
# workflow.json could point it at an arbitrary file the ComfyUI server then opens/parses). A
# workflow.json cannot set an env var, so the dev override QF_NATIVE_KEYFILE is safe; the shipped
# default is the package-bundled bin/<platform>/config.json.
def _resolve_keyfile():
    override = os.environ.get(qfe._ENV_KEYFILE_OVERRIDE, "").strip()
    if override:
        return override
    pkg = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg, "bin", qfe._BIN_SUBDIR, qfe._KEYFILE_BASENAME)


def _read_auth():
    # The API key is preferentially an env var (QUANTFUNC_API_KEY) — the harness/plugin convention;
    # only if absent do we read the bundled/overridden keyfile.
    key = os.environ.get("QUANTFUNC_API_KEY", "") or os.environ.get("QF_API_KEY", "")
    surl = os.environ.get("QF_SERVER_URL", "https://service.quantfunc.com")
    keyfile = _resolve_keyfile()
    if not key and keyfile and os.path.exists(keyfile):
        try:
            c = json.load(open(keyfile))
            key, surl = c.get("api_key", ""), c.get("server_url", surl)
        except Exception:  # noqa: BLE001
            pass
    return key, surl


# ── Flow-matching schedule shift (the CHECKPOINT'S OWN, not comfy's WAN default) ────────────────────
# comfy's WAN21_I2V model_sampling defaults to shift=8.0, but a QuantFunc wan svdq checkpoint carries
# its OWN scheduler (diffusers scheduler_config.json `flow_shift`, e.g. 5.0 for the a14b i2v distilled
# ckpt). The native seam has comfy's KSampler compute the sigma schedule that drives the engine forward,
# so that schedule MUST be the checkpoint's — a 4-step DISTILLED model is extremely schedule-sensitive:
# on comfy's shift-8.0 sigmas [1.0,0.96,0.889,0.727] instead of the checkpoint's shift-5.0 sigmas
# [1.0,0.938,0.833,0.625], the mid-range denoising is under-resolved → mangled fine structure
# (hands/hair) vs the engine's OWN pipeline (which uses flow_shift=5.0 UniPC). Read it and apply it.
def _comfy_device_index():
    """The CUDA index of the GPU ComfyUI computes on (0 when it is not a CUDA device or comfy is unavailable)."""
    try:
        dev = comfy.model_management.get_torch_device()
        return int(dev.index) if getattr(dev, "type", "") == "cuda" and dev.index is not None else 0
    except Exception:
        return 0


def _read_flow_shift(model_dir):
    """The checkpoint's flow-matching shift from its diffusers scheduler config
    (`flow_shift`, falling back to `shift`). Returns None (→ keep comfy's default) if the
    config is absent/unreadable, so a model dir without a scheduler config is byte-unchanged."""
    for rel in ("scheduler/scheduler_config.json", "scheduler_config.json"):
        p = os.path.join(model_dir, rel)
        if not os.path.exists(p):
            continue
        try:
            c = json.load(open(p))
        except Exception:  # noqa: BLE001
            return None
        v = c.get("flow_shift", c.get("shift"))
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _apply_checkpoint_flow_shift(model, model_dir):
    """Set comfy `model_sampling`'s shift to the checkpoint's own flow_shift so the stock KSampler
    drives the engine forward on the model's NATIVE schedule (the same the engine's own pipeline uses).
    No-op (comfy default kept, logged) when the checkpoint declares no flow_shift.

    ROOT CAUSE of the "native worse than the direct-engine path" report: comfy's WAN default shift=8.0
    vs this distilled checkpoint's flow_shift=5.0 mis-schedules the 4-step denoise (mid-range under-
    resolved → mangled hands / hair halo). VERIFIED fixed — run 20260808-234500-native-optionc-final-
    wan-svdq, legA_f0030: the hand resolved to individual fingers vs the pre-fix flesh smear."""
    ms = getattr(model, "model_sampling", None)
    shift = _read_flow_shift(model_dir)
    if shift is None or ms is None or not hasattr(ms, "set_parameters"):
        logging.warning("[qf_native] flow_shift: no scheduler_config flow_shift under %s — keeping "
                        "comfy's WAN default shift (schedule may not match the checkpoint)", model_dir)
        return
    ms.set_parameters(shift=float(shift))   # multiplier (1000) preserved; recomputes the sigma table
    print(f"[qf_native] flow_shift: set comfy model_sampling shift={float(shift)} from the checkpoint's "
          f"scheduler_config (comfy's WAN default 8.0 mis-schedules this distilled ckpt)", flush=True)


# Pipeline CACHE: reuse the created engine handle for a repeated config → a re-executed workflow does
# NOT leak a fresh pipeline. Keyed by the create-determining inputs.
#  • VRAM residency is scheduled by ComfyUI. Cache selection must not independently
#    evict other live handles; their resource adapters execute host reclaim requests.
#  • HOST RAM (bounded by the set of LIVE patchers): `_sweep_dead_pipelines` DESTROYS a handle only
#    once its QFWanModel has been garbage-collected (comfy dropped the patcher) — a dead model cannot
#    be use-after-freed, so destroy is safe there. Without this the unload_vram-only design leaks a
#    multi-GB CPU backup per distinct config forever (a resolution/model sweep). `_PIPELINE_MODELS`
#    holds weakrefs to ALL live consumers: QFLazyEngine acquisition pins plus family models.
_PIPELINE_CACHE = {}     # ckey -> QFEngineHandle
_PREPARED_CACHE = weakref.WeakValueDictionary()  # cold identities retained by lazy consumers
_ENGINE_IDENTITY_LOCK = threading.RLock()
_PIPELINE_MODELS = {}    # ckey -> [weakref.ref(consumer), ...] — ALL live consumers of that config's
#                          handle. A LIST, not one ref: two loader nodes on the same package share
#                          ONE handle, and a single last-load-wins ref made the FIRST model invisible
#                          to liveness (measured: a sibling's release/sweep could destroy the shared
#                          handle under a still-live patcher → NULL pipeline at its next denoise +
#                          a ledger that kept reporting the destroyed backup).
# CONCURRENCY CONTRACT: the three cache dictionaries above are protected by short
# `_ENGINE_IDENTITY_LOCK` lookup/publication/removal sections.  The lock is NEVER
# held while constructing QFPreparedEntry, checking native grants, creating or
# destroying a pipeline, or closing a losing prepared candidate.  Cold creation
# is single-flight under domain.transaction_lock -> entry._materialize_lock; that
# path may briefly take this cache lock, and no inverse cache-lock -> domain-lock
# edge is permitted.


def _unpin_pipeline_consumer(ckey, consumer):
    """Rollback one acquisition-created pin without touching model bindings."""
    with _ENGINE_IDENTITY_LOCK:
        refs = []
        for ref in _PIPELINE_MODELS.get(ckey, []):
            current = ref()
            if current is not None and current is not consumer:
                refs.append(ref)
        if refs:
            _PIPELINE_MODELS[ckey] = refs
        else:
            _PIPELINE_MODELS.pop(ckey, None)


def _pin_pipeline_consumer_locked(ckey):
    """Pin the current QFLazyEngine in the same transaction as cache selection.

    Caller holds `_ENGINE_IDENTITY_LOCK`.  Returning a live handle without this
    context is forbidden: otherwise retire can remove/destroy it before the
    family factory reaches its later model-binding loop.
    """
    acquisition = qfmp._current_engine_cache_acquisition()
    if acquisition is None:
        raise qfe.NativeContractUnavailable(
            "QuantFunc engine cache lookup requires a QFLazyEngine consumer acquisition")
    consumer = acquisition.consumer
    refs = [ref for ref in _PIPELINE_MODELS.get(ckey, []) if ref() is not None]
    if not any(ref() is consumer for ref in refs):
        refs.append(weakref.ref(consumer))
        _PIPELINE_MODELS[ckey] = refs
        acquisition.register_pin(lambda: _unpin_pipeline_consumer(ckey, consumer))
    elif refs:
        _PIPELINE_MODELS[ckey] = refs
    return consumer


def _bind_pipeline_model(ckey, model):
    """Register `model` as a live consumer of ckey's cached handle (called by every family builder
    after constructing its model). Prunes dead refs so the list tracks the true live set."""
    with _ENGINE_IDENTITY_LOCK:
        # Cold preparation also binds consumers but owns no pipeline. Keep that
        # metadata bounded by live consumers, not every historical cold recipe.
        for old_key in list(_PIPELINE_MODELS):
            if old_key not in _PIPELINE_CACHE:
                _live_pipeline_models(old_key)
        refs = [r for r in _PIPELINE_MODELS.get(ckey, []) if r() is not None]
        if not any(r() is model for r in refs):   # re-materialize of a live model must not accumulate
            refs.append(weakref.ref(model))
        _PIPELINE_MODELS[ckey] = refs


def _live_pipeline_models(ckey):
    """The models still alive on ckey's handle (prunes dead refs in place)."""
    with _ENGINE_IDENTITY_LOCK:
        refs = [r for r in _PIPELINE_MODELS.get(ckey, []) if r() is not None]
        if refs:
            _PIPELINE_MODELS[ckey] = refs
        else:
            _PIPELINE_MODELS.pop(ckey, None)
        return [r() for r in refs]


def _retire_handle(ckey, eng, requester=None, *, keep_binding=False, reason=""):
    """THE ONLY place in this plugin allowed to call .destroy() on an engine handle — enforced
    as a real AST property by the dispatch suite (attribute calls AND getattr-alias forms,
    file set derived from the package; reviewer-F: the previous per-site hand-written gates +
    a substring scan were the hollow-lint shape, and the unconditional-destroy class had
    already recurred twice).

    GATE (the one liveness discriminator, ex-_may_release_handle): retire is REFUSED while any
    live FOREIGN consumer is bound to ckey — a model that is neither `requester` itself nor
    sharing its `_qf` wrapper (pair-mates share ONE QFLazyEngine instance and re-materialize
    coherently; a SEPARATE load's sibling would be left holding a destroyed handle = the
    measured friendly-fire class). requester=None means ANY live consumer refuses (the sweep's
    only-all-dead semantics). `requester` accepts either the QFLazyEngine wrapper or a bound
    model (its `_qf` is used).

    ON GRANT: pop the cache entry, pop the binding list unless keep_binding (release() keeps it
    — the SAME ckey's next handle re-binds its surviving consumers; the LoRA reconcile re-creates
    under a NEW ckey and the sweep retires dead entries, so both drop it), then destroy."""
    retired_entry = None
    with _ENGINE_IDENTITY_LOCK:
        if _PIPELINE_CACHE.get(ckey) is not eng:
            return False
        req_qf = getattr(requester, "_qf", requester)   # model -> wrapper; wrapper -> itself; None -> None
        foreign = []
        for consumer in _live_pipeline_models(ckey):
            consumer_qf = getattr(consumer, "_qf", consumer)
            if requester is not None and consumer_qf is req_qf:
                continue
            foreign.append(consumer)
        if foreign:
            print(f"[qf_native] retire refused ({reason or 'unspecified'}): {len(foreign)} live "
                  "foreign consumer(s) still bound to this cache entry", flush=True)
            return False
        _PIPELINE_CACHE.pop(ckey, None)
        prepared_key = (qfe.library_identity(eng.lib), ckey)
        candidate = _PREPARED_CACHE.get(prepared_key)
        if candidate is not None and candidate.resource is eng.resource:
            _PREPARED_CACHE.pop(prepared_key, None)
            # Publish non-creatable in the same identity transaction as cache
            # removal. retire_materialized performs the domain-locked cleanup.
            candidate._cache_usable = False
            retired_entry = candidate
        if not keep_binding:
            _PIPELINE_MODELS.pop(ckey, None)
    if retired_entry is not None:
        retired_entry.retire_materialized()
    # Native teardown may be slow and must never run under the cache lock.
    try:
        eng.destroy()   # idempotent (pipeline→None); closes any session first
    except Exception:  # noqa: BLE001 — retire must never mask the caller's continuation
        pass
    return True


def _sweep_dead_pipelines(keep_key):
    """Reclaim HOST RAM: destroy any cached handle whose model comfy has GC'd (no live reference → no
    UAF). A still-live model is kept; ComfyUI decides its VRAM residency. Never
    touches keep_key or a handle not yet bound to a model (its load() may still be in flight)."""
    with _ENGINE_IDENTITY_LOCK:
        keys = list(_PIPELINE_CACHE.keys())
    for k in keys:
        if k == keep_key:
            continue
        with _ENGINE_IDENTITY_LOCK:
            if not _PIPELINE_MODELS.get(k):
                continue   # UNBOUND (load may be in flight) → never touch; only ever-bound entries sweep
            eng = _PIPELINE_CACHE.get(k)
        # requester=None ⇒ _retire_handle refuses while ANY consumer is live (only-all-dead sweeps)
        if eng is not None and _retire_handle(k, eng, None, reason="host-RAM sweep"):
            print("[qf_native] host-RAM sweep: destroyed a cached pipeline whose QFWanModel was GC'd "
                  "(comfy dropped its patcher) — freed its CPU backup", flush=True)


def _engine_recipe(model_dir, create_cfg=None, device_idx=0):
    """Resolve immutable create inputs without entering a cache critical section."""
    lib = qfe.load_lib()
    ckey = (qfe.resolve_so_path(), model_dir, "svdq", int(device_idx),
            json.dumps(create_cfg or {}, sort_keys=True))
    return lib, ckey, (qfe.library_identity(lib), ckey), create_cfg


def _prepare_params(model_dir, create_cfg, device_idx):
    """Build retained create params only after both engine caches miss."""
    key, surl = _read_auth()
    cfg = dict(create_cfg or {})
    if key:
        cfg["api_key"] = key
        cfg["server_url"] = surl
    return qfe.make_create_params(model_dir=model_dir, model_backend="svdq",
                                  device_idx=int(device_idx), config_json=cfg or None)


def _get_or_prepare_entry(lib, ckey, prepared_key, model_dir, create_cfg, device_idx):
    """Return the cached handle/entry, constructing a publish candidate outside the cache lock."""
    with _ENGINE_IDENTITY_LOCK:
        eng = _PIPELINE_CACHE.get(ckey)
        if eng is not None and eng.pipeline is not None:
            if qfe.library_identity(eng.lib) != qfe.library_identity(lib):
                raise RuntimeError("QuantFunc cache entry belongs to a different loaded native image")
            _pin_pipeline_consumer_locked(ckey)
            return eng
        entry = _PREPARED_CACHE.get(prepared_key)
        if entry is not None and not entry._cache_usable:
            _PREPARED_CACHE.pop(prepared_key, None)
            entry = None
    if entry is not None:
        return entry

    params = _prepare_params(model_dir, create_cfg, device_idx)
    candidate = qfmp.QFPreparedEntry(lib, int(params.device_idx), create_params=params)
    winner = None
    mismatch = False
    with _ENGINE_IDENTITY_LOCK:
        eng = _PIPELINE_CACHE.get(ckey)
        if eng is not None and eng.pipeline is not None:
            if qfe.library_identity(eng.lib) != qfe.library_identity(lib):
                mismatch = True
            else:
                _pin_pipeline_consumer_locked(ckey)
                winner = eng
        else:
            winner = _PREPARED_CACHE.get(prepared_key)
            if winner is not None and not winner._cache_usable:
                _PREPARED_CACHE.pop(prepared_key, None)
                winner = None
            if winner is None:
                _PREPARED_CACHE[prepared_key] = candidate
                winner, candidate = candidate, None
    if candidate is not None:
        candidate.retire_unpublished()
    if mismatch:
        raise RuntimeError("QuantFunc cache entry belongs to a different loaded native image")
    return winner


def _materialize_engine(entry, lib, ckey):
    """Single-flight cold create under domain -> prepared-entry lock order."""
    acquisition = qfmp._current_engine_cache_acquisition()
    if acquisition is None:
        raise qfe.NativeContractUnavailable(
            "QuantFunc engine materialization requires a QFLazyEngine consumer acquisition")
    # Validate weak-reference support before native creation; publication itself
    # must not discover an unpinnable consumer after allocating a live handle.
    weakref.ref(acquisition.consumer)
    with qfmp._domain_transaction(entry._owner_adapter):
        with entry._materialize_lock:
            if not entry._cache_usable:
                raise qfe.NativeContractUnavailable(
                    "QuantFunc prepared identity was retired before materialization")
            with _ENGINE_IDENTITY_LOCK:
                eng = _PIPELINE_CACHE.get(ckey)
                if eng is not None and eng.pipeline is not None:
                    if qfe.library_identity(eng.lib) != qfe.library_identity(lib):
                        raise RuntimeError("QuantFunc cache entry belongs to a different loaded native image")
                    _pin_pipeline_consumer_locked(ckey)
                    return eng, ckey
            # Creation is reachable only after the canonical Comfy dependency has
            # installed finite Owned/Shared/device grants. A direct factory caller has
            # no host-admitted capacity and therefore cannot bypass the common seam.
            if acquisition.consumer not in entry._materializers:
                raise qfe.NativeContractUnavailable(
                    "QuantFunc cold creation requires a live ComfyUI materializer binding")
            qfmp._require_engine_host_grants(entry)
            _sweep_dead_pipelines(ckey)
            # The exact configured-Prepared native capacity Comfy admitted is
            # retained on this identity. It is VRAM authority, not CPU-backup
            # footprint, and is never re-estimated from paths at create time.
            eng = qfe.QFEngineHandle.create(lib, create_params=entry.create_params,
                                           capacity_bytes=int(entry.capacity_bytes),
                                           prepared_resource=entry.resource)
            eng._qf_resource_adapters = entry._qf_resource_adapters
            eng._qf_resource_adapters[0]._prepared = False
            with _ENGINE_IDENTITY_LOCK:
                _PIPELINE_CACHE[ckey] = eng
                _pin_pipeline_consumer_locked(ckey)
            return eng, ckey


def _get_engine(model_dir, create_cfg=None, device_idx=0):
    """Create (or reuse) the engine for a model PACKAGE dir. The native library path is
    resolved internally (resolve_so_path — NEVER a node input, #vuln). Create is MINIMAL:
    a PREQUANT svdq package carries its own layout/precision in its metadata; anything
    supplied on top competes with it and loses. The transformer weights live INSIDE the
    package (engine loads model_dir/transformer[_2]/ directly — no path override).
    create_cfg carries the per-family create keys (e.g. wan text_precision) + the
    declarative lora stack from chained QuantFuncNativeLoRA nodes."""
    # [session-knobs] the session-knob-≠-create-key guard is sealed INSIDE
    # qf_engine.create_pipeline (the real quantfunc_create boundary — construction-enforced,
    # unbypassable by a future direct caller), not duplicated here (one truth source).
    # Cache lookup/publication is atomic; prepare/grant/create use the independent
    # domain -> prepared-entry order and never run under the global cache lock.
    lib, ckey, prepared_key, create_cfg = _engine_recipe(model_dir, create_cfg, device_idx)
    entry = _get_or_prepare_entry(lib, ckey, prepared_key, model_dir, create_cfg, device_idx)
    if isinstance(entry, qfe.QFEngineHandle):
        return entry, ckey
    if qfe.FACTORY_PREPARE_ONLY.get():
        return entry, ckey
    return _materialize_engine(entry, lib, ckey)


if _IMPORT_OK:
    # ── family REGISTRY: assembled from the per-family modules. Family LOGIC lives in the family
    #    modules; what remains here is the shared NODE SURFACE — transformer1/transformer2 FILE
    #    dropdowns + model_type (the LoRA node's target is WIRE-derived,
    #    no combo — chaining on the high output acts on the high expert). That
    #    surface is family-neutral as long as a new family fits the "1-2 transformer files +
    #    shipped config bundle" shape; one needing a NEW input must extend INPUT_TYPES here, so
    #    "add a family = one module + one _FAMILY_MODULES line" holds for the common case, not
    #    unconditionally. ──
    # Each module owns ONE model family end to end (its comfy model subclass, its builder, and its
    # detection rule) and exposes exactly three names: FAMILY / matches() / register(deps).
    # Adding a family = write qf_<name>_modelpatcher.py + add it to _FAMILY_MODULES. No edit to the
    # node, the dispatch or the detection lives here, so families cannot bleed into each other.
    _FAMILY_MODULES = ("qf_wan_modelpatcher", "qf_ltx_modelpatcher", "qf_h3_modelpatcher", "qf_krea2_modelpatcher",
                       "qf_qwenimage21_modelpatcher")
    _FAMILY_BUILDERS = {}     # family key -> build(...)
    _FAMILY_MATCHERS = []     # (family key, matches) in registration order. The FILE-based
    #    loader takes model_type EXPLICITLY (a bare .safetensors has no model_index to
    #    detect from), so matches() is no longer called at load — it stays exported as each
    #    family's transcription of its ENGINE detector predicate (parity documentation) and
    #    this registry is the model_type choices' name/order source.

    def _register_families():
        """Import each family module and register its builder. A family whose module fails to
        import is SKIPPED WITH A LOUD WARNING (its models then say 'no registered native seam')
        — one broken family must not take the whole plugin's registration down."""
        import importlib
        deps = {"get_engine": _get_engine, "bind_pipeline_model": _bind_pipeline_model,
                "retire_handle": _retire_handle,
                "apply_checkpoint_flow_shift": _apply_checkpoint_flow_shift}
        for mod_name in _FAMILY_MODULES:
            try:
                mod = importlib.import_module("." + mod_name, __name__)
                _FAMILY_BUILDERS[mod.FAMILY] = mod.register(deps)
                _FAMILY_MATCHERS.append((mod.FAMILY, mod.matches))
            except Exception as exc:  # noqa: BLE001 — never break plugin import
                logging.warning("[qf_native] family module %s not registered: %r", mod_name, exc)

    _register_families()



    def _run_family_load(expect_family, transformer1, model_config,
                         transformer2, sparse_opts=None):
        """The SHARED loader core behind the per-family nodes (user 2026-08-21 pivot). All
        validation is preserved verbatim from the original single-node load(); the per-family
        nodes add only (a) a family-filtered preset dropdown and (b) this family guard —
        defense-in-depth against a preset dir whose manifest family drifted after the dropdown
        rendered. Returns the family builder's result AS-IS (wan: a (model_high, model_low)
        patcher pair; single-expert families: one patcher)."""
        bundle_dir, manifest = _load_model_config(model_config)
        family = str(manifest["family"])
        if family != expect_family:
            raise RuntimeError(
                f"qf_native: model_config '{model_config}' declares family '{family}' but this "
                f"loader node is the '{expect_family}' loader — pick a '{expect_family}' preset "
                f"(the dropdown lists only those; this mismatch means the preset dir changed "
                f"after the UI rendered).")
        builder = _FAMILY_BUILDERS.get(family)
        if builder is None:
            raise RuntimeError(
                f"qf_native: model_config '{model_config}' routes to family '{family}', which "
                f"has no registered native seam in this install (available: "
                f"{sorted(_FAMILY_BUILDERS)}). An import of the seam module probably failed "
                f"at startup — check the log for a [qf_native] warning.")
        xfm1 = _resolve_transformer(transformer1)
        xfm2 = None if transformer2 in (_XFM_NONE, "", None) else _resolve_transformer(transformer2)
        if bool(manifest.get("dual_expert")) and xfm2 is None:
            raise RuntimeError(
                f"qf_native: model_config '{model_config}' is DUAL-expert — transformer1 = the "
                f"HIGH-noise expert AND transformer2 = the LOW-noise expert are both required "
                f"(the export ships them as a *-high-* / *-low-* pair).")
        # file_hints validation (DATA-driven; the manifest names what its transformers look
        # like). DESIGN BOUNDARY, recorded deliberately: comfy combo values must be the REAL
        # relative filenames (they resolve through folder_paths) and INPUT_TYPES is rendered
        # statically — so the dropdown CANNOT dynamically filter by the sibling model_config
        # widget without frontend JS. The correspondence contract is therefore enforced HERE,
        # fail-loud at load, with the expected patterns named.
        from .qf_file_hints import name_matches_hints  # one predicate, pinned by tests/published_names_test.py
        hints = manifest.get("file_hints") or {}
        for arm, val in (("transformer1", transformer1),
                         ("transformer2", None if xfm2 is None else transformer2)):
            pats = hints.get(arm) or []
            if val is None or not pats:
                continue
            if not name_matches_hints(val, pats):
                raise RuntimeError(
                    f"qf_native: {arm}={val!r} does not look like a '{model_config}' "
                    f"{arm} weight (expected a name matching {pats}). Pick the file the "
                    f"preset names — see the model_config tooltip — or choose the preset "
                    f"matching this file.")
        if not manifest.get("dual_expert") and xfm2 is not None:
            raise RuntimeError(
                f"qf_native: model_config '{model_config}' is single-transformer — leave "
                f"transformer2 = \"(none)\" (a second expert here would be silently ignored "
                f"at best; refused instead).")
        # NO aux resolution (user 2026-08-22 "引擎层不应该依赖这个"): the loader depends on
        # nothing but the transformer file(s) themselves. The retired [aux-auto] manifest
        # fallback layer (te/audio_vae/connectors conventional-filename resolution) served
        # only the transitional split exports; an incomplete single-file export is refused
        # loud by the family builder instead of being silently completed.
        # [#659] ALL runtime dials — the two cache thresholds AND the sparse dial —
        # are SESSION knobs (never create keys): armed
        # POST-construction on the returned patcher(s)' model via the mixin
        # setters, injected into every denoise_begin by residency_opts().
        # Deliberately NOT builder/create kwargs → they can never enter
        # create_cfg/ckey (NO pipeline rebuild on any widget change — user
        # 2026-08-28 "调整sparse要重建pipeline完全没必要" + "调整block/step cache
        # 能复用pipeline"). OFF values (0.0 / 1.0) omit the begin keys entirely →
        # the engine paths are byte-identical.
        _kw = {}
        if sparse_opts:
            # create-LEVEL keys only (attention backend / quant toggles). The
            # sparse dial itself is a SESSION knob (below) and never rides here.
            _kw["sparse_opts"] = sparse_opts
        out = builder(transformer1_path=xfm1, transformer2_path=xfm2,
                      bundle_dir=bundle_dir, **_kw)
        # [cache/sparse surface REMOVED, user 2026-08-29 「移除所有loader的cache以及
        # 稀疏入口 整体默认不生效」] The per-model set_step_cache/set_block_cache/
        # set_sparse arming that lived here is GONE with the loader widgets — the
        # mixin defaults (0.0/0.0/1.0) already omit every begin key, so the engine
        # paths are byte-identical without any call. Re-enabling is a plugin-side
        # revert of THIS commit (widgets + this arming loop); the engine session
        # dials are untouched and stay available.
        return out

    # [cache surface RE-ENABLED, user 2026-08-31 「step cache 以及 fbcache 的开关重新开启」]
    # The step_cache (EasyCache) + block_cache (First-Block Cache = "fbcache") widgets
    # restored to the video loaders (wan/LTX/H3) — the two loader widgets + their arming
    # loop that the 2026-08-29 removal dropped. SPARSE is deliberately NOT re-enabled
    # (the user named only the two caches). Both are RUNTIME SESSION knobs (mixin
    # set_step_cache/set_block_cache → residency_opts begin keys; 0.0 = OFF = byte-identical,
    # no create key, no pipeline rebuild on a widget change). Engine EasyCache/FBCache
    # (lighting_step_cache.h) is untouched on main.
    _STEP_CACHE_INPUT = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                         "tooltip": "Speed-up that reuses earlier work while the result is barely changing. 0 (default) = off. "
                                    "Higher values are faster but can move the result away from the full render; "
                                    "0.02–0.05 is typical. Takes effect on the next run."})
    _BLOCK_CACHE_INPUT = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                          "tooltip": "A second speed-up that reuses work inside each pass when little changes. 0 (default) = "
                                     "off. 0.05–0.12 is typical; higher is faster but can lose detail. Can be combined "
                                     "with step_cache. Takes effect on the next run."})
    # [quality — user 2026-09-24] ONE speed/quality choice on the four QuantFunc loaders (H3, LTX-2.5, Krea2, Qwen-Image-2.1). It
    # replaces the quality_enhance switch (the engine's video_enhance). The option names are the user's; what each one does is
    # ENGINE law, one table per model family (the engine's QualityLaw) — the loader sends only the name as the session's
    # `quality` and the engine resolves it per run from that run's own step count. A quality change never rebuilds or reloads
    # (user 「更换quality的时候不应该重建pipeline」): the create config never depends on it, so all four options share one cached
    # pipeline; the engine prepares its fast mode in place at the first fast step of a run and drops it at a run that uses none.
    # The two fast options exist only where the ENGINE says they can run on this GPU (quantfunc_quality_fast_available) — the
    # plugin keeps no GPU list; everywhere else a two-way choice.
    # User-facing text states only the speed / quality trade (user 「介绍上不要透露技术细节」).
    _QUALITY_FAST_OPTIONS = ["super_fast", "fast", "balance", "best_quality"]
    _QUALITY_BASE_OPTIONS = ["balance", "best_quality"]
    _QUALITY_DEFAULT = "balance"   # every GPU; the fast options are opt-in (user 「默认balance」)
    _QUALITY_TOOLTIP_FAST = ("Speed or quality. super_fast: the fastest; details can differ from best_quality. fast: faster, and "
                             "closer to best_quality. balance (default): can be a little faster than best_quality, with almost "
                             "the same result. best_quality: the highest quality, and the slowest. On turbo models, super_fast and fast "
                             "can produce a different variation of the same seed.")
    _QUALITY_TOOLTIP_BASE = ("Speed or quality. balance (default): can be faster, with almost the same result as best_quality. "
                             "best_quality: the highest quality, and slower.")
    # Saved workflows (migration): they carry the retired quality_enhance switch — API-format prompts under its NAME (declared
    # hidden, in ComfyUI's (type, options) input form, so ComfyUI hands it to load() and can validate it when it is linked; an
    # undeclared key would be dropped silently), UI workflows as a boolean in this widget's POSITION (widget values are stored by
    # position). Old ON (full quality) → best_quality, old OFF (the speed default) → balance.
    _QUALITY_LEGACY_HIDDEN = {"quality_enhance": ("BOOLEAN", {})}
    _quality_engine_cache = {}

    def _quality_engine(idx=None):
        """(speaks, fast) for a GPU (default: the one ComfyUI computes on — a load passes the device IT captured), asked of the
        ENGINE once per device: speaks = the engine takes the
        `quality` session key (it exports quantfunc_quality_fast_available — the key and the query ship together); fast =
        super_fast / fast can take effect on that GPU (the engine's own arming rule; the plugin keeps no GPU list). An older
        engine, no CUDA device, or any failure → (False, False): the two-way form, never a silently inert option."""
        idx = _comfy_device_index() if idx is None else int(idx)
        if idx not in _quality_engine_cache:
            try:
                lib = qfe.load_lib()
                speaks = hasattr(lib, "quantfunc_quality_fast_available")
                _quality_engine_cache[idx] = (speaks, bool(speaks and lib.quantfunc_quality_fast_available(idx) == 1))
            except Exception:
                _quality_engine_cache[idx] = (False, False)
        return _quality_engine_cache[idx]

    def _quality_fast_for_file(transformer, model_config, idx=None):
        """super_fast / fast also depend on the model FILE (a checkpoint with no layer the fast mode speeds up, or one stored in
        a form it cannot use): the engine answers for this file on this GPU (quantfunc_quality_fast_available_file). An engine
        without that query keeps its GPU answer; any other answer than yes — including an error — runs balance."""
        try:
            key, surl = _read_auth()
            ans = qfe.quality_fast_available_file(qfe.load_lib(), _load_model_config(model_config)[0],
                                                  _resolve_transformer(transformer),
                                                  _comfy_device_index() if idx is None else idx, surl, key)
        except Exception:
            return False
        return ans is None or ans == 1

    def _quality_fast_tier(idx=None):
        return _quality_engine(idx)[1]

    def _loaded_device_index(patcher):
        """The CUDA index the family load captured (its load_device) — the ONE device capture of a load drives the quality
        decision too, never a second read of ComfyUI's device."""
        dev = getattr(patcher, "load_device", None)
        return int(dev.index) if getattr(dev, "type", "") == "cuda" and dev.index is not None else 0

    def _quality_input():
        fast = _quality_fast_tier()
        return (list(_QUALITY_FAST_OPTIONS if fast else _QUALITY_BASE_OPTIONS),
                {"default": _QUALITY_DEFAULT, "tooltip": _QUALITY_TOOLTIP_FAST if fast else _QUALITY_TOOLTIP_BASE})

    def _validate_quality(quality):
        """The loaders' VALIDATE_INPUTS body (it replaces ComfyUI's own list check for `quality`): any of the four names on every
        GPU (a workflow saved on a GPU with the fast options still opens), a boolean (the retired switch, positional), or None —
        ComfyUI's value for a LINKED input (resolved at run time, where _resolve_quality refuses anything else)."""
        if quality is None or isinstance(quality, bool) or quality in _QUALITY_FAST_OPTIONS:
            return True
        return f"quality must be one of {', '.join(_QUALITY_FAST_OPTIONS)} (got {quality!r})"

    def _resolve_quality(quality=None, quality_enhance=None, transformer=None, model_config=None, device_idx=None):
        """The node's quality → the mode this run uses. An explicit quality wins; the retired switch (a boolean in quality's
        position, or by name when quality is absent) maps old ON → best_quality, OFF → balance; nothing given → the default. A
        fast option that cannot run — on this GPU, or (given the loader's transformer + model_config) for this model file —
        runs balance, with one console line, whatever the reason."""
        if isinstance(quality, bool):
            q = "best_quality" if quality else "balance"
        elif quality is not None:
            q = str(quality)
        elif quality_enhance is not None:
            q = "best_quality" if bool(quality_enhance) else "balance"
        else:
            q = _QUALITY_DEFAULT
        if q not in _QUALITY_FAST_OPTIONS:
            raise ValueError(_validate_quality(q))
        if q in ("super_fast", "fast") and not (_quality_fast_tier(device_idx) and
                                                (transformer is None or _quality_fast_for_file(transformer, model_config, device_idx))):
            print(f"[QuantFunc] '{q}' is not available here; using balance.", flush=True)   # this GPU / file, or an older engine
            q = "balance"
        return q

    def _apply_quality(_mm, q, device_idx=None):
        """Hand the resolved quality to the model's sessions. An engine that predates `quality` refuses that key, so it gets the
        retired switch instead (best_quality = video_enhance ON, else OFF — the engine's speed policy); the fast options never
        reach it — with no query symbol _resolve_quality already ran them as balance."""
        if _quality_engine(device_idx)[0]:
            _mm.set_quality(q)   # mandatory + unguarded, like the retired switch: a patcher without it is a wiring error
        else:
            _mm.set_video_enhance(q == "best_quality")


    # [audio_enhance switch, user 2026-09-13] H3-only. OFF (default) = byte-identical to no knob.
    # ON = after the normal (video) denoise, run EXTRA AUDIO-ONLY sub-steps so video_steps +
    # extra-audio steps total 16 — refining audio against the finished video at fixed per-sub-step
    # cost (no video recompute; video latent/frames byte-identical). No-op when the video already
    # runs >= 16 steps. Drives engine extra_audio_steps; the exact top-up is computed at session
    # begin from the sampler's step count (qf_h3_modelpatcher.set_audio_enhance / _begin).
    _AUDIO_ENHANCE_INPUT = ("BOOLEAN", {"default": False,
                     "tooltip": "MiniMax-H3 only. On: adds a short extra pass after the video is finished that makes the "
                                "sound clearer and sharper; the video itself is unchanged. Off (default): no extra time. "
                                "Most useful with fast turbo settings; it has no effect at long, high-quality settings."})

    # [sol-tau dial 2026-08-31] the ONE user-facing Sol-Attn knob (user "就一个就好"). Applies to
    # the flash/sage backends — the engine's applySolTauDial engages the Sol-Attn keep-ratio per
    # block regardless of attention_backend (it is NOT tied to the removed qfa choice). Same
    # runtime-session-knob class as step_cache/sparse: re-sent each run, no rebuild; 1.0 default is
    # omitted (older engines refuse unknown keys loud) and the engine resets an absent key to 1.0.
    _SOL_TAU_INPUT = ("FLOAT", {"default": 1.0, "min": 0.02, "max": 1.0, "step": 0.01,
                     "tooltip": "Attention speed-up. 1.0 (default) turns it off. Lower values are faster and give up "
                                "some quality: 0.15–0.2 is a good start, below 0.1 check the result carefully, 0.3–0.5 "
                                "keeps more quality. Takes effect on the next run."})

    def _arm_session_caches(_mm, step_cache, block_cache):
        """Arm the EasyCache (step) + FBCache (block) session knobs on a loaded model.
        Both are runtime session knobs (never create keys); 0.0 = OFF = byte-identical.
        The mixin setters + residency_opts threading are on qf_modelpatcher.py (intact
        through the 2026-08-29 removal — only the loader widgets + this arming were dropped)."""
        if _mm is None:
            return
        if hasattr(_mm, "set_step_cache"):
            _mm.set_step_cache(float(step_cache or 0.0))
        if hasattr(_mm, "set_block_cache"):
            _mm.set_block_cache(float(block_cache or 0.0))
    _COMMON_LIMITS = (
        "Notes: ControlNet is not supported. Interrupting stops at the next safe point of the run. The loader checks "
        "that ComfyUI and this plugin use a matching CUDA version; if they do not, it stops with a message that "
        "explains the fix.")

    # [attention backend selector, user 2026-08-27] one user-facing dropdown per loader.
    # SM-GATED: SM80+ offers the full set; SM75 (Turing) has NO int8-QK sage and NO
    # flash_attn build, so only fp16_native is valid there. The widget VALUE is a
    # display name; _attn_backend_to_engine maps it to the engine's comp_opts string
    # ("fp16_native" -> "native"). "auto" = the engine's per-SM resolution (default), and
    # is passed through so the user's choice is always the single source of truth.
    # [qfa REMOVED as a user-facing choice — user 2026-09-13] the qfa (int8-QK + Hadamard /
    # sol) backend is no longer offered in the dropdown. "auto" is unaffected (the engine may
    # still resolve to qfa internally per-SM); only the explicit user choice is gone.
    _ATTN_BACKEND_SM80PLUS = ["auto", "flash", "sage", "fp16_native"]
    _ATTN_BACKEND_SM75 = ["fp16_native"]

    def _attn_backend_choices():
        choices = _ATTN_BACKEND_SM80PLUS
        try:
            import torch
            if torch.cuda.is_available():
                maj, _min = torch.cuda.get_device_capability(0)
                if maj < 8:  # SM75 Turing (sm_7x): no sage int8-QK, no flash_attn
                    choices = _ATTN_BACKEND_SM75
        except Exception:  # noqa: BLE001 — no torch/CUDA at import → assume modern; engine validates
            pass
        return choices

    def _attn_backend_input(default="auto"):
        choices = _attn_backend_choices()
        # per-loader default; falls back to the first valid choice when the requested
        # default isn't offered on this SM (e.g. H3 wants 'flash' but SM75 has no flash).
        d = default if default in choices else choices[0]
        return (choices, {"default": d,
                "tooltip": "How attention is computed. auto picks the best choice for your GPU. flash is the most "
                           "robust; sage can be faster on newer GPUs; fp16_native works on every GPU. Takes effect "
                           "on the next run."})

    def _attn_backend_to_engine(v):
        # widget display name -> engine comp_opts attention_backend string
        return "native" if v == "fp16_native" else (v or "auto")

    def _merge_attn_backend(create_opts, attention_backend):
        """Fold the user's backend choice into the create-time opts dict. An EXPLICIT
        choice (anything but 'auto') always WINS; 'auto' leaves create_opts untouched
        (engine per-model default). Returns the dict (possibly newly created) or None
        when nothing to pass."""
        eng = _attn_backend_to_engine(attention_backend)
        if eng == "auto":
            return create_opts or None
        d = dict(create_opts or {})
        d["attention_backend"] = eng
        return d

    class QuantFuncWanLoader:
        """Wan 2.x loader — DUAL MODEL outputs (high-noise, low-noise) over ONE shared engine,
        mirroring the official two-UNETLoader wan2.2 A14B workflow 1:1: wire model_high to the
        first KSamplerAdvanced stage and model_low to the second, keep CLIP/VAE/latent/video
        nodes stock. Expert selection is engine-side per-step sigma (boundary_ratio from the
        preset), so sub-range stages (start/end_at_step) are fully supported — and even a
        mis-wired stage still computes correctly."""

        @classmethod
        def INPUT_TYPES(cls):
            _xfms = _transformer_choices()
            return {"required": {
                "transformer1": (_xfms, {"tooltip": "The HIGH-noise expert .safetensors under "
                                                    "models/diffusion_models."}),
                "transformer2": (_xfms, {"tooltip": "The LOW-noise expert .safetensors (wan A14B "
                                                    "ships them as a *-high-* / *-low-* pair)."}),
                "model_config": (_model_config_choices(family="wan"),
                                 {"tooltip": "The Wan 2.2 preset that matches the chosen model files."}),
            }, "optional": {
                "attention_backend": _attn_backend_input(),
                "step_cache": _STEP_CACHE_INPUT,
                "block_cache": _BLOCK_CACHE_INPUT,
                "act_scale_g32": ("BOOLEAN", {"default": False,
                    "tooltip": "svdq int4 激活-scale 组宽开关 (#565). OFF=g64 (默认, 与旧版逐字节一致); "
                               "ON=g32 (更细的激活量化组, 实测 -9.1% 激活量化误差, 前向 +~45%, 仅 SM89/86 是真杠杆; "
                               "SM120 上 int4 已用更细的 E0M3 g16, 此开关 no-op). 仅对 svdq int4 生效. "
                               "把 int4 画质往 fp8 靠的实验开关 —— 温和收益, 单靠它通常不足以完全追平 fp8 "
                               "(根因是 int4 激活精度; 干净对齐 fp8 需 a8w4). "
                               "切换此项会重新加载模型 (引擎只在创建模型时读取它)."}),
            }}

        RETURN_TYPES = ("MODEL", "MODEL")
        RETURN_NAMES = ("model_high", "model_low")
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = (
            "QuantFunc Wan loader (svdq, denoise_only): TWO MODEL outputs (high/low-noise expert) "
            "over ONE shared engine — a drop-in for the official wan2.2 A14B dual-UNETLoader "
            "workflow (dual KSamplerAdvanced stages + trimmed step ranges fully supported; the "
            "engine picks the expert per step by sigma). LoRA: chain QuantFuncNativeLoRA on an "
            "output — the wire IS the target (high output ⇒ high expert, low ⇒ low), no target "
            "widget; both wires keep sharing the one engine. "
            + _COMMON_LIMITS)

        def load(self, transformer1, transformer2, model_config,
                 attention_backend="auto", step_cache=0.0, block_cache=0.0, act_scale_g32=False):
            # [#659] the sparse dial is a SESSION knob (never a create key — no rebuild
            # on change); sparse_opts carries CREATE-level keys only.
            # [runtime dial 2026-08-29] attention_backend is a SESSION knob now — NOT a
            # create key (widget change no longer re-keys the loader = no rebuild).
            sparse_opts = {"act_scale_g32": True} if act_scale_g32 else None
            pair = _run_family_load("wan", transformer1, model_config,
                                    transformer2,
                                    sparse_opts=(sparse_opts or None))
            eng_b = _attn_backend_to_engine(attention_backend)
            for _p in pair:
                _mm = getattr(_p, "model", None)   # the session mixin lives on the MODEL
                if _mm is not None and hasattr(_mm, "set_attn_backend"):
                    _mm.set_attn_backend(eng_b)
                _arm_session_caches(_mm, step_cache, block_cache)
            return pair

    class QuantFuncLTXLoader:
        """LTX-2 loader — single MODEL output (single-expert family)."""

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "transformer": (_transformer_choices(),
                                {"tooltip": "The QuantFunc LTX-2 model file in models/diffusion_models."}),
                "model_config": (_model_config_choices(family="ltx2"),
                                 {"tooltip": "The LTX-2 preset that matches the chosen model file."}),
            }, "optional": {
                "attention_backend": _attn_backend_input(),
                "sol_tau": _SOL_TAU_INPUT,
                "quality": _quality_input(),
                "step_cache": _STEP_CACHE_INPUT,
                "block_cache": _BLOCK_CACHE_INPUT,
            }, "hidden": dict(_QUALITY_LEGACY_HIDDEN)}

        @classmethod
        def VALIDATE_INPUTS(cls, quality=None):
            return _validate_quality(quality)

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = ("Loads a QuantFunc LTX-2 model (video with sound) for ComfyUI's standard samplers — use it in place "
                       "of the usual diffusion-model loader. For image-to-video, add the official LTXVImgToVideoInplace node on "
                       "the latent input. " + _COMMON_LIMITS)

        def load(self, transformer, model_config,
                 attention_backend="auto", sol_tau=1.0, quality=None, step_cache=0.0, block_cache=0.0, quality_enhance=None):
            # [aux-auto] NO aux file widgets and NO image socket (user 2026-08-22 "只保留
            # transformer/block/model_config … 只关注latent"): te/audio-vae/connectors
            # resolve from the preset manifest's aux_files inside _run_family_load; i2v is
            # the workflow's own latent conditioning (LTXVImgToVideoInplace), exactly like
            # wan's cond-latent shape.
            _p = _run_family_load("ltx2", transformer, model_config,
                                   None,
                                   sparse_opts=None)
            _dev = _loaded_device_index(_p)
            q = _resolve_quality(quality, quality_enhance, transformer, model_config, _dev)
            _mm = getattr(_p, "model", None)
            if _mm is not None and hasattr(_mm, "set_attn_backend"):
                _mm.set_attn_backend(_attn_backend_to_engine(attention_backend))
            if _mm is not None and hasattr(_mm, "set_sol_tau"):
                _mm.set_sol_tau(sol_tau)
            _apply_quality(_mm, q, _dev)
            _arm_session_caches(_mm, step_cache, block_cache)
            return (_p,)

    class QuantFuncKrea2Loader:
        """Krea-2 Turbo loader (svdq, denoise_only, t2i) — the first IMAGE family on the
        native seam: one MODEL a stock KSampler drives with latents; CLIP (type krea2) +
        VAE + sampler stay comfy-owned (drop-in for the official UNETLoader slot)."""

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "transformer": (_transformer_choices(),
                                {"tooltip": "The QuantFunc Krea-2 Turbo model file in models/diffusion_models."}),
                "model_config": (_model_config_choices(family="krea2"),
                                 {"tooltip": "The Krea-2 Turbo preset that matches the chosen model file."}),
            }, "optional": {
                "attention_backend": _attn_backend_input(),
                "quality": _quality_input(),
            }, "hidden": dict(_QUALITY_LEGACY_HIDDEN)}

        @classmethod
        def VALIDATE_INPUTS(cls, quality=None):
            return _validate_quality(quality)

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = ("Loads a QuantFunc Krea-2 Turbo model (text-to-image) for ComfyUI's standard samplers — use it in "
                       "place of the usual diffusion-model loader. " + _COMMON_LIMITS)

        def load(self, transformer, model_config, attention_backend="auto",
                 quality=None, quality_enhance=None):
            # [runtime dials] backend + quality are SESSION knobs (NOT create keys — a widget change never re-keys the
            # engine = no rebuild).
            _p = _run_family_load("krea2", transformer, model_config,
                                  None,
                                  sparse_opts=None)
            _dev = _loaded_device_index(_p)
            q = _resolve_quality(quality, quality_enhance, transformer, model_config, _dev)
            _mm = getattr(_p, "model", None)
            if _mm is not None and hasattr(_mm, "set_attn_backend"):
                _mm.set_attn_backend(_attn_backend_to_engine(attention_backend))
            _apply_quality(_mm, q, _dev)
            return (_p,)

    class QuantFuncQwenImage21Loader:
        """Qwen-Image-2.1 loader (svdq, denoise_only, text-to-image + image edit): one MODEL a stock
        KSampler drives with latents; CLIP (type qwen_image, TextEncodeQwenImage21) + VAE + sampler stay
        comfy-owned (drop-in for the official UNETLoader slot). Edit = TextEncodeQwenImage21 with a VAE and
        reference images: the references ride every step into the engine (quantfunc_denoise_step_refs)."""

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "transformer": (_transformer_choices(),
                                {"tooltip": "The QuantFunc Qwen-Image-2.1 model file in models/diffusion_models."}),
                "model_config": (_model_config_choices(family="qwenimage21"),
                                 {"tooltip": "The Qwen-Image-2.1 preset that matches the chosen model file."}),
            }, "optional": {
                "attention_backend": _attn_backend_input(),
                "quality": _quality_input(),
            }, "hidden": dict(_QUALITY_LEGACY_HIDDEN)}

        @classmethod
        def VALIDATE_INPUTS(cls, quality=None):
            return _validate_quality(quality)

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = ("Loads a QuantFunc Qwen-Image-2.1 model for ComfyUI's standard samplers: text-to-image, and image "
                       "editing with TextEncodeQwenImage21 reference images (plus its VAE). Transparent images: VAE Decode + "
                       "Save Image keep the transparency. " + _COMMON_LIMITS)

        def load(self, transformer, model_config, attention_backend="auto", quality=None, quality_enhance=None):
            # [runtime dials] backend + quality are SESSION knobs (NOT create keys — a widget change never re-keys the engine =
            # no rebuild), exactly like the Krea2 node.
            _p = _run_family_load("qwenimage21", transformer, model_config,
                                  None,
                                  sparse_opts=None)
            _dev = _loaded_device_index(_p)
            q = _resolve_quality(quality, quality_enhance, transformer, model_config, _dev)
            _mm = getattr(_p, "model", None)
            if _mm is not None and hasattr(_mm, "set_attn_backend"):
                _mm.set_attn_backend(_attn_backend_to_engine(attention_backend))
            _apply_quality(_mm, q, _dev)
            return (_p,)


    class QuantFuncH3Loader:
        """MiniMax-H3 loader — single MODEL output (single-expert AV family)."""

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "transformer": (_transformer_choices(),
                                {"tooltip": "The QuantFunc MiniMax-H3 model file in models/diffusion_models."}),
                "model_config": (_model_config_choices(family="minimax-h3"),
                                 {"tooltip": "The MiniMax-H3 preset that matches the chosen model file."}),
            }, "optional": {
                # [sparse, user 2026-08-25 ONE-number dial; #659 session knob — no rebuild]
                # H3 default = flash: this model's auto resolves to sage2 int8-QK, which is
                # BROKEN on H3's post-qk-RMSNorm γ-outliers at high-res (blank/NaN — measured
                # 928²/S=31538: attn out absmax 0 → step-1 all-NaN → audio avcodec crash +
                # video blur). flash (fp16) is the verified-clean default; user can still pick
                # auto/sage/native. (Wan/LTX → auto is fine → they keep 'auto'.)
                "attention_backend": _attn_backend_input("flash"),
                "sol_tau": _SOL_TAU_INPUT,
                "quality": _quality_input(),
                "audio_enhance": _AUDIO_ENHANCE_INPUT,
                "step_cache": _STEP_CACHE_INPUT,
                "block_cache": _BLOCK_CACHE_INPUT,
                "allow_partial_denoise": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Opt in to split/trimmed sigma schedules for intentional H3 double-sampling workflows.",
                }),
            }, "hidden": dict(_QUALITY_LEGACY_HIDDEN)}

        @classmethod
        def VALIDATE_INPUTS(cls, quality=None):
            return _validate_quality(quality)

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = ("Loads a QuantFunc MiniMax-H3 model (video with sound) for ComfyUI's standard samplers — use it in "
                       "place of the usual diffusion-model loader. " + _COMMON_LIMITS)

        def load(self, transformer, model_config,
                 attention_backend="flash", sol_tau=1.0, quality=None, audio_enhance=False,
                 step_cache=0.0, block_cache=0.0, allow_partial_denoise=False, quality_enhance=None):  # H3: flash default (auto→sage is broken)
            _p = _run_family_load("minimax-h3", transformer, model_config,
                                   None,
                                   sparse_opts=None)
            _dev = _loaded_device_index(_p)
            q = _resolve_quality(quality, quality_enhance, transformer, model_config, _dev)
            _mm = getattr(_p, "model", None)
            if _mm is not None and hasattr(_mm, "set_attn_backend"):
                _mm.set_attn_backend(_attn_backend_to_engine(attention_backend))
            if _mm is not None and hasattr(_mm, "set_sol_tau"):
                _mm.set_sol_tau(sol_tau)
            _apply_quality(_mm, q, _dev)
            if _mm is not None and hasattr(_mm, "set_audio_enhance"):
                _mm.set_audio_enhance(audio_enhance)
            _mm.set_allow_partial_denoise(bool(allow_partial_denoise))
            _arm_session_caches(_mm, step_cache, block_cache)
            return (_p,)

    class QuantFuncNativeLoRA:
        """Sidecar LoRA for the QuantFunc native loader — MODEL in, MODEL out (LoraLoaderModelOnly
        shape). Chain several to stack them.

        Single-expert families (LTX-2.5, H3, Krea-2, Qwen-Image-2.1; user rule 2026-09-24 「换 LoRA 也不重建」):
        the pipeline is created WITHOUT LoRA, so every LoRA set of one model shares it, and the chained set is
        applied in place before each run (one declarative quantfunc_pipeline_update {"lora": [...]}; QFLazyEngine
        runtime_lora). Wan still merges its per-expert union at CREATE time (the engine cannot yet route a
        high/low target at runtime), so a LoRA change there re-creates the pair's pipeline. Either way the create
        is deferred (QFLazyEngine), so a chain of N nodes still builds ONE pipeline.
        Comfy-level patches applied upstream (ModelSampling*, set_model_* …) are TRANSPLANTED onto
        the rebuilt patcher, so this node may sit anywhere in the chain.
        """
        @classmethod
        def INPUT_TYPES(cls):
            # NO target widget (user directive 2026-08-22): the target is derived from WIRING —
            # chaining this node on the wan loader's high output MEANS it acts on the high
            # expert (low likewise); single-expert families derive "all".
            return {"required": {
                "model": ("MODEL",),
                "lora_name": (_lora_choices(),),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }}

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "apply"
        # comfy's own convention, measured: LoraLoader / LoraLoaderModelOnly — the nodes this one
        # is shaped after — use "model/loaders" (nodes.py); bare "loaders" is for file-loading
        # nodes. (ModelSampling* use "model/patch*", so the rule is per-precedent, not universal.)
        CATEGORY = "model/loaders"
        DESCRIPTION = ("Attaches a sidecar LoRA to a QuantFunc native MODEL (wire downstream of a "
                       "QuantFunc loader; chain several to stack). Changing the LoRA keeps the loaded "
                       "model (no reload). On the Wan loader a LoRA change reloads the model.")

        @staticmethod
        def _refuse_foreign_lora_format(path):
            """[one-format policy, user 2026-08-30] the native path adapts ONE
            mainstream format — diffusers/PEFT canonical (<module>.lora_A/B.weight).
            Anything else (kohya sd-scripts / ai-toolkit underscore keys, LyCORIS
            LoHa/LoKr) is refused LOUD with the exact converter command — never
            half-applied. Header-only sniff (8-byte len + json), no tensor reads."""
            import json as _j, struct as _st
            try:
                with open(path, "rb") as f:
                    n = _st.unpack("<Q", f.read(8))[0]
                    if n > 512 * 1024 * 1024:
                        raise ValueError("implausible safetensors header size")
                    keys = [k for k in _j.loads(f.read(n)) if k != "__metadata__"]
            except Exception as e:
                raise RuntimeError(
                    f"QuantFuncNativeLoRA: cannot read '{path}' as safetensors ({e!r})")
            conv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "scripts", "qf_lora_convert.py")
            if any(".hada_" in k or ".lokr_" in k for k in keys):
                raise RuntimeError(
                    "QuantFuncNativeLoRA: this is a LyCORIS (LoHa/LoKr) file — a factored "
                    "decomposition the native path does not consume. Merge/re-export it to a "
                    "standard LoRA first.")
            if any(k.startswith(("lora_unet_", "lora_transformer_", "lora_te_",
                                 "lora_te1_", "lora_te2_")) for k in keys):
                raise RuntimeError(
                    "QuantFuncNativeLoRA: kohya/ai-toolkit-format LoRA detected. The native "
                    "path adapts ONE format (diffusers/PEFT canonical) — convert once with:\n"
                    f"  python3 {conv} --in '{path}' --out '<same-dir>/<name>-diff.safetensors'\n"
                    "then pick the converted file in this node.")
            if not any(".lora_A." in k or ".lora_B." in k or ".lora_down." in k
                       or ".lora_up." in k for k in keys):
                raise RuntimeError(
                    "QuantFuncNativeLoRA: no recognizable LoRA keys (lora_A/lora_B/"
                    "lora_down/lora_up) in this file — not a LoRA, or an unsupported "
                    f"format. If it is a LoRA, convert it: python3 {conv} --in ... --out ...")

        def apply(self, model, lora_name, strength):
            rebuild = qfmp.rebuild_of(model)
            if rebuild is None:
                raise RuntimeError(
                    "QuantFuncNativeLoRA: this MODEL is not a QuantFunc native model — wire it "
                    "downstream of the QuantFunc Native Loader. (For a stock comfy model use the "
                    "built-in LoraLoaderModelOnly instead.)")
            # [R2-generality fix] the ONE-FORMAT refusal applies only to families the
            # engine actually restricts (Krea2/LTX2/H3 — mirror of the engine's E1 arm);
            # Wan keeps native kohya support (WAN_RULES) and marks itself exempt via
            # _qf_kohya_lora_ok on its model class.
            if not getattr(model.model, "_qf_kohya_lora_ok", False):
                self._refuse_foreign_lora_format(_resolve_lora(lora_name))
            stack = qfmp.lora_stack_of(model)
            stack.append({"path": _resolve_lora(lora_name), "scale": float(strength),
                          "target": qfmp.expert_of(model)})   # [wiring-lora] wire-derived side
            rebuilt = rebuild(stack)
            # CR regression fix: carry the UPSTREAM comfy state (ModelSampling* object patches,
            # set_model_* options, callbacks/wrappers/hooks) onto the re-created patcher, so a
            # LoRA node placed after a ModelSampling node cannot silently drop its shift.
            return (rebuilt.adopt_comfy_state_from(model),)

    # merge into (not replace) the mappings — matches the real plugin's multi-file NODE_CLASS_MAPPINGS.update
    NODE_CLASS_MAPPINGS.update({"QuantFuncWanLoader": QuantFuncWanLoader,
                                "QuantFuncLTXLoader": QuantFuncLTXLoader,
                                "QuantFuncH3Loader": QuantFuncH3Loader,
                                "QuantFuncKrea2Loader": QuantFuncKrea2Loader,
                                "QuantFuncQwenImage21Loader": QuantFuncQwenImage21Loader,
                                "QuantFuncNativeLoRA": QuantFuncNativeLoRA})
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "QuantFuncWanLoader": "QuantFunc Wan Loader (high+low)",
        "QuantFuncLTXLoader": "QuantFunc LTX-2 Loader",
        "QuantFuncH3Loader": "QuantFunc MiniMax-H3 Loader",
        "QuantFuncKrea2Loader": "QuantFunc Krea-2 Loader",
        "QuantFuncQwenImage21Loader": "QuantFunc Qwen-Image-2.1 Loader"})

    # AUTOMATION — a mechanism must not depend on someone remembering to run it (CR): run the reject-list
    # completeness scan AT IMPORT so a comfy upgrade that adds a consumable conditioning key emits a loud
    # logging.warning HERE, instead of waiting for a human to run the CLI on every comfy upgrade. Loaded by
    # file path (tests/ is not a package) and fully guarded — a tripwire must never break registration.
    try:
        import importlib.util as _ilu
        _rc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests",
                                "reject_list_completeness.py")
        _spec = _ilu.spec_from_file_location("qf_native_reject_list_completeness", _rc_path)
        _rc = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_rc)
        _rc.warn_if_stale(comfy_root=os.path.dirname(os.path.dirname(comfy.model_management.__file__)))
    except Exception as _rc_exc:  # noqa: BLE001 — the self-check must never break plugin import
        logging.debug("[qf_native] reject-list self-check skipped: %r", _rc_exc)
    # (R7: the old single-node "QuantFuncNativeLoader" display entry is GONE with the class —
    # a display mapping for an unregistered class is dead weight; the three per-family loaders
    # register their display names beside their class mappings above.)
    NODE_DISPLAY_NAME_MAPPINGS.update({"QuantFuncNativeLoRA": "QuantFunc Native LoRA"})


# ── QuantFunc Cloud TE encode node (design v11) ──────────────────────────────────
# Module-level merge (never replace), independent of the family-loader flow, so the
# cloud-TE node registers even when the family loaders are unavailable. Fully guarded —
# a failure (e.g. torch missing outside ComfyUI) must never break plugin import.
try:
    from . import qf_cloud_te_node as _qf_cloud_te
    NODE_CLASS_MAPPINGS.update(_qf_cloud_te.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_qf_cloud_te.NODE_DISPLAY_NAME_MAPPINGS)
except Exception as _qf_cloud_te_exc:  # noqa: BLE001
    import logging as _qf_lg
    _qf_lg.warning("[qf_native] cloud-TE node not registered: %r", _qf_cloud_te_exc)
# ── QuantFunc LTX-2.5 AV ancestral-sampler audio fix ─────────────────────────────
# The engine's STATELESS flow-match forward requires a non-re-noised trajectory; comfy's
# ancestral samplers (euler_ancestral auto-routes to *_RF for CONST/flow models) re-noise x
# every step, which collapses the low-dim AV AUDIO lane to silence. Neutralize the audio-lane
# re-noise for QF LTX-2.5 AV models ONLY (video ancestral stochasticity preserved). Fully
# guarded — a failure must never break plugin import.
try:
    from . import qf_ltx_ancestral_audio_fix as _qf_ltx_afix
    _qf_ltx_afix.install()
except Exception as _qf_ltx_afix_exc:  # noqa: BLE001
    import logging as _qf_lg2
    _qf_lg2.warning("[qf_native] LTX-2.5 AV audio fix not installed: %r", _qf_ltx_afix_exc)


# ── Engine log detail ─────────────────────────────────────────────────────────
# One HIDDEN `log_level` input on EVERY QuantFunc loader (family loaders and the cloud-TE loader alike): never
# shown, so users do not choose it; a prompt that carries it (the test harness's) still sets it. Default warning
# (warnings and errors only). The value is handed to qf_engine before the loader runs and applied to the
# engine library as soon as it is (or once it gets) loaded; asking never loads it. Process-wide. This runs LAST,
# after every NODE_CLASS_MAPPINGS registration above, so no loader is missed (tests/log_level_input_test.py
# checks that no registration comes after it). Fully guarded: it must never break plugin import.
try:
    from . import qf_engine as _qf_ll_engine
    from .qf_log_level import add_log_level_input as _qf_add_log_level
    for _qf_name, _qf_cls in list(NODE_CLASS_MAPPINGS.items()):
        if _qf_name.startswith("QuantFunc") and _qf_name.endswith("Loader"):
            _qf_add_log_level(_qf_cls, _qf_ll_engine.set_log_level)
except Exception as _qf_ll_exc:  # noqa: BLE001
    import logging as _qf_ll_lg
    _qf_ll_lg.warning("[qf_native] log-level input not attached: %r", _qf_ll_exc)
