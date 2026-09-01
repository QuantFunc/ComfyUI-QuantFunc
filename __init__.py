"""qf_native — native ComfyUI loader for the QuantFunc engine (wan svdq).

ONE loader node creates a QuantFunc pipeline and returns a native comfy MODEL (a QFModelPatcher)
that native KSampler / KSamplerAdvanced drive via quantfunc_denoise_step. CLIP + VAE stay NATIVE
comfy nodes (maximize comfy-ecosystem compatibility).
"""
import os
import json
import logging
import weakref

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


def _package_weight_paths(pkg):
    """The svdq transformer weight files inside a package (footprint estimate)."""
    outs = []
    for sub in ("transformer", "transformer_2"):
        d = os.path.join(pkg, sub)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".safetensors"):
                    outs.append(os.path.join(d, f))
    return outs


def _estimate_package_footprint(pkg):
    """Engine-resident transformer weight bytes for a package — computed from the FILES, so the
    memory ledger has a real number before the pipeline is created (QFLazyEngine)."""
    try:
        return qfe.estimate_footprint_bytes(*_package_weight_paths(pkg))
    except Exception:  # noqa: BLE001 — a bad estimate must not break loading
        return 1


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


def _preset_file_expectations():
    """One line per shipped preset naming its expected transformer files (tooltip text)."""
    out = []
    try:
        for d in _model_config_choices():
            mf = os.path.join(_CONFIGS_DIR, d, "qf_native.json")
            if os.path.isfile(mf):
                notes = (json.load(open(mf)).get("notes") or "")
                exp = notes.split("expected files:")[-1].strip() if "expected files:" in notes else ""
                if exp:
                    out.append(f"{d}: {exp}")
    except Exception:  # noqa: BLE001
        pass
    return (" Expected files — " + "; ".join(out)) if out else ""


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
# NOT leak a fresh pipeline. Keyed by the create-determining inputs. TWO bounds:
#  • VRAM (at most ONE pipeline resident): `_evict_other_pipelines` frees every OTHER cached handle's
#    VRAM via quantfunc_unload_sync — the handles STAY VALID (reload lazily on their next generate).
#    NO destroy of a LIVE handle → no use-after-destroy against a QFModelPatcher comfy still holds.
#  • HOST RAM (bounded by the set of LIVE patchers): `_sweep_dead_pipelines` DESTROYS a handle only
#    once its QFWanModel has been garbage-collected (comfy dropped the patcher) — a dead model cannot
#    be use-after-freed, so destroy is safe there. Without this the unload_vram-only design leaks a
#    multi-GB CPU backup per distinct config forever (a resolution/model sweep). `_PIPELINE_MODELS`
#    holds weakrefs to ALL of a config's live models as the liveness signal (bound in each build).
_PIPELINE_CACHE = {}     # ckey -> QFEngineHandle
_PIPELINE_MODELS = {}    # ckey -> [weakref.ref(model), ...] — ALL live consumers of that config's
#                          handle. A LIST, not one ref: two loader nodes on the same package share
#                          ONE handle, and a single last-load-wins ref made the FIRST model invisible
#                          to liveness (measured: a sibling's release/sweep could destroy the shared
#                          handle under a still-live patcher → NULL pipeline at its next denoise +
#                          a ledger that kept reporting the destroyed backup).
# KNOWN RESIDUALS — both UNREACHABLE in the production consumer (comfy's PromptExecutor runs every node on
# a SINGLE execution thread, so load() is never concurrent with another load()); recorded, not silent:
#  • These two dicts are mutated WITHOUT a lock. A hypothetical MULTI-THREADED caller could race the
#    check-then-create in _get_engine. NOT locked deliberately: a lock would have to span the multi-second
#    create_pipeline (a heavy critical section) to guard a path the single-threaded executor cannot reach —
#    defensive wiring for an unreachable race. Revisit IF comfy ever executes nodes concurrently.
#  • If load() raises AFTER _get_engine caches the handle but BEFORE it binds the weakref
#    (_PIPELINE_MODELS[ckey]=), that handle reads as "unbound" to _sweep_dead_pipelines forever (never
#    destroyed). Its VRAM is still freed by _evict_other_pipelines; only a DISTINCT config that fails
#    mid-load AND is never retried leaks ONE CPU-backup handle. A retry of the SAME config REUSES the
#    cached handle (faster) and, on success, binds the weakref → it becomes sweepable — so the pin is a
#    reuse-on-retry feature except in the never-retried case; evicting on failure would forfeit it.


def _bind_pipeline_model(ckey, model):
    """Register `model` as a live consumer of ckey's cached handle (called by every family builder
    after constructing its model). Prunes dead refs so the list tracks the true live set."""
    refs = [r for r in _PIPELINE_MODELS.get(ckey, []) if r() is not None]
    if not any(r() is model for r in refs):   # re-materialize of a live model must not accumulate
        refs.append(weakref.ref(model))
    _PIPELINE_MODELS[ckey] = refs


def _live_pipeline_models(ckey):
    """The models still alive on ckey's handle (prunes dead refs in place)."""
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
    req_qf = getattr(requester, "_qf", requester)   # model -> wrapper; wrapper -> itself; None -> None
    foreign = []
    for m in _live_pipeline_models(ckey):
        if requester is not None and (m is requester or getattr(m, "_qf", None) is req_qf):
            continue
        foreign.append(m)
    if foreign:
        print(f"[qf_native] retire refused ({reason or 'unspecified'}): {len(foreign)} live "
              "foreign consumer(s) still bound to this cache entry", flush=True)
        return False
    _PIPELINE_CACHE.pop(ckey, None)
    if not keep_binding:
        _PIPELINE_MODELS.pop(ckey, None)
    try:
        eng.destroy()   # idempotent (pipeline→None); closes any session first
    except Exception:  # noqa: BLE001 — retire must never mask the caller's continuation
        pass
    return True


def _evict_other_pipelines(keep_key):
    """Free the VRAM of every OTHER cached pipeline (unload_sync → CPU backup; handle stays valid)."""
    for k, e in list(_PIPELINE_CACHE.items()):
        if k != keep_key and e is not None and e.pipeline is not None and not e.unloaded:
            try:
                e.unload_vram()
            except Exception:  # noqa: BLE001
                pass


def _sweep_dead_pipelines(keep_key):
    """Reclaim HOST RAM: destroy any cached handle whose model comfy has GC'd (no live reference → no
    UAF). A still-live model is kept (its VRAM is freed separately by _evict_other_pipelines). Never
    touches keep_key or a handle not yet bound to a model (its load() may still be in flight)."""
    for k in list(_PIPELINE_CACHE.keys()):
        if k == keep_key:
            continue
        if not _PIPELINE_MODELS.get(k):
            continue   # UNBOUND (load may be in flight) → never touch; only ever-bound entries sweep
        eng = _PIPELINE_CACHE.get(k)
        # requester=None ⇒ _retire_handle refuses while ANY consumer is live (only-all-dead sweeps)
        if eng is not None and _retire_handle(k, eng, None, reason="host-RAM sweep"):
            print("[qf_native] host-RAM sweep: destroyed a cached pipeline whose QFWanModel was GC'd "
                  "(comfy dropped its patcher) — freed its CPU backup", flush=True)


def _get_engine(model_dir, create_cfg=None, device_idx=0):
    """Create (or reuse) the engine for a model PACKAGE dir. The native library path is
    resolved internally (resolve_so_path — NEVER a node input, #vuln). Create is MINIMAL:
    a PREQUANT svdq package carries its own layout/precision in its metadata; anything
    supplied on top competes with it and loses. The transformer weights live INSIDE the
    package (engine loads model_dir/transformer[_2]/ directly — no path override).
    create_cfg carries the per-family create keys (e.g. wan text_precision) + the
    declarative lora stack from chained QuantFuncNativeLoRA nodes."""
    # [manual-residency] the session-knob-≠-create-key guard is sealed INSIDE
    # qf_engine.create_pipeline (the real quantfunc_create boundary — construction-enforced,
    # unbypassable by a future direct caller), not duplicated here (one truth source).
    # [EXPERIMENT-ONLY 2026-08-23, internal A/B — remove after measurement; shipped
    # form will be loader-node widgets, per the no-env-production-switch rule]:
    # QF_NATIVE_CREATE_EXTRA merges extra create keys (internal dials /
    # attention_backend...) BEFORE the cache key is computed, so每个 extra 配置有独立
    # pipeline cache 身份 (never collides with the default config's handle).
    _extra = os.environ.get("QF_NATIVE_CREATE_EXTRA")
    if _extra:
        create_cfg = {**(create_cfg or {}), **json.loads(_extra)}
        print(f"[qf_native] EXPERIMENT create-extra merged: {_extra}", flush=True)
    lib = qfe.load_lib()
    # device_idx follows COMFY's torch device (the builders pass get_torch_device().index), so
    # a ComfyUI started on a different GPU — or an in-process device choice — drives the engine
    # on the SAME card comfy computes on. Part of the cache key: two devices = two handles.
    ckey = (qfe.resolve_so_path(), model_dir, "svdq", int(device_idx),
            json.dumps(create_cfg or {}, sort_keys=True))
    _sweep_dead_pipelines(ckey)        # reclaim host RAM from configs whose patchers comfy dropped
    _evict_other_pipelines(ckey)       # keep only THIS config's VRAM resident (others reload lazily)
    eng = _PIPELINE_CACHE.get(ckey)
    if eng is not None and eng.pipeline is not None:
        return eng, ckey
    key, surl = _read_auth()
    cfg = dict(create_cfg or {})       # minimal: svdq metadata drives layout/precision
    if key:
        cfg["api_key"] = key
        cfg["server_url"] = surl
    pipeline = qfe.create_pipeline(lib, model_dir=model_dir, transformer_path=None,
                                   model_backend="svdq", device_idx=int(device_idx),
                                   config_json=(cfg if cfg else None))
    # Footprint = the ENGINE-RESIDENT transformer weight bytes only (dual-expert). VAE + text_encoder
    # stay NATIVE comfy nodes (comfy already accounts for them), so they must NOT be added here — an
    # over-report would make comfy's ledger evict siblings that actually fit.
    footprint = _estimate_package_footprint(model_dir)
    eng = qfe.QFEngineHandle(lib, pipeline, footprint_bytes=footprint)
    _PIPELINE_CACHE[ckey] = eng
    return eng, ckey


if _IMPORT_OK:
    # ── family REGISTRY: assembled from the per-family modules. Family LOGIC lives in the family
    #    modules; what remains here is the shared NODE SURFACE — transformer1/transformer2 FILE
    #    dropdowns + model_type + resident_block_count (the LoRA node's target is WIRE-derived,
    #    no combo — chaining on the high output acts on the high expert). That
    #    surface is family-neutral as long as a new family fits the "1-2 transformer files +
    #    shipped config bundle" shape; one needing a NEW input must extend INPUT_TYPES here, so
    #    "add a family = one module + one _FAMILY_MODULES line" holds for the common case, not
    #    unconditionally. ──
    # Each module owns ONE model family end to end (its comfy model subclass, its builder, and its
    # detection rule) and exposes exactly three names: FAMILY / matches() / register(deps).
    # Adding a family = write qf_<name>_modelpatcher.py + add it to _FAMILY_MODULES. No edit to the
    # node, the dispatch or the detection lives here, so families cannot bleed into each other.
    _FAMILY_MODULES = ("qf_wan_modelpatcher", "qf_ltx_modelpatcher", "qf_h3_modelpatcher", "qf_krea2_modelpatcher")
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
                "estimate_footprint": _estimate_package_footprint,
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
                         resident_block_count, transformer2, sparse_opts=None):
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
        import fnmatch
        hints = manifest.get("file_hints") or {}
        for arm, val in (("transformer1", transformer1),
                         ("transformer2", None if xfm2 is None else transformer2)):
            pats = hints.get(arm) or []
            if val is None or not pats:
                continue
            base = os.path.basename(val).lower()
            if not any(fnmatch.fnmatch(base, p.lower()) for p in pats):
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
        # are SESSION knobs (same class as resident_block_count): armed
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
                      resident_block_count=int(resident_block_count),
                      bundle_dir=bundle_dir, **_kw)
        # [cache/sparse surface REMOVED, user 2026-08-29 「移除所有loader的cache以及
        # 稀疏入口 整体默认不生效」] The per-model set_step_cache/set_block_cache/
        # set_sparse arming that lived here is GONE with the loader widgets — the
        # mixin defaults (0.0/0.0/1.0) already omit every begin key, so the engine
        # paths are byte-identical without any call. Re-enabling is a plugin-side
        # revert of THIS commit (widgets + this arming loop); the engine session
        # dials are untouched and stay available.
        return out

    _RESIDENT_BLOCKS_INPUT = ("INT", {"default": 999, "min": 1, "max": 1024,
                                      "tooltip": "GPU-resident transformer blocks — the native "
                                                 "seam's ONLY residency knob. The engine clamps "
                                                 "to the model's block count, so the default "
                                                 "keeps every block resident on a card that "
                                                 "fits."})
    # [cache surface RE-ENABLED, user 2026-08-31 「step cache 以及 fbcache 的开关重新开启」]
    # The step_cache (EasyCache) + block_cache (First-Block Cache = "fbcache") widgets
    # restored to the video loaders (wan/LTX/H3) — the two loader widgets + their arming
    # loop that the 2026-08-29 removal dropped. SPARSE is deliberately NOT re-enabled
    # (the user named only the two caches). Both are RUNTIME SESSION knobs (mixin
    # set_step_cache/set_block_cache → residency_opts begin keys; 0.0 = OFF = byte-identical,
    # no create key, no pipeline rebuild on a widget change). Engine EasyCache/FBCache
    # (lighting_step_cache.h) is untouched on main.
    _STEP_CACHE_INPUT = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                         "tooltip": "Step cache threshold (EasyCache; 0 = OFF, byte-identical). "
                                    ">0 lets the engine SKIP whole denoise steps whose "
                                    "predicted change is below this relative budget and "
                                    "reuse the cached trajectory (typical 0.02-0.05; larger "
                                    "= faster but drifts more). Runtime session knob — "
                                    "takes effect next run, never rebuilds the pipeline."})
    _BLOCK_CACHE_INPUT = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                          "tooltip": "Block cache threshold (First-Block Cache; 0 = OFF, byte-identical). "
                                     ">0 runs block 0 every step and SKIPS blocks 1..N-1 when "
                                     "block 0's output change is below this relative budget, "
                                     "reconstructing from the cached residual — effective even "
                                     "on few-step distilled workflows (typical 0.05-0.12). "
                                     "Runtime session knob — takes effect next run, never "
                                     "rebuilds. COMPOSABLE with step_cache (EC skips whole "
                                     "steps; FBC skips blocks inside computed steps)."})
    # [sol-tau dial 2026-08-31] the ONE user-facing Sol-Attn knob (user "就一个就好").
    # Only meaningful under attention_backend=qfa (qfa = sol+qfa on H3/LTX). Same
    # runtime-session-knob class as step_cache/sparse: re-sent each run, no rebuild;
    # 1.0 default is omitted (older engines refuse unknown keys loud) and the engine
    # resets an absent key to 1.0 (dialing back truly restores the default).
    _TOKEN_PRUNE_INPUT = ("FLOAT", {"default": 1.0, "min": 0.2, "max": 1.0, "step": 0.05,
                     "tooltip": "Token-prune keep fraction (CAT). 1.0 = OFF; e.g. 0.4-0.5 = "
                                "recompute only that fraction of VIDEO tokens per step "
                                "(~1.4-1.5x faster, quality floor: last step always full; "
                                "audio never pruned). Auto-disabled while a step/block cache "
                                "is armed (they don't compose)."})
    _SOL_TAU_INPUT = ("FLOAT", {"default": 1.0, "min": 0.02, "max": 1.0, "step": 0.01,
                     "tooltip": "Sol-Attn keep-ratio (qfa/flash/sage backends). 1.0 = OFF "
                                "(the backend's original dense attention). <1 = sol engaged; "
                                "value ~= fraction of attention blocks computed exactly: "
                                "SMALLER = sparser = FASTER. 0.15-0.2 = the measured speed "
                                "optimum (recommended start); 0.05-0.1 = faster (watch "
                                "quality); 0.3-0.5 = quality-leaning."})

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
        "Sampler/scheduler/CFG stay ENTIRELY ComfyUI-side — the engine only denoises per step "
        "(latents in, velocity out). Limits: (1) ControlNet is not consumed by this seam "
        "(refused loud). (2) Interrupt stops BETWEEN denoise steps. (3) On Linux a fail-closed "
        "CUDA-toolchain check refuses a torch/.so CUDA-major mismatch; on Windows/macOS set "
        "QF_NATIVE_ALLOW_UNVERIFIED_TOOLCHAIN=1 after confirming they share a CUDA major.")

    # [attention backend selector, user 2026-08-27] one user-facing dropdown per loader.
    # SM-GATED: SM80+ offers the full set; SM75 (Turing) has NO int8-QK sage and NO
    # flash_attn build, so only qfa + fp16_native are valid there. The widget VALUE is a
    # display name; _attn_backend_to_engine maps it to the engine's comp_opts string
    # ("fp16_native" -> "native"). "auto" = the engine's per-SM resolution (default), and
    # is passed through so the user's choice is always the single source of truth.
    # [qfa = sol+qfa — user 2026-08-31 "不要新增任何开关 qfa就等于 sol+qfa"] there is NO separate
    # "qfa-sol" option: on the video families (H3 / LTX2) selecting "qfa" routes self-attn through
    # the engine's Sol-Attn NO-DROP arm at head_dim==128 automatically. The engine still ACCEPTS
    # "qfa-sol" as a transparent alias, but it is deliberately NOT surfaced as a user-facing choice.
    _ATTN_BACKEND_SM80PLUS = ["auto", "qfa", "flash", "sage", "fp16_native"]
    _ATTN_BACKEND_SM75 = ["qfa", "fp16_native"]

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
                "tooltip": "Self-attention backend. auto = engine picks per-SM (default). "
                           "flash = fp16 flash-attn (most robust; H3 high-res needs this). "
                           "sage = int8-QK sage (fastest, SM80+). qfa = int8-QK + Hadamard "
                           "rotation. fp16_native = portable fp16/fp32-score fallback. "
                           "SM75 (Turing) offers only qfa + fp16_native. CREATE key — "
                           "changing it re-creates the pipeline."})

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
                                 {"tooltip": "The OFFICIAL wan model config preset (arch + VAE "
                                             "geometry + expected-file naming). "
                                             + _preset_file_expectations()}),
                "resident_block_count": _RESIDENT_BLOCKS_INPUT,
            }, "optional": {
                "attention_backend": _attn_backend_input(),
                "step_cache": _STEP_CACHE_INPUT,
                "block_cache": _BLOCK_CACHE_INPUT,
                "act_scale_g32": ("BOOLEAN", {"default": False,
                    "tooltip": "svdq int4 激活-scale 组宽开关 (#565). OFF=g64 (默认, 与旧版逐字节一致); "
                               "ON=g32 (更细的激活量化组, 实测 -9.1% 激活量化误差, 前向 +~45%, 仅 SM89/86 是真杠杆; "
                               "SM120 上 int4 已用更细的 E0M3 g16, 此开关 no-op). 仅对 svdq int4 生效. "
                               "把 int4 画质往 fp8 靠的实验开关 —— 温和收益, 单靠它通常不足以完全追平 fp8 "
                               "(根因是 int4 激活精度; 干净对齐 fp8 需 a8w4)."}),
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

        def load(self, transformer1, transformer2, model_config, resident_block_count=999,
                 attention_backend="auto", step_cache=0.0, block_cache=0.0, act_scale_g32=False):
            # [#659] the sparse dial is a SESSION knob (never a create key — no rebuild
            # on change); sparse_opts carries CREATE-level keys only.
            # [runtime dial 2026-08-29] attention_backend is a SESSION knob now — NOT a
            # create key (widget change no longer re-keys the loader = no rebuild).
            sparse_opts = {"act_scale_g32": True} if act_scale_g32 else None
            pair = _run_family_load("wan", transformer1, model_config,
                                    resident_block_count, transformer2,
                                    sparse_opts=sparse_opts)
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
                                {"tooltip": "The LTX-2 transformer .safetensors under "
                                            "models/diffusion_models (the QuantFunc int4 "
                                            "single-file export — connector + audio blocks "
                                            "packed inside)."}),
                "model_config": (_model_config_choices(family="ltx2"),
                                 {"tooltip": "The OFFICIAL LTX-2 model config preset. "
                                             + _preset_file_expectations()}),
                "resident_block_count": _RESIDENT_BLOCKS_INPUT,
            }, "optional": {
                "attention_backend": _attn_backend_input(),
                "sol_tau": _SOL_TAU_INPUT,
                "token_prune": _TOKEN_PRUNE_INPUT,
                "step_cache": _STEP_CACHE_INPUT,
                "block_cache": _BLOCK_CACHE_INPUT,
            }}

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = ("QuantFunc LTX-2 loader (svdq, denoise_only): one native MODEL a stock "
                       "sampler drives with latents. i2v: wire the official "
                       "LTXVImgToVideoInplace into the LATENT path (wan-aligned — the model "
                       "only consumes latents; comfy's sampler applies the frame-0 mask). "
                       + _COMMON_LIMITS)

        def load(self, transformer, model_config, resident_block_count=999,
                 attention_backend="auto", sol_tau=1.0, token_prune=1.0, step_cache=0.0, block_cache=0.0):
            # [aux-auto] NO aux file widgets and NO image socket (user 2026-08-22 "只保留
            # transformer/block/model_config … 只关注latent"): te/audio-vae/connectors
            # resolve from the preset manifest's aux_files inside _run_family_load; i2v is
            # the workflow's own latent conditioning (LTXVImgToVideoInplace), exactly like
            # wan's cond-latent shape.
            _p = _run_family_load("ltx2", transformer, model_config,
                                   resident_block_count, None, sparse_opts=None)
            _mm = getattr(_p, "model", None)
            if _mm is not None and hasattr(_mm, "set_attn_backend"):
                _mm.set_attn_backend(_attn_backend_to_engine(attention_backend))
            if _mm is not None and hasattr(_mm, "set_sol_tau"):
                _mm.set_sol_tau(sol_tau)
            if _mm is not None and hasattr(_mm, "set_token_prune"):
                _mm.set_token_prune(token_prune)
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
                                {"tooltip": "The Krea-2 Turbo transformer .safetensors under "
                                            "models/diffusion_models (the QuantFunc int4 "
                                            "export)."}),
                "model_config": (_model_config_choices(family="krea2"),
                                 {"tooltip": "The OFFICIAL Krea-2 model config preset. "
                                             + _preset_file_expectations()}),
            }, "optional": {
                "attention_backend": _attn_backend_input(),
                "resident_block_count": ("INT", {
                    "default": 0, "min": 0, "max": 999,
                    "tooltip": "0 = auto (engine-managed residency, the default). "
                               ">0 = MANUAL: pin the first N transformer blocks "
                               "GPU-resident (999 = as many as fit; never-OOM "
                               "guarded). SESSION knob — applied at the next "
                               "sampling run, no model rebuild."}),
                # token-prune (CAT): image tokens only — text conditioning never
                # pruned. SESSION knob (rides begin options), no rebuild.
                # IMAGE-tuned range/guidance (e2e 2026-09-01, 8-step distilled @1024²,
                # user ruling: recommend 0.75+): 0.7 measured clean vs dense (1.2x);
                # 0.5 = subject clean but BACKGROUND mosaic patches (1.4x); 0.3 =
                # mosaic everywhere (1.8x). A single image has no temporal masking,
                # so stale patches show directly — hence the conservative floor.
                "token_prune": ("FLOAT", {
                    "default": 1.0, "min": 0.2, "max": 1.0, "step": 0.05,
                    "tooltip": "Token-prune keep fraction (CAT). 1.0 = OFF. "
                               "IMAGE recommendation: 0.75 or higher (clean, "
                               "~1.2x faster). Below 0.6 expect mosaic patches "
                               "on this 8-step distilled model (text conditioning "
                               "never pruned; last step always full)."}),
            }}

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = ("QuantFunc Krea-2 Turbo loader (svdq, denoise_only): one native t2i "
                       "MODEL a stock sampler drives with latents. " + _COMMON_LIMITS)

        def load(self, transformer, model_config, attention_backend="auto",
                 resident_block_count=0, token_prune=1.0):
            # [runtime dials] backend + residency are SESSION knobs (engine
            # applyAttnBackendDial / applyManualResidencyImageSessions): NOT create
            # keys, so a widget change never re-keys the engine = no rebuild.
            # resident_block_count 0 = auto (key omitted from begin — engine
            # warmup/auto residency, byte-unchanged legacy); >0 = manual pin.
            # The WIDGET value rides the BUILD kwargs (single writer): the
            # NativeLoRA chain's internal rebuild constructs a FRESH model wrapper
            # from the CAPTURED build kwargs, so a post-load setter alone is
            # overwritten on the LoRA path (measured R1: begin carried the old
            # hardcoded 999 while the widget said 0).
            _p = _run_family_load("krea2", transformer, model_config,
                                  int(resident_block_count), None, sparse_opts=None)
            _mm = getattr(_p, "model", None)
            if _mm is not None and hasattr(_mm, "set_attn_backend"):
                _mm.set_attn_backend(_attn_backend_to_engine(attention_backend))
            if _mm is not None and hasattr(_mm, "set_resident_block_count"):
                _mm.set_resident_block_count(int(resident_block_count))
            if _mm is not None and hasattr(_mm, "set_token_prune"):
                _mm.set_token_prune(token_prune)
            return (_p,)


    class QuantFuncH3Loader:
        """MiniMax-H3 loader — single MODEL output (single-expert AV family)."""

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "transformer": (_transformer_choices(),
                                {"tooltip": "The MiniMax-H3 transformer .safetensors under "
                                            "models/diffusion_models."}),
                "model_config": (_model_config_choices(family="minimax-h3"),
                                 {"tooltip": "The OFFICIAL MiniMax-H3 model config preset. "
                                             + _preset_file_expectations()}),
                "resident_block_count": _RESIDENT_BLOCKS_INPUT,
            }, "optional": {
                # [sparse, user 2026-08-25 ONE-number dial; #659 session knob — no rebuild]
                # H3 default = flash: this model's auto resolves to sage2 int8-QK, which is
                # BROKEN on H3's post-qk-RMSNorm γ-outliers at high-res (blank/NaN — measured
                # 928²/S=31538: attn out absmax 0 → step-1 all-NaN → audio avcodec crash +
                # video blur). flash (fp16) is the verified-clean default; user can still pick
                # auto/sage/qfa/native. (Wan→auto→qfa, LTX→auto are fine → they keep 'auto'.)
                "attention_backend": _attn_backend_input("flash"),
                "sol_tau": _SOL_TAU_INPUT,
                "token_prune": _TOKEN_PRUNE_INPUT,
                "step_cache": _STEP_CACHE_INPUT,
                "block_cache": _BLOCK_CACHE_INPUT,
            }}

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = ("QuantFunc MiniMax-H3 loader (svdq, denoise_only): one native AV MODEL a "
                       "stock sampler drives with latents. " + _COMMON_LIMITS)

        def load(self, transformer, model_config, resident_block_count=999,
                 attention_backend="flash", sol_tau=1.0, token_prune=1.0, step_cache=0.0, block_cache=0.0):  # H3: flash default (auto→sage is broken)
            _p = _run_family_load("minimax-h3", transformer, model_config,
                                   resident_block_count, None, sparse_opts=None)
            _mm = getattr(_p, "model", None)
            if _mm is not None and hasattr(_mm, "set_attn_backend"):
                _mm.set_attn_backend(_attn_backend_to_engine(attention_backend))
            if _mm is not None and hasattr(_mm, "set_sol_tau"):
                _mm.set_sol_tau(sol_tau)
            if _mm is not None and hasattr(_mm, "set_token_prune"):
                _mm.set_token_prune(token_prune)
            _arm_session_caches(_mm, step_cache, block_cache)
            return (_p,)

    class QuantFuncNativeLoRA:
        """Sidecar LoRA for the QuantFunc native loader — MODEL in, MODEL out (LoraLoaderModelOnly
        shape). Chain several to stack them.

        The engine merges sidecar LoRA at pipeline CREATE time (runtime hot-swap is not wired for
        the video pipelines), so this node RE-CREATES the pipeline for the accumulated set — the
        create itself is deferred (QFLazyEngine), so a chain of N nodes still builds ONE pipeline.
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
        DESCRIPTION = ("Attaches a sidecar LoRA to a QuantFunc native MODEL (wire downstream of the "
                       "QuantFunc Native Loader; chain several to stack). The engine merges sidecar "
                       "LoRA at create time, so the pipeline is re-created for the new set.")

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
                                "QuantFuncNativeLoRA": QuantFuncNativeLoRA})
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "QuantFuncWanLoader": "QuantFunc Wan Loader (high+low)",
        "QuantFuncLTXLoader": "QuantFunc LTX-2 Loader",
        "QuantFuncH3Loader": "QuantFunc MiniMax-H3 Loader",
        "QuantFuncKrea2Loader": "QuantFunc Krea-2 Loader"})

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


