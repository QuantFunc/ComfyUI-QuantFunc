"""qf_native — native ComfyUI loaders for the QuantFunc engine (LTX-2.5, MiniMax-H3, Krea-2, Qwen-Image-2.1; svdq).

ONE loader node creates a QuantFunc pipeline and returns a native comfy MODEL (a QFModelPatcher)
that native KSampler / KSamplerAdvanced drive via quantfunc_denoise_step. CLIP + VAE stay NATIVE
comfy nodes (maximize comfy-ecosystem compatibility).
"""
import os
import sys
import json
import inspect
import weakref
import threading

# qf_engine is stdlib only (nothing from comfy); every warning below prints through its console-safe logger (#738: one
# character the console's code page cannot hold must never raise). Its import is guarded like the rest: a broken
# plugin file still degrades to zero nodes + a warning.
try:
    from . import qf_engine as qfe
    _log = qfe.logger(__name__)
except Exception as _qfe_exc:  # noqa: BLE001 - never break registration
    import logging
    qfe, _log = None, logging.getLogger(__name__)
    _log.warning("[qf_native] disabled - qf_engine failed to import: %s", ascii(_qfe_exc))
# qf_api_key (the loaders' API key field) is stdlib only too. Without it no loader registers (the block below), so a key
# typed into a field is never dropped silently in favour of config.json.
try:
    from . import qf_api_key
except Exception as _qak_exc:  # noqa: BLE001 - never break registration
    qf_api_key = None
    _log.warning("[qf_native] disabled - qf_api_key failed to import: %s", ascii(_qak_exc))

# GUARDED imports (mirror the real plugin __init__.py) — a broken comfy-internals import must NOT
# take down node registration on a ComfyUI upgrade; degrade to zero nodes + a loud warning.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"   # ComfyUI serves it to the browser: the loaders' API key field (web/quantfunc_api_key.js)
try:
    if qfe is None or qf_api_key is None:
        raise ImportError("qf_engine or qf_api_key did not import (see the warning above)")
    import comfy.model_management
    import comfy.supported_models
    from . import qf_modelpatcher as qfmp
    from .qf_modelpatcher import QFModelPatcher
    _IMPORT_OK = True
except Exception as _exc:  # noqa: BLE001 - never break registration; report loudly
    _log.warning("[qf_native] disabled - a required import failed (ComfyUI API drift?): %s", ascii(_exc))
    _IMPORT_OK = False


# folder_paths gives the model-file listing surface (INT8-Fast-aligned: FILES, not dirs).
try:
    import folder_paths as _folder_paths
except Exception as _fp_exc:  # noqa: BLE001 - never break registration
    _folder_paths = None
    _log.warning("[qf_native] folder_paths unavailable: %s", ascii(_fp_exc))

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
    PER-FAMILY loader nodes (user 2026-08-21 pivot: one loader node per model family), so an H3
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
                except Exception:  # noqa: BLE001 - unreadable manifest: hide from filtered lists
                    continue
            out.append(d)
        return out or [_NO_CFG_HINT]
    except Exception:  # noqa: BLE001
        return [_NO_CFG_HINT]


def _shipped_presets(family):
    """The configs/ presets this plugin ships for a family (the listing without its no-preset hint)."""
    return [c for c in _model_config_choices(family=family) if c != _NO_CFG_HINT]


def _family_preset(family):
    """The ONE preset a family loader uses (user 2026-09-24 「model_config也不是设置啊 就一个选项没意义啊」): every family
    ships exactly one configs/ preset, so the loaders show no model_config choice. A family with more than one would need
    that choice again: the loader refuses instead of guessing — bring the dropdown back when a family ships a second one."""
    presets = _shipped_presets(family)
    if len(presets) == 1:
        return presets[0]
    if not presets:
        _raise_if_a_manifest_is_unreadable(family)
        raise RuntimeError(f"qf_native: this plugin ships no model config for {family} (configs/ has no preset for it).")
    raise RuntimeError(f"qf_native: this plugin ships {len(presets)} model configs for {family} ({', '.join(presets)}) "
                       f"and the loader cannot choose between them.")


def _load_model_config(name):
    """Resolve + read a preset's manifest. The name comes from a saved workflow (workflow-serializable =
    untrusted): it must be exactly one of the listed preset dirs — no separators, no traversal."""
    if name == _NO_CFG_HINT or os.sep in name or "/" in name or "\\" in name or name in ("", ".", ".."):
        raise RuntimeError(f"qf_native: invalid model_config {name!r} - pick one of the shipped "
                           f"presets ({_model_config_choices()}).")
    bundle = os.path.join(_CONFIGS_DIR, name)
    mf = os.path.join(bundle, "qf_native.json")
    if not os.path.isfile(mf):
        raise RuntimeError(f"qf_native: model_config {name!r} has no qf_native.json manifest "
                           f"(shipped presets: {_model_config_choices()}).")
    try:
        with open(mf, encoding="utf-8") as fh:   # never the OS code page: Windows would read it as cp936 (#738)
            manifest = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"qf_native: model_config {name!r} manifest unreadable ({mf}): {exc}. "
                           f"It ships with the plugin: reinstall or update the QuantFunc plugin.") from exc
    if not isinstance(manifest, dict) or not manifest.get("family"):
        raise RuntimeError(f"qf_native: model_config {name!r} manifest ({mf}) must declare a family. "
                           f"It ships with the plugin: reinstall or update the QuantFunc plugin.")
    return bundle, manifest


def _raise_if_a_manifest_is_unreadable(family):
    """The family listings HIDE a preset whose manifest cannot be read, because a broken file must not take down node
    registration. A load that then finds no preset for its family would report the preset as not shipped. So every
    shipped manifest is read here, and a broken one is reported instead: all of them, since a broken manifest's family
    cannot be read either."""
    broken = []
    for name in _shipped_presets(None):
        try:
            _load_model_config(name)
        except RuntimeError as exc:
            broken.append(str(exc))
    if broken:
        raise RuntimeError(f"qf_native: no readable model config for {family}: " + " | ".join(broken))


_NO_XFM_HINT = "(no .safetensors in models/diffusion_models)"


def _transformer_choices():
    """The .safetensors FILES under comfy's models/diffusion_models — the SAME surface the
    reference INT8-Fast UNetLoaderINTW8A8 lists (folder_paths.get_filename_list). The user picks a
    transformer weight FILE directly. NOT a package DIRECTORY — the engine's denoise-only
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
    if _folder_paths is None or name in ("", _NO_XFM_HINT):
        raise RuntimeError("qf_native: no transformer weight selected - put the svdq .safetensors "
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
        raise RuntimeError("qf_native: no LoRA available - put .safetensors files in models/loras/")
    return _folder_paths.get_full_path_or_raise("loras", name)


# Auth resolves from a loader's API key field (user 2026-09-27), the PROCESS ENVIRONMENT and the package-bundled
# keyfile. The field is never a widget value or a prompt value: the prompt carries a reference that only this process
# resolves (qf_api_key). The keyfile PATH is never a node input: `keyfile` was previously a workflow STRING widget, same
# class as `so_path` (a shared workflow.json could point it at an arbitrary file the ComfyUI server then opens/parses). A
# workflow.json cannot set an env var, so the dev override QF_NATIVE_KEYFILE is safe; the default is the user's
# bin/<platform>/config.json, read as the shipped bin/<platform>/config.default.json until the first save
# (qf_api_key.config_to_read: the plugin never ships config.json, so an update never drops a saved key).
def _resolve_keyfile():
    override = os.environ.get(qfe._ENV_KEYFILE_OVERRIDE, "").strip()
    if override:
        return override
    pkg = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg, "bin", qfe._BIN_SUBDIR, qfe._KEYFILE_BASENAME)


def _read_auth(ui_key=None):
    # `ui_key`: the key of the loader's API key field (qf_api_key.field_key checked it). It wins, and then neither
    # QUANTFUNC_API_KEY nor the keyfile is read. Without one the API key is preferentially an env var (QUANTFUNC_API_KEY)
    # — the harness/plugin convention; only if absent do we read the bundled/overridden keyfile.
    surl = os.environ.get("QF_SERVER_URL", "https://service.quantfunc.com")
    if ui_key:
        return ui_key, surl
    key = os.environ.get("QUANTFUNC_API_KEY", "")
    keyfile = qf_api_key.config_to_read(_resolve_keyfile())
    if not key and keyfile and os.path.exists(keyfile):
        # A keyfile that is there but unreadable is an error, never "no key": swallowing it hid a Windows cp936 decode
        # failure (#738) behind a later auth failure.
        try:
            with open(keyfile, encoding="utf-8") as fh:
                c = json.load(fh)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"qf_native: the keyfile {keyfile} is unreadable: {exc}. Set QUANTFUNC_API_KEY (the file "
                               f"is then not read), or restore the file: reinstall or update the QuantFunc plugin.") from exc
        if not isinstance(c, dict):
            raise RuntimeError(f"qf_native: the keyfile {keyfile} must hold a JSON object. Set QUANTFUNC_API_KEY (the file "
                               f"is then not read), or restore the file: reinstall or update the QuantFunc plugin.")
        key, surl = c.get("api_key", ""), c.get("server_url", surl)
    return key, surl


def _comfy_device_index():
    """The CUDA index of the GPU ComfyUI computes on (0 when it is not a CUDA device or comfy is unavailable)."""
    try:
        dev = comfy.model_management.get_torch_device()
        return int(dev.index) if getattr(dev, "type", "") == "cuda" and dev.index is not None else 0
    except Exception:
        return 0


# Pipeline CACHE: reuse the created engine handle for a repeated config → a re-executed workflow does
# NOT leak a fresh pipeline. Keyed by the create-determining inputs.
#  • VRAM residency is scheduled by ComfyUI. Cache selection must not independently
#    evict other live handles; their resource adapters execute host reclaim requests.
#  • HOST RAM (bounded by the set of LIVE patchers): `_sweep_dead_pipelines` DESTROYS a handle only
#    once its family model has been garbage-collected (comfy dropped the patcher) — a dead model cannot
#    be use-after-freed, so destroy is safe there. Without this a cache that only released VRAM leaks a
#    multi-GB CPU backup per distinct config forever (a resolution/model sweep). `_PIPELINE_MODELS`
#    holds weakrefs to ALL live consumers: QFLazyEngine acquisition pins plus family models.
_PIPELINE_CACHE = {}     # ckey -> QFEngineHandle
_PREPARED_CACHE = weakref.WeakValueDictionary()  # cold identities retained by lazy consumers
_ENGINE_IDENTITY_LOCK = threading.RLock()
_PIPELINE_MODELS = {}    # ckey -> [weakref.ref(consumer), ...] - ALL live consumers of that config's
#                          handle. A LIST, not one ref: two loader nodes on the same package share
#                          ONE handle, and a single last-load-wins ref made the FIRST model invisible
#                          to liveness (measured: a sibling's release/sweep could destroy the shared
#                          handle under a still-live patcher → NULL pipeline at its next denoise +
#                          a ledger that kept reporting the destroyed backup).
_PIPELINE_SIGS = {}      # ckey -> {_loader_sig of each loader whose models bound it}: lives and dies with the
#                          binding list. The sweep reads it: a dead pipeline that a loader of the RUNNING prompt makes
#                          again (a settings-only change: ComfyUI drops the old patcher before the new loader runs)
#                          is that loader's, not dead (_pending_loader_sigs).
# CONCURRENCY CONTRACT: the three cache dictionaries above are protected by short
# `_ENGINE_IDENTITY_LOCK` lookup/publication/removal sections.  The lock is NEVER
# held while constructing QFPreparedEntry, creating or destroying a pipeline, or
# closing a losing prepared candidate.  Cold creation is single-flight under
# entry._materialize_lock; that path may briefly take this cache lock, and no
# inverse cache-lock -> entry-lock edge is permitted.


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
            _PIPELINE_SIGS.pop(ckey, None)


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
    after constructing its model). Drops dead refs so the list tracks the true live set."""
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
        sig = getattr(model, "_qf_loader_sig", None)
        if sig is not None:
            _PIPELINE_SIGS.setdefault(ckey, set()).add(sig)


def _live_pipeline_models(ckey):
    """The models still alive on ckey's handle (drops dead refs in place)."""
    with _ENGINE_IDENTITY_LOCK:
        refs = [r for r in _PIPELINE_MODELS.get(ckey, []) if r() is not None]
        if refs:
            _PIPELINE_MODELS[ckey] = refs
        else:
            _PIPELINE_MODELS.pop(ckey, None)
            _PIPELINE_SIGS.pop(ckey, None)
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
            qfe.info(f"[qf_native] retire refused ({reason or 'unspecified'}): {len(foreign)} live "
                  "foreign consumer(s) still bound to this cache entry", flush=True)
            return False
        _PIPELINE_CACHE.pop(ckey, None)
        prepared_key = (qfe.library_identity(eng.lib), ckey)
        candidate = _PREPARED_CACHE.get(prepared_key)
        if candidate is not None and candidate.resource is eng.resource:
            _PREPARED_CACHE.pop(prepared_key, None)
            # Publish non-creatable in the same identity transaction as cache
            # removal. retire_materialized performs the entry-locked cleanup.
            candidate._cache_usable = False
            retired_entry = candidate
        if not keep_binding:
            _PIPELINE_MODELS.pop(ckey, None)
            _PIPELINE_SIGS.pop(ckey, None)
    if retired_entry is not None:
        retired_entry.retire_materialized()
    # Native teardown may be slow and must never run under the cache lock.
    try:
        eng.destroy()   # idempotent (pipeline->None); closes any session first
    except Exception:  # noqa: BLE001 - retire must never mask the caller's continuation
        pass
    # Destroying the pipeline Closed its Owned identity, and Comfy may still list that owner until gc collects it:
    # say so where it happens, so a partial unload of it frees nothing (a Closed identity has nothing eligible) without
    # a native read that may stay BUSY.
    for owner in getattr(eng, "_qf_resource_adapters", ())[:1]:
        owner._closed_identity = True
    return True


def _sweep_dead_pipelines(keep_key):
    """Reclaim HOST RAM: destroy any cached handle whose model comfy has GC'd (no live reference → no
    UAF) and that no loader of the running prompt makes again (_dead_keys). A still-live model is kept; ComfyUI
    decides its VRAM residency. Never touches keep_key or a handle not yet bound to a model (its load() may still be
    in flight)."""
    for k in _dead_keys():
        if k == keep_key:
            continue
        with _ENGINE_IDENTITY_LOCK:
            eng = _PIPELINE_CACHE.get(k)
        # requester=None ⇒ _retire_handle refuses while ANY consumer is live (a consumer bound since _dead_keys read)
        if eng is not None and _retire_handle(k, eng, None, reason="host-RAM sweep"):
            qfe.info("[qf_native] host-RAM sweep: destroyed a cached pipeline whose model was GC'd "
                  "(comfy dropped its patcher) - freed its CPU backup", flush=True)


def _loader_sig(family, transformer, model_config, pinned_memory):
    """A loader's create inputs, as its load() hands them to _run_family_load (ComfyUI's validation already made
    pinned_memory a bool, in the queued prompt too): everything else on a loader is a session knob, so two loaders with
    equal signatures make the same pipeline."""
    return family, transformer, model_config, pinned_memory


def _pending_loader_sigs():
    """The _loader_sig of every QuantFunc loader the running prompt executes (the nodes its outputs depend on), from
    ComfyUI's queue. After a settings-only change ComfyUI drops the old patcher at the prompt's first model load, and
    the new loader may run only after another node staged its model (measured #748: the TE, 14.6 GB, on a 32 GB PC):
    the pipeline those inputs made is that loader's. Keeping it holds what a same-settings rerun holds (its patcher
    alive); destroying it cost a full rebuild. A linked create input is unknown until its node runs: no signature."""
    queue = getattr(getattr(getattr(sys.modules.get("server"), "PromptServer", None), "instance", None), "prompt_queue", None)
    if queue is None:
        return set()
    sigs = set()
    try:   # the queue's layout is ComfyUI's, not an API: a changed shape must not fail every cold create
        with queue.mutex:
            running = list(queue.currently_running.values())
        for item in running:   # (number, prompt_id, prompt, extra_data, outputs to execute, ...)
            prompt, todo, seen = item[2], [str(o) for o in item[4]], set()
            while todo:
                nid = todo.pop()
                node = prompt.get(nid)
                if nid in seen or not isinstance(node, dict):
                    continue
                seen.add(nid)
                ins = node.get("inputs") or {}
                todo += [str(v[0]) for v in ins.values() if isinstance(v, list) and v]
                cls = NODE_CLASS_MAPPINGS.get(node.get("class_type"))
                family = getattr(cls, "QF_FAMILY", None)
                if family is None or any(isinstance(ins.get(k), list)
                                         for k in ("transformer", "model_config", "pinned_memory")):
                    continue
                defaults = inspect.signature(cls.load).parameters   # what load() takes for an input the prompt omits
                sigs.add(_loader_sig(family, ins.get("transformer"),
                                     ins.get("model_config", defaults["model_config"].default),
                                     ins.get("pinned_memory", defaults["pinned_memory"].default)))
    except Exception as exc:  # noqa: BLE001
        _log.warning("[qf_native] the running prompt could not be read from ComfyUI's queue (%s); a dead pipeline is "
                     "released as if no loader were waiting for it", ascii(exc))
        return set()
    return sigs


def _dead_keys():
    """The cached handles bound only to consumers ComfyUI has garbage-collected, less those a loader of the running prompt
    makes again (_pending_loader_sigs, read only when something is dead). Read-only (it never drops an entry from a
    binding list: the sweep's bound-only gate reads them) and cheap, because ComfyUI's staging asks before every pin."""
    with _ENGINE_IDENTITY_LOCK:
        dead = [k for k in _PIPELINE_CACHE
                if _PIPELINE_MODELS.get(k) and all(r() is None for r in _PIPELINE_MODELS[k])]
    if not dead:
        return []
    pending = _pending_loader_sigs()
    with _ENGINE_IDENTITY_LOCK:
        return [k for k in dead if not (_PIPELINE_SIGS.get(k, set()) & pending)]


def _dead_pipelines():
    """(count, bytes): the _dead_keys handles, and the engine's own capacity figure summed over them. capacity_bytes is
    the Prepared capacity the engine reported for the model; it approximates the pipeline's host copy (Krea-2: 10.1 GB
    reported against its 8.85 + 1.27 GB backup, measured); an exact host-copy read is a post-release engine API."""
    dead = _dead_keys()
    with _ENGINE_IDENTITY_LOCK:
        return len(dead), sum(int(getattr(_PIPELINE_CACHE.get(k), "capacity_bytes", 0) or 0) for k in dead)


def _install_host_ram_release():
    """Wrap comfy.memory_management.extra_ram_release. ComfyUI calls it before it pins host memory for a model's weights
    (while the model runs, a forward-time cast, a partial unload); not on --fast-disk or --disable-pinned-memory, which
    pin nothing, and not from its own release between nodes, which goes to its cache directly. After ComfyUI's own
    release, once the available host RAM (ComfyUI's own reading: get_free_memory(cpu)) is less than ComfyUI's headroom
    (its target) plus what the pipelines of models ComfyUI dropped still hold, those pipelines are destroyed
    (_sweep_dead_pipelines: only-all-dead, no use-after-free). With RAM ample, or nothing dead, nothing changes. The
    wrapper passes every argument through and returns ComfyUI's own answer; a failure of ours only logs."""
    import torch
    import comfy.memory_management as cmm
    orig, cpu = cmm.extra_ram_release, torch.device("cpu")

    def extra_ram_release(target, *args, **kwargs):
        freed = orig(target, *args, **kwargs)
        try:
            count, held = _dead_pipelines()
            if count:
                available = comfy.model_management.get_free_memory(cpu)
                if available < target + held:
                    qfe.info(f"[qf_native] host RAM: {int(available) >> 20} MB available, under ComfyUI's "
                             f"{int(target) >> 20} MB headroom plus the {held >> 20} MB that {count} pipeline(s) of a "
                             "model ComfyUI dropped still hold: releasing them", flush=True)
                    _sweep_dead_pipelines(None)
        except Exception as exc:  # noqa: BLE001 - ComfyUI's pinning must never fail on our release
            _log.warning("[qf_native] host-RAM release skipped: %s", ascii(exc))
        return freed

    extra_ram_release._qf_host_ram_release = True
    cmm.extra_ram_release = extra_ram_release


def _engine_recipe(model_dir, create_cfg=None, device_idx=0):
    """Resolve immutable create inputs without entering a cache critical section."""
    lib = qfe.load_lib()
    # the library THIS process loaded, not the resolver's current answer: a newer pair marked mid-session must not
    # split one model's cache entry (the lookup also re-hashed both files every time)
    ckey = (qfe.loaded_so_path(), model_dir, "svdq", int(device_idx),
            json.dumps(create_cfg or {}, sort_keys=True))
    return lib, ckey, (qfe.library_identity(lib), ckey), create_cfg


def _prepare_params(model_dir, create_cfg, device_idx, api_key=None):
    """Build retained create params only after both engine caches miss. api_key: the loader field's key, or None."""
    key, surl = _read_auth(api_key)
    cfg = dict(create_cfg or {})
    if key:
        cfg["api_key"] = key
        cfg["server_url"] = surl
    return qfe.make_create_params(model_dir=model_dir, model_backend="svdq",
                                  device_idx=int(device_idx), config_json=cfg or None)


def _get_or_prepare_entry(lib, ckey, prepared_key, model_dir, create_cfg, device_idx, api_key=None):
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

    params = _prepare_params(model_dir, create_cfg, device_idx, api_key)
    candidate = qfmp.QFPreparedEntry(lib, int(params.device_idx), create_params=params, api_key=api_key)
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
    """Single-flight cold create under the prepared entry's lock."""
    acquisition = qfmp._current_engine_cache_acquisition()
    if acquisition is None:
        raise qfe.NativeContractUnavailable(
            "QuantFunc engine materialization requires a QFLazyEngine consumer acquisition")
    # Validate weak-reference support before native creation; publication itself
    # must not discover an unpinnable consumer after allocating a live handle.
    weakref.ref(acquisition.consumer)
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
        # Creation runs only for a lazy engine bound as this entry's materializer: a direct factory caller
        # cannot bypass the common seam.
        if acquisition.consumer not in entry._materializers:
            raise qfe.NativeContractUnavailable(
                "QuantFunc cold creation requires a live ComfyUI materializer binding")
        _sweep_dead_pipelines(ckey)
        # The exact configured-Prepared native capacity is retained on this
        # identity, never re-estimated from paths at create time. Its authority
        # is VRAM. It has ONE second, approximate use: the host-RAM release
        # (_install_host_ram_release) reads a dead pipeline's capacity as the
        # size of its host copy, which it matches within a few percent
        # (measured per family in the commit that added that release).
        try:
            eng = qfe.QFEngineHandle.create(lib, create_params=entry.create_params,
                                           capacity_bytes=int(entry.capacity_bytes),
                                           prepared_resource=entry.resource)
        except BaseException:
            # A refused create (its key refused, say) retires this identity, so the next run prepares again with the
            # key it carries then (the loader field's, or config.json's) instead of sending this recipe's key again.
            with _ENGINE_IDENTITY_LOCK:
                prepared_key = (qfe.library_identity(lib), ckey)
                if _PREPARED_CACHE.get(prepared_key) is entry:
                    _PREPARED_CACHE.pop(prepared_key, None)
            entry.retire_materialized()
            raise
        eng._qf_resource_adapters = entry._qf_resource_adapters
        eng.applied_api_key = entry.api_key   # created with the key its recipe carries
        with _ENGINE_IDENTITY_LOCK:
            _PIPELINE_CACHE[ckey] = eng
            _pin_pipeline_consumer_locked(ckey)
        return eng, ckey


def _get_engine(model_dir, create_cfg=None, device_idx=0, api_key=None):
    """Create (or reuse) the engine for a model PACKAGE dir. The native library path is
    resolved internally (resolve_so_path — NEVER a node input, #vuln). Create is MINIMAL:
    a PREQUANT svdq package carries its own layout/precision in its metadata; anything
    supplied on top competes with it and loses. The transformer weights live INSIDE the
    package (engine loads model_dir/transformer[_2]/ directly — no path override).
    create_cfg carries the per-family create keys only: never a LoRA set or a session setting, so
    every setting of one model's weights shares one cached pipeline. api_key (the loader field's key, or None) is not a
    create key either: it signs in the create of a missed pipeline, and QFLazyEngine switches a cached one in place."""
    # [session-knobs] the session-knob-≠-create-key guard is sealed INSIDE
    # qf_engine.create_pipeline (the real quantfunc_create boundary — construction-enforced,
    # unbypassable by a future direct caller), not duplicated here (one truth source).
    # Cache lookup/publication is atomic; prepare and create never run under the
    # global cache lock (create is single-flight under the prepared entry's lock).
    lib, ckey, prepared_key, create_cfg = _engine_recipe(model_dir, create_cfg, device_idx)
    entry = _get_or_prepare_entry(lib, ckey, prepared_key, model_dir, create_cfg, device_idx, api_key)
    if isinstance(entry, qfe.QFEngineHandle):
        return entry, ckey
    if qfe.FACTORY_PREPARE_ONLY.get():
        return entry, ckey
    return _materialize_engine(entry, lib, ckey)


if _IMPORT_OK:
    # ── family REGISTRY: assembled from the per-family modules. Family LOGIC lives in the family
    #    modules; what remains here is the shared NODE SURFACE — the transformer FILE
    #    dropdown + model_type (the LoRA node applies to the model it is wired to). That
    #    surface is family-neutral as long as a new family fits the "one transformer file +
    #    shipped config bundle" shape; one needing a NEW input must extend INPUT_TYPES here, so
    #    "add a family = one module + one _FAMILY_MODULES line" holds for the common case, not
    #    unconditionally. ──
    # Each module owns ONE model family end to end (its comfy model subclass, its builder, and its
    # detection rule) and exposes exactly three names: FAMILY / matches() / register(deps).
    # Adding a family = write qf_<name>_modelpatcher.py + add it to _FAMILY_MODULES. No edit to the
    # node, the dispatch or the detection lives here, so families cannot bleed into each other.
    _FAMILY_MODULES = ("qf_ltx_modelpatcher", "qf_h3_modelpatcher", "qf_krea2_modelpatcher",
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
        deps = {"get_engine": _get_engine, "bind_pipeline_model": _bind_pipeline_model, "read_auth": _read_auth}
        for mod_name in _FAMILY_MODULES:
            try:
                mod = importlib.import_module("." + mod_name, __name__)
                _FAMILY_BUILDERS[mod.FAMILY] = mod.register(deps)
                _FAMILY_MATCHERS.append((mod.FAMILY, mod.matches))
            except Exception as exc:  # noqa: BLE001 - never break plugin import
                _log.warning("[qf_native] family module %s not registered: %s", mod_name, ascii(exc))

    _register_families()



    def _run_family_load(expect_family, transformer1, model_config=None, pinned_memory=False):
        """The SHARED loader core behind the per-family nodes (user 2026-08-21 pivot). The loaders show no
        model_config choice (each family ships ONE preset: _family_preset); `model_config` is only a saved workflow's
        value of the retired widget (a hidden input). It is honoured when it names this family's preset and refused
        otherwise, naming what the plugin ships. The family guard below is defense-in-depth against a preset dir whose
        manifest family changed between the listing and the read. Returns the family builder's result AS-IS."""
        sig = _loader_sig(expect_family, transformer1, model_config, pinned_memory)   # before model_config resolves
        if model_config is None:
            model_config = _family_preset(expect_family)
        else:
            shipped = _shipped_presets(expect_family)
            if model_config not in shipped:
                _raise_if_a_manifest_is_unreadable(expect_family)
                raise RuntimeError(
                    f"qf_native: this workflow was saved with model_config {model_config!r}, which is not a "
                    f"{expect_family} model config of this plugin (it ships: {', '.join(shipped) or 'none'}). Omit "
                    f"model_config (the loader picks its model config itself) or name the one it ships.")
        bundle_dir, manifest = _load_model_config(model_config)
        family = str(manifest["family"])
        if family != expect_family:
            raise RuntimeError(
                f"qf_native: model_config '{model_config}' declares family '{family}' but this "
                f"loader node is the '{expect_family}' loader (the preset dir changed while it was "
                f"being read).")
        builder = _FAMILY_BUILDERS.get(family)
        if builder is None:
            raise RuntimeError(
                f"qf_native: model_config '{model_config}' routes to family '{family}', which "
                f"has no registered native seam in this install (available: "
                f"{sorted(_FAMILY_BUILDERS)}). An import of the seam module probably failed "
                f"at startup - check the log for a [qf_native] warning.")
        xfm1 = _resolve_transformer(transformer1)
        # file_hints validation (DATA-driven; the manifest names what its transformers look
        # like). DESIGN BOUNDARY, recorded deliberately: comfy combo values must be the REAL
        # relative filenames (they resolve through folder_paths) and INPUT_TYPES is rendered
        # statically — so the transformer dropdown lists every file, not only this family's.
        # The correspondence contract is therefore enforced HERE, fail-loud at load, with the
        # expected patterns named.
        from .qf_file_hints import name_matches_hints  # one predicate, pinned by tests/published_names_test.py
        pats = (manifest.get("file_hints") or {}).get("transformer1") or []
        if pats and not name_matches_hints(transformer1, pats):
            raise RuntimeError(
                f"qf_native: transformer1={transformer1!r} does not look like a '{model_config}' "
                f"transformer1 weight (expected a name matching {pats}). Pick a file with such a "
                f"name, or use the loader for this file's model.")
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
        token = qfmp.LOADER_SIG.set(sig)   # family_build stamps it on every model it makes (_PIPELINE_SIGS)
        try:
            out = builder(transformer1_path=xfm1, bundle_dir=bundle_dir, pinned_memory=bool(pinned_memory))
        finally:
            qfmp.LOADER_SIG.reset(token)
        # [cache/sparse surface REMOVED, user 2026-08-29 「移除所有loader的cache以及
        # 稀疏入口 整体默认不生效」] The per-model set_step_cache/set_block_cache/
        # set_sparse arming that lived here is GONE with the loader widgets — the
        # mixin defaults (0.0/0.0/1.0) already omit every begin key, so the engine
        # paths are byte-identical without any call. Re-enabling is a plugin-side
        # revert of THIS commit (widgets + this arming loop); the engine session
        # dials are untouched and stay available.
        return out

    # [cache surface RE-ENABLED, user 2026-08-31 「step cache 以及 fbcache 的开关重新开启」]
    # The step_cache + block_cache widgets
    # restored to the video loaders (LTX/H3) — the two loader widgets + their arming
    # loop that the 2026-08-29 removal dropped. SPARSE is deliberately NOT re-enabled
    # (the user named only the two caches). Both are RUNTIME SESSION knobs (mixin
    # set_step_cache/set_block_cache → residency_opts begin keys; 0.0 = OFF = byte-identical,
    # no create key, no pipeline rebuild on a widget change).
    _STEP_CACHE_INPUT = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                         "tooltip": "Speed-up that reuses earlier work while the result is barely changing. 0 (default) = off. "
                                    "Higher values are faster but can move the result away from the full render; "
                                    "0.02-0.05 is typical. Takes effect on the next run."})
    _BLOCK_CACHE_INPUT = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                          "tooltip": "A second speed-up that reuses work inside each pass when little changes. 0 (default) = "
                                     "off. 0.05-0.12 is typical; higher is faster but can lose detail. Can be combined "
                                     "with step_cache. Takes effect on the next run."})
    # [quality_enhance — user 2026-09-25] ONE switch on the four QuantFunc loaders (H3, LTX-2.5, Krea-2, Qwen-Image-2.1), the same
    # on every GPU: OFF (default) = the engine's faster default, ON = maximum quality. Every session sends only the engine's
    # switch `video_enhance`; what each state does, per model family, is ENGINE law (user rule 2026-09-19: the plugin carries no
    # implementation detail). It is a session knob: the create config never depends on it, so toggling it never rebuilds or
    # reloads the pipeline. The tooltip states the measured trade of OFF against ON: the scene is kept, details can move.
    _QUALITY_ENHANCE_INPUT = ("BOOLEAN", {"default": False,
                              "tooltip": "OFF (default): faster; the subject and scene stay the same, but details such as poses, "
                                         "faces or small objects can differ. ON: the highest quality, a little slower. Takes "
                                         "effect on the next run."})
    # [pinned_memory — user 2026-09-26 「pin能用 透出个开关让用户选择开启」] ONE load-time switch on the four loaders. ON sends the
    # engine's use_pinned_memory create key: a CREATE key, not a session knob, so its two states are two cached pipelines and
    # changing it reloads the model (never a faked runtime toggle). The engine then keeps a model's host copy in page-locked
    # memory when the host has room for it; it turns pinned memory on for the whole process and has no call that turns it off,
    # so once any loader has turned it on, later loads can pin too until restart. Defaults (user 2026-09-26, measured on the
    # final engine): ON for LTX-2.5 — with little VRAM it moves the most model data (7.5 GiB free: 0.675 -> 0.309 s/step), and
    # the plugin before this switch always turned it on for LTX-2.5, so OFF there would be a regression; OFF for the others
    # (the user's no-pin default). It is each loader's LAST optional input: widget values are stored by position, so saved
    # workflows keep their slots; load() takes the same default, so an API prompt without the input gets it too.
    def _pinned_memory_input(default_on):
        return ("BOOLEAN", {"default": default_on, "tooltip": (
            "ON (default for LTX-2.5, which moves the most model data on graphics cards with little VRAM): faster on such "
            "cards. " if default_on else
            "OFF (default): the model is kept in ordinary system memory. ON: faster on graphics cards with little VRAM. ") +
            "When enough RAM is free, the model is kept in locked system memory, which the rest of the PC cannot use while "
            "the model is loaded; on a PC with little RAM this can make the system unstable. " +
            ("OFF: the model is kept in ordinary system memory. " if default_on else "") +
            "Changing it reloads the model. Once a QuantFunc loader has turned it on (the LTX-2.5 loader does by default), "
            "it stays on for every model until ComfyUI restarts."})
    _PINNED_MEMORY_INPUT = _pinned_memory_input(False)       # MiniMax-H3, Krea-2, Qwen-Image-2.1
    _PINNED_MEMORY_INPUT_LTX = _pinned_memory_input(True)    # LTX-2.5
    # Saved workflows: the published loaders had this same switch in this same widget slot, so their workflows open unchanged.
    # A workflow saved with the unpublished quality dropdown (0.0.07 release candidate) carries `quality`: an API-format prompt by
    # NAME, declared hidden so ComfyUI hands it to load() (an undeclared key is dropped silently) and mapped best_quality -> ON,
    # any other value -> OFF. A UI workflow stores widget values by POSITION, so its dropdown string lands in this switch's slot,
    # where ComfyUI's BOOLEAN conversion (bool(value)) turns it ON before any node code runs.
    _QUALITY_LEGACY_HIDDEN = {"quality": ("STRING", {})}

    def _quality_enhance_on(quality_enhance=None, quality=None):
        """This run's switch. It wins when the prompt has it; else a legacy dropdown value (best_quality -> ON, anything else ->
        OFF); else OFF (the default)."""
        return bool(quality_enhance) if quality_enhance is not None else quality == "best_quality"

    def _model_config_input(family):
        """The loaders' model_config input: the FIRST optional input, right after the transformer (widget index 1, where the
        dropdown always was). Each family ships one preset, and a setting with one choice is not a setting (user 2026-09-24
        「model_config也不是设置啊 就一个选项没意义啊」), so the widget is HIDDEN: ComfyUI's input option `hidden` keeps the widget, invisible, in its slot, and `socketless` = no input dot.
        Widget values are stored by POSITION, so every saved workflow keeps each value in its place, and a saved preset
        value is honoured by _run_family_load. Should a family ever ship a second preset, the dropdown shows again, so the
        user chooses and the loader never guesses."""
        presets = _shipped_presets(family) or [_NO_CFG_HINT]
        opts = {"default": presets[0], "tooltip": "The model config that matches the chosen model file."}
        if len(presets) == 1:
            opts.update(hidden=True, socketless=True)
        return presets, opts


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

    # [sol-tau dial 2026-08-31] the ONE user-facing attention dial (user "就一个就好"). Applies to
    # the flash/sage backends — the engine applies it regardless of attention_backend (it is NOT
    # tied to a removed backend choice). Same
    # runtime-session-knob class as step_cache/sparse: re-sent each run, no rebuild; 1.0 default is
    # omitted (older engines refuse unknown keys loud) and the engine resets an absent key to 1.0.
    _SOL_TAU_INPUT = ("FLOAT", {"default": 1.0, "min": 0.02, "max": 1.0, "step": 0.01,
                     "tooltip": "Attention speed-up. 1.0 (default) turns it off. Lower values are faster and give up "
                                "some quality: 0.15-0.2 is a good start, below 0.1 check the result carefully, 0.3-0.5 "
                                "keeps more quality. Takes effect on the next run."})

    def _arm_session_caches(_mm, step_cache, block_cache):
        """Arm the step-cache and block-cache session knobs on a loaded model.
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
    # SM-GATED: SM80+ offers the full set; SM75 (Turing) has NO sage backend and NO
    # flash_attn build, so only fp16_native is valid there. The widget VALUE is a
    # display name; _attn_backend_to_engine maps it to the engine's comp_opts string
    # ("fp16_native" -> "native"). "auto" = the engine's per-SM resolution (default), and
    # is passed through so the user's choice is always the single source of truth.
    # [a backend REMOVED as a user-facing choice — user 2026-09-13] one engine-internal backend is
    # no longer offered in the dropdown. "auto" is unaffected (the engine may still pick it
    # internally per-SM); only the explicit user choice is gone.
    _ATTN_BACKEND_SM80PLUS = ["auto", "flash", "sage", "fp16_native"]
    _ATTN_BACKEND_SM75 = ["fp16_native"]

    def _attn_backend_choices():
        choices = _ATTN_BACKEND_SM80PLUS
        try:
            import torch
            if torch.cuda.is_available():
                maj, _min = torch.cuda.get_device_capability(0)
                if maj < 8:  # SM75 Turing (sm_7x): no sage, no flash_attn
                    choices = _ATTN_BACKEND_SM75
        except Exception:  # noqa: BLE001 - no torch/CUDA at import -> assume modern; engine validates
            pass
        return choices

    def _attn_backend_input(default="auto"):
        choices = _attn_backend_choices()
        # per-loader default; falls back to the first valid choice when the requested
        # default isn't offered on this SM (e.g. H3 wants 'flash' but SM75 has no flash).
        d = default if default in choices else choices[0]
        return (choices, {"default": d,
                "tooltip": "auto (default) picks the best setting for your GPU. Try another setting only if a "
                           "result looks wrong or a run fails on your GPU. Takes effect on the next run."})

    def _attn_backend_to_engine(v):
        # widget display name -> engine comp_opts attention_backend string
        return "native" if v == "fp16_native" else (v or "auto")

    class QuantFuncLTXLoader:
        """LTX-2 loader — single MODEL output (single-expert family)."""

        QF_FAMILY = "ltx2"   # the family its create inputs belong to (_loader_sig, _pending_loader_sigs)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "transformer": (_transformer_choices(),
                                {"tooltip": "The QuantFunc LTX-2 model file in models/diffusion_models."}),
            }, "optional": {
                "model_config": _model_config_input("ltx2"),   # hidden; widget index 1 (see _model_config_input)
                "attention_backend": _attn_backend_input(),
                "sol_tau": _SOL_TAU_INPUT,
                "quality_enhance": _QUALITY_ENHANCE_INPUT,
                "step_cache": _STEP_CACHE_INPUT,
                "block_cache": _BLOCK_CACHE_INPUT,
                "pinned_memory": _PINNED_MEMORY_INPUT_LTX,
            }, "hidden": dict(_QUALITY_LEGACY_HIDDEN)}

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = ("Loads a QuantFunc LTX-2 model (video with sound) for ComfyUI's standard samplers - use it in place "
                       "of the usual diffusion-model loader. For image-to-video, add the official LTXVImgToVideoInplace node on "
                       "the latent input. " + _COMMON_LIMITS)

        def load(self, transformer, model_config=None,
                 attention_backend="auto", sol_tau=1.0, quality_enhance=None, step_cache=0.0, block_cache=0.0, quality=None,
                 pinned_memory=True):   # LTX-2.5: ON by default (_PINNED_MEMORY_INPUT_LTX)
            # NO aux file widgets and NO image socket (user 2026-08-22 "只保留
            # transformer/block/model_config … 只关注latent"): i2v is the workflow's own latent
            # conditioning (LTXVImgToVideoInplace).
            _p = _run_family_load(self.QF_FAMILY, transformer, model_config, pinned_memory)
            _mm = getattr(_p, "model", None)
            if _mm is not None and hasattr(_mm, "set_attn_backend"):
                _mm.set_attn_backend(_attn_backend_to_engine(attention_backend))
            if _mm is not None and hasattr(_mm, "set_sol_tau"):
                _mm.set_sol_tau(sol_tau)
            _mm.set_video_enhance(_quality_enhance_on(quality_enhance, quality))   # mandatory + unguarded: a patcher without it is a wiring error
            _arm_session_caches(_mm, step_cache, block_cache)
            return (_p,)

    class QuantFuncKrea2Loader:
        """Krea-2 Turbo loader (svdq, denoise_only, t2i) — the first IMAGE family on the
        native seam: one MODEL a stock KSampler drives with latents; CLIP (type krea2) +
        VAE + sampler stay comfy-owned (drop-in for the official UNETLoader slot)."""

        QF_FAMILY = "krea2"   # the family its create inputs belong to (_loader_sig, _pending_loader_sigs)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "transformer": (_transformer_choices(),
                                {"tooltip": "The QuantFunc Krea-2 Turbo model file in models/diffusion_models."}),
            }, "optional": {
                "model_config": _model_config_input("krea2"),   # hidden; widget index 1 (see _model_config_input)
                "attention_backend": _attn_backend_input(),
                "quality_enhance": _QUALITY_ENHANCE_INPUT,
                "pinned_memory": _PINNED_MEMORY_INPUT,
            }, "hidden": dict(_QUALITY_LEGACY_HIDDEN)}

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = ("Loads a QuantFunc Krea-2 Turbo model (text-to-image) for ComfyUI's standard samplers - use it in "
                       "place of the usual diffusion-model loader. " + _COMMON_LIMITS)

        def load(self, transformer, model_config=None, attention_backend="auto",
                 quality_enhance=None, quality=None, pinned_memory=False):
            # [runtime dials] backend + quality_enhance are SESSION knobs (NOT create keys — a widget change never re-keys the
            # engine = no rebuild).
            _p = _run_family_load(self.QF_FAMILY, transformer, model_config, pinned_memory)
            _mm = getattr(_p, "model", None)
            if _mm is not None and hasattr(_mm, "set_attn_backend"):
                _mm.set_attn_backend(_attn_backend_to_engine(attention_backend))
            _mm.set_video_enhance(_quality_enhance_on(quality_enhance, quality))   # mandatory + unguarded: a patcher without it is a wiring error
            return (_p,)

    class QuantFuncQwenImage21Loader:
        """Qwen-Image-2.1 loader (svdq, denoise_only, text-to-image + image edit): one MODEL a stock
        KSampler drives with latents; CLIP (type qwen_image, TextEncodeQwenImage21) + VAE + sampler stay
        comfy-owned (drop-in for the official UNETLoader slot). Edit = TextEncodeQwenImage21 with a VAE and
        reference images: the references ride every step into the engine (quantfunc_denoise_step_refs)."""

        QF_FAMILY = "qwenimage21"   # the family its create inputs belong to (_loader_sig, _pending_loader_sigs)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "transformer": (_transformer_choices(),
                                {"tooltip": "The QuantFunc Qwen-Image-2.1 model file in models/diffusion_models."}),
            }, "optional": {
                "model_config": _model_config_input("qwenimage21"),   # hidden; widget index 1 (see _model_config_input)
                "attention_backend": _attn_backend_input(),
                "quality_enhance": _QUALITY_ENHANCE_INPUT,
                "pinned_memory": _PINNED_MEMORY_INPUT,
            }, "hidden": dict(_QUALITY_LEGACY_HIDDEN)}

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = ("Loads a QuantFunc Qwen-Image-2.1 model for ComfyUI's standard samplers: text-to-image, and image "
                       "editing with TextEncodeQwenImage21 reference images (plus its VAE). Transparent images: VAE Decode + "
                       "Save Image keep the transparency. " + _COMMON_LIMITS)

        def load(self, transformer, model_config=None, attention_backend="auto", quality_enhance=None, quality=None,
                 pinned_memory=False):
            # [runtime dials] backend + quality_enhance are SESSION knobs (NOT create keys — a widget change never re-keys the engine =
            # no rebuild), exactly like the Krea2 node.
            _p = _run_family_load(self.QF_FAMILY, transformer, model_config, pinned_memory)
            _mm = getattr(_p, "model", None)
            if _mm is not None and hasattr(_mm, "set_attn_backend"):
                _mm.set_attn_backend(_attn_backend_to_engine(attention_backend))
            _mm.set_video_enhance(_quality_enhance_on(quality_enhance, quality))   # mandatory + unguarded: a patcher without it is a wiring error
            return (_p,)


    class QuantFuncH3Loader:
        """MiniMax-H3 loader — single MODEL output (single-expert AV family)."""

        QF_FAMILY = "minimax-h3"   # the family its create inputs belong to (_loader_sig, _pending_loader_sigs)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "transformer": (_transformer_choices(),
                                {"tooltip": "The QuantFunc MiniMax-H3 model file in models/diffusion_models."}),
            }, "optional": {
                "model_config": _model_config_input("minimax-h3"),   # hidden; widget index 1 (see _model_config_input)
                # [sparse, user 2026-08-25 ONE-number dial; #659 session knob — no rebuild]
                # H3 default = flash: on this model auto picks a backend that is
                # BROKEN at high resolution (blank/NaN — measured
                # 928²/S=31538: attn out absmax 0 → step-1 all-NaN → audio avcodec crash +
                # video blur). flash (fp16) is the verified-clean default; user can still pick
                # auto/sage/native. (Wan/LTX → auto is fine → they keep 'auto'.)
                "attention_backend": _attn_backend_input("flash"),
                "sol_tau": _SOL_TAU_INPUT,
                "quality_enhance": _QUALITY_ENHANCE_INPUT,
                "audio_enhance": _AUDIO_ENHANCE_INPUT,
                "step_cache": _STEP_CACHE_INPUT,
                "block_cache": _BLOCK_CACHE_INPUT,
                "allow_partial_denoise": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Opt in to split/trimmed sigma schedules for intentional H3 double-sampling workflows.",
                }),
                "pinned_memory": _PINNED_MEMORY_INPUT,
            }, "hidden": dict(_QUALITY_LEGACY_HIDDEN)}

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = ("Loads a QuantFunc MiniMax-H3 model (video with sound) for ComfyUI's standard samplers - use it in "
                       "place of the usual diffusion-model loader. " + _COMMON_LIMITS)

        def load(self, transformer, model_config=None,
                 attention_backend="flash", sol_tau=1.0, quality_enhance=None, audio_enhance=False,
                 step_cache=0.0, block_cache=0.0, allow_partial_denoise=False, quality=None,
                 pinned_memory=False):  # H3: flash default (auto->sage is broken)
            _p = _run_family_load(self.QF_FAMILY, transformer, model_config, pinned_memory)
            _mm = getattr(_p, "model", None)
            if _mm is not None and hasattr(_mm, "set_attn_backend"):
                _mm.set_attn_backend(_attn_backend_to_engine(attention_backend))
            if _mm is not None and hasattr(_mm, "set_sol_tau"):
                _mm.set_sol_tau(sol_tau)
            _mm.set_video_enhance(_quality_enhance_on(quality_enhance, quality))   # mandatory + unguarded: a patcher without it is a wiring error
            if _mm is not None and hasattr(_mm, "set_audio_enhance"):
                _mm.set_audio_enhance(audio_enhance)
            _mm.set_allow_partial_denoise(bool(allow_partial_denoise))
            _arm_session_caches(_mm, step_cache, block_cache)
            return (_p,)

    class QuantFuncNativeLoRA:
        """Sidecar LoRA for the QuantFunc native loader — MODEL in, MODEL out (LoraLoaderModelOnly
        shape). Chain several to stack them.

        User rule 2026-09-24 「换 LoRA 也不重建」: the pipeline is created WITHOUT LoRA, so every LoRA set of one
        model shares it, and the chained set is applied in place before each run (one declarative
        quantfunc_pipeline_update {"lora": [...]}; QFLazyEngine._apply_runtime_lora). The create is deferred
        (QFLazyEngine), so a chain of N nodes builds at most ONE pipeline.
        Comfy-level patches applied upstream (ModelSampling*, set_model_* …) are TRANSPLANTED onto
        the rebuilt patcher, so this node may sit anywhere in the chain.
        """
        @classmethod
        def INPUT_TYPES(cls):
            # NO target widget (user directive 2026-08-22): every family in this release is single-expert, so the
            # target is always "all".
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
        DESCRIPTION = ("Applies a LoRA to a QuantFunc model (connect it after a QuantFunc loader; chain "
                       "several to stack them). Changing or removing a LoRA keeps the loaded model (no reload).")

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
            # the kinds no key rename turns into a plain (A,B) LoRA: refused here with the same words as the engine and
            # scripts/qf_lora_convert.py (never sent to the converter, which refuses them too)
            for kind, marks in (("LyCORIS (LoHa/LoKr)", (".hada_", ".lokr_")),
                                ("DoRA", (".dora_scale", "lora_magnitude_vector")),
                                ("OFT/BOFT", (".oft_", ".boft_"))):
                if any(m in k for k in keys for m in marks):
                    raise RuntimeError(
                        f"QuantFuncNativeLoRA: this is {'an' if kind[0] in 'AEIOU' else 'a'} {kind} file, which is not supported - the native "
                        "path does not load it and converting it does not help. Merge/re-export it to a "
                        "standard (A,B) LoRA first.")
            if any(k.startswith(("lora_unet_", "lora_transformer_", "lora_te_",
                                 "lora_te1_", "lora_te2_")) for k in keys):
                raise RuntimeError(
                    "QuantFuncNativeLoRA: kohya/ai-toolkit-format LoRA detected. The native "
                    "path adapts ONE format (diffusers/PEFT canonical) - convert once with:\n"
                    f"  python3 {conv} --in '{path}' --out '<same-dir>/<name>-diff.safetensors'\n"
                    "then pick the converted file in this node.")
            if not any(".lora_A." in k or ".lora_B." in k or ".lora_down." in k
                       or ".lora_up." in k for k in keys):
                raise RuntimeError(
                    "QuantFuncNativeLoRA: no recognizable LoRA keys (lora_A/lora_B/"
                    "lora_down/lora_up) in this file - not a LoRA, or an unsupported "
                    f"format. If it is a LoRA, convert it: python3 {conv} --in ... --out ...")

        def apply(self, model, lora_name, strength):
            rebuild = qfmp.rebuild_of(model)
            if rebuild is None:
                raise RuntimeError(
                    "QuantFuncNativeLoRA: this MODEL is not a QuantFunc native model - wire it "
                    "downstream of the QuantFunc Native Loader. (For a stock comfy model use the "
                    "built-in LoraLoaderModelOnly instead.)")
            # The ONE-FORMAT refusal (mirror of the engine's E1 arm) applies to every family in this release.
            self._refuse_foreign_lora_format(_resolve_lora(lora_name))
            stack = qfmp.lora_stack_of(model)
            stack.append({"path": _resolve_lora(lora_name), "scale": float(strength),
                          "target": "all"})
            rebuilt = rebuild(stack)
            # CR regression fix: carry the UPSTREAM comfy state (ModelSampling* object patches,
            # set_model_* options, callbacks/wrappers/hooks) onto the re-created patcher, so a
            # LoRA node placed after a ModelSampling node cannot silently drop its shift.
            return (rebuilt.adopt_comfy_state_from(model),)

    # merge into (not replace) the mappings — matches the real plugin's multi-file NODE_CLASS_MAPPINGS.update
    NODE_CLASS_MAPPINGS.update({"QuantFuncLTXLoader": QuantFuncLTXLoader,
                                "QuantFuncH3Loader": QuantFuncH3Loader,
                                "QuantFuncKrea2Loader": QuantFuncKrea2Loader,
                                "QuantFuncQwenImage21Loader": QuantFuncQwenImage21Loader,
                                "QuantFuncNativeLoRA": QuantFuncNativeLoRA})
    NODE_DISPLAY_NAME_MAPPINGS.update({
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
    except Exception as _rc_exc:  # noqa: BLE001 - the self-check must never break plugin import
        _log.debug("[qf_native] reject-list self-check skipped: %s", ascii(_rc_exc))
    # (R7: the old single-node "QuantFuncNativeLoader" display entry is GONE with the class —
    # a display mapping for an unregistered class is dead weight; the three per-family loaders
    # register their display names beside their class mappings above.)
    NODE_DISPLAY_NAME_MAPPINGS.update({"QuantFuncNativeLoRA": "QuantFunc Native LoRA"})

    def _serve_api_key_route():
        """On a SERVING ComfyUI (its PromptServer exists while custom nodes load), add the route the loaders' API key
        field hands its key to (qf_api_key). Without the route the field fails the queue loudly: it never falls back to
        queuing the key itself."""
        srv = getattr(getattr(sys.modules.get("server"), "PromptServer", None), "instance", None)
        if srv is not None:
            qf_api_key.register_route(srv.routes, _resolve_keyfile)

    try:
        _serve_api_key_route()
    except Exception as _qf_route_exc:  # noqa: BLE001 - never break plugin import
        _log.warning("[qf_native] API key route not added (a loader's API key field then refuses to queue): %s",
                     ascii(_qf_route_exc))

    # Engine library install / update (option C): on a daemon thread, so node registration never waits. A loader
    # run before it finishes says plainly "still downloading" or why it failed (qf_engine.engine_install_status).
    # Only a SERVING ComfyUI installs (its PromptServer exists while custom nodes load): importing the plugin in a
    # test or a script never downloads an engine into the package.
    try:
        import sys as _qf_sys
        _qf_srv = getattr(_qf_sys.modules.get("server"), "PromptServer", None)
        if getattr(_qf_srv, "instance", None) is not None:
            qfe.start_engine_install(_comfy_device_index())
    except Exception as _qf_install_exc:  # noqa: BLE001 - installing must never break plugin import
        _log.warning("[qf_native] engine install not started: %s", ascii(_qf_install_exc))


# ── Host RAM: a dropped model's pipeline releases its backup when ComfyUI runs short (see _install_host_ram_release)
if _IMPORT_OK:
    try:
        _install_host_ram_release()
    except Exception as _qf_hr_exc:  # noqa: BLE001 - never break plugin import
        _log.warning("[qf_native] host-RAM release hook not installed: %s", ascii(_qf_hr_exc))


# ── QuantFunc LTX-2.5 AV ancestral-sampler audio fix ─────────────────────────────
# The engine's STATELESS flow-match forward requires a non-re-noised trajectory; comfy's
# ancestral samplers (euler_ancestral auto-routes to *_RF for CONST/flow models) re-noise x
# every step, which collapses the low-dim AV AUDIO lane to silence. Neutralize the audio-lane
# re-noise for QF LTX-2.5 AV models ONLY (video ancestral stochasticity preserved). Fully
# guarded — a failure must never break plugin import.
try:
    from . import qf_ltx_ancestral_audio_fix as _qf_ltx_afix
    qfe.logger(_qf_ltx_afix.__name__)   # its warnings: console-safe
    _qf_ltx_afix.install(guard=qfe.console_safe_errors)
except Exception as _qf_ltx_afix_exc:  # noqa: BLE001
    _log.warning("[qf_native] LTX-2.5 AV audio fix not installed: %s", ascii(_qf_ltx_afix_exc))


# ── Engine log detail, and the API key field ──────────────────────────────────
# One HIDDEN `log_level` input on EVERY QuantFunc loader: never
# shown, so users do not choose it; a prompt that carries it (the test harness's) still sets it. Default warning
# (warnings and errors only). The value is handed to qf_engine before the loader runs and applied to the
# engine library as soon as it is (or once it gets) loaded; asking never loads it. Process-wide.
# One HIDDEN `api_key` input on every QuantFunc loader too (user 2026-09-27: a valid key in the loader's field wins, and
# config.json is then not read): web/quantfunc_api_key.js draws the field on exactly the QuantFunc nodes that declare it,
# and qf_api_key.add_api_key_input resolves it and publishes the key (qfmp.LOADER_API_KEY) while the loader runs.
# This runs LAST, after every NODE_CLASS_MAPPINGS registration above, so no loader is missed (tests/log_level_input_test.py
# checks that no registration comes after it). Fully guarded: it must never break plugin import.
try:
    from . import qf_engine as _qf_ll_engine
    from .qf_log_level import add_log_level_input as _qf_add_log_level
    for _qf_name, _qf_cls in list(NODE_CLASS_MAPPINGS.items()):
        if _qf_name.startswith("QuantFunc") and _qf_name.endswith("Loader"):
            qf_api_key.add_api_key_input(_qf_cls, qfmp.LOADER_API_KEY)
            _qf_add_log_level(_qf_cls, _qf_ll_engine.set_log_level)
except Exception as _qf_ll_exc:  # noqa: BLE001
    _log.warning("[qf_native] hidden loader inputs (api_key, log_level) not attached: %s", ascii(_qf_ll_exc))


# -- The console-safe boundary (#738) ---------------------------------------------------------------------------------
# ComfyUI logs an uncaught node exception and its traceback to its strict console; a character the code page lacks (a
# Chinese username in a path, on an English Windows) made that logging raise a second exception inside ComfyUI's error
# handling. Every node's FUNCTION rewrites an exception leaving it console-safe (qf_engine.console_safe_nodes), and the
# classes ComfyUI calls into carry @qfe.console_safe_methods. This runs LAST, after every registration and the log-level
# wrap, so it is the outermost layer and no node skips it (tests/text_encoding_test.py, the boundary arm).
if qfe is not None:
    qfe.console_safe_nodes(NODE_CLASS_MAPPINGS)
