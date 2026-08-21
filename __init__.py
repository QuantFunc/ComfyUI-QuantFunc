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


def _model_config_choices():
    """The OFFICIAL model-config presets shipped with the plugin — one subdir of configs/ per
    model, each carrying a qf_native.json manifest (family routing + shape) beside the arch/VAE
    config JSONs. The dropdown lists the DIRECTORY NAMES, so adding a model preset = drop in a
    config dir; no code change (configs are data, not code)."""
    try:
        return sorted(d for d in os.listdir(_CONFIGS_DIR)
                      if os.path.isfile(os.path.join(_CONFIGS_DIR, d, "qf_native.json")))                or [_NO_CFG_HINT]
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


def _may_release_handle(ckey, requester):
    """May `requester`'s wrapper DESTROY ckey's shared handle? Only when no OTHER live model is
    bound to it — a sibling loader node on the same package would otherwise be left holding a
    destroyed handle (NULL pipeline at its next denoise) while still reporting its backup to
    comfy's ledger. On grant, the cache entry is dropped (the caller destroys the handle; the
    next load re-creates from disk). The requester itself STAYS bound: if it re-materializes
    later it becomes a live consumer of the ckey's NEXT handle."""
    for m in _live_pipeline_models(ckey):
        if m is not requester:
            return False
    _PIPELINE_CACHE.pop(ckey, None)
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
        refs = _PIPELINE_MODELS.get(k)
        if not refs or any(r() is not None for r in refs):
            continue   # unbound (load in flight) or ANY consumer still live → do NOT destroy (UAF-safe)
        eng = _PIPELINE_CACHE.pop(k, None)
        _PIPELINE_MODELS.pop(k, None)
        if eng is not None:
            try:
                eng.destroy()   # closes any session + quantfunc_destroy; idempotent (pipeline→None)
                print("[qf_native] host-RAM sweep: destroyed a cached pipeline whose QFWanModel was GC'd "
                      "(comfy dropped its patcher) — freed its CPU backup", flush=True)
            except Exception:  # noqa: BLE001
                pass


def _get_engine(model_dir, create_cfg=None, device_idx=0):
    """Create (or reuse) the engine for a model PACKAGE dir. The native library path is
    resolved internally (resolve_so_path — NEVER a node input, #vuln). Create is MINIMAL:
    a PREQUANT svdq package carries its own layout/precision in its metadata; anything
    supplied on top competes with it and loses. The transformer weights live INSIDE the
    package (engine loads model_dir/transformer[_2]/ directly — no path override).
    create_cfg carries the per-family create keys (e.g. wan text_precision) + the
    declarative lora stack from chained QuantFuncNativeLoRA nodes."""
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
    #    dropdowns + model_type + resident_block_count (+ the LoRA node's target combo). That
    #    surface is family-neutral as long as a new family fits the "1-2 transformer files +
    #    shipped config bundle" shape; one needing a NEW input must extend INPUT_TYPES here, so
    #    "add a family = one module + one _FAMILY_MODULES line" holds for the common case, not
    #    unconditionally. ──
    # Each module owns ONE model family end to end (its comfy model subclass, its builder, and its
    # detection rule) and exposes exactly three names: FAMILY / matches() / register(deps).
    # Adding a family = write qf_<name>_modelpatcher.py + add it to _FAMILY_MODULES. No edit to the
    # node, the dispatch or the detection lives here, so families cannot bleed into each other.
    _FAMILY_MODULES = ("qf_wan_modelpatcher", "qf_ltx_modelpatcher", "qf_h3_modelpatcher")
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
                "may_release_handle": _may_release_handle,
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



    class QuantFuncNativeLoader:
        """ONE loader for every QuantFunc native family — like the reference INT8-Fast
        UNetLoaderINTW8A8: you pick a transformer .safetensors FILE from models/diffusion_models
        and a model_type; it returns a native comfy MODEL that a STOCK KSampler drives with
        LATENTS. CLIP + VAE + latent + sampler + video nodes all stay stock comfy nodes — only the
        transformer (denoise) is this engine.

        wan A14B is dual-expert, so there are TWO transformer slots: transformer1 (high-noise) and
        an optional transformer2 (low-noise). Single-expert families (LTX-2.5 / MiniMax-H3) use
        transformer1 only and leave transformer2 = "(none)". LoRAs attach DOWNSTREAM via
        QuantFuncNativeLoRA.
        """

        @classmethod
        def INPUT_TYPES(cls):
            _xfms = _transformer_choices()
            return {"required": {
                "transformer1": (_xfms,
                                 {"tooltip": "The transformer weight .safetensors under "
                                             "models/diffusion_models (wan A14B: the HIGH-noise "
                                             "expert; single-expert LTX-2.5 / H3: the only "
                                             "transformer)."}),
                "model_config": (_model_config_choices(),
                                 {"tooltip": "The OFFICIAL model config for this transformer "
                                             "(shipped with the plugin: arch + VAE geometry + "
                                             "family routing + expected-file naming). Pick the "
                                             "preset matching your weights — a bare .safetensors "
                                             "has no metadata to auto-detect from. "
                                             + _preset_file_expectations()}),
                "resident_block_count": ("INT", {"default": 999, "min": 1, "max": 1024,
                                                 "tooltip": "GPU-resident transformer blocks — the "
                                                            "native seam's ONLY residency knob. The "
                                                            "engine clamps to the model's block "
                                                            "count, so the default keeps every "
                                                            "block resident on a card that fits."}),
            }, "optional": {
                "transformer2": ([_XFM_NONE] + _xfms,
                                 {"tooltip": "wan A14B ONLY: the LOW-noise expert .safetensors. "
                                             "Leave \"(none)\" for single-expert families "
                                             "(LTX-2.5 / MiniMax-H3) — an expert2 there is refused "
                                             "loud."}),
            }}
            # NOTE: NO `so_path` / `keyfile` widgets — the native library + auth keyfile resolve from
            # the package bundle + the PROCESS ENVIRONMENT only, never from workflow JSON (#vuln: a
            # workflow-serializable path into ctypes.CDLL is an arbitrary-code-execution primitive).

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = (
            "Loads a QuantFunc svdq TRANSFORMER .safetensors (wan / LTX-2 / MiniMax-H3) from "
            "models/diffusion_models and exposes it as a native comfy MODEL a STOCK KSampler drives "
            "with LATENTS — only the transformer is swapped in; CLIP, VAE, latent, sampler and "
            "video nodes stay stock (INT8-Fast-aligned: the inference path takes latents only). "
            "wan A14B is dual-expert → set transformer2. Limits: (1) A SINGLE full-range KSampler: a "
            "trimmed/partial denoise (KSamplerAdvanced start_step/last_step, denoise<1) mis-times "
            "the engine's internal schedule and is refused loud. (2) ControlNet is not consumed by "
            "this seam (refused loud). (3) Interrupt stops BETWEEN denoise steps. (4) On Linux a "
            "fail-closed CUDA-toolchain check refuses a torch/.so CUDA-major mismatch; on "
            "Windows/macOS set QF_NATIVE_ALLOW_UNVERIFIED_TOOLCHAIN=1 after confirming they share a "
            "CUDA major.")

        def load(self, transformer1, model_config, resident_block_count=999, transformer2=_XFM_NONE):
            bundle_dir, manifest = _load_model_config(model_config)
            family = str(manifest["family"])
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
            return (builder(transformer1_path=xfm1, transformer2_path=xfm2,
                            resident_block_count=int(resident_block_count),
                            bundle_dir=bundle_dir),)

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
            return {"required": {
                "model": ("MODEL",),
                "lora_name": (_lora_choices(),),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }, "optional": {
                "target": (["all", "high", "low"],
                           {"tooltip": "wan A14B per-expert routing. Single-transformer families "
                                       "(LTX / H3) take 'all'; an explicit high/low there is "
                                       "refused engine-side rather than silently mis-routed."}),
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

        def apply(self, model, lora_name, strength, target="all"):
            rebuild = qfmp.rebuild_of(model)
            if rebuild is None:
                raise RuntimeError(
                    "QuantFuncNativeLoRA: this MODEL is not a QuantFunc native model — wire it "
                    "downstream of the QuantFunc Native Loader. (For a stock comfy model use the "
                    "built-in LoraLoaderModelOnly instead.)")
            stack = qfmp.lora_stack_of(model)
            stack.append({"path": _resolve_lora(lora_name), "scale": float(strength),
                          "target": target})
            rebuilt = rebuild(stack)
            # CR regression fix: carry the UPSTREAM comfy state (ModelSampling* object patches,
            # set_model_* options, callbacks/wrappers/hooks) onto the re-created patcher, so a
            # LoRA node placed after a ModelSampling node cannot silently drop its shift.
            return (rebuilt.adopt_comfy_state_from(model),)

    # merge into (not replace) the mappings — matches the real plugin's multi-file NODE_CLASS_MAPPINGS.update
    NODE_CLASS_MAPPINGS.update({"QuantFuncNativeLoader": QuantFuncNativeLoader,
                                "QuantFuncNativeLoRA": QuantFuncNativeLoRA})

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
    NODE_DISPLAY_NAME_MAPPINGS.update({"QuantFuncNativeLoader": "QuantFunc Native Loader",
                                       "QuantFuncNativeLoRA": "QuantFunc Native LoRA"})


