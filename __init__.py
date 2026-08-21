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
    from .qf_modelpatcher import QFModelPatcher, QFLazyEngine
    _IMPORT_OK = True
except Exception as _exc:  # noqa: BLE001 — never break registration; report loudly
    logging.warning("[qf_native] disabled — a required import failed (ComfyUI API drift?): %r", _exc)
    _IMPORT_OK = False


# ── official-loader UX: scan the STANDARD models/diffusion_models folder ──
# A QuantFunc engine model is a PACKAGE DIRECTORY (model_index.json + transformer[_2]/ with
# the svdq weights + vae/ + text_encoder/ + tokenizer/ [+ scheduler/]) — the engine detects the
# pipeline family from model_index.json and loads transformer[_2]/ itself. Those packages live
# in comfy's OWN models/diffusion_models next to the single-file UNETs (that folder holds both),
# so they are listed from there with DiffusersLoader's walk-for-model_index.json mechanism
# pointed at diffusion_models — no private model folder.
try:
    import folder_paths as _folder_paths
except Exception as _fp_exc:  # noqa: BLE001 — never break registration
    _folder_paths = None
    logging.warning("[qf_native] folder_paths unavailable: %r", _fp_exc)

_NO_MODELS_HINT = "(no QuantFunc model package in models/diffusion_models)"
_NO_LORA_HINT = "(no LoRA in models/loras)"


def _model_roots():
    """The folders scanned for QuantFunc model packages: comfy's models/diffusion_models (where
    these packages are kept — CR raised switching to the `diffusers` key instead; MEASURED, that
    would not change comfy's native UNETLoader dropdown, which already recurses into any package
    dir placed there: with k9b present it lists k9b/transformer/model.safetensors etc. whether or
    not this plugin reads that folder. The listing is a consequence of the on-disk layout, not of
    our scan, so switching our READ key would move nothing off that dropdown while contradicting
    where the models actually live) PLUS models/diffusers for users who keep packages there."""
    roots = []
    for key in ("diffusion_models", "diffusers"):
        try:
            roots.extend(_folder_paths.get_folder_paths(key))
        except Exception:  # noqa: BLE001 — a missing key must not break listing
            pass
    return roots


def _list_quantfunc_packages():
    """Relative paths of every QuantFunc model PACKAGE (a dir holding model_index.json) under
    comfy's models/diffusion_models roots. Single-file .safetensors UNETs are deliberately NOT
    listed: the engine needs the package (arch metadata + its own transformer/vae/TE layout), so
    offering a bare file would be a choice that can only fail."""
    out = []
    if _folder_paths is None:
        return [_NO_MODELS_HINT]
    for root_dir in _model_roots():
        if not os.path.isdir(root_dir):
            continue
        for r, dirs, files in os.walk(root_dir, followlinks=True):
            if "model_index.json" in files:
                out.append(os.path.relpath(r, start=root_dir))
                dirs[:] = []          # a package is a leaf — do not descend into its subdirs
    return sorted(out) or [_NO_MODELS_HINT]


def _resolve_package(name):
    """Resolve a listed package name to its directory — CONFINED to the registered model roots.

    #vuln (CR): a widget value is workflow-serializable, so `name` is UNTRUSTED. comfy's own
    get_full_path_or_raise does this containment for FILES (that is what _resolve_lora and the
    connector dropdown use); there is no folder_paths equivalent for a package DIRECTORY, so the
    containment is done here explicitly: resolve both sides with realpath and require the
    candidate to sit under the root, so "../../.." / an absolute path / a symlink pointing out
    cannot escape models/diffusion_models (or models/diffusers)."""
    if _folder_paths is None:
        raise RuntimeError("qf_native: comfy folder_paths unavailable")
    if name == _NO_MODELS_HINT:
        raise RuntimeError(
            "qf_native: no QuantFunc model package found. Put the model DIRECTORY (the one with "
            "model_index.json + transformer/) under ComfyUI/models/diffusion_models/ and refresh.")
    if os.path.isabs(name):
        raise RuntimeError(f"qf_native: model_name must be a name inside a model folder, not an "
                           f"absolute path ({name!r})")
    for root_dir in _model_roots():
        root = os.path.realpath(root_dir)
        cand = os.path.realpath(os.path.join(root, name))
        if os.path.commonpath([root, cand]) != root:
            continue                      # escaped the root — not a candidate, keep looking
        if os.path.isfile(os.path.join(cand, "model_index.json")):
            return cand
    raise RuntimeError(
        f"qf_native: model package '{name}' not found inside the model folders "
        f"(a package dir must contain model_index.json, and must live under one of "
        f"{[os.path.basename(r.rstrip('/')) for r in _model_roots()]})")


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
#    holds a weakref to each config's model as the liveness signal (set in load() after construction).
_PIPELINE_CACHE = {}     # ckey -> QFEngineHandle
_PIPELINE_MODELS = {}    # ckey -> weakref.ref(QFWanModel)  (liveness tracker for safe host-RAM eviction)
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
        ref = _PIPELINE_MODELS.get(k)
        if ref is None or ref() is not None:
            continue   # unbound (load in flight) or model still live → do NOT destroy (UAF-safe)
        eng = _PIPELINE_CACHE.pop(k, None)
        _PIPELINE_MODELS.pop(k, None)
        if eng is not None:
            try:
                eng.destroy()   # closes any session + quantfunc_destroy; idempotent (pipeline→None)
                print("[qf_native] host-RAM sweep: destroyed a cached pipeline whose QFWanModel was GC'd "
                      "(comfy dropped its patcher) — freed its CPU backup", flush=True)
            except Exception:  # noqa: BLE001
                pass


def _get_engine(model_dir, create_cfg=None):
    """Create (or reuse) the engine for a model PACKAGE dir. The native library path is
    resolved internally (resolve_so_path — NEVER a node input, #vuln). Create is MINIMAL:
    a PREQUANT svdq package carries its own layout/precision in its metadata; anything
    supplied on top competes with it and loses. The transformer weights live INSIDE the
    package (engine loads model_dir/transformer[_2]/ directly — no path override).
    create_cfg carries the per-family create keys (e.g. wan text_precision) + the
    declarative lora stack from chained QuantFuncNativeLoRA nodes."""
    lib = qfe.load_lib()
    ckey = (qfe.resolve_so_path(), model_dir, "svdq",
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
                                   model_backend="svdq", device_idx=0,
                                   config_json=(cfg if cfg else None))
    # Footprint = the ENGINE-RESIDENT transformer weight bytes only (dual-expert). VAE + text_encoder
    # stay NATIVE comfy nodes (comfy already accounts for them), so they must NOT be added here — an
    # over-report would make comfy's ledger evict siblings that actually fit.
    footprint = _estimate_package_footprint(model_dir)
    eng = qfe.QFEngineHandle(lib, pipeline, footprint_bytes=footprint)
    _PIPELINE_CACHE[ckey] = eng
    return eng, ckey


if _IMPORT_OK:
    # ── family REGISTRY: assembled from the per-family modules, nothing family-specific here ──
    # Each module owns ONE model family end to end (its comfy model subclass, its builder, and its
    # detection rule) and exposes exactly three names: FAMILY / matches() / register(deps).
    # Adding a family = write qf_<name>_modelpatcher.py + add it to _FAMILY_MODULES. No edit to the
    # node, the dispatch or the detection lives here, so families cannot bleed into each other.
    _FAMILY_MODULES = ("qf_wan_modelpatcher", "qf_ltx_modelpatcher", "qf_h3_modelpatcher")
    _FAMILY_BUILDERS = {}     # family key -> build(...)
    _FAMILY_MATCHERS = []     # (family key, matches(pipeline_class)) in registration order

    def _register_families():
        """Import each family module and register its builder. A family whose module fails to
        import is SKIPPED WITH A LOUD WARNING (its models then say 'no registered native seam')
        — one broken family must not take the whole plugin's registration down."""
        import importlib
        deps = {"get_engine": _get_engine, "pipeline_models": _PIPELINE_MODELS,
                "estimate_footprint": _estimate_package_footprint,
                "QFLazyEngine": QFLazyEngine,
                "apply_checkpoint_flow_shift": _apply_checkpoint_flow_shift}
        for mod_name in _FAMILY_MODULES:
            try:
                mod = importlib.import_module("." + mod_name, __name__)
                _FAMILY_BUILDERS[mod.FAMILY] = mod.register(deps)
                _FAMILY_MATCHERS.append((mod.FAMILY, mod.matches))
            except Exception as exc:  # noqa: BLE001 — never break plugin import
                logging.warning("[qf_native] family module %s not registered: %r", mod_name, exc)

    _register_families()
    _FAMILY_LABELS = ["auto"] + [f for f, _ in _FAMILY_MATCHERS]

    def _detect_family(model_dir):
        """Ask each registered family whether the package's declared pipeline class is theirs.
        The predicates are the ENGINE's own detector strings (each transcribed in its family
        module), so plugin and engine cannot disagree; an unimplemented family is refused LOUD
        with the class it reported — never silently routed to the wrong seam."""
        try:
            with open(os.path.join(model_dir, "model_index.json"), "r") as f:
                cls = str(json.load(f).get("_class_name", ""))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"qf_native: cannot read model_index.json in {model_dir} ({exc}) — a QuantFunc "
                f"model package must contain it (it is what names the pipeline family).")
        for fam, matches in _FAMILY_MATCHERS:
            try:
                if matches(cls):
                    return fam, cls
            except Exception:  # noqa: BLE001 — a broken predicate must not mask the others
                continue
        known = [f for f, _ in _FAMILY_MATCHERS]
        raise RuntimeError(
            f"qf_native: model_index.json reports _class_name='{cls}', which no registered native "
            f"family claims (registered: {known if known else 'NONE - a family module failed to '
            'import; check the startup log'}). Pick a different package, or set model_type "
            f"explicitly if the metadata is wrong.")

    class QuantFuncNativeLoader:
        """ONE loader for every QuantFunc native family — the model dropdown picks which.

        Shape follows the reference INT8 loader (UNetLoaderINTW8A8): a model dropdown + a
        model_type selector + CATEGORY "loaders". Everything else comes from the workflow's own
        stock nodes — length/size from the empty-latent / image-to-video node, steps from
        KSampler, fps from CreateVideo, flow shift from ModelSamplingSD3 (wan/LTX) or
        ModelSamplingMiniMaxH3 (H3). LoRAs attach DOWNSTREAM via QuantFuncNativeLoRA.
        """

        @classmethod
        def INPUT_TYPES(cls):
            _ckpts = []
            if _folder_paths is not None:
                try:
                    _ckpts = list(_folder_paths.get_filename_list("checkpoints"))
                except Exception:  # noqa: BLE001
                    _ckpts = []
            return {"required": {
                "model_name": (_list_quantfunc_packages(),
                               {"tooltip": "QuantFunc model PACKAGE (the directory holding "
                                           "model_index.json + transformer/) under "
                                           "models/diffusion_models or models/diffusers."}),
                "model_type": (_FAMILY_LABELS,
                               {"default": "auto",
                                "tooltip": "auto reads the family from the package's "
                                           "model_index.json; pick one explicitly to override."}),
                "resident_block_count": ("INT", {"default": 999, "min": 1, "max": 1024,
                                                 "tooltip": "GPU-resident transformer blocks — the "
                                                            "native seam's ONLY residency knob. The "
                                                            "engine clamps to the model's block "
                                                            "count, so the default keeps every "
                                                            "block resident on a card that fits."}),
            }, "optional": {
                "start_image": ("IMAGE", {"tooltip": "i2v reference frame (wan / LTX video-only). "
                                                     "Leave the stock node's own start_image EMPTY "
                                                     "— the engine VAE-encodes these pixels itself. "
                                                     "Omit for t2v."}),
                "connector_ckpt": (["(none)"] + _ckpts,
                                   {"tooltip": "LTX-2.3 / 19B VIDEO-ONLY packages only: the comfy "
                                               "checkpoint carrying video_embeddings_connector. "
                                               "Unused by wan, H3 and LTX-2.5 joint-AV."}),
            }}
            # NOTE: NO `so_path` / `keyfile` widgets — the native library + auth keyfile resolve from
            # the package bundle + the PROCESS ENVIRONMENT only, never from workflow JSON (#vuln: a
            # workflow-serializable path into ctypes.CDLL is an arbitrary-code-execution primitive).

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = (
            "Loads a QuantFunc svdq model PACKAGE (wan / LTX-2 / MiniMax-H3) and exposes it as a "
            "native comfy MODEL a STOCK KSampler drives — only this loader is swapped in; CLIP, "
            "VAE, latent, sampler and video nodes stay stock. Limits: (1) i2v — fan a LoadImage "
            "into this loader's start_image and leave the stock node's own start_image EMPTY. "
            "(2) A SINGLE full-range KSampler: a trimmed/partial denoise (KSamplerAdvanced "
            "start_step/last_step, denoise<1) mis-times the engine's internal schedule and is "
            "refused loud. (3) ControlNet is not consumed by this seam (refused loud). (4) Interrupt "
            "stops BETWEEN denoise steps. (5) On Linux a fail-closed CUDA-toolchain check refuses a "
            "torch/.so CUDA-major mismatch; on Windows/macOS set "
            "QF_NATIVE_ALLOW_UNVERIFIED_TOOLCHAIN=1 after confirming they share a CUDA major.")

        def load(self, model_name, model_type="auto", resident_block_count=999,
                 start_image=None, connector_ckpt="(none)"):
            model_dir = _resolve_package(model_name)
            detected, reported = _detect_family(model_dir)
            family = detected if model_type == "auto" else model_type
            if model_type != "auto" and family != detected:
                logging.warning("[qf_native] model_type=%s overrides the package's own "
                                "_class_name=%s (detected %s) — override honored, but a mismatch "
                                "usually means the wrong package is selected.",
                                model_type, reported, detected)
            builder = _FAMILY_BUILDERS.get(family)
            if builder is None:
                raise RuntimeError(
                    f"qf_native: family '{family}' has no registered native seam in this install "
                    f"(available: {sorted(_FAMILY_BUILDERS)}). An import of the seam module "
                    f"probably failed at startup — check the log for a [qf_native] warning.")
            return (builder(model_dir=model_dir, model_name=model_name,
                            resident_block_count=int(resident_block_count),
                            start_image=start_image, connector_ckpt=connector_ckpt),)

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
        CATEGORY = "loaders"
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


