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
    from .qf_modelpatcher import QFWanModel, QFModelPatcher
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


def _list_quantfunc_packages():
    """Relative paths of every QuantFunc model PACKAGE (a dir holding model_index.json) under
    comfy's models/diffusion_models roots. Single-file .safetensors UNETs are deliberately NOT
    listed: the engine needs the package (arch metadata + its own transformer/vae/TE layout), so
    offering a bare file would be a choice that can only fail."""
    out = []
    if _folder_paths is None:
        return [_NO_MODELS_HINT]
    for root_dir in _folder_paths.get_folder_paths("diffusion_models"):
        if not os.path.isdir(root_dir):
            continue
        for r, dirs, files in os.walk(root_dir, followlinks=True):
            if "model_index.json" in files:
                out.append(os.path.relpath(r, start=root_dir))
                dirs[:] = []          # a package is a leaf — do not descend into its subdirs
    return sorted(out) or [_NO_MODELS_HINT]


def _resolve_package(name):
    if _folder_paths is None:
        raise RuntimeError("qf_native: comfy folder_paths unavailable")
    if name == _NO_MODELS_HINT:
        raise RuntimeError(
            "qf_native: no QuantFunc model package found. Put the model DIRECTORY (the one with "
            "model_index.json + transformer/) under ComfyUI/models/diffusion_models/ and refresh.")
    for root_dir in _folder_paths.get_folder_paths("diffusion_models"):
        cand = os.path.join(root_dir, name)
        if os.path.isfile(os.path.join(cand, "model_index.json")):
            return cand
    raise RuntimeError(
        f"qf_native: model package '{name}' not found under models/diffusion_models "
        f"(a package dir must contain model_index.json)")


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
    footprint = qfe.estimate_footprint_bytes(*_package_weight_paths(model_dir))
    eng = qfe.QFEngineHandle(lib, pipeline, footprint_bytes=footprint)
    _PIPELINE_CACHE[ckey] = eng
    return eng, ckey


if _IMPORT_OK:
    class QuantFuncNativeWanLoader:
        """Load a QuantFunc wan svdq model PACKAGE → native comfy MODEL (stock-KSampler seam).

        OFFICIAL-LOADER SHAPE (UNETLoader/DiffusersLoader precedent): the loader takes the
        MODEL only. Everything else comes from the workflow's own stock nodes —
        steps/seed/cfg from KSampler, width/height/length from WanImageToVideo or
        EmptyHunyuanLatentVideo, fps from CreateVideo, the flow shift from
        ModelSamplingSD3 (or the package's scheduler_config, applied here as the default).
        LoRAs attach DOWNSTREAM via QuantFuncNativeLoRA (MODEL -> MODEL), like LoraLoaderModelOnly.
        """
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "model_name": (_list_quantfunc_packages(),),
                # [manual-residency] GPU-resident transformer blocks (the native seam's ONLY
                # residency mechanism; engine clamps to the model's block count, so the
                # default keeps every block resident on a card that fits it).
                "resident_block_count": ("INT", {"default": 999, "min": 1, "max": 1024}),
            }, "optional": {
                # Option C: the i2v reference frame arrives as an IMAGE GRAPH input — fan a stock
                # LoadImage into THIS (leave WanImageToVideo.start_image EMPTY). The engine VAE-encodes
                # the original pixels itself; it cannot consume comfy's encoded concat_latent_image
                # (no ref-latent field in the denoise ABI). Omit it entirely for t2v.
                "start_image": ("IMAGE",),
            }}
            # NOTE: NO `so_path` / `keyfile` widgets — the native library + auth keyfile are resolved
            # from the package bundle + the PROCESS ENVIRONMENT only, never from workflow JSON (#vuln:
            # a workflow-serializable path into ctypes.CDLL is an arbitrary-code-execution primitive).
            # Dev override: QF_NATIVE_SO_PATH / QF_NATIVE_KEYFILE env vars (see qf_engine.resolve_so_path).

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "QuantFunc/native"
        DESCRIPTION = (
            "Loads a QuantFunc wan svdq model PACKAGE from models/diffusion_models (the model "
            "DIRECTORY holding model_index.json + transformer/) and exposes it as a native comfy "
            "MODEL that a STOCK KSampler drives (only this loader is swapped in; CLIP, VAE, latent, "
            "sampler and video nodes stay stock). Requirements & limits: (1) i2v — fan a LoadImage "
            "into this loader's 'start_image' (leave WanImageToVideo.start_image EMPTY); omit it for "
            "t2v. (2) Use a SINGLE full-range KSampler — a KSamplerAdvanced two-stage / partial "
            "denoise (start_step/last_step) is NOT supported: it trims the sigma range, which "
            "mis-times the engine's internal dual-expert swap (the node fails loud if you try, it "
            "never renders silently-wrong). (3) ControlNet is not consumed by this seam (fails loud). "
            "(4) The Interrupt button stops BETWEEN denoise steps. (5) The engine loads in-process; on "
            "LINUX a fail-closed CUDA-toolchain check refuses a torch/.so CUDA-major mismatch. On "
            "WINDOWS/macOS, after confirming your torch and the engine binary share a CUDA major, set "
            "QF_NATIVE_ALLOW_UNVERIFIED_TOOLCHAIN=1 to load.")

        def load(self, model_name, resident_block_count=999, start_image=None):
            model_dir = _resolve_package(model_name)

            def _build(lora_entries):
                """Create (or reuse) the pipeline for THIS lora set and wrap it in a patcher.
                Re-invoked by QuantFuncNativeLoRA downstream: the engine applies sidecar LoRA at
                CREATE time, so attaching one re-creates the pipeline for the new set (the handle
                cache keys on it; the previous set's VRAM is released by _evict_other_pipelines)."""
                # text_precision pinned to int8 (#329-verified W8A16). WHY (mechanism traced to source
                # after a §6.5 round caught an earlier wrong explanation here): with NO text_precision
                # the engine's resolve-at-entry falls to its SM-DEFAULT tier (est::smDefaultTextPrecision
                # = fp4 on SM120+, int4 below) and WAN's UMT5 factory REJECTS both 4-bit tiers fail-loud.
                # The engine TE is not even used by this seam (conditioning comes from comfy's NATIVE
                # CLIP), but create still builds it; int8 is the verified quantized-UMT5 tier.
                cfg = {"text_precision": "int8"}
                if lora_entries:
                    cfg["lora"] = list(lora_entries)   # WanVideoPipeline splits target-tagged entries per expert
                engine, ckey = _get_engine(model_dir, create_cfg=cfg)

                # Wan A14B uses the Wan 2.1 VAE (16-ch AutoencoderKLWan) → WAN21_I2V latent_format
                # (Wan21, 16-ch). NOT WAN22_T2V (48-ch, that is the 5B TI2V VAE). UNet build DISABLED
                # (no 14B torch weights — _apply_model is overridden to drive the engine).
                unet_config = {"image_model": "wan2.1", "model_type": "i2v",
                               "disable_unet_model_creation": True}
                model_config = comfy.supported_models.WAN21_I2V(unet_config)
                for attr, default in (("manual_cast_dtype", None), ("custom_operations", None),
                                      ("optimizations", {}), ("scaled_fp8", None)):
                    if not hasattr(model_config, attr):
                        setattr(model_config, attr, default)

                device = comfy.model_management.get_torch_device()
                offload = comfy.model_management.unet_offload_device()
                model = QFWanModel(model_config, engine, start_image, device=device,
                                   resident_block_count=resident_block_count)
                # Liveness tracker for the host-RAM cache sweep: when comfy GC's this patcher's model,
                # the weakref goes dead → _sweep_dead_pipelines may safely destroy the handle.
                _PIPELINE_MODELS[ckey] = weakref.ref(model)
                # Default the sampler to the CHECKPOINT'S OWN flow schedule when the package declares
                # one (a stock ModelSamplingSD3 downstream still overrides it — as on any comfy model).
                _apply_checkpoint_flow_shift(model, model_dir)
                model._qf_lora_stack = list(lora_entries)   # what QuantFuncNativeLoRA appends to
                model._qf_rebuild = _build                  # how it re-creates with the new set
                patcher = QFModelPatcher(model, load_device=device, offload_device=offload)
                print(f"[qf_native] loaded QuantFuncNativeWanLoader (wan svdq) package={model_name} "
                      f"resident_blocks={resident_block_count} loras={len(lora_entries)} "
                      f"footprint={engine.footprint_bytes // (1024*1024)}MB")
                return patcher

            return (_build([]),)

    class QuantFuncNativeLoRA:
        """Sidecar LoRA for the QuantFunc native loaders — MODEL in, MODEL out (LoraLoaderModelOnly
        shape). Chain several to stack them.

        The engine applies sidecar LoRA at pipeline CREATE time (runtime hot-swap is not wired for
        the video pipelines), so this node RE-CREATES the pipeline for the accumulated LoRA set.
        The engine merges each LoRA onto the svdq slots' sidecar branches — the int4 base weights
        are never touched — and a wrong-arch / typo'd file fails LOUD at create, never silently.
        """
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "model": ("MODEL",),
                "lora_name": (_lora_choices(),),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }, "optional": {
                # wan A14B per-expert routing: all (default) | high | low. Single-transformer
                # families (LTX/H3) take "all"; an explicit high/low there fails loud engine-side
                # rather than silently mis-routing.
                "target": (["all", "high", "low"],),
            }}

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "apply"
        CATEGORY = "QuantFunc/native"
        DESCRIPTION = ("Attaches a sidecar LoRA to a QuantFunc native MODEL (wire it downstream of a "
                       "QuantFunc native loader; chain several to stack). Applying or changing a LoRA "
                       "re-creates the engine pipeline, because the engine merges sidecar LoRA at "
                       "create time.")

        def apply(self, model, lora_name, strength, target="all"):
            inner = getattr(model, "model", None)
            rebuild = getattr(inner, "_qf_rebuild", None)
            if rebuild is None:
                raise RuntimeError(
                    "QuantFuncNativeLoRA: this MODEL is not a QuantFunc native model — wire it "
                    "downstream of a QuantFunc Native Wan/LTX/H3 Loader. (For a stock comfy model "
                    "use the built-in LoraLoaderModelOnly instead.)")
            stack = list(getattr(inner, "_qf_lora_stack", []))
            stack.append({"path": _resolve_lora(lora_name), "scale": float(strength),
                          "target": target})
            return (rebuild(stack),)

    # merge into (not replace) the mappings — matches the real plugin's multi-file NODE_CLASS_MAPPINGS.update
    NODE_CLASS_MAPPINGS.update({"QuantFuncNativeWanLoader": QuantFuncNativeWanLoader,
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
    NODE_DISPLAY_NAME_MAPPINGS.update({"QuantFuncNativeWanLoader": "QuantFunc Native Wan Loader",
                                   "QuantFuncNativeLoRA": "QuantFunc Native LoRA"})

    # LTX-2 native seam (qf_ltx_modelpatcher). Additive + defensively guarded: a bug in the LTX file
    # must NEVER break the wan loader's registration. _get_engine + _PIPELINE_MODELS are threaded in
    # (they live here, not in qf_modelpatcher) to avoid a circular import.
    try:
        from . import qf_ltx_modelpatcher as _qf_ltx
        _qf_ltx.register(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, _get_engine, _PIPELINE_MODELS,
                         _list_quantfunc_packages, _resolve_package)
    except Exception as _ltx_exc:  # noqa: BLE001 — LTX registration must never break plugin import
        logging.warning("[qf_native] LTX loader registration skipped: %r", _ltx_exc)

    # MiniMax-H3 native joint-AV seam (qf_h3_modelpatcher). Same additive + defensively-guarded pattern:
    # a bug in the H3 file must NEVER break the wan/LTX loaders' registration.
    try:
        from . import qf_h3_modelpatcher as _qf_h3
        _qf_h3.register(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, _get_engine, _PIPELINE_MODELS,
                        _list_quantfunc_packages, _resolve_package)
    except Exception as _h3_exc:  # noqa: BLE001 — H3 registration must never break plugin import
        logging.warning("[qf_native] H3 loader registration skipped: %r", _h3_exc)
