"""qf_native -- ComfyUI plugin for QuantFunc PC Engine (LTX-2.5, MiniMax-H3, Krea-2, Qwen-Image-2.1)."""
import inspect
import json
import os
import struct
import sys
import threading
import weakref

try:
    from . import qf_engine as qfe
    _log = qfe.logger(__name__)
except Exception as _qfe_exc:  # noqa: BLE001
    import logging
    qfe, _log = None, logging.getLogger(__name__)
    _log.warning("[qf_native] disabled - qf_engine failed to import: %s", ascii(_qfe_exc))

try:
    from . import qf_api_key
except Exception as _qak_exc:  # noqa: BLE001
    qf_api_key = None
    _log.warning("[qf_native] disabled - qf_api_key failed to import: %s", ascii(_qak_exc))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

try:
    if qfe is None or qf_api_key is None:
        raise ImportError("qf_engine or qf_api_key did not import")
    import comfy.model_management
    import comfy.supported_models
    from . import qf_modelpatcher as qfmp
    from . import qf_sol
    _IMPORT_OK = True
except Exception as _exc:  # noqa: BLE001
    _log.warning("[qf_native] disabled - a required import failed (ComfyUI API drift?): %s", ascii(_exc))
    _IMPORT_OK = False

try:
    import folder_paths as _folder_paths
except Exception as _fp_exc:  # noqa: BLE001
    _folder_paths = None
    _log.warning("[qf_native] folder_paths unavailable: %s", ascii(_fp_exc))

_NO_LORA_HINT = "(no LoRA in models/loras)"
_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
_NO_CFG_HINT = "(no official model config shipped)"
_NO_XFM_HINT = "(no .safetensors in models/diffusion_models)"
_MAX_LORA_HEADER_BYTES = 512 * 1024 * 1024


def _model_config_choices(family=None):
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
                except Exception:  # noqa: BLE001
                    continue
            out.append(d)
        return out or [_NO_CFG_HINT]
    except Exception:  # noqa: BLE001
        return [_NO_CFG_HINT]


def _shipped_presets(family):
    return [c for c in _model_config_choices(family=family) if c != _NO_CFG_HINT]


def _family_preset(family):
    presets = _shipped_presets(family)
    if len(presets) == 1:
        return presets[0]
    if not presets:
        _raise_if_a_manifest_is_unreadable(family)
        raise RuntimeError(f"qf_native: this plugin ships no model config for {family} (configs/ has no preset for it).")
    raise RuntimeError(f"qf_native: this plugin ships {len(presets)} model configs for {family} ({', '.join(presets)}) "
                       f"and the loader cannot choose between them.")


def _load_model_config(name):
    if name == _NO_CFG_HINT or os.sep in name or "/" in name or "\\" in name or name in ("", ".", ".."):
        raise RuntimeError(f"qf_native: invalid model_config {name!r} - pick one of {_model_config_choices()}.")
    bundle = os.path.join(_CONFIGS_DIR, name)
    mf = os.path.join(bundle, "qf_native.json")
    if not os.path.isfile(mf):
        raise RuntimeError(f"qf_native: model_config {name!r} has no qf_native.json manifest "
                           f"(shipped presets: {_model_config_choices()}).")
    try:
        with open(mf, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"qf_native: model_config {name!r} manifest unreadable ({mf}): {exc}. "
                           f"It ships with the plugin: reinstall or update the QuantFunc plugin.") from exc
    if not isinstance(manifest, dict) or not manifest.get("family"):
        raise RuntimeError(f"qf_native: model_config {name!r} manifest ({mf}) must declare a family. "
                           f"It ships with the plugin: reinstall or update the QuantFunc plugin.")
    return bundle, manifest


def _raise_if_a_manifest_is_unreadable(family):
    broken = []
    for name in _shipped_presets(None):
        try:
            _load_model_config(name)
        except RuntimeError as exc:
            broken.append(str(exc))
    if broken:
        raise RuntimeError(f"qf_native: no readable model config for {family}: " + " | ".join(broken))


def _transformer_choices():
    if _folder_paths is None:
        return [_NO_XFM_HINT]
    try:
        files = [f for f in _folder_paths.get_filename_list("diffusion_models")
                 if f.lower().endswith(".safetensors")]
    except Exception:  # noqa: BLE001
        files = []
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
    if _folder_paths is None or name in ("", _NO_XFM_HINT):
        raise RuntimeError("qf_native: no transformer weight selected - put the .safetensors "
                           "under ComfyUI/models/diffusion_models/ and pick it in transformer.")
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
        raise RuntimeError("qf_native: no LoRA available - put .safetensors files in models/loras/")
    return _folder_paths.get_full_path_or_raise("loras", name)


def _resolve_keyfile():
    override = os.environ.get(qfe._ENV_KEYFILE_OVERRIDE, "").strip()
    if override:
        return override
    pkg = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg, "bin", qfe._BIN_SUBDIR, qfe._KEYFILE_BASENAME)


def _read_auth(ui_key=None):
    surl = os.environ.get("QF_SERVER_URL", "https://service.quantfunc.com")
    if ui_key:
        return ui_key, surl
    key = os.environ.get("QUANTFUNC_API_KEY", "")
    keyfile = qf_api_key.config_to_read(_resolve_keyfile())
    if not key and keyfile and os.path.exists(keyfile):
        try:
            with open(keyfile, encoding="utf-8") as fh:
                c = json.load(fh)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"qf_native: the keyfile {keyfile} is unreadable: {exc}. Set QUANTFUNC_API_KEY (the file "
                f"is then not read), or restore the file: reinstall or update the QuantFunc plugin.") from exc
        if not isinstance(c, dict):
            raise RuntimeError(
                f"qf_native: the keyfile {keyfile} must hold a JSON object. Set QUANTFUNC_API_KEY (the file "
                f"is then not read), or restore the file: reinstall or update the QuantFunc plugin.")
        key, surl = c.get("api_key", ""), c.get("server_url", surl)
    return key, surl


def _comfy_device_index():
    try:
        dev = comfy.model_management.get_torch_device()
        return int(dev.index) if getattr(dev, "type", "") == "cuda" and dev.index is not None else 0
    except Exception:  # noqa: BLE001
        return 0


# ── Transformer handle cache & host-RAM lifecycle ────────────────────────────

_PIPELINE_CACHE = {}
_ENGINE_IDENTITY_LOCK = threading.RLock()
_CREATE_LOCKS = {}
_PIPELINE_MODELS = {}
_PIPELINE_SIGS = {}


def _create_lock_for(ckey):
    with _ENGINE_IDENTITY_LOCK:
        lock = _CREATE_LOCKS.get(ckey)
        if lock is None:
            lock = threading.Lock()
            _CREATE_LOCKS[ckey] = lock
        return lock


def _unpin_pipeline_consumer(ckey, consumer):
    with _ENGINE_IDENTITY_LOCK:
        refs = [ref for ref in _PIPELINE_MODELS.get(ckey, []) if ref() is not None and ref() is not consumer]
        if refs:
            _PIPELINE_MODELS[ckey] = refs
        else:
            _PIPELINE_MODELS.pop(ckey, None)
            _PIPELINE_SIGS.pop(ckey, None)


def _pin_pipeline_consumer_locked(ckey):
    acquisition = qfmp._current_engine_cache_acquisition()
    if acquisition is None:
        return None
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
    with _ENGINE_IDENTITY_LOCK:
        for old_key in list(_PIPELINE_MODELS):
            if old_key not in _PIPELINE_CACHE:
                _live_pipeline_models(old_key)
        refs = [r for r in _PIPELINE_MODELS.get(ckey, []) if r() is not None]
        if not any(r() is model for r in refs):
            refs.append(weakref.ref(model))
        _PIPELINE_MODELS[ckey] = refs
        sig = getattr(model, "_qf_loader_sig", None)
        if sig is not None:
            _PIPELINE_SIGS.setdefault(ckey, set()).add(sig)


def _live_pipeline_models(ckey):
    with _ENGINE_IDENTITY_LOCK:
        refs = [r for r in _PIPELINE_MODELS.get(ckey, []) if r() is not None]
        if refs:
            _PIPELINE_MODELS[ckey] = refs
        else:
            _PIPELINE_MODELS.pop(ckey, None)
            _PIPELINE_SIGS.pop(ckey, None)
        return [r() for r in refs]


def _retire_handle(ckey, eng, requester=None, *, keep_binding=False, reason=""):
    """Single authorized .destroy() chokepoint: refuses while any live foreign consumer is bound."""
    with _ENGINE_IDENTITY_LOCK:
        if _PIPELINE_CACHE.get(ckey) is not eng:
            return False
        req_qf = getattr(requester, "_qf", requester)
        foreign = [c for c in _live_pipeline_models(ckey)
                   if requester is None or getattr(c, "_qf", c) is not req_qf]
        if foreign:
            qfe.info(f"[qf_native] retire refused ({reason or 'unspecified'}): {len(foreign)} live "
                     "foreign consumer(s) still bound to this cache entry", flush=True)
            return False
        _PIPELINE_CACHE.pop(ckey, None)
        if not keep_binding:
            _PIPELINE_MODELS.pop(ckey, None)
            _PIPELINE_SIGS.pop(ckey, None)
    try:
        eng.destroy()
    except Exception:  # noqa: BLE001
        pass
    return True


def _sweep_dead_pipelines(keep_key):
    for k in _dead_keys():
        if k == keep_key:
            continue
        with _ENGINE_IDENTITY_LOCK:
            eng = _PIPELINE_CACHE.get(k)
        if eng is not None and _retire_handle(k, eng, None, reason="host-RAM sweep"):
            qfe.info("[qf_native] host-RAM sweep: destroyed cached transformer whose model was GC'd", flush=True)


def _loader_sig(family, transformer, model_config, pinned_memory):
    return family, transformer, model_config, pinned_memory


def _pending_loader_sigs():
    queue = getattr(getattr(getattr(sys.modules.get("server"), "PromptServer", None), "instance", None),
                    "prompt_queue", None)
    if queue is None:
        return set()
    sigs = set()
    try:
        with queue.mutex:
            running = list(queue.currently_running.values())
        for item in running:
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
                defaults = inspect.signature(cls.load).parameters
                sigs.add(_loader_sig(family, ins.get("transformer"),
                                     ins.get("model_config", defaults["model_config"].default),
                                     ins.get("pinned_memory", defaults["pinned_memory"].default)))
    except Exception as exc:  # noqa: BLE001
        _log.warning("[qf_native] running prompt unreadable from ComfyUI queue (%s)", ascii(exc))
        return set()
    return sigs


def _dead_keys():
    with _ENGINE_IDENTITY_LOCK:
        dead = [k for k in _PIPELINE_CACHE
                if _PIPELINE_MODELS.get(k) and all(r() is None for r in _PIPELINE_MODELS[k])]
    if not dead:
        return []
    pending = _pending_loader_sigs()
    with _ENGINE_IDENTITY_LOCK:
        return [k for k in dead if not (_PIPELINE_SIGS.get(k, set()) & pending)]


def _dead_pipelines():
    dead = _dead_keys()
    with _ENGINE_IDENTITY_LOCK:
        return len(dead), sum(int(getattr(_PIPELINE_CACHE.get(k), "capacity_bytes", 0) or 0) for k in dead)


def _install_host_ram_release():
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
                    qfe.info(
                        f"[qf_native] host RAM: {int(available) >> 20} MB available, under ComfyUI's "
                        f"{int(target) >> 20} MB headroom plus {held >> 20} MB in {count} dropped model(s): releasing",
                        flush=True)
                    _sweep_dead_pipelines(None)
        except Exception as exc:  # noqa: BLE001
            _log.warning("[qf_native] host-RAM release skipped: %s", ascii(exc))
        return freed

    extra_ram_release._qf_host_ram_release = True
    cmm.extra_ram_release = extra_ram_release


def _engine_recipe(weights_path, create_cfg=None, device_id=0):
    lib = qfe.load_lib()
    ckey = (qfe.loaded_so_path(), weights_path, int(device_id),
            json.dumps(create_cfg or {}, sort_keys=True))
    return lib, ckey, create_cfg


def _prepare_params(weights_path, create_cfg, device_id, api_key=None):
    key, surl = _read_auth(api_key)
    cfg = dict(create_cfg or {})
    if key:
        cfg["api_key"] = key
        cfg["server_url"] = surl
    return qfe.make_create_params(weights_path=weights_path, device_id=int(device_id), config_json=cfg or None)


def _peek_engine(weights_path, create_cfg=None, device_id=0):
    """Non-creating cache lookup for VRAM ledger queries."""
    so = qfe.loaded_so_path()
    if so is None:
        return None
    ckey = (so, weights_path, int(device_id), json.dumps(create_cfg or {}, sort_keys=True))
    with _ENGINE_IDENTITY_LOCK:
        eng = _PIPELINE_CACHE.get(ckey)
        if eng is not None and eng.xfm is not None:
            return eng, ckey
    return None


def _estimate_engine(weights_path, create_cfg=None, device_id=0, api_key=None):
    """Non-creating header-only VRAM estimate via qf_xfm_estimate_vram."""
    lib = qfe._LIB
    if lib is None:
        return qfe.ColdNeed(None, None)
    params = _prepare_params(weights_path, create_cfg, device_id, api_key)
    return qfe.estimate_vram(lib, params)


def _get_engine(weights_path, create_cfg=None, device_id=0, api_key=None):
    """Create (or reuse) one pure transformer handle per immutable weight recipe."""
    lib, ckey, create_cfg = _engine_recipe(weights_path, create_cfg, device_id)
    with _ENGINE_IDENTITY_LOCK:
        eng = _PIPELINE_CACHE.get(ckey)
        if eng is not None and eng.xfm is not None:
            if qfe.library_identity(eng.lib) != qfe.library_identity(lib):
                raise RuntimeError("QuantFunc cache entry belongs to a different loaded native image")
            _pin_pipeline_consumer_locked(ckey)
            return eng, ckey
    with _create_lock_for(ckey):
        with _ENGINE_IDENTITY_LOCK:
            eng = _PIPELINE_CACHE.get(ckey)
            if eng is not None and eng.xfm is not None:
                if qfe.library_identity(eng.lib) != qfe.library_identity(lib):
                    raise RuntimeError("QuantFunc cache entry belongs to a different loaded native image")
                _pin_pipeline_consumer_locked(ckey)
                return eng, ckey
        _sweep_dead_pipelines(ckey)
        params = _prepare_params(weights_path, create_cfg, device_id, api_key)
        cold = qfe.estimate_vram(lib, params)
        eng = qfe.QFEngineHandle.create(lib, create_params=params, capacity_bytes=int(cold.bytes or 0))
        eng.applied_api_key = api_key
        with _ENGINE_IDENTITY_LOCK:
            _PIPELINE_CACHE[ckey] = eng
            _pin_pipeline_consumer_locked(ckey)
        return eng, ckey


# ── Family registry & ComfyUI nodes ──────────────────────────────────────────

if _IMPORT_OK:
    _FAMILY_MODULES = ("qf_ltx_modelpatcher", "qf_h3_modelpatcher", "qf_krea2_modelpatcher",
                       "qf_qwenimage21_modelpatcher")
    _FAMILY_BUILDERS = {}
    _FAMILY_MATCHERS = []

    def _register_families():
        import importlib
        deps = {
            "get_engine": _get_engine,
            "bind_pipeline_model": _bind_pipeline_model,
            "read_auth": _read_auth,
            "peek_engine": _peek_engine,
            "estimate_engine": _estimate_engine,
        }
        for mod_name in _FAMILY_MODULES:
            try:
                mod = importlib.import_module("." + mod_name, __name__)
                _FAMILY_BUILDERS[mod.FAMILY] = mod.register(deps)
                _FAMILY_MATCHERS.append((mod.FAMILY, mod.matches))
            except Exception as exc:  # noqa: BLE001
                _log.warning("[qf_native] family module %s not registered: %s", mod_name, ascii(exc))

    _register_families()

    def _coerce_bool_switch(value, default, name, *, allow_none=False):
        """Safely coerce corrupted non-boolean trailing switches from qf_layout: 2 workflows."""
        if value is None:
            return None if allow_none else default
        if isinstance(value, bool):
            return value
        _log.warning("[qf_native] migrated non-boolean %s=%r -> %s (legacy qf_layout: 2 workflow)",
                     name, value, default)
        return default

    def _run_family_load(expect_family, transformer1, model_config=None, pinned_memory=False):
        sig = _loader_sig(expect_family, transformer1, model_config, pinned_memory)
        if model_config is None:
            model_config = _family_preset(expect_family)
        else:
            shipped = _shipped_presets(expect_family)
            if model_config not in shipped:
                _raise_if_a_manifest_is_unreadable(expect_family)
                raise RuntimeError(
                    f"qf_native: this workflow was saved with model_config {model_config!r}, which is not a "
                    f"{expect_family} model config of this plugin (it ships: {', '.join(shipped) or 'none'}).")
        bundle_dir, manifest = _load_model_config(model_config)
        family = str(manifest["family"])
        if family != expect_family:
            raise RuntimeError(
                f"qf_native: model_config '{model_config}' declares family '{family}' but this "
                f"loader node is the '{expect_family}' loader.")
        builder = _FAMILY_BUILDERS.get(family)
        if builder is None:
            raise RuntimeError(
                f"qf_native: model_config '{model_config}' routes to family '{family}', which "
                f"has no registered native seam (available: {sorted(_FAMILY_BUILDERS)}).")
        xfm1 = _resolve_transformer(transformer1)
        from .qf_file_hints import name_matches_hints
        pats = (manifest.get("file_hints") or {}).get("transformer1") or []
        if pats and not name_matches_hints(transformer1, pats):
            raise RuntimeError(
                f"qf_native: transformer1={transformer1!r} does not look like a '{model_config}' "
                f"transformer1 weight (expected a name matching {pats}).")
        token = qfmp.LOADER_SIG.set(sig)
        try:
            return builder(transformer1_path=xfm1, bundle_dir=bundle_dir, pinned_memory=bool(pinned_memory))
        finally:
            qfmp.LOADER_SIG.reset(token)

    _STEP_CACHE_INPUT = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                         "tooltip": "Speed-up that reuses earlier work while the result is barely changing. "
                                    "0 (default) = off; 0.1-0.3 is typical. Higher is faster with softer detail. "
                                    "Takes effect on the next run."})
    _BLOCK_CACHE_INPUT = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                          "tooltip": "Speed-up that reuses work inside each pass when little changes. "
                                     "0 (default) = off; 0.02-0.08 is typical. Higher is faster with softer detail; "
                                     "can be combined with step_cache. Takes effect on the next run."})
    _QUALITY_ENHANCE_INPUT = ("BOOLEAN", {"default": False,
                              "tooltip": "OFF (default): faster; the subject and scene stay the same, but details such as poses, "
                                         "faces or small objects can differ. ON: the highest quality, a little slower. Takes "
                                         "effect on the next run."})

    def _pinned_memory_input(default_on):
        return ("BOOLEAN", {"default": default_on, "tooltip": (
            "ON (default for LTX-2.5, which moves the most model data on graphics cards with little VRAM): faster on such "
            "cards. " if default_on else
            "OFF (default): the model is kept in ordinary system memory. ON: faster on graphics cards with little VRAM. ") +
            "When enough RAM is free, the model is kept in locked system memory, which the rest of the PC cannot use while "
            "the model is loaded; on a PC with little RAM this can make the system unstable. " +
            ("OFF: the model is kept in ordinary system memory. " if default_on else "") +
            "Changing it reloads the model."})

    _PINNED_MEMORY_INPUT = _pinned_memory_input(False)
    _PINNED_MEMORY_INPUT_LTX = _pinned_memory_input(True)
    _QUALITY_LEGACY_HIDDEN = {"quality": ("STRING", {})}

    def _quality_enhance_on(quality_enhance=None, quality=None):
        return bool(quality_enhance) if quality_enhance is not None else quality == "best_quality"

    def _model_config_input(family):
        presets = _shipped_presets(family) or [_NO_CFG_HINT]
        opts = {"default": presets[0], "tooltip": "The model config that matches the chosen model file."}
        if len(presets) == 1:
            opts.update(hidden=True, socketless=True)
        return presets, opts

    _AUDIO_ENHANCE_INPUT = ("BOOLEAN", {"default": False,
                            "tooltip": "Legacy workflow compatibility toggle (no longer needed in QuantFunc PC Engine; ignored)."})

    _SOL_TAU_INPUT = ("FLOAT", {"default": 1.3, "min": -4.0, "max": 4.0, "step": 0.05,
                      "tooltip": "Higher values favor speed; lower values favor detail. "
                                 "Negative values saved in older workflows remain supported."})
    _SOL_ENABLED_INPUT = ("BOOLEAN", {"default": False,
                          "tooltip": "Enable Sparse attention acceleration (Sol) to skip low-weight attention blocks on long sequences."})
    _SOL_WINDOW_INPUT = ("BOOLEAN", {"default": True, "tooltip": "Limit Sol to the selected sampling interval."})
    _SOL_START_INPUT = ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01})
    _SOL_END_INPUT = ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01})
    _SOL_MIN_INPUT = ("INT", {"default": 12288, "min": 0, "max": 1048576, "step": 512})
    _SOL_SINK_INPUT = (["exact_kv_and_rows", "exact_kv", "off"], {"default": "exact_kv_and_rows"})
    _SOL_VERSION_INPUT = ("INT", {"default": 2, "min": 1, "max": 2, "hidden": True, "socketless": True})

    def _arm_session_caches(mm, step_cache, block_cache):
        if mm is None:
            return
        if hasattr(mm, "set_step_cache"):
            mm.set_step_cache(float(step_cache or 0.0))
        if hasattr(mm, "set_block_cache"):
            mm.set_block_cache(float(block_cache or 0.0))

    _COMMON_LIMITS = (
        "Notes: ControlNet is not supported. Interrupting stops at the next safe point of the run. The loader checks "
        "that ComfyUI and this plugin use a matching CUDA version; if they do not, it stops with a message that "
        "explains the fix.")

    # Unified PC Engine attention backends across all 4 consumer SMs (SM75/86/89/120).
    _ATTN_BACKENDS = ["auto", "int8", "i4xi8", "int4", "fp16"]
    _ATTN_BACKEND_ALIASES = {
        "auto": "auto",
        "int8": "int8",
        "i4xi8": "i4xi8",
        "int4_pv8": "i4xi8",
        "int4": "int4",
        "fp16": "fp16",
        "qfa": "int8",
        "sage": "int8",
        "fp16_native": "fp16",
        "native": "fp16",
        "flash": "fp16",
    }

    def _attn_backend_choices(_cc=None):
        return list(_ATTN_BACKENDS)

    def _attn_backend_default(_cc=None, _platform=None):
        return ("auto", "")

    def _attn_backend_input():
        return (_attn_backend_choices(), {"default": "auto",
                "tooltip": "auto (recommended) selects the best built-in attention mode for your GPU. "
                           "Modes trade accuracy for speed (fp16 is reference; i4xi8 balances speed and quality); "
                           "older workflow values map automatically. Takes effect on the next run."})

    def _attn_backend_to_engine(v):
        if v is None or v == "":
            return "auto"
        key = str(v).strip().lower()
        mapped = _ATTN_BACKEND_ALIASES.get(key)
        if mapped is None:
            raise ValueError(f"qf_native: unsupported attention_backend {v!r}; expected one of {_ATTN_BACKENDS}")
        return mapped

    def _validate_attn_backend(attention_backend="auto"):
        try:
            _attn_backend_to_engine(attention_backend)
        except ValueError as exc:
            return str(exc)
        return True

    def _apply_common_loader_dials(patcher, attention_backend, quality_enhance, quality,
                                   sol_version, sol_tau, sol_enabled, sol_window_enabled,
                                   sol_start_percent, sol_end_percent, sol_min_tokens, sol_sink_conditioning):
        mm = getattr(patcher, "model", None)
        if mm is not None and hasattr(mm, "set_attn_backend"):
            mm.set_attn_backend(_attn_backend_to_engine(attention_backend))
        mm.set_sol_options(qf_sol.from_loader(
            sol_version=sol_version, sol_tau=sol_tau, sol_enabled=sol_enabled,
            sol_window_enabled=sol_window_enabled, sol_start_percent=sol_start_percent,
            sol_end_percent=sol_end_percent, sol_min_tokens=sol_min_tokens,
            sol_sink_conditioning=sol_sink_conditioning))
        mm.set_video_enhance(_quality_enhance_on(quality_enhance, quality))
        return mm

    class QuantFuncLTXLoader:
        """LTX-2.5 joint audio+video loader."""

        QF_FAMILY = "ltx2"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "transformer": (_transformer_choices(),
                                {"tooltip": "The QuantFunc LTX-2 model file in models/diffusion_models."}),
            }, "optional": {
                "model_config": _model_config_input("ltx2"),
                "attention_backend": _attn_backend_input(),
                "sol_tau": _SOL_TAU_INPUT,
                "sol_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sol_end_percent": _SOL_END_INPUT,
                "sol_min_tokens": _SOL_MIN_INPUT,
                "sol_sink_conditioning": _SOL_SINK_INPUT,
                "sol_version": _SOL_VERSION_INPUT,
                "step_cache": _STEP_CACHE_INPUT,
                "block_cache": _BLOCK_CACHE_INPUT,
                "sol_enabled": ("BOOLEAN", {"default": True}),
                "sol_window_enabled": _SOL_WINDOW_INPUT,
                "quality_enhance": _QUALITY_ENHANCE_INPUT,
                "pinned_memory": _PINNED_MEMORY_INPUT_LTX,
            }, "hidden": dict(_QUALITY_LEGACY_HIDDEN)}

        @classmethod
        def VALIDATE_INPUTS(cls, attention_backend="auto"):
            return _validate_attn_backend(attention_backend)

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = (
            "Loads a QuantFunc LTX-2 model (video with sound) for ComfyUI's standard samplers - use it in place "
            "of the usual diffusion-model loader. For image-to-video, add the official LTXVImgToVideoInplace node on "
            "the latent input. " + _COMMON_LIMITS)

        def load(self, transformer, model_config=None,
                 attention_backend="auto", sol_tau=None, quality_enhance=None, step_cache=0.0, block_cache=0.0,
                 quality=None, pinned_memory=True, sol_version=None, sol_enabled=None, sol_window_enabled=None,
                 sol_start_percent=None, sol_end_percent=None, sol_min_tokens=None, sol_sink_conditioning=None):
            pinned_memory = _coerce_bool_switch(pinned_memory, True, "pinned_memory")
            quality_enhance = _coerce_bool_switch(quality_enhance, False, "quality_enhance", allow_none=True)
            p = _run_family_load(self.QF_FAMILY, transformer, model_config, pinned_memory)
            mm = _apply_common_loader_dials(
                p, attention_backend, quality_enhance, quality,
                sol_version, sol_tau, sol_enabled, sol_window_enabled,
                sol_start_percent, sol_end_percent, sol_min_tokens, sol_sink_conditioning)
            _arm_session_caches(mm, step_cache, block_cache)
            return (p,)

    class QuantFuncKrea2Loader:
        """Krea-2 Turbo image t2i loader."""

        QF_FAMILY = "krea2"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "transformer": (_transformer_choices(),
                                {"tooltip": "The QuantFunc Krea-2 Turbo model file in models/diffusion_models."}),
            }, "optional": {
                "model_config": _model_config_input("krea2"),
                "attention_backend": _attn_backend_input(),
                "sol_tau": _SOL_TAU_INPUT,
                "sol_start_percent": _SOL_START_INPUT,
                "sol_end_percent": _SOL_END_INPUT,
                "sol_min_tokens": _SOL_MIN_INPUT,
                "sol_sink_conditioning": _SOL_SINK_INPUT,
                "sol_version": _SOL_VERSION_INPUT,
                "sol_enabled": _SOL_ENABLED_INPUT,
                "sol_window_enabled": _SOL_WINDOW_INPUT,
                "quality_enhance": _QUALITY_ENHANCE_INPUT,
                "pinned_memory": _PINNED_MEMORY_INPUT,
            }, "hidden": dict(_QUALITY_LEGACY_HIDDEN)}

        @classmethod
        def VALIDATE_INPUTS(cls, attention_backend="auto"):
            return _validate_attn_backend(attention_backend)

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = ("Loads a QuantFunc Krea-2 Turbo model (text-to-image) for ComfyUI's standard samplers - use it in "
                       "place of the usual diffusion-model loader. " + _COMMON_LIMITS)

        def load(self, transformer, model_config=None, attention_backend="auto",
                 quality_enhance=None, quality=None, pinned_memory=False, sol_tau=None, sol_version=None,
                 sol_enabled=None, sol_window_enabled=None, sol_start_percent=None, sol_end_percent=None,
                 sol_min_tokens=None, sol_sink_conditioning=None):
            pinned_memory = _coerce_bool_switch(pinned_memory, False, "pinned_memory")
            quality_enhance = _coerce_bool_switch(quality_enhance, False, "quality_enhance", allow_none=True)
            p = _run_family_load(self.QF_FAMILY, transformer, model_config, pinned_memory)
            _apply_common_loader_dials(
                p, attention_backend, quality_enhance, quality,
                sol_version, sol_tau, sol_enabled, sol_window_enabled,
                sol_start_percent, sol_end_percent, sol_min_tokens, sol_sink_conditioning)
            return (p,)

    class QuantFuncQwenImage21Loader:
        """Qwen-Image-2.1 (text-to-image + reference edit) loader."""

        QF_FAMILY = "qwenimage21"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "transformer": (_transformer_choices(),
                                {"tooltip": "The QuantFunc Qwen-Image-2.1 model file in models/diffusion_models."}),
            }, "optional": {
                "model_config": _model_config_input("qwenimage21"),
                "attention_backend": _attn_backend_input(),
                "sol_tau": _SOL_TAU_INPUT,
                "sol_start_percent": _SOL_START_INPUT,
                "sol_end_percent": _SOL_END_INPUT,
                "sol_min_tokens": _SOL_MIN_INPUT,
                "sol_sink_conditioning": _SOL_SINK_INPUT,
                "sol_version": _SOL_VERSION_INPUT,
                "sol_enabled": _SOL_ENABLED_INPUT,
                "sol_window_enabled": _SOL_WINDOW_INPUT,
                "quality_enhance": _QUALITY_ENHANCE_INPUT,
                "pinned_memory": _PINNED_MEMORY_INPUT,
            }, "hidden": dict(_QUALITY_LEGACY_HIDDEN)}

        @classmethod
        def VALIDATE_INPUTS(cls, attention_backend="auto"):
            return _validate_attn_backend(attention_backend)

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = (
            "Loads a QuantFunc Qwen-Image-2.1 model for ComfyUI's standard samplers: text-to-image, and image "
            "editing with TextEncodeQwenImage21 reference images (plus its VAE). Transparent images: VAE Decode + "
            "Save Image keep the transparency. " + _COMMON_LIMITS)

        def load(self, transformer, model_config=None, attention_backend="auto", quality_enhance=None, quality=None,
                 pinned_memory=False, sol_tau=None, sol_version=None, sol_enabled=None, sol_window_enabled=None,
                 sol_start_percent=None, sol_end_percent=None, sol_min_tokens=None, sol_sink_conditioning=None):
            pinned_memory = _coerce_bool_switch(pinned_memory, False, "pinned_memory")
            quality_enhance = _coerce_bool_switch(quality_enhance, False, "quality_enhance", allow_none=True)
            p = _run_family_load(self.QF_FAMILY, transformer, model_config, pinned_memory)
            _apply_common_loader_dials(
                p, attention_backend, quality_enhance, quality,
                sol_version, sol_tau, sol_enabled, sol_window_enabled,
                sol_start_percent, sol_end_percent, sol_min_tokens, sol_sink_conditioning)
            return (p,)

    class QuantFuncH3Loader:
        """MiniMax-H3 joint audio+video loader."""

        QF_FAMILY = "minimax-h3"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "transformer": (_transformer_choices(),
                                {"tooltip": "The QuantFunc MiniMax-H3 model file in models/diffusion_models."}),
            }, "optional": {
                "model_config": _model_config_input("minimax-h3"),
                "attention_backend": _attn_backend_input(),
                "sol_tau": _SOL_TAU_INPUT,
                "sol_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sol_end_percent": _SOL_END_INPUT,
                "sol_min_tokens": _SOL_MIN_INPUT,
                "sol_sink_conditioning": _SOL_SINK_INPUT,
                "sol_version": _SOL_VERSION_INPUT,
                "step_cache": _STEP_CACHE_INPUT,
                "block_cache": _BLOCK_CACHE_INPUT,
                "sol_enabled": ("BOOLEAN", {"default": True}),
                "sol_window_enabled": _SOL_WINDOW_INPUT,
                "quality_enhance": _QUALITY_ENHANCE_INPUT,
                "audio_enhance": _AUDIO_ENHANCE_INPUT,
                "pinned_memory": _PINNED_MEMORY_INPUT,
            }, "hidden": dict(_QUALITY_LEGACY_HIDDEN)}

        @classmethod
        def VALIDATE_INPUTS(cls, attention_backend="auto"):
            return _validate_attn_backend(attention_backend)

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "load"
        CATEGORY = "loaders"
        DESCRIPTION = ("Loads a QuantFunc MiniMax-H3 model (video with sound) for ComfyUI's standard samplers - use it in "
                       "place of the usual diffusion-model loader. " + _COMMON_LIMITS)

        def load(self, transformer, model_config=None,
                 attention_backend="auto", sol_tau=None, quality_enhance=None, audio_enhance=False,
                 step_cache=0.0, block_cache=0.0, quality=None,
                 pinned_memory=False, sol_version=None, sol_enabled=None, sol_window_enabled=None,
                 sol_start_percent=None, sol_end_percent=None, sol_min_tokens=None, sol_sink_conditioning=None):
            pinned_memory = _coerce_bool_switch(pinned_memory, False, "pinned_memory")
            quality_enhance = _coerce_bool_switch(quality_enhance, False, "quality_enhance", allow_none=True)
            audio_enhance = _coerce_bool_switch(audio_enhance, False, "audio_enhance")
            p = _run_family_load(self.QF_FAMILY, transformer, model_config, pinned_memory)
            mm = _apply_common_loader_dials(
                p, attention_backend, quality_enhance, quality,
                sol_version, sol_tau, sol_enabled, sol_window_enabled,
                sol_start_percent, sol_end_percent, sol_min_tokens, sol_sink_conditioning)
            if mm is not None and hasattr(mm, "set_audio_enhance"):
                mm.set_audio_enhance(audio_enhance)
            _arm_session_caches(mm, step_cache, block_cache)
            return (p,)

    class QuantFuncNativeLoRA:
        """Unified in-place LoRA stacker via qf_xfm_set_loras (no rebuild)."""

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "model": ("MODEL",),
                "lora_name": (_lora_choices(),),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }}

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "apply"
        CATEGORY = "model/loaders"
        DESCRIPTION = (
            "Applies a LoRA to a QuantFunc model with instant in-place hot-swap (no model reload; "
            "connect after a QuantFunc loader and chain several to stack them).")

        @staticmethod
        def _refuse_foreign_lora_format(path):
            try:
                with open(path, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    if n > _MAX_LORA_HEADER_BYTES:
                        raise ValueError("implausible safetensors header size")
                    keys = [k for k in json.loads(f.read(n)) if k != "__metadata__"]
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"QuantFuncNativeLoRA: cannot read '{path}' as safetensors ({e!r})")
            conv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "qf_lora_convert.py")
            for kind, marks in (("LyCORIS (LoHa/LoKr)", (".hada_", ".lokr_")),
                                ("DoRA", (".dora_scale", "lora_magnitude_vector")),
                                ("OFT/BOFT", (".oft_", ".boft_"))):
                if any(m in k for k in keys for m in marks):
                    raise RuntimeError(
                        f"QuantFuncNativeLoRA: this is {'an' if kind[0] in 'AEIOU' else 'a'} {kind} file, "
                        "which is not supported - merge/re-export to a standard (A,B) LoRA first.")
            if any(k.startswith(("lora_unet_", "lora_transformer_", "lora_te_", "lora_te1_", "lora_te2_"))
                   for k in keys):
                raise RuntimeError(
                    "QuantFuncNativeLoRA: kohya/ai-toolkit-format LoRA detected. Convert once with:\n"
                    f"  python3 {conv} --in '{path}' --out '<same-dir>/<name>-diff.safetensors'\n"
                    "then pick the converted file in this node.")
            if not any(".lora_A." in k or ".lora_B." in k or ".lora_down." in k or ".lora_up." in k for k in keys):
                raise RuntimeError(
                    "QuantFuncNativeLoRA: no recognizable LoRA keys (lora_A/lora_B/lora_down/lora_up) in this file. "
                    f"If it is a LoRA, convert it: python3 {conv} --in ... --out ...")

        def apply(self, model, lora_name, strength):
            rebuild = qfmp.rebuild_of(model)
            if rebuild is None:
                raise RuntimeError(
                    "QuantFuncNativeLoRA: this MODEL is not a QuantFunc native model - wire it "
                    "downstream of a QuantFunc loader.")
            path = _resolve_lora(lora_name)
            try:
                self._refuse_foreign_lora_format(path)
            except RuntimeError:
                qfmp.warn_lora_not_loaded()
                raise
            stack = qfmp.lora_stack_of(model)
            stack.append({"path": path, "scale": float(strength)})
            rebuilt = rebuild(stack)
            return (rebuilt.adopt_comfy_state_from(model),)

    class QuantFuncH3LatentUpscale:
        """Latent-space upscale for MiniMax-H3 two-stage workflows."""

        @classmethod
        def INPUT_TYPES(cls):
            import nodes as _comfy_nodes
            return {"required": {
                "samples": ("LATENT", {"tooltip": "The stage 1 sampler's output (MiniMax-H3 audio+video latent)."}),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05,
                                       "tooltip": "High-res size = the stage 1 size x this, rounded to a multiple of 32."}),
                "upscale_method": (_comfy_nodes.LatentUpscaleBy.upscale_methods, {"default": "bilinear"}),
            }}

        RETURN_TYPES = ("LATENT", "INT", "INT")
        RETURN_NAMES = ("samples", "width", "height")
        FUNCTION = "upscale"
        CATEGORY = "model/latent"

        def upscale(self, samples, scale_by, upscale_method):
            import comfy.latent_formats
            import comfy.nested_tensor
            import comfy.utils
            from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE
            fmt = comfy.latent_formats.MiniMaxH3Video
            parts = samples["samples"].unbind() if isinstance(samples["samples"], comfy.nested_tensor.NestedTensor) else ()
            if len(parts) != 2 or parts[0].ndim != 5 or parts[0].shape[1] != fmt.latent_channels:
                raise ValueError("QuantFunc MiniMax-H3 Latent Upscale takes the audio+video latent a MiniMax-H3 sampler outputs")
            if samples.get("noise_mask") is not None:
                raise ValueError("QuantFunc MiniMax-H3 Latent Upscale: this latent carries a noise mask, which it cannot scale")
            px = fmt.spacial_downscale_ratio
            if CANVAS_MULTIPLE % px:
                raise ValueError(f"QuantFunc MiniMax-H3 Latent Upscale: canvas multiple {CANVAS_MULTIPLE} is not a whole "
                                 f"number of {px}-pixel latent cells")
            video, audio = parts
            out_w, out_h = (max(CANVAS_MULTIPLE, round(n * px * scale_by / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                            for n in (video.shape[-1], video.shape[-2]))
            video = comfy.utils.common_upscale(video, out_w // px, out_h // px, upscale_method, "disabled")
            out = samples.copy()
            out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
            return (out, out_w, out_h)

    NODE_CLASS_MAPPINGS.update({
        "QuantFuncLTXLoader": QuantFuncLTXLoader,
        "QuantFuncH3Loader": QuantFuncH3Loader,
        "QuantFuncKrea2Loader": QuantFuncKrea2Loader,
        "QuantFuncQwenImage21Loader": QuantFuncQwenImage21Loader,
        "QuantFuncNativeLoRA": QuantFuncNativeLoRA,
        "QuantFuncH3LatentUpscale": QuantFuncH3LatentUpscale,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "QuantFuncLTXLoader": "QuantFunc LTX-2 Loader",
        "QuantFuncH3Loader": "QuantFunc MiniMax-H3 Loader",
        "QuantFuncKrea2Loader": "QuantFunc Krea-2 Loader",
        "QuantFuncQwenImage21Loader": "QuantFunc Qwen-Image-2.1 Loader",
        "QuantFuncNativeLoRA": qfmp.NATIVE_LORA_NODE,
        "QuantFuncH3LatentUpscale": "QuantFunc MiniMax-H3 Latent Upscale",
    })

    try:
        from .qf_h3_conditioning import QuantFuncH3ImageToVideoStage2, QuantFuncH3ReferenceToVideoStage2
        NODE_CLASS_MAPPINGS.update({
            "QuantFuncH3ImageToVideoStage2": QuantFuncH3ImageToVideoStage2,
            "QuantFuncH3ReferenceToVideoStage2": QuantFuncH3ReferenceToVideoStage2,
        })
    except ImportError as _h3_cond_exc:
        _log.warning("[qf_native] H3 Stage 2 conditioning unavailable: %s", ascii(_h3_cond_exc))

    try:
        import importlib.util as _ilu
        _rc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "reject_list_completeness.py")
        _spec = _ilu.spec_from_file_location("qf_native_reject_list_completeness", _rc_path)
        _rc = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_rc)
        _rc.warn_if_stale(comfy_root=os.path.dirname(os.path.dirname(comfy.model_management.__file__)))
    except Exception as _rc_exc:  # noqa: BLE001
        _log.debug("[qf_native] reject-list self-check skipped: %s", ascii(_rc_exc))

    def _serve_api_key_route():
        srv = getattr(getattr(sys.modules.get("server"), "PromptServer", None), "instance", None)
        if srv is not None:
            qf_api_key.register_route(srv.routes, _resolve_keyfile)

    try:
        _serve_api_key_route()
    except Exception as _qf_route_exc:  # noqa: BLE001
        _log.warning("[qf_native] API key route not added: %s", ascii(_qf_route_exc))

    try:
        _qf_srv = getattr(sys.modules.get("server"), "PromptServer", None)
        if getattr(_qf_srv, "instance", None) is not None:
            qfe.start_engine_install(_comfy_device_index())
    except Exception as _qf_install_exc:  # noqa: BLE001
        _log.warning("[qf_native] engine install not started: %s", ascii(_qf_install_exc))

if _IMPORT_OK:
    try:
        _install_host_ram_release()
    except Exception as _qf_hr_exc:  # noqa: BLE001
        _log.warning("[qf_native] host-RAM release hook not installed: %s", ascii(_qf_hr_exc))

try:
    from . import qf_ltx_ancestral_audio_fix as _qf_ltx_afix
    qfe.logger(_qf_ltx_afix.__name__)
    _qf_ltx_afix.install(guard=qfe.console_safe_errors)
except Exception as _qf_ltx_afix_exc:  # noqa: BLE001
    _log.warning("[qf_native] LTX-2.5 AV audio fix not installed: %s", ascii(_qf_ltx_afix_exc))

try:
    from . import qf_engine as _qf_ll_engine
    from .qf_log_level import add_log_level_input as _qf_add_log_level
    for _qf_name, _qf_cls in list(NODE_CLASS_MAPPINGS.items()):
        if _qf_name.startswith("QuantFunc") and _qf_name.endswith("Loader"):
            qf_api_key.add_api_key_input(_qf_cls, qfmp.LOADER_API_KEY)
            _qf_add_log_level(_qf_cls, _qf_ll_engine.set_log_level)
except Exception as _qf_ll_exc:  # noqa: BLE001
    _log.warning("[qf_native] hidden loader inputs (api_key, log_level) not attached: %s", ascii(_qf_ll_exc))

if qfe is not None:
    qfe.console_safe_nodes(NODE_CLASS_MAPPINGS)
