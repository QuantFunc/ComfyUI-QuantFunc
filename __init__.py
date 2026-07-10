"""ComfyUI-QuantFunc: GPU-accelerated quantized diffusion inference via QuantFunc C API."""

import os as _os, logging
_plugin_dir = _os.path.dirname(_os.path.abspath(__file__))

# Plugin updates are EXPLICIT, never silent. By default this plugin does NOT
# self-update its own Python code on import. A silent `git pull --rebase` at
# startup would fetch and then execute remote code on every ComfyUI launch —
# i.e. unattended remote-code-execution and a supply-chain risk. Update the
# plugin through ComfyUI-Manager instead (the user's chosen update channel).
# Power users who deliberately want the old auto-pull behavior can opt in by
# setting the environment variable QUANTFUNC_PLUGIN_AUTOPULL=1.
if _os.environ.get("QUANTFUNC_PLUGIN_AUTOPULL") == "1":
    try:
        import subprocess as _subprocess
        _r = _subprocess.run(
            ["git", "pull", "--rebase"],
            cwd=_plugin_dir, capture_output=True, text=True, timeout=30,
        )
        if _r.returncode == 0 and "Already up to date" not in _r.stdout:
            print("[QuantFunc] Plugin updated (QUANTFUNC_PLUGIN_AUTOPULL=1): {}".format(_r.stdout.strip()))
    except Exception:
        pass

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Sprint 1: format-adapter nodes (LoadDiffusionModel / LoadCLIP / LoadVAE /
# LoadCheckpoint / LoadLoRA / LoadPrecisionMap / BuildPipeline). These plug
# into ComfyUI's standard model directories and feed QuantFuncGenerate.
try:
    from .nodes_format_adapters import (
        NODE_CLASS_MAPPINGS as _FA_NODE_CLS,
        NODE_DISPLAY_NAME_MAPPINGS as _FA_NODE_NAMES,
    )
    NODE_CLASS_MAPPINGS.update(_FA_NODE_CLS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_FA_NODE_NAMES)
except Exception as _e:
    logging.getLogger("QuantFunc").warning(
        "format_adapters nodes failed to load: %s", _e)

# Install monkey-patches on comfy.sd loaders so the loaded MODEL/CLIP/VAE
# objects carry the source file path — needed because ComfyUI itself does
# not retain it. Required for QuantFuncBuildPipeline to accept official
# loader outputs (UNETLoader / CLIPLoader / VAELoader / CheckpointLoaderSimple).
try:
    from .nodes_pipeline_builder import install_loader_path_patches as _pb_install_patches
    _pb_install_patches()
except Exception as _e:
    logging.getLogger("QuantFunc").warning(
        "comfy loader path patches failed to install: %s", _e)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Ensure models directories exist
try:
    from .model_auto_loader import get_models_dir
    _models = get_models_dir()
    _os.makedirs(_os.path.join(_models, "transformer"), exist_ok=True)
except Exception:
    pass

# Auto-update check on startup (background, non-blocking)
try:
    from .auto_update import check_for_updates
    check_for_updates()
except Exception as e:
    logging.getLogger("QuantFunc").debug("Auto-update check skipped: %s", e)

# Refresh resource cache for ModelAutoLoader dropdowns (background)
try:
    from .model_auto_loader import refresh_cache_background
    refresh_cache_background()
except Exception as e:
    logging.getLogger("QuantFunc").debug("Resource cache refresh skipped: %s", e)

# Discover base model repos for BaseModelAutoLoader dropdowns (background)
try:
    from .model_auto_loader import refresh_base_model_repos_background
    refresh_base_model_repos_background()
except Exception as e:
    logging.getLogger("QuantFunc").debug("Base model repo discovery skipped: %s", e)
