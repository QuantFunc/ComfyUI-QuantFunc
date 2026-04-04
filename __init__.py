"""ComfyUI-QuantFunc: GPU-accelerated quantized diffusion inference via QuantFunc C API."""

# Pull latest plugin code before importing anything else so this process loads the newest modules
import subprocess as _subprocess, os as _os, logging
_plugin_dir = _os.path.dirname(_os.path.abspath(__file__))
try:
    _r = _subprocess.run(
        ["git", "pull", "--rebase"],
        cwd=_plugin_dir, capture_output=True, text=True, timeout=30,
    )
    if _r.returncode == 0 and "Already up to date" not in _r.stdout:
        print("[QuantFunc] Plugin updated: {}".format(_r.stdout.strip()))
except Exception:
    pass

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Ensure models directories exist
try:
    from .model_auto_loader import get_models_dir
    _models = get_models_dir()
    _os.makedirs(_os.path.join(_models, "lora"), exist_ok=True)
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
