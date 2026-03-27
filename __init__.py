"""ComfyUI-QuantFunc: GPU-accelerated quantized diffusion inference via QuantFunc C API."""

import logging

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Auto-update check on startup (background, non-blocking)
try:
    from .auto_update import check_for_updates
    check_for_updates()
except Exception as e:
    logging.getLogger("QuantFunc").debug("Auto-update check skipped: %s", e)
