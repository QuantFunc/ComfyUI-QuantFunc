"""Format adapters: convert any input format → standardized HF-style staging dir.

The C++ engine accepts a single canonical layout. Adapters in this package
handle each input variant (our prequant / ComfyUI single-file / bundled
multi-component checkpoint / NVFP4) and produce a small staging dir (~5 MB:
symlinks + configs + tokenizer + a quantfunc_config.json with format hints).
C++ reads the staging dir and uses PrefixStrippingProvider /
PrefixFilterProvider to view weights without copying.

Entry point:

    >>> from format_adapters import build_pipeline_inputs
    >>> result = build_pipeline_inputs(sources, context)
    >>> # result.model_dir is the staging dir to pass to quantfunc_create
"""

from .base import (
    FileRef,
    FormatAdapter,
    LoRARef,
    SourceBundle,
    BuildContext,
    StagingResult,
    UnsupportedFormatError,
)
from .factory import AdapterRegistry, adapter, build_pipeline_inputs

# Self-registering adapter modules (import for side effects)
from . import hf_native          # noqa: F401  (HF model_dir layout adapter — renamed from prequant_ours)
from . import nunchaku_svdq      # noqa: F401
from . import comfyui_unet       # noqa: F401
from . import comfyui_clip       # noqa: F401
from . import comfyui_vae        # noqa: F401
from . import comfyui_lora       # noqa: F401
from . import bundled_checkpoint  # noqa: F401
from . import comfyui_wan_remap   # noqa: F401  (Wan single-file TI2V-5B trio adapter)

__all__ = [
    "FileRef",
    "FormatAdapter",
    "LoRARef",
    "SourceBundle",
    "BuildContext",
    "StagingResult",
    "UnsupportedFormatError",
    "AdapterRegistry",
    "adapter",
    "build_pipeline_inputs",
]
