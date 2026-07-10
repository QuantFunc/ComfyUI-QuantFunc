"""Model auto-download and resource cache for QuantFuncModelAutoLoader node.

Handles:
- GPU variant detection: all series → 50x-below (SM<120, INT4 text-encoder) /
  50x-above (SM120+, FP4 text-encoder) base-model. Klein-4B/9B additionally
  ship 3-tier transformers + precision-configs (50x/40x/30x-below) that are
  selected per-file, independently of which base-model is downloaded.
- Base model download from HuggingFace or ModelScope
- Resource listing (transformer, prequant, precision-config) cached from ModelScope
- Data source selection controls download only; listing always from ModelScope
"""

import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading

logger = logging.getLogger("QuantFunc.ModelAutoLoader")

_IS_WINDOWS = platform.system() == "Windows"
_BIN_SUBDIR = "windows" if _IS_WINDOWS else "linux"

# ============================================================================
# Model series configuration
# ============================================================================

MODEL_SERIES_LIST = [
    "QuantFunc/Qwen-Image-Edit-Series",
    "QuantFunc/Qwen-Image-Series",
    "QuantFunc/Z-Image-Series",
    "QuantFunc/Klein-4B-Series",
    "QuantFunc/Klein-9B-Series",
]

# Config-only series: repos that ship ONLY a precision-config/ dir (no
# transformer/, prequant/, or base-model dir — weights stay on the official
# upstream; we share only the per-layer precision recipe). They must NOT appear
# in the model-download `model_series` dropdowns (nothing to download there), but
# their precision configs SHOULD be auto-discovered + offered by the Precision
# Config Auto Loader, exactly like the full series' configs.
_PRECISION_CONFIG_ONLY_SERIES = [
    "QuantFunc/Ideogram-4-Series",
    "QuantFunc/Qwen-Image-Layered-Series",
]

# Every series scanned for downloadable resources (transformer / prequant /
# precision-config). Config-only series contribute precision configs only; their
# absent subdirs are skipped gracefully by _refresh_cache_for_series. Resource
# discovery + selection-resolution iterate THIS; the model-series dropdowns in
# nodes.py keep using MODEL_SERIES_LIST (full models only).
_ALL_RESOURCE_SERIES = MODEL_SERIES_LIST + _PRECISION_CONFIG_ONLY_SERIES

# Per-series: base model directory naming pattern in the repo
# Actual dirs: qwen-image-edit-series-50x-below-base-model, etc.
_BASE_MODEL_PATTERN = {
    "QuantFunc/Qwen-Image-Edit-Series": "qwen-image-edit-series-{variant}-base-model",
    "QuantFunc/Qwen-Image-Series": "qwen-image-series-{variant}-base-model",
    "QuantFunc/Z-Image-Series": "z-image-series-{variant}-base-model",
    "QuantFunc/Klein-4B-Series": "klein-4b-series-{variant}-base-model",
    "QuantFunc/Klein-9B-Series": "klein-9b-series-{variant}-base-model",
}

_SERIES_WITH_PREQUANT = {
    "QuantFunc/Qwen-Image-Edit-Series",
    "QuantFunc/Qwen-Image-Series",
    # Klein-4B/9B ship no separate prequant/ dir (the transformer weights are
    # already prequantized in transformer/), so they are intentionally absent.
}

_SUBDIR_PREQUANT = "prequant"
_SUBDIR_TRANSFORMER = "transformer"
_SUBDIR_PRECISION_CONFIG = "precision-config"

_DATA_SOURCES = ["modelscope", "huggingface"]

# ============================================================================
# Path helpers
# ============================================================================

def _get_pkg_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _get_bin_dir():
    return os.path.join(_get_pkg_dir(), "bin", _BIN_SUBDIR)


def get_models_dir():
    """Return ComfyUI/models/QuantFunc/ directory (sibling to custom_nodes)."""
    pkg_dir = _get_pkg_dir()
    comfyui_dir = os.path.dirname(os.path.dirname(pkg_dir))
    return os.path.join(comfyui_dir, "models", "QuantFunc")


def _assert_within(base, target):
    """Guard against path traversal: `target` must resolve inside `base`.

    Catalog combo values (transformer / prequant / precision_config /
    base_model_repo) skip ComfyUI's combo validation, so a hand-edited or
    API-submitted value could contain '..' segments. Refuse any path that
    escapes the intended models directory before creating or downloading it.
    """
    real_base = os.path.realpath(base)
    real_target = os.path.realpath(target)
    if real_target != real_base and not real_target.startswith(real_base + os.sep):
        raise ValueError("Refusing path outside {}: {!r}".format(base, target))


def detect_gpu_variant(series=None):
    """Return the base-model variant for the detected GPU.

    Every series publishes exactly two base-models, picked by GPU tier:
      - '50x-above' : Blackwell (SM120+, e.g. RTX 50xx) — FP4 text-encoder base
      - '50x-below' : everything else (SM<120, incl. RTX 40 / Ada) — INT4 base

    `series` is accepted for call-site compatibility but no longer changes the
    name. The published Klein-4B/9B repos use the same 50x-above/50x-below
    layout as Qwen / Z-Image — there is NO 30x-below/50x base directory. The
    transformer tiers (30x/40x/50x) are a separate per-file choice and do not
    affect which base-model directory is downloaded.
    """
    sm = 0
    try:
        from .lib_setup import _detect_gpu_sm
        sm = _detect_gpu_sm()
    except Exception:
        sm = 0
    return "50x-above" if sm >= 120 else "50x-below"


# ============================================================================
# Unified resource cache
# ============================================================================
#
# Cache structure (resource_cache.json):
# {
#   "QuantFunc/Qwen-Image-Edit-Series": {
#     "transformer": ["name1.safetensors", "name2.safetensors"],
#     "prequant": ["weight-a.safetensors"],
#     "precision-config": ["config-a.json"]
#   },
#   ...
# }

_CACHE_FILE = "resource_cache.json"
_resource_cache = {}
_cache_lock = threading.Lock()


def _get_cache_path():
    return os.path.join(_get_bin_dir(), _CACHE_FILE)


def _load_cache():
    """Load resource cache from disk."""
    global _resource_cache
    path = _get_cache_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _resource_cache = json.load(f)
    except Exception as e:
        logger.debug("Failed to load resource cache: %s", e)
        _resource_cache = {}


def _save_cache():
    """Save resource cache to disk."""
    path = _get_cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_resource_cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.debug("Failed to save resource cache: %s", e)


def _list_local_resource_names(short, resource_type):
    """Filenames already downloaded under models/QuantFunc/<short>/<resource_type>/.

    Mirrors _download_single_file's local layout. Lets the dropdown offer a
    model the user already has on disk even when the remote catalog cache is
    empty/stale or the background refresh has not finished — which is what
    otherwise makes ComfyUI flag a perfectly-present model as "missing".
    """
    subdir = os.path.join(get_models_dir(), short, resource_type)
    if not os.path.isdir(subdir):
        return []
    ext = ".json" if resource_type == _SUBDIR_PRECISION_CONFIG else ".safetensors"
    try:
        return [n for n in os.listdir(subdir)
                if n.endswith(ext) and os.path.isfile(os.path.join(subdir, n))]
    except OSError:
        return []


def _build_dropdown(resource_type, include_none=True):
    """Build dropdown options: ['None', 'SeriesShort/name', ...].

    Merges the remote catalog cache with files already present on disk, so a
    usable value is ALWAYS in the list regardless of network reachability or
    background-refresh timing. Without the local merge, a downloaded model
    races the catalog refresh and ComfyUI rejects the saved value as a
    "missing model" until a refresh wins (or forever, if offline).
    """
    options = ["None"] if include_none else []
    seen = set()
    with _cache_lock:
        cached = {s: list(_resource_cache.get(s, {}).get(resource_type, []))
                  for s in _ALL_RESOURCE_SERIES}
    for series in _ALL_RESOURCE_SERIES:
        short = series.split("/")[-1]
        names = cached[series]
        for name in _list_local_resource_names(short, resource_type):
            if name not in names:
                names.append(name)
        for name in sorted(names):
            opt = "{}/{}".format(short, name)
            if opt not in seen:
                seen.add(opt)
                options.append(opt)
    return options if options else (["None"] if include_none else [""])


def get_prequant_options():
    return _build_dropdown(_SUBDIR_PREQUANT)


def get_precision_config_options():
    return _build_dropdown(_SUBDIR_PRECISION_CONFIG)


# ============================================================================
# Transformer GPU-compatibility (tier token → minimum SM) + [auto-detect]
# ============================================================================
#
# A transformer weight filename carries a GPU-tier token that dictates its
# quantization format and therefore the MINIMUM CUDA compute capability (SM)
# the weight can run on. Picking a weight above the GPU's SM → the engine
# __trap()s / raises "invalid configuration argument" at runtime, so the
# dropdown must (a) HIDE incompatible weights and (b) default to [auto-detect],
# which resolves to the highest-tier weight the target GPU can run.
#
# The token→SM map is CROSS-CHECKED against the engine's precision rules
# (src/gemm CLAUDE.md quant table + interop/sage_capability.h) AND mirrors the
# plugin's own precision-config picker (nodes_format_adapters.py:486-501):
#
#   token (observed in real weight names)   format               min SM
#   ------------------------------------    ------------------    ------
#   50x / 50x-above (e.g. klein-9b-50x,     FP4 W4A4              120  (Blackwell-only;
#     qwen-image-50x-above)                                            FP4 __trap()s < SM120)
#   40x   (klein-*-40x, 40x-int4-f8)        INT4 + FP8 islands     89  (FP8 needs SM89+)
#   30x-below / 50x-below (qwen 30x-below,  INT4 + INT8 islands    75  (INT4/INT8 target SM80;
#     50x-below-int4)                                                  SM75 via BF16 fallback)
#
# An unrecognized name (no tier token) → min SM 0 → NEVER filtered out
# (conservative: an unknown weight is always offered).
AUTO_DETECT = "[auto-detect]"

_SM_BLACKWELL = 120   # FP4 W4A4 tensor cores (RTX 50xx)
_SM_ADA_FP8 = 89      # FP8 tensor cores (RTX 40xx / Hopper)
_SM_TURING = 75       # lowest supported GPU; the INT4/INT8 tier BF16-fallbacks here
_SM_UNKNOWN = 0       # SM could not be determined → do not filter


def _has_tier_token(name, token):
    """True if `token` appears in `name` as a whole '-'/'_'/'.'-delimited segment.

    Anchored (not a raw substring) so an unrelated digit run can't collide — e.g.
    a resolution-tagged '...1440x...' must NOT match the '40x' tier token."""
    return re.search(r"(?:^|[-_.])" + re.escape(token) + r"(?=$|[-_.])", name) is not None


# Ordered (tier-token → minimum-SM) table. ORDER MATTERS and is the whole point
# of the ordered list: '50x-below' / '30x-below' (the INT4 tier — they merely NAME
# the GPUs they run *below*) must be matched BEFORE the bare '50x' FP4 token, else
# '50x-below' would be misread as FP4. First delimiter-anchored match wins.
# Data-driven so a FUTURE tier is ONE new row here — no logic edit per tier.
#
# The tokens (30x/40x/50x) are CONSUMER-tier names, but the min-SM VALUES are what
# the whole filter/auto-detect compares against — so DATACENTER GPUs are handled
# purely by NUMERIC min-SM comparison, no consumer-name aliasing needed:
#   A100 (SM80) → INT4 (30x tier); H100/H200 (SM90) → FP8 (40x tier);
#   B200/GB200 (SM100) → FP8 (40x tier); RTX 50xx (SM120) → FP4 (50x tier).
_TIER_TOKEN_MAP = [
    ("50x-below", _SM_TURING),     # INT4 (Qwen/Z-Image variant that "runs below 50x")
    ("30x",       _SM_TURING),     # INT4 + INT8 islands (30x-below); SM75 via BF16 fallback
    ("40x",       _SM_ADA_FP8),    # INT4 + FP8 islands — FP8 tensor cores need SM89+
    # FP4 = SM120 ONLY today. The engine's sage3 + W4A4 FP4 is built sm_120a-only
    # (CUDA_ARCHITECTURES "120a"); SM100 (sm_100a datacenter Blackwell, e.g.
    # B200/GB200) is a DIFFERENT arch with NO engine FP4 path, so an SM100 card
    # correctly falls to FP8 (SM89+) via numeric comparison — NOT this FP4 tier.
    # If the engine ever ships an sm_100a FP4 build, extend the FP4-eligible set
    # to include SM100 (e.g. a second FP4 row / an SM-set) — do NOT lower this to 100.
    ("50x",       _SM_BLACKWELL),  # 50x-above / bare 50x → FP4 W4A4, Blackwell SM120+
]


def _transformer_min_sm(filename):
    """Minimum CUDA SM the transformer weight `filename` can run on.

    Returns _SM_UNKNOWN (0) for a name with no recognized tier token, so an
    unrecognized weight is never hidden. Iterates _TIER_TOKEN_MAP in priority
    order; the first delimiter-anchored token match wins (see the map's ordering
    note for why '50x-below'/'30x' precede the bare '50x' FP4 token)."""
    n = (filename or "").lower()
    for token, min_sm in _TIER_TOKEN_MAP:
        if _has_tier_token(n, token):
            return min_sm
    return _SM_UNKNOWN


# Memoized once — the default CUDA device is fixed for the process, and
# get_transformer_options() runs on ComfyUI's INPUT_TYPES / /object_info hot path,
# so we must NOT re-query on every call. A failed/0 probe is NOT cached, so a
# transient hiccup simply retries next time. (Module-level so tests can stub it.)
_default_device_sm_cache = None


def _default_device_sm():
    """SM of the DEFAULT CUDA device (torch device 0) — the device BuildPipeline
    runs the transformer on by default. Memoized; 0 if no CUDA / CPU-only."""
    global _default_device_sm_cache
    if _default_device_sm_cache:
        return _default_device_sm_cache
    try:
        from .lib_setup import _detect_default_device_sm
        sm = _detect_default_device_sm()
    except Exception:
        sm = _SM_UNKNOWN
    if sm:
        _default_device_sm_cache = sm
    return sm


def _target_gpu_sm():
    """The SM to key transformer GPU-compatibility on: the DEFAULT CUDA device
    (torch device 0) — the GPU BuildPipeline runs the transformer on by default.

    Keying on the ACTUAL default run-device (NOT max/min over the whole GPU set)
    means a weight offered / auto-picked here is runnable on that device under ANY
    `CUDA_VISIBLE_DEVICES` mask or `CUDA_DEVICE_ORDER` — torch's "device 0" is the
    same index BuildPipeline defaults to (and the worker inherits the identical
    CVD/order env), so the keyed GPU and the run-device agree and it will not
    __trap — WHEN BuildPipeline's `device` is left at its default (0). Under the
    default FASTEST_FIRST ordering device 0 IS the best GPU, so on a 3060+4090 box
    device 0 is the 4090 (SM89) → the 40x/FP8 tier shows + auto-picks (the user's
    reported ask). Undetectable (CPU-only / no torch) → _SM_UNKNOWN (0), and the
    caller then does NOT filter.

    KNOWN LIMITATION (accepted trade-off — the orchestrator chose device-0 keying
    OVER a build()-time SM-vs-device guardrail): this node has no wiring to
    BuildPipeline's `device` selector, so if the user MANUALLY routes the pipeline
    to a NON-DEFAULT, WEAKER GPU (e.g. device 1 = the 3060 while device 0 = the
    4090), an auto-picked tier keyed to device 0 could exceed the chosen device's
    SM and __trap at run time. In that case pick a matching lower-tier weight
    explicitly. (Earlier keyings were WORSE, not merely limited: MIN over all GPUs
    over-hid the user's 40x on EVERY config; MAX over the physical set could offer
    a tier the CVD-visible / PCI-slot-0 DEFAULT run-device couldn't run — a __trap
    on the common no-override path, which device-0 keying eliminates.)"""
    return _default_device_sm()


def _available_transformer_names(model_series):
    """Transformer filenames known for `model_series` (remote catalog ∪ on-disk)."""
    short = model_series.split("/")[-1]
    with _cache_lock:
        names = list(_resource_cache.get(model_series, {}).get(_SUBDIR_TRANSFORMER, []))
    for name in _list_local_resource_names(short, _SUBDIR_TRANSFORMER):
        if name not in names:
            names.append(name)
    return names


def _pick_best_transformer(names, sm):
    """From candidate filenames, pick the highest-tier one runnable at `sm`.

    Returns None (→ caller defers to the base model's own transformer) when
    `names` is empty OR — critically — when the GPU is KNOWN (sm > 0) but NO
    RECOGNIZED tier is compatible. Returning a known-incompatible weight would
    reproduce the exact __trap() this feature prevents, so we must NOT fall
    through to 'lowest tier' in that case. Only when the GPU is UNKNOWN
    (sm == 0, cannot filter) do we best-effort the LOWEST KNOWN tier (runs on the
    most GPUs). 'Highest-tier compatible' = the largest min-SM ≤ sm (a LOWER-tier
    weight on a HIGHER GPU is fine — INT4 runs on Blackwell). Ties break by name
    (deterministic).

    An UNRECOGNIZED token scores _SM_UNKNOWN (0) = "unknown compatibility", which
    is NOT "runs on any GPU" — auto-detect must never PICK it (it may need a
    higher SM → __trap). So BOTH branches exclude ms == _SM_UNKNOWN from the
    auto-pick pool (sm>0: from `compat`; sm==0: prefer `known`), falling back to
    None / an unknown name only when there is nothing recognized to choose. The
    dropdown still SHOWS unrecognized weights (get_transformer_options never
    hides ms==0) for a deliberate MANUAL pick — auto-detect is known-safe-only."""
    scored = [(n, _transformer_min_sm(n)) for n in names]
    if not scored:
        return None
    if sm > 0:
        # RECOGNIZED (_SM_UNKNOWN < ms) AND runnable (ms <= sm). ms==0 is excluded:
        # it means "unknown", not "runs everywhere" — mirrors the sm==0 branch.
        compat = [(n, ms) for (n, ms) in scored if _SM_UNKNOWN < ms <= sm]
        if not compat:
            return None  # no recognized-compatible weight → base model's default transformer
        top = max(ms for (_n, ms) in compat)
        return sorted(n for (n, ms) in compat if ms == top)[0]
    # GPU SM unknown → best-effort the lowest KNOWN tier (runs on the most GPUs);
    # only fall back to an unknown-token name when EVERY candidate is unrecognized.
    known = [(n, ms) for (n, ms) in scored if ms > _SM_UNKNOWN]
    pool = known if known else scored
    low = min(ms for (_n, ms) in pool)
    return sorted(n for (n, ms) in pool if ms == low)[0]


def get_transformer_options():
    """Transformer dropdown: ['[auto-detect]', 'None', <GPU-compatible weights>].

    '[auto-detect]' (the node default) resolves at load time to the highest-tier
    weight the target GPU can run. Weights whose min-SM exceeds the target GPU
    are HIDDEN (e.g. a 50x FP4 weight is not offered on an SM86 card), so the
    user cannot pick a weight that __trap()s at runtime. Both the remote-catalog
    names and the on-disk-merged names (from `_build_dropdown`) are filtered.
    When the GPU SM is undetectable, nothing is filtered (conservative)."""
    base = _build_dropdown(_SUBDIR_TRANSFORMER)  # ['None', 'Short/name', ...]
    sm = _target_gpu_sm()
    kept = []
    for opt in base:
        if opt == "None":
            kept.append(opt)
            continue
        name = opt.split("/", 1)[1] if "/" in opt else opt
        if sm > 0 and _transformer_min_sm(name) > sm:
            continue  # weight cannot run on the target GPU → hide it
        kept.append(opt)
    return [AUTO_DETECT] + kept


# ============================================================================
# ModelScope file listing (single source for cache data)
# ============================================================================

def _ensure_modelscope():
    """Return True if modelscope is importable; do NOT install it at runtime.

    Runtime `pip install` from a custom node is prohibited by the ComfyUI registry
    security policy (it bypasses the user's package manager). modelscope is declared
    in requirements.txt and installed at install time by ComfyUI-Manager / pip; if it
    is missing we fail gracefully with an actionable message instead of installing.
    """
    try:
        import modelscope  # noqa: F401
        return True
    except ImportError:
        print("[QuantFunc] modelscope is not installed; ModelScope features are "
              "unavailable. Install it with `pip install modelscope` (declared in "
              "requirements.txt) and restart ComfyUI.")
        return False


def _ensure_huggingface_hub():
    """Return True if huggingface_hub is importable; do NOT install it at runtime.

    Same ComfyUI-registry-compliance reasoning as _ensure_modelscope: no runtime
    `pip install`. huggingface_hub is declared in requirements.txt; if missing we
    fail gracefully instead of installing.
    """
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        print("[QuantFunc] huggingface_hub is not installed; HuggingFace features "
              "are unavailable. Install it with `pip install huggingface_hub` "
              "(declared in requirements.txt) and restart ComfyUI.")
        return False


def _list_ms_dir(repo_id, subdir):
    """List files in a ModelScope repo subdirectory via SDK.
    Returns list of dicts with 'Name' and 'Type' keys, or None on failure.
    """
    try:
        if not _ensure_modelscope():
            return None
        from modelscope.hub.api import HubApi
        api = HubApi()
        items = api.get_model_files(model_id=repo_id, root=subdir)
        return items
    except Exception as e:
        logger.debug("MS listing %s/%s failed: %s", repo_id, subdir, e)
        return None


def _list_files_in_subdir(repo_id, subdir, extension=None):
    """List filenames in a repo subdirectory, filtered by extension."""
    items = _list_ms_dir(repo_id, subdir)
    if items is None:
        return None
    names = [f["Name"] for f in items
             if f.get("Type") == "blob" and f.get("Name")]
    if extension:
        names = [n for n in names if n.endswith(extension)]
    return names


# ============================================================================
# Cache refresh
# ============================================================================

def _refresh_cache_for_series(series):
    """Refresh all resource caches for one series from ModelScope."""
    series_cache = {}
    updated = False

    # Transformer: .safetensors files in transformer/
    tf_files = _list_files_in_subdir(series, _SUBDIR_TRANSFORMER, ".safetensors")
    if tf_files is not None:
        series_cache[_SUBDIR_TRANSFORMER] = sorted(tf_files)
        updated = True

    # Prequant: .safetensors files in prequant/ (only for series that have it)
    if series in _SERIES_WITH_PREQUANT:
        pq_files = _list_files_in_subdir(series, _SUBDIR_PREQUANT, ".safetensors")
        if pq_files is not None:
            series_cache[_SUBDIR_PREQUANT] = sorted(pq_files)
            updated = True

    # Precision config: .json files in precision-config/
    pc_files = _list_files_in_subdir(series, _SUBDIR_PRECISION_CONFIG, ".json")
    if pc_files is not None:
        series_cache[_SUBDIR_PRECISION_CONFIG] = sorted(pc_files)
        updated = True

    if updated:
        with _cache_lock:
            if series not in _resource_cache:
                _resource_cache[series] = {}
            _resource_cache[series].update(series_cache)

    return updated


def _refresh_all_caches():
    """Refresh resource caches for all series."""
    any_updated = False
    for series in _ALL_RESOURCE_SERIES:
        try:
            if _refresh_cache_for_series(series):
                any_updated = True
                logger.info("[QuantFunc] Resource cache updated for %s", series)
        except Exception as e:
            logger.debug("Cache refresh failed for %s: %s", series, e)

    if any_updated:
        with _cache_lock:
            _save_cache()
        print("[QuantFunc] Model resource cache updated successfully")


def refresh_cache_background():
    """Start background thread to refresh all resource caches."""
    t = threading.Thread(target=_refresh_all_caches, daemon=True,
                         name="QuantFunc-ResourceCache")
    t.start()


# ============================================================================
# Model & resource download
# ============================================================================

_DOWNLOAD_MARKER = ".quantfunc_download_complete"

# Vision encoder model.safetensors should be ~650MB; anything under 600MB is corrupt/truncated
_VISION_ENCODER_MIN_SIZE = 600 * 1024 * 1024  # 600 MB


def _check_vision_encoder(local_dir, marker_path):
    """Check vision_encoder/model.safetensors size for Qwen-Edit models.

    If the file exists but is smaller than 600MB, it's likely corrupt or
    truncated. Delete it, remove the download marker and quantfunc_config.json
    so they get re-downloaded together.
    """
    ve_model = os.path.join(local_dir, "vision_encoder", "model.safetensors")
    if not os.path.exists(ve_model):
        return
    size = os.path.getsize(ve_model)
    if size < _VISION_ENCODER_MIN_SIZE:
        size_mb = size / (1024 * 1024)
        print("[QuantFunc] vision_encoder/model.safetensors is too small "
              "({:.1f}MB < 600MB), likely corrupt. Re-downloading...".format(size_mb))
        try:
            os.remove(ve_model)
            # Remove quantfunc_config.json so it gets refreshed on re-download
            nc_path = os.path.join(local_dir, "quantfunc_config.json")
            if os.path.exists(nc_path):
                os.remove(nc_path)
                print("[QuantFunc] Removed quantfunc_config.json for re-download")
            if os.path.exists(marker_path):
                os.remove(marker_path)
        except OSError as e:
            print("[QuantFunc] Failed to remove corrupt file: {}".format(e))


def download_base_model(series, gpu_variant, data_source):
    """Download base model. Returns local model directory path.

    Uses a marker file (.quantfunc_download_complete) to track completeness.
    If marker is missing, removes any partial download and re-downloads.
    """
    pattern = _BASE_MODEL_PATTERN.get(series)
    if not pattern:
        raise RuntimeError("Unknown model series: {}".format(series))

    remote_dir = pattern.format(variant=gpu_variant)
    short_name = series.split("/")[-1]
    local_base = os.path.join(get_models_dir(), short_name)
    local_dir = os.path.join(local_base, remote_dir)
    marker = os.path.join(local_dir, _DOWNLOAD_MARKER)

    # Already downloaded and verified?
    if os.path.exists(marker):
        # Qwen-Edit: check vision_encoder model.safetensors integrity
        if "image-edit" in series.lower():
            _check_vision_encoder(local_dir, marker)
        # Re-check marker (may have been removed by integrity check)
        if os.path.exists(marker):
            return local_dir

    os.makedirs(local_base, exist_ok=True)
    if os.path.isdir(local_dir):
        print("[QuantFunc] Resuming incomplete download: {}/{}...".format(
            series, remote_dir))
    else:
        print("[QuantFunc] Downloading base model: {}/{} from {}...".format(
            series, remote_dir, data_source))

    if data_source == "huggingface":
        if not _ensure_huggingface_hub():
            raise RuntimeError("huggingface_hub is required but not installed (declared in requirements.txt)")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=series,
            allow_patterns=["{}/**".format(remote_dir)],
            local_dir=local_base,
        )
    else:
        if not _ensure_modelscope():
            raise RuntimeError("modelscope is required but not installed (declared in requirements.txt)")
        from modelscope import snapshot_download as ms_download
        ms_download(
            model_id=series,
            allow_patterns=["{}/**".format(remote_dir)],
            local_dir=local_base,
        )

    if not os.path.exists(os.path.join(local_dir, "model_index.json")):
        raise RuntimeError(
            "Download completed but model_index.json not found in {}.\n"
            "Check the repo structure.".format(local_dir))

    # Mark as complete
    with open(marker, "w") as f:
        f.write("ok")
    print("[QuantFunc] Base model ready: {}".format(local_dir))
    return local_dir


def _download_single_file(series, remote_path, data_source):
    """Download a single file from repo. Returns local file path."""
    short_name = series.split("/")[-1]
    local_base = os.path.join(get_models_dir(), short_name)
    local_file = os.path.join(local_base, remote_path)
    # Guard against the per-series base, not just models/QuantFunc/: a crafted
    # '..' in the resolved filename must not redirect the write into another
    # series' subdirectory.
    _assert_within(local_base, local_file)

    if os.path.exists(local_file):
        return local_file

    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    print("[QuantFunc] Downloading: {}/{} from {}...".format(
        series, remote_path, data_source))

    if data_source == "huggingface":
        if not _ensure_huggingface_hub():
            raise RuntimeError("huggingface_hub is required but not installed (declared in requirements.txt)")
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=series, filename=remote_path, local_dir=local_base)
    else:
        if not _ensure_modelscope():
            raise RuntimeError("modelscope is required but not installed (declared in requirements.txt)")
        from modelscope.hub.file_download import model_file_download
        model_file_download(
            model_id=series, file_path=remote_path, local_dir=local_base)

    if not os.path.exists(local_file):
        raise RuntimeError("Download failed: {}".format(local_file))

    return local_file


def download_transformer(series, filename, data_source):
    """Download transformer .safetensors file. Returns local path."""
    remote_path = "{}/{}".format(_SUBDIR_TRANSFORMER, filename)
    path = _download_single_file(series, remote_path, data_source)
    print("[QuantFunc] Transformer ready: {}".format(path))
    return path


def download_prequant(series, filename, data_source):
    """Download prequant weight file. Returns local path."""
    remote_path = "{}/{}".format(_SUBDIR_PREQUANT, filename)
    path = _download_single_file(series, remote_path, data_source)
    print("[QuantFunc] Prequant weights ready: {}".format(path))
    return path


def download_precision_config(series, filename, data_source):
    """Download precision config file. Returns local path."""
    remote_path = "{}/{}".format(_SUBDIR_PRECISION_CONFIG, filename)
    path = _download_single_file(series, remote_path, data_source)
    print("[QuantFunc] Precision config ready: {}".format(path))
    return path


# ============================================================================
# Selection resolution — parse dropdown "SeriesShort/name" format
# ============================================================================

def _resolve_selection(selection, model_series, resource_label):
    """Parse a 'SeriesShort/name' dropdown value.
    Returns (series_full_name, name) or (None, None) if 'None'.
    Validates match with model_series.
    """
    if not selection or selection == "None":
        return None, None

    if "/" not in selection:
        return None, None

    short_name, name = selection.split("/", 1)
    for s in _ALL_RESOURCE_SERIES:
        if s.endswith("/" + short_name):
            if s != model_series:
                raise ValueError(
                    "{} '{}' belongs to {} but selected model series is {}. "
                    "Please select a matching option or 'None'.".format(
                        resource_label, name, s, model_series))
            return s, name

    raise ValueError("Unknown series in {} selection: {}".format(
        resource_label, short_name))


def resolve_selection_no_series(selection, resource_label):
    """Parse a 'SeriesShort/name' dropdown value without model_series validation.
    Returns (series_full_name, name) or (None, None) if 'None'.
    """
    if not selection or selection == "None":
        return None, None
    if "/" not in selection:
        return None, None
    short_name, name = selection.split("/", 1)
    for s in _ALL_RESOURCE_SERIES:
        if s.endswith("/" + short_name):
            return s, name
    raise ValueError("Unknown series in {} selection: {}".format(
        resource_label, short_name))


def resolve_transformer_selection(selection, model_series):
    """Resolve a transformer dropdown value to (series_full_name, name).

    '[auto-detect]' (the default) → the highest-tier weight the target GPU can
    run, chosen from `model_series`'s available weights; (None, None) when the
    series ships no separate transformer weights OR none are compatible with the
    detected GPU (→ use the base model's own transformer, which is GPU-tier-matched
    by detect_gpu_variant). Any explicit 'SeriesShort/name' value still resolves
    exactly as before (backward compatible)."""
    if selection == AUTO_DETECT:
        names = _available_transformer_names(model_series)
        best = _pick_best_transformer(names, _target_gpu_sm())
        if not best:
            return None, None
        return model_series, best
    return _resolve_selection(selection, model_series, "Transformer")


def resolve_prequant_selection(selection, model_series):
    return _resolve_selection(selection, model_series, "Prequant")


def resolve_precision_config_selection(selection, model_series):
    return _resolve_selection(selection, model_series, "Precision config")


# ============================================================================
# Base model repo discovery and download
# ============================================================================

# Search configs: (org, keyword_filter)
_BASE_MODEL_SEARCH_CONFIGS = [
    ("Qwen", lambda name: "image" in name.lower()),
    ("Tongyi-MAI", lambda name: "z-image" in name.lower() or "zimage" in name.lower()),
    # Flux.2 Klein 4B/9B — original-precision checkpoints. Both the distilled
    # (FLUX.2-klein-{4B,9B}) and the non-distilled base (FLUX.2-klein-base-{4B,9B})
    # variants carry "klein" in the name, so a single filter finds all four.
    ("black-forest-labs", lambda name: "klein" in name.lower()),
]

_BASE_MODEL_CACHE_KEY = "__base_model_repos__"
_BASE_MODEL_FALLBACK = [
    "Qwen/Qwen-Image",
    "Qwen/Qwen-Image-2512",
    "Qwen/Qwen-Image-Edit",
    "Qwen/Qwen-Image-Edit-2509",
    "Qwen/Qwen-Image-Edit-2511",
    "Qwen/Qwen-Image-Layered",
    "Tongyi-MAI/Z-Image",
    "Tongyi-MAI/Z-Image-Turbo",
    # Flux.2 Klein 4B/9B — "base" = non-distilled, no "base" = distilled
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-klein-base-4B",
    "black-forest-labs/FLUX.2-klein-9B",
    "black-forest-labs/FLUX.2-klein-base-9B",
]
_base_model_repos = []  # list of "org/repo" strings
_base_model_lock = threading.Lock()


def _search_base_model_repos():
    """Search ModelScope for available base model repositories."""
    if not _ensure_modelscope():
        return []
    from modelscope.hub.api import HubApi
    api = HubApi()
    repos = []
    for org, name_filter in _BASE_MODEL_SEARCH_CONFIGS:
        try:
            result = api.list_models(org, page_size=100)
            models = result.get("Models", []) if isinstance(result, dict) else result
            if not models:
                continue
            for m in models:
                name = m.get("Name", "") or ""
                path = m.get("Path", "") or org
                model_id = "{}/{}".format(path, name)
                if name_filter(name):
                    if model_id not in repos:
                        repos.append(model_id)
        except Exception as e:
            logger.debug("Base model search failed for %s: %s", org, e)
    return sorted(repos)


def _refresh_base_model_repos():
    """Refresh the list of available base model repos."""
    global _base_model_repos
    # ModelScope is a China service; proxies may interfere
    saved = {}
    for key in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    try:
        repos = _search_base_model_repos()
    finally:
        os.environ.update(saved)
    if repos:
        with _base_model_lock:
            _base_model_repos = repos
            # Persist to resource cache
            _resource_cache[_BASE_MODEL_CACHE_KEY] = repos
            _save_cache()
        print("[QuantFunc] Base model repos discovered: {}".format(repos))


def _load_base_model_repos_from_cache():
    """Load base model repo list from resource cache."""
    global _base_model_repos
    with _base_model_lock:
        cached = _resource_cache.get(_BASE_MODEL_CACHE_KEY, [])
        if cached:
            _base_model_repos = cached


def get_base_model_repo_options():
    """Get dropdown options for base model repos."""
    with _base_model_lock:
        return list(_base_model_repos) if _base_model_repos else list(_BASE_MODEL_FALLBACK)


def refresh_base_model_repos_background():
    """Start background thread to refresh base model repo list."""
    t = threading.Thread(target=_refresh_base_model_repos, daemon=True,
                         name="QuantFunc-BaseModelRepoSearch")
    t.start()


def download_base_model_repo(repo_id, data_source):
    """Download a base model from its upstream repo (e.g. Qwen/Qwen-Image-2512).

    Downloads the full repo to ComfyUI/models/QuantFunc/<repo_name>/.
    Returns local model directory path.
    """
    # Use org/repo structure: models/QuantFunc/Qwen/Qwen-Image-2512/
    local_dir = os.path.join(get_models_dir(), *repo_id.split("/"))
    _assert_within(get_models_dir(), local_dir)
    marker = os.path.join(local_dir, _DOWNLOAD_MARKER)

    if os.path.exists(marker):
        # Check vision_encoder integrity for Qwen-Image-Edit models
        if "image-edit" in repo_id.lower():
            _check_vision_encoder(local_dir, marker)
        if os.path.exists(marker):
            return local_dir

    os.makedirs(local_dir, exist_ok=True)
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print("[QuantFunc] Resuming incomplete download: {}...".format(repo_id))
    else:
        print("[QuantFunc] Downloading base model: {} from {}...".format(
            repo_id, data_source))

    if data_source == "huggingface":
        if not _ensure_huggingface_hub():
            raise RuntimeError("huggingface_hub is required but not installed (declared in requirements.txt)")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
    else:
        if not _ensure_modelscope():
            raise RuntimeError("modelscope is required but not installed (declared in requirements.txt)")
        from modelscope import snapshot_download as ms_download
        ms_download(model_id=repo_id, local_dir=local_dir)

    if not os.path.exists(os.path.join(local_dir, "model_index.json")):
        raise RuntimeError(
            "Download completed but model_index.json not found in {}.\n"
            "Check the repo structure.".format(local_dir))

    with open(marker, "w") as f:
        f.write("ok")
    print("[QuantFunc] Base model ready: {}".format(local_dir))
    return local_dir


def download_base_model_to_diffusers(repo_id, data_source):
    """Download a base model to ComfyUI/models/diffusers/<org>/<repo>/.

    Returns local model directory path.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    comfyui_dir = os.path.dirname(os.path.dirname(pkg_dir))
    local_dir = os.path.join(comfyui_dir, "models", "diffusers", *repo_id.split("/"))
    _assert_within(os.path.join(comfyui_dir, "models", "diffusers"), local_dir)
    marker = os.path.join(local_dir, _DOWNLOAD_MARKER)

    if os.path.exists(marker):
        if "image-edit" in repo_id.lower():
            _check_vision_encoder(local_dir, marker)
        if os.path.exists(marker):
            return local_dir

    os.makedirs(local_dir, exist_ok=True)
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print("[QuantFunc] Resuming incomplete download: {}...".format(repo_id))
    else:
        print("[QuantFunc] Downloading base model: {} from {} to models/diffusers/...".format(
            repo_id, data_source))

    if data_source == "huggingface":
        if not _ensure_huggingface_hub():
            raise RuntimeError("huggingface_hub is required but not installed (declared in requirements.txt)")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
    else:
        if not _ensure_modelscope():
            raise RuntimeError("modelscope is required but not installed (declared in requirements.txt)")
        from modelscope import snapshot_download as ms_download
        ms_download(model_id=repo_id, local_dir=local_dir)

    if not os.path.exists(os.path.join(local_dir, "model_index.json")):
        raise RuntimeError(
            "Download completed but model_index.json not found in {}.\n"
            "Check the repo structure.".format(local_dir))

    with open(marker, "w") as f:
        f.write("ok")
    print("[QuantFunc] Base model ready: {}".format(local_dir))
    return local_dir


# ── Load cache on import ──
_load_cache()
_load_base_model_repos_from_cache()
