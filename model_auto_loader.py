"""Model auto-download and resource cache for QuantFuncModelAutoLoader node.

Handles:
- GPU variant detection (50x-below / 50x-above based on SM version)
- Base model download from HuggingFace or ModelScope
- Resource listing (transformer, prequant, precision-config) cached from ModelScope
- Data source selection controls download only; listing always from ModelScope
"""

import json
import logging
import os
import platform
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
]

# Per-series: base model directory naming pattern in the repo
# Actual dirs: qwen-image-edit-series-50x-below-base-model, etc.
_BASE_MODEL_PATTERN = {
    "QuantFunc/Qwen-Image-Edit-Series": "qwen-image-edit-series-{variant}-base-model",
    "QuantFunc/Qwen-Image-Series": "qwen-image-series-{variant}-base-model",
    "QuantFunc/Z-Image-Series": "z-image-series-{variant}-base-model",
}

_SERIES_WITH_PREQUANT = {
    "QuantFunc/Qwen-Image-Edit-Series",
    "QuantFunc/Qwen-Image-Series",
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


def detect_gpu_variant():
    """Return '50x-above' for SM120+ GPUs (RTX 50 series), '50x-below' otherwise."""
    try:
        from .lib_setup import _detect_gpu_sm
        sm = _detect_gpu_sm()
        if sm >= 120:
            return "50x-above"
    except Exception:
        pass
    return "50x-below"


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


def _build_dropdown(resource_type, include_none=True):
    """Build dropdown options: ['None', 'SeriesShort/name', ...]."""
    options = ["None"] if include_none else []
    with _cache_lock:
        for series in MODEL_SERIES_LIST:
            short = series.split("/")[-1]
            for name in _resource_cache.get(series, {}).get(resource_type, []):
                options.append("{}/{}".format(short, name))
    return options if options else (["None"] if include_none else [""])


def get_transformer_options():
    return _build_dropdown(_SUBDIR_TRANSFORMER)


def get_prequant_options():
    return _build_dropdown(_SUBDIR_PREQUANT)


def get_precision_config_options():
    return _build_dropdown(_SUBDIR_PRECISION_CONFIG)


# ============================================================================
# ModelScope file listing (single source for cache data)
# ============================================================================

def _ensure_modelscope():
    """Install modelscope if not available."""
    try:
        import modelscope  # noqa: F401
        return True
    except ImportError:
        print("[QuantFunc] Installing modelscope...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "modelscope", "-q"],
                stdout=subprocess.DEVNULL)
            print("[QuantFunc] modelscope installed successfully")
            return True
        except Exception as e:
            print("[QuantFunc] Failed to install modelscope: {}".format(e))
            return False


def _ensure_huggingface_hub():
    """Install huggingface_hub if not available."""
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        print("[QuantFunc] Installing huggingface_hub...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"],
                stdout=subprocess.DEVNULL)
            print("[QuantFunc] huggingface_hub installed successfully")
            return True
        except Exception as e:
            print("[QuantFunc] Failed to install huggingface_hub: {}".format(e))
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
    for series in MODEL_SERIES_LIST:
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
    truncated. Delete it and remove the download marker so it gets re-downloaded.
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
        if "Qwen-Image-Edit" in series:
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
            raise RuntimeError("Cannot install huggingface_hub")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=series,
            allow_patterns=["{}/**".format(remote_dir)],
            local_dir=local_base,
        )
    else:
        if not _ensure_modelscope():
            raise RuntimeError("Cannot install modelscope")
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

    if os.path.exists(local_file):
        return local_file

    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    print("[QuantFunc] Downloading: {}/{} from {}...".format(
        series, remote_path, data_source))

    if data_source == "huggingface":
        if not _ensure_huggingface_hub():
            raise RuntimeError("Cannot install huggingface_hub")
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=series, filename=remote_path, local_dir=local_base)
    else:
        if not _ensure_modelscope():
            raise RuntimeError("Cannot install modelscope")
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
    for s in MODEL_SERIES_LIST:
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
    for s in MODEL_SERIES_LIST:
        if s.endswith("/" + short_name):
            return s, name
    raise ValueError("Unknown series in {} selection: {}".format(
        resource_label, short_name))


def resolve_transformer_selection(selection, model_series):
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
]

_BASE_MODEL_CACHE_KEY = "__base_model_repos__"
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
        return list(_base_model_repos) if _base_model_repos else ["None"]


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
    repo_name = repo_id.split("/")[-1]
    local_dir = os.path.join(get_models_dir(), repo_name)
    marker = os.path.join(local_dir, _DOWNLOAD_MARKER)

    if os.path.exists(marker):
        # Check vision_encoder integrity for Qwen-Image-Edit models
        if "image-edit" in repo_name.lower() or "Image-Edit" in repo_name:
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
            raise RuntimeError("Cannot install huggingface_hub")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
    else:
        if not _ensure_modelscope():
            raise RuntimeError("Cannot install modelscope")
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
