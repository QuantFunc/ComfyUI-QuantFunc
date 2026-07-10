"""ComfyUI nodes for the format-adapter pipeline (Sprint 1).

These nodes mirror the official Load Diffusion Model / Load CLIP / Load VAE /
Load LoRA UX (dropdown scanning of standard ComfyUI directories) and feed
into a QuantFuncBuildPipeline node that runs the adapter factory and
constructs a QuantFunc pipeline.

Naming summary:
  QuantFuncLoadDiffusionModel  → QF_XFM     (xfm_ref)
  QuantFuncLoadCLIP            → QF_TE      (te_ref)
  QuantFuncLoadVAE             → QF_VAE     (vae_ref)
  QuantFuncLoadCheckpoint      → QF_XFM + QF_TE + QF_VAE (3 outputs)
  QuantFuncLoadLoRA            → QF_LORA_LIST  (chainable)
  QuantFuncLoadPrecisionMap    → QF_PRECISION_MAP
  QuantFuncSchedulerConfig     → QF_SCHED   (scheduler JSON path)
  QuantFuncBuildPipeline       → QUANTFUNC_PIPELINE  (consumed by Generate)

The pipeline output is the same QUANTFUNC_PIPELINE type the existing
QuantFuncGenerate node accepts.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

# ComfyUI exports folder_paths globally; if missing (e.g. testing standalone),
# fall back to no-op stubs so this module can at least import.
try:
    import folder_paths  # type: ignore[import-not-found]
except ImportError:
    class _StubFolderPaths:
        def get_filename_list(self, _key):  # noqa: D401
            return []
        def get_full_path(self, _key, name):
            return name
    folder_paths = _StubFolderPaths()  # type: ignore[assignment]

from .format_adapters import (
    AdapterRegistry,
    BuildContext,
    FileRef,
    SourceBundle,
    UnsupportedFormatError,
    build_pipeline_inputs,
)
from .format_adapters.tools import (
    fingerprint_arch_from_keys,
    fingerprint_kind_from_metadata,
)

logger = logging.getLogger("QuantFunc")


# ============================================================================
# Loader helpers (used only by QuantFuncLoadLoRA — the rest of the loaders
# are now superseded by official ComfyUI loader nodes consumed via the
# monkey-patches installed at plugin import time).
# ============================================================================

def _list_files(folder_key: str) -> list[str]:
    """Names from ComfyUI's `folder_paths`. Empty if folder doesn't exist."""
    try:
        return list(folder_paths.get_filename_list(folder_key))  # type: ignore[no-any-return]
    except Exception:
        return []


def _resolve(folder_key: str, name: str) -> str:
    try:
        path = folder_paths.get_full_path(folder_key, name)
        return path or os.path.join(folder_key, name)
    except Exception:
        return name


# ============================================================================
# Loader nodes
# ============================================================================

class _QFPathStub:
    """Stand-in for ComfyUI MODEL / CLIP / VAE that carries only the source
    file path. Avoids ComfyUI's `comfy.sd.load_*` doing a full FP8→BF16 torch
    cast (slow on multi-GB checkpoint files + may misclassify the model_type).

    Plugs into QuantFunc Build Pipeline via the same MODEL / CLIP / VAE
    sockets that official ComfyUI loaders use. Any non-QuantFunc node
    downstream (KSampler, etc.) will crash on this stub — by design.

    Optional QuantFunc-loader hints (qf_model_dir / qf_backend_hint /
    qf_prequant_weights) let QuantFunc Model Loader / Auto Loader pass
    model-series metadata that comfy native loaders don't carry.
    """
    __slots__ = ("qf_source_path", "qf_is_checkpoint", "qf_lora_chain",
                 "qf_kind", "qf_model_dir", "qf_backend_hint",
                 "qf_prequant_weights",
                 # [auto-detect] intent stashed by QuantFuncModelAutoLoader so
                 # build() can RE-RESOLVE the transformer weight for the SELECTED
                 # run-device (device_idx), not the auto-loader's device-0 pick.
                 "qf_auto_transformer_series", "qf_data_source")

    def __init__(self, path: str, kind: str = ""):
        self.qf_source_path = path
        self.qf_is_checkpoint = (kind == "bundled_checkpoint")
        self.qf_lora_chain: list = []
        self.qf_kind = kind  # informational: "transformer" / "te" / "vae" / "bundled_checkpoint"
        self.qf_model_dir = ""
        self.qf_backend_hint = ""
        self.qf_prequant_weights = ""
        self.qf_auto_transformer_series = ""   # non-empty ⇒ [auto-detect] re-resolve at build()
        self.qf_data_source = ""


def _scan_files(*folder_keys: str) -> list[str]:
    """Combined dropdown over multiple ComfyUI folder roots."""
    seen: set[str] = set()
    out: list[str] = []
    for k in folder_keys:
        for n in _list_files(k):
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def _resolve_first(name: str, *folder_keys: str) -> str:
    for k in folder_keys:
        p = _resolve(k, name)
        if p and os.path.isfile(p):
            return p
    raise RuntimeError(
        f"File not found in any of {folder_keys}: {name}")


class _AnyType(str):
    """ComfyUI wildcard-type sentinel: equality is permissive so a slot
    typed `_AnyType("*")` connects to any input type (rgthree / kjnodes
    pattern). Used so QuantFunc Precision Config Loader can wire into
    BuildPipeline's COMBO `precision_config` once converted to input —
    ComfyUI otherwise rejects STRING→COMBO connections."""
    def __ne__(self, other):
        return False


_QF_ANY = _AnyType("*")


class QuantFuncPrecisionConfigLoader:
    """Load a precision-config JSON by absolute path.

    Wires into BuildPipeline by right-clicking the `precision_config`
    dropdown → "Convert Widget to Input" → connect this node's output
    to the resulting socket. BuildPipeline detects an absolute path and
    uses it directly, bypassing the preset resolution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "",
                    "placeholder": "/abs/path/to/precision.json",
                    "tooltip": "Absolute path to a precision-config JSON.",
                }),
            }
        }

    # Wildcard-typed output so it connects to BuildPipeline's COMBO
    # `precision_config` when converted to input (ComfyUI's strict type
    # check rejects STRING→COMBO; `*` bypasses via _AnyType.__ne__).
    RETURN_TYPES = (_QF_ANY,)
    RETURN_NAMES = ("precision_config",)
    FUNCTION = "load"
    CATEGORY = "QuantFunc/v2"

    def load(self, path: str):
        p = (path or "").strip()
        if not p:
            raise RuntimeError("Precision config path is empty")
        if not os.path.isabs(p):
            raise RuntimeError(
                f"Precision config path must be absolute: {p!r}")
        if not os.path.isfile(p):
            raise RuntimeError(
                f"Precision config file not found: {p}")
        return (p,)


class QuantFuncPickDiffusionModel:
    """Pick a transformer / bundled-checkpoint file by name. ZERO torch
    load — just records the path in a stub MODEL object that QuantFunc
    Build Pipeline reads.

    Scans models/diffusion_models/, models/checkpoints/, models/unet/
    (combined dropdown). Use this when the file is QuantFunc-bound only;
    use ComfyUI's official UNETLoader / CheckpointLoaderSimple if you also
    need the MODEL on a non-QuantFunc branch (e.g., KSampler).
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = _scan_files("diffusion_models", "checkpoints", "unet")
        return {"required": {"name": (files or ["(empty)"],)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "QuantFunc/v2"

    def load(self, name: str):
        if name == "(empty)":
            raise RuntimeError("No diffusion_models/ / checkpoints/ / unet/ entries")
        path = _resolve_first(name, "diffusion_models", "checkpoints", "unet")
        # Header probe to detect bundled multi-component checkpoint;
        # QuantFunc Build Pipeline also re-checks, so worst case we just
        # skip this here.
        kind = ""
        try:
            from .format_adapters.tools.safetensors_io import has_keys_starting_with
            hits = has_keys_starting_with(
                path, ["model.diffusion_model.", "text_encoders.", "vae."])
            if len(hits) >= 2:
                kind = "bundled_checkpoint"
        except Exception:
            pass
        return (_QFPathStub(path, kind=kind or "transformer"),)


def _scan_quantfunc_te_files() -> list[tuple[str, str]]:
    """Find Qwen2.5-VL TE files inside `models/QuantFunc/<series>/<base-model-dir>/text_encoder/`.

    These are the BF16 official text encoders ModelAutoLoader downloads —
    higher precision than community FP8 variants, gives noticeably better
    quality (esp. Chinese prompts) when paired with INT4 transformer.

    Returns list of (display_name, absolute_path) tuples.
    """
    out: list[tuple[str, str]] = []
    try:
        comfyui_root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))  # plugin parent (custom_nodes/) → up
        comfyui_root = os.path.dirname(comfyui_root)  # → ComfyUI/
        qf_root = os.path.join(comfyui_root, "models", "QuantFunc")
        if not os.path.isdir(qf_root):
            return out
        for series in os.listdir(qf_root):
            series_dir = os.path.join(qf_root, series)
            if not os.path.isdir(series_dir):
                continue
            for sub in os.listdir(series_dir):
                te_dir = os.path.join(series_dir, sub, "text_encoder")
                if not os.path.isdir(te_dir):
                    continue
                for f in sorted(os.listdir(te_dir)):
                    if f.endswith(".safetensors"):
                        full = os.path.join(te_dir, f)
                        # Display: "[QF] Qwen-Image-Series/qwen-image-series-50x-below-base-model"
                        display = f"[QF] {series}/{sub}"
                        out.append((display, full))
    except Exception:
        pass
    return out


class QuantFuncPickCLIP:
    """Pick a text encoder file by name (zero load).

    Scans:
      - models/text_encoders/, models/clip/  (standard ComfyUI dirs)
      - models/QuantFunc/<series>/<base>/text_encoder/  (BF16 TE downloaded
        by QuantFunc Model Auto Loader — best quality, esp. for Chinese)
    """

    @classmethod
    def INPUT_TYPES(cls):
        std_files = _scan_files("text_encoders", "clip")
        qf_files = _scan_quantfunc_te_files()
        files = std_files + [d for d, _ in qf_files]
        return {"required": {"name": (files or ["(empty)"],)}}

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"
    CATEGORY = "QuantFunc/v2"

    def load(self, name: str):
        if name == "(empty)":
            raise RuntimeError("No text_encoders/ / clip/ / QuantFunc/ entries")
        # If QuantFunc series prefix → resolve via map
        if name.startswith("[QF] "):
            qf_files = dict(_scan_quantfunc_te_files())
            path = qf_files.get(name)
            if not path:
                raise RuntimeError(f"QuantFunc TE not found: {name}")
            return (_QFPathStub(path, kind="te"),)
        path = _resolve_first(name, "text_encoders", "clip")
        return (_QFPathStub(path, kind="te"),)


class QuantFuncPickVAE:
    """Pick a VAE file by name (zero load)."""

    @classmethod
    def INPUT_TYPES(cls):
        files = _scan_files("vae")
        return {"required": {"name": (files or ["(empty)"],)}}

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "QuantFunc/v2"

    def load(self, name: str):
        if name == "(empty)":
            raise RuntimeError("No vae/ entries")
        path = _resolve_first(name, "vae")
        return (_QFPathStub(path, kind="vae"),)


class QuantFuncPickCheckpoint:
    """Pick a single-file checkpoint that bundles transformer + TE + VAE
    (zero load).

    Drop-in replacement for ComfyUI's `CheckpointLoaderSimple`:
      - same dropdown UX (scans models/checkpoints/)
      - same 3-output shape: MODEL, CLIP, VAE
      - but ZERO torch load — outputs are stubs carrying just the path

    All three outputs share the same source path tagged as
    `bundled_checkpoint`. BuildPipeline routes via the bundled-checkpoint
    adapter (transformer + TE + VAE all sliced from the one file by key
    prefix).
    """

    @classmethod
    def INPUT_TYPES(cls):
        files = _scan_files("checkpoints")
        return {"required": {"ckpt_name": (files or ["(empty)"],)}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    CATEGORY = "QuantFunc/v2"

    def load(self, ckpt_name: str):
        if ckpt_name == "(empty)":
            raise RuntimeError("No checkpoints/ entries")
        path = _resolve_first(ckpt_name, "checkpoints")
        stub = _QFPathStub(path, kind="bundled_checkpoint")
        return (stub, stub, stub)


# ============================================================================
# Build Pipeline
# ============================================================================

# Map an arch fingerprint to its ModelScope precision-config series. Each series
# ships per-layer precision configs (Z-Image/Qwen: 50x-above-fp4 / 50x-below-int4;
# Klein: 50x-fp4-f8 / 40x-int4-f8 / 30x-below-int4-i8) whose keys match THAT arch's
# real (engine-internal) layer structure, so downloading the matching series' config
# is correct regardless of the source weight layout.
_ARCH_TO_SERIES = {
    "QwenImage":        "QuantFunc/Qwen-Image-Series",
    "QwenImageEdit":    "QuantFunc/Qwen-Image-Edit-Series",
    "QwenImageLayered": "QuantFunc/Qwen-Image-Layered-Series",
    "ZImage":        "QuantFunc/Z-Image-Series",
    "Ideogram4":     "QuantFunc/Ideogram-4-Series",
    # Klein 4B (K=3072) and 9B (K=4096) deliberately share ONE precision-config:
    # the keys are layer-NAME patterns (transformer_blocks.attn / .ff / .ff_context
    # / single_transformer_blocks.attn / modulation / embedders / head), NOT
    # dimension-specific shapes, so the same file applies to both. fingerprint_arch
    # returns "Flux2Klein" for both and does not disambiguate size. Klein-9B-Series
    # exists for prequant-WEIGHT downloads; if it ever ships its own precision-config,
    # add a size-disambiguated entry here (and teach the fingerprint to tell 4B/9B apart).
    "Flux2Klein":    "QuantFunc/Klein-4B-Series",
}


def _device_sm(device_idx: int) -> int:
    """Compute capability (e.g. 120, 89, 86) of the user-selected CUDA device.
    Returns 0 if it can't be determined.

    Thin convenience wrapper — delegates to the canonical per-index detector
    `lib_setup._detect_device_sm` (torch `get_device_capability(idx)`, CVD-aware,
    bounds-checked), which is the single source of truth for per-index SM (callers
    may also import `_detect_device_sm` directly). We deliberately do NOT probe a
    first GPU / nvidia-smi here:
    nvidia-smi orders by PCI bus while CUDA orders by capability, so it can return
    the WRONG device's SM and wrongly pick FP4 on a non-Blackwell card (the 本地
    4090/3060 trap — FP4 __trap()s below SM120). 0 → the caller falls back to INT4
    (50x-below), the conservative choice that runs on EVERY GPU."""
    try:
        from .lib_setup import _detect_device_sm
        return _detect_device_sm(device_idx)
    except Exception:
        return 0


# POSITIVE allowlist of kinds that get a config injected — only genuine
# full-precision weights. A blocklist was fragile: any quant format the detector
# can't name (kind=="") would slip through and get a fresh-quant config injected
# over already-quantized weights (double-quant / shape mismatch). Kinds come from
# `fingerprint_kind_from_metadata` (already on `xfm_ref.kind`):
#   raw_highprec       — a plain FP16/BF16/F32 transformer (needs online-quant)
#   bundled_checkpoint — an all-in-one 全家桶 checkpoint; MAY itself be a
#                        QuantFunc-stamped (already-quantized) export, so it gets
#                        an extra stamped-metadata check below before injecting.
# Everything else (nvfp4_disk / raw_fp8 / raw_int8 / prequant_lighting_separate /
# unknown "") is left untouched — the engine / SVDQ path uses its on-disk precision.
# EXCEPTION — Ideogram-4: its official distribution IS an fp8 base the a4w4 recipe
# is measured on, so it does NOT use this strict allowlist; it uses the broader
# "anything not pre-quantized" denylist gate below (`_PREQUANTIZED_KINDS`), applied
# only when series == "QuantFunc/Ideogram-4-Series" in `_autopick_*` below.
_FULL_PRECISION_KINDS = frozenset({"raw_highprec", "bundled_checkpoint"})
# Kinds that are ALREADY pre-quantized by a final pipeline (QuantFunc-stamped
# lighting export, or a nunchaku NVFP4 disk format) — their on-disk precision is
# the answer, so the auto-config is NEVER injected over them, for any family.
_PREQUANTIZED_KINDS = frozenset({"prequant_lighting_separate", "nvfp4_disk"})


def _autopick_precision_for_full_model(precision_map_xfm, xfm_ref, device_idx,
                                       data_source="modelscope"):
    """Auto-pick a precision config for a config-eligible base model when the
    user left precision_config on [auto-derive] (no explicit config).

    A full-precision diffusers base model OR an all-in-one (全家桶) checkpoint
    carries no quant metadata, so the engine's [auto-derive] would leave it at
    full precision. Instead, IDENTIFY the model — reusing the arch + kind that
    `build()` already fingerprinted onto `xfm_ref` — and load its CORRESPONDING
    precision config through the precision auto loader: the matching series' own
    per-layer config, at the variant suited to the selected GPU (FP4
    `50x-above` on Blackwell SM120+, INT4 `50x-below` otherwise);
    `download_precision_config` fetches + caches it.

    Eligibility differs by family. For most families ONLY genuine full-precision
    weights get a config injected — already-quantized inputs (nunchaku NVFP4, raw
    FP8/INT8/FP4, or any QuantFunc-stamped export) keep their on-disk precision.
    Ideogram-4 is the exception: its official distribution IS an fp8 base its a4w4
    recipe is measured on, so ANY non-pre-quantized Ideogram base (fp16/bf16 AND
    fp8) is eligible; only a stamped / nunchaku-final export is left untouched."""
    # Only act on the [auto-derive] preset (empty path, preset == 'auto').
    if not (isinstance(precision_map_xfm, dict)
            and precision_map_xfm.get("preset") == "auto"
            and not (precision_map_xfm.get("path") or "").strip()):
        return precision_map_xfm
    arch = getattr(xfm_ref, "arch", "") or ""
    path = getattr(xfm_ref, "path", "") or ""
    kind = getattr(xfm_ref, "kind", "") or ""
    series = _ARCH_TO_SERIES.get(arch)
    if not series:
        logger.info("[BuildPipeline] [auto-derive]: arch '%s' has no precision-"
                    "config series; leaving to engine auto-derive", arch or "?")
        return precision_map_xfm
    # Only inject onto GENUINE full-precision weights; everything else keeps its
    # on-disk precision (already quantized, or an unrecognized kind we won't touch).
    # Eligibility gate — which inputs get the auto-config injected:
    #   • Ideogram-4 ships as an FP8 base (and may also come as fp16/bf16); the
    #     a4w4 recipe was MEASURED on that base. Per the product rule, activate
    #     for ANY model that is NOT already pre-quantized — fp16/bf16 AND fp8
    #     both get the recipe; only a QuantFunc-stamped / nunchaku-final export
    #     is left untouched. (Empty/unknown kind is treated as not-eligible.)
    #   • Every OTHER family keeps the strict full-precision-only gate: an FP8 /
    #     INT8 / FP4 Klein/Qwen/ZImage is a final quantization, used as-is.
    if series == "QuantFunc/Ideogram-4-Series":
        is_eligible = bool(kind) and kind not in _PREQUANTIZED_KINDS
    else:
        is_eligible = kind in _FULL_PRECISION_KINDS
    if not is_eligible:
        logger.info("[BuildPipeline] [auto-derive]: %s kind=%s is pre-quantized / "
                    "not a config-eligible base; engine uses its on-disk precision",
                    arch, kind or "?")
        return precision_map_xfm
    if kind == "bundled_checkpoint":
        # A 全家桶 bundle may itself be a QuantFunc-stamped (already-quantized)
        # export whose marker is hidden behind force_kind='bundled_checkpoint'.
        # Inject ONLY when positively confirmed NOT stamped; if we can't read it
        # (no path) or the probe errors, skip — the safe default (the engine's own
        # [auto-derive] still reads any stamped map from the bundle).
        try:
            from .format_adapters.tools.auto_precision import _precision_map_from_metadata
            full_precision_bundle = bool(path) and not _precision_map_from_metadata(path)
        except Exception:
            full_precision_bundle = False
        if not full_precision_bundle:
            logger.info("[BuildPipeline] [auto-derive]: %s bundle is stamped or "
                        "unverifiable; engine uses its on-disk precision", arch)
            return precision_map_xfm
    try:
        from .model_auto_loader import download_precision_config
        sm = _device_sm(device_idx)
        if series in ("QuantFunc/Klein-4B-Series", "QuantFunc/Klein-9B-Series"):
            # Klein 3-tier (FP4 needs Blackwell SM120; FP8 needs SM89+; else INT8):
            if sm >= 120:
                fname = "50x-fp4-f8-sample.json"        # Blackwell: FP4 + FP8 islands
            elif sm >= 89:
                fname = "40x-int4-f8-sample.json"       # Ada/Hopper FP8: INT4 + FP8 islands
            else:
                fname = "30x-below-int4-i8-sample.json"  # no FP8: INT4 + INT8 islands
        elif series == "QuantFunc/Ideogram-4-Series":
            # Single GPU-adaptive map: ideogram4_a4w4.json uses AUTO_4 (FP4 on
            # SM120 / INT4 on SM89) + AUTO_8 (FP8 on SM89+ / INT8 older), so one
            # file fits every supported GPU — no 50x-above/below split.
            fname = "ideogram4_a4w4.json"
        else:
            fname = ("50x-above-fp4-sample.json" if sm >= 120  # native NVFP4
                     else "50x-below-int4-sample.json")         # INT4 (RTX 20/30/40)
        local = download_precision_config(series, fname, data_source)
        logger.info("[BuildPipeline] full-precision %s (kind=%s) + [auto-derive]: "
                     "device %d (SM%d) -> %s / %s",
                     arch, kind or "?", device_idx, sm, series, fname)
        return {"path": local, "target": "transformer", "preset": fname}
    except Exception as e:
        logger.warning("[BuildPipeline] precision auto-pick failed (%s); "
                        "falling back to engine [auto-derive]", e)
        return precision_map_xfm


class _NoCompatibleWeightError(RuntimeError):
    """The [auto-detect] BACKSTOP: the SELECTED run-device can run NO weight in the
    series. A clean, user-actionable pipeline-build error (choose a higher-capability
    device or pick a weight explicitly) — NOT a device __trap. A dedicated subclass
    so `_reresolve_auto_transformer_for_device` can re-raise ONLY this and let a
    generic download RuntimeError fall back to the auto-loader's pick."""


def _reresolve_auto_transformer_for_device(model, xfm_path, device_idx):
    """Device-aware [auto-detect] transformer re-resolution at pipeline-build time.

    QuantFuncModelAutoLoader resolves the transformer weight BEFORE the run-device
    is known (its node has no `device`; the device lives on THIS Build Pipeline
    node), so it keys on the DEFAULT GPU (device 0). Here `device_idx` — the device
    the pipeline will ACTUALLY run on — IS known, so re-pick the best weight for it,
    overriding the auto-loader's device-0 pick. This makes a device switch (e.g. a
    4090 on device 0 → a 3060 on device 1) load a weight the SELECTED GPU can run
    instead of the device-0 tier (which would `__trap` on the weaker card).

    - No-op (returns `xfm_path` unchanged) when `model` carries no [auto-detect]
      marker — i.e. an explicit user pick or a non-auto-loader source (plain
      UNETLoader). So it never overrides a deliberate selection.
    - BACKSTOP: raises a clean `_NoCompatibleWeightError` (never a device `__trap`)
      when the series HAS weights but NONE runs on the selected device — telling the
      user to choose a higher-capability device or pick a compatible weight explicitly.
    - Undetectable device SM (no CUDA/torch) → the resolver best-efforts the lowest
      tier (safe); a genuine resolve/download error (network, hf/modelscope missing)
      FALLS BACK to the auto-loader's already-valid pick (never crashes build()).
    """
    series = getattr(model, "qf_auto_transformer_series", "") or ""
    if not series:
        return xfm_path  # not an [auto-detect] auto-loader selection → leave as-is
    data_source = getattr(model, "qf_data_source", "") or "modelscope"
    try:
        from .model_auto_loader import (
            resolve_transformer_selection, download_transformer, AUTO_DETECT,
            _available_transformer_names,
        )
        from .lib_setup import _detect_device_sm
        sm = _detect_device_sm(device_idx)
        t_series, t_name = resolve_transformer_selection(AUTO_DETECT, series, device_idx)
        if not t_name:
            if _available_transformer_names(series):
                # DISTINCT subclass (not a bare RuntimeError) so the except below can
                # tell THIS intentional backstop from a generic download failure —
                # download_transformer raises plain RuntimeError on hf/modelscope
                # missing or a transient network error, which must FALL BACK, not crash.
                raise _NoCompatibleWeightError(
                    "[QuantFunc] No transformer weight in {} runs on the selected "
                    "device (CUDA device {}, SM{}). Choose a device with a higher "
                    "compute capability, or pick a compatible weight explicitly in "
                    "the QuantFunc Model Auto Loader.".format(series, device_idx, sm))
            return xfm_path  # series ships no separate weights → keep base default
        new_path = download_transformer(t_series, t_name, data_source)
        if os.path.abspath(new_path or "") != os.path.abspath(xfm_path or ""):
            logger.info("[BuildPipeline] [auto-detect] re-resolved transformer for "
                        "device %d (SM%d): %s -> %s", device_idx, sm,
                        os.path.basename(xfm_path or "?"), os.path.basename(new_path))
        return new_path
    except _NoCompatibleWeightError:
        raise  # the clean, user-actionable backstop — surface it (never a __trap)
    except Exception as e:
        # ANY other failure (download/network/library-missing/resolve bug) → degrade
        # gracefully to the auto-loader's already-valid pick rather than crash build().
        logger.warning("[BuildPipeline] [auto-detect] device re-resolve failed (%s); "
                        "keeping the auto-loader's pick %s",
                        e, os.path.basename(xfm_path or "?"))
        return xfm_path


# tiny-VAE (TAEHV) opt-in fast-preview decoder — Wan video only. The engine's
# TinyVAEDecoder is a per-variant TRUSTED decoder that validates the incoming
# latent channel count z against the variant (engine origin/main b13721da
# ComponentImpl.cpp wan_vae_factory: `vd=="taew2_2"||vd=="taew2_1"`, taew2_1 =
# 16-ch, taew2_2 = 48-ch — verify via `git show b13721da:src/ComponentImpl.cpp`;
# a stale engine checkout/lib may predate the taew2_1 arm, which is exactly what
# _engine_lib_supports_taew guards against). We pick the variant from that SAME
# value — the staged Wan transformer's `out_channels` (= the latent channel
# count) — instead of a new parallel arch heuristic, so the plugin's choice can
# never disagree with the engine's own check.
_TAEW_LATENT_TO_VARIANT = {16: "taew2_1", 48: "taew2_2"}  # latent z_dim → taehv variant
_TAEW_WEIGHTS_SUBDIR = "taew"  # <ComfyUI>/models/QuantFunc/<subdir>/<variant>.safetensors


def _resolve_tiny_vae_decoder(staging_model_dir: str,
                              taew_dir: Optional[str] = None) -> tuple[str, str]:
    """Resolve the taew (variant, weights_path) for a staged Wan video model.

    Reads the SAME staged configs the engine loads from (`model_index.json`
    `_class_name` for the Wan-family gate, `transformer/config.json`
    `out_channels` for the latent channel count) — no new parallel arch
    detection. Fails LOUD (RuntimeError) if the model is not Wan video, the
    latent channel count is unsupported, or the weight file is absent; never
    silently degrades to a wrong-variant / full-VAE decode.

    `taew_dir` defaults to `<ComfyUI>/models/QuantFunc/taew` (dependency-
    injected so the unit test can point it at a fixture dir).
    """
    import json as _json
    # (1) Wan-video FAMILY gate — REUSE the plugin's single Wan-family
    #     predicate (`_is_wan_diffusers_dir`, comfyui_wan_remap.py). Its
    #     model_index + transformer arms exactly mirror the engine's wan_detect
    #     (WanVideoPipeline.cpp @ b13721da: pipeline_class starts "Wan" OR
    #     transformer_class=="WanTransformer3DModel"); its A14B-layout and
    #     vae-substring arms are PERMISSIVE family signals only — the strict
    #     consumer-side check is step (1b) below. Predicate is fail-open
    #     (False on unreadable configs); the REFUSAL here is the fail-loud part.
    from .format_adapters.comfyui_wan_remap import _is_wan_diffusers_dir
    if not _is_wan_diffusers_dir(staging_model_dir):
        # Best-effort class name, purely for the error message.
        cls_name = ""
        try:
            with open(os.path.join(staging_model_dir, "model_index.json"),
                      "r", encoding="utf-8") as _f:
                cls_name = str(_json.load(_f).get("_class_name", ""))
        except Exception:  # noqa: BLE001 — message detail only
            pass
        raise RuntimeError(
            f"tiny_vae is only supported for Wan video models (Wan2.1/A14B or "
            f"Wan2.2-5B); no Wan signal in {staging_model_dir!r} "
            f"(model_index.json/transformer/vae _class_name; pipeline is "
            f"'{cls_name or 'unknown'}'). Disable tiny_vae for this model.")
    # (1b) CONSUMER-exact check: the injected keys are consumed by the engine's
    #     Wan VAE factory, whose registry dispatch is wan_vae_match =
    #     vaeClassOf(config) == "AutoencoderKLWan" EXACT equality
    #     (ComponentImpl.cpp:6466 @ b13721da) — STRICTER than the family
    #     predicate's permissive vae-substring arm. A "Wan-ish" but non-exact
    #     VAE class would dispatch a DIFFERENT VAE backend that ignores
    #     vae_decoder/vae_decoder_weights → silent full-VAE fallback. Refuse
    #     loud instead. An ABSENT vae config / class stays permitted (the
    #     family gate + the engine's own fail-loud load carry those).
    vae_cfg_path = os.path.join(staging_model_dir, "vae", "config.json")
    if os.path.isfile(vae_cfg_path):
        try:
            with open(vae_cfg_path, "r", encoding="utf-8") as _f:
                vae_cls = str(_json.load(_f).get("_class_name", "") or "")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"tiny_vae: cannot read {vae_cfg_path} ({e}) — needed to "
                f"confirm the Wan VAE class before injecting tiny-VAE keys.")
        if vae_cls and vae_cls != "AutoencoderKLWan":
            raise RuntimeError(
                f"tiny_vae: the staged VAE class '{vae_cls}' is not the exact "
                f"'AutoencoderKLWan' the engine's Wan VAE factory dispatches "
                f"on — the tiny-VAE keys would be silently ignored by the "
                f"selected VAE backend. Disable tiny_vae for this model.")

    # (2) Variant by latent channel count = staged transformer out_channels,
    #     the exact value the engine's TinyVAEDecoder validates z against.
    xfm_cfg_path = os.path.join(staging_model_dir, "transformer", "config.json")
    try:
        with open(xfm_cfg_path, "r", encoding="utf-8") as _f:
            latent_ch = int(_json.load(_f).get("out_channels"))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"tiny_vae: cannot read out_channels from {xfm_cfg_path} ({e}) — "
            f"needed to pick the taew variant.")
    variant = _TAEW_LATENT_TO_VARIANT.get(latent_ch)
    if variant is None:
        raise RuntimeError(
            f"tiny_vae: unsupported Wan latent channel count "
            f"out_channels={latent_ch}; the tiny VAE supports Wan2.1/A14B "
            f"(16-ch → taew2_1) and Wan2.2-5B (48-ch → taew2_2) only.")
    # (3) Weight path — <ComfyUI>/models/QuantFunc/taew/<variant>.safetensors.
    if taew_dir is None:
        from .model_auto_loader import get_models_dir
        taew_dir = os.path.join(get_models_dir(), _TAEW_WEIGHTS_SUBDIR)
    weights_path = os.path.join(taew_dir, f"{variant}.safetensors")
    if not os.path.isfile(weights_path):
        raise RuntimeError(
            f"tiny_vae: {variant} weights not found at {weights_path}. Place "
            f"the taehv {variant} weights file there (create the folder if "
            f"needed): <ComfyUI>/models/QuantFunc/{_TAEW_WEIGHTS_SUBDIR}/"
            f"{variant}.safetensors")
    return variant, weights_path


# Engine-capability probe cache: lib identity (path, mtime, size) → set of the
# taew variant tokens found in the lib binary. The engine's factory gates on the
# LITERAL variant strings ("taew2_1"/"taew2_2" in ComponentImpl.cpp), so a
# supporting .so necessarily carries them in .rodata (string LITERALS survive
# symbol stripping — `strip` removes symbol tables, not constants) — their
# absence means the installed engine PREDATES tiny-VAE support and would
# SILENTLY IGNORE the injected comp_opts (unknown create-time keys are not
# rejected), i.e. a silent full-VAE fallback. Probing turns that version skew
# into a fail-LOUD error. Transition-window belt-and-braces: releases pair
# plugin+engine, but a manual `git pull` of the plugin alone would otherwise
# silently no-op.
# WHY NOT auto_update._read_lib_version() + _ver_cmp (simplicity-CR question):
# MEASURED — QUANTFUNC_VERSION_STRING bumps only at SHIP time, not per commit:
# it is "0.0.12" (engine CMakeLists.txt:242) at BOTH 81ddedc9 (pre-taew2_1) AND
# b13721da (has taew2_1). So the shipped 0.0.12 lib (probe: no "taew2_1" token)
# and a self-built lib from b13721da (has it) report the IDENTICAL version —
# a version floor cannot distinguish them, on either side of the skew. The
# capability token IS the discriminating signal (same signal used to diagnose
# the live lib's skew in the first place).
_ENGINE_TAEW_PROBE_CACHE: dict[tuple, frozenset] = {}
_TAEW_PROBE_CHUNK_BYTES = 8 * 1024 * 1024  # streaming scan chunk (avoid loading a ~300MB .so at once)


def _engine_lib_supports_taew(variant: str,
                              lib_path: Optional[str] = None) -> Optional[bool]:
    """Best-effort: does the installed engine lib support this taew variant?

    Returns True/False when the lib file is resolvable (scan for the variant's
    literal token), or None when no lib path can be resolved (non-ComfyUI test
    context) — callers treat None as "cannot verify" (log, don't block).
    """
    if lib_path is None:
        try:
            from .nodes import _LIB_PATH
            lib_path = _LIB_PATH
        except Exception:  # noqa: BLE001 — nodes not importable outside ComfyUI
            return None
    if not lib_path or not os.path.isfile(lib_path):
        return None
    try:
        st = os.stat(lib_path)
        cache_key = (lib_path, st.st_mtime_ns, st.st_size)
        found = _ENGINE_TAEW_PROBE_CACHE.get(cache_key)
        if found is None:
            tokens = {v.encode("ascii") for v in _TAEW_LATENT_TO_VARIANT.values()}
            hits = set()
            overlap = max(len(t) for t in tokens) - 1
            tail = b""
            with open(lib_path, "rb") as f:
                while tokens - hits:
                    chunk = f.read(_TAEW_PROBE_CHUNK_BYTES)
                    if not chunk:
                        break
                    window = tail + chunk
                    for t in tokens - hits:
                        if t in window:
                            hits.add(t)
                    tail = chunk[-overlap:]
            found = frozenset(h.decode("ascii") for h in hits)
            _ENGINE_TAEW_PROBE_CACHE[cache_key] = found
        return variant in found
    except Exception as e:  # noqa: BLE001 — unreadable lib: cannot verify
        logger.warning("[BuildPipeline] tiny_vae engine probe failed on %s: %s",
                       lib_path, e)
        return None


def _apply_tiny_vae(options: dict, tiny_vae: bool, staging_model_dir: str,
                    taew_dir: Optional[str] = None,
                    lib_path: Optional[str] = None) -> None:
    """Inject the tiny-VAE comp_opts keys when `tiny_vae` is ON.

    OFF (the default) is a strict no-op — NO key is added or touched, so the
    engine comp_opts stay byte-identical to the full-VAE path. ON resolves the
    taew variant + weights via `_resolve_tiny_vae_decoder` (fail-LOUD), verifies
    the INSTALLED engine lib actually supports the variant (an engine that
    predates tiny-VAE support would silently ignore the keys = silent full-VAE
    fallback — refuse instead), and injects `vae_decoder` + `vae_decoder_weights`.
    """
    if not tiny_vae:
        return
    variant, weights_path = _resolve_tiny_vae_decoder(staging_model_dir, taew_dir)
    supported = _engine_lib_supports_taew(variant, lib_path)
    if supported is False:
        raise RuntimeError(
            f"tiny_vae: the installed QuantFunc engine library predates "
            f"tiny-VAE ({variant}) support and would silently ignore it "
            f"(falling back to the full VAE). Update the engine library "
            f"(plugin auto-update / matching release), or disable tiny_vae.")
    if supported is None:
        logger.warning(
            "[BuildPipeline] tiny_vae: could not verify engine support for %s "
            "(engine lib not resolvable in this context) — proceeding; an "
            "unsupported engine would fall back to the full VAE.", variant)
    options["vae_decoder"] = variant
    options["vae_decoder_weights"] = weights_path
    logger.info(
        "[BuildPipeline] tiny_vae ON → %s (%s) — lossy fast preview",
        variant, weights_path)


class QuantFuncBuildPipeline:
    """Assemble a QuantFunc pipeline from official ComfyUI loaders.

    Wire pattern (all sockets accept official ComfyUI types):
      UNETLoader / CheckpointLoaderSimple ─→ model
      CLIPLoader / DualCLIPLoader         ─→ clip   (optional)
      VAELoader / CheckpointLoaderSimple  ─→ vae    (optional)
      QuantFuncLoadLoRA chain             ─→ lora_list (optional)

    Source paths recovered via monkey-patches installed at plugin import
    (see nodes_pipeline_builder).

    Precision config is one inline dropdown merging:
      [none] / [auto-derive] sentinels
      [builtin] JSON files from <plugin>/configs or $QUANTFUNC_CONFIGS_DIR
      [series]  QuantFunc model-series presets (downloaded on demand
                from ModelScope via model_auto_loader)

    Output: QUANTFUNC_PIPELINE consumed by QuantFuncGenerate.
    """

    @classmethod
    def INPUT_TYPES(cls):
        try:
            from .nodes import _AVAILABLE_DEVICES  # type: ignore[attr-defined]
            devices = _AVAILABLE_DEVICES
        except Exception:
            devices = ["0: GPU"]
        from .nodes_pipeline_builder import (
            build_precision_preset_options,
            PRECISION_AUTO_LABEL,
        )
        presets = build_precision_preset_options()
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "device": (devices,),
                "precision_config": (presets, {"default": PRECISION_AUTO_LABEL,
                    "tooltip": "[auto-derive] (default) — for a full-precision "
                               "diffusers base / AIO checkpoint with no quant metadata, "
                               "identify the model and load its matching precision config "
                               "for the selected GPU (FP4 50x-above on Blackwell SM120+, "
                               "INT4 50x-below otherwise); SVDQ / pre-quantized models keep "
                               "their own per-layer config from safetensors metadata.\n"
                               "[none] — never inject a precision_map.\n"
                               "[builtin] / [series] — use a fixed JSON config.",
                }),
            },
            "optional": {
                "pipeline_config": ("QUANTFUNC_CONFIG", {
                    "tooltip": "Optional. Connect a QuantFunc Pipeline Config node to "
                               "override knobs (precision / vae_precision / text_precision "
                               "/ vision_quant / act_quant_mode / attention_backend / "
                               "tiled_vae). When not connected, BuildPipeline uses the "
                               "same defaults as Pipeline Config (auto_optimize default).",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "QuantFunc API key (qf_xxx) for key-protected models. "
                               "Empty = falls back to api_key in config.json next to "
                               "libquantfunc.so. Explicit value here overrides that.",
                }),
                "tiny_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Lossy FAST-preview VAE (Wan video only). Swaps the "
                               "full Wan VAE decoder for the tiny TAEHV decoder — "
                               "measured decode speedups range from ~5x (Wan2.1/A14B "
                               "@384x384) to ~28x (Wan2.2-5B), varying with model and "
                               "resolution. Output stays coherent but is SOFTER — "
                               "use for draft/preview, DISABLE for final quality. "
                               "The taew variant is auto-selected by model "
                               "(Wan2.1/A14B = taew2_1, Wan2.2-5B = taew2_2); the "
                               "weights must sit at "
                               "<ComfyUI>/models/QuantFunc/taew/<variant>.safetensors.",
                }),
            },
        }

    RETURN_TYPES = ("QUANTFUNC_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "build"
    CATEGORY = "QuantFunc/v2"

    @classmethod
    def IS_CHANGED(cls, model=None, clip=None, vae=None, device=None,
                    precision_config=None, pipeline_config=None, api_key="",
                    tiny_vae=False):
        # Force re-execution every prompt: this node creates a fresh tmp
        # staging dir on each call, so caching the previous prompt's
        # output (which references a now-deleted staging dir) would crash
        # the engine with "Failed to load safetensors: /tmp/quantfunc_staging_*".
        import time
        return f"build@{time.time_ns()}"

    def build(self, model, clip, vae, device, precision_config,
              pipeline_config=None, api_key="", tiny_vae=False):
        # P0 diagnostic — surface the exact `precision_config` arg ComfyUI
        # delivered. User reported wiring `Precision Config Loader` →
        # converted-to-input `precision_config` socket but generation came
        # out as if `[auto-derive]` ran. This log proves whether the wired
        # path actually reaches build() or not.
        logger.info("[BuildPipeline] precision_config arg=%r  type=%s",
                     precision_config, type(precision_config).__name__)
        # No-config fallback: same defaults as QuantFunc Pipeline Config so
        # workflows that don't wire a config node behave identically to the
        # previous in-node-widget defaults.
        if not isinstance(pipeline_config, dict):
            pipeline_config = {
                "tiled_vae": False,
                "attention_backend": "auto",
                "precision": "bf16",
                "text_precision": "int4",
                "vision_quant": "int8",
                "vae_precision": "auto",
                "act_quant_mode": "absmax",
            }
        cfg_dict = dict(pipeline_config)
        # Defaults match the in-node PipelineConfig widget defaults exactly so
        # workflows that don't wire PipelineConfig get the same behaviour
        # (act_quant_mode default is absmax — fast single-pass, user choice).
        vae_precision  = cfg_dict.pop("vae_precision",  "auto")
        text_precision = cfg_dict.pop("text_precision", "int4")
        act_quant_mode = cfg_dict.pop("act_quant_mode", "absmax")
        from .nodes_pipeline_builder import (
            extract_qf_source_path, resolve_precision_preset,
            detect_scheduler_config,
        )
        from .format_adapters.tools import (
            fingerprint_arch_from_keys, fingerprint_kind_from_metadata,
        )
        # `precision_config` accepts either:
        #   - a preset label from the inline dropdown ([none] / [auto-derive]
        #     / [builtin] xxx.json / [series] yyy.json)
        #   - an absolute filesystem path (when the dropdown is converted to
        #     an input socket and wired from QuantFunc Precision Config Loader).
        pcfg_value = (precision_config or "").strip() if isinstance(
            precision_config, str) else ""
        if pcfg_value and os.path.isabs(pcfg_value):
            if not os.path.isfile(pcfg_value):
                raise RuntimeError(
                    f"precision_config path not found: {pcfg_value}")
            precision_map_xfm = {
                "path": pcfg_value,
                "target": "transformer",
                "preset": os.path.basename(pcfg_value),
            }
        else:
            precision_map_xfm = resolve_precision_preset(
                pcfg_value, "", "transformer", "modelscope")
        precision_map_te = None  # use uniform `text_precision` instead

        # Recover source paths from official ComfyUI MODEL/CLIP/VAE objects
        # (monkey-patched at plugin import to attach qf_source_path).
        def _ref_from_path(path: str, force_kind: Optional[str] = None) -> FileRef:
            try:
                arch = fingerprint_arch_from_keys(path) or ""
                kind = force_kind or (fingerprint_kind_from_metadata(path) or "")
            except Exception:
                arch, kind = "", (force_kind or "")
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            return FileRef(path=path, arch=arch, kind=kind, mtime=mtime)

        xfm_path = extract_qf_source_path(model, "diffusion model (UNETLoader)")
        # [auto-detect] DEVICE-AWARE re-resolution: QuantFuncModelAutoLoader picked
        # xfm_path for the DEFAULT GPU (device 0), but THIS pipeline runs on the
        # user-SELECTED `device`. Re-pick the best weight for the selected device so
        # switching devices (e.g. 4090 device 0 → 3060 device 1) never loads a tier
        # the selected GPU can't run (→ __trap). No-op unless the auto-loader stashed
        # its [auto-detect] marker on `model`. Done BEFORE the is_ckpt probe/staging
        # so they operate on the weight actually used.
        # SELECTED run-device index (also reused below by the precision-config
        # auto-pick — computed once here, the earliest point it's needed).
        device_idx = int(device.split(":")[0]) if isinstance(device, str) else int(device)
        xfm_path = _reresolve_auto_transformer_for_device(model, xfm_path, device_idx)
        is_ckpt = bool(getattr(model, "qf_is_checkpoint", False))
        # CheckpointLoaderSimple sets qf_is_checkpoint, but UNETLoader doesn't —
        # users may also wire a bundled-checkpoint file
        # (e.g. model/checkpoints/Qwen-Rapid-*-Bundle.safetensors) through
        # UNETLoader. Detect the bundle layout by header inspection: the
        # bundle has `model.diffusion_model.` + `text_encoders.` + `vae.`
        # key prefixes coexisting in one file.
        if not is_ckpt:
            try:
                from .format_adapters.tools.safetensors_io import has_keys_starting_with
                # Probe for both upstream (`text_encoders.` plural — qwen25 /
                # qwen2vl) and qf_flat (`text_encoder.` singular) layouts so
                # QuantFunc-native bundle exports get auto-promoted to ckpt
                # mode without requiring a CheckpointLoader node.
                hits = has_keys_starting_with(
                    xfm_path,
                    ["model.diffusion_model.", "text_encoders.",
                     "text_encoder.", "vae."])
                te_hit = ("text_encoders." in hits) or ("text_encoder." in hits)
                xfm_hit = ("model.diffusion_model." in hits)
                vae_hit = ("vae." in hits)
                if int(te_hit) + int(xfm_hit) + int(vae_hit) >= 2:
                    is_ckpt = True
                    logger.info(
                        "[BuildPipeline] auto-detected bundled checkpoint "
                        "from key prefixes %s in %s",
                        sorted(hits), os.path.basename(xfm_path))
            except Exception as e:
                logger.debug("Bundle header probe failed for %s: %s", xfm_path, e)
        xfm_ref = _ref_from_path(xfm_path,
                                  force_kind="bundled_checkpoint" if is_ckpt else None)
        te_ref = vae_ref = None
        if clip is not None:
            te_path = extract_qf_source_path(clip, "CLIP (CLIPLoader)")
            if not (is_ckpt and te_path == xfm_path):
                te_ref = _ref_from_path(te_path)
        if vae is not None:
            vae_path = extract_qf_source_path(vae, "VAE (VAELoader)")
            if not (is_ckpt and vae_path == xfm_path):
                vae_ref = _ref_from_path(vae_path)

        # Auto-detect Lightning / distilled variants and pick a bundled
        # scheduler config; otherwise let the engine use its FlowMatchEuler
        # default.
        scheduler_config = detect_scheduler_config(xfm_path)

        # LoRA: chain via existing QuantFuncLoRALoader downstream (operates
        # on QUANTFUNC_PIPELINE output of this node). Not wired here.

        # Build SourceBundle
        sources = SourceBundle(
            transformer=xfm_ref if not is_ckpt else None,
            text_encoder=te_ref,
            vae=vae_ref,
            checkpoint=xfm_ref if is_ckpt else None,
            loras=[],
            scheduler_config=scheduler_config,
        )

        # device_idx already computed above (right after xfm_path extraction).
        # Full-precision auto-pick: a full-precision diffusers base / all-in-one
        # checkpoint with no quant metadata + no explicit precision_config would
        # stay at full precision under [auto-derive]. Identify the model (via the
        # arch + kind already fingerprinted onto xfm_ref) and load its matching
        # precision config for the selected GPU (FP4 on Blackwell SM120+, INT4
        # otherwise); already-quantized inputs are left untouched.
        precision_map_xfm = _autopick_precision_for_full_model(
            precision_map_xfm, xfm_ref, device_idx)
        context = BuildContext(
            precision_map_xfm=precision_map_xfm,
            precision_map_te=precision_map_te,
            vae_precision=vae_precision,
            text_precision=text_precision,
            device_idx=device_idx,
            backend="lighting",
            api_key="",  # paid-tier key; not exposed in this minimal node
        )

        # Run adapter factory
        try:
            staging = build_pipeline_inputs(sources, context)
        except UnsupportedFormatError as e:
            raise RuntimeError(
                f"No QuantFunc adapter handles this combination: {e}. "
                f"Likely cause: an unrecognized weight format (e.g. GGUF, "
                f"NVFP4-disk on consumer GPUs, or proprietary INT4 packing).")

        # Free ComfyUI's pre-loaded torch tensors — they're dead weight, we
        # reload from disk through QuantFunc. Route through the plugin helper
        # (which calls the ORIGINAL, un-hooked free_memory) so this self-cleanup
        # of ComfyUI-native weights does NOT trip our free_memory hook into
        # tearing down sibling QuantFunc pipelines — that self-inflicted
        # blanket-destroy made two pipelines rebuild each other every run.
        try:
            from .nodes import free_comfy_native_models
            free_comfy_native_models()
        except Exception as e:
            logger.debug("free_comfy_native_models() failed: %s", e)

        # Build the cfg dict consumed by QuantFuncGenerate. The Lighting
        # quality knobs below mirror what the proven base-model path sets
        # so bundled / runtime-quant outputs match diffusers-format quality:
        #
        # - rotation_block_size=256: CRITICAL — enables H256 Hadamard rotation
        #   for INT4. Engine auto-enables `rht_seed=0x52485421` and MSE
        #   activation-scale search when rotation>0 (ComponentImpl.cpp:2073).
        #   Without it, INT4 outputs blurry across 60 layers.
        # - quant_method="higgs+hqq": HIGGS gaussian-optimal scales + HQQ
        #   grid-search refinement. Engine default already, set explicit so
        #   it shows in logs and doesn't depend on a default that may shift.
        # - cub_fp4 NOT set → defaults to false → mma.sync FP4 (default
        #   optimal). cuBLASLt NVFP4 has BF16 round-trip in MLP causing
        #   60-layer error accumulation; the mma.sync path uses fused
        #   GELU+quant. Only enable cuBLASLt FP4 via opt-in.
        # - act_quant_mode NOT set → engine auto-enables MSE search when
        #   rotation>0 (more accurate than absmax for INT4 activations).
        options: dict[str, Any] = {
            "auto_optimize": True,
            "vae_precision": vae_precision,
            # `text_precision` was popped off pipeline_config above and was
            # ONLY propagated into cfg["precision"] (which is unrelated —
            # that one is the transformer's BF16/FP16 compute precision).
            # Without this line the engine never saw text_precision and
            # defaulted TE to FP16 — Qwen2.5-VL 7B would ship in the bundle
            # at ~21 GB FP16 instead of ~4 GB INT4. Forward it explicitly so
            # any adapter path (HF-native online_quant, BundledCheckpoint
            # qwen25_bundle, ComfyUI-trio) lands on the user's choice.
            "text_precision": text_precision,
            "rotation_block_size": 256,
            "quant_method": "higgs+hqq",
        }
        # Forward every other knob from PipelineConfig (vision_quant,
        # attention_backend, tiled_vae, vae_tile_size, pinned_memory_limit, …)
        # so adding a knob to PipelineConfig automatically propagates here.
        options.update(cfg_dict)
        options.setdefault("vision_quant", "int8")
        # #516: runtime int4/fp4 TE (Qwen3) MUST use Hadamard rotation. The engine
        # qwen3_te_factory defaults use_rotation=false for the RUNTIME (raw/online-
        # quant) TE, so without this the outlier-heavy Qwen3 hidden-state activations
        # collapse under int4/fp4 W4A4 ACTIVATION quant → the conditioning embedding
        # goes to ~0 → solid-gray t2i (edit survives via image-token conditioning).
        # The transformer already rotates (rotation_block_size=256 above + engine
        # default use_rotation=true); forward the same to the TE. rotation_block_size
        # is already 256 here so the paired H256 rotation is available. setdefault →
        # an explicit user use_rotation still wins; 8-bit/fp16 TE tolerate outliers
        # and are left untouched. Mirrors what the pre-quantized Model-Auto-Loader
        # path bakes in (its int4/fp4 TE ships already-rotated).
        if str(text_precision).lower() in ("int4", "i4", "4", "fp4", "f4"):
            options.setdefault("use_rotation", True)
        # `act_quant_mode="auto"` ⇒ leave key unset so engine auto-enables
        # MSE search when rotation_block_size > 0 (best INT4 quality).
        # Explicit "absmax" / "mse" matches the QuantFuncModelAutoLoader knob.
        if act_quant_mode in ("absmax", "mse"):
            options["act_quant_mode"] = act_quant_mode

        # tiny-VAE (TAEHV) fast-preview opt-in — Wan video only, default OFF.
        # OFF injects NOTHING (comp_opts byte-identical to the full-VAE path);
        # ON resolves the taew variant from the SAME latent channel count the
        # engine's TinyVAEDecoder validates z against, so a non-Wan model /
        # unsupported variant / missing weight file all fail LOUD here rather
        # than silently mis-decoding. See _apply_tiny_vae/_resolve_tiny_vae_decoder.
        _apply_tiny_vae(options, tiny_vae, staging.model_dir)

        # Pick up QuantFunc-loader-only hints stashed on the `_QFPathStub`
        # by QuantFunc Model Loader / Auto Loader (no equivalent on stock
        # comfy MODEL objects). Silently no-op when wired from a comfy
        # native loader.
        prequant_weights = getattr(model, "qf_prequant_weights", "")
        if prequant_weights:
            options["mod_weights"] = prequant_weights

        # API key + server URL — needed to load key-protected models like the
        # BF16 Qwen2.5-VL TE downloaded by QuantFuncModelAutoLoader (which
        # are obfuscated/encrypted; engine decrypts at load using the key).
        # Priority: explicit `api_key` socket > lib config.json > none.
        explicit_key = api_key.strip() if isinstance(api_key, str) else ""
        if explicit_key.lower() == "none":
            explicit_key = ""
        try:
            from .nodes import _load_lib_config
            lib_config = _load_lib_config()
            ak = explicit_key or lib_config.get("api_key", "")
            su = lib_config.get("server_url", "")
            if ak:
                options["api_key"] = ak
            if su:
                options["server_url"] = su
        except Exception:
            if explicit_key:
                options["api_key"] = explicit_key
        # Resolve precision_config to a concrete file path passed to the engine:
        #   - [none]         → never inject (precision_map_xfm is None)
        #   - [builtin] xxx  → precision_map_xfm["path"] is the bundled JSON
        #   - [series]  xxx  → adapter downloaded the JSON, path is set
        #   - custom path    → set directly
        #   - [auto-derive]  → always runs auto_derive_precision_map(), which
        #     itself reads `transformer.precision_map` / `quantization_config`
        #     metadata from the transformer .safetensors when available
        #     (any prequant export — separated or bundle — stamps it) and
        #     only falls back to the per-tensor dtype scan for raw FP16/BF16
        #     AIO checkpoints.  No per-method_hint skip branch — the
        #     skip-on-prequant version silently dropped layers whose
        #     transformer keys were obfuscated to UUIDs (img_mod, txt_mod)
        #     and produced "weight tensor is empty" at forward.
        if precision_map_xfm and precision_map_xfm.get("path"):
            options["precision_config"] = precision_map_xfm["path"]
        elif precision_map_xfm and precision_map_xfm.get("preset") == "auto":
            try:
                from .format_adapters.tools.auto_precision import (
                    auto_derive_precision_map,
                )
                derived = auto_derive_precision_map(
                    xfm_ref.path,
                    target_quant="i4",
                    key_strip_prefix="model.diffusion_model.",
                )
                auto_path = os.path.join(staging.model_dir, "auto_precision.json")
                import json as _json
                with open(auto_path, "w") as _f:
                    _json.dump(derived, _f, indent=2)
                options["precision_config"] = auto_path
                logger.info(
                    "[BuildPipeline] auto-derive: %d entries → %s "
                    "(method=%s)",
                    len(derived), os.path.basename(auto_path),
                    staging.method_hint)
            except Exception as e:
                logger.warning(
                    "[BuildPipeline] auto-derive failed: %s — "
                    "falling back to engine default (no precision_config)", e)
        # Fused INT8 GEMV for W8A8 modulation: SiLU(temb) → GEMV → +bias →
        # split_mod<6> in one kernel. QwenImage / QwenImageEdit only — Klein /
        # ZImage transformers ignore this flag (no fused-mod kernel path).
        # Always on for Qwen; not user-configurable.
        # #270-residual: prefix-match the Qwen-Image family instead of an
        # enumerated tuple — the 2511 edit pipeline's class is
        # QwenImageEditPlusPipeline (arch "QwenImageEditPlus"), which the old
        # tuple missed -> fused_mod silently not injected on export -> bundle
        # carried tiled (non-GEMV) mod weights while the reimport (arch from
        # bundle metadata, gate hit) requested fused -> noise. The engine now
        # ALSO defaults fused_mod on for QwenImage Lighting (belt+braces);
        # this keeps the plugin's intent general for future Qwen variants.
        if staging.arch.startswith("QwenImage"):
            options["fused_mod"] = True


        # Backend dispatch:
        #   - prequant_svdq_separate (Nunchaku/MIT SVDQuant) → backend=svdq,
        #     transformer file fed directly via cfg["transformer"]; engine
        #     reads metadata to detect proj_down/proj_up/smooth_factor naming.
        #   - everything else (online_quant / prequant_lighting_separate /
        #     prequant_lighting_bundle) → backend=lighting, transformer read
        #     from the staging dir.
        if staging.method_hint == "prequant_svdq_separate":
            backend = "svdq"
            # Pull the original file path the adapter recorded in staging
            # quantfunc_config.json (set via layout.set_extra).
            transformer_override = ""
            try:
                with open(os.path.join(staging.model_dir,
                                        "quantfunc_config.json")) as _f:
                    import json as _json
                    transformer_override = _json.load(_f).get(
                        "svdq_transformer_path", "")
            except Exception:
                pass
            transformer_path = transformer_override or xfm_ref.path
            # Drop Lighting-only knobs that don't belong on the SVDQ path.
            # The legacy QuantFuncModelAutoLoader (nodes.py:1208) ONLY sets
            # rotation_block_size when backend=="lighting"; for SVDQ it
            # leaves these unset. Sending them on SVDQ degrades quality
            # (verified blurry Chinese output otherwise — user-confirmed).
            # The Nunchaku transformer is prequant W4A4 from MIT, and the
            # obfuscated TE is prequant INT4+rotation, so neither needs
            # online HIGGS/HQQ/H256 dispatching.
            options.pop("rotation_block_size", None)
            options.pop("quant_method", None)
            options.pop("fused_mod", None)
            # #516: the SVDQ obfuscated TE is prequant INT4+rotation (baked), so
            # the runtime TE use_rotation default set above is a Lighting-only knob
            # here — drop it for parity (the prequant restore path applies the
            # baked rotation regardless).
            options.pop("use_rotation", None)
        else:
            backend = "lighting"
            transformer_path = ""

        # #411: robustly recover the Qwen-Image-Layered identity. The per-file
        # fingerprint (fingerprint_arch_from_keys) collapses a layered model to
        # plain "QwenImage" on several load layouts — a flat lighting export whose
        # metadata _class_name is the SHARED QwenImageTransformer2DModel; a
        # non-shard-1 shard read with no sibling config; a model_dir handed in as a
        # directory. The ENGINE picks QwenImageLayered — and refImageChannels()==4 —
        # from model_index.json _class_name, so mirror that exact source of truth
        # here. Without it cfg["_arch"] != "QwenImageLayered" -> QuantFunc Generate
        # stages the RGBA ref as an RGB-only QFRAW blob -> engine load_image(
        # channels=4) bypasses the QFRAW decoder -> cv::imread on a raw blob ->
        # "Failed to load image .qfraw" (regressed #397). A false positive is
        # harmless (a real PNG loads at 3 OR 4 channels); a false negative crashes,
        # so bias toward detecting layered.
        eff_arch = staging.arch
        if not str(eff_arch or "").startswith("QwenImageLayered"):
            _layered = False
            try:
                import json as _json
                _mi = os.path.join(str(staging.model_dir or ""), "model_index.json")
                if os.path.isfile(_mi):
                    with open(_mi, "r", encoding="utf-8") as _f:
                        if _json.load(_f).get("_class_name") == "QwenImageLayeredPipeline":
                            _layered = True
            except Exception:
                pass
            if not _layered:
                _pc = precision_config if isinstance(precision_config, str) else ""
                if ("layered" in os.path.basename(str(staging.model_dir or "")).lower()
                        or "layered" in _pc.lower()):
                    _layered = True
            if _layered:
                logger.info(
                    "[BuildPipeline] arch upgraded %r -> QwenImageLayered "
                    "(model_index/precision_config layered marker, #411)", eff_arch)
                eff_arch = "QwenImageLayered"

        cfg = {
            "model_dir": staging.model_dir,
            "transformer": transformer_path,
            "backend": backend,
            "precision": text_precision,
            "scheduler": scheduler_config or "",
            "device": device_idx,
            "options": options,
            "unload": False,
            "_arch": eff_arch,
            "_method_hint": staging.method_hint,
            "_staging_cleanup": staging.cleanup_dir,
        }
        logger.info(
            "[BuildPipeline] arch=%s method=%s xfm=%s te=%s vae=%s "
            "precision=%s vae_prec=%s text_prec=%s",
            eff_arch, staging.method_hint,
            os.path.basename(xfm_ref.path),
            os.path.basename(te_ref.path) if te_ref else "(none)",
            os.path.basename(vae_ref.path) if vae_ref else "(none)",
            (precision_map_xfm or {}).get("preset", "none"),
            vae_precision, text_precision,
        )
        return (cfg,)


# ============================================================================
# Registration helpers (consumed by __init__.py)
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "QuantFuncPickDiffusionModel":    QuantFuncPickDiffusionModel,
    "QuantFuncPickCLIP":              QuantFuncPickCLIP,
    "QuantFuncPickVAE":               QuantFuncPickVAE,
    "QuantFuncPickCheckpoint":        QuantFuncPickCheckpoint,
    "QuantFuncPrecisionConfigLoader": QuantFuncPrecisionConfigLoader,
    "QuantFuncBuildPipeline":         QuantFuncBuildPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuantFuncPickDiffusionModel":    "QuantFunc Pick Diffusion Model (zero-load)",
    "QuantFuncPickCLIP":              "QuantFunc Pick CLIP (zero-load)",
    "QuantFuncPickVAE":               "QuantFunc Pick VAE (zero-load)",
    "QuantFuncPickCheckpoint":        "QuantFunc Pick Checkpoint (zero-load, bundled)",
    "QuantFuncPrecisionConfigLoader": "QuantFunc Precision Config Loader (path)",
    "QuantFuncBuildPipeline":         "QuantFunc Build Pipeline",
}
