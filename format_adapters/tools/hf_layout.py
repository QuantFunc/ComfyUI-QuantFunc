"""HF-style staging directory builder.

Adapters use HFLayout to populate a temp directory with the canonical layout
that quantfunc_create() expects:

    staging_dir/
    ├── model_index.json
    ├── transformer/
    │   ├── config.json
    │   └── diffusion_pytorch_model.safetensors  → symlink to source
    ├── text_encoder/
    │   ├── config.json
    │   └── model.safetensors                    → symlink
    ├── vae/
    │   ├── config.json
    │   └── diffusion_pytorch_model.safetensors  → symlink
    ├── tokenizer/                               (copied from plugin bundle)
    ├── scheduler/
    │   └── scheduler_config.json                (optional)
    └── quantfunc_config.json                    (our hints)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from .fs_util import link_or_copy

from .safetensors_io import read_safetensors_header

logger = logging.getLogger(__name__)


# Krea-2 guidance-distillation marker, for adapters that SYNTHESISE a
# model_index.json. The engine picks Krea-2's timestep shift from
# model_index.json `is_distilled` (Krea2Pipeline.cpp): true => FIXED (turbo)
# shift; key ABSENT => computed (base) shift, the reference default (the engine
# logs which branch it took). The two schedules differ SILENTLY, so the marker
# must never be guessed:
#   * positive NAME signal (`turbo` / `distill`)  => True
#   * anything else                               => None (omit the key) + WARN
# A ComfyUI single-file carries no model_index.json and no distillation flag in
# its metadata (the released krea2 single-files expose only
# `_quantization_metadata` + `format`), so the checkpoint NAME is the only signal
# available to a synthesising adapter.
#
# The match is TOKEN-based with DIRECTIONAL negation, NOT a naive substring: a bare
# `"distill" in name` reads `krea2_UNdistilled` / `krea2_NON_distilled` /
# `krea2_distill-FREE` as POSITIVE and stamps the turbo FIXED shift onto a
# checkpoint whose name says the opposite — silently off-schedule, via the
# "positive" branch so it wouldn't even WARN.
#
# A negation token only counts as a DISTILLATION negation when it actually points
# at a distillation token in reading order (English scope), NOT merely co-occurs:
#   * prefix negators (`no`/`not`/`non`/`un`/`without`/`anti`) negate the first
#     distill-or-cfg token WITHIN THE NEXT 2 tokens (`not_a_turbo` -> turbo at +2;
#     `no_longer_distilled` -> distilled at +2); plus the postfix idiom
#     `<distilled> not/no anymore|longer`.
#   * the suffix negator `free` negates the first distill-or-cfg token within the
#     PREVIOUS 2 tokens (`distill-free`).
#   * fused `undistilled`/`nondistilled` self-negate.
# When the negation's actual target is CFG/GUIDANCE, not distillation, it is IGNORED
# — `turbo-no-cfg` / `cfg-free` / `guidance-free` is standard naming for a
# guidance-DISTILLED (turbo) model and must resolve True; whereas `cfg-no-turbo`
# ("no turbo") negates the distillation and resolves ambiguous. This directionality
# is what separates them, and it stops unrelated qualifier words from false-
# negating a real turbo name (`turbo-de-noise`, `turbo_anti_aliased`,
# `turbo_no_artifact`, `turbo_non_square` all stay True — their negation-looking
# token points at a non-distillation word).
#
# NOTE: an HF-diffusers SOURCE DIR never needs this — HFLayoutAdapter returns
# such a dir AS-IS (hf_native.py), so its own model_index.json (and therefore its
# real is_distilled) is preserved untouched. This helper is only for the
# synthesised-index paths.
_KREA2_DISTILLED_TOKENS = ("turbo", "distill", "distilled")
_KREA2_PREFIX_NEGATORS = ("no", "not", "non", "un", "without", "anti")
_KREA2_SUFFIX_NEGATORS = ("free",)
_KREA2_FUSED_NEGATORS = ("undistilled", "nondistilled")
_KREA2_IDIOM_COMPLETERS = ("anymore", "longer", "more")
# The negation's actual target: distillation words disclaim distillation; cfg/
# guidance words do NOT (a guidance-free / no-cfg model is still distilled).
_KREA2_CFG_TOKENS = ("cfg", "guidance")
_KREA2_NEG_TARGETS = _KREA2_DISTILLED_TOKENS + _KREA2_CFG_TOKENS


def _krea2_negates_distillation(tokens: list[str]) -> bool:
    """True iff a negation token in `tokens` points, in reading order, at a
    DISTILLATION token (not a cfg/guidance token)."""
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if tok in _KREA2_FUSED_NEGATORS:
            return True
        forward = tok in _KREA2_PREFIX_NEGATORS
        backward = tok in _KREA2_SUFFIX_NEGATORS
        if not (forward or backward):
            continue
        window = ((i + 1, i + 2) if forward else (i - 1, i - 2))
        target = ""
        for j in window:
            if 0 <= j < n and tokens[j] in _KREA2_NEG_TARGETS:
                target = tokens[j]
                break
        if target in _KREA2_DISTILLED_TOKENS:
            return True          # e.g. `non_distilled`, `not_a_turbo`, `distill-free`
        if target in _KREA2_CFG_TOKENS:
            continue             # `no-cfg` / `cfg-free` — negates CFG, keep positive
        # postfix idiom: `<distilled> not/no anymore|longer` (nothing forward)
        if (forward and i > 0 and tokens[i - 1] in _KREA2_DISTILLED_TOKENS
                and i + 1 < n and tokens[i + 1] in _KREA2_IDIOM_COMPLETERS):
            return True
    return False


def krea2_is_distilled(arch: str, src_path: str | Path) -> Optional[bool]:
    """Positive Krea-2 distillation signal, else None (omit + WARN). Non-Krea2 => None.

    A positive marker (`turbo`/`distill`/`distilled`) resolves True ONLY when no
    negation in the name actually points at a distillation token (see the module
    comment for the directional rule). Any real distillation-negation makes the name
    AMBIGUOUS → omit the key + WARN, so we never silently stamp the FIXED (turbo)
    schedule onto a checkpoint whose own name disclaims it. Safe by construction:
    the two schedules differ silently, so when in doubt we fall to the engine's
    reference-default computed (base) shift, never guess turbo.
    """
    if arch != "Krea2":
        return None
    name = Path(src_path).name.lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", name) if t]
    positive = any(t in _KREA2_DISTILLED_TOKENS for t in tokens)
    negated = _krea2_negates_distillation(tokens)
    if positive and not negated:
        logger.info("[hf_layout] Krea-2: '%s' names a distilled/turbo checkpoint "
                    "→ model_index is_distilled=true (FIXED timestep shift).", name)
        return True
    if positive and negated:
        logger.warning(
            "[hf_layout] Krea-2: '%s' carries BOTH a distillation marker and a "
            "negation — the name is AMBIGUOUS, so is_distilled is omitted and the "
            "engine uses its reference-default COMPUTED (base) timestep shift. If "
            "this IS a Krea-2-Turbo checkpoint, rename it unambiguously (e.g. "
            "'krea2_turbo') or wire the HF-diffusers model dir (staged as-is, so "
            "its model_index.json is_distilled is honoured verbatim).", name)
        return None
    logger.warning(
        "[hf_layout] Krea-2: cannot determine distillation from '%s' (this "
        "checkpoint carries no is_distilled marker) — omitting the key, so the "
        "engine uses its reference-default COMPUTED (base) timestep shift. If this "
        "IS a Krea-2-Turbo checkpoint it will generate OFF-SCHEDULE: put 'turbo' in "
        "the filename, or wire the HF-diffusers model dir (which is staged as-is, "
        "so its model_index.json is_distilled is honoured verbatim).", name)
    return None



# Map our internal arch tag → diffusers _class_name (what detectPipelineKind expects)
ARCH_TO_PIPELINE_CLASS = {
    "QwenImage":        "QwenImagePipeline",
    "QwenImageEdit":    "QwenImageEditPipeline",
    "QwenImageLayered": "QwenImageLayeredPipeline",
    "Flux2Klein":       "Flux2KleinPipeline",
    "ZImage":           "ZImagePipeline",
    "Krea2":            "Krea2Pipeline",
}

ARCH_TO_TRANSFORMER_CLASS = {
    "QwenImage":        "QwenImageTransformer2DModel",
    "QwenImageEdit":    "QwenImageTransformer2DModel",  # same class, edit_mode=true
    "QwenImageLayered": "QwenImageTransformer2DModel",  # same class, use_additional_t_cond=true
    "Flux2Klein":       "Flux2Transformer2DModel",
    "ZImage":           "ZImageTransformer2DModel",
    "Krea2":            "Krea2Transformer2DModel",  # engine krea2_xfm_match keys on this
}

# model_index.json text_encoder class per arch (informational — the engine's TE
# factory dispatches on the TE config's model_type/hidden_size, not this entry).
# Krea-2 uses a Qwen3-VL 4B text tower (model_type qwen3_vl, hidden 2560).
ARCH_TO_TE_CLASS = {
    "QwenImageEdit": "Qwen2_5VLForConditionalGeneration",
    "Krea2":         "Qwen3VLForConditionalGeneration",
}

# Arch variants whose bundled TEXT-ENCODER / TOKENIZER / SCHEDULER assets are
# byte-identical to a base arch's — reuse the base's bundle (single source of
# truth) instead of duplicating files. NOTE: the VAE config is intentionally NOT
# aliased here: QwenImageLayered's VAE is 4-channel RGBA (AutoencoderKLQwenImage
# with input_channels=4), distinct from base QwenImage's 3-channel — it ships its
# own bin/vae_configs/QwenImageLayered.json. The transformer is likewise NOT
# aliased (its dims are weight-derived per arch).
BUNDLED_ASSET_ARCH_ALIAS = {
    # QwenImageLayered reuses the base QwenImage Qwen3/Qwen2.5-VL text encoder
    # (hidden 3584 / 28 layers / 28 heads / 4 kv / 18944 ffn — identical), the
    # same tokenizer, and the same FlowMatchEuler scheduler.
    "QwenImageLayered": "QwenImage",
}


def _bundled_asset_arch(arch: str) -> str:
    """Resolve `arch` to the arch whose bundled TE/tokenizer/scheduler asset to
    use. Identity for most archs; aliases an identical variant onto its base
    (see BUNDLED_ASSET_ARCH_ALIAS). Does NOT affect VAE/transformer lookups."""
    return BUNDLED_ASSET_ARCH_ALIAS.get(arch, arch)


class HFLayout:
    """Builder for the HF-style staging directory.

    Adapters call methods on this object; nothing touches the filesystem until
    a method is invoked. Weight references use link_or_copy (hardlink → symlink
    → copy fallback, see tools/fs_util.py); only configs
    and tokenizer files are actually written.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._hints: dict = {}              # quantfunc_config.json contents
        self._pipeline_class: str = ""

    # ── Component placement ───────────────────────────────────────────────

    def add_transformer(self, source_path: str | Path,
                         config: Optional[dict] = None,
                         arch: Optional[str] = None) -> Path:
        """Symlink the transformer weights under BOTH canonical filenames:
        - `diffusion_pytorch_model.safetensors` — diffusers convention used
          by ComfyUI's standard loader probes and runtime-quantize / SVDQ
          paths in the C engine.
        - `model.safetensors` — hard-coded filename the Lighting prequant
          loader (ComponentImpl.cpp) uses; without this alias prequant
          reload errors with "Pre-quantized Lighting model directory
          missing model.safetensors". One symlink solves both lookups.

        `config` is augmented with weight-derived architecture dims so a
        size-variant single-file model (e.g. Klein 4B vs 9B) gets the CORRECT
        `num_layers` / `num_attention_heads` / … instead of inheriting the
        engine's hard-coded family-canonical default → no shape-mismatch crash.
        `arch` (internal arch tag) may be passed to skip a re-fingerprint."""
        cfg = self._with_weight_derived_transformer(source_path, config, arch)
        sub = self._add_component(
            "transformer", source_path,
            "diffusion_pytorch_model.safetensors", cfg)
        # Add the alias as a sibling link (target the same source so both
        # names resolve to the same file).
        alias = sub / "model.safetensors"
        link_or_copy(source_path, alias)
        return sub

    def _with_weight_derived_transformer(
            self, source_path: str | Path,
            config: Optional[dict], arch: Optional[str] = None) -> dict:
        """Return `config` with weight-derived transformer dims merged IN.

        The weight tensors are the ground truth for architecture dims, so the
        derived values OVERRIDE the incoming config: a correct diffusers /
        prequant config already EQUALS the derived dims (same weights), while a
        minimal `{_class_name}` (single-file UNETLoader path) or a
        variant-mismatched bundled config gets corrected. Only high-confidence
        fields are produced by the probe; the rest are left to the caller's
        config / the engine default. Best-effort — any failure leaves the
        config untouched (existing behaviour)."""
        cfg = dict(config or {})
        try:
            from .weight_derived_config import derive_transformer_config
            from .arch_fingerprint import fingerprint_arch_from_keys
            a = arch or fingerprint_arch_from_keys(str(source_path))
            derived = derive_transformer_config(str(source_path), a) if a else None
            if derived:
                cfg.update(derived)
                logger.info("[staging] transformer dims derived from weights "
                             "(arch=%s): %s", a, derived)
        except Exception as e:
            logger.debug("[staging] transformer dim derive skipped: %s", e)
        return cfg

    def add_transformer_remapped(self, source_path: str | Path,
                                  remap_fn,
                                  config: Optional[dict] = None) -> Path:
        """Write a remapped transformer (not a symlink) using `remap_fn`.

        `remap_fn(src_path: Path, dst_path: Path) -> None` is called once to
        write the diffusers-keyed safetensors at the staging path. Used for
        BFL → diffusers ZImage and similar key-rename cases that can't be
        expressed via simple `set_key_strip` / `set_key_filter`.
        """
        sub = self.root / "transformer"
        sub.mkdir(parents=True, exist_ok=True)
        dst = sub / "diffusion_pytorch_model.safetensors"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        remap_fn(Path(source_path), dst)
        # Derive dims from the staged file so a size variant is sized correctly
        # here too, consistent with add_transformer. Only WRITE when the merged
        # config carries a `_class_name` (the engine needs it to dispatch the
        # transformer); a dims-only config — config=None + derive succeeded —
        # is an untested semantic, so skip it rather than emit a class-less file.
        cfg = self._with_weight_derived_transformer(dst, config)
        if cfg.get("_class_name"):
            (sub / "config.json").write_text(json.dumps(cfg, indent=2))
        elif cfg:
            # Defensive: a future caller passing config without `_class_name`
            # (current callers always pass one) would otherwise SILENTLY get no
            # config.json → engine mis-dispatch. Make the skip LOUD, not silent.
            logger.warning("[staging] remapped transformer: derived dims but NO "
                           "_class_name (config=%r) — skipping config.json to avoid "
                           "a class-less file; pass a config with _class_name.", config)
        return sub

    def add_text_encoder(self, source_path: str | Path,
                          config: Optional[dict] = None) -> Path:
        cfg = self._with_weight_derived_te(source_path, config)
        return self._add_component(
            "text_encoder", source_path, "model.safetensors", cfg)

    def _with_weight_derived_te(self, source_path: str | Path,
                                config: Optional[dict]) -> dict:
        """Override the TE config's SIZE dims (hidden_size / num_hidden_layers /
        heads / intermediate_size / vocab_size) with values DERIVED from the
        actual TE weights, so a same-family size variant loaded via a bare
        "Load CLIP" (no model_dir config) is allocated correctly. The plugin's
        bundled per-arch TE config is a single size — e.g. Klein bundles
        Qwen3-2560 (4B), but the 9B TE is 4096 → the engine allocated TE buffers
        at 2560 and loaded the 4096 weights → copy_ shape mismatch → noise
        (#267). The weights are ground truth; for an already-correct config the
        derived values EQUAL it (no-op). Best-effort — failure leaves config
        untouched."""
        cfg = dict(config or {})
        try:
            from .weight_derived_config import derive_te_config
            derived = derive_te_config(str(source_path))
            if derived:
                cfg.update(derived)
                logger.info("[staging] TE dims derived from weights: %s", derived)
        except Exception as e:
            logger.debug("[staging] TE dim derive skipped: %s", e)
        return cfg

    def add_vision_encoder(self, source_path: str | Path,
                           config: Optional[dict] = None) -> Path:
        """Place an edit-pipeline vision encoder (Qwen2.5-VL `visual.*`),
        deriving its size dims from the weights so a bare standalone vision
        encoder (no sibling config) is allocated correctly — the same
        derive-from-weights routing as add_text_encoder (#267 class). For a
        QuantFunc export with a sibling config the derived dims equal it
        (no-op)."""
        cfg = self._with_weight_derived_ve(source_path, config)
        return self._add_component(
            "vision_encoder", source_path, "model.safetensors", cfg)

    def _with_weight_derived_ve(self, source_path: str | Path,
                                config: Optional[dict]) -> dict:
        """Override the vision-encoder config's SIZE dims (hidden_size/depth)
        with values derived from the `visual.*` weights. Best-effort; leaves
        config untouched on failure or no-match."""
        cfg = dict(config or {})
        try:
            from .weight_derived_config import derive_vision_encoder_config
            derived = derive_vision_encoder_config(str(source_path))
            if derived:
                cfg.update(derived)
                logger.info("[staging] vision-encoder dims derived from weights: %s",
                             derived)
        except Exception as e:
            logger.debug("[staging] vision-encoder dim derive skipped: %s", e)
        return cfg

    def add_vae(self, source_path: str | Path,
                config: Optional[dict] = None) -> Path:
        cfg = self._with_weight_derived_vae(source_path, config)
        return self._add_component(
            "vae", source_path, "diffusion_pytorch_model.safetensors", cfg)

    def _with_weight_derived_vae(self, source_path: str | Path,
                                 config: Optional[dict]) -> dict:
        """Correct the VAE `_class_name` from the weight keys when they carry a
        recognised family signature (generalizes the engine #257 detection into
        the plugin). A standalone "Load VAE" file ships no config.json, so the
        declared class is empty / a generic AutoencoderKL guess — the wrong 2-D
        decoder against a 3-D AutoencoderKLQwenImage / Flux2 VAE crashes at load.
        Best-effort; leaves config untouched on any failure or no-match."""
        cfg = dict(config or {})
        try:
            from .weight_derived_config import derive_vae_class
            detected = derive_vae_class(str(source_path))
            if detected and cfg.get("_class_name", "") != detected:
                logger.info("[staging] VAE _class_name from weights: %r (was %r)",
                             detected, cfg.get("_class_name") or None)
                cfg["_class_name"] = detected
        except Exception as e:
            logger.debug("[staging] VAE class derive skipped: %s", e)
        return cfg

    def add_vae_remapped(self, source_path: str | Path,
                          remap_fn,
                          config: Optional[dict] = None) -> Path:
        """Write a remapped VAE (not a symlink) using `remap_fn`.

        `remap_fn(src_path: Path, dst_path: Path) -> None`. Used for
        BFL → HF AutoencoderKL key renames.
        """
        sub = self.root / "vae"
        sub.mkdir(parents=True, exist_ok=True)
        dst = sub / "diffusion_pytorch_model.safetensors"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        remap_fn(Path(source_path), dst)
        # Correct the VAE `_class_name` from the remapped output's weight keys
        # (zero-copy remap → `dst` is a symlink to the original, keys intact),
        # so the #257 family detection also covers this BFL-remap VAE path.
        # Previously this path hardcoded `{"_class_name":"AutoencoderKL"}` →
        # #257 was BYPASSED here (a native QwenImage/Flux2 VAE wired through the
        # BFL-remap path would still build the wrong 2-D decoder → load crash).
        cfg = self._with_weight_derived_vae(dst, config)
        if cfg.get("_class_name"):
            (sub / "config.json").write_text(json.dumps(cfg, indent=2))
        elif cfg:
            logger.warning("[staging] remapped VAE: config without _class_name "
                           "(config=%r) — skipping config.json; pass a config with "
                           "_class_name.", config)
        return sub

    def _add_component(self, subdir: str, source_path: str | Path,
                        weight_filename: str, config: Optional[dict]) -> Path:
        sub = self.root / subdir
        sub.mkdir(parents=True, exist_ok=True)
        weight_link = sub / weight_filename
        link_or_copy(source_path, weight_link)
        # Sharded model fan-out: when source is one shard of a multi-shard
        # safetensors, symlink ALL siblings + the index.json. Without this
        # the C engine's ShardedSafeTensors only sees one shard and silently
        # drops keys living in the other shards (with `partial=true` in
        # loadParams), producing empty weights at quant time
        # (e.g. Klein 4B TE down_proj for layers 20-35 lives in
        # `model-00002-of-00002.safetensors`; staging only the first shard
        # makes layer 20's down_proj.weight invalid → setQuantMode crashes).
        src_path = os.path.abspath(str(source_path))
        src_dir = os.path.dirname(src_path)
        src_name = os.path.basename(src_path)
        # Detect sharded layout by `<prefix>-NNNNN-of-NNNNN.safetensors`
        # naming, or by sibling `<source>.index.json`.
        import re as _re
        m = _re.match(r"^(.+?)-(\d{5})-of-(\d{5})\.safetensors$", src_name)
        if m:
            prefix = m.group(1)
            try:
                shards = sorted(
                    f for f in os.listdir(src_dir)
                    if _re.match(rf"^{_re.escape(prefix)}-\d{{5}}-of-\d{{5}}\.safetensors$", f)
                )
            except OSError:
                shards = []
            # Containment guard: a shard `f` from os.listdir could be a symlink
            # that escapes src_dir (and `prefix` derives from an untrusted source
            # filename). `basename` already strips `../` from prefix, but resolve
            # every staged source through realpath and skip anything outside
            # src_dir — defense-in-depth, mirroring _all_related_safetensors. The
            # destination `sub / f` is inherently safe (f is a regex-filtered
            # bare filename, no separators).
            real_src_dir = os.path.realpath(src_dir)

            def _within_src(p: str) -> bool:
                return os.path.realpath(p).startswith(real_src_dir + os.sep)

            # Always link every shard under its ORIGINAL name (engine /
            # ShardedSafeTensors reads them via the index.json which
            # references original names, not the alias).
            for f in shards:
                s = os.path.join(src_dir, f)
                if not _within_src(s):
                    continue  # shard escapes the model dir → skip (untrusted)
                link_or_copy(s, sub / f)
            # Also link index.json under BOTH the source's natural name
            # and the alias the C side expects (it scans for any
            # `*.safetensors.index.json`).
            idx_src = os.path.join(src_dir, f"{prefix}.safetensors.index.json")
            if os.path.isfile(idx_src) and _within_src(idx_src):
                # Source-named index (what the inner shards reference)
                idx_link_src = sub / f"{prefix}.safetensors.index.json"
                link_or_copy(idx_src, idx_link_src)
                # Engine-side filename based on `weight_filename`
                wf_stem = weight_filename[:-len(".safetensors")] \
                    if weight_filename.endswith(".safetensors") else weight_filename
                idx_link_dst = sub / f"{wf_stem}.safetensors.index.json"
                if idx_link_dst != idx_link_src:
                    link_or_copy(idx_src, idx_link_dst)
        if config:
            (sub / "config.json").write_text(json.dumps(config, indent=2))
        return sub

    # ── Hint accumulation ─────────────────────────────────────────────────

    def set_method(self, method: str) -> None:
        """e.g. "online_quant" / "nvfp4_disk" / "prequant_lighting_separate" / "prequant_lighting_bundle" / "prequant_svdq_separate" — or, when the qf_flat_bundle adapter is mirroring the bundle's own metadata, the raw C++-facing value ("lighting_precomputed" / "svdq")."""
        self._hints["method"] = method

    def set_key_strip(self, kind: str, prefix: str) -> None:
        """kind ∈ {"transformer", "te", "vae"}.  Additive lookup."""
        if prefix:
            self._hints[f"{kind}_key_strip"] = prefix

    def set_key_filter(self, kind: str, prefix: str) -> None:
        """kind ∈ {"transformer", "te", "vae"}.  Strict filter."""
        if prefix:
            self._hints[f"{kind}_key_filter"] = prefix

    def set_extra(self, key: str, value) -> None:
        """Additional hint (e.g. nvfp4_block_size)."""
        self._hints[key] = value

    def set_pipeline_class(self, arch: str) -> None:
        """Set _class_name for model_index.json (from arch tag)."""
        self._pipeline_class = ARCH_TO_PIPELINE_CLASS.get(arch, "")

    # ── Index files ───────────────────────────────────────────────────────

    def write_model_index(self, arch: str,
                           extra_components: Optional[dict] = None,
                           is_distilled: Optional[bool] = None) -> None:
        """Write model_index.json so detectPipelineKind() can identify the arch.

        `is_distilled` (Krea-2): the engine reads this key to pick the timestep
        shift — True => FIXED (turbo) shift; key ABSENT => computed (base) shift,
        which is the reference default (Krea2Pipeline.cpp logs either way). It is
        written ONLY when the CALLER positively determined it, because the two
        values produce different (and silently different) schedules — a staged
        turbo checkpoint missing the key generates off-schedule, and a base
        checkpoint forced to True does too. Pass None when there is no signal;
        the engine then takes its reference default and says so in the log.
        """
        cls = ARCH_TO_PIPELINE_CLASS.get(arch, "")
        if not cls:
            raise ValueError(f"Unknown arch: {arch!r}")
        idx: dict = {
            "_class_name": cls,
            "_diffusers_version": "0.30.0",
            "transformer": ["diffusers", ARCH_TO_TRANSFORMER_CLASS.get(arch, "")],
        }
        if (self.root / "vae").exists():
            idx["vae"] = ["diffusers", "AutoencoderKL"]
        if (self.root / "text_encoder").exists():
            idx["text_encoder"] = ["transformers",
                                    ARCH_TO_TE_CLASS.get(arch, "Qwen3ForCausalLM")]
        # Krea-2 guidance-distillation marker. Written ONLY from a CALLER-supplied
        # positive determination (see the docstring): never defaulted to True here.
        # An unconditional True would silently force the FIXED (turbo) shift onto a
        # base checkpoint — a schedule error with no error message, exactly the
        # silent-quality class this codebase treats as highest severity.
        if is_distilled is not None:
            idx["is_distilled"] = bool(is_distilled)
        if extra_components:
            idx.update(extra_components)
        (self.root / "model_index.json").write_text(json.dumps(idx, indent=2))

    def apply_user_precisions(
            self,
            text_precision: Optional[str] = None,
            vae_precision: Optional[str] = None) -> None:
        """Propagate user-selected precisions from BuildContext into the
        nested per-component config blocks the C++ engine reads. Adapters
        whose source has no per-component metadata (raw HF layouts, separate
        UNet/CLIP/VAE files, upstream FP8-mixed Qwen bundles) MUST call this
        before write_quantfunc_config — otherwise the C++ TE/VAE silently
        default to FP16 and the export bloats by tens of GB."""
        if text_precision:
            te = dict(self._hints.get("text_encoder") or {})
            te.setdefault("text_precision", text_precision)
            self._hints["text_encoder"] = te
        if vae_precision:
            v = dict(self._hints.get("vae") or {})
            v.setdefault("vae_precision", vae_precision)
            self._hints["vae"] = v

    def write_quantfunc_config(self) -> None:
        """Flush accumulated hints to staging/quantfunc_config.json."""
        if not self._hints:
            return
        (self.root / "quantfunc_config.json").write_text(
            json.dumps(self._hints, indent=2))
        logger.debug("[staging] quantfunc_config.json: %s", self._hints)

    # ── Scheduler / tokenizer ────────────────────────────────────────────

    def add_scheduler(self, scheduler_config_path: Optional[str],
                       arch: Optional[str] = None) -> None:
        """Copy a scheduler_config.json into staging.

        Priority:
          1. Caller-provided `scheduler_config_path` (e.g. user's diffusers
             model dir scheduler/scheduler_config.json) — copy as-is.
          2. Bundled fallback at <plugin>/bin/scheduler_configs/<arch>.json —
             used when the caller doesn't have a scheduler config but we know
             the arch. This is required because the C++ engine fails to
             open the file if `scheduler/scheduler_config.json` is missing.

        If neither is available, the caller's pipeline load will fail at
        the C++ side with a clear "Cannot open file" error.
        """
        sub = self.root / "scheduler"
        sub.mkdir(parents=True, exist_ok=True)
        dst = sub / "scheduler_config.json"
        if scheduler_config_path:
            shutil.copy2(scheduler_config_path, dst)
            return
        if arch:
            asset_arch = _bundled_asset_arch(arch)
            plugin_root = Path(__file__).resolve().parent.parent.parent
            bundled = plugin_root / "bin" / "scheduler_configs" / f"{asset_arch}.json"
            if bundled.is_file():
                shutil.copy2(bundled, dst)
                return
        logger.warning("[staging] no scheduler_config.json (caller=None, "
                        "arch=%r → bundled missing); pipeline will fail to "
                        "load if the engine demands one.", arch)

    def tokenizer_dir(self) -> Path:
        sub = self.root / "tokenizer"
        sub.mkdir(parents=True, exist_ok=True)
        return sub


def copy_tokenizer_bundle(arch: str, dst: Path) -> None:
    """Copy the bundled tokenizer for `arch` from <plugin>/bin/tokenizers/<arch>/.

    Identical-tokenizer variants (e.g. QwenImageLayered) resolve onto the base
    arch's bundle via _bundled_asset_arch (avoids duplicating the ~5 MB tokenizer).
    """
    asset_arch = _bundled_asset_arch(arch)
    plugin_root = Path(__file__).resolve().parent.parent.parent
    src = plugin_root / "bin" / "tokenizers" / asset_arch
    if not src.is_dir():
        logger.warning("[staging] no bundled tokenizer for arch=%s at %s", arch, src)
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    logger.debug("[staging] tokenizer bundle copied for arch=%s (%d files)",
                  arch, sum(1 for _ in src.iterdir()))


def copy_tokenizer(arch: str, dst: Path, candidate_model_dirs=()) -> None:
    """Populate the staging tokenizer/ dir.

    Prefer a real ``tokenizer/`` already present in a source model_dir: HF base
    model downloads (e.g. the QuantFunc auto-loader's base model) ship a full
    tokenizer, so we should use that instead of depending on a per-arch
    tokenizer being bundled inside the plugin. Only when no source model_dir
    carries one do we fall back to <plugin>/bin/tokenizers/<arch>/.

    `candidate_model_dirs` is tried in order; the first dir with a
    `tokenizer/tokenizer.json` or `tokenizer/vocab.json` wins.
    """
    for md in candidate_model_dirs:
        if not md:
            continue
        cand = Path(md) / "tokenizer"
        if (cand / "tokenizer.json").is_file() or (cand / "vocab.json").is_file():
            dst.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in cand.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst / f.name)
                    n += 1
            logger.info("[staging] tokenizer from model_dir %s (%d files)",
                         cand, n)
            return
    copy_tokenizer_bundle(arch, dst)


def ensure_engine_tokenizer(tokenizer_dir) -> bool:
    """Materialise the engine-native split tokenizer files from a fused
    ``tokenizer.json`` when ``vocab.json`` / ``merges.txt`` are absent.

    The C engine's ``Qwen3Tokenizer::load`` reads ``tokenizer/vocab.json`` +
    ``tokenizer/merges.txt`` (the legacy GPT-2/Qwen split layout) and cannot
    parse the modern fused HF ``tokenizer.json``. Most model packages ship the
    split files, but some (e.g. ``ideogram-ai/ideogram-4-fp8``) ship ONLY the
    fused ``tokenizer.json`` → the engine fails at load with
    "Failed to open vocab.json". A fused HF BPE tokenizer.json embeds the same
    data under ``model.vocab`` (token→id) and ``model.merges`` (BPE pairs), so
    derive the split files from it losslessly here — using the model's OWN
    tokenizer (correct vocab/merges for that exact model), not a generic bundle.

    Special/added tokens are intentionally NOT written into vocab.json: the
    working split-layout models don't carry them there either (the engine maps
    them via its own hard-coded special-token table), so this mirrors the
    canonical layout exactly.

    Idempotent + best-effort: a no-op when both split files already exist, when
    there is no ``tokenizer.json`` to derive from, or when it is not a fused BPE
    tokenizer. Returns True only when it wrote the split files.
    """
    td = Path(tokenizer_dir)
    vocab_p = td / "vocab.json"
    merges_p = td / "merges.txt"
    if vocab_p.is_file() and merges_p.is_file():
        return False  # already engine-native
    tj = td / "tokenizer.json"
    if not tj.is_file():
        return False  # nothing to derive from — let the engine report
    try:
        with open(tj, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = data.get("model") or {}
        vocab = model.get("vocab")
        merges = model.get("merges")
        # Only a fused BPE tokenizer (dict vocab + list merges) is splittable;
        # a unigram/sentencepiece tokenizer.json (no BPE merges) is left for the
        # engine to handle. The list check also rejects a malformed string merges.
        if (not isinstance(vocab, dict) or not vocab
                or not isinstance(merges, list) or not merges):
            return False
        # Each split file is written under its own guard so an interrupted run
        # that produced only one of the two self-heals on the next call.
        wrote = []
        if not vocab_p.is_file():
            with open(vocab_p, "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False)
            wrote.append("vocab.json")
        if not merges_p.is_file():
            with open(merges_p, "w", encoding="utf-8") as f:
                f.write("#version: 0.2\n")  # GPT-2/HF BPE header; engine skips line 1
                for pair in merges:
                    # New HF form: ["a", "b"]; legacy form: "a b" string.
                    if isinstance(pair, (list, tuple)):
                        f.write("{} {}\n".format(pair[0], pair[1]))
                    else:
                        f.write("{}\n".format(pair))
            wrote.append("merges.txt")
        if wrote:
            logger.info("[staging] derived engine tokenizer %s from tokenizer.json "
                        "(vocab %d, merges %d) in %s",
                        "+".join(wrote), len(vocab), len(merges), td)
        return True
    except Exception as e:  # never block model load on a best-effort backfill
        logger.warning("[staging] could not derive vocab.json/merges.txt from "
                       "%s: %s", tj, e)
        return False


def bundled_transformer_config(arch: str) -> Optional[dict]:
    """Return bundled transformer config (architecture dims) for `arch`, or None.

    Used by adapters when the source file (NVFP4 disk-load, etc.) doesn't ship
    a per-component config.json. Without proper dims (num_attention_heads,
    num_layers, ...) the C++ side falls back to defaults that may not match
    the actual weight shapes (e.g. Klein 4B vs 9B).
    """
    plugin_root = Path(__file__).resolve().parent.parent.parent
    p = plugin_root / "bin" / "transformer_configs" / f"{arch}.json"
    if not p.is_file():
        logger.warning("[staging] no bundled transformer config for arch=%s at %s", arch, p)
        return None
    return json.loads(p.read_text())


def bundled_te_config(arch: str) -> Optional[dict]:
    """Return the bundled text-encoder config.json for `arch`, or None.

    Bundled at <plugin>/bin/text_encoder_configs/<arch>.json. Used by adapters
    when the source file (e.g. a bundled multi-component checkpoint)
    doesn't ship per-component config.json — without this, our C++ TE
    loader falls back to defaults (Qwen3 2.5B) and crashes when the actual
    weights are Qwen2.5-VL 7B. Identical-TE variants (e.g. QwenImageLayered)
    resolve onto the base arch's bundle via _bundled_asset_arch.
    """
    asset_arch = _bundled_asset_arch(arch)
    plugin_root = Path(__file__).resolve().parent.parent.parent
    p = plugin_root / "bin" / "text_encoder_configs" / f"{asset_arch}.json"
    if not p.is_file():
        logger.warning("[staging] no bundled TE config for arch=%s at %s", arch, p)
        return None
    return json.loads(p.read_text())


def bundled_vae_config(arch: str) -> Optional[dict]:
    """Same idea but for the VAE config.json."""
    plugin_root = Path(__file__).resolve().parent.parent.parent
    p = plugin_root / "bin" / "vae_configs" / f"{arch}.json"
    if not p.is_file():
        logger.warning("[staging] no bundled VAE config for arch=%s at %s", arch, p)
        return None
    return json.loads(p.read_text())


# Archs whose VAE MUST be 4-channel RGBA: the layered decomposition emits N RGBA
# layers, each carrying a per-layer ALPHA channel, so its VAE decodes 4 channels
# (the official Qwen-Image-Layered VAE has decoder out_channels=4). Base QwenImage
# and QwenImageEdit output plain RGB (3 channels, no per-pixel alpha) — they are NOT
# in this set and correctly accept the standard 3-channel qwen_image_vae.
_RGBA_VAE_ARCHS = {"QwenImageLayered"}

# Candidate "final decoder output convolution" tensor-name suffixes across the two
# QwenImage VAE key conventions: diffusers AutoencoderKLQwenImage (`decoder.conv_out.*`)
# and the ComfyUI-flat layout (`decoder.head.2.*` — the Conv ending the `head` =
# [RMSNorm, SiLU, Conv] sequence). shape[0] of either is the output channel count
# (3 = RGB, 4 = RGBA). Suffix-matched so a `vae.` wrapper prefix is tolerated.
_VAE_OUT_CONV_SUFFIXES = (
    "decoder.conv_out.bias", "decoder.conv_out.weight",
    "decoder.head.2.bias", "decoder.head.2.weight",
)


def vae_decoder_out_channels(vae_path) -> Optional[int]:
    """Best-effort: the VAE decoder's OUTPUT channel count (3=RGB / 4=RGBA), read
    from the final decoder-conv tensor shape in the safetensors header. Handles both
    the diffusers (`decoder.conv_out`) and ComfyUI-flat (`decoder.head.2`) QwenImage
    VAE namings. Returns None when the file can't be read or no recognised output
    conv is found — callers MUST treat None as 'unknown' (never a mismatch), so an
    unreadable / differently-keyed VAE is never falsely blocked."""
    try:
        hdr = read_safetensors_header(vae_path)
    except Exception:
        return None
    for suffix in _VAE_OUT_CONV_SUFFIXES:
        for k, info in hdr.items():
            if k == "__metadata__":
                continue
            if k.endswith(suffix):
                shape = (info or {}).get("shape") or []
                if shape:
                    return int(shape[0])
    return None


def assert_vae_matches_arch(arch: str, vae_path, vae_name: str = "") -> None:
    """Fail LOUD with an ACTIONABLE message when the WIRED VAE's channel count is
    wrong for `arch`, BEFORE the engine reaches a cryptic deep `conv_out [N] vs [M]`
    copy-overflow. Guards the RGBA-VAE archs (QwenImageLayered): a layered
    decomposition emits per-layer alpha, so it REQUIRES a 4-channel RGBA VAE; the
    standard 3-channel RGB qwen_image_vae cannot represent the alpha. A NO-OP for
    every other arch (QwenImage / QwenImageEdit keep the standard 3-ch VAE) and when
    the channel count can't be determined (best-effort, never a false block)."""
    if arch not in _RGBA_VAE_ARCHS:
        return
    ch = vae_decoder_out_channels(vae_path)
    if ch is None or ch == 4:
        return
    name = vae_name or Path(vae_path).name
    raise ValueError(
        f"{arch} requires a 4-channel RGBA VAE (the layered decomposition emits "
        f"per-layer alpha), but the wired VAE '{name}' decodes {ch} channels — this "
        f"is a standard RGB QwenImage VAE, which cannot produce layered RGBA output. "
        f"Wire the Qwen-Image-Layered model's OWN 4-channel RGBA VAE (the official "
        f"diffusers model's vae/ — its decoder.conv_out has 4 output channels), not "
        f"the base qwen_image_vae.")
