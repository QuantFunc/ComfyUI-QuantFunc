"""Adapter for HF-diffusers `model_dir` layouts.

Two distinct cases, one common path:
  • QuantFunc prequant export — transformer .safetensors carrying a
    QuantFunc metadata marker (method / quantfunc_obfuscated / …).
    method_hint = "prequant_lighting_separate". Engine reloads as-is.
  • Plain BF16/FP16 base — no QuantFunc metadata, vanilla HF model_dir
    (e.g. `flux.2-klein-4B/`). method_hint = "online_quant". Engine
    quantizes at load time via the user-supplied precision_config.

Detection (either signal sufficient):
  - Path-level: source transformer's parent chain (≤3 levels up) has a
    `model_index.json` sibling. HF diffusers convention.
  - Metadata-level: file carries a QuantFunc marker (legacy: standalone
    prequant file dropped into ComfyUI's models/diffusion_models/).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .base import BuildContext, FormatAdapter, SourceBundle, StagingResult
from .factory import adapter
from .tools import (
    fingerprint_arch_from_keys,
    read_safetensors_metadata,
)
from .tools.arch_fingerprint import QUANTFUNC_MARKERS

logger = logging.getLogger(__name__)


def _has_qf_marker(path: str) -> bool:
    try:
        meta = read_safetensors_metadata(path)
    except Exception:
        return False
    return any(m in meta for m in QUANTFUNC_MARKERS)


def _walk_to_model_dir(start: Path) -> Path | None:
    """Walk parent chain up to 3 levels looking for a `model_index.json`
    sibling (HF diffusers convention). Returns the dir or None."""
    cur = start
    for _ in range(3):
        if (cur / "model_index.json").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


@adapter(priority=100)
class HFLayoutAdapter(FormatAdapter):
    """HF-diffusers model_dir handler. Two cases, one path:
      - QuantFunc prequant export (has metadata marker) → `prequant_ours`
      - Plain BF16/FP16 base (no quant metadata) → `online_quant`
        (the engine quantizes at load via the user-supplied
         precision_config).

    Detection: source transformer's parent chain (up to 3 levels) has a
    `model_index.json` sibling, OR the file carries a QuantFunc metadata
    marker (legacy standalone prequant files).
    """

    @classmethod
    def detect(cls, sources: SourceBundle) -> bool:
        ref = sources.transformer or sources.checkpoint
        if not ref:
            return False
        # Path-level signal: HF model_dir layout.
        if _walk_to_model_dir(Path(ref.path).parent) is not None:
            return True
        # Metadata-level signal: legacy QF prequant file sitting standalone.
        # BUT defer to BundledCheckpointAdapter if the file is a multi-
        # component bundle (qf_flat_bundle / qwen25_bundle / qwen2vl_bundle).
        # qf_flat exports carry the QuantFunc method marker AND bundle
        # prefixes (model.diffusion_model.* + text_encoder.* + vae.*); without
        # this guard we'd stage only the transformer slice and fail at VAE
        # load with "Failed to load safetensors: <staging>/vae" because
        # add_text_encoder/add_vae never ran.
        if _has_qf_marker(ref.path):
            try:
                from .bundled_checkpoint import _select_scheme
                if _select_scheme(ref.path) is not None:
                    return False
            except Exception:
                pass
            return True
        return False

    def adapt(self, sources: SourceBundle, staging_dir: Path,
              context: BuildContext) -> StagingResult:
        ref = sources.transformer or sources.checkpoint
        # Method hint: prequant if metadata says so, else online_quant
        # (BF16/FP16 base getting runtime-quantized via precision_config).
        method_hint = "prequant_lighting_separate" if _has_qf_marker(ref.path) else "online_quant"
        src = Path(ref.path)
        existing = _walk_to_model_dir(src.parent)
        if existing is not None:
            arch = fingerprint_arch_from_keys(ref.path) or ref.arch
            if not arch:
                # #270-residual: the key-fingerprint can fail on a SHARD of a
                # sharded transformer (e.g. qwen diffusion_pytorch_model-00001-
                # of-00005.safetensors) -> arch stayed "" -> the downstream
                # arch-gated knobs (fused_mod for QwenImage/QwenImageEdit) were
                # silently skipped -> the export produced TILED (non-GEMV) mod
                # weights while a later fused reimport expected GEMV -> noise.
                # For an existing HF model_dir the model_index.json _class_name
                # is AUTHORITATIVE (the same signal the engine itself dispatches
                # on) -- derive arch from it generically (strip "Pipeline").
                try:
                    import json as _json
                    with open(Path(existing) / "model_index.json") as _f:
                        _cls = _json.load(_f).get("_class_name", "") or ""
                    arch = _cls[:-len("Pipeline")] if _cls.endswith("Pipeline") else _cls
                    # SAFETY (vuln-CR): arch flows into path components
                    # (bin/tokenizers/<arch>/, transformer_configs/<arch>.json)
                    # — accept only a plain identifier (all real archs are
                    # CamelCase alnum: QwenImageEditPlus, Flux2Klein, ZImage).
                    # A crafted _class_name (e.g. "../../x") is rejected.
                    if arch and not arch.isalnum():
                        logger.warning("[%s] ignoring non-identifier _class_name "
                                       "%r from model_index.json", method_hint, _cls)
                        arch = ""
                    if arch:
                        logger.info("[%s] arch from model_index.json _class_name: %s",
                                    method_hint, arch)
                except Exception:
                    pass
            logger.info("[%s] using existing model_dir %s", method_hint, existing)
            # The model_dir is used as-is. A well-formed HF / QuantFunc export
            # ships its own tokenizer/, but some exports omit it — then the
            # engine fails at load with "Failed to open vocab.json". Backfill
            # from the plugin bundle (idempotent: skipped when one is present)
            # so an existing model_dir is always loadable.
            from .tools.hf_layout import (
                copy_tokenizer as _copy_tokenizer,
                ensure_engine_tokenizer as _ensure_engine_tokenizer,
            )
            _ex_tok = Path(existing) / "tokenizer"
            # The engine needs vocab.json+merges.txt, NOT the fused
            # tokenizer.json — first derive the split files from a fused
            # tokenizer.json if that's all the dir ships (e.g. ideogram-4-fp8).
            _ensure_engine_tokenizer(_ex_tok)
            # Still no engine-native vocab.json (and nothing to derive from) →
            # fall back to the bundled per-arch tokenizer.
            if not (_ex_tok / "vocab.json").is_file():
                _copy_tokenizer(arch, _ex_tok, [])
            # Krea-2 fp8 TE: this branch hands the model_dir to the engine AS-IS —
            # no add_text_encoder(), so the staging-route sweep does not cover it —
            # yet the engine still loads that dir's text_encoder/. An unsupported
            # fp8 layout there hits the implicit-1.0-scale blind spot (silent
            # garbage), so guard the dir's own TE weights too. This dir is only
            # resolvable as Krea2 because this feature added the arch fingerprint,
            # so the hole is ours to close.
            if arch == "Krea2":
                from .tools.krea2_fp8_te import guard_krea2_te_fp8
                # The dir is passed to the engine AS-IS, INCLUDING its own
                # quantfunc_config.json — which may declare the TE prequantized.
                # The engine reads that as skip_fp8_dequant=true and would
                # byte-reinterpret the fp8 into its BF16 container, so the guard
                # must see the SAME flag the engine will (fp8 + prequantized is a
                # refusal, not a pass).
                _preq = False
                try:
                    _qc = Path(existing) / "quantfunc_config.json"
                    if _qc.is_file():
                        _obj = json.loads(_qc.read_text())
                        _te_h = (_obj or {}).get("text_encoder") or {}
                        _preq = bool(_te_h.get("prequantized", False))
                except Exception as e:                    # never fail staging on this
                    logger.debug("[hf_native] Krea-2 prequantized probe failed: %s", e)
                for _te in sorted((Path(existing) / "text_encoder").glob("*.safetensors")):
                    guard_krea2_te_fp8(str(_te), prequantized_hint=_preq)
            return StagingResult(
                model_dir=str(existing),
                arch=arch,
                method_hint=method_hint,
                cleanup_dir=None,
            )

        # No sibling model_index.json found — file lives standalone
        # (e.g. user dropped it into ComfyUI's models/diffusion_models).
        # Build a minimal staging dir referencing it.
        from .tools.hf_layout import (
            HFLayout, copy_tokenizer, bundled_transformer_config,
            bundled_te_config, bundled_vae_config, ARCH_TO_TRANSFORMER_CLASS,
            assert_vae_matches_arch,
        )
        arch = fingerprint_arch_from_keys(ref.path) or ref.arch or "QwenImage"
        layout = HFLayout(staging_dir)
        # Build transformer/config.json:
        #   - Start from bundled per-arch config (correct dims/layers).
        #   - Override num_attention_heads + num_layers + num_single_layers
        #     + joint_attention_dim by probing the source file. This handles
        #     arch variants that share class_name but differ in size
        #     (e.g. Klein 4B vs 9B both use Flux2Transformer2DModel but
        #     have 5+20 vs 8+24 blocks and dim 3072 vs 4096).
        # Without these overrides the C engine reads the bundled config and
        # mis-allocates blocks, producing shape mismatches at load.
        xfm_config = bundled_transformer_config(arch) or {
            # Central arch→transformer-class table (covers QwenImageLayered etc.);
            # the per-arch weight-derived dims are merged in by add_transformer.
            "_class_name": ARCH_TO_TRANSFORMER_CLASS.get(arch, ""),
        }
        # add_transformer derives the architecture dims from the weights and
        # overrides the bundled base (single header read inside the layout) — no
        # separate dim-probe pass is needed here. Pass arch so add_transformer
        # skips re-fingerprinting.
        layout.add_transformer(ref.path, config=xfm_config, arch=arch)
        # Surface sibling config.json files from the source dirs to the
        # staging dir. Without text_encoder/config.json the C engine
        # falls back to defaults (hidden_size=2560, num_heads=32) and
        # computes head_dim = 2560/32 = 80, but real Qwen3 TE has
        # head_dim=128 → tensor copy mismatch [80] vs [128] at load.
        # Same for VAE config.json.
        import os as _os, json as _json
        def _sibling_config(weight_path: str) -> dict | None:
            cfg_path = _os.path.join(_os.path.dirname(weight_path), "config.json")
            try:
                with open(cfg_path) as _f:
                    return _json.load(_f)
            except Exception:
                return None
        # Component config priority: sibling config.json next to the
        # weight file (HF model_dir convention) → bundled per-arch config
        # (covers Pick CLIP/VAE wired from `models/clip/`, `models/vae/`
        # which have no sibling config.json). Without a config the C
        # engine falls back to wrong defaults (e.g. hidden_size=2560 /
        # num_heads=32 → head_dim=80 vs real Qwen2.5-VL head_dim=128 →
        # cudaMemcpy illegal memory access at TE load).
        if sources.text_encoder:
            # Krea-2 fp8 TE: this branch STAGES the TE for the engine, so it must
            # pass the shared fp8 guard — an unsupported fp8 layout would be
            # dequantized with an implicit 1.0 scale => silent garbage.
            if arch == "Krea2":
                from .tools.krea2_fp8_te import guard_krea2_te_fp8
                guard_krea2_te_fp8(sources.text_encoder.path)
            layout.add_text_encoder(
                sources.text_encoder.path,
                config=(_sibling_config(sources.text_encoder.path)
                        or bundled_te_config(arch)))
        if sources.vae:
            # QwenImageLayered needs a 4-channel RGBA VAE: fail LOUD on a wrong
            # (3-ch) wired VAE instead of a cryptic engine conv_out [4]vs[3] crash.
            assert_vae_matches_arch(arch, sources.vae.path)
            layout.add_vae(
                sources.vae.path,
                config=(_sibling_config(sources.vae.path)
                        or bundled_vae_config(arch)))
        # QuantFunc edit-mode prequant exports place vision_encoder weights
        # in a sibling `vision_encoder/` subdir of the TE file (per src/gemm/
        # CLAUDE.md note 26). ComfyUI has no separate VisionEncoderLoader
        # node, so the user can't wire it explicitly — auto-detect by
        # probing for `<TE_dir>/../vision_encoder/model.safetensors`.
        # Without this the C++ engine sees zero `visual.*` keys and edit
        # mode degrades to text-only encoding (no reference-image features
        # → output ignores the input image).
        ve_extra_cfg: dict = {}
        if sources.text_encoder:
            te_parent = _os.path.dirname(_os.path.dirname(
                _os.path.abspath(sources.text_encoder.path)))
            ve_dir = _os.path.join(te_parent, "vision_encoder")
            ve_weights = _os.path.join(ve_dir, "model.safetensors")
            if _os.path.isfile(ve_weights):
                logger.info("[%s] auto-detected sibling vision_encoder/: %s",
                             method_hint, ve_weights)
                # Route through add_vision_encoder so the vision tower's size
                # dims (hidden_size/depth) are DERIVED from the weights — a bare
                # standalone vision encoder with no sibling config would
                # otherwise inherit a wrong hidden_size → the same #267-class
                # shape mismatch the TE fix addresses. For a QuantFunc export
                # the sibling config already carries the dims (derive == no-op).
                layout.add_vision_encoder(
                    ve_weights, _sibling_config(ve_weights))
                ve_extra_cfg["prequantized"] = True
        # Forward the source quantfunc_config.json's per-component metadata
        # if present — prequant exports carry the prequant precision flags
        # there. Without merging, our staging clobbers the source's
        # `vision_encoder.prequantized=true` / `vision_quant=int8` etc., and
        # C++ defaults to fp16 → the freshly-symlinked vision encoder file
        # is loaded with the wrong quant config → garbage embeddings.
        for cfg_root in {_os.path.dirname(p) for p in (
                getattr(sources.text_encoder, "path", None),
                getattr(sources.vae, "path", None),
                getattr(sources.transformer, "path", None),
                getattr(sources.checkpoint, "path", None)) if p}:
            if not cfg_root:
                continue
            # Walk up to 3 parents looking for source quantfunc_config.json
            cur = cfg_root
            for _ in range(3):
                qf = _os.path.join(cur, "quantfunc_config.json")
                if _os.path.isfile(qf):
                    try:
                        src_cfg = _json.loads(open(qf).read())
                    except Exception:
                        break
                    for comp in ("text_encoder", "vision_encoder", "vae",
                                  "vae_encoder", "transformer"):
                        if comp in src_cfg and isinstance(src_cfg[comp], dict):
                            existing = layout._hints.get(comp) or {}
                            # Source wins for per-component prequant flags;
                            # don't overwrite user's text_precision selection.
                            for k, v in src_cfg[comp].items():
                                existing.setdefault(k, v)
                            layout._hints[comp] = existing
                    if src_cfg.get("obfuscated"):
                        layout._hints.setdefault("obfuscated", True)
                    break
                parent = _os.path.dirname(cur)
                if parent == cur:
                    break
                cur = parent
        if ve_extra_cfg:
            ve_h = layout._hints.get("vision_encoder") or {}
            for k, v in ve_extra_cfg.items():
                ve_h.setdefault(k, v)
            layout._hints["vision_encoder"] = ve_h
        # Tokenizer: prefer the one shipped inside the source model_dir (HF
        # base-model downloads — incl. the QuantFunc auto-loader's base model —
        # include a full tokenizer/), so we don't depend on a per-arch tokenizer
        # being bundled in the plugin. Fall back to the bundle otherwise.
        _tok_dirs: list[str] = []
        for _p in (getattr(sources.text_encoder, "path", None),
                    getattr(sources.transformer, "path", None),
                    getattr(sources.vae, "path", None),
                    getattr(sources.checkpoint, "path", None)):
            if _p:
                _ap = _os.path.abspath(_p)
                _tok_dirs.append(_os.path.dirname(_os.path.dirname(_ap)))
                _tok_dirs.append(_os.path.dirname(_ap))
        copy_tokenizer(arch, layout.tokenizer_dir(), _tok_dirs)
        layout.add_scheduler(sources.scheduler_config, arch=arch)
        layout.set_method(method_hint)
        # Plain HF model dirs have no per-component precision metadata —
        # propagate the user's choice so the C++ TE/VAE actually quantize
        # rather than defaulting to FP16 and bloating the export.
        layout.apply_user_precisions(
            text_precision=context.text_precision,
            vae_precision=context.vae_precision)
        # Krea-2 fp8 TE, FINAL check: the sibling-quantfunc_config merge above can
        # set `text_encoder.prequantized=true` from an UNTRUSTED sidecar AFTER the
        # early guard ran. The engine reads that flag as skip_fp8_dequant=true and
        # byte-reinterprets the fp8 into its BF16 container => silent garbage. So
        # re-run the guard against the hints that will ACTUALLY be written, i.e.
        # the same state the engine will see. (fp8 + prequantized => refuse.)
        if arch == "Krea2" and sources.text_encoder is not None:
            from .tools.krea2_fp8_te import guard_krea2_te_fp8 as _guard_te
            _te_hints = layout._hints.get("text_encoder") or {}
            _guard_te(sources.text_encoder.path,
                      prequantized_hint=bool(_te_hints.get("prequantized", False)))
        layout.write_quantfunc_config()
        # When the source carried `vision_encoder/`, the runtime arch is
        # really QwenImageEdit (or another edit pipeline) regardless of
        # what the bare transformer file's keys would suggest. Bump the
        # arch so model_index.json points the C++ loader at the edit
        # pipeline class.
        if ve_extra_cfg and arch == "QwenImage":
            arch = "QwenImageEdit"
        # Krea-2: this branch SYNTHESISES model_index.json (the source had no
        # sibling index), so the distillation marker must be derived from a real
        # signal — never left to a silent default. (An existing source model_dir
        # is returned AS-IS above, preserving its own is_distilled verbatim.)
        from .tools.hf_layout import krea2_is_distilled as _krea2_is_distilled
        layout.write_model_index(
            arch, is_distilled=_krea2_is_distilled(arch, ref.path))
        return StagingResult(
            model_dir=str(staging_dir),
            arch=arch,
            method_hint=method_hint,
            cleanup_dir=str(staging_dir),
        )
