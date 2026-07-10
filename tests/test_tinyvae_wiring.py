"""Tests for the Build Pipeline tiny-VAE (TAEHV fast-preview) wiring.

The engine (merged main) supports an opt-in tiny Wan video VAE decoder via
comp_opts `vae_decoder` ("taew2_1" = Wan2.1/A14B 16-ch | "taew2_2" = Wan2.2-5B
48-ch) + `vae_decoder_weights` (operator-provided .safetensors). This suite
covers the plugin-side wiring added to QuantFuncBuildPipeline:

  1. OFF (default) is a strict byte-no-op — the options dict gains NO key.
  2. ON auto-selects the variant from the staged transformer's `out_channels`
     (the SAME latent channel count the engine's TinyVAEDecoder validates z
     against): 16 → taew2_1, 48 → taew2_2 — and injects exactly
     {vae_decoder, vae_decoder_weights}.
  3. Fail-LOUD paths: missing weights (message names the exact expected path
     + the <ComfyUI>/models/QuantFunc/taew/ convention), non-Wan model,
     unsupported latent channel count, unreadable staged configs.
  4. Node surface: `tiny_vae` is an optional BOOLEAN widget defaulting False,
     and build()/IS_CHANGED accept it with default False (old workflows that
     never send the widget keep the exact previous behaviour).

Pure filesystem fixtures — no engine .so, no GPU, no ComfyUI runtime.

Run:  python3 -m pytest tests/test_tinyvae_wiring.py -q     (also plain python3)
"""
import copy
import inspect
import json
import os
import sys
import tempfile
import types
import importlib

import pytest

_PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PLUGIN)
_PKG = os.path.basename(_PLUGIN)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
# Stub heavy optional deps so a logic-only import doesn't drag ComfyUI/torch in.
for _n in ("comfy", "torch", "folder_paths", "comfy.model_management"):
    sys.modules.setdefault(_n, types.ModuleType(_n))

nfa = importlib.import_module(f"{_PKG}.nodes_format_adapters")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _mk_staged_dir(mi_class="WanPipeline", out_channels=16,
                   write_mi=True, write_xfm_cfg=True):
    """A minimal staged model dir: model_index.json + transformer/config.json
    (the two files _resolve_tiny_vae_decoder reads)."""
    d = tempfile.mkdtemp(prefix="qftaew_staged_")
    if write_mi:
        json.dump({"_class_name": mi_class},
                  open(os.path.join(d, "model_index.json"), "w"))
    if write_xfm_cfg:
        os.makedirs(os.path.join(d, "transformer"), exist_ok=True)
        json.dump({"_class_name": "WanTransformer3DModel",
                   "out_channels": out_channels},
                  open(os.path.join(d, "transformer", "config.json"), "w"))
    return d


def _mk_taew_dir(*variants):
    """A taew weights dir containing dummy weight files for the variants."""
    d = tempfile.mkdtemp(prefix="qftaew_weights_")
    for v in variants:
        with open(os.path.join(d, f"{v}.safetensors"), "wb") as f:
            f.write(b"\0" * 16)  # existence is all the resolver checks
    return d


# --------------------------------------------------------------------------- #
# 1. OFF = strict byte-no-op
# --------------------------------------------------------------------------- #
def test_off_injects_nothing():
    options = {"auto_optimize": True, "vae_precision": "auto",
               "text_precision": "int4"}
    before = copy.deepcopy(options)
    # staging dir deliberately NON-Wan + no weights: OFF must not even look.
    nfa._apply_tiny_vae(options, False, "/nonexistent/never/read")
    assert options == before, "tiny_vae=False must be a strict no-op"
    assert "vae_decoder" not in options
    assert "vae_decoder_weights" not in options


def _mk_engine_lib(*tokens, pad_mb=1):
    """A fake engine .so: binary blob optionally embedding the taew tokens
    (mirrors the real gate's string literals living in .rodata)."""
    fd, p = tempfile.mkstemp(prefix="qftaew_lib_", suffix=".so")
    with os.fdopen(fd, "wb") as f:
        f.write(b"\x7fELF" + b"\0" * (pad_mb * 1024 * 1024))
        for t in tokens:
            f.write(t.encode("ascii") + b"\0" * 64)
    return p


# --------------------------------------------------------------------------- #
# 2. ON → correct variant + exact keys
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("out_ch,variant", [(16, "taew2_1"), (48, "taew2_2")])
def test_on_injects_variant_by_latent_channels(out_ch, variant):
    staged = _mk_staged_dir(out_channels=out_ch)
    taew = _mk_taew_dir("taew2_1", "taew2_2")
    lib = _mk_engine_lib("taew2_1", "taew2_2")  # engine supports both
    options = {"auto_optimize": True}
    before = copy.deepcopy(options)
    nfa._apply_tiny_vae(options, True, staged, taew_dir=taew, lib_path=lib)
    added = {k: v for k, v in options.items() if k not in before}
    assert added == {
        "vae_decoder": variant,
        "vae_decoder_weights": os.path.join(taew, f"{variant}.safetensors"),
    }, "ON must inject exactly vae_decoder + vae_decoder_weights"


@pytest.mark.parametrize("mi_class", ["WanPipeline", "WanImageToVideoPipeline"])
def test_on_accepts_any_wan_pipeline_class(mi_class):
    # Engine wan_detect accepts ANY `Wan…`-prefixed pipeline class; mirror it.
    staged = _mk_staged_dir(mi_class=mi_class, out_channels=16)
    taew = _mk_taew_dir("taew2_1")
    variant, weights = nfa._resolve_tiny_vae_decoder(staged, taew_dir=taew)
    assert variant == "taew2_1"
    assert weights.endswith("taew2_1.safetensors")


# --------------------------------------------------------------------------- #
# 3. fail-LOUD paths
# --------------------------------------------------------------------------- #
def test_missing_weights_message_names_exact_path_and_convention():
    staged = _mk_staged_dir(out_channels=16)
    taew = _mk_taew_dir()  # empty — no weight files
    with pytest.raises(RuntimeError) as ei:
        nfa._resolve_tiny_vae_decoder(staged, taew_dir=taew)
    msg = str(ei.value)
    expected = os.path.join(taew, "taew2_1.safetensors")
    assert expected in msg, "error must name the exact missing path"
    assert "models/QuantFunc/taew/taew2_1.safetensors" in msg, \
        "error must state the <ComfyUI>/models/QuantFunc/taew/ convention"


def test_non_wan_model_rejected():
    staged = _mk_staged_dir(mi_class="QwenImagePipeline", out_channels=16)
    taew = _mk_taew_dir("taew2_1")
    with pytest.raises(RuntimeError, match="only supported for Wan video"):
        nfa._resolve_tiny_vae_decoder(staged, taew_dir=taew)


def test_unsupported_latent_channels_rejected():
    staged = _mk_staged_dir(out_channels=32)  # neither 16 nor 48
    taew = _mk_taew_dir("taew2_1", "taew2_2")
    with pytest.raises(RuntimeError, match="unsupported Wan latent channel"):
        nfa._resolve_tiny_vae_decoder(staged, taew_dir=taew)


def test_missing_model_index_rejected():
    staged = _mk_staged_dir(write_mi=False)
    with pytest.raises(RuntimeError, match="model_index.json"):
        nfa._resolve_tiny_vae_decoder(staged, taew_dir=_mk_taew_dir("taew2_1"))


def test_missing_transformer_config_rejected():
    staged = _mk_staged_dir(write_xfm_cfg=False)
    with pytest.raises(RuntimeError, match="out_channels"):
        nfa._resolve_tiny_vae_decoder(staged, taew_dir=_mk_taew_dir("taew2_1"))


# --------------------------------------------------------------------------- #
# 3b. engine version-skew guard (an old .so would SILENTLY ignore the keys)
# --------------------------------------------------------------------------- #
def test_old_engine_lib_without_variant_rejected():
    staged = _mk_staged_dir(out_channels=16)
    taew = _mk_taew_dir("taew2_1")
    old_lib = _mk_engine_lib("taew2_2")  # pre-taew2_1 engine (or none at all)
    options = {}
    with pytest.raises(RuntimeError, match="predates"):
        nfa._apply_tiny_vae(options, True, staged, taew_dir=taew,
                            lib_path=old_lib)
    assert "vae_decoder" not in options, "must refuse BEFORE injecting"


def test_engine_probe_finds_token_across_chunk_boundary():
    # Token straddling the streaming-scan chunk boundary must still be found.
    fd, p = tempfile.mkstemp(prefix="qftaew_lib_", suffix=".so")
    tok = b"taew2_1"
    split = 3  # bytes of the token before the boundary
    with os.fdopen(fd, "wb") as f:
        f.write(b"\0" * (nfa._TAEW_PROBE_CHUNK_BYTES - split))
        f.write(tok)
    assert nfa._engine_lib_supports_taew("taew2_1", lib_path=p) is True


def test_unresolvable_lib_warns_but_proceeds():
    # Non-ComfyUI context (no lib on disk): cannot verify → proceed (log-only).
    staged = _mk_staged_dir(out_channels=48)
    taew = _mk_taew_dir("taew2_2")
    options = {}
    nfa._apply_tiny_vae(options, True, staged, taew_dir=taew,
                        lib_path="/nonexistent/libquantfunc.so")
    assert options.get("vae_decoder") == "taew2_2"


# --------------------------------------------------------------------------- #
# 4. node surface / backward-compat
# --------------------------------------------------------------------------- #
def test_widget_is_optional_boolean_default_false():
    spec = nfa.QuantFuncBuildPipeline.INPUT_TYPES()
    assert "tiny_vae" in spec["optional"], "tiny_vae must be an OPTIONAL widget"
    typ, opts = spec["optional"]["tiny_vae"]
    assert typ == "BOOLEAN"
    assert opts["default"] is False
    assert "tiny_vae" not in spec["required"], \
        "must stay optional so pre-existing workflows load unchanged"


def test_build_and_ischanged_default_false():
    for fn in (nfa.QuantFuncBuildPipeline.build,
               nfa.QuantFuncBuildPipeline.IS_CHANGED.__func__):
        sig = inspect.signature(fn)
        assert "tiny_vae" in sig.parameters
        assert sig.parameters["tiny_vae"].default is False, \
            f"{fn.__name__} must default tiny_vae=False (old workflows)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
