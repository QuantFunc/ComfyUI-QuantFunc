"""Tests for the QuantFunc LoRA Auto Loader `transformer` targeting dropdown
(Wan2.2 A14B two-expert per-LoRA routing — high/low/all, default all).

Contract under test (plugin side of the engine wan-LoRA work):
  all  -> options["lora"]        (flat list, byte-identical to the old behavior)
  high -> options["lora_high"]   (engine merges ONLY into transformer)
  low  -> options["lora_low"]    (engine merges ONLY into transformer_2)
  high/low on a NON-Wan pipeline -> warning + fall back to the flat list.

Run:  python3 -m pytest tests/test_lora_auto_loader.py -q
"""
import os
import sys
import json
import tempfile

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)
from test_video_nodes_wan_ltx import _make_model_dir  # noqa: E402 (stubs ride along)
import nodes  # noqa: E402


def _fake_lora(tmpdir, name):
    lora_dir = os.path.join(tmpdir, "models", "loras", "wan2.2")
    os.makedirs(lora_dir, exist_ok=True)
    p = os.path.join(lora_dir, name)
    with open(p, "wb") as f:
        f.write(b"\0" * 16)
    return os.path.join("wan2.2", name)


def _with_fake_comfyui_dir(fn):
    tmp = tempfile.mkdtemp(prefix="qflora_")
    old = nodes._get_comfyui_dir
    nodes._get_comfyui_dir = lambda: tmp
    try:
        return fn(tmp)
    finally:
        nodes._get_comfyui_dir = old


def test_widget_has_transformer_dropdown_appended_after_scale():
    old = nodes._get_lora_file_options
    nodes._get_lora_file_options = lambda: ["None"]
    try:
        it = nodes.QuantFuncLoRAAutoLoader.INPUT_TYPES()["required"]
    finally:
        nodes._get_lora_file_options = old
    assert list(it["transformer"][0]) == ["all", "high", "low"]
    assert it["transformer"][1]["default"] == "all"
    # appended AFTER scale so old saved graphs keep their widget slots
    keys = list(it.keys())
    assert keys.index("transformer") > keys.index("scale")


def test_default_all_is_byte_identical_flat_list():
    def run(tmp):
        rel = _fake_lora(tmp, "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors")
        d = _make_model_dir(class_name="WanPipeline")
        node = nodes.QuantFuncLoRAAutoLoader()
        (cfg,) = node.add_lora({"model_dir": d, "options": {}}, rel, 1.0)
        assert list(cfg["options"].keys()) == ["lora"]          # flat key only
        assert cfg["options"]["lora"][0].endswith("_high_noise.safetensors")
        assert ":" not in os.path.basename(cfg["options"]["lora"][0])  # no scale suffix at 1.0
    _with_fake_comfyui_dir(run)


def test_high_low_route_to_separate_keys_on_wan():
    def run(tmp):
        hi = _fake_lora(tmp, "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors")
        lo = _fake_lora(tmp, "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors")
        d = _make_model_dir(class_name="WanPipeline")
        node = nodes.QuantFuncLoRAAutoLoader()
        # chain: high then low (the official pair wiring)
        (cfg,) = node.add_lora({"model_dir": d, "options": {}}, hi, 1.0, transformer="high")
        (cfg,) = node.add_lora(cfg, lo, 0.8, transformer="low")
        assert cfg["options"]["lora_high"][0].endswith("_high_noise.safetensors")
        assert cfg["options"]["lora_low"][0].endswith("_low_noise.safetensors:0.8")
        assert "lora" not in cfg["options"]                     # nothing leaked to flat
    _with_fake_comfyui_dir(run)


def test_high_on_non_wan_falls_back_to_flat_with_warning():
    def run(tmp):
        rel = _fake_lora(tmp, "some_style_lora.safetensors")
        d = _make_model_dir(class_name="QwenImagePipeline")
        node = nodes.QuantFuncLoRAAutoLoader()
        import logging
        recs = []
        h = logging.Handler(); h.emit = lambda r: recs.append(r.getMessage())
        logging.getLogger().addHandler(h)
        try:
            (cfg,) = node.add_lora({"model_dir": d, "options": {}}, rel, 1.0,
                                   transformer="high")
        finally:
            logging.getLogger().removeHandler(h)
        assert "lora_high" not in cfg["options"]
        assert cfg["options"]["lora"][0].endswith("some_style_lora.safetensors")
        assert any("not Wan" in m for m in recs), recs
    _with_fake_comfyui_dir(run)


def test_missing_file_still_raises():
    def run(tmp):
        d = _make_model_dir(class_name="WanPipeline")
        node = nodes.QuantFuncLoRAAutoLoader()
        try:
            node.add_lora({"model_dir": d, "options": {}}, "nope/missing.safetensors",
                          1.0, transformer="high")
        except RuntimeError as e:
            assert "not found" in str(e)
        else:
            raise AssertionError("expected RuntimeError for missing file")
    _with_fake_comfyui_dir(run)
