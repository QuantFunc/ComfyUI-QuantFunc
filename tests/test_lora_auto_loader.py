"""Tests for the QuantFunc LoRA Auto Loader `transformer` targeting dropdown
(Wan2.2 A14B two-expert per-LoRA routing — high/low/all, default all).

ENGINE CONTRACT under test (orchestrator-confirmed): ONE options["lora"] list;
  all  -> legacy "path[:scale]" STRING entry (byte-identical default)
  high -> OBJECT entry {"path","scale","target":"high"}  (engine -> transformer only)
  low  -> OBJECT entry {"path","scale","target":"low"}   (engine -> transformer_2 only,
                                                          fail-loud on single-transformer)
The dropdown value passes LITERALLY; no plugin-side family sniffing.

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


def test_default_all_is_byte_identical_legacy_string():
    def run(tmp):
        rel = _fake_lora(tmp, "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors")
        d = _make_model_dir(class_name="WanPipeline")
        node = nodes.QuantFuncLoRAAutoLoader()
        (cfg,) = node.add_lora({"model_dir": d, "options": {}}, rel, 1.0)
        entries = cfg["options"]["lora"]
        assert len(entries) == 1 and isinstance(entries[0], str)   # legacy STRING
        assert entries[0].endswith("_high_noise.safetensors")
        # scale=1.0 -> bare path (no ":1.0" suffix), exactly the old behavior
        assert ":" not in os.path.basename(entries[0])
        # scale != 1.0 -> "path:scale" string, still legacy shape
        (cfg2,) = node.add_lora({"model_dir": d, "options": {}}, rel, 0.8)
        assert isinstance(cfg2["options"]["lora"][0], str)
        assert cfg2["options"]["lora"][0].endswith(":0.8")
    _with_fake_comfyui_dir(run)


def test_high_low_emit_target_object_entries():
    def run(tmp):
        hi = _fake_lora(tmp, "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors")
        lo = _fake_lora(tmp, "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors")
        d = _make_model_dir(class_name="WanPipeline")
        node = nodes.QuantFuncLoRAAutoLoader()
        # chain: high then low (the official pair wiring) — ONE shared list
        (cfg,) = node.add_lora({"model_dir": d, "options": {}}, hi, 1.0, transformer="high")
        (cfg,) = node.add_lora(cfg, lo, 0.8, transformer="low")
        entries = cfg["options"]["lora"]
        assert len(entries) == 2
        assert entries[0] == {"path": entries[0]["path"], "scale": 1.0, "target": "high"}
        assert entries[0]["path"].endswith("_high_noise.safetensors")
        assert entries[1]["target"] == "low" and entries[1]["scale"] == 0.8
        assert entries[1]["path"].endswith("_low_noise.safetensors")
        # entries are JSON-serializable (they ride options into the engine create)
        json.dumps(entries)
    _with_fake_comfyui_dir(run)


def test_mixed_chain_string_and_object_coexist():
    def run(tmp):
        style = _fake_lora(tmp, "some_style_lora.safetensors")
        hi = _fake_lora(tmp, "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors")
        d = _make_model_dir(class_name="WanPipeline")
        node = nodes.QuantFuncLoRAAutoLoader()
        (cfg,) = node.add_lora({"model_dir": d, "options": {}}, style, 1.0)          # all
        (cfg,) = node.add_lora(cfg, hi, 1.0, transformer="high")                     # targeted
        entries = cfg["options"]["lora"]
        assert isinstance(entries[0], str) and isinstance(entries[1], dict)
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
