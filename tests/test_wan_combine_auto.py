"""Tests for the QuantFunc Wan Combine Experts (Auto) loader node + its scan layer.

The Auto node is a zero-typing loader: it scans the model dirs for Wan A14B
single-file experts + Wan diffusers dirs, offers them as dropdowns (high / low /
shared), and on run COMBINES the two experts into the engine's two-transformer
model_dir INTERNALLY (delegating to the gen-verified `stage_two_expert`) then
outputs the three standard MODEL / CLIP / VAE handles straight into Build Pipeline.
Staging is hidden; no user-facing model_dir.

Covered (pure filesystem + a MOCKED stage; no engine .so, no GPU):
  - list_wan_a14b_choices: expert listing (weight-confirmed, 5b-excluded, renamed
    still detected), shared-dir listing (Wan-only, A14B-family first), labels.
  - the node: dropdowns populated; load() resolves labels→paths, delegates the
    EXACT (high,low,shared) tuple to stage_two_expert (byte-identical to the manual
    node), and returns MODEL/CLIP/VAE stubs that let Build Pipeline's hf_native
    recover the WHOLE staged dir (transformer_2/ + boundary preserved).

Run:  python3 -m pytest tests/test_wan_combine_auto.py -q
"""
import os
import sys
import json
import importlib
import tempfile

import pytest

_PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PLUGIN)
_PKG = os.path.basename(_PLUGIN)
_TESTS = os.path.dirname(os.path.abspath(__file__))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

W = importlib.import_module(f"{_PKG}.format_adapters.comfyui_wan_remap")
twe = importlib.import_module("test_wan_combine_experts")   # DRY: reuse builders
_HAS_TORCH = twe._HAS_TORCH
_write_expert = twe._write_expert


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mk_root(*named_experts):
    root = tempfile.mkdtemp(prefix="qfwan_root_")
    for fname, in_ch, out_ch in named_experts:
        _write_expert(os.path.join(root, fname), in_ch=in_ch, out_ch=out_ch)
    return root


def _mk_wan_diffusers_dir(name="wan2.2-i2v-A14B-Diffusers", a14b=True,
                          mi_class="WanImageToVideoPipeline"):
    """A Wan diffusers dir (A14B = dual transformer_2/) with vae/ + text_encoder/."""
    parent = tempfile.mkdtemp(prefix="qfwan_diff_")
    d = os.path.join(parent, name)
    subs = ["transformer", "transformer_2"] if a14b else ["transformer"]
    for sub in subs:
        os.makedirs(os.path.join(d, sub))
        json.dump({"_class_name": "WanTransformer3DModel", "in_channels": 36,
                   "out_channels": 16},
                  open(os.path.join(d, sub, "config.json"), "w"))
    for sub in ("vae", "text_encoder", "tokenizer", "scheduler"):
        os.makedirs(os.path.join(d, sub))
    json.dump({"_class_name": "AutoencoderKLWan", "z_dim": 16},
              open(os.path.join(d, "vae", "config.json"), "w"))
    json.dump({"_class_name": mi_class}, open(os.path.join(d, "model_index.json"), "w"))
    return d


def _mk_nonwan_diffusers_dir(name="Qwen-Image", mi_class="QwenImagePipeline"):
    parent = tempfile.mkdtemp(prefix="qfnw_")
    d = os.path.join(parent, name)
    for sub in ("transformer", "vae", "text_encoder"):
        os.makedirs(os.path.join(d, sub))
    json.dump({"_class_name": "AutoencoderKLQwenImage"},
              open(os.path.join(d, "vae", "config.json"), "w"))
    json.dump({"_class_name": mi_class}, open(os.path.join(d, "model_index.json"), "w"))
    return d


# --------------------------------------------------------------------------- #
# list_wan_a14b_choices — the dropdown data
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_choices_list_experts_and_shared():
    root = _mk_root(
        ("wan2.2_t2v_high_14B_fp8_scaled.safetensors", 16, 16),
        ("wan2.2_t2v_low_14B_fp8_scaled.safetensors", 16, 16),
        ("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 36, 16),
        ("wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", 36, 16),
    )
    a14b = _mk_wan_diffusers_dir()
    experts, shared = W.list_wan_a14b_choices([root, os.path.dirname(a14b)])
    # all 4 expert files listed, each label → its abs path
    assert len(experts) == 4
    assert any("t2v_high" in k for k in experts) and any("i2v_low" in k for k in experts)
    for label, path in experts.items():
        assert os.path.isfile(path) and label
    # the A14B diffusers dir is a shared choice
    assert os.path.basename(a14b) in shared
    assert os.path.isdir(shared[os.path.basename(a14b)])


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_choices_exclude_ti2v_5b_expert():
    root = _mk_root(("wan2.2_ti2v_5B_fp16.safetensors", 48, 48))
    experts, _ = W.list_wan_a14b_choices([root])
    assert experts == {}, "the 5B single model (no high/low, 5b) is not an A14B expert"


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_choices_renamed_expert_still_listed():
    # no `14b` token, but a high/low token + Wan header → still an A14B expert.
    root = _mk_root(("wan_high_noise.safetensors", 36, 16),
                    ("wan_low_noise.safetensors", 36, 16))
    experts, _ = W.list_wan_a14b_choices([root])
    assert len(experts) == 2


def test_choices_shared_is_wan_only_a14b_first():
    a14b = _mk_wan_diffusers_dir(name="wan2.2-I2V-A14B-Diffusers", a14b=True)
    b5 = _mk_wan_diffusers_dir(name="wan2.2-TI2V-5B-Diffusers", a14b=False,
                               mi_class="WanPipeline")
    qwen = _mk_nonwan_diffusers_dir(name="Qwen-Image")
    roots = [os.path.dirname(a14b), os.path.dirname(b5), os.path.dirname(qwen)]
    _, shared = W.list_wan_a14b_choices(roots)
    labels = list(shared)
    assert "Qwen-Image" not in labels, "a non-Wan dir must not be a shared choice"
    assert set(labels) == {"wan2.2-I2V-A14B-Diffusers", "wan2.2-TI2V-5B-Diffusers"}
    assert labels[0] == "wan2.2-I2V-A14B-Diffusers", "A14B-family shared dir listed first"


def test_is_wan_diffusers_dir():
    assert W._is_wan_diffusers_dir(_mk_wan_diffusers_dir(mi_class="WanImageToVideoPipeline"))
    assert W._is_wan_diffusers_dir(_mk_wan_diffusers_dir(a14b=False, mi_class="WanPipeline"))
    assert not W._is_wan_diffusers_dir(_mk_nonwan_diffusers_dir())


def test_labeled_disambiguates_basename_collision():
    # same basename in two roots → disambiguated with the parent dir name.
    import shutil
    a = tempfile.mkdtemp(prefix="qflabA_"); b = tempfile.mkdtemp(prefix="qflabB_")
    pa = os.path.join(a, "dup.safetensors"); pb = os.path.join(b, "dup.safetensors")
    open(pa, "w").close(); open(pb, "w").close()
    lab = W._labeled([pa, pb])
    assert len(lab) == 2 and len(set(lab.values())) == 2  # two distinct paths kept
    shutil.rmtree(a, ignore_errors=True); shutil.rmtree(b, ignore_errors=True)


# --------------------------------------------------------------------------- #
# the node — dropdowns + load() → MODEL/CLIP/VAE (staging hidden)
# --------------------------------------------------------------------------- #
def _import_nodes():
    import types as _t
    for _n in ("comfy", "comfy.model_management", "comfy.utils", "comfy.sd"):
        sys.modules.setdefault(_n, _t.ModuleType(_n))
    fp = sys.modules.get("folder_paths") or _t.ModuleType("folder_paths")
    fp.get_folder_paths = getattr(fp, "get_folder_paths", lambda k: [])
    fp.get_temp_directory = getattr(fp, "get_temp_directory", lambda: tempfile.gettempdir())
    sys.modules["folder_paths"] = fp
    return importlib.import_module(f"{_PKG}.nodes")


def _staged_two_expert_dir():
    """A dir shaped like stage_two_expert's output (transformer + transformer_2 +
    shared + model_index with boundary)."""
    d = tempfile.mkdtemp(prefix="qfwan_staged_")
    for sub in ("transformer", "transformer_2", "text_encoder", "vae",
                "tokenizer", "scheduler"):
        os.makedirs(os.path.join(d, sub))
    open(os.path.join(d, "transformer", "diffusion_pytorch_model.safetensors"), "wb").write(b"HI")
    open(os.path.join(d, "transformer_2", "diffusion_pytorch_model.safetensors"), "wb").write(b"LO")
    open(os.path.join(d, "text_encoder", "model.safetensors"), "wb").write(b"TE")
    open(os.path.join(d, "vae", "diffusion_pytorch_model.safetensors"), "wb").write(b"VAE")
    json.dump({"_class_name": "WanPipeline", "boundary_ratio": 0.9},
              open(os.path.join(d, "model_index.json"), "w"))
    return d


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_node_dropdowns_and_output_shape(monkeypatch):
    nodes = _import_nodes()
    root = _mk_root(
        ("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 36, 16),
        ("wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", 36, 16),
    )
    a14b = _mk_wan_diffusers_dir()
    monkeypatch.setattr(nodes, "_wan_a14b_scan_roots",
                        lambda: [root, os.path.dirname(a14b)])
    nodes.refresh_wan_a14b_choices()
    it = nodes.QuantFuncWanCombineExpertsAuto.INPUT_TYPES()["required"]
    assert any("high" in x for x in it["high_noise_expert"][0])
    assert any("low" in x for x in it["low_noise_expert"][0])
    assert os.path.basename(a14b) in it["shared_components"][0]
    assert nodes.QuantFuncWanCombineExpertsAuto.RETURN_TYPES == ("MODEL", "CLIP", "VAE")


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_node_load_delegates_identical_tuple_and_outputs_stubs(monkeypatch):
    nodes = _import_nodes()
    root = _mk_root(
        ("wan2.2_t2v_high_14B_fp8_scaled.safetensors", 16, 16),
        ("wan2.2_t2v_low_14B_fp8_scaled.safetensors", 16, 16),
    )
    a14b = _mk_wan_diffusers_dir()
    monkeypatch.setattr(nodes, "_wan_a14b_scan_roots",
                        lambda: [root, os.path.dirname(a14b)])
    nodes.refresh_wan_a14b_choices()

    staged = _staged_two_expert_dir()
    captured = {}

    def fake_stage(high, low, shared, out, *, boundary_ratio=None, force=False):
        captured.update(high=high, low=low, shared=shared, boundary=boundary_ratio)
        return staged
    monkeypatch.setattr(W, "stage_two_expert", fake_stage)

    experts, shared = nodes._get_wan_a14b_choices(force_rescan=True)
    high_lbl = [k for k in experts if "high" in k][0]
    low_lbl = [k for k in experts if "low" in k][0]
    sh_lbl = list(shared)[0]
    model, clip, vae = nodes.QuantFuncWanCombineExpertsAuto().load(
        high_lbl, low_lbl, sh_lbl, boundary_ratio=0.0)

    # the node passed the EXACT resolved (high,low,shared) to stage_two_expert —
    # byte-identical to what the manual node would pass (its own high/low/shared).
    assert captured["high"] == experts[high_lbl]
    assert captured["low"] == experts[low_lbl]
    assert captured["shared"] == shared[sh_lbl]
    assert captured["boundary"] is None            # 0 → AUTO

    # outputs the three standard handles pointing INTO the staged two-expert dir
    assert model.qf_kind == "transformer" and model.qf_model_dir == staged
    assert model.qf_source_path == os.path.join(staged, "transformer",
                                                "diffusion_pytorch_model.safetensors")
    assert clip.qf_source_path == os.path.join(staged, "text_encoder", "model.safetensors")
    assert vae.qf_source_path == os.path.join(staged, "vae",
                                             "diffusion_pytorch_model.safetensors")

    # two-expert preservation: Build Pipeline's hf_native walks the transformer file
    # to the model_index and loads the WHOLE dir (transformer_2/ + boundary).
    hf = importlib.import_module(f"{_PKG}.format_adapters.hf_native")
    from pathlib import Path
    found = hf._walk_to_model_dir(Path(model.qf_source_path).parent)
    assert str(found) == staged
    assert os.path.isdir(os.path.join(str(found), "transformer_2"))
    assert json.load(open(os.path.join(str(found), "model_index.json")))["boundary_ratio"] == 0.9


def test_node_errors_on_unresolvable_selection(monkeypatch):
    nodes = _import_nodes()
    monkeypatch.setattr(nodes, "_wan_a14b_scan_roots", lambda: [])
    nodes.refresh_wan_a14b_choices()
    it = nodes.QuantFuncWanCombineExpertsAuto.INPUT_TYPES()["required"]
    assert it["high_noise_expert"][0] == [nodes._WAN_A14B_NO_EXPERTS]
    assert it["shared_components"][0] == [nodes._WAN_A14B_NO_SHARED]
    with pytest.raises(RuntimeError, match="could not resolve"):
        nodes.QuantFuncWanCombineExpertsAuto().load(
            nodes._WAN_A14B_NO_EXPERTS, nodes._WAN_A14B_NO_EXPERTS,
            nodes._WAN_A14B_NO_SHARED)


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_choices_recurses_into_subfolders():
    # experts + a diffusers dir nested one level under the root (per-model subfolder
    # layout) must be found — matches the plugin's other os.walk dropdown scanners.
    root = tempfile.mkdtemp(prefix="qfnest_")
    sub = os.path.join(root, "wan2.2-a14b"); os.makedirs(sub)
    _write_expert(os.path.join(sub, "wan_high_noise_14B.safetensors"), in_ch=36, out_ch=16)
    _write_expert(os.path.join(sub, "wan_low_noise_14B.safetensors"), in_ch=36, out_ch=16)
    # a REAL nested Wan diffusers dir two levels down (os.walk, like the sibling
    # scanners, does not follow symlinked dirs — so build it as a real tree).
    dd = os.path.join(root, "diffusers_sub", "wan2.2-I2V-A14B-Diffusers")
    for sub in ("transformer", "transformer_2", "vae", "text_encoder"):
        os.makedirs(os.path.join(dd, sub))
    json.dump({"_class_name": "WanTransformer3DModel", "in_channels": 36,
               "out_channels": 16}, open(os.path.join(dd, "transformer", "config.json"), "w"))
    json.dump({"_class_name": "AutoencoderKLWan"}, open(os.path.join(dd, "vae", "config.json"), "w"))
    json.dump({"_class_name": "WanImageToVideoPipeline"}, open(os.path.join(dd, "model_index.json"), "w"))
    experts, shared = W.list_wan_a14b_choices([root])
    assert len(experts) == 2, f"nested experts missed: {list(experts)}"
    assert "wan2.2-I2V-A14B-Diffusers" in shared, f"nested shared dir missed: {list(shared)}"
