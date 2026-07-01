"""Tests for the Wan A14B two-expert combine-picker (comfyui_wan_remap + node).

Covers, CPU-only (no engine .so, no GPU):
  - remap_key(): original-Wan → diffusers key translation (spot checks).
  - is_comfyui_wan_single_file() / detect_wan_modality() on synthetic experts.
  - stage_two_expert(): full staging (dequant + raw modes) into the two-expert
    diffusers layout — transformer/config channel+depth, remapped+dequant'd
    weights, transformer_2, shared symlinks, VAE absent-key fix, model_index
    with the engine-loadable _class_name ("WanPipeline" for BOTH modalities; the
    engine dispatches i2v by channels, and matches no other Wan class) + boundary_ratio.
  - caching (idempotent skip), modality-mismatch guard.
  - QuantFuncWanCombineExperts node: INPUT_TYPES + registration + required inputs.
  - GUARDED real-file check: single-file → diffusers key-exactness (skips if the
    local Wan A14B models are absent).

Run:  python3 tests/test_wan_combine_experts.py   (also pytest-compatible)
"""
import os
import sys
import json
import types
import struct
import importlib
import tempfile

import pytest

_PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PLUGIN)
_PKG = os.path.basename(_PLUGIN)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

W = importlib.import_module(f"{_PKG}.format_adapters.comfyui_wan_remap")

try:
    import torch
    from safetensors.torch import save_file, load_file
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


# --------------------------- remap_key spot checks ---------------------------
def test_remap_key_block_self_attn():
    assert W.remap_key("blocks.3.self_attn.q.weight") == "blocks.3.attn1.to_q.weight"
    assert W.remap_key("blocks.3.self_attn.o.weight") == "blocks.3.attn1.to_out.0.weight"
    assert W.remap_key("blocks.3.self_attn.norm_q.weight") == "blocks.3.attn1.norm_q.weight"


def test_remap_key_block_cross_attn_ffn_norm():
    assert W.remap_key("blocks.0.cross_attn.k.weight") == "blocks.0.attn2.to_k.weight"
    assert W.remap_key("blocks.0.cross_attn.o.weight") == "blocks.0.attn2.to_out.0.weight"
    assert W.remap_key("blocks.0.ffn.0.weight") == "blocks.0.ffn.net.0.proj.weight"
    assert W.remap_key("blocks.0.ffn.2.weight") == "blocks.0.ffn.net.2.weight"
    assert W.remap_key("blocks.0.norm3.weight") == "blocks.0.norm2.weight"
    assert W.remap_key("blocks.0.modulation") == "blocks.0.scale_shift_table"


def test_remap_key_top_level():
    assert W.remap_key("text_embedding.0.weight") == \
        "condition_embedder.text_embedder.linear_1.weight"
    assert W.remap_key("time_embedding.2.bias") == \
        "condition_embedder.time_embedder.linear_2.bias"
    assert W.remap_key("time_projection.1.weight") == "condition_embedder.time_proj.weight"
    assert W.remap_key("head.head.weight") == "proj_out.weight"
    assert W.remap_key("head.modulation") == "scale_shift_table"
    # pass-through
    assert W.remap_key("patch_embedding.weight") == "patch_embedding.weight"


def test_model_index_class_name_always_engine_loadable():
    """The synthesized model_index must carry the ONE class the engine's family
    detect exact-matches ("WanPipeline") — for BOTH modalities. The engine has no
    "WanImageToVideoPipeline" registration; writing it would throw at load
    (t2v-vs-i2v is channel-driven in the engine, not class-string-driven)."""
    assert W._ENGINE_WAN_PIPELINE_CLASS == "WanPipeline"
    mi = W.synthesize_model_index(None, W._ENGINE_WAN_PIPELINE_CLASS, 0.9)
    assert mi["_class_name"] == "WanPipeline"


def test_vae_absent_key_fix_only_fills_absent():
    fixed = W.synthesize_vae_config({"base_dim": 96, "z_dim": 16})
    assert fixed["decoder_base_dim"] == 96
    assert fixed["is_residual"] is False
    assert fixed["patch_size"] == 1
    assert fixed["out_channels"] == 3
    assert fixed["scale_factor_spatial"] == 8
    # present values are preserved (byte-safe for a Wan2.2/5B VAE)
    keep = W.synthesize_vae_config(
        {"base_dim": 160, "decoder_base_dim": 256, "is_residual": True, "patch_size": 2})
    assert keep["decoder_base_dim"] == 256
    assert keep["is_residual"] is True
    assert keep["patch_size"] == 2


def test_model_index_two_expert_boundary():
    mi = W.synthesize_model_index({"_diffusers_version": "x"}, "WanPipeline", 0.9)
    assert mi["_class_name"] == "WanPipeline"
    assert mi["boundary_ratio"] == 0.9
    assert mi["transformer"][1] == "WanTransformer3DModel"
    assert mi["transformer_2"][1] == "WanTransformer3DModel"


# --------------------------- synthetic experts ---------------------------
def _write_expert(path, in_ch=16, out_ch=16, dim=32, ffn=64, n_blocks=2, fp8=True):
    """A tiny original-Wan single-file expert (t2v when in_ch==out_ch)."""
    assert _HAS_TORCH
    pt, ph, pw = 1, 2, 2
    f8 = torch.float8_e4m3fn
    sd = {}
    sd["patch_embedding.weight"] = torch.randn(dim, in_ch, pt, ph, pw).half()
    sd["patch_embedding.bias"] = torch.randn(dim).half()
    sd["text_embedding.0.weight"] = torch.randn(dim, dim).half()
    sd["text_embedding.0.bias"] = torch.randn(dim).half()
    sd["text_embedding.2.weight"] = torch.randn(dim, dim).half()
    sd["text_embedding.2.bias"] = torch.randn(dim).half()
    sd["time_embedding.0.weight"] = torch.randn(dim, dim).half()
    sd["time_embedding.0.bias"] = torch.randn(dim).half()
    sd["time_embedding.2.weight"] = torch.randn(dim, dim).half()
    sd["time_embedding.2.bias"] = torch.randn(dim).half()
    sd["time_projection.1.weight"] = torch.randn(6 * dim, dim).half()
    sd["time_projection.1.bias"] = torch.randn(6 * dim).half()
    sd["head.head.weight"] = torch.randn(out_ch * pt * ph * pw, dim).half()
    sd["head.head.bias"] = torch.randn(out_ch * pt * ph * pw).half()
    sd["head.modulation"] = torch.randn(2, dim).half()

    def lin(rows, cols):
        if fp8:
            w = (torch.randn(rows, cols) * 0.1).to(f8)
            return w, torch.tensor(0.05)
        return (torch.randn(rows, cols) * 0.1).half(), None

    for b in range(n_blocks):
        for proj in ("q", "k", "v", "o"):
            for attn in ("self_attn", "cross_attn"):
                w, s = lin(dim, dim)
                sd[f"blocks.{b}.{attn}.{proj}.weight"] = w
                if s is not None:
                    sd[f"blocks.{b}.{attn}.{proj}.scale_weight"] = s
        for attn in ("self_attn", "cross_attn"):
            sd[f"blocks.{b}.{attn}.norm_q.weight"] = torch.randn(dim).half()
            sd[f"blocks.{b}.{attn}.norm_k.weight"] = torch.randn(dim).half()
        w, s = lin(ffn, dim)
        sd[f"blocks.{b}.ffn.0.weight"] = w
        if s is not None:
            sd[f"blocks.{b}.ffn.0.scale_weight"] = s
        w, s = lin(dim, ffn)
        sd[f"blocks.{b}.ffn.2.weight"] = w
        if s is not None:
            sd[f"blocks.{b}.ffn.2.scale_weight"] = s
        sd[f"blocks.{b}.norm3.weight"] = torch.randn(dim).half()
        sd[f"blocks.{b}.modulation"] = torch.randn(6, dim).half()
    if fp8:
        sd["scaled_fp8"] = torch.tensor(0.0, dtype=f8)
    save_file(sd, path)


def _make_shared_dir(base_dim=96):
    d = tempfile.mkdtemp(prefix="qfwan_shared_")
    os.makedirs(os.path.join(d, "transformer"))
    json.dump({"_class_name": "WanTransformer3DModel", "in_channels": 36,
               "out_channels": 16, "num_layers": 40, "num_attention_heads": 40,
               "attention_head_dim": 128},
              open(os.path.join(d, "transformer", "config.json"), "w"))
    for sub in ("text_encoder", "tokenizer", "scheduler"):
        os.makedirs(os.path.join(d, sub))
        open(os.path.join(d, sub, "marker.txt"), "w").write(sub)
    os.makedirs(os.path.join(d, "vae"))
    if _HAS_TORCH:
        save_file({"decoder.conv_in.weight": torch.randn(4, 16).half()},
                  os.path.join(d, "vae", "diffusion_pytorch_model.safetensors"))
    json.dump({"_class_name": "AutoencoderKLWan", "base_dim": base_dim, "z_dim": 16},
              open(os.path.join(d, "vae", "config.json"), "w"))
    json.dump({"_class_name": "WanImageToVideoPipeline", "_diffusers_version": "0.35.0"},
              open(os.path.join(d, "model_index.json"), "w"))
    return d


def _hdr(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


# --------------------------- detection on synthetic ---------------------------
@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch to build synthetic experts")
def test_detect_and_modality_synthetic():
    d = tempfile.mkdtemp(prefix="qfwan_det_")
    t2v = os.path.join(d, "t2v.safetensors")
    i2v = os.path.join(d, "i2v.safetensors")
    _write_expert(t2v, in_ch=16, out_ch=16)
    _write_expert(i2v, in_ch=36, out_ch=16)
    assert W.is_comfyui_wan_single_file(t2v)
    assert W.is_comfyui_wan_single_file(i2v)
    assert W.detect_wan_modality(t2v) == (16, 16)
    assert W.detect_wan_modality(i2v) == (36, 16)


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_is_comfyui_wan_false_for_diffusers_keys():
    d = tempfile.mkdtemp(prefix="qfwan_neg_")
    p = os.path.join(d, "diff.safetensors")
    save_file({"blocks.0.attn1.to_q.weight": torch.randn(4, 4).half()}, p)
    assert not W.is_comfyui_wan_single_file(p)


# --------------------------- full staging (dequant) ---------------------------
@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_stage_two_expert_dequant():
    src = tempfile.mkdtemp(prefix="qfwan_src_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high, in_ch=16, out_ch=16, n_blocks=2, fp8=True)
    _write_expert(low, in_ch=16, out_ch=16, n_blocks=2, fp8=True)
    shared = _make_shared_dir(base_dim=96)
    out = tempfile.mkdtemp(prefix="qfwan_out_") + "/stage"

    model_dir = W.stage_two_expert(high, low, shared, out, boundary_ratio=0.9,
                                   dequant_fp8=True)
    assert model_dir == os.path.abspath(out)

    # transformer config: t2v channels + depth from the expert
    cfg = json.load(open(os.path.join(out, "transformer", "config.json")))
    assert cfg["in_channels"] == 16 and cfg["out_channels"] == 16
    assert cfg["num_layers"] == 2
    cfg2 = json.load(open(os.path.join(out, "transformer_2", "config.json")))
    assert cfg2["in_channels"] == 16

    # remapped + dequant'd weights: diffusers keys, fp16, no scale/marker
    hdr = _hdr(os.path.join(out, "transformer", "diffusion_pytorch_model.safetensors"))
    assert "blocks.0.attn1.to_q.weight" in hdr
    assert "blocks.0.ffn.net.0.proj.weight" in hdr
    assert "condition_embedder.text_embedder.linear_1.weight" in hdr
    assert "proj_out.weight" in hdr
    assert not any(k.endswith(".scale_weight") for k in hdr)
    assert "scaled_fp8" not in hdr
    assert all(v["dtype"] in ("F16", "F32") for v in hdr.values())

    # shared symlinks + VAE absent-key fix + model_index
    for sub in ("text_encoder", "tokenizer", "scheduler"):
        assert os.path.islink(os.path.join(out, sub))
    vcfg = json.load(open(os.path.join(out, "vae", "config.json")))
    assert vcfg["decoder_base_dim"] == 96 and vcfg["is_residual"] is False
    assert vcfg["patch_size"] == 1
    # vae weights linked in via link_or_copy (hardlink → symlink → copy); just exists
    assert os.path.exists(os.path.join(out, "vae",
                                       "diffusion_pytorch_model.safetensors"))
    mi = json.load(open(os.path.join(out, "model_index.json")))
    assert mi["_class_name"] == "WanPipeline"    # t2v
    assert mi["boundary_ratio"] == 0.9

    # dequant fidelity spot-check: value == fp8 × scale within fp16 tolerance
    orig = load_file(high)
    staged = load_file(os.path.join(out, "transformer",
                                    "diffusion_pytorch_model.safetensors"))
    w8 = orig["blocks.0.self_attn.q.weight"]
    sc = orig["blocks.0.self_attn.q.scale_weight"]
    ref = (w8.to(torch.float32) * sc.to(torch.float32)).to(torch.float16)
    assert torch.equal(staged["blocks.0.attn1.to_q.weight"], ref)


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_stage_two_expert_raw_preserves_fp8():
    src = tempfile.mkdtemp(prefix="qfwan_raw_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high, fp8=True)
    _write_expert(low, fp8=True)
    shared = _make_shared_dir()
    out = tempfile.mkdtemp(prefix="qfwan_rawout_") + "/stage"
    W.stage_two_expert(high, low, shared, out, dequant_fp8=False)
    wpath = os.path.join(out, "transformer", "diffusion_pytorch_model.safetensors")
    hdr = _hdr(wpath)
    # keys remapped, fp8 preserved, scale/marker dropped
    assert "blocks.0.attn1.to_q.weight" in hdr
    assert hdr["blocks.0.attn1.to_q.weight"]["dtype"] == "F8_E4M3"
    assert not any(k.endswith(".scale_weight") for k in hdr)
    assert "scaled_fp8" not in hdr
    # CRITICAL: the file must be a VALID safetensors after dropping keys — load via
    # the REAL consumer (repacked data_offsets must partition [0,N) with no gaps).
    from safetensors import safe_open
    with safe_open(wpath, framework="pt") as f:  # raises InvalidOffset if corrupt
        keys = set(f.keys())
        assert "blocks.0.attn1.to_q.weight" in keys
        assert f.get_tensor("blocks.0.attn1.to_q.weight").dtype == torch.float8_e4m3fn
        assert f.get_tensor("proj_out.weight").shape[0] > 0   # tail tensor readable (head.head→proj_out)


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_dequant_on_non_fp8_expert_passthrough():
    """dequant_fp8=True on fully-fp16 (non-fp8) experts: non-fp8 tensors pass through,
    keys still remapped, output loads clean."""
    src = tempfile.mkdtemp(prefix="qfwan_nofp8_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high, fp8=False)
    _write_expert(low, fp8=False)
    shared = _make_shared_dir()
    out = tempfile.mkdtemp(prefix="qfwan_nofp8out_") + "/stage"
    W.stage_two_expert(high, low, shared, out, dequant_fp8=True)
    wpath = os.path.join(out, "transformer", "diffusion_pytorch_model.safetensors")
    sd = load_file(wpath)
    assert "blocks.0.attn1.to_q.weight" in sd
    assert sd["blocks.0.attn1.to_q.weight"].dtype == torch.float16
    assert not any(k.endswith(".scale_weight") for k in sd)


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_out_dir_overlapping_source_refused():
    src = tempfile.mkdtemp(prefix="qfwan_ovl_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = _make_shared_dir()
    # out_dir == shared_dir (worst case: a naive rmtree would destroy the source)
    with pytest.raises(RuntimeError, match="overlaps a source"):
        W.stage_two_expert(high, low, shared, shared)
    # out_dir inside shared_dir
    with pytest.raises(RuntimeError, match="overlaps a source"):
        W.stage_two_expert(high, low, shared, os.path.join(shared, "sub"))


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_foreign_nonempty_out_dir_refused():
    src = tempfile.mkdtemp(prefix="qfwan_foreign_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = _make_shared_dir()
    out = tempfile.mkdtemp(prefix="qfwan_foreignout_")
    # a pre-existing user file (no .qf_stage_complete marker) → refuse to touch it
    open(os.path.join(out, "my_important_file.txt"), "w").write("do not delete")
    with pytest.raises(RuntimeError, match="not created by this node"):
        W.stage_two_expert(high, low, shared, out)
    assert os.path.isfile(os.path.join(out, "my_important_file.txt"))  # untouched


def test_remap_file_raw_rejects_out_of_range_offsets():
    """An adversarial/corrupt header (data_offsets beyond the data region) must fail
    LOUD, not emit a silently-truncated file."""
    src = tempfile.mkdtemp(prefix="qfwan_bad_") + "/bad.safetensors"
    hdr = {"blocks.0.self_attn.q.weight": {"dtype": "F16", "shape": [2, 2],
                                           "data_offsets": [0, 999999]},  # far past EOF
           "patch_embedding.weight": {"dtype": "F16", "shape": [1, 16, 1, 2, 2],
                                      "data_offsets": [0, 8]}}
    nh = json.dumps(hdr).encode()
    with open(src, "wb") as f:
        f.write(struct.pack("<Q", len(nh)))
        f.write(nh)
        f.write(b"\x00" * 8)   # only 8 bytes of data region
    with pytest.raises(RuntimeError, match="out of range"):
        W.remap_file_raw(src, src + ".out")


def test_remap_file_raw_rejects_implausible_header_len():
    src = tempfile.mkdtemp(prefix="qfwan_bighdr_") + "/big.safetensors"
    with open(src, "wb") as f:
        f.write(struct.pack("<Q", 1 << 40))   # 1 TiB header on a tiny file
        f.write(b"{}")
    with pytest.raises(RuntimeError, match="implausible"):
        W.remap_file_raw(src, src + ".out")


def test_symlink_replace_never_deletes_real_dir():
    """_stage_shared_dir must never delete a real directory at dst (idempotent skip)."""
    import tempfile as _t
    src = _t.mkdtemp(prefix="qfwan_src2_"); open(os.path.join(src, "a.txt"), "w").write("x")
    parent = _t.mkdtemp(prefix="qfwan_dst2_"); dst = os.path.join(parent, "te")
    os.makedirs(dst); open(os.path.join(dst, "real.txt"), "w").write("keep me")
    W._stage_shared_dir(src, dst)                 # dst is a real dir → skip, no delete
    assert os.path.isfile(os.path.join(dst, "real.txt"))


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_stage_caching_skips_second_run():
    src = tempfile.mkdtemp(prefix="qfwan_cache_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = _make_shared_dir()
    out = tempfile.mkdtemp(prefix="qfwan_cacheout_") + "/stage"
    W.stage_two_expert(high, low, shared, out)
    wpath = os.path.join(out, "transformer", "diffusion_pytorch_model.safetensors")
    mtime1 = os.stat(wpath).st_mtime_ns
    W.stage_two_expert(high, low, shared, out)  # cache hit → no rewrite
    assert os.stat(wpath).st_mtime_ns == mtime1


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_modality_mismatch_raises():
    src = tempfile.mkdtemp(prefix="qfwan_mm_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high, in_ch=16, out_ch=16)
    _write_expert(low, in_ch=36, out_ch=16)   # different variant
    shared = _make_shared_dir()
    out = tempfile.mkdtemp(prefix="qfwan_mmout_") + "/stage"
    with pytest.raises(RuntimeError, match="modality mismatch"):
        W.stage_two_expert(high, low, shared, out)


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_expert_depth_mismatch_raises():
    src = tempfile.mkdtemp(prefix="qfwan_depth_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high, in_ch=16, out_ch=16, n_blocks=3)
    _write_expert(low, in_ch=16, out_ch=16, n_blocks=2)   # same channels, diff depth
    shared = _make_shared_dir()
    out = tempfile.mkdtemp(prefix="qfwan_depthout_") + "/stage"
    with pytest.raises(RuntimeError, match="depth mismatch"):
        W.stage_two_expert(high, low, shared, out)


def test_i2v_experts_stage_engine_loadable_class_with_i2v_channels():
    """Two i2v experts (in>out): the staged model_index carries the ENGINE-LOADABLE
    "WanPipeline" (NOT "WanImageToVideoPipeline", which no engine pipeline matches —
    it would throw at load); i2v-ness is carried by the transformer config channels,
    which is what the engine's is_i2v/vae-encoder gates actually read."""
    if not _HAS_TORCH:
        pytest.skip("needs torch")
    src = tempfile.mkdtemp(prefix="qfwan_i2v_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high, in_ch=36, out_ch=16)
    _write_expert(low, in_ch=36, out_ch=16)
    shared = _make_shared_dir()
    out = tempfile.mkdtemp(prefix="qfwan_i2vout_") + "/stage"
    W.stage_two_expert(high, low, shared, out)
    mi = json.load(open(os.path.join(out, "model_index.json")))
    # the literal rule of the engine's wan_detect: pipeline_class == "WanPipeline"
    # (the transformer_class fallback never fires when _class_name is present)
    assert mi["_class_name"] == "WanPipeline"
    for sub in ("transformer", "transformer_2"):
        cfg = json.load(open(os.path.join(out, sub, "config.json")))
        assert cfg["in_channels"] == 36 and cfg["out_channels"] == 16
        assert cfg["_class_name"] == "WanTransformer3DModel"


# --------------------------- node registration ---------------------------
def test_node_registered_and_input_types():
    for _n in ("comfy", "comfy.model_management", "comfy.utils", "folder_paths"):
        sys.modules.setdefault(_n, types.ModuleType(_n))
    _torch = types.ModuleType("torch")
    _torch.from_numpy = lambda a: a
    sys.modules.setdefault("torch", _torch)
    nodes = importlib.import_module(f"{_PKG}.nodes")
    assert "QuantFuncWanCombineExperts" in nodes.NODE_CLASS_MAPPINGS
    assert "QuantFuncWanCombineExperts" in nodes.NODE_DISPLAY_NAME_MAPPINGS
    cls = nodes.NODE_CLASS_MAPPINGS["QuantFuncWanCombineExperts"]
    it = cls.INPUT_TYPES()
    req = it["required"]
    for k in ("high_noise_expert", "low_noise_expert", "shared_components", "boundary_ratio"):
        assert k in req
    assert cls.RETURN_TYPES == ("STRING",)
    assert cls.RETURN_NAMES == ("model_dir",)


def test_node_missing_inputs_raise():
    for _n in ("comfy", "comfy.model_management", "comfy.utils", "folder_paths"):
        sys.modules.setdefault(_n, types.ModuleType(_n))
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    nodes = importlib.import_module(f"{_PKG}.nodes")
    node = nodes.NODE_CLASS_MAPPINGS["QuantFuncWanCombineExperts"]()
    try:
        node.combine("", "", "", 0.9)
        assert False, "expected RuntimeError for missing inputs"
    except RuntimeError as e:
        assert "required" in str(e)


# --------------------------- GUARDED real-file key exactness ---------------------------
_DM = "/media/jonathan/Data/ComfyUI/models/diffusion_models"
_REF = "/media/jonathan/Data/ComfyUI/models/diffusers/wan2.2-I2V-A14B-Diffusers"
_HIGH = f"{_DM}/wan2.2_t2v_high_14B_fp8_scaled_wan2.2_t2v.safetensors"


@pytest.mark.skipif(not (os.path.isfile(_HIGH) and os.path.isdir(_REF)),
                    reason="local Wan A14B models not present")
def test_real_single_file_key_exact_vs_diffusers():
    man = W.build_wan_xfm_manifest(_HIGH)
    produced = set(man["rename"].keys())
    dk = set()
    for f in os.listdir(f"{_REF}/transformer"):
        if f.endswith(".safetensors"):
            dk |= set(_hdr(os.path.join(f"{_REF}/transformer", f)).keys())
    assert produced == dk, \
        f"missing={sorted(dk - produced)[:5]} extra={sorted(produced - dk)[:5]}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
