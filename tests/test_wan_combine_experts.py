"""Tests for the Wan A14B two-expert combine-picker (comfyui_wan_remap + node).

Covers, CPU-only (no engine .so, no GPU):
  - remap_key(): original-Wan → diffusers key translation (spot checks).
  - is_comfyui_wan_single_file() / detect_wan_modality() on synthetic experts.
  - stage_two_expert(): full staging (dequant; atomic lock+tmp+swap) into the two-expert
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
    # Capability probe (not just importability): a foreign sys.modules torch STUB
    # (e.g. another test file's) lacks real dtypes — skip cleanly, never fail.
    _HAS_TORCH = hasattr(torch, "float8_e4m3fn")
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

    model_dir = W.stage_two_expert(high, low, shared, out, boundary_ratio=0.9)
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
def test_dequant_on_non_fp8_expert_passthrough():
    """Dequant staging on fully-fp16 (non-fp8) experts: non-fp8 tensors pass through,
    keys still remapped, output loads clean."""
    src = tempfile.mkdtemp(prefix="qfwan_nofp8_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high, fp8=False)
    _write_expert(low, fp8=False)
    shared = _make_shared_dir()
    out = tempfile.mkdtemp(prefix="qfwan_nofp8out_") + "/stage"
    W.stage_two_expert(high, low, shared, out)
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


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_remap_dequant_rejects_out_of_range_offsets():
    """An adversarial/corrupt header (data_offsets beyond the data region) must fail
    LOUD at plan time, not emit a silently-truncated file."""
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
        W.remap_dequant_file(src, src + ".out")


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_remap_dequant_rejects_implausible_header_len():
    src = tempfile.mkdtemp(prefix="qfwan_bighdr_") + "/big.safetensors"
    with open(src, "wb") as f:
        f.write(struct.pack("<Q", 1 << 40))   # 1 TiB header on a tiny file
        f.write(b"{}")
    with pytest.raises(RuntimeError, match="implausible"):
        W.remap_dequant_file(src, src + ".out")


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


# --------------------------- fixup-round guards (S1-S4, C1-C3, G1) ---------------------------
@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_same_file_both_experts_refused():
    """S1 — the same file staged as both experts = silent quality loss -> raise."""
    src = tempfile.mkdtemp(prefix="qfwan_same_")
    one = os.path.join(src, "wan_high_and_low.safetensors")
    _write_expert(one)
    shared = _make_shared_dir()
    out = tempfile.mkdtemp(prefix="qfwan_sameout_") + "/stage"
    with pytest.raises(RuntimeError, match="SAME file"):
        W.stage_two_expert(one, one, shared, out)


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_swapped_high_low_filenames_refused():
    """C3 — both filename hints present but REVERSED -> raise (silent quality loss)."""
    src = tempfile.mkdtemp(prefix="qfwan_swap_")
    a = os.path.join(src, "wan2.2_t2v_low_noise_14B.safetensors")
    b = os.path.join(src, "wan2.2_t2v_high_noise_14B.safetensors")
    _write_expert(a)
    _write_expert(b)
    shared = _make_shared_dir()
    out = tempfile.mkdtemp(prefix="qfwan_swapout_") + "/stage"
    with pytest.raises(RuntimeError, match="SWAPPED"):
        W.stage_two_expert(a, b, shared, out)   # 'low' passed as high + vice versa


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_fp8_marker_with_missing_scale_sibling_refused():
    """S2 — 'scaled_fp8' marker present but a .scale_weight sibling MISSING -> raise
    (a bare fp8->fp16 cast would be ~20x numerically wrong)."""
    d = tempfile.mkdtemp(prefix="qfwan_noscale_")
    src = os.path.join(d, "bad.safetensors")
    f8 = torch.float8_e4m3fn
    sd = {"patch_embedding.weight": torch.randn(8, 16, 1, 2, 2).half(),
          "blocks.0.self_attn.q.weight": (torch.randn(8, 8) * 0.1).to(f8),
          # NOTE: no blocks.0.self_attn.q.scale_weight sibling
          "blocks.0.ffn.0.weight": (torch.randn(8, 8) * 0.1).to(f8),
          "blocks.0.ffn.0.scale_weight": torch.tensor(0.05),
          "scaled_fp8": torch.tensor(0.0, dtype=f8)}
    save_file(sd, src)
    with pytest.raises(RuntimeError, match="scale_weight"):
        W.remap_dequant_file(src, src + ".out")


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_stage_atomic_no_tmp_leftover_and_stale_tmp_cleaned():
    """S3 — staging is tmp+rename: after success no `<out>.tmp-*` sibling remains,
    and a stale sentinel-carrying tmp dir from an aborted run is cleaned up."""
    src = tempfile.mkdtemp(prefix="qfwan_atomic_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = _make_shared_dir()
    parent = tempfile.mkdtemp(prefix="qfwan_atomicout_")
    out = os.path.join(parent, "stage")
    # plant a stale aborted-run tmp dir (with our sentinel, DEAD pid) + a foreign lookalike
    import subprocess
    _dead = subprocess.Popen(["true"]); _dead.wait()
    stale = out + f".tmp-{_dead.pid}"
    os.makedirs(stale)
    open(os.path.join(stale, W._TMP_SENTINEL), "w").write("x")
    foreign = out + ".tmp-alien"
    os.makedirs(foreign)
    open(os.path.join(foreign, "users_own.txt"), "w").write("keep")
    W.stage_two_expert(high, low, shared, out)
    assert os.path.isfile(os.path.join(out, ".qf_stage_complete"))
    assert not os.path.exists(stale)                       # ours -> cleaned
    assert os.path.isfile(os.path.join(foreign, "users_own.txt"))  # foreign -> kept
    leftovers = [n for n in os.listdir(parent)
                 if n.startswith("stage.tmp-") and n != os.path.basename(foreign)]
    assert leftovers == []                                 # no tmp residue
    assert not os.path.exists(os.path.join(out, W._TMP_SENTINEL))  # sentinel gone


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_estimator_matches_actual_output_bytes():
    """S4 — the free-disk estimator returns EXACTLY the data bytes the dequant writes
    (verified against the real staged file: file size == 8 + header + estimate)."""
    d = tempfile.mkdtemp(prefix="qfwan_est_")
    src = os.path.join(d, "e.safetensors")
    _write_expert(src, fp8=True)
    est = W.estimate_dequant_output_bytes(src)
    dst = os.path.join(d, "out.safetensors")
    W.remap_dequant_file(src, dst)
    with open(dst, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
    assert os.path.getsize(dst) == 8 + hlen + est


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_stage_refuses_when_disk_too_small(monkeypatch):
    """S4 — a too-small free-disk report -> actionable raise BEFORE any write."""
    import shutil as _sh
    src = tempfile.mkdtemp(prefix="qfwan_disk_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = _make_shared_dir()
    out = tempfile.mkdtemp(prefix="qfwan_diskout_") + "/stage"
    Usage = type(_sh.disk_usage("/"))
    monkeypatch.setattr(W.shutil, "disk_usage",
                        lambda p: Usage(total=10**9, used=10**9 - 1024, free=1024))
    with pytest.raises(RuntimeError, match="free disk"):
        W.stage_two_expert(high, low, shared, out)
    assert not os.path.exists(out)                          # nothing written


def test_resolve_boundary_ratio_precedence():
    """C1/C2 — explicit override > shared model_index value > published default;
    explicit 0/out-of-range -> raise."""
    # explicit override wins
    v, srcl = W.resolve_boundary_ratio(0.8, {"boundary_ratio": 0.875}, 16, 16)
    assert v == 0.8 and "override" in srcl
    # inherit the published value from the shared model_index
    v, srcl = W.resolve_boundary_ratio(None, {"boundary_ratio": 0.875}, 16, 16)
    assert v == 0.875 and "inherited" in srcl
    # published per-modality defaults when the base carries none
    v, srcl = W.resolve_boundary_ratio(None, {}, 16, 16)
    assert v == W._PUBLISHED_BOUNDARY_T2V and "t2v" in srcl
    v, srcl = W.resolve_boundary_ratio(None, {}, 36, 16)
    assert v == W._PUBLISHED_BOUNDARY_I2V and "i2v" in srcl
    # explicit 0 / out-of-range -> raise (the engine would silently drop the low expert)
    with pytest.raises(RuntimeError, match="invalid"):
        W.resolve_boundary_ratio(0.0, None, 16, 16)
    with pytest.raises(RuntimeError, match="invalid"):
        W.resolve_boundary_ratio(1.5, None, 16, 16)
    # modality gate: a cross-modality shared dir's boundary is NOT inherited
    v, srcl = W.resolve_boundary_ratio(None, {"boundary_ratio": 0.9}, 16, 16, (36, 16))
    assert v == W._PUBLISHED_BOUNDARY_T2V and "t2v" in srcl
    v, srcl = W.resolve_boundary_ratio(None, {"boundary_ratio": 0.875}, 36, 16, (16, 16))
    assert v == W._PUBLISHED_BOUNDARY_I2V and "i2v" in srcl
    # matching modality -> inherited; unknown shared modality -> inherited (trusted)
    v, srcl = W.resolve_boundary_ratio(None, {"boundary_ratio": 0.85}, 16, 16, (16, 16))
    assert v == 0.85 and "inherited" in srcl
    v, srcl = W.resolve_boundary_ratio(None, {"boundary_ratio": 0.85}, 16, 16, None)
    assert v == 0.85 and "inherited" in srcl


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_stage_inherits_published_boundary_from_shared_model_index():
    """C1 — auto (boundary_ratio=None) inherits the shared model_index's published
    value when the shared dir's MODALITY MATCHES the experts'. Uses 0.85 (== no
    published default) to prove genuine inheritance, and a t2v-shaped shared
    transformer config matching the t2v synthetic experts."""
    src = tempfile.mkdtemp(prefix="qfwan_inh_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = _make_shared_dir()
    json.dump({"_class_name": "WanTransformer3DModel", "in_channels": 16,
               "out_channels": 16, "num_layers": 40},
              open(os.path.join(shared, "transformer", "config.json"), "w"))
    json.dump({"_class_name": "WanPipeline", "boundary_ratio": 0.85},
              open(os.path.join(shared, "model_index.json"), "w"))
    out = tempfile.mkdtemp(prefix="qfwan_inhout_") + "/stage"
    W.stage_two_expert(high, low, shared, out)              # auto
    mi = json.load(open(os.path.join(out, "model_index.json")))
    assert mi["boundary_ratio"] == 0.85


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_cross_modality_shared_dir_boundary_not_inherited():
    """Generality NO-GO repro: t2v experts + an I2V-shaped shared dir (the common
    real pairing — shared components are byte-identical across the A14B releases,
    but its model_index carries the i2v boundary 0.9). AUTO must NOT inherit 0.9;
    it must use the experts' published t2v default 0.875."""
    src = tempfile.mkdtemp(prefix="qfwan_xmod_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high, in_ch=16, out_ch=16)      # t2v experts
    _write_expert(low, in_ch=16, out_ch=16)
    shared = _make_shared_dir()                    # i2v-shaped (in=36,out=16)
    json.dump({"_class_name": "WanImageToVideoPipeline", "boundary_ratio": 0.9},
              open(os.path.join(shared, "model_index.json"), "w"))
    out = tempfile.mkdtemp(prefix="qfwan_xmodout_") + "/stage"
    W.stage_two_expert(high, low, shared, out)     # auto
    mi = json.load(open(os.path.join(out, "model_index.json")))
    assert mi["boundary_ratio"] == W._PUBLISHED_BOUNDARY_T2V   # 0.875, NOT 0.9
    # and the reverse: i2v experts + a t2v-shaped shared dir carrying 0.875
    src2 = tempfile.mkdtemp(prefix="qfwan_xmod2_")
    h2 = os.path.join(src2, "high.safetensors")
    l2 = os.path.join(src2, "low.safetensors")
    _write_expert(h2, in_ch=36, out_ch=16)         # i2v experts
    _write_expert(l2, in_ch=36, out_ch=16)
    shared2 = _make_shared_dir()
    json.dump({"_class_name": "WanTransformer3DModel", "in_channels": 16,
               "out_channels": 16, "num_layers": 40},
              open(os.path.join(shared2, "transformer", "config.json"), "w"))
    json.dump({"_class_name": "WanPipeline", "boundary_ratio": 0.875},
              open(os.path.join(shared2, "model_index.json"), "w"))
    out2 = tempfile.mkdtemp(prefix="qfwan_xmod2out_") + "/stage"
    W.stage_two_expert(h2, l2, shared2, out2)      # auto
    mi2 = json.load(open(os.path.join(out2, "model_index.json")))
    assert mi2["boundary_ratio"] == W._PUBLISHED_BOUNDARY_I2V  # 0.9, NOT 0.875


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_unscaled_fp8_without_marker_dequants():
    """A legitimately-unscaled fp8 file (no scaled_fp8 marker, no scale siblings)
    still dequants via the bare fp8->fp16 cast (no scale requirement absent the
    marker)."""
    d = tempfile.mkdtemp(prefix="qfwan_unscaled_")
    src = os.path.join(d, "u.safetensors")
    f8 = torch.float8_e4m3fn
    w8 = (torch.randn(8, 8) * 0.1).to(f8)
    sd = {"patch_embedding.weight": torch.randn(8, 16, 1, 2, 2).half(),
          "blocks.0.self_attn.q.weight": w8}     # no marker, no scale sibling
    save_file(sd, src)
    dst = os.path.join(d, "u.out.safetensors")
    W.remap_dequant_file(src, dst)
    out = load_file(dst)
    assert torch.equal(out["blocks.0.attn1.to_q.weight"], w8.to(torch.float16))


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_missing_shared_components_refused():
    """G1 — shared dir without vae/ or text_encoder/ -> actionable stage-time error."""
    src = tempfile.mkdtemp(prefix="qfwan_g1_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = tempfile.mkdtemp(prefix="qfwan_g1shared_")   # empty: no vae/text_encoder
    out = tempfile.mkdtemp(prefix="qfwan_g1out_") + "/stage"
    with pytest.raises(RuntimeError, match="missing"):
        W.stage_two_expert(high, low, shared, out)


def test_stage_root_never_plugin_tree(monkeypatch):
    """S5 — without QUANTFUNC_CACHE_DIR and without ComfyUI folder_paths, the staging
    root falls back to the SYSTEM temp dir, never the plugin tree."""
    for _n in ("comfy", "comfy.model_management", "comfy.utils", "folder_paths"):
        sys.modules.setdefault(_n, types.ModuleType(_n))
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    nodes = importlib.import_module(f"{_PKG}.nodes")
    monkeypatch.delenv("QUANTFUNC_CACHE_DIR", raising=False)
    # the stubbed folder_paths has no get_temp_directory -> Exception branch
    root = nodes._wan_combine_stage_root()
    plugin_root = os.path.dirname(os.path.abspath(nodes.__file__))
    assert not root.startswith(plugin_root)
    assert os.path.basename(root) == "qf_wan_combined"


# ------------------- SF1-SF3 concurrency + crash safety -------------------
def _stage_in_child(high, low, shared, out, q):
    """Child-process worker for the concurrency test (module-level for picklability)."""
    try:
        import importlib
        Wm = importlib.import_module(f"{_PKG}.format_adapters.comfyui_wan_remap")
        Wm.stage_two_expert(high, low, shared, out)
        q.put("ok")
    except Exception as e:  # noqa: BLE001
        q.put(f"fail: {e}")


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_concurrent_staging_same_out_dir_both_succeed():
    """SF1 — THE defect: two PROCESSES staging the same out_dir concurrently. The
    lock must serialize them (B waits, then cache-hits A's completed stage) — no
    reap of a live tmp, no half-swapped dir, both succeed, no residue."""
    import multiprocessing as mp
    src = tempfile.mkdtemp(prefix="qfwan_conc_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = _make_shared_dir()
    parent = tempfile.mkdtemp(prefix="qfwan_concout_")
    out = os.path.join(parent, "stage")
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=_stage_in_child, args=(high, low, shared, out, q))
             for _ in range(2)]
    for pr in procs:
        pr.start()
    results = [q.get(timeout=120) for _ in procs]
    for pr in procs:
        pr.join(timeout=120)
    assert results == ["ok", "ok"], results
    # final dir intact + marker valid; zero tmp/trash/half-swap residue
    assert os.path.isfile(os.path.join(out, ".qf_stage_complete"))
    assert os.path.isfile(os.path.join(out, "transformer",
                                       "diffusion_pytorch_model.safetensors"))
    residue = [n for n in os.listdir(parent)
               if ".tmp-" in n or ".trash-" in n]
    assert residue == [], residue


def test_cleanup_never_reaps_live_pid_tmp():
    """SF1 defense-in-depth: a tmp dir owned by a LIVE pid is NEVER reaped by
    `_cleanup_stale_dirs` (liveness gate), while a DEAD pid's tmp is."""
    import subprocess, time as _t
    parent = tempfile.mkdtemp(prefix="qfwan_live_")
    out = os.path.join(parent, "stage")
    live_proc = subprocess.Popen(["sleep", "30"])
    try:
        live_tmp = f"{out}.tmp-{live_proc.pid}"
        os.makedirs(live_tmp)
        open(os.path.join(live_tmp, W._TMP_SENTINEL), "w").write("x")
        dead_proc = subprocess.Popen(["true"]); dead_proc.wait(); _t.sleep(0.05)
        dead_tmp = f"{out}.tmp-{dead_proc.pid}"
        os.makedirs(dead_tmp)
        open(os.path.join(dead_tmp, W._TMP_SENTINEL), "w").write("x")
        W._cleanup_stale_dirs(out)
        assert os.path.isdir(live_tmp)          # live → skipped
        assert not os.path.exists(dead_tmp)     # dead → reaped
    finally:
        live_proc.kill()


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_crash_recovery_trash_and_tmp_reaped_then_stage_succeeds():
    """SF2 — a crash mid-swap leaves `out.trash-<pid>` (old complete, marker inside)
    and/or `out.tmp-<pid>` (sentinel inside) with NO out_dir. The next run must reap
    both (dead pid) and stage successfully."""
    import subprocess
    src = tempfile.mkdtemp(prefix="qfwan_crash_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = _make_shared_dir()
    parent = tempfile.mkdtemp(prefix="qfwan_crashout_")
    out = os.path.join(parent, "stage")
    dead = subprocess.Popen(["true"]); dead.wait()
    trash = f"{out}.trash-{dead.pid}"
    os.makedirs(trash)
    open(os.path.join(trash, ".qf_stage_complete"), "w").write("oldfp")
    tmp = f"{out}.tmp-{dead.pid}"
    os.makedirs(tmp)
    open(os.path.join(tmp, W._TMP_SENTINEL), "w").write("x")
    W.stage_two_expert(high, low, shared, out)
    assert os.path.isfile(os.path.join(out, ".qf_stage_complete"))
    assert not os.path.exists(trash)
    assert not os.path.exists(tmp)


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_own_tmp_reaped_on_exception():
    """SF3 — an exception mid-build (disk full, simulated) leaves NO orphan tmp."""
    import shutil as _sh
    src = tempfile.mkdtemp(prefix="qfwan_sf3_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = _make_shared_dir()
    parent = tempfile.mkdtemp(prefix="qfwan_sf3out_")
    out = os.path.join(parent, "stage")
    # force a failure AFTER tmp creation: break remap by removing the low expert
    # mid-flight is racy — instead patch _dump_json to raise on the model_index step.
    orig = W.remap_dequant_file
    def boom(*a, **k):
        raise RuntimeError("simulated mid-build failure")
    W.remap_dequant_file = boom
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            W.stage_two_expert(high, low, shared, out)
    finally:
        W.remap_dequant_file = orig
    residue = [n for n in os.listdir(parent) if ".tmp-" in n]
    assert residue == [], residue                # SF3: own tmp reaped
    assert not os.path.exists(out)               # nothing half-staged at the final path


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_swap_failure_second_replace_rolls_back_old_stage(monkeypatch):
    """SF2 rollback: a failure of the SECOND os.replace (tmp->out) must restore the
    OLD complete stage at out_dir (never leave it absent), keep the completed tmp
    (marker inside — NOT reaped), and a follow-up run must recover + succeed."""
    src = tempfile.mkdtemp(prefix="qfwan_swap2_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = _make_shared_dir()
    parent = tempfile.mkdtemp(prefix="qfwan_swap2out_")
    out = os.path.join(parent, "stage")
    W.stage_two_expert(high, low, shared, out)                 # old stage in place
    old_marker = open(os.path.join(out, ".qf_stage_complete")).read()

    real_replace = os.replace
    failed = {"n": 0}
    def failing_replace(a, b):
        # fail ONLY the first dst==out call (the tmp->out swap-in); the ROLLBACK's
        # trash->out rename (also dst==out) must be allowed through — it models a
        # transient swap-in fault (ENOSPC/EIO class) vs the metadata-only rollback.
        if os.path.abspath(b) == os.path.abspath(out) and failed["n"] == 0:
            failed["n"] = 1
            raise OSError("simulated failure of the second replace")
        return real_replace(a, b)
    monkeypatch.setattr(W.os, "replace", failing_replace)
    with pytest.raises(OSError, match="second replace"):
        W.stage_two_expert(high, low, shared, out, force=True)
    monkeypatch.setattr(W.os, "replace", real_replace)

    # OLD stage rolled back to the canonical path, marker intact
    assert os.path.isfile(os.path.join(out, ".qf_stage_complete"))
    assert open(os.path.join(out, ".qf_stage_complete")).read() == old_marker
    # the completed tmp was NOT reaped (it held the only new build)
    tmps = [n for n in os.listdir(parent) if ".tmp-" in n]
    assert len(tmps) == 1
    assert os.path.isfile(os.path.join(parent, tmps[0], ".qf_stage_complete"))
    # follow-up run recovers: reaps the marker-carrying tmp + stages successfully
    W.stage_two_expert(high, low, shared, out, force=True)
    assert os.path.isfile(os.path.join(out, ".qf_stage_complete"))
    residue = [n for n in os.listdir(parent) if ".tmp-" in n or ".trash-" in n]
    assert residue == [], residue


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_swap_failure_first_replace_leaves_old_intact(monkeypatch):
    """SF2: a failure of the FIRST os.replace (out->trash) leaves the OLD stage
    untouched at out_dir (rename is atomic: either moved or not)."""
    src = tempfile.mkdtemp(prefix="qfwan_swap1_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = _make_shared_dir()
    parent = tempfile.mkdtemp(prefix="qfwan_swap1out_")
    out = os.path.join(parent, "stage")
    W.stage_two_expert(high, low, shared, out)
    real_replace = os.replace
    def failing_replace(a, b):
        if ".trash-" in os.path.basename(b):                    # the out->trash aside
            raise OSError("simulated failure of the first replace")
        return real_replace(a, b)
    monkeypatch.setattr(W.os, "replace", failing_replace)
    with pytest.raises(OSError, match="first replace"):
        W.stage_two_expert(high, low, shared, out, force=True)
    monkeypatch.setattr(W.os, "replace", real_replace)
    assert os.path.isfile(os.path.join(out, ".qf_stage_complete"))   # old intact


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_compound_fault_both_replaces_fail_loud_and_recoverable(monkeypatch, caplog):
    """COMPOUND persistent fault: the swap-in AND the (retried) rollback both fail.
    Accepted degraded behavior, locked in: exception propagates LOUD, an ERROR names
    both recovery dirs, out_dir may be absent BUT both complete stages survive
    marker-carrying, and a follow-up run (fault cleared) self-heals."""
    import logging as _logging
    src = tempfile.mkdtemp(prefix="qfwan_cmp_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = _make_shared_dir()
    parent = tempfile.mkdtemp(prefix="qfwan_cmpout_")
    out = os.path.join(parent, "stage")
    W.stage_two_expert(high, low, shared, out)                 # old stage in place

    real_replace = os.replace
    def persistent_failure(a, b):
        if os.path.abspath(b) == os.path.abspath(out):          # swap-in AND rollback
            raise OSError("simulated persistent fault touching out_dir")
        return real_replace(a, b)
    monkeypatch.setattr(W.os, "replace", persistent_failure)
    with caplog.at_level(_logging.ERROR):
        with pytest.raises(OSError, match="persistent fault"):
            W.stage_two_expert(high, low, shared, out, force=True)
    monkeypatch.setattr(W.os, "replace", real_replace)

    # LOUD: the compound-fault error names both recovery dirs
    assert any("COMPOUND fault" in r.message for r in caplog.records)
    # both complete stages survive, marker-carrying (nothing lost)
    leftovers = [n for n in os.listdir(parent) if ".tmp-" in n or ".trash-" in n]
    assert len(leftovers) == 2, leftovers
    for n in leftovers:
        assert os.path.isfile(os.path.join(parent, n, ".qf_stage_complete"))
    # follow-up run (fault cleared) self-heals: reaps both + stages successfully
    W.stage_two_expert(high, low, shared, out, force=True)
    assert os.path.isfile(os.path.join(out, ".qf_stage_complete"))
    residue = [n for n in os.listdir(parent) if ".tmp-" in n or ".trash-" in n]
    assert residue == [], residue


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_swap_failure_rollback_retry_rescues_transient(monkeypatch):
    """The retry loop's value proposition: swap-in fails, the FIRST rollback attempt
    ALSO fails (transient), the SECOND rollback attempt succeeds -> old stage restored
    at out_dir."""
    src = tempfile.mkdtemp(prefix="qfwan_retry_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = _make_shared_dir()
    parent = tempfile.mkdtemp(prefix="qfwan_retryout_")
    out = os.path.join(parent, "stage")
    W.stage_two_expert(high, low, shared, out)
    old_marker = open(os.path.join(out, ".qf_stage_complete")).read()
    real_replace = os.replace
    fails = {"n": 0}
    def transient(a, b):
        if os.path.abspath(b) == os.path.abspath(out) and fails["n"] < 2:
            fails["n"] += 1          # fail the swap-in AND the 1st rollback attempt
            raise OSError("transient fault")
        return real_replace(a, b)
    monkeypatch.setattr(W.os, "replace", transient)
    with pytest.raises(OSError, match="transient"):
        W.stage_two_expert(high, low, shared, out, force=True)
    monkeypatch.setattr(W.os, "replace", real_replace)
    # the 2nd rollback attempt succeeded -> old stage back at the canonical path
    assert open(os.path.join(out, ".qf_stage_complete")).read() == old_marker


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_marker_written_before_sentinel_removed(monkeypatch):
    """N-a ordering invariant: at the instant the sentinel is unlinked, the
    completion marker must ALREADY exist in the tmp dir — so a SIGKILL between the
    two operations can never leave a neither-proof (unreapable) orphan."""
    src = tempfile.mkdtemp(prefix="qfwan_ord_")
    high = os.path.join(src, "high.safetensors")
    low = os.path.join(src, "low.safetensors")
    _write_expert(high)
    _write_expert(low)
    shared = _make_shared_dir()
    out = tempfile.mkdtemp(prefix="qfwan_ordout_") + "/stage"
    seen = {"checked": False}
    real_unlink = os.unlink
    def checking_unlink(path):
        if os.path.basename(path) == W._TMP_SENTINEL:
            # the marker must already be present in the same dir
            assert os.path.isfile(os.path.join(os.path.dirname(path),
                                               ".qf_stage_complete")),                 "sentinel removed BEFORE the marker was written (neither-proof window)"
            seen["checked"] = True
        return real_unlink(path)
    monkeypatch.setattr(W.os, "unlink", checking_unlink)
    W.stage_two_expert(high, low, shared, out)
    assert seen["checked"], "sentinel unlink was never observed"
    assert os.path.isfile(os.path.join(out, ".qf_stage_complete"))


class _FakeK32:
    """Mock kernel32 for the nt branch of _pid_alive (no real Windows needed)."""
    def __init__(self, handle, exit_code=259, last_error=0, gec_ok=True):
        self._h, self._code, self._le, self._ok = handle, exit_code, last_error, gec_ok
    def OpenProcess(self, access, inherit, pid):
        return self._h
    def GetLastError(self):
        return self._le
    def GetExitCodeProcess(self, h, code_ref):
        if not self._ok:
            return 0
        code_ref._obj.value = self._code
        return 1
    def CloseHandle(self, h):
        return 1


def test_pid_alive_nt_branch_mocked(monkeypatch):
    """Windows liveness probe, mock-exercised on Linux: OpenProcess denied => alive;
    handle + STILL_ACTIVE => alive; handle + exited => dead; GetExitCodeProcess
    failure => alive (leak-safe); never os.kill on nt."""
    import ctypes, types as _types
    monkeypatch.setattr(W.os, "name", "nt")
    def probe(k32):
        monkeypatch.setattr(ctypes, "windll",
                            _types.SimpleNamespace(kernel32=k32), raising=False)
        return W._pid_alive(4242)
    STILL_ACTIVE, ERROR_ACCESS_DENIED = 259, 5
    assert probe(_FakeK32(handle=0, last_error=ERROR_ACCESS_DENIED)) is True   # denied => alive
    assert probe(_FakeK32(handle=0, last_error=87)) is False                   # invalid pid => dead
    assert probe(_FakeK32(handle=123, exit_code=STILL_ACTIVE)) is True         # running
    assert probe(_FakeK32(handle=123, exit_code=0)) is False                   # exited
    assert probe(_FakeK32(handle=123, gec_ok=False)) is True                   # ambiguous => alive


def test_stage_lock_msvcrt_retry_loop_mocked(monkeypatch):
    """The Windows lock branch, mock-exercised on Linux: fcntl import blocked =>
    msvcrt path; LK_LOCK raising OSError twice (the documented ~10s-retry timeout)
    must be retried until it succeeds — the loop emulates flock's indefinite block."""
    import sys as _sys, types as _types
    calls = {"lock": 0, "unlock": 0}
    fake = _types.ModuleType("msvcrt")
    fake.LK_LOCK, fake.LK_UNLCK = 0, 2
    def locking(fd, mode, n):
        if mode == fake.LK_LOCK:
            calls["lock"] += 1
            if calls["lock"] < 3:
                raise OSError("lock timeout (simulated msvcrt 10s retry expiry)")
        else:
            calls["unlock"] += 1
    fake.locking = locking
    monkeypatch.setitem(_sys.modules, "fcntl", None)     # import fcntl -> ImportError
    monkeypatch.setitem(_sys.modules, "msvcrt", fake)
    d = tempfile.mkdtemp(prefix="qfwan_ntlock_")
    out = os.path.join(d, "stage")
    with W._stage_lock(out):
        pass
    assert calls["lock"] == 3      # retried through 2 timeouts, then acquired
    assert calls["unlock"] == 1    # released on exit
    assert os.path.isfile(out + W._STAGE_LOCK_SUFFIX)


def test_cleanup_reaps_marker_only_tmp():
    """The sentinel-off->marker-in crash WINDOW: a dead-pid tmp carrying the marker
    (sentinel already removed) must be recognized as OURS and reaped."""
    import subprocess
    parent = tempfile.mkdtemp(prefix="qfwan_mkonly_")
    out = os.path.join(parent, "stage")
    dead = subprocess.Popen(["true"]); dead.wait()
    tmp = f"{out}.tmp-{dead.pid}"
    os.makedirs(tmp)
    open(os.path.join(tmp, ".qf_stage_complete"), "w").write("fp")   # marker, NO sentinel
    W._cleanup_stale_dirs(out)
    assert not os.path.exists(tmp)


def test_published_boundary_values_pinned():
    """N1 — pin the published boundary VALUES (not just key presence): the module
    constants must equal the official releases' model_index values (t2v 0.875 /
    i2v 0.9); guarded cross-check against the REAL local i2v reference when present."""
    assert W._PUBLISHED_BOUNDARY_T2V == 0.875
    assert W._PUBLISHED_BOUNDARY_I2V == 0.9
    ref = "/media/jonathan/Data/ComfyUI/models/diffusers/wan2.2-I2V-A14B-Diffusers/model_index.json"
    if os.path.isfile(ref):
        mi = json.load(open(ref))
        assert mi["boundary_ratio"] == W._PUBLISHED_BOUNDARY_I2V


def test_hint_check_tokenized_no_false_positive():
    """N3 — 'flow'/'highway' substrings must NOT trigger the swap detector; genuine
    reversed high/low tokens still raise."""
    # no raise: hints are substrings of other words → treated as hint-less (warn only)
    W._expert_basename_hint_check("/x/flow_expert.safetensors",
                                  "/x/highway_expert.safetensors")
    # genuine reversed tokens still raise
    with pytest.raises(RuntimeError, match="SWAPPED"):
        W._expert_basename_hint_check("/x/wan_low_noise.safetensors",
                                      "/x/wan_high_noise.safetensors")


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
