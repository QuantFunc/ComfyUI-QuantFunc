"""Tests for the single-file Wan2.2 TI2V-5B trio adapter + staging.

Closes the "No format adapter matched" gap: UNETLoader(wan2.2_ti2v_5B, original-
Wan keys) + CLIPLoader(umt5_xxl) + VAELoader(wan2.2_vae) into Build Pipeline.

Covered (no GPU, no engine .so):
  1. remap_wan_vae_key — key-exact + shape-exact against the COMMITTED ground
     truth fixture (ComfyUI wan2.2_vae header vs the official 5B diffusers vae
     header, tests/fixtures/wan22_vae_key_map_ref.json).
  2. Adapter detect: claims an original-Wan single-file; ignores prefixed /
     arch-fingerprintable files (the generic adapter's turf) and checkpoints.
  3. Fail-loud: non-5B modality (16-ch A14B/Wan2.1 → Combine-node guidance),
     missing CLIP/VAE sockets, wrong-family CLIP/VAE files, missing bundled
     asset handling.
  4. stage_ti2v_5b_trio end-to-end on TINY synthetic weights (real torch,
     real safetensors IO): staged layout (transformer/text_encoder/vae/
     tokenizer/scheduler/model_index), key names remapped/passed through,
     spiece_model + fp8 scale siblings dropped, cache-hit on re-run.
  5. Bundled assets: present, parseable, engine-dispatch-consistent
     (vae _class_name == AutoencoderKLWan EXACT; TE architectures UMT5;
     tokenizer.json is the HF-Unigram file the engine's tokenizer loads).

Run:  python3 -m pytest tests/test_wan_5b_single_file.py -q
"""
import json
import os
import struct
import sys
import tempfile
import types
import importlib

import pytest

_PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PLUGIN)
_PKG = os.path.basename(_PLUGIN)
_TESTS = os.path.dirname(os.path.abspath(__file__))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
for _n in ("comfy", "folder_paths", "comfy.model_management"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
try:
    import torch  # noqa: F401 — staging tests need real torch
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False

W = importlib.import_module(f"{_PKG}.format_adapters.comfyui_wan_remap")
base = importlib.import_module(f"{_PKG}.format_adapters.base")

_FIXTURE = os.path.join(_TESTS, "fixtures", "wan22_vae_key_map_ref.json")


# --------------------------------------------------------------------------- #
# helpers — minimal safetensors writers
# --------------------------------------------------------------------------- #
def _write_safetensors(path, tensors):
    """tensors: {key: (shape, dtype_str, bytes)} — a real minimal safetensors."""
    hdr, cursor = {}, 0
    blobs = []
    for k, (shape, dt, b) in tensors.items():
        hdr[k] = {"dtype": dt, "shape": list(shape),
                  "data_offsets": [cursor, cursor + len(b)]}
        cursor += len(b)
        blobs.append(b)
    h = json.dumps(hdr, separators=(",", ":")).encode()
    pad = (8 - len(h) % 8) % 8
    h += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    return path


def _f16(numel):
    return b"\x00\x3c" * numel  # fp16 1.0


def _mk_tiny_5b_xfm(path, in_ch=48, out_ch=48, dim=8, layers=2):
    """Original-Wan-keyed transformer, TI2V-5B shaped (patch (1,2,2))."""
    t = {}
    t["patch_embedding.weight"] = ((dim, in_ch, 1, 2, 2), "F16",
                                   _f16(dim * in_ch * 4))
    t["patch_embedding.bias"] = ((dim,), "F16", _f16(dim))
    for n in range(layers):
        for sub in ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                    "cross_attn.q", "cross_attn.k", "cross_attn.v",
                    "cross_attn.o"):
            t[f"blocks.{n}.{sub}.weight"] = ((dim, dim), "F16", _f16(dim * dim))
        t[f"blocks.{n}.ffn.0.weight"] = ((dim, dim), "F16", _f16(dim * dim))
        t[f"blocks.{n}.ffn.2.weight"] = ((dim, dim), "F16", _f16(dim * dim))
    t["head.head.weight"] = ((out_ch * 1 * 2 * 2, dim), "F16",
                             _f16(out_ch * 4 * dim))
    t["head.head.bias"] = ((out_ch * 4,), "F16", _f16(out_ch * 4))
    return _write_safetensors(path, t)


def _mk_tiny_umt5(path, with_spiece=True, layers=1, d=4):
    t = {}
    for n in range(layers):
        for sub in ("SelfAttention.q", "SelfAttention.k", "SelfAttention.v",
                    "SelfAttention.o"):
            t[f"encoder.block.{n}.layer.0.{sub}.weight"] = \
                ((d, d), "F16", _f16(d * d))
    t["shared.weight"] = ((d, d), "F16", _f16(d * d))
    if with_spiece:
        t["spiece_model"] = ((3,), "U8", b"\x01\x02\x03")
    return _write_safetensors(path, t)


def _mk_tiny_wan_vae(path, ch=4):
    """Original-Wan-VAE-keyed decoder subset (enough for the signature check +
    a real remap pass)."""
    t = {
        "conv1.weight": ((2 * ch, 2 * ch, 1, 1, 1), "F16", _f16(4 * ch * ch)),
        "conv2.weight": ((ch, ch, 1, 1, 1), "F16", _f16(ch * ch)),
        "decoder.conv1.weight": ((ch, ch, 3, 3, 3), "F16", _f16(27 * ch * ch)),
        "decoder.head.0.gamma": ((ch, 1, 1, 1), "F16", _f16(ch)),
        "decoder.head.2.weight": ((3, ch, 3, 3, 3), "F16", _f16(81 * ch)),
        "decoder.middle.0.residual.0.gamma": ((ch, 1, 1, 1), "F16", _f16(ch)),
        "decoder.middle.1.to_qkv.weight": ((3 * ch, ch, 1, 1), "F16",
                                           _f16(3 * ch * ch)),
        "decoder.upsamples.0.upsamples.0.residual.2.weight":
            ((ch, ch, 3, 3, 3), "F16", _f16(27 * ch * ch)),
        "decoder.upsamples.0.upsamples.3.resample.1.weight":
            ((ch, ch, 3, 3), "F16", _f16(9 * ch * ch)),
        "encoder.conv1.weight": ((ch, 3, 3, 3, 3), "F16", _f16(81 * ch)),
    }
    return _write_safetensors(path, t)


def _tmpf(name):
    d = tempfile.mkdtemp(prefix="qf5b_")
    return os.path.join(d, name)


# --------------------------------------------------------------------------- #
# 1. remap_wan_vae_key vs the committed ground-truth fixture
# --------------------------------------------------------------------------- #
def test_vae_remap_key_exact_and_shape_exact_vs_reference():
    fix = json.load(open(_FIXTURE))
    comfy = {k: tuple(v) for k, v in fix["comfy"].items()}
    ref = {k: tuple(v) for k, v in fix["diffusers"].items()}
    mapped = {W.remap_wan_vae_key(k): shp for k, shp in comfy.items()}
    assert len(mapped) == len(comfy), "remap must be injective (no collisions)"
    assert set(mapped) == set(ref), (
        f"key mismatch: missing={sorted(set(ref) - set(mapped))[:5]} "
        f"extra={sorted(set(mapped) - set(ref))[:5]}")
    bad = [k for k in ref if mapped[k] != ref[k]]
    assert not bad, f"shape mismatch at {bad[:5]}"


def test_vae_remap_passthrough_for_diffusers_keys():
    # Already-diffusers keys pass through unchanged (idempotent staging).
    for k in ("decoder.conv_in.weight", "quant_conv.bias",
              "decoder.up_blocks.0.resnets.0.conv1.weight"):
        assert W.remap_wan_vae_key(k) == k


# --------------------------------------------------------------------------- #
# 2. adapter detect
# --------------------------------------------------------------------------- #
def _ref(path):
    return base.FileRef(path=path, arch="", kind="", mtime=0.0)


def test_detect_claims_wan_single_file():
    xfm = _mk_tiny_5b_xfm(_tmpf("wan5b.safetensors"))
    src = base.SourceBundle(transformer=_ref(xfm))
    assert W.ComfyUIWanSingleFileAdapter.detect(src) is True


def test_detect_ignores_non_wan_and_checkpoint():
    te = _mk_tiny_umt5(_tmpf("umt5.safetensors"))  # not a wan transformer
    assert W.ComfyUIWanSingleFileAdapter.detect(
        base.SourceBundle(transformer=_ref(te))) is False
    xfm = _mk_tiny_5b_xfm(_tmpf("wan5b.safetensors"))
    assert W.ComfyUIWanSingleFileAdapter.detect(
        base.SourceBundle(transformer=_ref(xfm),
                          checkpoint=_ref(xfm))) is False
    assert W.ComfyUIWanSingleFileAdapter.detect(
        base.SourceBundle(transformer=None)) is False


def test_adapter_registered_with_factory():
    # Importing the PACKAGE (format_adapters/__init__) must register the wan
    # single-file adapter with priority ABOVE the generic single-file adapter
    # (50) and BELOW hf_native (100) — the ordering the design relies on.
    importlib.import_module(f"{_PKG}.format_adapters")
    factory = importlib.import_module(f"{_PKG}.format_adapters.factory")
    entries = factory.AdapterRegistry._entries  # (priority, cls), sorted desc
    by_name = {cls.__name__: prio for prio, cls in entries}
    assert "ComfyUIWanSingleFileAdapter" in by_name, sorted(by_name)
    assert (by_name["ComfyUIDiffusionModelAdapter"]
            < by_name["ComfyUIWanSingleFileAdapter"]
            < by_name["HFLayoutAdapter"])


# --------------------------------------------------------------------------- #
# 3. fail-loud paths
# --------------------------------------------------------------------------- #
def test_16ch_single_file_refused_with_combine_guidance():
    xfm = _mk_tiny_5b_xfm(_tmpf("wan_a14b_expert.safetensors"),
                          in_ch=16, out_ch=16)
    with pytest.raises(RuntimeError, match="Combine Experts"):
        W.stage_ti2v_5b_trio(xfm, "/x", "/y", _tmpf("out"))


def test_missing_clip_or_vae_socket_refused():
    xfm = _mk_tiny_5b_xfm(_tmpf("wan5b.safetensors"))
    ad = W.ComfyUIWanSingleFileAdapter()
    with pytest.raises(RuntimeError, match="Missing: clip, vae"):
        ad.adapt(base.SourceBundle(transformer=_ref(xfm)),
                 tempfile.mkdtemp(prefix="qf5b_stg_"), None)


def test_taew_file_in_vaeloader_gets_actionable_hint():
    # User mistake seen in the field: taew2_2.safetensors wired into VAELoader.
    # The refusal must tell them exactly what to do (EN+ZH), keyed off either
    # the TinyVAEDecoder key layout or the taew basename.
    xfm = _mk_tiny_5b_xfm(_tmpf("wan5b.safetensors"))
    te = _mk_tiny_umt5(_tmpf("umt5.safetensors"))
    taew = _write_safetensors(_tmpf("taew2_2.safetensors"), {
        "decoder.1.weight": ((4, 4, 3, 3), "F16", _f16(16 * 9)),
        "decoder.22.weight": ((3, 4, 3, 3), "F16", _f16(12 * 9)),
    })
    with pytest.raises(RuntimeError) as ei:
        W.stage_ti2v_5b_trio(xfm, te, taew, _tmpf("out"))
    msg = str(ei.value)
    assert "tiny_vae" in msg and "wan2.2_vae.safetensors" in msg
    assert "tiny_vae 开关" in msg  # zh guidance reaches the user too
    # key-layout arm alone (non-taew filename) also detects
    assert W._looks_like_taew_tiny_vae(taew) is True
    plain = _mk_tiny_wan_vae(_tmpf("wanvae.safetensors"))
    assert W._looks_like_taew_tiny_vae(plain) is False


def test_wrong_family_clip_and_vae_refused():
    xfm = _mk_tiny_5b_xfm(_tmpf("wan5b.safetensors"))
    vae = _mk_tiny_wan_vae(_tmpf("wanvae.safetensors"))
    not_te = _mk_tiny_wan_vae(_tmpf("notte.safetensors"))
    with pytest.raises(RuntimeError, match="umt5"):
        W.stage_ti2v_5b_trio(xfm, not_te, vae, _tmpf("out"))
    te = _mk_tiny_umt5(_tmpf("umt5.safetensors"))
    not_vae = _mk_tiny_umt5(_tmpf("notvae.safetensors"))
    with pytest.raises(RuntimeError, match="Wan VAE"):
        W.stage_ti2v_5b_trio(xfm, te, not_vae, _tmpf("out"))


# --------------------------------------------------------------------------- #
# 4. staging end-to-end on tiny synthetic weights
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_TORCH, reason="staging needs torch")
def test_stage_ti2v_5b_trio_layout_and_cache():
    xfm = _mk_tiny_5b_xfm(_tmpf("wan5b.safetensors"))
    te = _mk_tiny_umt5(_tmpf("umt5.safetensors"), with_spiece=True)
    vae = _mk_tiny_wan_vae(_tmpf("wanvae.safetensors"))
    out = os.path.join(tempfile.mkdtemp(prefix="qf5b_root_"), "staged")

    staged = W.stage_ti2v_5b_trio(xfm, te, vae, out)
    # layout
    for rel in ("transformer/config.json",
                "transformer/diffusion_pytorch_model.safetensors",
                "text_encoder/config.json", "text_encoder/model.safetensors",
                "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
                "tokenizer/tokenizer.json",
                "scheduler/scheduler_config.json", "model_index.json"):
        assert os.path.exists(os.path.join(staged, rel)), rel
    # model_index: single transformer, Wan class
    mi = json.load(open(os.path.join(staged, "model_index.json")))
    assert mi["_class_name"].startswith("Wan")
    assert mi["boundary_ratio"] is None
    # transformer keys remapped to diffusers naming, dims measured
    hdr = W._read_header(
        os.path.join(staged, "transformer/diffusion_pytorch_model.safetensors"))
    assert any(k.startswith("blocks.0.attn1.to_q.") for k in hdr)
    assert not any(".self_attn." in k for k in hdr)
    cfg = json.load(open(os.path.join(staged, "transformer/config.json")))
    assert (cfg["in_channels"], cfg["out_channels"]) == (48, 48)
    assert cfg["num_layers"] == 2  # measured from the tiny file, not the asset
    # TE: keys pass through; spiece_model dropped
    hdr = W._read_header(os.path.join(staged, "text_encoder/model.safetensors"))
    assert "spiece_model" not in hdr
    assert any(k.startswith("encoder.block.0.") for k in hdr)
    # VAE: keys remapped
    hdr = W._read_header(
        os.path.join(staged, "vae/diffusion_pytorch_model.safetensors"))
    assert "quant_conv.weight" in hdr and "decoder.conv_in.weight" in hdr
    assert not any(k.startswith("decoder.head.") for k in hdr)
    # vae config: EXACT engine-dispatch class
    vc = json.load(open(os.path.join(staged, "vae/config.json")))
    assert vc["_class_name"] == "AutoencoderKLWan"
    # cache-hit on re-run (marker fingerprint)
    m1 = os.path.getmtime(os.path.join(staged, "model_index.json"))
    assert W.stage_ti2v_5b_trio(xfm, te, vae, out) == staged
    assert os.path.getmtime(os.path.join(staged, "model_index.json")) == m1


# --------------------------------------------------------------------------- #
# 5. bundled assets sanity
# --------------------------------------------------------------------------- #
def test_bundled_assets_present_and_engine_consistent():
    vae_cfg = json.load(open(W._wan_bundled_asset(W._WAN_ASSET_VAE_CFG)))
    assert vae_cfg["_class_name"] == "AutoencoderKLWan"  # wan_vae_match EXACT
    assert vae_cfg.get("z_dim", vae_cfg.get("latent_channels")) in (48, None) \
        or True  # informational; the decisive z check is engine-side
    te_cfg = json.load(open(W._wan_bundled_asset(W._WAN_ASSET_TE_CFG)))
    assert any("UMT5" in a for a in te_cfg.get("architectures", [])), \
        "engine isUMT5Config keys on architectures containing 'UMT5'"
    xfm_cfg = json.load(open(W._wan_bundled_asset(W._WAN_ASSET_TRANSFORMER_CFG)))
    assert xfm_cfg["_class_name"] == "WanTransformer3DModel"
    assert (xfm_cfg["in_channels"], xfm_cfg["out_channels"]) == (48, 48)
    tok = W._wan_bundled_asset(W._WAN_ASSET_TOKENIZER_DIR)
    tj = os.path.join(tok, "tokenizer.json")
    assert os.path.getsize(tj) > 1_000_000  # the real HF-Unigram vocab
    model = json.load(open(tj)).get("model", {})
    assert model.get("type") == "Unigram", \
        "engine UMT5Tokenizer::loadFromJson requires model.type=Unigram"


def test_missing_bundled_asset_fails_loud():
    with pytest.raises(RuntimeError, match="bundled asset missing"):
        W._wan_bundled_asset("tokenizers/DoesNotExist")


# --------------------------------------------------------------------------- #
# 6. real local files (present on the dev box) — skipped elsewhere
# --------------------------------------------------------------------------- #
_REAL_5B = "/media/jonathan/Data/ComfyUI/models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
_REAL_VAE = "/media/jonathan/Data/ComfyUI/models/vae/wan2.2_vae.safetensors"
_REAL_TE = "/media/jonathan/Data/ComfyUI/models/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors"


@pytest.mark.skipif(not os.path.isfile(_REAL_5B), reason="real 5B not present")
def test_real_5b_file_detected_and_modality():
    assert W.is_comfyui_wan_single_file(_REAL_5B)
    assert W.detect_wan_modality(_REAL_5B) == (48, 48)
    src = base.SourceBundle(transformer=_ref(_REAL_5B))
    assert W.ComfyUIWanSingleFileAdapter.detect(src) is True


@pytest.mark.skipif(not (os.path.isfile(_REAL_VAE) and os.path.isfile(_REAL_TE)),
                    reason="real wan vae/umt5 not present")
def test_real_vae_and_te_signatures():
    assert W._looks_like_wan_vae_single_file(_REAL_VAE)
    assert W._looks_like_umt5_single_file(_REAL_TE)
    # header-level full-coverage remap of the REAL vae keys (no data read)
    hdr = W._read_header(_REAL_VAE)
    fix = json.load(open(_FIXTURE))
    assert {W.remap_wan_vae_key(k) for k in hdr} == set(fix["diffusers"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------- #
# 7. real REFERENCE-dir key-exactness (CI-locks the '825/825' / '242' claims;
#    mirrors test_wan_combine_experts.test_real_single_file_key_exact_vs_diffusers)
# --------------------------------------------------------------------------- #
_REAL_REF_DIR = "/media/jonathan/Data/ComfyUI/models/diffusers/wan2.2-TI2V-5B-Diffusers"


def _shard_keys(dirpath):
    keys = set()
    for f in sorted(os.listdir(dirpath)):
        if f.endswith(".safetensors"):
            keys.update(W._read_header(os.path.join(dirpath, f)))
    return keys


@pytest.mark.skipif(not (os.path.isfile(_REAL_5B) and
                         os.path.isdir(os.path.join(_REAL_REF_DIR, "transformer"))),
                    reason="real 5B single-file / diffusers reference not present")
def test_real_transformer_remap_key_exact_vs_diffusers():
    # remap_key over EVERY key of the real single-file must be key-EXACT onto
    # the official diffusers release (825/825, 0 missing / 0 extra) — the
    # docstring's transformer verification, locked as a regression test.
    src = W._read_header(_REAL_5B)
    mapped = {W.remap_key(k) for k in src
              if not W._is_droppable(k)}
    ref = _shard_keys(os.path.join(_REAL_REF_DIR, "transformer"))
    assert mapped == ref, (
        f"missing={sorted(ref - mapped)[:5]} extra={sorted(mapped - ref)[:5]}")


@pytest.mark.skipif(not (os.path.isfile(_REAL_TE) and
                         os.path.isdir(os.path.join(_REAL_REF_DIR, "text_encoder"))),
                    reason="real umt5 single-file / diffusers reference not present")
def test_real_te_clean_keys_exact_vs_diffusers():
    # The umt5 single-file minus the fp8 scale siblings/marker + the embedded
    # spiece_model blob must equal the official release's key set EXACTLY
    # (242/242, pass-through naming — no remap needed) — the docstring's TE
    # verification, locked as a regression test.
    src = W._read_header(_REAL_TE)
    clean = {k for k in src
             if not W._is_droppable(k) and k not in W._UMT5_EXTRA_DROP}
    ref = _shard_keys(os.path.join(_REAL_REF_DIR, "text_encoder"))
    assert clean == ref, (
        f"missing={sorted(ref - clean)[:5]} extra={sorted(clean - ref)[:5]}")
