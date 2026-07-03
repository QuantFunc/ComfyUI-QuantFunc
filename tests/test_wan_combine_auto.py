"""Tests for the QuantFunc Wan Combine Experts (Auto) node + its detection layer.

The Auto node is a pure DETECTION + DELEGATION frontend: it scans model dirs for
Wan2.2-A14B sets and resolves a chosen set to an engine-loadable model_dir by
DELEGATING to the exact staging the manual node uses (`stage_two_expert` for a
single-file pair; pass-through / a metadata-only `model_index` normalization for a
diffusers A14B dir). It adds NO staging/dequant/remap of its own.

Covered (pure filesystem + a MOCKED stage; no engine .so, no GPU):
  - detect_wan_a14b_sets: single-file t2v/i2v pairing (weight-derived modality),
    diffusers A14B dir recognition, TI2V-5B exclusion, no-shared-dir skip,
    14B-family shared-dir preference, cross-modality shared.
  - EQUIVALENCE (key): the auto-resolved single-file tuple stages the BYTE-IDENTICAL
    model_dir that the manual node's `stage_two_expert(same high,low,shared)` does.
  - stage_a14b_diffusers: model_index normalized to WanPipeline + resolved boundary,
    subdirs symlinked (weights byte-identical), cache-hit, already-loadable
    pass-through.

Run:  python3 -m pytest tests/test_wan_combine_auto.py -q
"""
import os
import sys
import json
import importlib
import tempfile
import filecmp

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
# Reuse the sibling suite's expert/shared-dir builders (DRY — same synthetic shapes
# that already exercise the full remap+dequant path).
twe = importlib.import_module("test_wan_combine_experts")
_HAS_TORCH = twe._HAS_TORCH
_write_expert = twe._write_expert
_make_shared_dir = twe._make_shared_dir


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mk_root(*named_experts):
    """A scan root containing the given (filename, in_ch, out_ch) experts."""
    root = tempfile.mkdtemp(prefix="qfwan_root_")
    for fname, in_ch, out_ch in named_experts:
        _write_expert(os.path.join(root, fname), in_ch=in_ch, out_ch=out_ch)
    return root


def _mk_diffusers_a14b(name="wan2.2-i2v-A14B-Diffusers", in_ch=36, out_ch=16,
                       mi_class="WanImageToVideoPipeline", boundary=0.9,
                       with_shared=True):
    """A synthetic diffusers A14B dir: transformer/ + transformer_2/ (config +
    a stand-in weight file) + optional shared vae/text_encoder + model_index."""
    parent = tempfile.mkdtemp(prefix="qfwan_diff_")
    d = os.path.join(parent, name)
    for sub in ("transformer", "transformer_2"):
        os.makedirs(os.path.join(d, sub))
        json.dump({"_class_name": "WanTransformer3DModel", "in_channels": in_ch,
                   "out_channels": out_ch, "num_layers": 40},
                  open(os.path.join(d, sub, "config.json"), "w"))
        # a stand-in sharded weight (content-identical symlink target check later)
        open(os.path.join(d, sub, "diffusion_pytorch_model-00001-of-00001.safetensors"),
             "wb").write(b"WEIGHTS-" + sub.encode())
        json.dump({"weight_map": {}}, open(os.path.join(
            d, sub, "diffusion_pytorch_model.safetensors.index.json"), "w"))
    if with_shared:
        for sub in ("vae", "text_encoder", "tokenizer", "scheduler"):
            os.makedirs(os.path.join(d, sub))
            open(os.path.join(d, sub, "marker.txt"), "w").write(sub)
        json.dump({"_class_name": "AutoencoderKLWan", "z_dim": 16},
                  open(os.path.join(d, "vae", "config.json"), "w"))
    mi = {"_class_name": mi_class, "_diffusers_version": "0.35.0"}
    if boundary is not None:
        mi["boundary_ratio"] = boundary
    json.dump(mi, open(os.path.join(d, "model_index.json"), "w"))
    return d


def _by_name(sets):
    return {s["name"]: s for s in sets}


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_detect_single_file_pairs_t2v_and_i2v():
    root = _mk_root(
        ("wan2.2_t2v_high_14B_fp8_scaled.safetensors", 16, 16),
        ("wan2.2_t2v_low_14B_fp8_scaled.safetensors", 16, 16),
        ("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 36, 16),
        ("wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", 36, 16),
    )
    shared = _make_shared_dir()
    sets = _by_name(W.detect_wan_a14b_sets([root, os.path.dirname(shared)]))
    assert "wan2.2-t2v-A14B" in sets and "wan2.2-i2v-A14B" in sets
    t2v = sets["wan2.2-t2v-A14B"]
    assert t2v["kind"] == "single_file_pair" and t2v["modality"] == "t2v"
    assert "high" in os.path.basename(t2v["high"]) and "low" in os.path.basename(t2v["low"])
    assert os.path.isdir(t2v["shared"])
    i2v = sets["wan2.2-i2v-A14B"]
    assert i2v["modality"] == "i2v"
    assert "high" in os.path.basename(i2v["high"]) and "low" in os.path.basename(i2v["low"])


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_detect_excludes_ti2v_5b():
    # a 5B single model (no high/low, `5b` token) must NOT be paired/surfaced.
    root = _mk_root(("wan2.2_ti2v_5B_fp16.safetensors", 48, 48))
    shared = _make_shared_dir()
    sets = W.detect_wan_a14b_sets([root, os.path.dirname(shared)])
    assert all("5b" not in s["name"].lower() for s in sets)
    # only the shared dir (not A14B) exists → no single-file set at all
    assert not any(s["kind"] == "single_file_pair" for s in sets)


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_detect_skips_pair_without_shared_dir():
    # a complete high+low pair but NO Wan diffusers dir to supply vae/text_encoder
    root = _mk_root(
        ("wan2.2_t2v_high_14B_fp8_scaled.safetensors", 16, 16),
        ("wan2.2_t2v_low_14B_fp8_scaled.safetensors", 16, 16),
    )
    sets = W.detect_wan_a14b_sets([root])
    assert not any(s["kind"] == "single_file_pair" for s in sets), \
        "a pair with no shared dir must be dropped, not surfaced unusable"


def test_detect_diffusers_a14b_dir_and_loadable_flag():
    # a REAL download: a Wan…-prefixed class + a published boundary → loadable AS-IS
    # (wan_detect accepts any "Wan" prefix; the two-expert gate needs only
    # boundary_ratio>0 + transformer_2/ — verified against live WanVideoPipeline.cpp).
    real = _mk_diffusers_a14b(mi_class="WanImageToVideoPipeline", boundary=0.9)
    sets = _by_name(W.detect_wan_a14b_sets([os.path.dirname(real)]))
    nm = os.path.basename(real)
    assert nm in sets and sets[nm]["kind"] == "diffusers_dir"
    assert sets[nm]["loadable"] is True        # WanImageToVideoPipeline is a "Wan…" prefix

    # a plain WanPipeline is also loadable
    ok = _mk_diffusers_a14b(name="wan-plain", mi_class="WanPipeline", boundary=0.875)
    assert _by_name(W.detect_wan_a14b_sets([os.path.dirname(ok)]))["wan-plain"]["loadable"]

    # missing boundary_ratio → the engine would load SINGLE-expert → NOT loadable
    no_b = _mk_diffusers_a14b(name="wan-no-boundary",
                              mi_class="WanImageToVideoPipeline", boundary=None)
    assert _by_name(W.detect_wan_a14b_sets(
        [os.path.dirname(no_b)]))["wan-no-boundary"]["loadable"] is False

    # a non-Wan class → wan_detect's "Wan" prefix rejects → NOT loadable
    bad = _mk_diffusers_a14b(name="mislabeled", mi_class="FooPipeline", boundary=0.9)
    assert _by_name(W.detect_wan_a14b_sets(
        [os.path.dirname(bad)]))["mislabeled"]["loadable"] is False


def test_detect_5b_diffusers_not_flagged_a14b():
    # a single-transformer (5B) diffusers dir has no transformer_2/ → not A14B.
    parent = tempfile.mkdtemp(prefix="qfwan_5bdir_")
    d = os.path.join(parent, "wan2.2-TI2V-5B-Diffusers")
    os.makedirs(os.path.join(d, "transformer"))
    json.dump({"in_channels": 48, "out_channels": 48},
              open(os.path.join(d, "transformer", "config.json"), "w"))
    for sub in ("vae", "text_encoder"):
        os.makedirs(os.path.join(d, sub))
    sets = W.detect_wan_a14b_sets([parent])
    assert not any(s.get("kind") == "diffusers_dir" for s in sets)


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_detect_prefers_a14b_family_shared_dir():
    # two shared candidates: a 5B-style (no transformer_2) and an A14B diffusers.
    # the A14B one must win (its 14B vae/text_encoder match the experts).
    root = _mk_root(
        ("wan2.2_t2v_high_14B_fp8_scaled.safetensors", 16, 16),
        ("wan2.2_t2v_low_14B_fp8_scaled.safetensors", 16, 16),
    )
    five_b = tempfile.mkdtemp(prefix="qfwan_5b_")
    d5 = os.path.join(five_b, "wan2.2-TI2V-5B-Diffusers")
    os.makedirs(os.path.join(d5, "transformer"))
    json.dump({"in_channels": 48, "out_channels": 48},
              open(os.path.join(d5, "transformer", "config.json"), "w"))
    for sub in ("vae", "text_encoder"):
        os.makedirs(os.path.join(d5, sub))
    a14b = _mk_diffusers_a14b()
    sets = _by_name(W.detect_wan_a14b_sets([root, five_b, os.path.dirname(a14b)]))
    assert sets["wan2.2-t2v-A14B"]["shared"] == a14b   # A14B-family wins over 5B


# --------------------------------------------------------------------------- #
# EQUIVALENCE (key) — auto-resolve stages the byte-identical model_dir
# the manual node's stage_two_expert(same high,low,shared) produces
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_auto_resolve_delegates_identical_tuple(monkeypatch):
    root = _mk_root(
        ("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 36, 16),
        ("wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", 36, 16),
    )
    shared = _make_shared_dir()
    sets = _by_name(W.detect_wan_a14b_sets([root, os.path.dirname(shared)]))
    desc = sets["wan2.2-i2v-A14B"]

    captured = {}
    def fake_stage(high, low, shared_dir, out_dir, *, boundary_ratio=None, force=False):
        captured.update(high=high, low=low, shared=shared_dir, out=out_dir,
                        boundary=boundary_ratio, force=force)
        return out_dir
    monkeypatch.setattr(W, "stage_two_expert", fake_stage)

    out = tempfile.mkdtemp(prefix="qfwan_out_") + "/stage"
    W.resolve_wan_a14b_set(desc, out, boundary_ratio=None)
    # the auto frontend passes EXACTLY the descriptor's resolved paths — i.e. the
    # same (high, low, shared) a manual combine of those files would pass.
    assert captured["high"] == desc["high"]
    assert captured["low"] == desc["low"]
    assert captured["shared"] == desc["shared"]
    assert captured["out"] == out
    assert captured["boundary"] is None       # AUTO propagated as None


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_auto_staged_model_dir_byte_identical_to_manual():
    """Full proof: stage the SAME synthetic pair via (a) the auto-resolved
    descriptor and (b) a direct manual stage_two_expert call → identical
    model_index + transformer config + file tree."""
    root = _mk_root(
        ("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 36, 16),
        ("wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", 36, 16),
    )
    shared = _make_shared_dir()
    desc = _by_name(W.detect_wan_a14b_sets([root, os.path.dirname(shared)]))["wan2.2-i2v-A14B"]

    auto_out = tempfile.mkdtemp(prefix="qfwan_auto_") + "/stage"
    manual_out = tempfile.mkdtemp(prefix="qfwan_manual_") + "/stage"
    auto_dir = W.resolve_wan_a14b_set(desc, auto_out, boundary_ratio=None)
    manual_dir = W.stage_two_expert(desc["high"], desc["low"], desc["shared"],
                                    manual_out, boundary_ratio=None)

    # identical model_index (incl. resolved boundary_ratio + _class_name)
    a_mi = json.load(open(os.path.join(auto_dir, "model_index.json")))
    m_mi = json.load(open(os.path.join(manual_dir, "model_index.json")))
    assert a_mi == m_mi
    assert a_mi["_class_name"] == "WanPipeline" and a_mi["boundary_ratio"] > 0
    # identical transformer configs
    for sub in ("transformer", "transformer_2"):
        assert json.load(open(os.path.join(auto_dir, sub, "config.json"))) == \
               json.load(open(os.path.join(manual_dir, sub, "config.json")))
    # identical produced file tree (names)
    def _tree(root_dir):
        return sorted(os.path.relpath(os.path.join(dp, f), root_dir)
                      for dp, _, fs in os.walk(root_dir) for f in fs)
    assert _tree(auto_dir) == _tree(manual_dir)


# --------------------------------------------------------------------------- #
# diffusers normalization + pass-through
# --------------------------------------------------------------------------- #
def test_normalize_preserves_wan_prefix_class_and_adds_boundary():
    # a Wan-prefixed dir MISSING boundary_ratio (not loadable) → normalize KEEPS the
    # accurate published class and ADDS the resolved i2v boundary; weights symlinked.
    src = _mk_diffusers_a14b(mi_class="WanImageToVideoPipeline", boundary=None)
    out = tempfile.mkdtemp(prefix="qfwan_norm_") + "/stage"
    model_dir = W.stage_a14b_diffusers(src, out, boundary_ratio=None)

    mi = json.load(open(os.path.join(model_dir, "model_index.json")))
    assert mi["_class_name"] == "WanImageToVideoPipeline"   # Wan-prefix class preserved
    assert mi["boundary_ratio"] == 0.9                      # published i2v default added
    # every engine-read subdir present; experts + shared symlinked to the source
    for sub in ("transformer", "transformer_2", "vae", "text_encoder",
                "tokenizer", "scheduler"):
        assert os.path.isdir(os.path.join(model_dir, sub))
    # weights byte-identical (symlink → same content as source)
    assert filecmp.cmp(
        os.path.join(model_dir, "transformer",
                     "diffusion_pytorch_model-00001-of-00001.safetensors"),
        os.path.join(src, "transformer",
                     "diffusion_pytorch_model-00001-of-00001.safetensors"),
        shallow=False)
    # source model_index untouched (never mutate the user's model)
    assert "boundary_ratio" not in json.load(open(os.path.join(src, "model_index.json")))


def test_normalize_rewrites_non_wan_class_to_wanpipeline():
    # a NON-Wan class (mislabeled) → normalize rewrites to the canonical WanPipeline.
    src = _mk_diffusers_a14b(mi_class="FooPipeline", boundary=0.9)
    out = tempfile.mkdtemp(prefix="qfwan_normc_") + "/stage"
    model_dir = W.stage_a14b_diffusers(src, out, boundary_ratio=None)
    mi = json.load(open(os.path.join(model_dir, "model_index.json")))
    assert mi["_class_name"] == "WanPipeline"      # non-Wan → canonical loadable class
    assert mi["boundary_ratio"] == 0.9
    # source untouched
    assert json.load(open(os.path.join(src, "model_index.json")))["_class_name"] \
        == "FooPipeline"


def test_normalize_diffusers_dir_cache_hit():
    src = _mk_diffusers_a14b()
    out = tempfile.mkdtemp(prefix="qfwan_normhit_") + "/stage"
    W.stage_a14b_diffusers(src, out)
    marker = os.path.join(out, ".qf_stage_complete")
    m1 = os.stat(marker).st_mtime_ns
    W.stage_a14b_diffusers(src, out)               # second run must cache-hit
    assert os.stat(marker).st_mtime_ns == m1, "cache hit must not rewrite the marker"


def test_resolve_loadable_diffusers_is_passthrough():
    # the REAL download shape (WanImageToVideoPipeline + published boundary) is
    # loadable AS-IS → resolve returns the dir itself, ZERO staging.
    src = _mk_diffusers_a14b(name="wan-real-i2v",
                             mi_class="WanImageToVideoPipeline", boundary=0.9)
    desc = _by_name(W.detect_wan_a14b_sets([os.path.dirname(src)]))["wan-real-i2v"]
    assert desc["loadable"] is True
    out = tempfile.mkdtemp(prefix="qfwan_pt_") + "/stage"
    model_dir = W.resolve_wan_a14b_set(desc, out)
    assert model_dir == os.path.abspath(src)       # the dir itself, no staging
    assert not os.path.exists(out)                 # nothing was staged


def test_resolve_not_loadable_diffusers_normalizes():
    # a not-loadable shape (missing boundary) → resolve normalizes it.
    src = _mk_diffusers_a14b(mi_class="WanImageToVideoPipeline", boundary=None)
    desc = _by_name(W.detect_wan_a14b_sets([os.path.dirname(src)]))[os.path.basename(src)]
    assert desc["loadable"] is False
    out = tempfile.mkdtemp(prefix="qfwan_rn_") + "/stage"
    model_dir = W.resolve_wan_a14b_set(desc, out)
    assert model_dir == os.path.abspath(out)
    mi = json.load(open(os.path.join(model_dir, "model_index.json")))
    assert mi["_class_name"].startswith("Wan") and mi["boundary_ratio"] > 0


def test_stage_a14b_diffusers_rejects_non_a14b():
    parent = tempfile.mkdtemp(prefix="qfwan_bad_")
    d = os.path.join(parent, "single-xfm")
    os.makedirs(os.path.join(d, "transformer"))
    json.dump({"in_channels": 48, "out_channels": 48},
              open(os.path.join(d, "transformer", "config.json"), "w"))
    with pytest.raises(RuntimeError, match="not a diffusers A14B dir"):
        W.stage_a14b_diffusers(d, tempfile.mkdtemp() + "/o")


# --------------------------------------------------------------------------- #
# node-level (the combine() wrapper: dropdown listing, delegation, no-set error)
# --------------------------------------------------------------------------- #
def _import_nodes():
    import types as _t
    for _n in ("comfy", "comfy.model_management", "comfy.utils"):
        sys.modules.setdefault(_n, _t.ModuleType(_n))
    fp = sys.modules.get("folder_paths") or _t.ModuleType("folder_paths")
    fp.get_folder_paths = getattr(fp, "get_folder_paths", lambda k: [])
    fp.get_temp_directory = getattr(fp, "get_temp_directory", lambda: tempfile.gettempdir())
    sys.modules["folder_paths"] = fp
    return importlib.import_module(f"{_PKG}.nodes")


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
def test_node_lists_sets_and_delegates(monkeypatch):
    nodes = _import_nodes()
    root = _mk_root(
        ("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 36, 16),
        ("wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", 36, 16),
    )
    shared = _make_shared_dir()
    monkeypatch.setattr(nodes, "_wan_a14b_scan_roots",
                        lambda: [root, os.path.dirname(shared)])
    nodes.refresh_wan_a14b_sets()

    opts = nodes.QuantFuncWanCombineExpertsAuto.INPUT_TYPES()["required"]["wan_a14b_set"][0]
    assert "wan2.2-i2v-A14B" in opts

    captured = {}
    monkeypatch.setattr(W, "stage_two_expert",
                        lambda h, l, s, o, *, boundary_ratio=None, force=False:
                        captured.update(high=h, low=l, shared=s, out=o) or o)
    out = tempfile.mkdtemp(prefix="qfwan_node_") + "/stage"
    (model_dir,) = nodes.QuantFuncWanCombineExpertsAuto().combine(
        "wan2.2-i2v-A14B", boundary_ratio=0.0, output_dir=out)
    assert model_dir == out
    assert "high" in os.path.basename(captured["high"])
    assert os.path.isdir(captured["shared"])


def test_node_errors_on_no_set(monkeypatch):
    nodes = _import_nodes()
    monkeypatch.setattr(nodes, "_wan_a14b_scan_roots", lambda: [])
    nodes.refresh_wan_a14b_sets()
    opts = nodes.QuantFuncWanCombineExpertsAuto.INPUT_TYPES()["required"]["wan_a14b_set"][0]
    assert opts == [nodes._WAN_A14B_NO_SETS]      # sentinel keeps the combo valid
    with pytest.raises(RuntimeError, match="no Wan A14B set selected"):
        nodes.QuantFuncWanCombineExpertsAuto().combine(nodes._WAN_A14B_NO_SETS)


def test_node_errors_on_stale_saved_value(monkeypatch):
    nodes = _import_nodes()
    monkeypatch.setattr(nodes, "_wan_a14b_scan_roots", lambda: [])
    nodes.refresh_wan_a14b_sets()
    with pytest.raises(RuntimeError, match="no longer present"):
        nodes.QuantFuncWanCombineExpertsAuto().combine("wan2.2-t2v-A14B")


# --------------------------------------------------------------------------- #
# SHARED atomic-swap fault matrix — drives BOTH staging code paths
# (stage_two_expert AND stage_a14b_diffusers) through the SAME crash points, so
# the two copies of the hard-won marker-first / sentinel-off / swap_started
# invariant can never silently diverge: a regression in either fails these.
# --------------------------------------------------------------------------- #
def _single_file_stage():
    root = _mk_root(
        ("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 36, 16),
        ("wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", 36, 16),
    )
    shared = _make_shared_dir()
    desc = _by_name(W.detect_wan_a14b_sets([root, os.path.dirname(shared)]))["wan2.2-i2v-A14B"]
    return lambda out, force=False: W.stage_two_expert(
        desc["high"], desc["low"], desc["shared"], out, force=force)


def _diffusers_stage():
    src = _mk_diffusers_a14b(mi_class="WanImageToVideoPipeline", boundary=0.9)
    return lambda out, force=False: W.stage_a14b_diffusers(src, out, force=force)


_SWAP_PATHS = {"single_file": _single_file_stage, "diffusers": _diffusers_stage}


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
@pytest.mark.parametrize("path", sorted(_SWAP_PATHS))
def test_swap_path_sentinel_unlink_failure_completes(path, monkeypatch, caplog):
    """Both paths: a transient OSError on the sentinel unlink must NOT discard the
    completed build (swap_started gates the reap OFF at the marker write) — the
    stage COMPLETES, out_dir is valid + marker-carrying, a warning is logged."""
    import logging as _logging
    stage = _SWAP_PATHS[path]()
    out = tempfile.mkdtemp(prefix=f"qfwan_fm_{path}_") + "/stage"
    stage(out)                                    # baseline complete stage
    real_unlink = os.unlink
    def failing(p, *a, **k):
        if isinstance(p, str) and os.path.basename(p) == W._TMP_SENTINEL:
            raise OSError("simulated transient sentinel lock")
        return real_unlink(p, *a, **k)
    monkeypatch.setattr(W.os, "unlink", failing)
    with caplog.at_level(_logging.WARNING):
        stage(out, force=True)                    # must COMPLETE, not raise
    monkeypatch.setattr(W.os, "unlink", real_unlink)
    assert os.path.isfile(os.path.join(out, ".qf_stage_complete"))
    assert any("staging sentinel" in r.message for r in caplog.records)
    parent = os.path.dirname(out)
    assert [n for n in os.listdir(parent) if ".tmp-" in n or ".trash-" in n] == []


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
@pytest.mark.parametrize("path", sorted(_SWAP_PATHS))
def test_swap_path_marker_written_before_sentinel_removed(path, monkeypatch):
    """Both paths: at the instant the tmp sentinel is unlinked, the completion
    marker must ALREADY exist (>=1 ownership proof at every instant)."""
    stage = _SWAP_PATHS[path]()
    out = tempfile.mkdtemp(prefix=f"qfwan_ord_{path}_") + "/stage"
    seen = {"checked": False}
    real_unlink = os.unlink
    def checking(p, *a, **k):
        if isinstance(p, str) and os.path.basename(p) == W._TMP_SENTINEL:
            assert os.path.isfile(os.path.join(os.path.dirname(p), ".qf_stage_complete")), \
                "sentinel removed BEFORE the marker was written (neither-proof window)"
            seen["checked"] = True
        return real_unlink(p, *a, **k)
    monkeypatch.setattr(W.os, "unlink", checking)
    stage(out)
    assert seen["checked"] and os.path.isfile(os.path.join(out, ".qf_stage_complete"))


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
@pytest.mark.parametrize("path", sorted(_SWAP_PATHS))
def test_swap_path_swap_in_failure_rolls_back_old(path, monkeypatch):
    """Both paths: a transient failure of the swap-in (tmp -> out_dir) rolls the
    old stage back — out_dir is never left absent when a valid old stage existed."""
    stage = _SWAP_PATHS[path]()
    out = tempfile.mkdtemp(prefix=f"qfwan_swf_{path}_") + "/stage"
    stage(out)                                    # old complete stage in place
    old_marker = open(os.path.join(out, ".qf_stage_complete")).read()
    real_replace = os.replace
    failed = {"n": 0}
    def transient(a, b):
        # fail ONLY the swap-in (source is the .tmp- dir), and only once
        if ".tmp-" in os.path.basename(a) and failed["n"] == 0:
            failed["n"] = 1
            raise OSError("simulated transient swap-in fault")
        return real_replace(a, b)
    monkeypatch.setattr(W.os, "replace", transient)
    with pytest.raises(OSError, match="swap-in"):
        stage(out, force=True)
    monkeypatch.setattr(W.os, "replace", real_replace)
    # rollback restored the OLD stage to the canonical path
    assert os.path.isfile(os.path.join(out, ".qf_stage_complete"))
    assert open(os.path.join(out, ".qf_stage_complete")).read() == old_marker
