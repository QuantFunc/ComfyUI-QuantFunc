#!/usr/bin/env python3
"""Behavioural tests for the FILE-BASED loader node (INT8-Fast-aligned redesign) + the
liveness/LoRA substrate:

  1. UI surface — transformer1/transformer2 FILE dropdowns + model_config (official presets,
     data-driven from configs/), nothing else
  2. dispatch by the preset MANIFEST's family; ltx2/minimax-h3 FILE-MODE staging shape;
     traversal/shape-mismatch presets refused
  3. wan dual-expert STAGING — configs copied from the shipped bundle, weights SYMLINKED,
     denoise_only in the create cfg; single-file wan refused (A14B is dual-expert)
  4. transformer-name containment (comfy's own get_full_path_or_raise normalization)
  5. adopt_comfy_state_from — an upstream ModelSampling patch must SURVIVE a LoRA rebuild
  6. HOST-RAM accounting honesty — a never-created engine must report 0, not its estimate
  7. SHARED-handle sibling safety — a release under a live sibling must refuse (measured UAF)
  8. the REAL LoRA-node chain — two-instance state transplant + N-chain -> ONE deferred create
  9. the passive SWEEP path — spares a handle while ANY consumer lives, reclaims when all die

Run:  python tests/loader_dispatch_test.py          (needs the ComfyUI env; SKIPs without it)
"""
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.dirname(_HERE)


def _emit_skip(reason):
    print(f"LOADER_DISPATCH: SKIP — {reason}")
    return 0


def _load_plugin():
    """Import the plugin package the way ComfyUI does (by file path), with a real comfy on sys.path."""
    comfy_root = os.environ.get("COMFY_ROOT")
    if not comfy_root:
        guess = os.path.dirname(os.path.dirname(_PLUGIN))
        comfy_root = guess if os.path.isfile(os.path.join(guess, "folder_paths.py")) else None
    if not comfy_root:
        return None, "ComfyUI root not found (set COMFY_ROOT)"
    sys.path.insert(0, comfy_root)
    try:
        import importlib.util
        import folder_paths  # noqa: F401 — proves the env is real
        spec = importlib.util.spec_from_file_location("qfn_test_pkg",
                                                      os.path.join(_PLUGIN, "__init__.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["qfn_test_pkg"] = mod
        spec.loader.exec_module(mod)
        return mod, None
    except Exception as exc:  # noqa: BLE001
        return None, f"plugin import failed: {type(exc).__name__}: {exc}"


class _DummyEngine:
    """Stands in for a created QFEngineHandle — no CUDA, no model files."""
    lib = "LIB"
    footprint_bytes = 1234 * 1024 * 1024
    step_count = 0
    sampler_step_count = 0

    def __init__(self):
        self.pipeline = object()
        self.current_session = None
        self.unloaded = False

    def end_session_if_open(self):
        return (False, True)

    def unload_vram(self):
        self.unloaded = True
        return self.footprint_bytes

    def destroy(self):
        self.pipeline = None
        self.current_session = None


def main():
    qfn, why = _load_plugin()
    if qfn is None:
        return _emit_skip(why)
    if not getattr(qfn, "_IMPORT_OK", False):
        return _emit_skip("plugin imported but its comfy imports failed (_IMPORT_OK False)")
    import folder_paths

    bad = 0

    def check(label, ok, detail=""):
        nonlocal bad
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}{(' ' + detail) if detail else ''}")
        if not ok:
            bad += 1

    # fixture transformer FILES in a scratch diffusion_models root (registered with comfy itself —
    # the loader lists/resolves through folder_paths, exactly like production)
    tmp = tempfile.mkdtemp(prefix="qf_loader_test_")
    dm = os.path.join(tmp, "diffusion_models")
    os.makedirs(dm)
    for f in ("fx-t2v-4steps-high-quantfunc-int4.safetensors", "fx-t2v-4steps-low-quantfunc-int4.safetensors", "other.safetensors",
              "not-a-model.txt"):
        with open(os.path.join(dm, f), "wb") as fh:
            fh.write(b"\0" * 16)
    folder_paths.add_model_folder_path("diffusion_models", dm)
    # ltx2/h3 file-mode fixtures: extra weight surfaces the LTX loader resolves through
    # (text_encoders: the gemma with-proj file; vae: the ltx-2.5 audio vae) + single-file
    # transformer fixtures matching the shipped presets' file_hints.
    te_root = os.path.join(tmp, "text_encoders"); os.makedirs(te_root)
    vae_root = os.path.join(tmp, "vae"); os.makedirs(vae_root)
    for root, f in ((dm, "fx-ltx-2.5-quantfunc-4bit.safetensors"),
                    (dm, "fx-minimax-h3-quantfunc-int4.safetensors"),
                    (te_root, "fx-gemma4-with-proj.safetensors"),
                    (vae_root, "fx-ltx25-audio-vae.safetensors")):
        with open(os.path.join(root, f), "wb") as fh:
            fh.write(b"\0" * 16)
    # a VALID-header connectors fixture (the ltx2 build does a real safetensors header read to
    # detect embeddings_connector keys — a 16-byte dummy reads as keyless):
    import struct as _st
    def _mini_st(path, *keys):
        hdr = json.dumps({k: {"dtype": "F32", "shape": [1],
                              "data_offsets": [4 * i, 4 * i + 4]}
                          for i, k in enumerate(keys)}).encode()
        pad = (8 - (len(hdr) % 8)) % 8; hdr += b" " * pad
        with open(path, "wb") as fh:
            fh.write(_st.pack("<Q", len(hdr))); fh.write(hdr); fh.write(b"\0" * (4 * len(keys)))
    # connectors fixtures carry BOTH modality connector blocks (like the real file):
    # the AUDIO one is the wan-align AV discriminant (audio_vae staging no longer required).
    _mini_st(os.path.join(dm, "fx-ltx25-connectors.safetensors"),
             "model.diffusion_model.video_embeddings_connector.learnable_registers",
             "model.diffusion_model.audio_embeddings_connector.learnable_registers")
    # conf-1: the AV discriminant now mirrors the ENGINE's (audio_vae weights AND a
    # vocoder — 2.5 bundles vocoder.* inside the audio_vae file), so the audio-vae
    # fixture must carry a vocoder key to arm the AV path.
    _mini_st(os.path.join(vae_root, "fx-ltx25-audio-vae.safetensors"),
             "vocoder.bwe_generator.conv_pre.weight")
    # [aux-auto] manifest-named aux fixtures: the ltx2-2.5-22b preset's aux_files resolve
    # against these (audio_vae must carry a vocoder key for the AV discriminant; connectors
    # must carry a connector key for the explicit-source validation).
    _mini_st(os.path.join(vae_root, "ltx-2.5-audio-vae-bf16.safetensors"),
             "vocoder.bwe_generator.conv_pre.weight")
    _mini_st(os.path.join(vae_root, "ltx-2.5-video-vae-bf16.safetensors"),
             "decoder.conv_in.weight")
    _mini_st(os.path.join(te_root, "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
             "text_embedding_projection.video_aggregate_embed.weight")
    _mini_st(os.path.join(dm, "ltx25-connectors-bf16.safetensors"),
             "model.diffusion_model.video_embeddings_connector.learnable_registers",
             "model.diffusion_model.audio_embeddings_connector.learnable_registers")
    folder_paths.add_model_folder_path("text_encoders", te_root)
    folder_paths.add_model_folder_path("vae", vae_root)
    # an ISOLATED diffusion_models dir with NO connectors sibling — the auto-probe fallback
    # scans the transformer's OWN dir, so the loud-refusal arm needs a keyless transformer
    # with no probe candidates next to it.
    dm2 = os.path.join(tmp, "diffusion_models_iso"); os.makedirs(dm2)
    with open(os.path.join(dm2, "fx-iso-ltx-2.5-quantfunc-4bit.safetensors"), "wb") as fh:
        fh.write(b"\0" * 16)
    folder_paths.add_model_folder_path("diffusion_models", dm2)

    # OFFICIAL-CONFIG presets under a FIXTURE configs dir: the real shipped wan preset copied in,
    # plus synthetic manifests that exercise routing (ltx2/h3 not-wired, unknown family,
    # single-expert shape) without shipping fake production presets.
    import shutil
    cfgroot = os.path.join(tmp, "configs")
    shutil.copytree(os.path.join(_PLUGIN, "configs", "wan2.2-a14b-t2v"),
                    os.path.join(cfgroot, "wan2.2-a14b-t2v"))
    shutil.copytree(os.path.join(_PLUGIN, "configs", "wan2.2-a14b-i2v"),
                    os.path.join(cfgroot, "wan2.2-a14b-i2v"))
    shutil.copytree(os.path.join(_PLUGIN, "configs", "ltx2-2.5-22b"),
                    os.path.join(cfgroot, "ltx2-2.5-22b"))
    shutil.copytree(os.path.join(_PLUGIN, "configs", "minimax-h3-fl2va"),
                    os.path.join(cfgroot, "minimax-h3-fl2va"))
    for name, mf in (("fx-ltx", {"family": "ltx2"}),
                     ("fx-h3", {"family": "minimax-h3"}),
                     ("fx-alien", {"family": "no-such-family"}),
                     ("fx-single", {"family": "wan", "dual_expert": False})):
        os.makedirs(os.path.join(cfgroot, name))
        json.dump(mf, open(os.path.join(cfgroot, name, "qf_native.json"), "w"))
    qfn._CONFIGS_DIR = cfgroot

    creates = []
    fake_cache = {}

    def fake_get_engine(model_dir, create_cfg=None, device_idx=0):
        # Mirrors the REAL _get_engine contract the liveness layer is built against: ONE handle
        # per (model_dir, device, cfg) key, REUSED while its pipeline is valid.
        ck = (model_dir, int(device_idx), json.dumps(create_cfg or {}, sort_keys=True))
        eng = fake_cache.get(ck)
        if eng is not None and eng.pipeline is not None:
            return eng, ck
        creates.append(dict(create_cfg or {}))
        eng = _DummyEngine()
        fake_cache[ck] = eng
        return eng, ck
    qfn._get_engine = fake_get_engine
    # Builders CAPTURE deps at registration — re-register so they hold the stub.
    qfn._FAMILY_BUILDERS.clear()
    qfn._FAMILY_MATCHERS.clear()
    qfn._register_families()

    WanL = qfn.NODE_CLASS_MAPPINGS["QuantFuncWanLoader"]()
    LtxL = qfn.NODE_CLASS_MAPPINGS["QuantFuncLTXLoader"]()
    H3L = qfn.NODE_CLASS_MAPPINGS["QuantFuncH3Loader"]()

    # ── 1) UI surface (per-family pivot): wan = dual required transformers + DUAL MODEL outputs;
    #      single-expert nodes = one transformer; preset dropdowns are FAMILY-FILTERED ──
    it = WanL.INPUT_TYPES()
    check("wan required = transformer1/transformer2/model_config",
          list(it["required"].keys()) == ["transformer1", "transformer2", "model_config"],
          f"-> {list(it['required'].keys())}")
    check("wan has NO optional block (transformer2 is required)", not it.get("optional"))
    check("wan RETURN = two MODELs named high/low",
          WanL.RETURN_TYPES == ("MODEL", "MODEL")
          and WanL.RETURN_NAMES == ("model_high", "model_low"))
    cfgs = it["required"]["model_config"][0]
    check("wan model_config lists ONLY wan presets (family-filtered)",
          "wan2.2-a14b-t2v" in cfgs and "fx-single" in cfgs and "fx-ltx" not in cfgs
          and "fx-alien" not in cfgs, f"-> {cfgs}")
    check("the i2v preset ships and lists in the wan dropdown", "wan2.2-a14b-i2v" in cfgs,
          f"-> {cfgs}")
    ltx_cfgs = LtxL.INPUT_TYPES()["required"]["model_config"][0]
    check("ltx model_config lists ONLY ltx2 presets",
          "fx-ltx" in ltx_cfgs and "wan2.2-a14b-t2v" not in ltx_cfgs, f"-> {ltx_cfgs}")
    h3_cfgs = H3L.INPUT_TYPES()["required"]["model_config"][0]
    check("h3 model_config lists ONLY minimax-h3 presets",
          "fx-h3" in h3_cfgs and "wan2.2-a14b-t2v" not in h3_cfgs, f"-> {h3_cfgs}")
    t1 = it["required"]["transformer1"][0]
    check("transformer1 lists .safetensors FILES (and only those)",
          "fx-t2v-4steps-high-quantfunc-int4.safetensors" in t1 and "not-a-model.txt" not in t1)

    # ── 2) dispatch: family-node guard + ltx2/minimax-h3 FILE-MODE staging (wired now —
    #      the old not-wired refusal arms flipped into layout-shape arms);
    #      cross-family preset refused; model_config traversal refused ──
    # ltx2 file-mode: te_file REQUIRED; with audio_vae -> AV staging (same-target xfm links
    # into transformer/ AND connectors/ [#565 comfy25 branch], te + audio_vae links, cfg
    # carries denoise_only). The synthetic fx-ltx manifest (no file_hints) keeps exercising
    # bare routing; the REAL preset exercises the full staged shape.
    # ── CONNECTORS-SOURCE CONTRACT (user 2026-08-31, supersedes the 2026-08-22
    # one-file ruling "引擎层不应该依赖这个") ─────────────────────────────────────
    # A transformer-only export RESOLVES via a same-dir QUALIFYING completion sibling
    # (content-probed: BOTH modality connector prefixes — the joint-AV discriminant);
    # it refuses loud ONLY when no qualifying source exists. text projections are no
    # longer required (the with-proj clip applies them TE-side; the engine loads them
    # opportunistically with a te-dir fallback).
    _out_split = None
    try:
        _out_split = LtxL.load("fx-ltx-2.5-quantfunc-4bit.safetensors", "ltx2-2.5-22b")[0]
    except RuntimeError as e:
        check("ltx2 transformer-only resolves via the sibling completion file", False,
              f"-> unexpected refusal: {str(e)[:90]}")
    if _out_split is not None:
        _ = _out_split.model._qf.lib   # first touch materializes the staged pkg key
        _smd = _out_split.model._qf._ckey[0]
        _ssrc = os.path.realpath(os.path.join(_smd, "connectors", "model.safetensors"))
        check("ltx2 transformer-only resolves via the sibling completion file",
              _ssrc.endswith("fx-ltx25-connectors.safetensors"),
              f"-> staged connectors {_ssrc[-48:]}")
    # loud-refusal arm (the iso fixture built for exactly this): NO probe candidates
    # next to the transformer -> the new message names the probed dir, the
    # BOTH-modality requirement, and both remedies.
    try:
        LtxL.load("fx-iso-ltx-2.5-quantfunc-4bit.safetensors", "ltx2-2.5-22b")
        check("ltx2 transformer-only with NO completion source refuses loud",
              False, "-> no exception")
    except RuntimeError as e:
        check("ltx2 transformer-only with NO completion source refuses loud",
              "transformer-only" in str(e) and "connectors completion" in str(e)
              and "allin" in str(e) and "audio_embeddings_connector" in str(e),
              f"-> {str(e)[:90]}")
    # node surface: required = latent-only duo (transformer + model_config; the manual
    # block-count widget was REMOVED 2026-09-12 — residency is arena-managed); optional =
    # the runtime SESSION dials (attention_backend 2026-08-27; sol_tau; quality_enhance
    # [video_enhance boolean switch 2026-09-19; was the token-prune float mapper 2026-09-12]; step_cache +
    # block_cache). Every optional is a session knob (no create key / no rebuild) — sparse is
    # deliberately NOT among them (removed 2026-08-29, only the caches came back).
    _lit = LtxL.INPUT_TYPES()
    check("ltx node surface = latent-duo required + session-dial optionals (no sparse)",
          list(_lit["required"].keys()) == ["transformer", "model_config"]
          and list(_lit.get("optional", {}).keys()) == ["attention_backend", "sol_tau", "quality_enhance",
                                                        "step_cache", "block_cache"]
          and "sparse" not in _lit.get("optional", {}),
          f"-> req={list(_lit['required'].keys())} opt={list(_lit.get('optional', {}).keys())}")
    # h3 node surface (same latent-duo + session dials shape as ltx; block-count removed 2026-09)
    _h3it = H3L.INPUT_TYPES()
    check("h3 node surface = latent-duo required + session-dial optionals",
          list(_h3it["required"].keys()) == ["transformer", "model_config"]
          and list(_h3it.get("optional", {}).keys()) == ["attention_backend", "sol_tau",
                                                          "quality_enhance", "audio_enhance", "step_cache", "block_cache"],
          f"-> req={list(_h3it['required'].keys())} opt={list(_h3it.get('optional', {}).keys())}")
    # (B) quality_enhance widget -> the ONE boolean begin option `video_enhance` (user 2026-09-19: the keep ratio
    #     is engine law; the plugin carries no number). NOTE: the pixel-level A/B (quality_enhance ON vs OFF) is
    #     the USER's ComfyUI acceptance — this arm only proves the widget->engine key wiring, not the visual effect.
    from qfn_test_pkg import qf_modelpatcher as _qmp_tp
    class _TPProbe(_qmp_tp.QFSessionModelMixin):
        pass
    _tp_on = _TPProbe(); _tp_on.set_video_enhance(True)
    _d_tp_on = _tp_on.residency_opts()
    _tp_off = _TPProbe(); _tp_off.set_video_enhance(False)
    _d_tp_off = _tp_off.residency_opts()
    _d_tp_def = _TPProbe().residency_opts()
    check("video_enhance wiring: set_video_enhance(True/False) -> residency_opts video_enhance True/False; default False (still sent); never the raw key",
          _d_tp_on.get("video_enhance") is True and _d_tp_off.get("video_enhance") is False
          and _d_tp_def.get("video_enhance") is False
          and not any("token_prune_keep_ratio" in d for d in (_d_tp_on, _d_tp_off, _d_tp_def)),
          f"-> on={_d_tp_on.get('video_enhance')} off={_d_tp_off.get('video_enhance')} default={_d_tp_def.get('video_enhance')}")
    check("no plugin-side keep-ratio mapper survives (the number is engine law)",
          not hasattr(qfn, "_quality_enhance_to_token_prune") and not hasattr(_qmp_tp.QFSessionModelMixin, "set_token_prune"),
          "-> mapper/set_token_prune still present" )
    # qfa REMOVED as a user-facing attention_backend choice (2026-09-13)
    check("attention_backend choices drop qfa (SM80+ and SM75)",
          "qfa" not in qfn._ATTN_BACKEND_SM80PLUS and "qfa" not in qfn._ATTN_BACKEND_SM75,
          f"-> sm80+={qfn._ATTN_BACKEND_SM80PLUS} sm75={qfn._ATTN_BACKEND_SM75}")
    # the ALL-IN single file: projections + BOTH modality connector blocks packed (the audio
    # one is ALSO the AV discriminant — no audio_vae staging, comfy owns audio decode).
    import struct as _st2
    _allin_name = "fx-ltx-2.5-allin-quantfunc-4bit.safetensors"
    _hdr2 = json.dumps({
        "text_embedding_projection.video_aggregate_embed.weight":
            {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "model.diffusion_model.video_embeddings_connector.learnable_registers":
            {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        "model.diffusion_model.audio_embeddings_connector.learnable_registers":
            {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
    }).encode()
    _pad2 = (8 - (len(_hdr2) % 8)) % 8
    _hdr2 += b" " * _pad2
    with open(os.path.join(dm, _allin_name), "wb") as _fh:
        _fh.write(_st2.pack("<Q", len(_hdr2))); _fh.write(_hdr2); _fh.write(b"\0" * 12)
    # per-piece detection BOTH WAYS: missing projections alone / missing connectors alone
    # are each named precisely (proves the probe discriminates, not a single blanket check).
    _half1 = "fx-ltx-2.5-connonly-quantfunc-4bit.safetensors"   # connectors, NO projections
    _h1 = json.dumps({"model.diffusion_model.video_embeddings_connector.learnable_registers":
                      {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    _h1 += b" " * ((8 - (len(_h1) % 8)) % 8)
    with open(os.path.join(dm, _half1), "wb") as _fh:
        _fh.write(_st2.pack("<Q", len(_h1))); _fh.write(_h1); _fh.write(b"\0" * 4)
    _half2 = "fx-ltx-2.5-projonly-quantfunc-4bit.safetensors"   # projections, NO connectors
    _h2 = json.dumps({"text_embedding_projection.video_aggregate_embed.weight":
                      {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    _h2 += b" " * ((8 - (len(_h2) % 8)) % 8)
    with open(os.path.join(dm, _half2), "wb") as _fh:
        _fh.write(_st2.pack("<Q", len(_h2))); _fh.write(_h2); _fh.write(b"\0" * 4)
    # NEW-contract expectations for the two half shapes: a video-only-connector
    # transformer does NOT self-qualify (joint-AV needs BOTH modalities) and a
    # proj-only transformer has no connector blocks at all — with a qualifying
    # sibling present, BOTH resolve via the completion file (never the retired 2.3
    # connector_ckpt dead-letter, whose widget no longer exists on the node).
    _msgs = {}
    for _f in (_half1, _half2):
        try:
            _o = LtxL.load(_f, "ltx2-2.5-22b")[0]
            _ = _o.model._qf.lib   # first touch materializes the staged pkg key
            _md_ = _o.model._qf._ckey[0]
            _msgs[_f] = os.path.realpath(os.path.join(_md_, "connectors", "model.safetensors"))
        except RuntimeError as e:
            _msgs[_f] = f"REFUSED: {str(e)[:70]}"
    check("ltx2 partial shapes resolve via the sibling completion file (both ways)",
          _msgs[_half1].endswith("fx-ltx25-connectors.safetensors")
          and _msgs[_half2].endswith("fx-ltx25-connectors.safetensors"),
          f"-> connonly:{_msgs[_half1][-44:]} | projonly:{_msgs[_half2][-44:]}")
    # isolated video-only-connector (no sibling): the ACCURATE joint-AV refusal —
    # names the missing audio modality, never the unactionable connector_ckpt message.
    _half1_iso = "fx-iso-ltx-2.5-connonly-quantfunc-4bit.safetensors"
    import shutil as _sh
    _sh.copyfile(os.path.join(dm, _half1), os.path.join(dm2, _half1_iso))
    try:
        LtxL.load(_half1_iso, "ltx2-2.5-22b")
        check("ltx2 video-only-connector with NO completion source gets the joint-AV refusal",
              False, "-> no exception")
    except RuntimeError as e:
        check("ltx2 video-only-connector with NO completion source gets the joint-AV refusal",
              "NOT the audio" in str(e) and "joint-AV" in str(e)
              and "connector_ckpt" not in str(e),
              f"-> {str(e)[:90]}")
    # ── positive path: the all-in file alone loads (AV via the packed audio connector) ──
    out_ltx = LtxL.load(_allin_name, "ltx2-2.5-22b")[0]
    check("ltx2 AV file-mode returns a QFModelPatcher",
          type(out_ltx).__name__ == "QFModelPatcher")
    check("ltx2 AV model is QFLTXAVModel (packed audio connector -> AV path)",
          type(out_ltx.model).__name__ == "QFLTXAVModel", f"-> {type(out_ltx.model).__name__}")
    _n0 = len(creates)
    _ = out_ltx.model._qf.lib
    check("ltx2 create deferred until first touch", len(creates) == _n0 + 1)
    _lcfg = creates[-1]
    check("ltx2 create cfg carries denoise_only", _lcfg.get("denoise_only") is True, f"-> {_lcfg}")
    _lmd = out_ltx.model._qf._ckey[0]
    _rx = os.path.realpath(os.path.join(_lmd, "transformer", "model.safetensors"))
    _rc2 = os.path.realpath(os.path.join(_lmd, "connectors", "model.safetensors"))
    check("ltx2 staged: transformer AND connectors BOTH self-link to the ONE all-in file",
          _rx.endswith(_allin_name) and _rc2.endswith(_allin_name), f"-> {_rx[-40:]} | {_rc2[-40:]}")
    check("ltx2 staged: NO text_encoder and NO audio_vae weights (one-file contract)",
          not os.path.exists(os.path.join(_lmd, "text_encoder", "model.safetensors"))
          and not os.path.exists(os.path.join(_lmd, "audio_vae", "model.safetensors")))
    check("ltx2 staged: config-complete + transformer_2 pruned (single-expert)",
          all(os.path.isfile(os.path.join(_lmd, q)) for q in
              ("model_index.json", "transformer/config.json", "vae/config.json",
               "connectors/config.json", "audio_vae/config.json"))
          and not os.path.exists(os.path.join(_lmd, "transformer_2")))
    # wan-align (user 2026-08-22 "只关注latent"): NO image socket, NO plugin/engine i2v
    # plumbing. The i2v mechanism is comfy's own Inplace latent + noise_mask ->
    # KSamplerX0Inpaint -> the INHERITED BaseModel.scale_latent_inpaint. Assert the
    # mechanism BOTH ways, on the live class + live model:
    from qfn_test_pkg import qf_ltx_modelpatcher as _qlm
    import comfy.model_base as _mb
    check("ltx scale_latent_inpaint is INHERITED (comfy's OWN LTXV mask semantics), not loud-failed",
          "scale_latent_inpaint" not in _qlm.QFLTXModel.__dict__
          and "scale_latent_inpaint" not in _qlm.QFLTXAVModel.__dict__
          and _qlm.QFLTXModel.scale_latent_inpaint is _mb.LTXV.scale_latent_inpaint)
    check("denoise_mask is NOT in the LTX reject list (Inplace latent route rides it)",
          "denoise_mask" not in _qlm.QFLTXModel._ENGINE_IGNORED_COND_KEYS)
    import torch as _t2
    _mdl = out_ltx.model
    try:
        _ = _mdl.extra_conds(denoise_mask=_t2.ones(1, 1, 2, 2, 2))
        _dm_ok, _dm_msg = True, "(accepted)"
    except Exception as _e:  # noqa: BLE001
        _dm_ok, _dm_msg = False, str(_e)[:80]
    check("extra_conds ACCEPTS denoise_mask (comfy consumes it outside the model)",
          _dm_ok, f"-> {_dm_msg}")
    try:
        _mdl.extra_conds(keyframe_idxs=_t2.zeros(1, 1))
        check("extra_conds still REJECTS keyframe_idxs loud", False, "-> no exception")
    except RuntimeError as _e:
        check("extra_conds still REJECTS keyframe_idxs loud",
              "keyframe_idxs" in str(_e), f"-> {str(_e)[:70]}")
    # ── masked-latent i2v: _frame_scales_from_mask unit arms (Z review: pure-fn, no e2e) ──
    import types as _types
    _red = _qlm.QFLTXModel._frame_scales_from_mask
    _stub = _types.SimpleNamespace(latent_shapes=[[1, 4, 2, 2, 2], [1, 8, 3, 16]])
    # (a) 5D channel-repeated [B,C,F,H,W]: frame0=0.3, frame1=1.0
    _m5 = _t2.ones(1, 4, 2, 2, 2)
    _m5[:, :, 0] = 0.3
    check("mask->scales: 5D per-frame reduce",
          [round(v, 4) for v in _red(_stub, _m5)] == [0.3, 1.0],
          f"-> {_red(_stub, _m5)}")
    # (b) packed AV flat [B,1,total]: video part frame0=0.3, audio part all-1.0
    _nv = 4 * 2 * 2 * 2
    _na = 8 * 3 * 16
    _flat = _t2.ones(1, 1, _nv + _na)
    _v = _flat[0, 0, :_nv].reshape(1, 4, 2, 2, 2)
    _v[:, :, 0] = 0.3
    check("mask->scales: packed AV flat split via latent_shapes",
          [round(v, 4) for v in _red(_stub, _flat)] == [0.3, 1.0],
          f"-> {_red(_stub, _flat)}")
    # (c) AUDIO-lane masking refused loud
    _flat_am = _flat.clone()
    _flat_am[0, 0, _nv:] = 0.5
    try:
        _red(_stub, _flat_am)
        check("mask->scales: audio-lane mask refused loud", False, "-> no exception")
    except RuntimeError as _e:
        check("mask->scales: audio-lane mask refused loud", "AUDIO" in str(_e),
              f"-> {str(_e)[:60]}")
    # (d) frame-INTERNAL variation (spatial inpaint) refused loud
    _m5v = _t2.ones(1, 4, 2, 2, 2)
    _m5v[0, 0, 0, 0, 0] = 0.2
    try:
        _red(_stub, _m5v)
        check("mask->scales: spatial (within-frame) mask refused loud", False, "-> no exception")
    except RuntimeError as _e:
        check("mask->scales: spatial (within-frame) mask refused loud",
              "WITHIN a latent frame" in str(_e), f"-> {str(_e)[:60]}")
    # (e) unknown rank refused loud
    try:
        _red(_stub, _t2.ones(2, 2))
        check("mask->scales: unknown mask rank refused loud", False, "-> no exception")
    except RuntimeError as _e:
        check("mask->scales: unknown mask rank refused loud", "unsupported mask source" in str(_e),
              f"-> {str(_e)[:60]}")
    # (f) integration: all-ones mask -> omitted (scalar path); conditioned mask -> stashed
    _ = _mdl.extra_conds(denoise_mask=_t2.ones(1, 4, 2, 2, 2))
    check("extra_conds: all-ones mask omits the begin key (scalar path kept)",
          getattr(_mdl, "_pending_frame_t_scale", "MISSING") is None)
    _ = _mdl.extra_conds(denoise_mask=_m5)
    check("extra_conds: conditioned mask stashed as per-frame scales",
          [round(v, 4) for v in (_mdl._pending_frame_t_scale or [])] == [0.3, 1.0],
          f"-> {_mdl._pending_frame_t_scale}")
    # minimax-h3 file-mode: minimal staging (configs + single xfm; no extra links)
    out_h3 = H3L.load("fx-minimax-h3-quantfunc-int4.safetensors", "minimax-h3-fl2va")[0]
    check("h3 file-mode returns a QFModelPatcher + QFH3Model",
          type(out_h3).__name__ == "QFModelPatcher"
          and type(out_h3.model).__name__ == "QFH3Model", f"-> {type(out_h3.model).__name__}")
    _n0 = len(creates)
    _ = out_h3.model._qf.lib
    _hcfg = creates[-1] if len(creates) > _n0 else json.loads(out_h3.model._qf._ckey[-1])
    check("h3 create cfg carries denoise_only", _hcfg.get("denoise_only") is True, f"-> {_hcfg}")
    _hmd = out_h3.model._qf._ckey[0]
    check("h3 staged: config-complete + xfm linked + transformer_2 pruned",
          all(os.path.isfile(os.path.join(_hmd, q)) for q in
              ("model_index.json", "transformer/config.json", "vae/config.json"))
          and os.path.realpath(os.path.join(_hmd, "transformer", "model.safetensors")).endswith("fx-minimax-h3-quantfunc-int4.safetensors")
          and not os.path.exists(os.path.join(_hmd, "transformer_2")))
    # a preset whose manifest family does not match the NODE's family → the defense-in-depth guard
    try:
        WanL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors",
                  "fx-t2v-4steps-low-quantfunc-int4.safetensors", "fx-ltx")
        check("cross-family preset on the wan node refused", False, "-> no exception")
    except RuntimeError as e:
        check("cross-family preset on the wan node refused", "declares family" in str(e))
    try:
        LtxL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors", "fx-alien")
        check("unknown-family preset refused (family guard)", False, "-> no exception")
    except RuntimeError as e:
        check("unknown-family preset refused (family guard)", "declares family" in str(e))
    for evil_cfg in ("../wan2.2-a14b-t2v", "a/b", "..", ""):
        try:
            WanL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors",
                      "fx-t2v-4steps-low-quantfunc-int4.safetensors", evil_cfg)
            check(f"model_config refuses {evil_cfg!r}", False, "-> loaded!")
        except RuntimeError:
            check(f"model_config refuses {evil_cfg!r}", True)
    # manifest-driven SHAPE mismatches
    try:
        WanL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors",
                  "fx-t2v-4steps-low-quantfunc-int4.safetensors", "fx-single")
        check("single-expert preset + transformer2 refused", False, "-> no exception")
    except RuntimeError as e:
        check("single-expert preset + transformer2 refused", "single-transformer" in str(e))

    # ── 3) wan dual-expert staging + denoise_only create cfg + DUAL MODEL outputs ──
    try:
        pair = WanL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors",
                         "fx-t2v-4steps-low-quantfunc-int4.safetensors", "wan2.2-a14b-t2v")
        check("wan loader returns a (high, low) pair", isinstance(pair, tuple) and len(pair) == 2)
        out, low = pair
        check("both outputs are QFModelPatcher",
              type(out).__name__ == "QFModelPatcher" and type(low).__name__ == "QFModelPatcher")
        check("both outputs SHARE one engine object", low.model._qf is out.model._qf)
        check("low output is the SHADOW (flag on the MODEL, survives clones)",
              getattr(low.model, "_qf_shadow", False) is True
              and not getattr(out.model, "_qf_shadow", False))
        check("shadow ledger share is the tiny constant (no double-count)",
              low.model_size() == type(low)._QF_SHADOW_LEDGER_BYTES
              and out.model_size() == max(1, int(out.model._qf.footprint_bytes)),
              f"-> low={low.model_size()} out={out.model_size()}")
        check("shadow never drives shared-engine eviction (partially_unload -> 0)",
              low.partially_unload(None, 10 ** 12) == 0)
        check("shadow reports 0 host RAM (primary owns the backup line)",
              low.loaded_ram_size() == 0)
        n0 = len(creates)
        _ = out.model._qf.lib          # first touch materializes
        check("create deferred until first touch", len(creates) == n0 + 1)
        md = out.model._qf._ckey[0]
        cfg = creates[-1]
        check("create cfg carries denoise_only", cfg.get("denoise_only") is True, f"-> {cfg}")
        check("staged dir is config-complete",
              all(os.path.isfile(os.path.join(md, p)) for p in
                  ("model_index.json", "transformer/config.json", "transformer_2/config.json",
                   "vae/config.json")))
        r1 = os.path.realpath(os.path.join(md, "transformer", "model.safetensors"))
        r2 = os.path.realpath(os.path.join(md, "transformer_2", "model.safetensors"))
        check("expert weight links resolve to the PICKED files",
              r1.endswith("fx-t2v-4steps-high-quantfunc-int4.safetensors") and r2.endswith("fx-t2v-4steps-low-quantfunc-int4.safetensors"))
        mi = json.load(open(os.path.join(md, "model_index.json")))
        check("staged model_index is dual-expert (boundary_ratio>0)",
              float(mi.get("boundary_ratio", 0)) > 0)
        vae = json.load(open(os.path.join(md, "vae", "config.json")))
        check("staged vae config carries the A14B wan2.1 scales (8 spatial / 4 temporal)",
              vae.get("scale_factor_spatial") == 8 and vae.get("scale_factor_temporal") == 4)
        # R8 POSITIVE arm — the pair-mate exemption must GRANT, not just the shadow short-circuit:
        # with the SHADOW STILL ALIVE, the primary's host-RAM release succeeds (same lazy-wrapper
        # identity => coherent re-create), and the shared engine self-heals on next use.
        out.detach(unpatch_all=False)       # engine holds its CPU backup now (already materialized)
        held_pair = out.loaded_ram_size()
        check("primary reports the backup while the shadow lives", held_pair > 0,
              f"-> {held_pair}")
        freed_pair = out.partially_unload_ram(10 ** 12)
        check("pair-mate exemption GRANTS the primary's release (shadow alive)",
              freed_pair == held_pair, f"-> freed {freed_pair} vs held {held_pair}")
        check("released pair engine self-heals on next use",
              out.model._qf.ensure().pipeline is not None and low.model._qf is out.model._qf)
        # ZERO-LATENT GUARD arm (user black-video class): _apply_model on an ALL-ZERO latent must
        # refuse LOUD (naming add_noise) BEFORE any engine call; a noised latent must get PAST the
        # guard (it then fails at _begin on the fixture's fake lib — proving the guard is the ONLY
        # thing that fired for zeros, and that it does NOT fire for nonzero input).
        import torch as _t
        _sig = _t.tensor([1.0])
        _topts = {"sample_sigmas": _t.tensor([1.0, 0.5, 0.0])}
        _zero = _t.zeros(1, 16, 3, 8, 8)
        _ctx = _t.zeros(1, 8, 4096)   # cond content is irrelevant to this guard
        try:
            out.model._apply_model(_zero, _sig, c_crossattn=_ctx, transformer_options=_topts)
            check("zero-latent guard fires (black-video class)", False, "-> no exception")
        except RuntimeError as e:
            check("zero-latent guard fires (black-video class)",
                  "ALL ZEROS" in str(e) and "add_noise" in str(e), f"-> {str(e)[:80]}")
        try:
            out.model._apply_model(_t.randn(1, 16, 3, 8, 8), _sig, c_crossattn=_ctx,
                                   transformer_options=_topts)
            check("noised latent passes the guard (reaches _begin)", False, "-> no exception??")
        except Exception as e:  # noqa: BLE001 — ANY non-guard failure proves it got PAST the
            # guard (on the fixture the fake lib then fails inside _begin, e.g. AttributeError);
            # only the guard's own message would mean the guard misfired on nonzero input.
            check("noised latent passes the guard (reaches _begin)",
                  "ALL ZEROS" not in str(e), f"-> {type(e).__name__}: {str(e)[:60]}")

        # N3: the guard is ONE mechanism, N users — wan is exercised through _apply_model above;
        # LTX2/H3 file-mode is not wired (build() refuses) so their models can't be instantiated
        # here, but their EXACT call is behavior-tested via the shared helper per family tag, AND
        # the call-before-engine wiring is asserted structurally for all three modules.
        import torch as _t2, re as _re2
        _zero5 = _t2.zeros(1, 16, 3, 8, 8)
        _noise5 = _t2.randn(1, 16, 3, 8, 8)
        for _tag in ("LTX", "LTX-AV", "H3", "wan"):
            try:
                qfn.qfmp.refuse_all_zero_initial_latent(_zero5, _tag)
                check(f"shared guard fires for {_tag}", False, "-> no exception")
            except RuntimeError as _e:
                check(f"shared guard fires for {_tag} with its tag",
                      f"qf_native {_tag}:" in str(_e) and "ALL ZEROS" in str(_e), f"-> {str(_e)[:50]}")
            try:
                qfn.qfmp.refuse_all_zero_initial_latent(_noise5, _tag)
                check(f"shared guard passes a noised latent for {_tag}", True)
            except Exception as _e:  # noqa: BLE001
                check(f"shared guard passes a noised latent for {_tag}", False, f"-> {_e!r}")
        # structural: each family module calls the guard BEFORE it opens the session (self._begin).
        import os as _os
        for _mod, _tags in (("qf_wan_modelpatcher.py", 1), ("qf_ltx_modelpatcher.py", 2),
                            ("qf_h3_modelpatcher.py", 1)):
            _src = open(_os.path.join(_PLUGIN, _mod)).read()
            _n_guard = _src.count("refuse_all_zero_initial_latent(")
            # every guard call must be followed (in source) by a self._begin( before the next guard
            _ok = _n_guard == _tags
            for _m in _re2.finditer("refuse_all_zero_initial_latent", _src):
                _after = _src[_m.end():_m.end() + 400]
                if "self._begin(" not in _after:
                    _ok = False
            check(f"{_mod}: {_tags} guard call(s), each before self._begin", _ok,
                  f"-> found {_n_guard}")
    except Exception as e:  # noqa: BLE001
        check("wan dual-expert staging", False, f"-> raised {type(e).__name__}: {e}")

    # a NON-conforming file name for the preset must be refused loud (file_hints mechanism)
    try:
        WanL.load("other.safetensors", "fx-t2v-4steps-low-quantfunc-int4.safetensors",
                  "wan2.2-a14b-t2v")
        check("file_hints refuses a non-conforming transformer1", False, "-> loaded!")
    except RuntimeError as e:
        check("file_hints refuses a non-conforming transformer1",
              "does not look like" in str(e))

    # wan without a low expert must refuse (A14B is dual-expert; "" maps to the none-sentinel)
    try:
        WanL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors", "", "wan2.2-a14b-t2v")
        check("wan single-file refused (dual-expert required)", False, "-> no exception")
    except RuntimeError as e:
        check("wan single-file refused (dual-expert required)", "DUAL-expert" in str(e))

    # CROSS-task discrimination: the i2v preset must REFUSE a t2v file (and the t2v preset an
    # i2v-named file) — file_hints are the only guard between the two same-family presets.
    for f in ("fx-i2v-4steps-high-quantfunc-int4.safetensors",
              "fx-i2v-4steps-low-quantfunc-int4.safetensors"):
        with open(os.path.join(dm, f), "wb") as fh:
            fh.write(b"\0" * 16)
    try:
        WanL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors",
                  "fx-i2v-4steps-low-quantfunc-int4.safetensors", "wan2.2-a14b-i2v")
        check("i2v preset refuses a t2v transformer1", False, "-> loaded!")
    except RuntimeError as e:
        check("i2v preset refuses a t2v transformer1", "does not look like" in str(e))
    try:
        WanL.load("fx-i2v-4steps-high-quantfunc-int4.safetensors",
                  "fx-t2v-4steps-low-quantfunc-int4.safetensors", "wan2.2-a14b-t2v")
        check("t2v preset refuses an i2v transformer1", False, "-> loaded!")
    except RuntimeError as e:
        check("t2v preset refuses an i2v transformer1", "does not look like" in str(e))
    pair_i2v = WanL.load("fx-i2v-4steps-high-quantfunc-int4.safetensors",
                         "fx-i2v-4steps-low-quantfunc-int4.safetensors",
                         "wan2.2-a14b-i2v")
    check("i2v preset loads a conforming pair (dual outputs)",
          isinstance(pair_i2v, tuple) and len(pair_i2v) == 2)
    mi_i2v = json.load(open(os.path.join(cfgroot, "wan2.2-a14b-i2v", "model_index.json")))
    check("i2v preset carries the official boundary 0.9",
          abs(float(mi_i2v.get("boundary_ratio", 0)) - 0.9) < 1e-6)
    # i2v COND FLOW arm (the E2E-caught wiring gap, both ways): with the stub's shape carrier
    # armed from the STAGED config, comfy's stock concat machinery must build the [mask|image]
    # tail (20ch) for the i2v package — and must build NOTHING for a t2v package (in==16).
    import torch as _t3
    _kw = dict(noise=_t3.zeros(1, 16, 3, 8, 8), device="cpu",
               concat_latent_image=_t3.randn(1, 16, 3, 8, 8),
               concat_mask=_t3.cat([_t3.zeros(1, 1, 1, 8, 8), _t3.ones(1, 1, 2, 8, 8)], dim=2),
               cross_attn=_t3.randn(1, 8, 4096))
    _oc = pair_i2v[0].model.extra_conds(**_kw)
    check("i2v package: extra_conds emits c_concat", "c_concat" in _oc, f"-> {sorted(_oc)}")
    if "c_concat" in _oc:
        _cc = _oc["c_concat"].cond
        check("i2v tail is [mask|image] = 20 channels", int(_cc.shape[1]) == 20,
              f"-> {tuple(_cc.shape)}")
        # comfy inverts the mask (1-mask): our concat_mask frame0=0 -> tail mask frame0=1
        check("i2v tail mask frame0==1 (engine frame0-known semantics)",
              float(_cc[0, 0, 0].mean()) == 1.0 and float(_cc[0, 0, 1].mean()) == 0.0)
    _ot = out.model.extra_conds(**_kw)
    check("t2v package: NO c_concat (in==16, extra_channels 0)", "c_concat" not in _ot,
          f"-> {sorted(_ot)}")
    # inter-stage thrash fix (D3/D4-hardened): comfy's eviction decisions call
    # memory_required THROUGH sampler_helpers.estimate_memory — `memory_required(shape,
    # cond_shapes=cond_shapes)` (KEYWORD). D4: the old positional-only override TypeError'd
    # on every real KSampler run while the old positional-only arm stayed green — so this
    # arm now (a) calls EXACTLY like the real call site, (b) asserts signature
    # compatibility against comfy's own BaseModel.memory_required so future comfy drift
    # goes red here, (c) asserts the D3 honesty properties: geometry-proportional,
    # monotonic, and far below the torch-WAN activation estimate that caused the eviction
    # thrash.
    import inspect as _insp
    from comfy.model_base import BaseModel as _CB
    _base_params = [q for q in _insp.signature(_CB.memory_required).parameters
                    if q != "self"]
    _ours = _insp.signature(type(out.model).memory_required)
    try:
        _ours.bind(out.model, [1, 16, 21, 80, 80],
                   **{q: {} for q in _base_params[1:]})
        _sig_ok = True
    except TypeError:
        _sig_ok = False
    check("memory_required signature accepts every BaseModel caller form (D4)", _sig_ok,
          f"-> base params {_base_params} vs ours {list(_ours.parameters)}")
    _shape = [2, 16, 21, 80, 80]      # sampler_helpers doubles batch for cfg
    _conds = {"c_crossattn": [[1, 512, 4096]], "c_concat": [[1, 20, 21, 80, 80]]}
    _mr = out.model.memory_required(_shape, cond_shapes=_conds)   # the REAL call form
    _mr_small = out.model.memory_required([1, 16, 3, 8, 8], cond_shapes={})
    _mr_big = out.model.memory_required([2, 16, 21, 192, 192], cond_shapes=_conds)
    check("memory_required geometry-proportional + monotonic (D3)",
          _mr_small < _mr < _mr_big, f"-> {_mr_small} < {_mr} < {_mr_big}")
    check("memory_required stays far below the torch-WAN estimate (thrash fix preserved)",
          _mr <= 1 << 30, f"-> {_mr}")

    # ── B1 (CR-2b): the OLD-.so compat qfa pin is SM-GATED — NEVER on the SM80 fault tier ──
    # The engine's qfa forward THROWS on SM80 (kQfaForwardFaultsOnSm) even for explicit
    # requests, so an unconditional plugin pin turned A100/A800 from runs-with-collapse-risk
    # into first-attention hard crash. The gate lives in _wan_device_sm() + the (86, 89) span;
    # these arms pin BOTH directions by rebuilding with the SM query monkeypatched.
    _wan_mod = sys.modules[type(out.model).__module__]
    _orig_sm = _wan_mod._wan_device_sm
    try:
        for _sm_val, _want_pin, _why in ((89, True,  "sm89: consumable tier -> pin qfa"),
                                         (86, True,  "sm86: consumable tier -> pin qfa"),
                                         (80, False, "sm80: engine qfa-fwd FAULT tier -> NO pin (AUTO)"),
                                         (75, False, "sm75: outside compat-pin span -> NO pin"),
                                         (0,  False, "no CUDA -> NO pin (fail-safe AUTO)")):
            _wan_mod._wan_device_sm = (lambda v: (lambda: v))(_sm_val)
            _pair = WanL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors",
                              "fx-t2v-4steps-low-quantfunc-int4.safetensors", "wan2.2-a14b-t2v")
            _p = _pair[0]
            _n0 = len(creates)
            _ = _p.model._qf.lib          # touch -> create (or cache-hit on an identical cfg)
            if len(creates) > _n0:
                _cfg = creates[-1]
            else:
                # cache-hit: cfg identical to an earlier create — decode it from the cache
                # key. cfg-json is the LAST element of BOTH the real _get_engine ckey (5-tuple)
                # and the suite's fake ckey (3-tuple), so [-1] is shape-agnostic.
                _cfg = json.loads(_p.model._qf._ckey[-1])
            check("B1 " + _why,
                  (_cfg.get("attention_backend") == "qfa") if _want_pin
                  else ("attention_backend" not in _cfg),
                  f"-> cfg={_cfg}")
    finally:
        _wan_mod._wan_device_sm = _orig_sm

    # single-expert staging shape (direct helper call — no single-expert family is wired yet,
    # but the helper's contract must already hold for the one that will be)
    try:
        stage = qfn.qfmp.stage_denoise_only_package(
            os.path.join(_PLUGIN, "configs", "wan2.2-a14b-t2v"),
            os.path.join(dm, "other.safetensors"), None)
        check("single-expert staging PRUNES transformer_2",
              not os.path.exists(os.path.join(stage, "transformer_2")))
    except Exception as e:  # noqa: BLE001
        check("single-expert staging", False, f"-> raised {type(e).__name__}: {e}")

    # ── 4) containment: traversal/absolute names cannot escape the model roots ──
    for evil in ("../../../../etc/passwd", "/etc/passwd", "fx-t2v-4steps-high-quantfunc-int4.safetensors/../../x"):
        try:
            qfn._resolve_transformer(evil)
            check(f"containment refuses {evil!r}", False, "-> resolved!")
        except Exception:  # noqa: BLE001 — comfy raises its own error type
            check(f"containment refuses {evil!r}", True)
    try:
        ok = qfn._resolve_transformer("fx-t2v-4steps-high-quantfunc-int4.safetensors") == os.path.join(dm, "fx-t2v-4steps-high-quantfunc-int4.safetensors")
        check("containment still resolves a legit file", ok)
    except Exception as e:  # noqa: BLE001
        check("containment still resolves a legit file", False, f"-> {e!r}")

    # ── 5-8) substrate arms (adopt / RAM honesty / sibling safety / LoRA chain) ──
    # Fixture note: liveness is per-ckey = per (staged-dir, cfg). The staged dir is keyed by the
    # FILE PAIR, so distinct pairs give naturally isolated ckeys per arm.
    try:
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        lora_dir = os.path.join(tmp, "loras")
        os.makedirs(lora_dir, exist_ok=True)
        for f in ("a.safetensors", "b.safetensors"):
            open(os.path.join(lora_dir, f), "wb").write(b"\0" * 16)
        qfn._lora_choices = lambda: ["a.safetensors", "b.safetensors"]
        qfn._resolve_lora = lambda n: os.path.join(lora_dir, n)
        LoraNode = qfn.NODE_CLASS_MAPPINGS["QuantFuncNativeLoRA"]()

        base, base_low = WanL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors",
                                   "fx-t2v-4steps-low-quantfunc-int4.safetensors",
                                   "wan2.2-a14b-t2v")
        shifted = ModelSamplingSD3().patch(base, 11.0)[0]     # upstream comfy patch on the pair's high
        check("comfy patches still apply to the dual outputs (clone path intact)",
              "model_sampling" in shifted.object_patches)
        # [wiring-lora] v2 (retires the v1 refusal): chaining on the dual outputs WORKS with the
        # target derived from the WIRE (no widget), over ONE shared engine object.
        from qfn_test_pkg import qf_modelpatcher as _qmp2
        outH = LoraNode.apply(shifted, "a.safetensors", 0.8)[0]
        stH = _qmp2.lora_stack_of(outH)
        check("LoRA on the HIGH wire derives target=high (no widget)",
              len(stH) == 1 and stH[0]["target"] == "high" and stH[0]["scale"] == 0.8,
              f"-> {stH}")
        check("high-wire rebuild keeps the upstream comfy patch (adopt)",
              "model_sampling" in outH.object_patches)
        outL = LoraNode.apply(base_low, "b.safetensors", 0.5)[0]
        stL = _qmp2.lora_stack_of(outL)
        check("LoRA on the LOW wire derives target=low",
              len(stL) == 1 and stL[0]["target"] == "low", f"-> {stL}")
        engH, engL = outH.model._qf, outL.model._qf
        check("both wires share ONE engine object (no fork), union holds both sides",
              engH is engL and engH is base.model._qf
              and [e["target"] for e in engH.lora_union()] == ["high", "low"],
              f"-> same={engH is engL} union={engH.lora_union()}")
        outH2 = LoraNode.apply(outH, "b.safetensors", 0.3)[0]
        stH2 = _qmp2.lora_stack_of(outH2)
        check("second node on the same wire accumulates FUNCTIONALLY (no loader-cache mutation)",
              [(e["target"], e["scale"]) for e in stH2] == [("high", 0.8), ("high", 0.3)]
              and [e["target"] for e in engH.lora_union()] == ["high", "high", "low"]
              and _qmp2.lora_stack_of(base) == [],
              f"-> wire={stH2} union={engH.lora_union()} loader_stack={_qmp2.lora_stack_of(base)}")
        # DEAD-AUTHOR convergence (reviewer-C death rule): a registry side must never outlive
        # the node that authored it. Simulate "user deleted the LoRA nodes and rewired the
        # sampler to the RAW loader outputs": the raw models' run-start assert resets their
        # sides. Both directions: re-asserting the rebuilt wire restores it.
        base.model._assert_wire_lora()            # raw HIGH wire => high side reset to []
        _u1 = [e["target"] for e in engH.lora_union()]
        base_low.model._assert_wire_lora()        # raw LOW wire => low side reset too
        _u2 = [e["target"] for e in engH.lora_union()]
        outH2.model._assert_wire_lora()           # rebuilt wire re-asserts its stack
        _u3 = [e["target"] for e in engH.lora_union()]
        check("registry side never outlives its author node (raw assert resets; both ways)",
              _u1 == ["low"] and _u2 == [] and _u3 == ["high", "high"],
              f"-> after-raw-high={_u1} after-raw-low={_u2} re-asserted={_u3}")
        # RECONCILE unit, both ways: a materialized engine retires ONLY on a union change —
        # and the retire fires from inside ensure() itself (the chokepoint; no per-family
        # begin hook exists to forget).
        _drops = []

        class _FakeReal:
            pipeline = object()
            current_session = None
            step_count = 0
            sampler_step_count = 0
            footprint_bytes = 0

            def end_session_if_open(self):
                return (False, True)
        _le = _qmp2.QFLazyEngine(lambda: (_FakeReal(), "new-ck"), 0,
                                 retire=lambda _ck, _e, _req=None, **_kw: _drops.append(_ck) or True)
        _le.set_lora_side("high", [{"path": "x", "scale": 1.0, "target": "high"}])
        _le._real, _le._ckey = _FakeReal(), "old-ck"
        _le._created_lora_sig = _le._lora_sig()
        _same = _le.reconcile_lora()          # unchanged union -> no retire
        _le.set_lora_side("low", [{"path": "y", "scale": 1.0, "target": "low"}])
        _r2 = _le.ensure()                    # ensure() itself must drift-retire + re-create
        check("ensure() drift-retires ONLY on a union change (both ways) + re-creates",
              _same is False and _drops == ["old-ck"] and _le._ckey == "new-ck"
              and _r2 is not None and _le._created_lora_sig == _le._lora_sig(),
              f"-> same={_same} drops={_drops} ckey={_le._ckey}")
        # MID-SESSION drift must refuse LOUD (never retarget under a running generation) —
        # both ways: same union with an open session passes silently.
        _le._real.current_session = object()
        _mid_ok = True
        try:
            _le.reconcile_lora()              # sig == created sig -> no-op even mid-session
        except RuntimeError:
            _mid_ok = False
        _le.set_lora_side("high", [])         # drift while session open
        _mid_raised = False
        try:
            _le.reconcile_lora()
        except RuntimeError as _e:
            _mid_raised = "MID-GENERATION" in str(_e)
        check("mid-session drift refuses LOUD (both ways)", _mid_ok and _mid_raised,
              f"-> same-sig-ok={_mid_ok} drift-raised={_mid_raised}")
        # set_lora_side without retire must refuse LOUD (half-adoption leak guard):
        _naked = _qmp2.QFLazyEngine(lambda: (None, None), 0)
        try:
            _naked.set_lora_side("high", [])
            check("set_lora_side refuses without retire (half-adoption guard)", False,
                  "-> accepted!")
        except RuntimeError as _e:
            check("set_lora_side refuses without retire (half-adoption guard)",
                  "retire" in str(_e), f"-> {_e}")
        # identity-gated retire: a FOREIGN live consumer on the same ckey must SKIP the destroy.
        class _W:                       # two distinct wrapper identities
            pass
        _w1, _w2 = _W(), _W()

        class _M:                       # a bound model pinning the cache entry
            def __init__(self, w):
                self._qf = w
        import weakref as _wr
        _mine, _theirs = _M(_w1), _M(_w2)
        _destroyed = []

        class _Eng:
            def destroy(self):
                _destroyed.append(True)
        qfn._PIPELINE_CACHE["shared-ck"] = "sentinel"
        qfn._PIPELINE_MODELS["shared-ck"] = [_wr.ref(_mine), _wr.ref(_theirs)]
        _r1 = qfn._retire_handle("shared-ck", _Eng(), _w1, reason="arm")
        _kept = (_r1 is False and qfn._PIPELINE_CACHE.get("shared-ck") == "sentinel"
                 and not _destroyed)
        qfn._PIPELINE_MODELS["shared-ck"] = [_wr.ref(_mine)]          # only OUR consumer left
        _r2 = qfn._retire_handle("shared-ck", _Eng(), _w1, reason="arm")
        _dropped = (_r2 is True and "shared-ck" not in qfn._PIPELINE_CACHE
                    and _destroyed == [True])
        # keep_binding semantics (release path): binding list survives a granted retire
        qfn._PIPELINE_CACHE["kb-ck"] = "sentinel2"
        qfn._PIPELINE_MODELS["kb-ck"] = [_wr.ref(_mine)]
        _r3 = qfn._retire_handle("kb-ck", _Eng(), _w1, keep_binding=True, reason="arm")
        _kb = _r3 is True and "kb-ck" not in qfn._PIPELINE_CACHE and "kb-ck" in qfn._PIPELINE_MODELS
        qfn._PIPELINE_MODELS.pop("kb-ck", None)
        check("retire is identity-gated (foreign keeps; own-only destroys; keep_binding honored)",
              _kept and _dropped and _kb, f"-> kept={_kept} dropped={_dropped} kb={_kb}")
        # RETIRE-CHOKEPOINT property (reviewer-F: the substring scan was hollow — it matched
        # 'the word appeared', not 'a gate was applied'; an unconditional destroy containing a
        # .pop() passed it). REAL property: a `.destroy` CALL — attribute form OR the
        # getattr-alias form — may appear ONLY inside _retire_handle's own subtree. File set
        # DERIVED from the package (top modules + _FAMILY_MODULES), not a hand-written tuple.
        # Mutation-proven: reverting any path to a direct eng.destroy() turns this arm RED.
        import ast as _ast
        import os as _os
        import qfn_test_pkg as _pkg
        _viol = []
        _pkg_dir = _os.path.dirname(_qmp2.__file__)
        _scan_files = ["__init__.py", "qf_modelpatcher.py", "qf_engine.py"] + \
                      [_m + ".py" for _m in _pkg._FAMILY_MODULES]
        for _fn in _scan_files:
            _p = _os.path.join(_pkg_dir, _fn)
            if not _os.path.exists(_p):
                _viol.append(f"{_fn}:MISSING")
                continue
            _src = open(_p).read()
            _tree = _ast.parse(_src)
            _allowed = set()
            for _n in _ast.walk(_tree):
                if isinstance(_n, _ast.FunctionDef) and _n.name == "_retire_handle":
                    _allowed = {id(_c) for _c in _ast.walk(_n)}
            for _n in _ast.walk(_tree):
                if not isinstance(_n, _ast.Call):
                    continue
                _is_destroy = isinstance(_n.func, _ast.Attribute) and _n.func.attr == "destroy"
                _is_alias = (isinstance(_n.func, _ast.Name) and _n.func.id == "getattr"
                             and len(_n.args) >= 2 and isinstance(_n.args[1], _ast.Constant)
                             and _n.args[1].value == "destroy")
                if (_is_destroy or _is_alias) and id(_n) not in _allowed:
                    _viol.append(f"{_fn}:{_n.lineno}")
        check("destroy() is callable ONLY inside _retire_handle (AST call+alias, derived files)",
              not _viol, f"-> violations={_viol}")
        # recursive create-guard (reviewer-D LOW): a session knob nested at any depth refuses.
        from qfn_test_pkg import qf_engine as _rqe
        _nr = 0
        for _bad in ({"a": {"step_cache": 1}}, {"lora": [{"sparse": 2}]}):
            try:
                _rqe._refuse_session_knobs_in_create(_bad)
            except RuntimeError:
                _nr += 1
        _nok = True
        try:
            _rqe._refuse_session_knobs_in_create({"a": {"b": 1}, "lora": [{"path": "x"}]})
        except RuntimeError:
            _nok = False
        check("create guard refuses a session knob at ANY depth (both ways)", _nr == 2 and _nok,
              f"-> nested-raised={_nr}/2 clean-passed={_nok}")
        # STRING (pre-serialized JSON) form must ALSO refuse (qf_engine isinstance(cfg,str) branch —
        # its only prior coverage was the deleted resident_block_count block; re-added with a live key)
        _sr = False
        try:
            _rqe._refuse_session_knobs_in_create('{"denoise_only": true, "step_cache": 0.05}')
        except RuntimeError:
            _sr = True
        _sok = True
        try:
            _rqe._refuse_session_knobs_in_create('{"denoise_only": true}')
        except RuntimeError:
            _sok = False
        check("create guard refuses a session knob in the STRING (JSON) form too",
              _sr and _sok, f"-> str-with-key raised={_sr} str-clean passed={_sok}")
    except Exception as e:  # noqa: BLE001
        check("wiring-derived dual LoRA arm", False, f"-> raised {type(e).__name__}: {e}")

    # ── 6) HOST-RAM honesty on a DEDICATED file pair (isolated ckey) ──
    try:
        for f in ("ram-t2v-high-quantfunc-int4.safetensors", "ram-t2v-low-quantfunc-int4.safetensors"):
            open(os.path.join(dm, f), "wb").write(b"\0" * 16)
        fresh, fresh_low = WanL.load("ram-t2v-high-quantfunc-int4.safetensors",
                                     "ram-t2v-low-quantfunc-int4.safetensors",
                                     "wan2.2-a14b-t2v")
        eng = fresh.model._qf
        check("never-created engine reports 0 host RAM", fresh.loaded_ram_size() == 0)
        check("never-created engine frees 0 host RAM", fresh.partially_unload_ram(10 ** 12) == 0)
        _ = eng.lib
        fresh.detach(unpatch_all=False)
        held = fresh.loaded_ram_size()
        check("evicted engine DOES report its CPU backup", held > 0, f"-> {held}")
        freed = fresh.partially_unload_ram(10 ** 12)
        check("partially_unload_ram frees the backup", freed == held, f"-> {freed}")
        check("released handle re-creates on next use",
              getattr(eng, "materialized", True) is False)
        fresh.partially_unload_ram(10 ** 12, subsets=["patches"])
        check("partially_unload_ram accepts comfy's subsets kwarg", True)
    except Exception as e:  # noqa: BLE001
        check("host-RAM accounting", False, f"-> raised {type(e).__name__}: {e}")

    # ── 7) SHARED-handle sibling safety on a DEDICATED pair ──
    try:
        import gc
        for f in ("sh-t2v-high-quantfunc-int4.safetensors", "sh-t2v-low-quantfunc-int4.safetensors"):
            open(os.path.join(dm, f), "wb").write(b"\0" * 16)
        pa, pa_low = WanL.load("sh-t2v-high-quantfunc-int4.safetensors",
                               "sh-t2v-low-quantfunc-int4.safetensors", "wan2.2-a14b-t2v")
        pb, pb_low = WanL.load("sh-t2v-high-quantfunc-int4.safetensors",
                               "sh-t2v-low-quantfunc-int4.safetensors", "wan2.2-a14b-t2v")
        ra = pa.model._qf.ensure()
        rb = pb.model._qf.ensure()
        check("two loads of one file-pair share the real handle", ra is rb)
        pa.detach(unpatch_all=False)
        check("sibling-shared release REFUSES (freed 0, sibling alive)",
              pa.partially_unload_ram(10 ** 12) == 0)
        check("sibling's pipeline SURVIVES the refused release",
              pb.model._qf.pipeline is not None)
        check("sibling still honestly reports its backup", pb.loaded_ram_size() > 0)
        del pb, pb_low, rb
        gc.collect()
        freed = pa.partially_unload_ram(10 ** 12)
        check("sole-consumer release DOES free once the sibling is gone", freed > 0, f"-> {freed}")
        check("released wrapper reports 0 afterwards", pa.loaded_ram_size() == 0)
        check("released wrapper self-heals on next use",
              pa.model._qf.ensure().pipeline is not None)
        del pa, pa_low, ra
        gc.collect()
    except Exception as e:  # noqa: BLE001
        check("shared-handle sibling safety", False, f"-> raised {type(e).__name__}: {e}")

    # ── 8) the passive SWEEP path (real _PIPELINE_CACHE/_bind_pipeline_model/_sweep) ──
    try:
        import gc

        class _M:                      # weakref-able stand-in for a family model
            pass

        ck = ("sweep_pkg", "{}")
        other = ("sweep_other", "{}")
        h = _DummyEngine()
        qfn._PIPELINE_CACHE[ck] = h
        m1, m2 = _M(), _M()
        qfn._bind_pipeline_model(ck, m1)
        qfn._bind_pipeline_model(ck, m2)
        del m1
        gc.collect()
        qfn._sweep_dead_pipelines(other)
        check("sweep spares the handle while ANY consumer lives",
              h.pipeline is not None and ck in qfn._PIPELINE_CACHE)
        del m2
        gc.collect()
        qfn._sweep_dead_pipelines(other)
        check("sweep destroys once ALL consumers are dead",
              h.pipeline is None and ck not in qfn._PIPELINE_CACHE)
        ck2 = ("sweep_keep", "{}")
        h2 = _DummyEngine()
        qfn._PIPELINE_CACHE[ck2] = h2
        m3 = _M()
        qfn._bind_pipeline_model(ck2, m3)
        del m3
        gc.collect()
        qfn._sweep_dead_pipelines(ck2)
        check("sweep never touches keep_key", h2.pipeline is not None
              and ck2 in qfn._PIPELINE_CACHE)
        ck3 = ("sweep_unbound", "{}")
        h3 = _DummyEngine()
        qfn._PIPELINE_CACHE[ck3] = h3
        qfn._sweep_dead_pipelines(other)
        check("sweep never touches an unbound (in-flight) entry",
              h3.pipeline is not None and ck3 in qfn._PIPELINE_CACHE)
        for k in (ck2, ck3):
            qfn._PIPELINE_CACHE.pop(k, None)
            qfn._PIPELINE_MODELS.pop(k, None)
        ck4 = ("sweep_dedup", "{}")
        m4 = _M()
        qfn._bind_pipeline_model(ck4, m4)
        qfn._bind_pipeline_model(ck4, m4)
        check("re-binding a live model does not accumulate refs",
              len(qfn._PIPELINE_MODELS.get(ck4, [])) == 1)
        qfn._PIPELINE_MODELS.pop(ck4, None)
        del m4
    except Exception as e:  # noqa: BLE001
        check("sweep liveness coverage", False, f"-> raised {type(e).__name__}: {e}")

    # ── 9) file_hints CONTRACT completeness (mechanism, not memory — mirrors the family-module
    #       derivation arm): every SHIPPED preset must declare file_hints for its required arms,
    #       and each hint set must DISCRIMINATE (a matching name passes; a non-matching name is
    #       refused loud). A future preset landing without the contract turns the suite red HERE,
    #       instead of a user meeting an uncorresponding dropdown. ──
    try:
        import fnmatch as _fn
        real_cfg = os.path.join(_PLUGIN, "configs")
        presets = [d for d in sorted(os.listdir(real_cfg))
                   if os.path.isfile(os.path.join(real_cfg, d, "qf_native.json"))]
        check("at least one shipped preset exists", bool(presets), f"-> {presets}")
        for d in presets:
            mf = json.load(open(os.path.join(real_cfg, d, "qf_native.json")))
            hints = mf.get("file_hints")
            need = ["transformer1"] + (["transformer2"] if mf.get("dual_expert") else [])
            check(f"preset {d} declares file_hints for {need}",
                  isinstance(hints, dict) and all(hints.get(a) for a in need),
                  f"-> {hints and sorted(hints.keys())}")
            if not isinstance(hints, dict):
                continue
            for arm in need:
                pats = [str(x).lower() for x in (hints.get(arm) or [])]
                if not pats:
                    continue
                # BOTH-WAYS: derive a matching name from the first pattern, and use an
                # unmistakably-foreign name for the refusal direction.
                match_name = pats[0].replace("*", "x") + ".safetensors"                     if not pats[0].endswith(".safetensors") else pats[0].replace("*", "x")
                ok_match = any(_fn.fnmatch(match_name, p) for p in pats)
                ok_refuse = not any(_fn.fnmatch("totally-unrelated-model.safetensors", p) for p in pats)
                check(f"preset {d}/{arm} hints discriminate both ways", ok_match and ok_refuse,
                      f"-> match({match_name})={ok_match} refuse(unrelated)={ok_refuse}")
    except Exception as e:  # noqa: BLE001
        check("file_hints contract scan", False, f"-> raised {type(e).__name__}: {e}")

    print("LOADER_DISPATCH:", "PASS" if bad == 0 else f"FAIL ({bad} wrong)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
