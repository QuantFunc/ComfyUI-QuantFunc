#!/usr/bin/env python3
"""Behavioural tests for the FILE-BASED loader node (INT8-Fast-aligned redesign) + the
liveness/LoRA substrate:

  1. UI surface — exactly the four single-expert loaders (no Wan / cloud-TE node), FILE dropdowns +
     model_config (official presets, data-driven from configs/)
  2. dispatch by the preset MANIFEST's family; ltx2/minimax-h3 FILE-MODE staging shape;
     traversal / cross-family presets refused
  3. single-expert STAGING (Krea-2) — configs copied from the shipped bundle, weight SYMLINKED,
     denoise_only (and no LoRA) in the create cfg; the shared zero-latent guard; memory_required D3/D4
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
import types

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
        # Fake-engine test: run ComfyUI in its own --cpu mode (the contract tests' idiom), so a box with no visible GPU
        # (the CPU suite hides CUDA) imports comfy instead of skipping every arm. Must precede the first comfy import.
        sys.argv = [sys.argv[0], "--cpu"]
        import comfy.options
        comfy.options.enable_args_parsing()
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

    def end_session_if_open(self):
        return (False, True)

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
    for f in ("other.safetensors", "not-a-model.txt"):
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
                    (dm, "fx-krea2-turbo-quantfunc-int4.safetensors"),
                    (dm, "fx-qwen-image-2.1-quantfunc-int4.safetensors"),
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
    shutil.copytree(os.path.join(_PLUGIN, "configs", "ltx2-2.5-22b"),
                    os.path.join(cfgroot, "ltx2-2.5-22b"))
    shutil.copytree(os.path.join(_PLUGIN, "configs", "minimax-h3-fl2va"),
                    os.path.join(cfgroot, "minimax-h3-fl2va"))
    shutil.copytree(os.path.join(_PLUGIN, "configs", "krea2-turbo-int4"),
                    os.path.join(cfgroot, "krea2-turbo-int4"))
    shutil.copytree(os.path.join(_PLUGIN, "configs", "qwen-image-2.1-int4"),
                    os.path.join(cfgroot, "qwen-image-2.1-int4"))
    for name, mf in (("fx-ltx", {"family": "ltx2"}),
                     ("fx-h3", {"family": "minimax-h3"}),
                     ("fx-alien", {"family": "no-such-family"})):
        os.makedirs(os.path.join(cfgroot, name))
        json.dump(mf, open(os.path.join(cfgroot, name, "qf_native.json"), "w", encoding="utf-8"))
    qfn._CONFIGS_DIR = cfgroot

    creates = []
    create_devices = []
    fake_cache = {}
    prepared_cache = {}

    class _DummyPrepared(qfn.qfmp.QFPreparedEntry):
        """Prepared/resource half of the production lazy-factory contract."""
        def __init__(self):
            # an engine without the cold-need entry (#738): the cold ask keeps comfy's own floor
            self.resource = types.SimpleNamespace(cold_vram_need_bytes=lambda: qfn.qfmp.qfe.ColdNeed(None, None))
            self.capacity_bytes = 4096
            self._materializers = []

        def bind_materializer(self, engine, cache_key):
            self._materializers.append(engine)
            engine._bind_capacity(self.capacity_bytes)

        def materialize(self, preferred=None):
            target = preferred or self._materializers[0]
            return target._materialize_prepared(self)

    _upd_log, _upd_status = [], [0]

    class _ContractEngine(qfn.qfe.QFEngineHandle):
        """Materialized handle carrying the exact Prepared resource identity."""
        def __init__(self, resource):
            super().__init__(self, object(), footprint_bytes=_DummyEngine.footprint_bytes,
                             resource=resource, capacity_bytes=4096)

        # The quantfunc_pipeline_update ABI boundary (the handle is its own lib), recorded for the WHOLE file: a
        # LoRA-free run must never send one (checked before arm 5b); arm 5b reads the same log and status.
        @staticmethod
        def quantfunc_pipeline_update(_pipe, payload):
            _upd_log.append(json.loads(payload))
            return _upd_status[0]

        def end_session_if_open(self):
            return (False, True)

        def vram_need_bytes(self, latent_shape):
            # The quantfunc_vram_need_bytes ABI boundary, stubbed like the capacity ABI below: the engine's answer
            # is the working set of ONE forward of that latent, so a shape-proportional stand-in (4 B per latent
            # element) lets memory_required's real combination logic run. (memory_required asks the engine since
            # c71c315; without this the file aborted at the D3 arm and never reached its later checks.)
            n = 1
            for d in latent_shape:
                n *= int(d)
            return 4 * n

        def destroy(self):
            self.pipeline = None
            self.current_session = None

    def fake_get_engine(model_dir, create_cfg=None, device_idx=0, api_key=None, transformer_path=None):
        # Mirrors the real Prepared -> first-touch materialization contract (the transformer file is a create input).
        ck = (model_dir, int(device_idx), transformer_path, json.dumps(create_cfg or {}, sort_keys=True))
        eng = fake_cache.get(ck)
        if eng is not None and eng.pipeline is not None:
            return eng, ck
        entry = prepared_cache.setdefault(ck, _DummyPrepared())
        if qfn.qfe.FACTORY_PREPARE_ONLY.get():
            return entry, ck
        creates.append(dict(create_cfg or {}))
        create_devices.append(int(device_idx))
        eng = _ContractEngine(entry.resource)
        fake_cache[ck] = eng
        return eng, ck
    qfn._get_engine = fake_get_engine
    # This is a routing/liveness fixture: its 16-byte transformer files are not
    # native-loadable checkpoints. Stub the capacity ABI boundary explicitly,
    # just as creation is stubbed above; never depend on a production fallback
    # to file size (or a bundled engine .so) to keep later assertions running.
    qfn.qfe.load_lib = lambda *args, **kwargs: _DummyEngine.lib
    # Builders CAPTURE deps at registration — re-register so they hold the stub.
    qfn._FAMILY_BUILDERS.clear()
    qfn._FAMILY_MATCHERS.clear()
    qfn._register_families()

    LtxL = qfn.NODE_CLASS_MAPPINGS["QuantFuncLTXLoader"]()
    H3L = qfn.NODE_CLASS_MAPPINGS["QuantFuncH3Loader"]()
    KreaL = qfn.NODE_CLASS_MAPPINGS["QuantFuncKrea2Loader"]()

    # Every QuantFunc loader takes log_level as a HIDDEN input (qf_log_level, attached after every registration):
    # users never see or choose it, and the per-family surface checks below see only the family's own inputs.
    _ll_loaders = [n for n in qfn.NODE_CLASS_MAPPINGS if n.startswith("QuantFunc") and n.endswith("Loader")]
    _ll_bad = []
    for _n in _ll_loaders:
        _s = qfn.NODE_CLASS_MAPPINGS[_n].INPUT_TYPES()
        if ("log_level" in {**_s.get("required", {}), **_s.get("optional", {})}
                or _s.get("hidden", {}).get("log_level") != ("STRING", {})):
            _ll_bad.append(_n)
    check("log_level is hidden (never a visible input) on every QuantFunc loader",
          bool(_ll_loaders) and not _ll_bad, f"-> {len(_ll_loaders)} loaders, wrong: {_ll_bad}")

    # ── 1) UI surface: the release ships exactly the four single-expert loaders (user scope 2026-09-24
    #      「我们的新功能支持好LTX2.5 H3 Krea2 QI就好 还有wan的节点 还有cloud TE节点先干掉吧」); preset dropdowns are
    #      FAMILY-FILTERED ──
    _nodes = sorted(qfn.NODE_CLASS_MAPPINGS)
    check("the Wan loader and the cloud-TE node are not registered",
          not any("Wan" in k or "CloudTE" in k for k in _nodes)
          and {"QuantFuncLTXLoader", "QuantFuncH3Loader", "QuantFuncKrea2Loader", "QuantFuncQwenImage21Loader",
               "QuantFuncNativeLoRA"} <= set(_nodes), f"-> {_nodes}")
    # model_config (user 2026-09-24 「model_config也不是设置啊 就一个选项没意义啊」): the FIRST optional input, right after the
    # transformer (widget index 1, where the dropdown always was), HIDDEN + socketless while the family ships ONE preset
    # (the widget keeps its slot, so saved workflows keep every value in place); a family with
    # two presets shows it again (this fixture gives ltx2 and h3 a synthetic second one).
    ltx_cfgs = LtxL.INPUT_TYPES()["optional"]["model_config"][0]
    check("ltx model_config lists ONLY ltx2 presets",
          "fx-ltx" in ltx_cfgs and "krea2-turbo-int4" not in ltx_cfgs, f"-> {ltx_cfgs}")
    h3_cfgs = H3L.INPUT_TYPES()["optional"]["model_config"][0]
    check("h3 model_config lists ONLY minimax-h3 presets",
          "fx-h3" in h3_cfgs and "krea2-turbo-int4" not in h3_cfgs, f"-> {h3_cfgs}")
    _mc_bad = {}
    for _n in ("QuantFuncLTXLoader", "QuantFuncH3Loader", "QuantFuncKrea2Loader", "QuantFuncQwenImage21Loader"):
        _it = qfn.NODE_CLASS_MAPPINGS[_n].INPUT_TYPES()
        _opt = _it.get("optional", {})
        if list(_it["required"]) != ["transformer"] or list(_opt)[:1] != ["model_config"]:
            _mc_bad[_n] = (list(_it["required"]), list(_opt)[:2])
            continue
        _choices, _o = _opt["model_config"]
        if bool(_o.get("hidden")) != (len(_choices) == 1) or bool(_o.get("socketless")) != (len(_choices) == 1) \
                or _o.get("default") != _choices[0]:
            _mc_bad[_n] = (_choices, _o)
    check("every loader: transformer alone is required; model_config is the first optional input, hidden + socketless "
          "exactly when the family ships one preset, defaulting to it", not _mc_bad, f"-> {_mc_bad}")
    _real_cfg, _saved_cfg = os.path.join(_PLUGIN, "configs"), qfn._CONFIGS_DIR
    qfn._CONFIGS_DIR = _real_cfg
    try:
        _real = {_n: qfn.NODE_CLASS_MAPPINGS[_n].INPUT_TYPES() for _n in
                 ("QuantFuncLTXLoader", "QuantFuncH3Loader", "QuantFuncKrea2Loader", "QuantFuncQwenImage21Loader")}
    finally:
        qfn._CONFIGS_DIR = _saved_cfg
    _real_mc = {_n: (_v["optional"]["model_config"][0], _v["optional"]["model_config"][1].get("hidden"))
                for _n, _v in _real.items()}
    check("with the SHIPPED configs every loader's model_config is hidden with its family's one preset (no dropdown)",
          all(len(c) == 1 and h is True for c, h in _real_mc.values()), f"-> {_real_mc}")
    # a QI-2.1 workflow saved with the published switch ([transformer, model_config, attention_backend, quality_enhance]) opens
    # with every value in its slot: widget values are stored by position, and model_config keeps slot 1.
    _qi_it = _real["QuantFuncQwenImage21Loader"]
    _old = ["qwen-image-2.1-quantfunc-int4-r128-i8sidecar.qfc.safetensors", "qwen-image-2.1-int4", "auto", False]
    _widgets = list(_qi_it["required"]) + list(_qi_it.get("optional", {}))
    _spec = {**_qi_it["required"], **_qi_it.get("optional", {})}
    _fits = [w == "transformer" or (isinstance(v, bool) if _spec[w][0] == "BOOLEAN" else v in _spec[w][0])
             for w, v in zip(_widgets, _old)]
    check("a saved QI-2.1 workflow's values land on the right widgets (model_config keeps slot 1)",
          _widgets[:4] == ["transformer", "model_config", "attention_backend", "quality_enhance"] and all(_fits),
          f"-> {list(zip(_widgets, _old))} fits={_fits}")
    _shifted = [w for w in _widgets if w != "model_config"]
    check("... and without the model_config slot they would not (the preset name would land in attention_backend)",
          _old[1] not in _spec[_shifted[1]][0], f"-> {_shifted[1]}")
    t1 = KreaL.INPUT_TYPES()["required"]["transformer"][0]
    check("transformer lists .safetensors FILES (and only those)",
          "fx-krea2-turbo-quantfunc-int4.safetensors" in t1 and "not-a-model.txt" not in t1)

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
    # node surface: required = the transformer alone (model_config is the hidden first optional since 2026-09-24, see
    # above; the manual block-count widget was REMOVED 2026-09-12 — residency is arena-managed); optional =
    # the runtime SESSION dials (attention_backend 2026-08-27; sol_tau; quality_enhance [the switch; the 2026-09-24
    # dropdown was withdrawn 2026-09-25]; step_cache +
    # block_cache). Every optional is a session knob (no create key / no rebuild) — sparse is
    # deliberately NOT among them (removed 2026-08-29, only the caches came back).
    _lit = LtxL.INPUT_TYPES()
    check("ltx node surface = transformer required + hidden model_config + session-dial optionals (no sparse)",
          list(_lit["required"].keys()) == ["transformer"]
          and list(_lit.get("optional", {}).keys()) == ["model_config", "attention_backend", "sol_tau", "step_cache",
                                                        "block_cache", "quality_enhance", "pinned_memory"]
          and "sparse" not in _lit.get("optional", {}),
          f"-> req={list(_lit['required'].keys())} opt={list(_lit.get('optional', {}).keys())}")
    # h3 node surface (same latent-duo + session dials shape as ltx; block-count removed 2026-09)
    _h3it = H3L.INPUT_TYPES()
    check("h3 node surface = transformer required + hidden model_config + session-dial optionals",
          list(_h3it["required"].keys()) == ["transformer"]
          and list(_h3it.get("optional", {}).keys()) == ["model_config", "attention_backend", "sol_tau", "step_cache",
                                                          "block_cache", "quality_enhance", "audio_enhance",
                                                          "pinned_memory"],
          f"-> req={list(_h3it['required'].keys())} opt={list(_h3it.get('optional', {}).keys())}")
    # (B) quality_enhance (user 2026-09-25): ONE switch on the four loaders, the same on every GPU. DEATH RULES: the switch sits
    #     in the widget slot of the earlier `quality` input (where the published loaders had it); the session sends only the
    #     engine's switch video_enhance (what each state does per model family is engine law: the plugin carries no
    #     implementation detail), never `quality` or a retired key (tests/_banned_terms.py, hashed); an API prompt saved with the
    #     earlier input maps best_quality -> ON, any other value -> OFF; the switch adds no create key.
    sys.path.insert(0, _HERE)
    import _banned_terms as _bt
    _FOUR = ("QuantFuncLTXLoader", "QuantFuncH3Loader", "QuantFuncKrea2Loader", "QuantFuncQwenImage21Loader")
    _sw = {n: qfn.NODE_CLASS_MAPPINGS[n].INPUT_TYPES() for n in _FOUR}
    _sw_bad = {n: list(it.get("optional", {})) for n, it in _sw.items()
               if it["optional"]["quality_enhance"][0] != "BOOLEAN"
               or it["optional"]["quality_enhance"][1].get("default") is not False
               or "quality" in it["optional"] or it.get("hidden", {}).get("quality") != ("STRING", {})}
    check("every loader: quality_enhance is a BOOLEAN, default OFF; the withdrawn dropdown is no input, its name is declared "
          "hidden as a (type, options) spec (an old API prompt reaches load())",
          not _sw_bad, f"-> {_sw_bad}")
    # _LOADER_LAYOUT (user 2026-09-28): on every loader the value inputs come first and the switches last (a workflow saved
    # with an earlier order is remapped on load by web/quantfunc_loader_layout.js).
    _order = {n: [(k, v[0]) for k, v in list(it["required"].items()) + list(it.get("optional", {}).items())]
              for n, it in _sw.items()}
    _late = {n: o for n, o in _order.items()
             if any(t != "BOOLEAN" for _, t in o[next((i for i, (_, t) in enumerate(o) if t == "BOOLEAN"), len(o)):])}
    check("every loader: every switch (BOOLEAN) comes after every value input",
          not _late and all(any(t == "BOOLEAN" for _, t in o) for o in _order.values()), f"-> {_late}")
    E = qfn._quality_enhance_on
    _row = (E(), E(False), E(True), E(None, "best_quality"), [E(None, q) for q in ("balance", "other", "")],
            E(True, "balance"), E(False, "best_quality"))
    check("the switch: default OFF; a saved earlier value maps best_quality -> ON and any other value -> OFF; the switch wins "
          "over a saved earlier value",
          _row == (False, False, True, True, [False] * 3, True, False), f"-> {_row}")
    check("no dropdown machinery or plugin-side keep table survives (no options, no GPU tier, no engine query, no "
          "VALIDATE_INPUTS, no create key, no number)",
          not any(hasattr(qfn, a) for a in ("_quality_fast_tier", "_quality_fast_cache", "_resolve_quality", "_validate_quality",
                                             "_quality_input", "_qi21_quality_input", "_qi21_resolve_quality", "_QUALITY_DROPPED",
                                             "_loaded_device_index", "_quality_create_opts"))
          and not [a for a in dir(qfn) if _bt.h(a) in _bt.NAMES]
          and not any(hasattr(qfn.NODE_CLASS_MAPPINGS[n], "VALIDATE_INPUTS") for n in _FOUR),
          "-> a dropdown helper is still present")
    from qfn_test_pkg import qf_modelpatcher as _qmp_q

    class _QProbe(_qmp_q.QFSessionModelMixin):
        pass
    _sess = {}
    for _on in (False, True):
        _qp = _QProbe()
        _qp.set_video_enhance(_on)
        _sess[_on] = _qp.residency_opts()
    _qa = _QProbe()
    _qa.set_video_enhance(False)
    _qa.set_attn_backend("flash")
    _qa_flash = _qa.residency_opts().get("attention_backend")
    _qa.set_attn_backend("auto")
    check("session: attention_backend is ALWAYS sent, auto included (the engine keeps its previous backend when the key "
          "is absent, so an omitted auto left a prior flash run's choice in force)",
          _sess[False].get("attention_backend") == "auto" and _qa_flash == "flash"
          and _qa.residency_opts().get("attention_backend") == "auto" and _qa.dial_opts().get("attention_backend") == "auto",
          f"-> unset={_sess[False].get('attention_backend')!r} flash={_qa_flash!r} back={_qa.residency_opts().get('attention_backend')!r}")
    # the add-on attention choice (user 2026-09-29) needs an engine that carries it (0.0.17+, one export marks it): an older
    # engine does not run it as chosen (0.0.16: Qwen-Image-2.1 native, MiniMax-H3 dense sage), so set_attn_backend - the one
    # setter every loader calls on the model it builds - refuses it loudly there, before any model loads; every other
    # choice, and that one on an engine that has it, is set and then sent as is (dial_opts does not second-guess it).
    _sym = getattr(_qmp_q, "_ATTN_ROUTE_SYMBOL", None)

    class _FakeLib:
        pass
    _ld0, _gate = _qmp_q.qfe.load_lib, {}
    try:
        for _has in (False, True):
            _fake = _FakeLib()
            if _has and _sym:
                setattr(_fake, _sym, lambda: 1)
            _qmp_q.qfe.load_lib = lambda _f=_fake: _f
            for _b in ("qfa", "auto", "flash"):   # the setter itself, fed as a loader feeds it (a loader that skips the
                _qa.set_attn_backend("auto")        # widget mapping still meets the refusal)
                try:
                    _qa.set_attn_backend(_b)
                    _gate[(_has, _b)] = _qa.dial_opts()["attention_backend"]
                except RuntimeError as _e:
                    _gate[(_has, _b)] = "refused: " + str(_e)
    finally:
        _qmp_q.qfe.load_lib = _ld0
        _qa.set_attn_backend("auto")
    _old = _gate[(False, "qfa")]
    check("attention_backend: every loader's setter refuses the add-on attention choice on an engine without it, naming "
          "the way out (auto, or an engine update); on an engine with it, and every other choice on any engine, the "
          "choice is set and sent as is",
          _sym == "quantfunc_attention_auto_route" and _old.startswith("refused: ") and "auto" in _old and "update" in _old
          and _gate[(True, "qfa")] == "qfa"
          and all(_gate[(h, "auto")] == "auto" and _gate[(h, "flash")] == "flash" for h in (False, True))
          and qfn._attn_backend_to_engine("fp16_native") == "native" and qfn._attn_backend_to_engine("qfa") == "qfa",
          f"-> {_sym!r} {_gate}")
    try:
        _QProbe().residency_opts()
        _unset = "sent"
    except RuntimeError:
        _unset = "refused"
    check("session: the engine's switch video_enhance is ALWAYS sent (both states) and nothing else decides quality: never "
          "`quality` or a retired key; a model its loader never configured refuses to begin",
          all(o.get("video_enhance") is on and not any(k == "quality" or _bt.h(k) in _bt.KEYS for k in o)
              for on, o in _sess.items())
          and _unset == "refused", f"-> {_sess} / unset {_unset}")
    # (B) the four loaders' user-visible text states only the speed / quality trade (user 「介绍上不要透露技术细节」,
    #     「四个 loader 的所有说明都改」): no technique word in any DESCRIPTION or tooltip — the words live hashed in
    #     _banned_terms (DESC_*). Widget NAMES are the user's and stay (a whole-word match, so the step_cache widget name
    #     is not a hit); the attention options are named, never explained (A4c).
    _texts = []
    for _n in _FOUR + ("QuantFuncNativeLoRA",):   # the LoRA node's text is user-visible too (A, round 2)
        _c = qfn.NODE_CLASS_MAPPINGS[_n]
        _texts.append((f"{_n}.DESCRIPTION", getattr(_c, "DESCRIPTION", "")))
        for _sec in ("required", "optional"):
            for _k, _v in _c.INPUT_TYPES().get(_sec, {}).items():
                if len(_v) > 1 and isinstance(_v[1], dict) and _v[1].get("tooltip"):
                    _texts.append((f"{_n}.{_k}", _v[1]["tooltip"]))
    _hits = [(w, x) for w, t in _texts for x in _bt.desc_hits(t) + _bt.term_hits(t) + _bt.mode_id_hits(t)]
    check("the four loaders' user-visible text carries no technique word (DESCRIPTIONs + every tooltip)",
          not _hits and len(_texts) > 20, f"-> {len(_texts)} texts, hits {_hits[:4]}")
    _qe_tips = {w: t for w, t in _texts if w.endswith(".quality_enhance")}
    check("every loader's quality_enhance tooltip states the measured trade (OFF keeps the subject and scene, details can "
          "differ; ON is the highest quality)",
          len(_qe_tips) == 4 and all("subject and scene stay the same" in t and "can differ" in t and "highest quality" in t
                                     for t in _qe_tips.values()), f"-> {_qe_tips}")
    _refused = []
    for _knob in ({"quality": "balance"}, {"video_enhance": False}):
        try:
            qfn.qfe._refuse_session_knobs_in_create(_knob)
            _refused.append(False)
        except RuntimeError:
            _refused.append(True)
    check("create boundary refuses `quality` and video_enhance (session knobs) in a create config",
          _refused == [True, True], f"-> {_refused}")
    from qfn_test_pkg import qf_modelpatcher as _qmp_tp
    check("the session carries the engine's switch (set_video_enhance) and no retired setter or mapper survives",
          hasattr(_qmp_tp.QFSessionModelMixin, "set_video_enhance")
          and not [a for a in dir(_qmp_tp.QFSessionModelMixin) + dir(qfn) if _bt.h(a) in _bt.NAMES]
          and not hasattr(_qmp_tp.QFSessionModelMixin, "set_quality") and not hasattr(qfn, "_apply_quality"),
          "-> a retired setter / mapper is still present")
    # (C) pinned_memory (user 2026-09-26 「pin能用 透出个开关让用户选择开启」): ONE load-time switch on the four loaders; default ON
    #     on LTX-2.5 (the user's decision: the plugin before it always turned it on there, and LTX-2.5 moves the most model data
    #     with little VRAM), OFF on the other three. It is a CREATE key (the engine's use_pinned_memory), so its two states are
    #     two cached pipelines: changing it reloads the model. It is the LAST optional input, so every saved workflow's widget
    #     values keep their slots.
    _PM_DEFAULT_ON = {"QuantFuncLTXLoader"}
    _pm_bad = {}
    for _n, _it in _sw.items():
        _opt = list(_it.get("optional", {}))
        _pms = _it["optional"].get("pinned_memory")
        if _opt[-1:] != ["pinned_memory"] or _pms[0] != "BOOLEAN" or _pms[1].get("default") is not (_n in _PM_DEFAULT_ON):
            _pm_bad[_n] = (_opt[-2:], _pms)
    check("every loader: pinned_memory is a BOOLEAN, its LAST optional input (saved widget values keep their slots), default ON "
          "on LTX-2.5 and OFF on MiniMax-H3, Krea-2 and Qwen-Image-2.1", not _pm_bad, f"-> {_pm_bad}")
    _pm_tips = {w: t for w, t in _texts if w.endswith(".pinned_memory")}
    check("every loader's pinned_memory tooltip states its own default (ON for LTX-2.5, OFF elsewhere), names LTX-2.5, and says "
          "that changing it reloads the model and that, once on, it stays on until ComfyUI restarts (the engine cannot turn "
          "it back off in a running process)",
          len(_pm_tips) == 4 and all("reloads the model" in t and "until ComfyUI restarts" in t and "LTX-2.5" in t
                                     and t.startswith("ON (default for LTX-2.5" if w.split(".")[0] in _PM_DEFAULT_ON
                                                      else "OFF (default)") for w, t in _pm_tips.items()),
          f"-> {_pm_tips}")
    # The 2026-09-13 technique-word rule still applies to prose and every other backend label. User 2026-09-29 explicitly
    # restored one backend (the add-on attention) as a user-facing choice, so that ONE exact option value is the exception
    # (tests/shipped_terms_test.py allows it only as that literal).
    import _banned_terms as _bt
    _backend_labels = qfn._ATTN_BACKEND_SM80PLUS + qfn._ATTN_BACKEND_ADDON + qfn._ATTN_BACKEND_SM75
    check("attention_backend choices expose the add-on attention but no other removed/banned backend",
          not any(_bt.h(c) in _bt.WORDS for c in _backend_labels if c != "qfa"),
          f"-> sm80+={qfn._ATTN_BACKEND_SM80PLUS} sm75={qfn._ATTN_BACKEND_SM75}")
    # attention_backend defaults (user 2026-09-29): auto on every loader and every GPU (the engine resolves auto per GPU).
    # The add-on attention is user-selectable on the SMs its library ships for (SM75/86/89/120, the engine's AUTO table),
    # and not offered on SM80/90/100/103. Only the offered list and the default move: a saved value stays as saved.
    import inspect as _insp
    import torch as _torch
    _dev0, _cap0 = qfn.qfmp.comfy.model_management.get_torch_device, _torch.cuda.get_device_capability
    _attn = {}
    try:
        qfn.qfmp.comfy.model_management.get_torch_device = lambda: _torch.device("cuda", 0)
        for _sm in ((7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0)):
            _torch.cuda.get_device_capability = lambda _d=None, _sm=_sm: _sm
            _attn[_sm] = {n: qfn.NODE_CLASS_MAPPINGS[n].INPUT_TYPES()["optional"]["attention_backend"] for n in _FOUR}
    finally:
        qfn.qfmp.comfy.model_management.get_torch_device, _torch.cuda.get_device_capability = _dev0, _cap0
    _expected_attn = {
        (7, 5): ["auto", "qfa", "fp16_native"],
        (8, 0): ["auto", "flash", "sage", "fp16_native"],
        (8, 6): ["auto", "qfa", "flash", "sage", "fp16_native"],
        (8, 9): ["auto", "qfa", "flash", "sage", "fp16_native"],
        (9, 0): ["auto", "flash", "sage", "fp16_native"],
        (10, 0): ["auto", "flash", "sage", "fp16_native"],
        (10, 3): ["auto", "flash", "sage", "fp16_native"],
        (12, 0): ["auto", "qfa", "flash", "sage", "fp16_native"],
    }
    check("attention_backend: every loader offers the add-on attention exactly on SM75/86/89/120 and defaults to auto",
          all(c[0] == _expected_attn[sm] and c[1]["default"] == "auto"
              for sm, per in _attn.items() for c in per.values()), f"-> {_attn}")
    _sig = {n: _insp.signature(qfn.NODE_CLASS_MAPPINGS[n].load).parameters["attention_backend"].default for n in _FOUR}
    check("attention_backend: every loader's load() defaults to auto (an API prompt without the input runs auto)",
          set(_sig.values()) == {"auto"}, f"-> {_sig}")
    # tests-32 2026-09-29: only MiniMax-H3 turns auto into flash on an engine without the H3-safe auto. The one other reader
    # of that export is qf_modelpatcher.engine_routes_attention, which the loaders' add-on attention check calls; the two
    # modules name the same export (a cross-repo name: a typo on either side silently mis-routes every engine).
    import glob as _glob
    _h3src = open(os.path.join(_PLUGIN, "qf_h3_modelpatcher.py"), encoding="utf-8").read()
    _capname = next((ln.split("=", 1)[1].split("#")[0].strip().strip("\"'") for ln in _h3src.splitlines()
                     if ln.startswith("_H3_SAFE_AUTO_SYMBOL = ")), None)
    _capref = sorted(os.path.basename(p) for p in _glob.glob(os.path.join(_PLUGIN, "*.py"))
                     if any(t in open(p, encoding="utf-8").read() for t in ("_H3_SAFE_AUTO_SYMBOL", _capname or "\0")))
    check("attention_backend: the auto-route export is read only by MiniMax-H3's auto and engine_routes_attention, under "
          "one name",
          bool(_capname) and _capref == ["qf_h3_modelpatcher.py", "qf_modelpatcher.py"]
          and getattr(_qmp_q, "_ATTN_ROUTE_SYMBOL", None) == _capname, f"-> {_capname!r} in {_capref}")
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

    # Every deferred family captures one Comfy device and reuses it for both
    # the logical patcher and the Prepared/materialized cache identity.
    _device2 = qfn.qfmp.torch.device("cuda", 2)
    _device_calls = [0]
    _original_get_torch_device = qfn.qfmp.comfy.model_management.get_torch_device
    _original_unet_offload_device = qfn.qfmp.comfy.model_management.unet_offload_device

    def _device2_once():
        _device_calls[0] += 1
        return _device2

    qfn.qfmp.comfy.model_management.get_torch_device = _device2_once
    qfn.qfmp.comfy.model_management.unet_offload_device = lambda: qfn.qfmp.torch.device("cpu")
    try:
        _device_cases = (
            ("h3", lambda: H3L.load(
                "fx-minimax-h3-quantfunc-int4.safetensors", "minimax-h3-fl2va")[0]),
            ("krea2", lambda: KreaL.load(
                "fx-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4")[0]),
            ("ltx-av", lambda: LtxL.load(_allin_name, "ltx2-2.5-22b")[0]),
        )
        for _label, _build_device_case in _device_cases:
            _calls_before = _device_calls[0]
            _creates_before = len(create_devices)
            _device_patcher = _build_device_case()
            _ = _device_patcher.model._qf.lib
            check(f"device=2 {_label}: one captured device drives logical + native identity",
                  _device_calls[0] == _calls_before + 1
                  and _device_patcher.load_device == _device2
                  and _device_patcher.model._qf._ckey[1] == 2
                  and create_devices[_creates_before:] == [2],
                  f"-> calls={_device_calls[0] - _calls_before} "
                  f"load={_device_patcher.load_device} key={_device_patcher.model._qf._ckey}")
    finally:
        qfn.qfmp.comfy.model_management.get_torch_device = _original_get_torch_device
        qfn.qfmp.comfy.model_management.unet_offload_device = _original_unet_offload_device

    # ── positive path: the all-in file alone loads (AV via the packed audio connector) ──
    out_ltx = LtxL.load(_allin_name, "ltx2-2.5-22b")[0]
    _lk = lambda **kw: getattr(LtxL.load(_allin_name, "ltx2-2.5-22b", **kw)[0].model, "_video_enhance", "MISSING")
    _ltx_k = [_lk(), _lk(quality_enhance=True), _lk(quality="best_quality"), _lk(quality="balance")]
    check("ltx2 load(): OFF by default, ON when set; a saved earlier best_quality runs ON, any other value runs OFF",
          _ltx_k == [False, True, True, False], f"-> {_ltx_k}")
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
    check("ltx2 staged: config-complete + transformer_2 removed (single-expert)",
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
    # (C) pinned_memory through the REAL load() of each family: ON puts use_pinned_memory in the create config; OFF leaves the
    #     config exactly {denoise_only: true}. A load WITHOUT the input (an API prompt) gets the loader's own default: ON on
    #     LTX-2.5 (its create config as before the switch), OFF on the others (theirs as before). A LoRA rebuild keeps it.
    QiL = qfn.NODE_CLASS_MAPPINGS["QuantFuncQwenImage21Loader"]()
    _pm_loads = {"ltx2": (LtxL, _allin_name, "ltx2-2.5-22b"),
                 "minimax-h3": (H3L, "fx-minimax-h3-quantfunc-int4.safetensors", "minimax-h3-fl2va"),
                 "krea2": (KreaL, "fx-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4"),
                 "qwenimage21": (QiL, "fx-qwen-image-2.1-quantfunc-int4.safetensors", "qwen-image-2.1-int4")}

    def _pm_cfg(node, xfm, preset, **kw):
        out = node.load(xfm, preset, **kw)[0]
        _ = out.model._qf.lib   # first touch: the create (or the cached pipeline of the same config)
        return json.loads(out.model._qf._ckey[-1]), out
    _pm_rows, _pm_on = {}, {}
    for _fam, (_node, _xfm, _preset) in _pm_loads.items():
        _dflt = _pm_cfg(_node, _xfm, _preset)[0]
        _on, _pm_on[_fam] = _pm_cfg(_node, _xfm, _preset, pinned_memory=True)
        _pm_rows[_fam] = (_dflt, _on, _pm_cfg(_node, _xfm, _preset, pinned_memory=False)[0])
    _PM_ON, _PM_OFF = {"denoise_only": True, "use_pinned_memory": True}, {"denoise_only": True}
    # (C2) a LoRA that cannot be loaded (user 2026-09-30 「lora无法加载的时候让用户自己看 …/docs/lora-convert」): ComfyUI's own
    #      LoRA load (comfy.sd.load_lora_for_models, which LoraLoader / LoraLoaderModelOnly and other LoRA nodes call)
    #      patches torch weights, which a QuantFunc model never computes with. On every family it prints ONE warning that
    #      points at the page, next to ComfyUI's own per-key lines. A patch that is not a LoRA load prints none.
    import logging as _lg
    _LORA_URL = "https://www.quantfunc.com/docs/lora-convert"

    class _LoraHint(_lg.Handler):
        def __enter__(self):
            self.msgs = []
            qfn.qfmp._log.addHandler(self)
            return self.msgs

        def __exit__(self, *_exc):
            qfn.qfmp._log.removeHandler(self)

        def emit(self, record):
            if _LORA_URL in record.getMessage():
                self.msgs.append(record.getMessage())
    import comfy.sd as _csd
    _fake_lora = {"diffusion_model.blocks.0.attn.to_q.lora_A.weight": _torch.zeros(4, 8),
                  "diffusion_model.blocks.0.attn.to_q.lora_B.weight": _torch.zeros(8, 4)}
    _hint = {}
    for _fam, (_node, _xfm, _preset) in _pm_loads.items():
        _mp = _node.load(_xfm, _preset)[0]
        with _LoraHint() as _got:
            _csd.load_lora_for_models(_mp, None, dict(_fake_lora), 0.0, 1.0)   # strength 0 on the model: CLIP only
            _zero_n = len(_got)
            _csd.load_lora_for_models(_mp, None, dict(_fake_lora), 1.0, 0.0)
            _lora_n = len(_got)
            _mp.clone().add_patches({})   # not a LoRA load
        _hint[_fam] = (_zero_n, _lora_n, len(_got), _got[:1])
    check("LoRA hint: ComfyUI's own LoRA load on every family (LTX-2.5, MiniMax-H3, Krea-2, Qwen-Image-2.1) prints "
          "exactly ONE warning pointing at the LoRA page; a load at model strength 0 (CLIP only) and a patch that is "
          "not a LoRA load print none",
          len(_hint) == 4 and all(z == 0 and n == 1 and total == 1 for z, n, total, _m in _hint.values()),
          f"-> {_hint}")
    # (D) Windows without Developer Mode, the model on another drive or share (hotfix 2026-09-29, the user's 5060 Ti on an
    #     SMB share): a symlink is refused (WinError 1314) and a hardlink cannot cross volumes (EXDEV / WinError 17).
    #     Krea-2 and Qwen-Image-2.1 never link: the create names the picked file as its transformer_path, the package
    #     holds configs only, and two weight files are two pipelines. LTX-2.5 and MiniMax-H3 still link on this engine
    #     (LTX-2.5 reads its connectors from the package; MiniMax-H3's folded checkpoint needs transformer/config.json
    #     beside the weights, see the h3 arm) and, when they cannot, say what to do; never a copy.
    import errno as _errno
    from unittest import mock as _mock
    _far = tempfile.mkdtemp(prefix="qf_other_volume_")
    folder_paths.add_model_folder_path("diffusion_models", _far)
    _far_loads = {"krea2": (KreaL, "far-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4"),
                  "krea2-b": (KreaL, "far-b-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4"),
                  "qwenimage21": (QiL, "far-qwen-image-2.1-quantfunc-int4.safetensors", "qwen-image-2.1-int4")}
    for _xfm in [x for _n, x, _p in _far_loads.values()] + ["far-minimax-h3-quantfunc-int4.safetensors"]:
        with open(os.path.join(_far, _xfm), "wb") as _fh:
            _fh.write(b"\0" * 16)
    _link_calls = []

    def _refused_symlink(*a, **k):
        _link_calls.append("symlink")
        raise OSError(1314, "A required privilege is not held by the client")

    def _cross_volume_link(*a, **k):
        _link_calls.append("link")
        raise OSError(_errno.EXDEV, "Invalid cross-device link")
    _far_got, _ltx_err, _h3_err = {}, None, None
    with _mock.patch("os.symlink", _refused_symlink), _mock.patch("os.link", _cross_volume_link):
        for _fam, (_node, _xfm, _preset) in _far_loads.items():
            try:
                _o = _node.load(_xfm, _preset)[0]
                _ = _o.model._qf.lib
                _far_got[_fam] = _o.model._qf._ckey
            except Exception as _e:  # noqa: BLE001 - a refusal is this arm's FAIL, shown in its detail
                _far_got[_fam] = f"{type(_e).__name__}: {_e}"
        _far_calls = list(_link_calls)
        try:
            LtxL.load("fx-ltx-2.5-quantfunc-4bit.safetensors", "ltx2-2.5-22b")
        except RuntimeError as _e:
            _ltx_err = str(_e)
        try:
            H3L.load("far-minimax-h3-quantfunc-int4.safetensors", "minimax-h3-fl2va")
        except RuntimeError as _e:
            _h3_err = str(_e)

    def _far_ok(ck, xfm):
        return (isinstance(ck, tuple) and os.path.realpath(ck[2] or "") == os.path.realpath(os.path.join(_far, xfm))
                and os.path.isfile(os.path.join(ck[0], "model_index.json"))
                and not any(f.endswith(".safetensors") for _r, _d, _fs in os.walk(ck[0]) for f in _fs))
    check("links impossible (Windows 1314 / cross-volume EXDEV): Krea-2 and Qwen-Image-2.1 load without linking; the "
          "create names the picked file as transformer_path and the package holds configs only",
          not _far_calls and all(_far_ok(_far_got[f], x) for f, (_n, x, _p) in _far_loads.items()),
          f"-> calls={_far_calls} got={ {f: (v if isinstance(v, str) else v[2]) for f, v in _far_got.items()} }")
    _fk, _fkb = _far_got.get("krea2"), _far_got.get("krea2-b")
    check("two weight files of one family are two pipelines (the transformer file is part of the create key), each with "
          "its own package dir (the engine writes its VRAM measurement cache there: per weight file, never the plugin's)",
          isinstance(_fk, tuple) and isinstance(_fkb, tuple) and _fk[2] != _fkb[2] and _fk[0] != _fkb[0]
          and not os.path.commonpath([_fk[0], os.path.dirname(os.path.abspath(qfn.__file__))]) ==
          os.path.dirname(os.path.abspath(qfn.__file__)), f"-> {_fk} / {_fkb}")
    _stage_root = os.path.join(folder_paths.get_temp_directory(), "qf_native_stage")
    _copied = [os.path.join(r, f) for r, _d, fs in os.walk(_stage_root) for f in fs
               if f.endswith(".safetensors") and not os.path.islink(os.path.join(r, f))]
    check("links impossible: LTX-2.5 and MiniMax-H3 refuse with what to do (Developer Mode, or the model on the drive of "
          "ComfyUI's temp folder; a QuantFunc engine update lifts it) and copy nothing",
          all(e is not None and "Developer Mode" in e and "drive" in e and "LTX-2.5" in e and "MiniMax-H3" in e
              for e in (_ltx_err, _h3_err)) and not _copied,
          f"-> ltx={(_ltx_err or 'no refusal')[:160]} h3={(_h3_err or 'no refusal')[:160]} copied={_copied}")
    # (E) 0.0.08 + engine 0.0.17 (#777; engine-kv 525a39424): an engine exporting quantfunc_weight_paths reads every
    #     family's weights from paths - LTX-2.5's connectors from connectors_path, MiniMax-H3's fold count from the
    #     package's transformer/config.json - so with links impossible LTX-2.5 and MiniMax-H3 load link-free too: the
    #     package holds configs only, the create names the picked file as transformer_path (and LTX-2.5's connectors
    #     source as connectors_path), and LTX-2.5 is still the joint audio+video model (its discriminant reads that source).
    _wp_lib = types.SimpleNamespace(quantfunc_weight_paths=lambda: 1)
    _lib0 = qfn.qfe.load_lib
    qfn.qfe.load_lib = lambda *a, **k: _wp_lib
    _wp_got, _link_calls[:] = {}, []
    _wp_loads = {"minimax-h3": (H3L, "far-minimax-h3-quantfunc-int4.safetensors", "minimax-h3-fl2va"),
                 "ltx2": (LtxL, "fx-ltx-2.5-quantfunc-4bit.safetensors", "ltx2-2.5-22b")}
    try:
        with _mock.patch("os.symlink", _refused_symlink), _mock.patch("os.link", _cross_volume_link):
            for _fam, (_node, _xfm, _preset) in _wp_loads.items():
                try:
                    _o = _node.load(_xfm, _preset)[0]
                    _ = _o.model._qf.lib
                    _wp_got[_fam] = (_o.model._qf._ckey, json.loads(_o.model._qf._ckey[-1]), type(_o.model).__name__)
                except Exception as _e:  # noqa: BLE001 - a refusal is this arm's FAIL, shown in its detail
                    _wp_got[_fam] = f"{type(_e).__name__}: {_e}"
    finally:
        qfn.qfe.load_lib = _lib0
    _wp_calls = list(_link_calls)

    def _wp_ok(fam, xfm_path, conn=None):
        got = _wp_got.get(fam)
        if not isinstance(got, tuple):
            return False
        ck, cfg, _cls = got
        return (os.path.realpath(ck[2] or "") == os.path.realpath(xfm_path)
                and os.path.isfile(os.path.join(ck[0], "model_index.json"))
                and not any(f.endswith(".safetensors") for _r, _d, _fs in os.walk(ck[0]) for f in _fs)
                and cfg.get("denoise_only") is True
                and (conn is None or os.path.realpath(cfg.get("connectors_path") or "") == os.path.realpath(conn)))
    check("an engine that reads weight paths (quantfunc_weight_paths), links impossible: MiniMax-H3 and LTX-2.5 load "
          "link-free - configs-only package, the picked file as transformer_path, LTX-2.5's connectors source as "
          "connectors_path - and LTX-2.5 is still the joint audio+video model",
          not _wp_calls and _wp_ok("minimax-h3", os.path.join(_far, "far-minimax-h3-quantfunc-int4.safetensors"))
          and _wp_ok("ltx2", os.path.join(dm, "fx-ltx-2.5-quantfunc-4bit.safetensors"),
                     os.path.join(dm, "fx-ltx25-connectors.safetensors"))
          and isinstance(_wp_got.get("ltx2"), tuple) and _wp_got["ltx2"][2] == "QFLTXAVModel",
          f"-> calls={_wp_calls} got={ {f: (v if isinstance(v, str) else (v[0][2], v[1], v[2])) for f, v in _wp_got.items()} }")
    _pp = qfn._prepare_params("one-package", {"denoise_only": True}, 0, "test-not-a-key", "/w/a.safetensors")
    check("the real create params carry the package dir and the transformer file (quantfunc_create's model_dir and "
          "transformer_path)", _pp.model_dir == b"one-package" and _pp.transformer_path == b"/w/a.safetensors",
          f"-> {_pp.model_dir!r} {_pp.transformer_path!r}")
    _r1 = qfn._engine_recipe("one-package", {"denoise_only": True}, 0, "/w/a.safetensors")[1]
    _r2 = qfn._engine_recipe("one-package", {"denoise_only": True}, 0, "/w/b.safetensors")[1]
    _r0 = qfn._engine_recipe("one-package", {"denoise_only": True}, 0)[1]
    check("the real engine cache key carries the transformer file: one package, two weight files -> two keys (and no "
          "file -> a third)", len({_r0, _r1, _r2}) == 3 and "/w/a.safetensors" in _r1, f"-> {_r1}")
    check("pinned_memory ON: every family's create config carries use_pinned_memory=true and nothing else changes; OFF: the "
          "key is absent and the config is {denoise_only: true}; the default (the input absent): ON for LTX-2.5, OFF for "
          "MiniMax-H3, Krea-2 and Qwen-Image-2.1",
          all(_on == _PM_ON and _off == _PM_OFF and _dflt == (_PM_ON if _fam == "ltx2" else _PM_OFF)
              for _fam, (_dflt, _on, _off) in _pm_rows.items()), f"-> {_pm_rows}")
    from qfn_test_pkg import qf_modelpatcher as _qmp_pm
    _pm_rb = _qmp_pm.rebuild_of(_pm_on["krea2"])([])
    _ = _pm_rb.model._qf.lib
    check("a LoRA rebuild of a pinned_memory=ON model keeps the switch (the same create config)",
          json.loads(_pm_rb.model._qf._ckey[-1]) == _pm_rows["krea2"][1], f"-> {_pm_rb.model._qf._ckey[-1]}")
    try:
        qfn.qfe._refuse_session_knobs_in_create({"denoise_only": True, "use_pinned_memory": True})
        _pm_create_ok = True
    except RuntimeError:
        _pm_create_ok = False
    check("use_pinned_memory is a create key the create boundary accepts (not a session knob)", _pm_create_ok)
    # quality_enhance through the REAL load() (behaviour, not a signature read — a loader wrapped by another input layer keeps
    # it): OFF (default) / ON; a saved earlier value by name when the switch is absent (best_quality -> ON, anything else ->
    # OFF); the switch wins over a saved dropdown value. Krea-2 likewise.
    _hk = lambda **kw: getattr(H3L.load("fx-minimax-h3-quantfunc-int4.safetensors", "minimax-h3-fl2va", **kw)[0].model,
                               "_video_enhance", "MISSING")
    _kk = lambda **kw: getattr(KreaL.load("fx-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4", **kw)[0].model,
                               "_video_enhance", "MISSING")
    _hks = [_hk(), _hk(quality_enhance=True), _hk(quality_enhance=False), _hk(quality="best_quality"),
            _hk(quality="best_quality", quality_enhance=False), _hk(quality="other")]
    _kks = [_kk(), _kk(quality_enhance=True), _kk(quality="balance")]
    check("h3 load(): OFF by default, ON when set; a saved earlier value maps best_quality -> ON, any other -> OFF; the switch wins",
          _hks == [False, True, False, True, False, False], f"-> {_hks}")
    check("krea2 load(): OFF by default, ON when set; a saved other value runs OFF", _kks == [False, True, False], f"-> {_kks}")
    # MiniMax-H3 keeps the LINK on this engine (#777, measured on 远程-linux-d): the release checkpoint is adaln-folded and
    # its sealed resq_slot_count counts the 50 folded slots; the engine subtracts qf_adaln_fold_dropped_resq, which it
    # reads ONLY from the config.json beside the weights path (the package's transformer/config.json). Loaded through
    # transformer_path, that is the model file's own folder: no correction, "armed-slot count (200) != ... (250)".
    _hmd, _htp = out_h3.model._qf._ckey[0], out_h3.model._qf._ckey[2]
    try:
        _hcfg_fold = json.load(open(os.path.join(_hmd, "transformer", "config.json"), encoding="utf-8")).get("qf_adaln_fold_dropped_resq")
    except (OSError, ValueError):
        _hcfg_fold = None
    check("h3: config-complete package with the weights LINKED as transformer/model.safetensors beside the transformer "
          "config.json that carries qf_adaln_fold_dropped_resq (the engine reads it next to the weights); no "
          "transformer_path; transformer_2 removed",
          all(os.path.isfile(os.path.join(_hmd, q)) for q in
              ("model_index.json", "transformer/config.json", "vae/config.json"))
          and os.path.realpath(os.path.join(_hmd, "transformer", "model.safetensors")).endswith(
              "fx-minimax-h3-quantfunc-int4.safetensors")
          and isinstance(_hcfg_fold, int) and _hcfg_fold > 0 and _htp is None
          and not os.path.exists(os.path.join(_hmd, "transformer_2")), f"-> tp={_htp} fold={_hcfg_fold}")
    # a saved workflow's model_config of another family / of no family: not this family's preset -> refused, naming it
    try:
        KreaL.load("fx-krea2-turbo-quantfunc-int4.safetensors", "fx-ltx")
        check("cross-family saved model_config on the Krea-2 node refused", False, "-> no exception")
    except RuntimeError as e:
        check("cross-family saved model_config on the Krea-2 node refused",
              "not a krea2 model config" in str(e) and "krea2-turbo-int4" in str(e), f"-> {str(e)[:120]}")
    try:
        LtxL.load("fx-ltx-2.5-quantfunc-4bit.safetensors", "fx-alien")
        check("unknown-family saved model_config refused", False, "-> no exception")
    except RuntimeError as e:
        check("unknown-family saved model_config refused", "not a ltx2 model config" in str(e), f"-> {str(e)[:120]}")
    for evil_cfg in ("../krea2-turbo-int4", "a/b", "..", ""):
        try:
            KreaL.load("fx-krea2-turbo-quantfunc-int4.safetensors", evil_cfg)
            check(f"model_config refuses {evil_cfg!r}", False, "-> loaded!")
        except RuntimeError:
            check(f"model_config refuses {evil_cfg!r}", True)
    # model_config values (user 2026-09-24): an API prompt that omits it gets the family's one preset; a saved value is
    # honoured when it is one of the family's presets and refused otherwise, naming what ships; with two presets and no
    # value the loader refuses instead of guessing (the dropdown shows again then — see the surface arm).
    with open(os.path.join(dm, "mc-krea2-turbo-quantfunc-int4.safetensors"), "wb") as fh:
        fh.write(b"\0" * 16)
    _mc_a = KreaL.load("mc-krea2-turbo-quantfunc-int4.safetensors")[0]            # the API prompt omits model_config
    _mc_b = KreaL.load("mc-krea2-turbo-quantfunc-int4.safetensors", model_config="krea2-turbo-int4")[0]
    _ = (_mc_a.model._qf.lib, _mc_b.model._qf.lib)                                 # materialize both staged packages
    check("an API prompt without model_config loads the family's one preset (the same staged package as naming it)",
          _mc_a.model._qf._ckey == _mc_b.model._qf._ckey, f"-> {_mc_a.model._qf._ckey} vs {_mc_b.model._qf._ckey}")
    try:
        LtxL.load("fx-ltx-2.5-quantfunc-4bit.safetensors")
        check("no model_config with two presets for the family: refused, never a guess", False, "-> no exception")
    except RuntimeError as e:
        check("no model_config with two presets for the family: refused, never a guess",
              "cannot choose" in str(e) and "fx-ltx" in str(e) and "ltx2-2.5-22b" in str(e), f"-> {str(e)[:120]}")
    try:
        KreaL.load("mc-krea2-turbo-quantfunc-int4.safetensors", model_config="no-such-preset")
        check("an unknown saved model_config is refused, naming the shipped preset", False, "-> no exception")
    except RuntimeError as e:
        check("an unknown saved model_config is refused, naming the shipped preset",
              "no-such-preset" in str(e) and "krea2-turbo-int4" in str(e), f"-> {str(e)[:120]}")
    # the family guard (defense in depth): the listing says krea2, the manifest read says ltx2
    _listing = qfn._model_config_choices
    qfn._model_config_choices = lambda family=None: ["fx-ltx"]
    try:
        KreaL.load("mc-krea2-turbo-quantfunc-int4.safetensors", model_config="fx-ltx")
        check("a preset whose manifest family changed after the listing is refused (family guard)", False,
              "-> no exception")
    except RuntimeError as e:
        check("a preset whose manifest family changed after the listing is refused (family guard)",
              "declares family" in str(e), f"-> {str(e)[:120]}")
    finally:
        qfn._model_config_choices = _listing
    # ── 3) single-expert staging + denoise_only create cfg + the shared ledger / D3 arms (Krea-2; the Wan dual
    #      loader these arms used to ride left the release — user scope 2026-09-24) ──
    try:
        with open(os.path.join(dm, "st-krea2-turbo-quantfunc-int4.safetensors"), "wb") as fh:
            fh.write(b"\0" * 16)
        out = KreaL.load("st-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4")[0]
        check("krea2 loader returns one QFModelPatcher", type(out).__name__ == "QFModelPatcher")
        _module_size = qfn.qfmp.comfy.model_management.module_size
        n0 = len(creates)
        check("the logical output uses the ordinary Torch ledger; native bytes stay on dependencies",
              out.model_size() == _module_size(out.model)
              and out.loaded_size() == out.model.model_loaded_weight_memory
              and len(creates) == n0, f"-> model_size={out.model_size()}")
        _ = out.model._qf.lib          # first touch materializes
        check("create deferred until first touch", len(creates) == n0 + 1)
        md = out.model._qf._ckey[0]
        cfg = creates[-1]
        check("create cfg carries denoise_only and no LoRA", cfg.get("denoise_only") is True and "lora" not in cfg,
              f"-> {cfg}")
        check("staged dir is config-complete and single-expert",
              all(os.path.isfile(os.path.join(md, p)) for p in
                  ("model_index.json", "transformer/config.json", "vae/config.json"))
              and not os.path.exists(os.path.join(md, "transformer_2")))
        tp = out.model._qf._ckey[2]
        check("no weights in the package; the create names the PICKED file as its transformer_path",
              not os.path.lexists(os.path.join(md, "transformer", "model.safetensors"))
              and os.path.realpath(tp or "") == os.path.realpath(os.path.join(dm, "st-krea2-turbo-quantfunc-int4.safetensors")),
              f"-> {tp}")

        # ZERO-LATENT GUARD (user black-video class): ONE shared mechanism, N users — behavior-tested through the
        # shared helper per family tag, and its call-before-engine wiring asserted structurally for every family
        # module that calls it (the image families Krea-2 / QI-2.1 do not).
        import torch as _t2, re as _re2
        _zero5 = _t2.zeros(1, 16, 3, 8, 8)
        _noise5 = _t2.randn(1, 16, 3, 8, 8)
        for _tag in ("LTX", "LTX-AV", "H3"):
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
        for _mod, _tags in (("qf_ltx_modelpatcher.py", 1), ("qf_h3_modelpatcher.py", 1)):
            _src = open(_os.path.join(_PLUGIN, _mod), encoding="utf-8").read()
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
        check("single-expert staging", False, f"-> raised {type(e).__name__}: {e}")

    # a NON-conforming file name for the preset must be refused loud (file_hints mechanism)
    try:
        KreaL.load("other.safetensors", "krea2-turbo-int4")
        check("file_hints refuses a non-conforming transformer", False, "-> loaded!")
    except RuntimeError as e:
        check("file_hints refuses a non-conforming transformer", "does not look like" in str(e))

    # memory_required (D3/D4): comfy's eviction decisions call it THROUGH sampler_helpers.estimate_memory —
    # `memory_required(shape, cond_shapes=cond_shapes)` (KEYWORD). This arm (a) calls EXACTLY like the real call
    # site, (b) asserts signature compatibility against comfy's own BaseModel.memory_required so future comfy drift
    # goes red here, (c) asserts the D3 honesty properties: geometry-proportional, monotonic, and nowhere near a
    # torch activation estimate (the inter-stage eviction thrash). Krea-2 rides comfy's [B,C,T=1,H,W] latent.
    import inspect as _insp
    from comfy.model_base import BaseModel as _CB
    _base_params = [q for q in _insp.signature(_CB.memory_required).parameters
                    if q != "self"]
    _ours = _insp.signature(type(out.model).memory_required)
    try:
        _ours.bind(out.model, [1, 16, 1, 128, 128],
                   **{q: {} for q in _base_params[1:]})
        _sig_ok = True
    except TypeError:
        _sig_ok = False
    check("memory_required signature accepts every BaseModel caller form (D4)", _sig_ok,
          f"-> base params {_base_params} vs ours {list(_ours.parameters)}")
    _shape = [2, 16, 1, 128, 128]      # sampler_helpers doubles batch for cfg
    _conds = {"c_crossattn": [[1, 512, 30720]]}
    _mr = out.model.memory_required(_shape, cond_shapes=_conds)   # the REAL call form
    _mr_small = out.model.memory_required([1, 16, 1, 8, 8], cond_shapes={})
    _mr_big = out.model.memory_required([2, 16, 1, 256, 256], cond_shapes=_conds)
    check("memory_required geometry-proportional + monotonic (D3)",
          _mr_small < _mr < _mr_big, f"-> {_mr_small} < {_mr} < {_mr_big}")
    check("memory_required stays far below a torch activation estimate (thrash fix preserved)",
          _mr <= 1 << 30, f"-> {_mr}")
    # #716: while NO pipeline exists for these weights, comfy's OWN estimate (the next class after the mixin in the
    # real family MRO) is the floor; the hot model above never takes it (both ways, same shape).
    open(os.path.join(dm, "cold-krea2-turbo-quantfunc-int4.safetensors"), "wb").write(b"\0" * 16)
    _cold = KreaL.load("cold-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4")[0]
    _torch_est = int(super(qfn.qfmp.QFSessionModelMixin, _cold.model).memory_required(_shape, cond_shapes=_conds))
    _cold_side = _cold.model._qf_comfy_side_bytes(_shape, _conds)
    _n_cold = len(creates)
    _mr_cold = _cold.model.memory_required(_shape, cond_shapes=_conds)
    check("#716: a COLD model asks comfy's own estimate as the floor (no create); the hot one does not",
          _mr_cold == max(_torch_est, _cold_side) and len(creates) == _n_cold and _torch_est > _mr,
          f"-> cold={_mr_cold} torch={_torch_est} side={_cold_side} hot={_mr} creates+={len(creates) - _n_cold}")

    # ── 4) containment: traversal/absolute names cannot escape the model roots ──
    for evil in ("../../../../etc/passwd", "/etc/passwd", "fx-krea2-turbo-quantfunc-int4.safetensors/../../x"):
        try:
            qfn._resolve_transformer(evil)
            check(f"containment refuses {evil!r}", False, "-> resolved!")
        except Exception:  # noqa: BLE001 — comfy raises its own error type
            check(f"containment refuses {evil!r}", True)
    try:
        ok = qfn._resolve_transformer("fx-krea2-turbo-quantfunc-int4.safetensors") == os.path.join(dm, "fx-krea2-turbo-quantfunc-int4.safetensors")
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
        import struct as _lst
        _lh = json.dumps({"blocks.0.attn.to_q.lora_A.weight":
                          {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]}}).encode()
        for f in ("a.safetensors", "b.safetensors"):
            open(os.path.join(lora_dir, f), "wb").write(_lst.pack("<Q", len(_lh)) + _lh + b"\0\0")
        qfn._lora_choices = lambda: ["a.safetensors", "b.safetensors"]
        qfn._resolve_lora = lambda n: os.path.join(lora_dir, n)
        LoraNode = qfn.NODE_CLASS_MAPPINGS["QuantFuncNativeLoRA"]()

        base = KreaL.load("fx-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4")[0]
        shifted = ModelSamplingSD3().patch(base, 3.0)[0]     # upstream comfy patch on the loader output
        check("comfy patches still apply to a QuantFunc output (clone path intact)",
              "model_sampling" in shifted.object_patches)
        from qfn_test_pkg import qf_modelpatcher as _qmp2
        outA = LoraNode.apply(shifted, "a.safetensors", 0.8)[0]
        stA = _qmp2.lora_stack_of(outA)
        check("a LoRA node stacks target=all with its strength (no target widget)",
              len(stA) == 1 and stA[0]["target"] == "all" and stA[0]["scale"] == 0.8, f"-> {stA}")
        check("a LoRA rebuild keeps the upstream comfy patch (adopt)",
              "model_sampling" in outA.object_patches)
        outAB = LoraNode.apply(outA, "b.safetensors", 0.3)[0]
        stAB = _qmp2.lora_stack_of(outAB)
        check("a second node accumulates FUNCTIONALLY (no loader-cache mutation)",
              [(os.path.basename(e["path"]), e["scale"]) for e in stAB] == [("a.safetensors", 0.8), ("b.safetensors", 0.3)]
              and outAB.model._qf.lora_set() == stAB and _qmp2.lora_stack_of(base) == []
              and base.model._qf.lora_set() == [],
              f"-> wire={stAB} loader_stack={_qmp2.lora_stack_of(base)}")
        # the one-format refusal of this node still raises, and prints the same ONE LoRA-page warning
        _kh = json.dumps({"lora_unet_blocks_0_attn_to_q.lora_down.weight":
                          {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]}}).encode()
        open(os.path.join(lora_dir, "k.safetensors"), "wb").write(_lst.pack("<Q", len(_kh)) + _kh + b"\0\0")
        _k_err = ""
        with _LoraHint() as _k_got:
            try:
                LoraNode.apply(base, "k.safetensors", 1.0)
            except RuntimeError as _e:
                _k_err = str(_e)
        check("LoRA hint: the native LoRA node refuses a file it cannot load (raises as before) and prints ONE "
              "LoRA-page warning", "kohya" in _k_err and len(_k_got) == 1, f"-> {_k_err[:80]!r} {_k_got}")
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
            # _retire_handle reads the loaded-image identity (lib) and the retained native resource (the host-vram
            # prepared-entry check); a bare destroy()-only object raised AttributeError before the gate was reached.
            lib = "LIB"
            resource = None

            def destroy(self):
                _destroyed.append(True)
        _shared_eng = _Eng()
        qfn._PIPELINE_CACHE["shared-ck"] = _shared_eng
        qfn._PIPELINE_MODELS["shared-ck"] = [_wr.ref(_mine), _wr.ref(_theirs)]
        _stale = qfn._retire_handle("shared-ck", _Eng(), _w1, reason="stale arm")
        _r1 = qfn._retire_handle("shared-ck", _shared_eng, _w1, reason="arm")
        _kept = (_stale is False and _r1 is False
                 and qfn._PIPELINE_CACHE.get("shared-ck") is _shared_eng
                 and not _destroyed)
        qfn._PIPELINE_MODELS["shared-ck"] = [_wr.ref(_mine)]          # only OUR consumer left
        _r2 = qfn._retire_handle("shared-ck", _shared_eng, _w1, reason="arm")
        _dropped = (_r2 is True and "shared-ck" not in qfn._PIPELINE_CACHE
                    and _destroyed == [True])
        # keep_binding semantics (release path): binding list survives a granted retire
        _kb_eng = _Eng()
        qfn._PIPELINE_CACHE["kb-ck"] = _kb_eng
        qfn._PIPELINE_MODELS["kb-ck"] = [_wr.ref(_mine)]
        _r3 = qfn._retire_handle("kb-ck", _kb_eng, _w1, keep_binding=True, reason="arm")
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
            _src = open(_p, encoding="utf-8").read()
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
        check("LoRA chain + substrate arm", False, f"-> raised {type(e).__name__}: {e}")

    # ── 5b) RUNTIME LoRA on a single-expert family (user rule 2026-09-24 「换 LoRA 也不重建」) ──
    # The create carries the weights only, so every LoRA set of one model shares ONE pipeline, and each consumer's
    # chained set goes on in place with ONE quantfunc_pipeline_update {"lora": [...]} before its run ([] = the base).
    # A DEDICATED Krea-2 file gives this arm its own cache key. The C ABI is stubbed at the same boundary as the
    # capacity ABI above: the recorder takes the JSON body and answers the status in _rt_status.
    import struct as _rst
    check("runtime LoRA: every LoRA-free run before this arm sent no pipeline_update (an unchanged set sends none)",
          _upd_log == [], f"-> updates={_upd_log}")
    _rt_updates, _rt_status = _upd_log, _upd_status
    _ContractEngine.quantfunc_last_error = staticmethod(lambda: b"pipeline busy")
    _saved_resolve = qfn._resolve_lora
    try:
        rt_dir = os.path.join(tmp, "loras_rt"); os.makedirs(rt_dir, exist_ok=True)
        for _n in ("rt-a.safetensors", "rt-b.safetensors"):
            _hdr = json.dumps({"blocks.0.attn.to_q.lora_A.weight":
                               {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]}}).encode()
            with open(os.path.join(rt_dir, _n), "wb") as fh:   # a PEFT-shaped header passes the one-format sniff
                fh.write(_rst.pack("<Q", len(_hdr)) + _hdr + b"\0\0")
        qfn._resolve_lora = lambda n: os.path.join(rt_dir, n)
        with open(os.path.join(dm, "rt-krea2-turbo-quantfunc-int4.safetensors"), "wb") as fh:
            fh.write(b"\0" * 16)
        RtLora = qfn.NODE_CLASS_MAPPINGS["QuantFuncNativeLoRA"]()
        _rt_c0 = len(creates)
        rt_base = KreaL.load("rt-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4")[0]
        rt_a = RtLora.apply(rt_base, "rt-a.safetensors", 0.7)[0]
        rt_b = RtLora.apply(rt_base, "rt-b.safetensors", 1.0)[0]
        rt_ab = RtLora.apply(rt_a, "rt-b.safetensors", 0.5)[0]
        _pa = {"path": os.path.join(rt_dir, "rt-a.safetensors"), "scale": 0.7, "target": "all"}
        _pb1 = {"path": os.path.join(rt_dir, "rt-b.safetensors"), "scale": 1.0, "target": "all"}
        _pb5 = {"path": os.path.join(rt_dir, "rt-b.safetensors"), "scale": 0.5, "target": "all"}
        # A queue in the user's order: base, LoRA A, base, LoRA B, LoRA B again, A+B chained, base.
        _rt_seen = []
        for _m in (rt_base, rt_a, rt_base, rt_b, rt_b, rt_ab, rt_base):
            _ = _m.model._qf.lib                      # the run-start materialization (QFLazyEngine.ensure)
            _rt_seen.append(_m.model._qf._real)
        _rt_new = creates[_rt_c0:]
        check("runtime LoRA: every LoRA set of one Krea-2 model shares ONE create, and the create has no 'lora'",
              len(_rt_new) == 1 and "lora" not in _rt_new[0] and all(r is _rt_seen[0] for r in _rt_seen),
              f"-> creates={_rt_new} one-handle={all(r is _rt_seen[0] for r in _rt_seen)}")
        check("runtime LoRA: each set change is ONE in-place update (A, [], B, A+B, []); an unchanged set sends none",
              _rt_updates == [{"lora": [_pa]}, {"lora": []}, {"lora": [_pb1]}, {"lora": [_pa, _pb5]},
                              {"lora": []}],
              f"-> updates={_rt_updates}")
        # A refused update raises the engine's message and leaves the applied set UNKNOWN: the engine may have kept
        # the previous set or rolled back to the base mid-apply, so the next run re-sends its set — including the
        # set that was applied before the failure (A applied -> a refused B -> A again must send A).
        _rt_status[0] = 7
        _refused = ""
        with _LoraHint() as _rt_got:
            try:
                _ = rt_a.model._qf.lib
            except RuntimeError as _e:
                _refused = str(_e)
        _busy_hint = list(_rt_got)   # a busy engine is not a LoRA it cannot load: no LoRA-page hint
        _rt_real = rt_a.model._qf._real
        _unknown = _rt_real.applied_lora_sig
        _rt_status[0] = 0
        _ = rt_a.model._qf.lib                      # the retry puts A on
        _retried = _rt_updates[-1] == {"lora": [_pa]} and _rt_real.applied_lora_sig == rt_a.model._qf._lora_sig()
        _rt_status[0] = 7
        try:
            _ = rt_b.model._qf.lib                  # B refused while A was on
        except RuntimeError:
            pass
        _rt_status[0] = 0
        _n_before = len(_rt_updates)
        _ = rt_a.model._qf.lib                      # A again: must be RE-SENT, not trusted
        check("runtime LoRA: a refused update raises the engine's message, leaves the set unknown, and the next "
              "run re-sends (A -> refused B -> A re-sends A)",
              "pipeline_update failed (status 7)" in _refused and "pipeline busy" in _refused
              and _unknown is None and _retried and len(_rt_updates) == _n_before + 1
              and _rt_updates[-1] == {"lora": [_pa]},
              f"-> refused={_refused!r} unknown={_unknown!r} retried={_retried} "
              f"resent={_rt_updates[_n_before:]}")
        # The engine refusing a LoRA it cannot load names that file (runtime swap: "'<path>' matched 0 target
        # modules ..."): the refusal still raises the engine's message, and prints ONE LoRA-page warning.
        _busy_err = _ContractEngine.quantfunc_last_error
        _ContractEngine.quantfunc_last_error = staticmethod(
            lambda: f"runtime LoRA swap: '{_pb1['path']}' matched 0 target modules of this transformer".encode())
        _rt_status[0] = 7
        _named = ""
        # a console that cannot print a character of the path gets it escaped in the engine's message (last_err ->
        # console_safe, e.g. a non-ASCII path on a cp1252 console): the path is matched the way the message was written
        _cs0 = qfn.qfe.console_safe
        qfn.qfe.console_safe = lambda text: _cs0(text).replace("_", "\\x5f")   # a character the path has, the URL not
        try:
            with _LoraHint() as _named_got:
                try:
                    _ = rt_b.model._qf.lib
                except RuntimeError as _e:
                    _named = str(_e)
        finally:
            qfn.qfe.console_safe = _cs0
        _ContractEngine.quantfunc_last_error = _busy_err
        _rt_status[0] = 0
        _ = rt_a.model._qf.lib                      # A back on: the blocks below start from the same state
        check("LoRA hint: the engine refusing a LoRA it cannot load still raises its message and prints ONE "
              "LoRA-page warning; a busy refusal prints none",
              "matched 0 target modules" in _named and len(_named_got) == 1 and _busy_hint == [],
              f"-> {_named[:90]!r} {_named_got} busy={_busy_hint}")
        # Mid-generation: a set change while a session is still open refuses LOUD; the SAME set passes
        # without touching the engine (both ways).
        _rt_real.current_session = object()           # the fixture's end_session_if_open never closes it
        _n0 = len(_rt_updates)
        _same_ok = True
        try:
            _ = rt_a.model._qf.lib
        except RuntimeError:
            _same_ok = False
        _mid = ""
        try:
            _ = rt_b.model._qf.lib
        except RuntimeError as _e:
            _mid = str(_e)
        _rt_real.current_session = None
        check("runtime LoRA: a set change mid-generation refuses LOUD, the same set passes (both ways)",
              _same_ok and "still running" in _mid and len(_rt_updates) == _n0,
              f"-> same-ok={_same_ok} mid={_mid!r} updates-sent={len(_rt_updates) - _n0}")
        # An interrupted run's end is refused while its last step drains: ONE bounded retry (the begin path's 2 s),
        # then the set change goes through. time is shimmed inside qf_modelpatcher only (no real sleep, no global patch).
        _ends, _slept, _late = [], [], ""

        def _end_on_second_try():
            _ends.append(1)
            if len(_ends) == 2:
                _rt_real.current_session = None
            return (True, len(_ends) == 2)

        class _TimeShim:
            sleep = staticmethod(_slept.append)

            def __getattr__(self, name):
                return getattr(_real_time, name)
        _real_time = qfn.qfmp.time
        _rt_real.current_session = object()
        _rt_real.end_session_if_open = _end_on_second_try
        qfn.qfmp.time = _TimeShim()
        try:
            _ = rt_b.model._qf.lib
        except RuntimeError as _e:
            _late = str(_e)
        finally:
            qfn.qfmp.time = _real_time
            del _rt_real.end_session_if_open
            _rt_real.current_session = None
        try:
            qfn.qfmp.tag_lora_rebuild(_types.SimpleNamespace(model=_types.SimpleNamespace(_qf=_rt_real)), [_pa], lambda s: None)
            _typed = ""
        except TypeError as _e:
            _typed = str(_e)
        check("runtime LoRA: tag_lora_rebuild refuses a model whose engine is not a QFLazyEngine (no silent drop)",
              "must be a QFLazyEngine" in _typed, f"-> {_typed!r}")
        check("runtime LoRA: after an interrupted run's refused end, ONE 2 s retry, then the set change goes through",
              not _late and len(_ends) == 2 and _slept == [2.0] and _rt_updates[-1] == {"lora": [_pb1]},
              f"-> refused={_late!r} ends={len(_ends)} slept={_slept} last={_rt_updates[-1]}")
    except Exception as e:  # noqa: BLE001
        check("runtime LoRA arm", False, f"-> raised {type(e).__name__}: {e}")
    finally:
        qfn._resolve_lora = _saved_resolve
        _upd_status[0] = 0
        del _ContractEngine.quantfunc_last_error

    # ── 5c) a LoRA rebuild keeps the loader's session dials ──
    # The loader sets its widgets (attention backend, quality_enhance, caches, H3's audio switch) on the MODEL; a
    # QuantFuncNativeLoRA rebuild hands back a NEW model. Without the carry, a LoRA'd H3 silently ran "auto" attention
    # instead of its flash default and dropped the switches.
    import ast as _dast
    _saved_resolve = qfn._resolve_lora
    try:
        dl_dir = os.path.join(tmp, "loras_dials"); os.makedirs(dl_dir, exist_ok=True)
        _hdr = json.dumps({"blocks.0.attn.to_q.lora_A.weight":
                           {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]}}).encode()
        with open(os.path.join(dl_dir, "dial.safetensors"), "wb") as fh:
            fh.write(_rst.pack("<Q", len(_hdr)) + _hdr + b"\0\0")
        qfn._resolve_lora = lambda n: os.path.join(dl_dir, n)
        _h3 = H3L.load("fx-minimax-h3-quantfunc-int4.safetensors", "minimax-h3-fl2va",
                       attention_backend="flash", sol_tau=0.5, quality_enhance=True, audio_enhance=True,
                       step_cache=0.1, block_cache=0.2)[0]
        _h3l = qfn.NODE_CLASS_MAPPINGS["QuantFuncNativeLoRA"]().apply(_h3, "dial.safetensors", 0.5)[0]
        _src, _dst = _h3.model, _h3l.model
        _want = (_src.residency_opts(), _src._audio_enhance)
        _got = (_dst.residency_opts(), _dst._audio_enhance)
        check("a LoRA rebuild keeps the loader's session dials (H3: backend, sol_tau, quality_enhance, caches, audio)",
              _dst is not _src and _got == _want and _want[0].get("attention_backend") == "flash"
              and _want[0].get("video_enhance") is True
              and _want[1] is True,
              f"-> rebuilt={_dst is not _src} want={_want} got={_got}")
        # DEATH RULE: every set_* on the session mixin or a family model writes only attributes its class carries in
        # _SESSION_DIALS, so a new dial cannot be forgotten by the carry.
        from qfn_test_pkg import qf_modelpatcher as _dqmp
        import qfn_test_pkg as _dpkg
        _dmods = {"qf_modelpatcher": _dqmp}
        for _mn in _dpkg._FAMILY_MODULES:
            _dmods[_mn] = sys.modules[f"qfn_test_pkg.{_mn}"]
        _dviol, _dseen = [], 0
        for _mn, _mod in _dmods.items():
            for _cn in _dast.parse(open(_mod.__file__, encoding="utf-8").read()).body:
                if not isinstance(_cn, _dast.ClassDef):
                    continue
                _cls = getattr(_mod, _cn.name, None)
                if not (isinstance(_cls, type) and issubclass(_cls, _dqmp.QFSessionModelMixin)):
                    continue
                for _fn in _cn.body:
                    if not (isinstance(_fn, _dast.FunctionDef) and _fn.name.startswith("set_")):
                        continue
                    _dseen += 1
                    for _n in _dast.walk(_fn):
                        if isinstance(_n, _dast.Assign):
                            for _t in _n.targets:
                                if (isinstance(_t, _dast.Attribute) and isinstance(_t.value, _dast.Name)
                                        and _t.value.id == "self" and _t.attr not in _cls._SESSION_DIALS):
                                    _dviol.append(f"{_mn}.{_cn.name}.{_fn.name} writes {_t.attr}")
        check("every session-dial setter's attribute is carried across a LoRA rebuild (AST, family modules derived)",
              _dseen >= 6 and not _dviol, f"-> setters={_dseen} uncarried={_dviol}")   # 6: H3's partial switch is gone
        # DEATH RULE: the begin dials are emitted in ONE place, dial_opts — no other function reads them, so no family
        # can drift from the always-send rule (the image families once built their own copies that omitted "auto").
        _eviol, _ereads = [], 0
        for _mn, _mod in _dmods.items():
            for _fn in _dast.walk(_dast.parse(open(_mod.__file__, encoding="utf-8").read())):
                if not isinstance(_fn, _dast.FunctionDef):
                    continue
                for _n in _dast.walk(_fn):
                    _nm = None
                    if (isinstance(_n, _dast.Call) and isinstance(_n.func, _dast.Name) and _n.func.id == "getattr"
                            and len(_n.args) >= 2 and isinstance(_n.args[1], _dast.Constant)):
                        _nm = _n.args[1].value
                    elif isinstance(_n, _dast.Attribute) and isinstance(_n.ctx, _dast.Load):
                        _nm = _n.attr
                    if _nm in ("_attn_backend", "_video_enhance"):
                        _ereads += _fn.name == "dial_opts"
                        if _fn.name != "dial_opts":
                            _eviol.append(f"{_mn}.{_fn.name} reads {_nm}")
        check("the begin dials (attention backend, the enhance switch) are read ONLY by dial_opts (AST, family modules derived)",
              _ereads >= 2 and not _eviol, f"-> dial_opts reads={_ereads} others={_eviol}")
    except Exception as e:  # noqa: BLE001
        check("session-dial carry arm", False, f"-> raised {type(e).__name__}: {e}")
    finally:
        qfn._resolve_lora = _saved_resolve

    # ── 6) HOST-RAM: the LOGICAL patcher owns none. Native backing (VRAM pages and their host backup) belongs to the
    #    canonical resource adapters; ComfyUI 0.37 calls loaded_ram_size/partially_unload_ram only on a DYNAMIC patcher,
    #    and its base answers 0. So: 0 and nothing freed before create, after create and after the logical detach, which
    #    never evicts the engine. (The pre-canonical arms asserted a logical-detach eviction that no longer exists.)
    try:
        open(os.path.join(dm, "ram-krea2-turbo-quantfunc-int4.safetensors"), "wb").write(b"\0" * 16)
        fresh = KreaL.load("ram-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4")[0]
        eng = fresh.model._qf
        _ram = [(fresh.loaded_ram_size(), fresh.partially_unload_ram(10 ** 12))]
        _ = eng.lib
        _ram.append((fresh.loaded_ram_size(), fresh.partially_unload_ram(10 ** 12)))
        fresh.detach(unpatch_all=False)
        _ram.append((fresh.loaded_ram_size(), fresh.partially_unload_ram(10 ** 12)))
        check("the logical patcher reports and frees 0 host RAM (before create, after create, after its detach)",
              _ram == [(0, 0)] * 3 and not fresh.is_dynamic(), f"-> {_ram}")
        check("the logical detach never evicts the engine (native eviction is the resource adapters')",
              eng.materialized and eng.ensure().pipeline is not None)
    except Exception as e:  # noqa: BLE001
        check("host-RAM ownership arm", False, f"-> raised {type(e).__name__}: {e}")

    # ── 7) SHARED-handle sibling safety on a DEDICATED pair: one real handle, never torn down by a logical detach ──
    try:
        import gc
        open(os.path.join(dm, "sh-krea2-turbo-quantfunc-int4.safetensors"), "wb").write(b"\0" * 16)
        pa = KreaL.load("sh-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4")[0]
        pb = KreaL.load("sh-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4")[0]
        ra = pa.model._qf.ensure()
        rb = pb.model._qf.ensure()
        check("two loads of one file share the real handle", ra is rb)
        pa.detach(unpatch_all=False)
        check("a logical detach on one sibling leaves the shared pipeline alive for the other",
              pb.model._qf.pipeline is not None and ra.pipeline is not None)
        del pb, rb
        gc.collect()
        check("the surviving wrapper keeps its pipeline once the sibling is gone",
              pa.model._qf.ensure().pipeline is not None)
        del pa, ra
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
            mf = json.load(open(os.path.join(real_cfg, d, "qf_native.json"), encoding="utf-8"))
            hints = mf.get("file_hints")
            need = ["transformer1"]
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
