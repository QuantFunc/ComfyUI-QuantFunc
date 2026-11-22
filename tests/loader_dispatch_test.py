#!/usr/bin/env python3
"""Behavioural tests for QuantFunc PC Engine loaders, unified in-place LoRA, VRAM coexistence,
workflow coercion, and liveness substrate.

Run: COMFY_ROOT=... python3 tests/loader_dispatch_test.py
"""
import ast
import fnmatch
import gc
import importlib.util
import inspect
import json
import logging
import os
import re
import shutil
import struct
import sys
import tempfile
import types
import weakref
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.dirname(_HERE)


def _emit_skip(reason):
    print(f"LOADER_DISPATCH: SKIP — {reason}")
    return 77


def _load_plugin():
    comfy_root = os.environ.get("COMFY_ROOT")
    if not comfy_root:
        guess = os.path.dirname(os.path.dirname(_PLUGIN))
        comfy_root = guess if os.path.isfile(os.path.join(guess, "folder_paths.py")) else None
    if not comfy_root:
        return None, "ComfyUI root not found (set COMFY_ROOT)"
    sys.path.insert(0, comfy_root)
    try:
        sys.argv = [sys.argv[0], "--cpu"]
        import comfy.options
        comfy.options.enable_args_parsing()
        import folder_paths  # noqa: F401
        spec = importlib.util.spec_from_file_location("qfn_test_pkg", os.path.join(_PLUGIN, "__init__.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["qfn_test_pkg"] = mod
        spec.loader.exec_module(mod)
        return mod, None
    except Exception as exc:  # noqa: BLE001
        return None, f"plugin import failed: {type(exc).__name__}: {exc}"


class _DummyEngine:
    lib = "LIB"
    capacity_bytes = 4096
    step_count = 0
    sampler_step_count = 0

    def __init__(self):
        self.xfm = object()
        self.current_session = None

    def end_session_if_open(self):
        return (False, True)

    def destroy(self):
        self.xfm = None
        self.current_session = None


def main():
    qfn, why = _load_plugin()
    if qfn is None:
        return _emit_skip(why)
    if not getattr(qfn, "_IMPORT_OK", False):
        return _emit_skip("plugin imported but its comfy imports failed (_IMPORT_OK False)")
    import folder_paths
    import torch
    import comfy.model_base as _mb
    import comfy.sd as _csd
    import comfy.nested_tensor as _cnt
    import comfy.latent_formats as _clf
    from comfy_extras.nodes_model_advanced import ModelSamplingSD3
    from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE as _CM
    from qfn_test_pkg import qf_modelpatcher as qfmp
    from qfn_test_pkg import qf_ltx_modelpatcher as qlm

    sys.path.insert(0, _HERE)
    import _banned_terms as _bt

    bad = 0

    def check(label, ok, detail=""):
        nonlocal bad
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}{(' ' + detail) if detail else ''}")
        if not ok:
            bad += 1

    tmp = tempfile.mkdtemp(prefix="qf_loader_test_")
    dm = os.path.join(tmp, "diffusion_models")
    os.makedirs(dm)
    for f in ("other.safetensors", "not-a-model.txt",
              "fx-ltx-2.5-quantfunc-4bit.safetensors",
              "fx-minimax-h3-quantfunc-int4.safetensors",
              "fx-krea2-turbo-quantfunc-int4.safetensors",
              "fx-qwen-image-2.1-quantfunc-int4.safetensors"):
        open(os.path.join(dm, f), "wb").write(b"\0" * 16)
    folder_paths.add_model_folder_path("diffusion_models", dm)

    def _mini_st(path, *keys):
        hdr = json.dumps({k: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]}
                          for i, k in enumerate(keys)}).encode()
        hdr += b" " * ((8 - (len(hdr) % 8)) % 8)
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(hdr)))
            fh.write(hdr)
            fh.write(b"\0" * (4 * len(keys)))

    _mini_st(os.path.join(dm, "fx-ltx25-connectors.safetensors"),
             "model.diffusion_model.video_embeddings_connector.learnable_registers",
             "model.diffusion_model.audio_embeddings_connector.learnable_registers")
    _allin_name = "fx-ltx-2.5-allin-quantfunc-4bit.safetensors"
    _mini_st(os.path.join(dm, _allin_name),
             "text_embedding_projection.video_aggregate_embed.weight",
             "model.diffusion_model.video_embeddings_connector.learnable_registers",
             "model.diffusion_model.audio_embeddings_connector.learnable_registers")
    _half1 = "fx-ltx-2.5-connonly-quantfunc-4bit.safetensors"
    _mini_st(os.path.join(dm, _half1),
             "model.diffusion_model.video_embeddings_connector.learnable_registers")
    _half2 = "fx-ltx-2.5-projonly-quantfunc-4bit.safetensors"
    _mini_st(os.path.join(dm, _half2),
             "text_embedding_projection.video_aggregate_embed.weight")

    dm2 = os.path.join(tmp, "diffusion_models_iso")
    os.makedirs(dm2)
    open(os.path.join(dm2, "fx-iso-ltx-2.5-quantfunc-4bit.safetensors"), "wb").write(b"\0" * 16)
    shutil.copyfile(os.path.join(dm, _half1), os.path.join(dm2, "fx-iso-ltx-2.5-connonly-quantfunc-4bit.safetensors"))
    folder_paths.add_model_folder_path("diffusion_models", dm2)

    cfgroot = os.path.join(tmp, "configs")
    for preset in ("ltx2-2.5-22b", "minimax-h3-fl2va", "krea2-turbo-int4", "qwen-image-2.1-int4"):
        shutil.copytree(os.path.join(_PLUGIN, "configs", preset), os.path.join(cfgroot, preset))
    for name, mf in (("fx-ltx", {"family": "ltx2"}), ("fx-h3", {"family": "minimax-h3"}),
                     ("fx-alien", {"family": "no-such-family"})):
        os.makedirs(os.path.join(cfgroot, name))
        with open(os.path.join(cfgroot, name, "qf_native.json"), "w", encoding="utf-8") as fh:
            json.dump(mf, fh)
    qfn._CONFIGS_DIR = cfgroot

    creates, create_devices, fake_cache = [], [], {}
    _lora_log, _lora_status = [], [0]

    class _ContractEngine(qfn.qfe.QFEngineHandle):
        def __init__(self):
            super().__init__(self, object(), capacity_bytes=4096)
            self._resident = 4096

        @staticmethod
        def qf_xfm_set_loras(_xfm, payload):
            _lora_log.append(json.loads(payload))
            return _lora_status[0]

        @staticmethod
        def qf_last_error():
            return b"transformer busy"

        def end_session_if_open(self):
            return (False, True)

        def resident_vram_bytes(self):
            return self._resident

        def unload_vram(self, req=qfn.qfe.QF_UINT64_MAX):
            freed = self._resident if req >= self._resident else int(req)
            self._resident -= freed
            return freed

        def vram_need_bytes(self, latent_shape):
            n = 1
            for d in latent_shape:
                n *= int(d)
            return 4 * n

        def destroy(self):
            self.xfm = None
            self.current_session = None

    def fake_get_engine(weights_path, create_cfg=None, device_id=0, api_key=None):
        ck = (weights_path, int(device_id), json.dumps(create_cfg or {}, sort_keys=True))
        eng = fake_cache.get(ck)
        if eng is not None and eng.xfm is not None:
            return eng, ck
        creates.append((weights_path, dict(create_cfg or {})))
        create_devices.append(int(device_id))
        eng = _ContractEngine()
        fake_cache[ck] = eng
        return eng, ck

    def fake_peek_engine(weights_path, create_cfg=None, device_id=0):
        ck = (weights_path, int(device_id), json.dumps(create_cfg or {}, sort_keys=True))
        eng = fake_cache.get(ck)
        return (eng, ck) if (eng is not None and eng.xfm is not None) else None

    qfn._get_engine = fake_get_engine
    qfn._peek_engine = fake_peek_engine
    qfn._estimate_engine = lambda *a, **k: qfn.qfe.ColdNeed(None, None)
    qfn.qfe.load_lib = lambda *a, **k: _DummyEngine.lib
    qfn._FAMILY_BUILDERS.clear()
    qfn._FAMILY_MATCHERS.clear()
    qfn._register_families()

    LtxL = qfn.NODE_CLASS_MAPPINGS["QuantFuncLTXLoader"]()
    H3L = qfn.NODE_CLASS_MAPPINGS["QuantFuncH3Loader"]()
    KreaL = qfn.NODE_CLASS_MAPPINGS["QuantFuncKrea2Loader"]()
    QiL = qfn.NODE_CLASS_MAPPINGS["QuantFuncQwenImage21Loader"]()
    _FOUR = ("QuantFuncLTXLoader", "QuantFuncH3Loader", "QuantFuncKrea2Loader", "QuantFuncQwenImage21Loader")

    # ── 1) UI surface: 4 loaders, hidden log_level & api_key, attention_backend, widget order ──
    _nodes = sorted(qfn.NODE_CLASS_MAPPINGS)
    check("8-node surface: 4 loaders + LoRA + H3 upscale, no Wan/CloudTE",
          not any("Wan" in k or "CloudTE" in k for k in _nodes)
          and set(_FOUR) | {"QuantFuncNativeLoRA", "QuantFuncH3LatentUpscale"} <= set(_nodes), f"-> {_nodes}")
    _ll_bad = [n for n in _FOUR if qfn.NODE_CLASS_MAPPINGS[n].INPUT_TYPES().get("hidden", {}).get("log_level") != ("STRING", {})]
    check("log_level is hidden on all 4 loaders", not _ll_bad, f"-> {_ll_bad}")
    check("ltx/h3 model_config lists only own-family presets",
          "fx-ltx" in LtxL.INPUT_TYPES()["optional"]["model_config"][0]
          and "krea2-turbo-int4" not in LtxL.INPUT_TYPES()["optional"]["model_config"][0]
          and "fx-h3" in H3L.INPUT_TYPES()["optional"]["model_config"][0])

    _saved_cfg = qfn._CONFIGS_DIR
    qfn._CONFIGS_DIR = os.path.join(_PLUGIN, "configs")
    try:
        _real = {n: qfn.NODE_CLASS_MAPPINGS[n].INPUT_TYPES() for n in _FOUR}
    finally:
        qfn._CONFIGS_DIR = _saved_cfg
    check("shipped configs hide single-preset model_config on all 4 loaders",
          all(len(v["optional"]["model_config"][0]) == 1 and v["optional"]["model_config"][1].get("hidden") is True
              for v in _real.values()))

    _sol_values = ["sol_tau", "sol_start_percent", "sol_end_percent", "sol_min_tokens",
                   "sol_sink_conditioning", "sol_version"]
    _sol_switches = ["sol_enabled", "sol_window_enabled"]
    check("LTX node surface matches layout spec",
          list(LtxL.INPUT_TYPES()["required"]) == ["transformer"]
          and list(LtxL.INPUT_TYPES()["optional"]) == ["model_config", "attention_backend"] + _sol_values
          + ["step_cache", "block_cache"] + _sol_switches + ["quality_enhance", "pinned_memory"])
    check("H3 node surface matches layout spec",
          list(H3L.INPUT_TYPES()["required"]) == ["transformer"]
          and list(H3L.INPUT_TYPES()["optional"]) == ["model_config", "attention_backend"] + _sol_values
          + ["step_cache", "block_cache"] + _sol_switches + ["quality_enhance", "audio_enhance", "pinned_memory"])

    _sw = {n: qfn.NODE_CLASS_MAPPINGS[n].INPUT_TYPES() for n in _FOUR}
    _order = {n: [(k, v[0]) for k, v in list(it["required"].items()) + list(it.get("optional", {}).items())]
              for n, it in _sw.items()}
    _late = {n: o for n, o in _order.items()
             if any(t != "BOOLEAN" for _, t in o[next((i for i, (_, t) in enumerate(o) if t == "BOOLEAN"), len(o)):])}
    check("every loader: all BOOLEAN switches come after all value inputs", not _late, f"-> {_late}")

    # attention_backend: ["auto", "int8", "int4", "fp16"], default "auto", aliases mapped, VALIDATE_INPUTS works
    check("attention_backend dropdown is ['auto', 'int8', 'int4', 'fp16'] with default 'auto' on all loaders",
          all(it["optional"]["attention_backend"][0] == ["auto", "int8", "int4", "fp16"]
              and it["optional"]["attention_backend"][1]["default"] == "auto" for it in _sw.values()))
    check("attention_backend aliases map cleanly to PC Engine names",
          qfn._attn_backend_to_engine("qfa") == "int8"
          and qfn._attn_backend_to_engine("sage") == "int8"
          and qfn._attn_backend_to_engine("fp16_native") == "fp16"
          and qfn._attn_backend_to_engine("native") == "fp16"
          and qfn._attn_backend_to_engine("flash") == "fp16"
          and qfn._attn_backend_to_engine("auto") == "auto"
          and qfn._attn_backend_to_engine("int4") == "int4")
    check("VALIDATE_INPUTS accepts legacy aliases and refuses unknown backend",
          all(qfn.NODE_CLASS_MAPPINGS[n].VALIDATE_INPUTS("qfa") is True
              and qfn.NODE_CLASS_MAPPINGS[n].VALIDATE_INPUTS("sage") is True
              and isinstance(qfn.NODE_CLASS_MAPPINGS[n].VALIDATE_INPUTS("bogus"), str) for n in _FOUR))

    # User-visible text check (no banned terms in DESCRIPTION or tooltips)
    _texts = []
    for _n in _FOUR + ("QuantFuncNativeLoRA",):
        _c = qfn.NODE_CLASS_MAPPINGS[_n]
        _texts.append((f"{_n}.DESCRIPTION", getattr(_c, "DESCRIPTION", "")))
        for _sec in ("required", "optional"):
            for _k, _v in _c.INPUT_TYPES().get(_sec, {}).items():
                if len(_v) > 1 and isinstance(_v[1], dict) and _v[1].get("tooltip"):
                    _texts.append((f"{_n}.{_k}", _v[1]["tooltip"]))
    _hits = [(w, x) for w, t in _texts for x in _bt.desc_hits(t) + _bt.term_hits(t) + _bt.mode_id_hits(t)]
    check("all 4 loaders + LoRA node user-visible text carries zero banned words",
          not _hits and len(_texts) > 20, f"-> {len(_texts)} texts, hits={_hits[:4]}")

    # ── 2) qf_layout: 2 anomalous-tail workflow coercion in Python loaders ──
    _warn_msgs = []

    class _CapWarn(logging.Handler):
        def emit(self, record):
            _warn_msgs.append(record.getMessage())

    _wh = _CapWarn()
    qfn._log.addHandler(_wh)
    try:
        # Exact anomalous tails from QuantFunc-LTX25-t2v.json and QuantFunc-MiniMaxH3-ref2va*.json
        ltx_mig = LtxL.load(_allin_name, "ltx2-2.5-22b", quality_enhance=12288, pinned_memory="exact_kv_and_rows")[0]
        _ = ltx_mig.model._qf.lib
        ltx_mig_cfg = json.loads(ltx_mig.model._qf._ckey[-1])
        h3_mig = H3L.load("fx-minimax-h3-quantfunc-int4.safetensors", "minimax-h3-fl2va",
                          quality_enhance=12288, audio_enhance="exact_kv_and_rows", pinned_memory=2)[0]
        _ = h3_mig.model._qf.lib
        h3_mig_cfg = json.loads(h3_mig.model._qf._ckey[-1])
    finally:
        qfn._log.removeHandler(_wh)
    check("qf_layout: 2 anomalous non-boolean tails coerce safely to defaults and log migration warning",
          ltx_mig.model._video_enhance is False and ltx_mig_cfg.get("pinned_memory") is True
          and h3_mig.model._video_enhance is False and h3_mig.model._audio_enhance is False
          and "pinned_memory" not in h3_mig_cfg
          and sum("migrated non-boolean" in m for m in _warn_msgs) >= 5,
          f"-> warns={len(_warn_msgs)} ltx={ltx_mig_cfg} h3={h3_mig_cfg}")

    # ── 3) Zero-staging direct paths across ALL 4 families (no os.symlink/os.link, no stage_*) ──
    check("retired staging & BF16/i8-convrot helpers are absent",
          not hasattr(qfmp, "stage_denoise_only_package") and not hasattr(qfmp, "stage_config_package")
          and not hasattr(qlm, "_ltx_source_create_options"))
    _link_calls = []
    with mock.patch("os.symlink", side_effect=lambda *a, **k: _link_calls.append("symlink")), \
         mock.patch("os.link", side_effect=lambda *a, **k: _link_calls.append("link")):
        _pm_loads = {
            "ltx2": (LtxL, _allin_name, "ltx2-2.5-22b", "ltx25"),
            "minimax-h3": (H3L, "fx-minimax-h3-quantfunc-int4.safetensors", "minimax-h3-fl2va", "minimax_h3"),
            "krea2": (KreaL, "fx-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4", "krea2"),
            "qwenimage21": (QiL, "fx-qwen-image-2.1-quantfunc-int4.safetensors", "qwen-image-2.1-int4", "qwenimage21"),
        }
        _fam_cfgs = {}
        for fam, (node, xfm, preset, arch) in _pm_loads.items():
            p_def = node.load(xfm, preset)[0]
            _ = p_def.model._qf.lib
            c_def = json.loads(p_def.model._qf._ckey[-1])
            p_on = node.load(xfm, preset, pinned_memory=True)[0]
            _ = p_on.model._qf.lib
            c_on = json.loads(p_on.model._qf._ckey[-1])
            p_off = node.load(xfm, preset, pinned_memory=False)[0]
            _ = p_off.model._qf.lib
            c_off = json.loads(p_off.model._qf._ckey[-1])
            _fam_cfgs[fam] = (arch, p_def.model._qf._ckey[0], c_def, c_on, c_off)
    check("all 4 families load link-free with model_arch + config_path + pinned_memory (never denoise_only/use_pinned_memory)",
          not _link_calls and all(
              c_def.get("model_arch") == arch and os.path.isfile(c_def.get("config_path", ""))
              and "denoise_only" not in c_def and "use_pinned_memory" not in c_def
              and c_on.get("pinned_memory") is True and "pinned_memory" not in c_off
              and (c_def.get("pinned_memory") is True if fam == "ltx2" else "pinned_memory" not in c_def)
              and os.path.basename(wp) == _pm_loads[fam][1]
              for fam, (arch, wp, c_def, c_on, c_off) in _fam_cfgs.items()),
          f"-> links={_link_calls}")
    # Verify H3 transformer config retains qf_adaln_fold_dropped_resq: 50
    h3_cfg_doc = json.load(open(_fam_cfgs["minimax-h3"][2]["config_path"], encoding="utf-8"))
    check("MiniMax-H3 transformer/config.json retains qf_adaln_fold_dropped_resq == 50",
          h3_cfg_doc.get("qf_adaln_fold_dropped_resq") == 50, f"-> {h3_cfg_doc.get('qf_adaln_fold_dropped_resq')}")

    # ── 4) LTX connectors completion probe + mask / reject list / zero-latent guard ──
    _out_split = LtxL.load("fx-ltx-2.5-quantfunc-4bit.safetensors", "ltx2-2.5-22b")[0]
    _ = _out_split.model._qf.lib
    _split_cfg = json.loads(_out_split.model._qf._ckey[-1])
    check("ltx2 transformer-only resolves sibling connectors_path directly",
          os.path.basename(_split_cfg.get("connectors_path", "")) == "fx-ltx25-connectors.safetensors"
          and os.path.isfile(_split_cfg.get("config_path", "")), f"-> {_split_cfg}")
    for bad_ltx, expect_tok in (("fx-iso-ltx-2.5-quantfunc-4bit.safetensors", "connectors completion"),
                                ("fx-iso-ltx-2.5-connonly-quantfunc-4bit.safetensors", "audio_embeddings_connector")):
        try:
            LtxL.load(bad_ltx, "ltx2-2.5-22b")
            check(f"ltx2 isolated {bad_ltx} refuses loud", False)
        except RuntimeError as e:
            check(f"ltx2 isolated {bad_ltx} refuses loud", expect_tok in str(e), f"-> {str(e)[:80]}")
    check("ltx scale_latent_inpaint inherited & denoise_mask accepted & keyframe_idxs rejected",
          "scale_latent_inpaint" not in qlm.QFLTXModel.__dict__
          and qlm.QFLTXModel.scale_latent_inpaint is _mb.LTXV.scale_latent_inpaint
          and "denoise_mask" not in qlm.QFLTXModel._ENGINE_IGNORED_COND_KEYS)
    _mdl = _out_split.model
    _mdl.extra_conds(denoise_mask=torch.ones(1, 1, 2, 2, 2))
    try:
        _mdl.extra_conds(keyframe_idxs=torch.zeros(1, 1))
        check("extra_conds rejects keyframe_idxs loud", False)
    except RuntimeError as e:
        check("extra_conds rejects keyframe_idxs loud", "keyframe_idxs" in str(e))

    # Zero-latent guard + structural before self._begin
    for _tag in ("LTX", "LTX-AV", "H3"):
        try:
            qfmp.refuse_all_zero_initial_latent(torch.zeros(1, 16, 3, 8, 8), _tag)
            check(f"zero-latent guard fires for {_tag}", False)
        except RuntimeError as e:
            check(f"zero-latent guard fires for {_tag}", f"qf_native {_tag}:" in str(e) and "ALL ZEROS" in str(e))
        qfmp.refuse_all_zero_initial_latent(torch.randn(1, 16, 3, 8, 8), _tag)
    for _mod in ("qf_ltx_modelpatcher.py", "qf_h3_modelpatcher.py"):
        _src = open(os.path.join(_PLUGIN, _mod), encoding="utf-8").read()
        _ok = _src.count("refuse_all_zero_initial_latent(") == 1 and all(
            "self._begin(" in _src[m.end():m.end() + 400] for m in re.finditer("refuse_all_zero_initial_latent", _src))
        check(f"{_mod}: zero-latent guard precedes self._begin", _ok)

    # ── 5) memory_required D3/D4 + cold floor (#716) + path containment ──
    open(os.path.join(dm, "cold-krea2-turbo-quantfunc-int4.safetensors"), "wb").write(b"\0" * 16)
    _hot = KreaL.load("fx-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4")[0]
    _ = _hot.model._qf.lib
    _shape, _conds = [2, 16, 1, 128, 128], {"c_crossattn": [[1, 512, 30720]]}
    _mr_s = _hot.model.memory_required([1, 16, 1, 8, 8], cond_shapes={})
    _mr_m = _hot.model.memory_required(_shape, cond_shapes=_conds)
    _mr_b = _hot.model.memory_required([2, 16, 1, 256, 256], cond_shapes=_conds)
    check("memory_required geometry-proportional, monotonic, below 1 GiB", _mr_s < _mr_m < _mr_b and _mr_m <= (1 << 30))
    _cold = KreaL.load("cold-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4")[0]
    _c_before = len(creates)
    _torch_est = int(super(qfmp.QFSessionModelMixin, _cold.model).memory_required(_shape, cond_shapes=_conds))
    _mr_cold = _cold.model.memory_required(_shape, cond_shapes=_conds)
    check("cold model uses comfy floor without creating transformer",
          _mr_cold == max(_torch_est, _cold.model._qf_comfy_side_bytes(_shape, _conds)) and len(creates) == _c_before)
    for evil in ("../../../../etc/passwd", "/etc/passwd"):
        try:
            qfn._resolve_transformer(evil)
            check(f"containment refuses {evil!r}", False)
        except Exception:  # noqa: BLE001
            check(f"containment refuses {evil!r}", True)

    # ── 6) Unified in-place LoRA (qf_xfm_set_loras), dial carry, LoRA hints, and drain retry ──
    _LORA_URL = "https://www.quantfunc.com/docs/lora"

    class _LoraHint(logging.Handler):
        def __enter__(self):
            self.msgs = []
            qfmp._log.addHandler(self)
            return self.msgs

        def __exit__(self, *_exc):
            qfmp._log.removeHandler(self)

        def emit(self, record):
            if _LORA_URL in record.getMessage():
                self.msgs.append(record.getMessage())

    _fake_lora = {"diffusion_model.blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 8),
                  "diffusion_model.blocks.0.attn.to_q.lora_B.weight": torch.zeros(8, 4)}
    with _LoraHint() as _stock_got:
        _csd.load_lora_for_models(_hot, None, dict(_fake_lora), 1.0, 0.0)
    check("stock ComfyUI LoRA loader prints one warning pointing to QuantFunc Native LoRA docs",
          len(_stock_got) == 1 and qfn.NODE_DISPLAY_NAME_MAPPINGS["QuantFuncNativeLoRA"] in _stock_got[0])

    check("no qf_xfm_set_loras sent before any LoRA swap", _lora_log == [], f"-> {_lora_log}")
    lora_dir = os.path.join(tmp, "loras")
    os.makedirs(lora_dir)
    _lh = json.dumps({"blocks.0.attn.to_q.lora_A.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]}}).encode()
    for fn in ("a.safetensors", "b.safetensors"):
        open(os.path.join(lora_dir, fn), "wb").write(struct.pack("<Q", len(_lh)) + _lh + b"\0\0")
    _kh = json.dumps({"lora_unet_blocks_0_attn_to_q.lora_down.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]}}).encode()
    open(os.path.join(lora_dir, "k.safetensors"), "wb").write(struct.pack("<Q", len(_kh)) + _kh + b"\0\0")
    qfn._resolve_lora = lambda n: os.path.join(lora_dir, n)
    LoraNode = qfn.NODE_CLASS_MAPPINGS["QuantFuncNativeLoRA"]()

    open(os.path.join(dm, "rt-krea2-turbo-quantfunc-int4.safetensors"), "wb").write(b"\0" * 16)
    _rt_c0 = len(creates)
    rt_base = KreaL.load("rt-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4")[0]
    shifted = ModelSamplingSD3().patch(rt_base, 3.0)[0]
    rt_a = LoraNode.apply(shifted, "a.safetensors", 0.7)[0]
    rt_b = LoraNode.apply(rt_base, "b.safetensors", 1.0)[0]
    rt_ab = LoraNode.apply(rt_a, "b.safetensors", 0.5)[0]
    check("LoRA rebuild preserves upstream ModelSamplingSD3 patch", "model_sampling" in rt_a.object_patches)
    _pa = {"path": os.path.join(lora_dir, "a.safetensors"), "scale": 0.7}
    _pb1 = {"path": os.path.join(lora_dir, "b.safetensors"), "scale": 1.0}
    _pb5 = {"path": os.path.join(lora_dir, "b.safetensors"), "scale": 0.5}
    _seen = []
    for m in (rt_base, rt_a, rt_base, rt_b, rt_b, rt_ab, rt_base):
        _ = m.model._qf.lib
        _seen.append(m.model._qf._real)
    check("unified in-place LoRA: ONE transformer create across all stacks, exact qf_xfm_set_loras swaps",
          len(creates) - _rt_c0 == 1 and all(r is _seen[0] for r in _seen)
          and _lora_log == [[_pa], [], [_pb1], [_pa, _pb5], []], f"-> {_lora_log}")
    with _LoraHint() as _k_got:
        try:
            LoraNode.apply(rt_base, "k.safetensors", 1.0)
            _k_err = ""
        except RuntimeError as e:
            _k_err = str(e)
    check("foreign kohya LoRA refused loud with LoRA hint", "kohya" in _k_err and len(_k_got) == 1)

    # Refused swap leaves applied_lora_sig unknown and re-sends on next run
    _lora_status[0] = 6
    try:
        _ = rt_a.model._qf.lib
        _ref_msg = ""
    except RuntimeError as e:
        _ref_msg = str(e)
    _rt_real = rt_a.model._qf._real
    _unknown = _rt_real.applied_lora_sig
    _lora_status[0] = 0
    _ = rt_a.model._qf.lib
    check("refused qf_xfm_set_loras leaves applied_lora_sig=None and retries cleanly",
          "qf_xfm_set_loras failed" in _ref_msg and _unknown is None and _lora_log[-1] == [_pa])

    # Session dial carry across LoRA rebuild + AST death rules
    _h3 = H3L.load("fx-minimax-h3-quantfunc-int4.safetensors", "minimax-h3-fl2va",
                   attention_backend="int8", sol_tau=0.5, quality_enhance=True, audio_enhance=True,
                   step_cache=0.1, block_cache=0.2)[0]
    _h3l = LoraNode.apply(_h3, "a.safetensors", 0.5)[0]
    check("LoRA rebuild carries all session dials (backend, quality_enhance, caches, sol, audio compat)",
          _h3l.model is not _h3.model and _h3l.model.dial_opts() == _h3.model.dial_opts()
          and _h3l.model.dial_opts().get("attention_backend") == "int8"
          and _h3l.model.dial_opts().get("cat_keep_ratio") == 1.0
          and _h3l.model._audio_enhance is True)

    _dmods = {"qf_modelpatcher": qfmp, **{m: sys.modules[f"qfn_test_pkg.{m}"] for m in qfn._FAMILY_MODULES}}
    _dviol, _dseen = [], 0
    for _mn, _mod in _dmods.items():
        for _cn in ast.parse(open(_mod.__file__, encoding="utf-8").read()).body:
            if not isinstance(_cn, ast.ClassDef):
                continue
            _cls = getattr(_mod, _cn.name, None)
            if not (isinstance(_cls, type) and issubclass(_cls, qfmp.QFSessionModelMixin)):
                continue
            for _fn in _cn.body:
                if not (isinstance(_fn, ast.FunctionDef) and _fn.name.startswith("set_")):
                    continue
                _dseen += 1
                for _n in ast.walk(_fn):
                    if isinstance(_n, ast.Assign):
                        for _t in _n.targets:
                            if (isinstance(_t, ast.Attribute) and isinstance(_t.value, ast.Name)
                                    and _t.value.id == "self" and _t.attr not in _cls._SESSION_DIALS
                                    and not _t.attr.endswith("_warned")):
                                _dviol.append(f"{_mn}.{_cn.name}.{_fn.name} writes {_t.attr}")
    check("every session-dial setter attribute is in _SESSION_DIALS (AST)", _dseen >= 6 and not _dviol, f"-> {_dviol}")

    # ── 7) Direct QFModelPatcher VRAM coexistence + retire chokepoint AST + sweep + create guard ──
    _p_vram = KreaL.load("rt-krea2-turbo-quantfunc-int4.safetensors", "krea2-turbo-int4")[0]
    _ = _p_vram.model._qf.lib
    check("direct QFModelPatcher reports resident VRAM, unloads via partially_unload, detach(False) keeps, detach(True) unloads all",
          _p_vram.is_dynamic() is False and _p_vram.loaded_size() == 4096
          and _p_vram.partially_unload(_p_vram.offload_device, 1024) == 1024 and _p_vram.loaded_size() == 3072
          and (_p_vram.detach(unpatch_all=False), _p_vram.loaded_size() == 3072)[1]
          and (_p_vram.detach(unpatch_all=True), _p_vram.loaded_size() == 0)[1])

    _viol = []
    _scan_files = ["__init__.py", "qf_modelpatcher.py", "qf_engine.py"] + [m + ".py" for m in qfn._FAMILY_MODULES]
    for _fn in _scan_files:
        _tree = ast.parse(open(os.path.join(_PLUGIN, _fn), encoding="utf-8").read())
        _allowed = set()
        for _n in ast.walk(_tree):
            if isinstance(_n, ast.FunctionDef) and _n.name == "_retire_handle":
                _allowed = {id(c) for c in ast.walk(_n)}
        for _n in ast.walk(_tree):
            if isinstance(_n, ast.Call):
                _is_d = isinstance(_n.func, ast.Attribute) and _n.func.attr == "destroy"
                _is_a = (isinstance(_n.func, ast.Name) and _n.func.id == "getattr"
                         and len(_n.args) >= 2 and isinstance(_n.args[1], ast.Constant) and _n.args[1].value == "destroy")
                if (_is_d or _is_a) and id(_n) not in _allowed:
                    _viol.append(f"{_fn}:{_n.lineno}")
    check("destroy() is callable ONLY inside _retire_handle (AST)", not _viol, f"-> {_viol}")

    _nr = 0
    for _bad in ({"a": {"step_cache": 1}}, {"nested": [{"sparse": 2}]}, {"lora": []}, {"denoise_only": True},
                 json.dumps({"model_arch": "krea2", "cat_keep_ratio": 0.8})):
        try:
            qfn.qfe._refuse_session_knobs_in_create(_bad)
        except RuntimeError:
            _nr += 1
    qfn.qfe._refuse_session_knobs_in_create({"model_arch": "krea2", "pinned_memory": True})
    check("create guard refuses session & retired keys at any depth (dict and JSON string)", _nr == 5, f"-> {_nr}/5")

    # Passive sweep path
    class _M:
        pass
    ck, other = ("sweep_pkg", "{}"), ("sweep_other", "{}")
    h = _DummyEngine()
    qfn._PIPELINE_CACHE[ck] = h
    m1, m2 = _M(), _M()
    qfn._bind_pipeline_model(ck, m1)
    qfn._bind_pipeline_model(ck, m2)
    del m1
    gc.collect()
    qfn._sweep_dead_pipelines(other)
    kept_live = h.xfm is not None and ck in qfn._PIPELINE_CACHE
    del m2
    gc.collect()
    qfn._sweep_dead_pipelines(other)
    reclaimed = h.xfm is None and ck not in qfn._PIPELINE_CACHE
    check("passive sweep spares handle while any consumer lives and destroys when all die", kept_live and reclaimed)

    # ── 8) Shipped presets file_hints completeness + H3 latent upscale ──
    real_cfg = os.path.join(_PLUGIN, "configs")
    presets = [d for d in sorted(os.listdir(real_cfg)) if os.path.isfile(os.path.join(real_cfg, d, "qf_native.json"))]
    check("exactly 4 shipped model presets with valid file_hints",
          len(presets) == 4 and all(
              isinstance(json.load(open(os.path.join(real_cfg, d, "qf_native.json"), encoding="utf-8")).get("file_hints"), dict)
              for d in presets), f"-> {presets}")
    for d in presets:
        pats = [str(x).lower() for x in json.load(open(os.path.join(real_cfg, d, "qf_native.json"), encoding="utf-8"))["file_hints"]["transformer1"]]
        match_name = pats[0].replace("*", "x") + ("" if pats[0].endswith(".safetensors") else ".safetensors")
        check(f"preset {d} file_hints discriminate both ways",
              any(fnmatch.fnmatch(match_name, p) for p in pats)
              and not any(fnmatch.fnmatch("totally-unrelated.safetensors", p) for p in pats))

    _px = _clf.MiniMaxH3Video.spacial_downscale_ratio
    _lu = qfn.NODE_CLASS_MAPPINGS["QuantFuncH3LatentUpscale"]()
    _aud = torch.randn(1, 32, 2, 208)
    _o, _w2, _h2 = _lu.upscale({"samples": _cnt.NestedTensor((torch.randn(1, 24, 3, 34, 60), _aud))}, 2.0, "bilinear")
    _v2, _a2 = _o["samples"].unbind()
    check("QuantFuncH3LatentUpscale snaps to canvas multiple and preserves audio latent",
          (_w2, _h2) == (1920, 1088) and _w2 % _CM == 0 and _h2 % _CM == 0
          and (_v2.shape[-2] * _px, _v2.shape[-1] * _px) == (_h2, _w2) and torch.equal(_a2, _aud))

    shutil.rmtree(tmp, ignore_errors=True)
    print("LOADER_DISPATCH:", "PASS" if bad == 0 else f"FAIL ({bad} wrong)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
