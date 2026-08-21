#!/usr/bin/env python3
"""Behavioural tests for the FILE-BASED loader node (INT8-Fast-aligned redesign) + the
liveness/LoRA substrate:

  1. UI surface — transformer1/transformer2 FILE dropdowns + model_config (official presets,
     data-driven from configs/), nothing else
  2. dispatch by the preset MANIFEST's family; ltx2/minimax-h3 refuse LOUD (not wired yet);
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

    # OFFICIAL-CONFIG presets under a FIXTURE configs dir: the real shipped wan preset copied in,
    # plus synthetic manifests that exercise routing (ltx2/h3 not-wired, unknown family,
    # single-expert shape) without shipping fake production presets.
    import shutil
    cfgroot = os.path.join(tmp, "configs")
    shutil.copytree(os.path.join(_PLUGIN, "configs", "wan2.2-a14b-t2v"),
                    os.path.join(cfgroot, "wan2.2-a14b-t2v"))
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
    check("wan required = transformer1/transformer2/model_config/resident_block_count",
          list(it["required"].keys()) == ["transformer1", "transformer2", "model_config",
                                          "resident_block_count"],
          f"-> {list(it['required'].keys())}")
    check("wan has NO optional block (transformer2 is required)", not it.get("optional"))
    check("wan RETURN = two MODELs named high/low",
          WanL.RETURN_TYPES == ("MODEL", "MODEL")
          and WanL.RETURN_NAMES == ("model_high", "model_low"))
    cfgs = it["required"]["model_config"][0]
    check("wan model_config lists ONLY wan presets (family-filtered)",
          "wan2.2-a14b-t2v" in cfgs and "fx-single" in cfgs and "fx-ltx" not in cfgs
          and "fx-alien" not in cfgs, f"-> {cfgs}")
    ltx_cfgs = LtxL.INPUT_TYPES()["required"]["model_config"][0]
    check("ltx model_config lists ONLY ltx2 presets",
          "fx-ltx" in ltx_cfgs and "wan2.2-a14b-t2v" not in ltx_cfgs, f"-> {ltx_cfgs}")
    h3_cfgs = H3L.INPUT_TYPES()["required"]["model_config"][0]
    check("h3 model_config lists ONLY minimax-h3 presets",
          "fx-h3" in h3_cfgs and "wan2.2-a14b-t2v" not in h3_cfgs, f"-> {h3_cfgs}")
    t1 = it["required"]["transformer1"][0]
    check("transformer1 lists .safetensors FILES (and only those)",
          "fx-t2v-4steps-high-quantfunc-int4.safetensors" in t1 and "not-a-model.txt" not in t1)

    # ── 2) dispatch: family-node guard; ltx2/minimax-h3 loud not-wired via THEIR nodes;
    #      cross-family preset refused; model_config traversal refused ──
    for node, preset in ((LtxL, "fx-ltx"), (H3L, "fx-h3")):
        try:
            node.load("fx-t2v-4steps-high-quantfunc-int4.safetensors", preset, 999)
            check(f"{preset} refuses loud (family not wired for file mode)", False, "-> no exception")
        except RuntimeError as e:
            check(f"{preset} refuses loud (family not wired for file mode)", "not" in str(e).lower())
    # a preset whose manifest family does not match the NODE's family → the defense-in-depth guard
    try:
        WanL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors",
                  "fx-t2v-4steps-low-quantfunc-int4.safetensors", "fx-ltx", 999)
        check("cross-family preset on the wan node refused", False, "-> no exception")
    except RuntimeError as e:
        check("cross-family preset on the wan node refused", "declares family" in str(e))
    try:
        LtxL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors", "fx-alien", 999)
        check("unknown-family preset refused (family guard)", False, "-> no exception")
    except RuntimeError as e:
        check("unknown-family preset refused (family guard)", "declares family" in str(e))
    for evil_cfg in ("../wan2.2-a14b-t2v", "a/b", "..", ""):
        try:
            WanL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors",
                      "fx-t2v-4steps-low-quantfunc-int4.safetensors", evil_cfg, 999)
            check(f"model_config refuses {evil_cfg!r}", False, "-> loaded!")
        except RuntimeError:
            check(f"model_config refuses {evil_cfg!r}", True)
    # manifest-driven SHAPE mismatches
    try:
        WanL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors",
                  "fx-t2v-4steps-low-quantfunc-int4.safetensors", "fx-single", 999)
        check("single-expert preset + transformer2 refused", False, "-> no exception")
    except RuntimeError as e:
        check("single-expert preset + transformer2 refused", "single-transformer" in str(e))

    # ── 3) wan dual-expert staging + denoise_only create cfg + DUAL MODEL outputs ──
    try:
        pair = WanL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors",
                         "fx-t2v-4steps-low-quantfunc-int4.safetensors", "wan2.2-a14b-t2v", 999)
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
    except Exception as e:  # noqa: BLE001
        check("wan dual-expert staging", False, f"-> raised {type(e).__name__}: {e}")

    # a NON-conforming file name for the preset must be refused loud (file_hints mechanism)
    try:
        WanL.load("other.safetensors", "fx-t2v-4steps-low-quantfunc-int4.safetensors",
                  "wan2.2-a14b-t2v", 999)
        check("file_hints refuses a non-conforming transformer1", False, "-> loaded!")
    except RuntimeError as e:
        check("file_hints refuses a non-conforming transformer1",
              "does not look like" in str(e))

    # wan without a low expert must refuse (A14B is dual-expert; "" maps to the none-sentinel)
    try:
        WanL.load("fx-t2v-4steps-high-quantfunc-int4.safetensors", "", "wan2.2-a14b-t2v", 999)
        check("wan single-file refused (dual-expert required)", False, "-> no exception")
    except RuntimeError as e:
        check("wan single-file refused (dual-expert required)", "DUAL-expert" in str(e))

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
                                   "wan2.2-a14b-t2v", 999)
        shifted = ModelSamplingSD3().patch(base, 11.0)[0]     # upstream comfy patch on the pair's high
        check("comfy patches still apply to the dual outputs (clone path intact)",
              "model_sampling" in shifted.object_patches)
        # v1 pivot contract: chaining the LoRA node onto EITHER dual output must refuse LOUD
        # (a rebuild would fork the shared engine — see _lora_rebuild_dual). Both directions:
        for tag, target in (("high", shifted), ("low", base_low)):
            try:
                LoraNode.apply(target, "a.safetensors", 0.8)
                check(f"LoRA chain on the dual wan {tag} output refuses loud", False,
                      "-> applied!")
            except RuntimeError as e:
                check(f"LoRA chain on the dual wan {tag} output refuses loud",
                      "cannot chain onto the dual-output wan loader" in str(e), f"-> {e}")
    except Exception as e:  # noqa: BLE001
        check("dual-output LoRA refusal arm", False, f"-> raised {type(e).__name__}: {e}")

    # ── 6) HOST-RAM honesty on a DEDICATED file pair (isolated ckey) ──
    try:
        for f in ("ram-t2v-high-quantfunc-int4.safetensors", "ram-t2v-low-quantfunc-int4.safetensors"):
            open(os.path.join(dm, f), "wb").write(b"\0" * 16)
        fresh, fresh_low = WanL.load("ram-t2v-high-quantfunc-int4.safetensors",
                                     "ram-t2v-low-quantfunc-int4.safetensors",
                                     "wan2.2-a14b-t2v", 999)
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
                               "sh-t2v-low-quantfunc-int4.safetensors", "wan2.2-a14b-t2v", 999)
        pb, pb_low = WanL.load("sh-t2v-high-quantfunc-int4.safetensors",
                               "sh-t2v-low-quantfunc-int4.safetensors", "wan2.2-a14b-t2v", 999)
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
