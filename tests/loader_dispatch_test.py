#!/usr/bin/env python3
"""Behavioural tests for the SINGLE loader node's newest surfaces — the ones three independent
reviewers flagged as having zero automated coverage:

  1. family DISPATCH + auto-detect, including the engine's transformer_class fallback
  2. the explicit model_type OVERRIDE (it must win even for a package no family claims — the
     escape hatch the tooltip promises; it was measured UNREACHABLE once)
  3. _resolve_package CONTAINMENT (workflow-serializable widget value = untrusted)
  4. adopt_comfy_state_from — an upstream ModelSampling patch must SURVIVE a LoRA rebuild
     (the silent-shift regression this contract exists to prevent)
  5. HOST-RAM accounting honesty — a never-created engine must report 0, not its estimate
  6. SHARED-handle sibling safety — a release under a live sibling must refuse (measured UAF)
  7. the REAL LoRA-node chain — two-instance state transplant + N-chain -> ONE deferred create
  8. the passive SWEEP path — spares a handle while ANY consumer lives, reclaims when all die

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
        # the plugin normally lives at <comfy>/custom_nodes/<pkg>
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


def _make_pkg(root, name, pipeline_class=None, transformer_class=None):
    """Write a minimal model PACKAGE (what the loader's dropdown lists)."""
    pkg = os.path.join(root, name)
    os.makedirs(os.path.join(pkg, "transformer"), exist_ok=True)
    body = {} if pipeline_class is None else {"_class_name": pipeline_class}
    with open(os.path.join(pkg, "model_index.json"), "w") as f:
        json.dump(body, f)
    if transformer_class is not None:
        with open(os.path.join(pkg, "transformer", "config.json"), "w") as f:
            json.dump({"_class_name": transformer_class}, f)
    return pkg


def main():
    qfn, why = _load_plugin()
    if qfn is None:
        return _emit_skip(why)
    if not getattr(qfn, "_IMPORT_OK", False):
        return _emit_skip("plugin imported but its comfy imports failed (_IMPORT_OK False)")

    bad = 0

    def check(label, ok, detail=""):
        nonlocal bad
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}{(' ' + detail) if detail else ''}")
        if not ok:
            bad += 1

    tmp = tempfile.mkdtemp(prefix="qf_loader_test_")
    root = os.path.join(tmp, "models", "diffusion_models")
    os.makedirs(root, exist_ok=True)
    qfn._model_roots = lambda: [root]          # confine the scan to the fixture root

    creates = []
    fake_cache = {}

    def fake_get_engine(model_dir, create_cfg=None):
        # Mirrors the REAL _get_engine contract the liveness layer is built against: ONE handle
        # per (model_dir, cfg) key, REUSED while its pipeline is valid — that sharing is exactly
        # what the sibling-safety arms test, so a fresh-dummy-per-call stub would test nothing.
        ck = (model_dir, json.dumps(create_cfg or {}, sort_keys=True))
        eng = fake_cache.get(ck)
        if eng is not None and eng.pipeline is not None:
            return eng, ck
        creates.append(len((create_cfg or {}).get("lora", [])))
        eng = _DummyEngine()
        fake_cache[ck] = eng
        return eng, ck
    qfn._get_engine = fake_get_engine
    # The family builders CAPTURE their deps at registration time (deps["get_engine"]), so a module
    # -attribute monkeypatch alone would leave them calling the real engine loader — which then
    # fails on this box's CUDA-toolchain guard. Re-register so the stub is what they hold. (This is
    # also the architecture working as intended: families receive their deps, they do not reach
    # back into the package namespace.)
    qfn._FAMILY_BUILDERS.clear()
    qfn._FAMILY_MATCHERS.clear()
    qfn._register_families()

    # ── 1) dispatch + auto-detect, incl. the transformer_class fallback ────────────────────
    _make_pkg(root, "wan_pkg", pipeline_class="WanImageToVideoPipeline")
    _make_pkg(root, "ltx_pkg", pipeline_class="LTX2Pipeline")
    _make_pkg(root, "h3_pkg", pipeline_class="MiniMaxH3Pipeline")
    # a package with NO pipeline class but a wan transformer — the engine's own second detector half
    _make_pkg(root, "xfm_only_pkg", pipeline_class=None,
              transformer_class="WanTransformer3DModel")
    _make_pkg(root, "alien_pkg", pipeline_class="Flux2KleinPipeline")
    # INTERNALLY-INCONSISTENT package: unknown pipeline_class + a wan transformer config. The
    # ENGINE only consults transformer_class when pipeline_class is EMPTY (PipelineLoader
    # detectPipelineKind), so auto-detect must NOT claim this for wan — engine input-parity.
    _make_pkg(root, "mixed_pkg", pipeline_class="Flux2KleinPipeline",
              transformer_class="WanTransformer3DModel")
    # DEDICATED packages for the lifecycle arms: liveness is per-ckey (per package+cfg), and any
    # still-alive patcher from another arm on the same package legitimately pins its handle — the
    # refusal that caused would be the mechanism WORKING, not the property each arm tests.
    _make_pkg(root, "ram_pkg", pipeline_class="WanPipeline")
    _make_pkg(root, "shared_pkg", pipeline_class="WanPipeline")

    for name, want in (("wan_pkg", "wan"), ("ltx_pkg", "ltx2"), ("h3_pkg", "minimax-h3")):
        got, _rep = qfn._detect_family(os.path.join(root, name))
        check(f"auto-detect {name}", got == want, f"-> {got!r} (want {want!r})")
    got, _rep = qfn._detect_family(os.path.join(root, "xfm_only_pkg"))
    check("auto-detect via transformer_class fallback", got == "wan", f"-> {got!r}")
    got, rep = qfn._detect_family(os.path.join(root, "alien_pkg"))
    check("unknown family reports None (does NOT raise)", got is None and "Flux2Klein" in rep,
          f"-> {got!r}, reported={rep!r}")
    got, _rep = qfn._detect_family(os.path.join(root, "mixed_pkg"))
    check("engine input-parity: transformer half ignored when pipeline_class present",
          got is None, f"-> {got!r} (engine would never consult the transformer half here)")

    Loader = qfn.NODE_CLASS_MAPPINGS["QuantFuncNativeLoader"]()

    # auto on a package no family claims -> refuse LOUD
    try:
        Loader.load("alien_pkg")
        check("auto on an unclaimed package refuses", False, "-> no exception")
    except RuntimeError as e:
        check("auto on an unclaimed package refuses", "Flux2Klein" in str(e),
              "-> names the reported class")

    # ── 2) the explicit override must WIN (the escape hatch the tooltip promises) ──────────
    if "wan" in qfn._FAMILY_BUILDERS:
        try:
            out = Loader.load("alien_pkg", model_type="wan")[0]
            check("explicit model_type overrides an unclaimed package",
                  type(out).__name__ == "QFModelPatcher", f"-> {type(out).__name__}")
        except Exception as e:  # noqa: BLE001
            check("explicit model_type overrides an unclaimed package", False, f"-> raised {e!r}")

        # ── 3) containment of the untrusted widget value ──────────────────────────────────
        for evil in ("../../../../etc", "/etc", "wan_pkg/../../..", ""):
            try:
                qfn._resolve_package(evil)
                check(f"containment refuses {evil!r}", False, "-> resolved!")
            except RuntimeError:
                check(f"containment refuses {evil!r}", True)
        try:
            ok = qfn._resolve_package("wan_pkg") == os.path.realpath(os.path.join(root, "wan_pkg"))
            check("containment still resolves a legit package", ok)
        except Exception as e:  # noqa: BLE001
            check("containment still resolves a legit package", False, f"-> {e!r}")

        # ── 4) an upstream ModelSampling patch must survive a LoRA rebuild ────────────────
        try:
            from comfy_extras.nodes_model_advanced import ModelSamplingSD3
            base = Loader.load("wan_pkg")[0]
            patched = ModelSamplingSD3().patch(base, 11.0)[0]
            patched.patch_model()
            before = float(getattr(patched.model.model_sampling, "shift", -1))
            rebuilt = patched.adopt_comfy_state_from(patched)   # same-shape transplant
            rebuilt.patch_model()
            after = float(getattr(rebuilt.model.model_sampling, "shift", -1))
            check("ModelSampling patch survives a patcher rebuild",
                  before == 11.0 and after == 11.0 and "model_sampling" in rebuilt.object_patches,
                  f"-> {before} -> {after}")
        except Exception as e:  # noqa: BLE001
            check("ModelSampling patch survives a patcher rebuild", False, f"-> raised {e!r}")

        # ── 5) HOST-RAM honesty: a never-created engine holds NOTHING ─────────────────────
        try:
            fresh = Loader.load("ram_pkg")[0]
            eng = fresh.model._qf
            never_ram = fresh.loaded_ram_size()
            never_freed = fresh.partially_unload_ram(10 ** 12)
            check("never-created engine reports 0 host RAM", never_ram == 0, f"-> {never_ram}")
            check("never-created engine frees 0 host RAM", never_freed == 0, f"-> {never_freed}")
            _ = eng.lib                      # materialize (what a session begin does)
            fresh.detach(unpatch_all=False)  # comfy dropping the model -> VRAM released
            held = fresh.loaded_ram_size()
            check("evicted engine DOES report its CPU backup", held > 0, f"-> {held}")
            freed = fresh.partially_unload_ram(10 ** 12)
            check("partially_unload_ram frees the backup", freed == held, f"-> {freed}")
            check("released handle re-creates on next use",
                  getattr(eng, "materialized", True) is False)
            # comfy's dynamic path passes a `subsets` kwarg — must not TypeError
            fresh.partially_unload_ram(10 ** 12, subsets=["patches"])
            check("partially_unload_ram accepts comfy's subsets kwarg", True)
        except Exception as e:  # noqa: BLE001
            check("host-RAM accounting", False, f"-> raised {type(e).__name__}: {e}")

        # ── 6) SHARED-handle sibling safety (the round-2 NO-GO, exact measured shape):
        #       two loader nodes on the SAME package share one cached handle; one sibling's
        #       partially_unload_ram must NOT destroy it under the other ─────────────────────
        try:
            import gc
            pa = Loader.load("shared_pkg")[0]
            pb = Loader.load("shared_pkg")[0]
            ra = pa.model._qf.ensure()
            rb = pb.model._qf.ensure()
            check("two loads of one package share the real handle", ra is rb)
            pa.detach(unpatch_all=False)     # evict -> both wrappers now see the CPU backup
            freed_a = pa.partially_unload_ram(10 ** 12)
            check("sibling-shared release REFUSES (freed 0, sibling alive)", freed_a == 0,
                  f"-> {freed_a}")
            check("sibling's pipeline SURVIVES the refused release",
                  pb.model._qf.pipeline is not None)
            check("sibling still honestly reports its backup", pb.loaded_ram_size() > 0)
            # drop sibling B entirely -> A becomes the sole consumer -> release now proceeds
            del pb, rb
            gc.collect()
            freed_a2 = pa.partially_unload_ram(10 ** 12)
            check("sole-consumer release DOES free once the sibling is gone", freed_a2 > 0,
                  f"-> {freed_a2}")
            check("released wrapper reports 0 afterwards", pa.loaded_ram_size() == 0)
            check("released wrapper self-heals on next use",
                  pa.model._qf.ensure().pipeline is not None)
            del pa, ra
            gc.collect()
        except Exception as e:  # noqa: BLE001
            check("shared-handle sibling safety", False, f"-> raised {type(e).__name__}: {e}")

        # ── 7) REAL LoRA-node chain (production call shape): two-instance state transplant +
        #       the deferred-create contract (N chained nodes -> ONE pipeline create) ─────────
        try:
            from comfy_extras.nodes_model_advanced import ModelSamplingSD3
            lora_dir = os.path.join(tmp, "loras")
            os.makedirs(lora_dir, exist_ok=True)
            for f in ("a.safetensors", "b.safetensors"):
                open(os.path.join(lora_dir, f), "wb").write(b"\0" * 16)
            qfn._lora_choices = lambda: ["a.safetensors", "b.safetensors"]
            qfn._resolve_lora = lambda n: os.path.join(lora_dir, n)
            LoraNode = qfn.NODE_CLASS_MAPPINGS["QuantFuncNativeLoRA"]()

            base = Loader.load("wan_pkg")[0]
            shifted = ModelSamplingSD3().patch(base, 11.0)[0]     # upstream comfy patch
            n0 = len(creates)
            chained1 = LoraNode.apply(shifted, "a.safetensors", 0.8)[0]
            chained2 = LoraNode.apply(chained1, "b.safetensors", 0.5)[0]
            check("LoRA chain defers ALL creates (0 so far)", len(creates) == n0,
                  f"-> {len(creates) - n0} eager creates")
            check("LoRA node returns a DIFFERENT patcher (two-instance transplant)",
                  chained2 is not shifted and chained2.model is not shifted.model)
            check("upstream ModelSampling patch transplanted onto the rebuilt patcher",
                  "model_sampling" in chained2.object_patches)
            chained2.patch_model()
            got_shift = float(getattr(chained2.model.model_sampling, "shift", -1))
            check("transplanted shift survives across instances", got_shift == 11.0,
                  f"-> {got_shift}")
            stack = qfn.qfmp.lora_stack_of(chained2)
            check("chained stack accumulated both entries", len(stack) == 2 and
                  stack[0]["scale"] == 0.8 and stack[1]["scale"] == 0.5, f"-> {stack}")
            _ = chained2.model._qf.lib          # what the sampler's first touch does
            check("first touch creates EXACTLY ONE pipeline for the whole chain",
                  len(creates) == n0 + 1, f"-> {len(creates) - n0}")
            check("that one create carries the full accumulated LoRA set", creates[-1] == 2,
                  f"-> {creates[-1]} loras in create_cfg")
        except Exception as e:  # noqa: BLE001
            check("real LoRA-node chain", False, f"-> raised {type(e).__name__}: {e}")

        # ── 8) the passive SWEEP shares the liveness fix (round-3 review: this path had zero
        #       committed coverage). One dead + one alive consumer on a ckey: the sweep must NOT
        #       destroy (the round-2 blind spot — single last-load-wins ref did); all dead: must
        #       destroy + drop the cache entry; keep_key + unbound entries are protected. Drives
        #       the REAL _PIPELINE_CACHE/_bind_pipeline_model/_sweep_dead_pipelines directly ──
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
            qfn._sweep_dead_pipelines(other)          # one consumer dead, one ALIVE
            check("sweep spares the handle while ANY consumer lives",
                  h.pipeline is not None and ck in qfn._PIPELINE_CACHE)
            del m2
            gc.collect()
            qfn._sweep_dead_pipelines(other)          # ALL consumers dead
            check("sweep destroys once ALL consumers are dead",
                  h.pipeline is None and ck not in qfn._PIPELINE_CACHE)
            # keep_key protection: an all-dead entry passed as keep_key must survive
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
            # unbound entry (load in flight): no liveness list yet -> protected
            ck3 = ("sweep_unbound", "{}")
            h3 = _DummyEngine()
            qfn._PIPELINE_CACHE[ck3] = h3
            qfn._sweep_dead_pipelines(other)
            check("sweep never touches an unbound (in-flight) entry",
                  h3.pipeline is not None and ck3 in qfn._PIPELINE_CACHE)
            for k in (ck2, ck3):                       # leave no fixture state behind
                qfn._PIPELINE_CACHE.pop(k, None)
                qfn._PIPELINE_MODELS.pop(k, None)
            # bind dedup: re-binding the SAME live model must not grow the consumer list
            ck4 = ("sweep_dedup", "{}")
            m4 = _M()
            qfn._bind_pipeline_model(ck4, m4)
            qfn._bind_pipeline_model(ck4, m4)
            check("re-binding a live model does not accumulate refs",
                  len(qfn._PIPELINE_MODELS.get(ck4, [])) == 1,
                  f"-> {len(qfn._PIPELINE_MODELS.get(ck4, []))}")
            qfn._PIPELINE_MODELS.pop(ck4, None)
            del m4
        except Exception as e:  # noqa: BLE001
            check("sweep liveness coverage", False, f"-> raised {type(e).__name__}: {e}")
    else:
        check("wan family registered", False, "-> builders: %s" % sorted(qfn._FAMILY_BUILDERS))

    print("LOADER_DISPATCH:", "PASS" if bad == 0 else f"FAIL ({bad} wrong)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
