#!/usr/bin/env python3
"""Batch CPU integration: actual plugin factories + official host; only native ABI doubled."""
import ast
import ctypes
import gc
import sys
import threading
import types
import unittest
import weakref
from unittest import mock

from host_loader_contract_test import plugin, torch

qfe, qfm = plugin.qfe, plugin.qfmp
mm = qfm.comfy.model_management


class Library:
    """The native C ABI double. It exports no grant verb: #751 deleted them from the engine."""
    def __init__(self):
        self.resources = {1: dict(epoch=0, device=0, held=0, state=0)}
        self.events = []
        self.capabilities = 3
        self.full_state = 0
        self.capacity_state = 0
        self.capacity_bytes = 4096
        self.domain_state = 0
        self.quantfunc_last_error = lambda: b"native contract refusal"
        self.quantfunc_resource_acquire_shared = self.shared
        # Function objects permit ctypes signature annotations.
        for name in ("acquire_shared", "prepare", "query", "query_residency", "destroy", "enroll_host",
                     "release_eligible", "release_all", "acquire", "query_lifecycle", "configure",
                     "query_capacity", "query_domain_residency"):
            method = getattr(self, name if name != "acquire_shared" else "shared")
            setattr(self, "quantfunc_resource_" + name, lambda *a, fn=method: fn(*a))
        # The residency-ABI-2 engine's enrollment symbol (#751); the old name is its refusing legacy stub.
        self.quantfunc_resource_enroll_host_v2 = self.quantfunc_resource_enroll_host

    def shared(self, device, version, out):
        self.events.append(("shared", device))
        out._obj.value = 1
        return 0

    def prepare(self, device, version, out):
        key = max(self.resources) + 1
        self.resources[key] = dict(epoch=key, device=device, held=0, state=0)
        self.events.append(("prepare", key))
        out._obj.value = key
        return 0

    def acquire(self, *args):
        raise AssertionError("must retain existing view, not migrate an Attached owner")

    def configure(self, params, pointer):
        self.events.append(("configure", pointer.value))
        self.resources[pointer.value]["recipe"] = params._obj.config_json
        return 0

    def query_capacity(self, pointer, out):
        self.events.append(("capacity", pointer.value))
        out._obj.state = self.capacity_state
        out._obj.component_count = 1
        out._obj.required_persistent_bytes = self.capacity_bytes
        return 0

    def query_domain_residency(self, pointer, out):
        self.events.append(("domain_residency", pointer.value))
        out._obj.state = self.domain_state
        out._obj.resident_bytes = sum(resource["held"] for resource in self.resources.values())
        return 0

    def query(self, pointer, out):
        r = self.resources[pointer.value]
        out._obj.state = r["state"]
        out._obj.device = r["device"]
        out._obj.owner_epoch = r["epoch"]
        out._obj.capabilities = self.capabilities
        out._obj.cca_live = r["held"]
        return 0

    def query_residency(self, pointer, out):
        r = self.resources[pointer.value]
        out._obj.state = r["state"]
        out._obj.resident_bytes = r["held"]
        return 0

    def query_lifecycle(self, pointer, out):
        r = self.resources[pointer.value]
        out._obj.state = 0 if r["state"] == qfe.QUANTFUNC_RESOURCE_CLOSED else r["state"]
        out._obj.phase = r.get("phase", qfe.QUANTFUNC_RESOURCE_PHASE_CLOSED if r["state"] == qfe.QUANTFUNC_RESOURCE_CLOSED
                              else qfe.QUANTFUNC_RESOURCE_PHASE_PREPARED if r["epoch"] else qfe.QUANTFUNC_RESOURCE_PHASE_SHARED)
        return 0

    def destroy(self, pointer):
        self.events.append(("close", pointer.value))

    def enroll_host(self, pointer):
        self.events.append(("enroll", pointer.value))
        return 0

    def release_eligible(self, pointer, requested, out):
        r = self.resources[pointer.value]
        self.events.append(("release", pointer.value, requested))
        out._obj.state = r["state"]
        if not r["state"]:
            freed = min(requested, r["held"])
            r["held"] -= freed
            out._obj.freed_bytes = freed
        return 0

    def release_all(self, pointer, out):
        self.events.append(("full", pointer.value))
        r = self.resources[pointer.value]
        out._obj.state = self.full_state
        if self.full_state == 0:
            out._obj.freed_bytes, r["held"] = r["held"], 0
        return 0


class TrackingRLock:
    """Test lock exposing only whether the current thread owns the cache lock."""
    def __init__(self):
        self._lock = threading.RLock()
        self._local = threading.local()

    def acquire(self, *args, **kwargs):
        acquired = self._lock.acquire(*args, **kwargs)
        if acquired:
            self._local.depth = getattr(self._local, "depth", 0) + 1
        return acquired

    def release(self):
        self._local.depth -= 1
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def held_by_current_thread(self):
        return bool(getattr(self._local, "depth", 0))


class CanonicalIntegration(unittest.TestCase):
    def setUp(self):
        self.lib = Library()
        for patch in (
            mock.patch.dict(qfm._RESOURCE_DOMAINS, {}, clear=True),
            mock.patch.dict(plugin._PIPELINE_CACHE, {}, clear=True),
            mock.patch.dict(plugin._PIPELINE_MODELS, {}, clear=True),
            mock.patch.dict(plugin._PREPARED_CACHE, {}, clear=True),
            mock.patch.object(qfe, "load_lib", return_value=self.lib),
            # the pipeline cache keys on the library THIS process loaded (qf_engine.loaded_so_path), never on
            # the resolver's current answer: resolve_so_path is reached only by load_lib (mocked here)
            mock.patch.object(qfe, "loaded_so_path", return_value="contract.so"),
            mock.patch.object(plugin, "_read_auth", return_value=("", "")),
            # No free memory unless a test says otherwise: ComfyUI's free_memory then unloads what it is asked to.
            mock.patch.object(mm, "get_free_memory", return_value=0),
            mock.patch.object(mm, "current_loaded_models", []),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def wrapper(self, recipe="A", engine=None):
        if engine is None:
            # Deliberately NOT make_engine_factory: custom family factory path.
            engine = qfm.QFLazyEngine(lambda: plugin._get_engine(recipe))
        model = torch.nn.Module()
        model.device = torch.device("cuda:0")
        model._qf = engine
        return qfm.QFModelPatcher(model, model.device, torch.device("cpu"))

    def parallel(self, calls, timeout=3):
        """Start calls together; barriers inside a call establish any finer ordering."""
        start = threading.Barrier(len(calls) + 1)
        results = [None] * len(calls)
        errors = [None] * len(calls)

        def run(index, call):
            try:
                start.wait(timeout)
                results[index] = call()
            except BaseException as error:
                errors[index] = error

        threads = [threading.Thread(target=run, args=(index, call), daemon=True)
                   for index, call in enumerate(calls)]
        for thread in threads:
            thread.start()
        start.wait(timeout)
        for thread in threads:
            thread.join(timeout)
        self.assertFalse(any(thread.is_alive() for thread in threads), "concurrent call deadlocked")
        return results, [error for error in errors if error is not None]

    def warm_native_model(self, recipe="A"):
        patcher = self.wrapper(recipe)
        owner, shared = patcher.model_patches_models()
        self.lib.resources[1]["held"] = 1024

        def create(lib, *, capacity_bytes, prepared_resource, create_params):
            key = prepared_resource._pointer.value
            self.lib.resources[key]["held"] = 3072
            self.lib.resources[key]["phase"] = qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED
            return qfe.QFEngineHandle(lib, ctypes.c_void_p(70 + key), resource=prepared_resource,
                                      capacity_bytes=capacity_bytes)

        with mock.patch.object(qfe.QFEngineHandle, "create", side_effect=create):
            owner.partially_load(owner.load_device, 4096)
        return patcher, owner, shared

    def official_loaded_model(self, patcher):
        """Put the real host wrapper on the real scheduler list for contract tests."""
        if not hasattr(mm, "LoadedModel") or not hasattr(mm, "current_loaded_models"):
            self.skipTest("tested ComfyUI lacks LoadedModel/current_loaded_models")
        loaded = mm.LoadedModel(patcher)
        # model_unload() normally receives these after model_load().  The resource
        # adapter has no torch materialization, so use a no-op finalizer while still
        # executing the official LoadedModel.model_unload implementation.
        loaded.model_finalizer = types.SimpleNamespace(detach=lambda: None)
        loaded.real_model = lambda: patcher.model
        mm.current_loaded_models.append(loaded)

        def remove_loaded():
            if loaded in mm.current_loaded_models:
                mm.current_loaded_models.remove(loaded)

        self.addCleanup(remove_loaded)
        return loaded

    def exercise_shared_full_detach_state(self, state):
        _patcher, _owner, shared = self.warm_native_model()
        loaded_shared = self.official_loaded_model(shared)
        self.assertIn(loaded_shared, mm.current_loaded_models)
        self.assertEqual(loaded_shared.model_loaded_memory(), 1024)
        self.lib.capabilities = 7
        self.lib.full_state = state
        with mock.patch.object(mm, "DISABLE_SMART_MEMORY", True), \
             mock.patch.object(mm, "cleanup_models_gc", return_value=None, create=True), \
             mock.patch.object(mm, "soft_empty_cache", return_value=None, create=True):
            # Comfy's unload hook has no failure channel: free_memory() runs unguarded on the prompt worker
            # (OOM handler, POST /free). Whatever native answered, nothing may escape it, and Comfy completes
            # its own deregistration. MEASURED (host-vram-h3 attempt 8): the old raise ended the worker thread.
            mm.free_memory(1 << 60, shared.load_device)
        self.assertIn(("full", 1), self.lib.events)
        self.assertNotIn(loaded_shared, mm.current_loaded_models)
        self.assertIsNone(loaded_shared.model_finalizer)
        self.assertTrue(shared._resource._finalizer.alive)

    def test_cache_key_is_the_loaded_library_not_the_resolvers_current_answer(self):
        """G-1 (tests-07 re-CR round 2, both reviewers): a newer pair marked while ComfyUI runs moves resolve_so_path's
        answer. The pipeline cache key must stay the library THIS process loaded (else the same weights miss the cache
        and build a second pipeline), and no lookup may re-resolve the pair (each resolve re-hashes host + kernel)."""
        calls = []
        with mock.patch.object(qfe, "resolve_so_path",
                               side_effect=lambda: calls.append(1) or f"/bin/linux/0.0.1{len(calls)}-consumer-cu13/x.so"):
            before = plugin._engine_recipe("pkg-a")
            after = plugin._engine_recipe("pkg-a")            # "after a background install marked a newer pair"
            again = [plugin._engine_recipe("pkg-a") for _ in range(4)]   # the lookups of one sampler run
        self.assertEqual(before[1], after[1])
        self.assertEqual(before[2], after[2])
        self.assertEqual(before[1][0], "contract.so")
        self.assertTrue(all(r[1] == before[1] for r in again))
        self.assertEqual(calls, [], "a cache lookup re-resolved (re-hashed) the engine pair")

    def test_cold_factory_graph_is_canonical_before_any_model_create(self):
        a, same, b = self.wrapper(), self.wrapper(), self.wrapper("B")
        sibling = self.wrapper(engine=a.model._qf)   # a second patcher over the SAME lazy engine
        groups = [p.model_patches_models() for p in (a, same, sibling, a.clone(), b)]
        self.assertTrue(all(group == groups[0] for group in groups[:4]))
        self.assertIs(groups[0][1], groups[4][1])
        self.assertIsNot(groups[0][0], groups[4][0])
        self.assertEqual(a.get_nested_additional_models(), groups[0])
        self.assertEqual([p.loaded_size() for p in (a, same, sibling, b)], [0, 0, 0, 0])
        self.assertEqual(len(self.lib.resources), 3)  # Shared + A + B, not one per clone.
        # Host enrollment is the device's (its Shared view), before any Owned identity exists; it grants nothing.
        self.assertLess(self.lib.events.index(("enroll", 1)), self.lib.events.index(("prepare", 2)))
        self.assertEqual({event[1] for event in self.lib.events if event[0] == "enroll"}, {1})
        self.assertLess(self.lib.events.index(("configure", 2)), self.lib.events.index(("capacity", 2)))
        self.assertEqual(plugin._PIPELINE_CACHE, {})
        self.assertTrue(all(r["held"] == 0 for r in self.lib.resources.values()))

    def test_native_capacity_is_the_only_cold_authority_for_arbitrary_descriptors(self):
        engine = qfm.QFLazyEngine(lambda: plugin._get_engine(
            "descriptor", create_cfg={"lora": [{"path": "opaque"}],
                                      "future_component": {"enabled": True}}))
        patcher = self.wrapper(engine=engine)
        owner, _ = patcher.model_patches_models()
        self.assertEqual(owner.model_size(), self.lib.capacity_bytes)
        self.assertEqual(engine.capacity_bytes, self.lib.capacity_bytes)
        self.assertEqual(plugin._PIPELINE_CACHE, {})

        self.lib.capacity_state = qfe.QUANTFUNC_RESOURCE_CAPACITY_UNSUPPORTED
        unsupported = self.wrapper("unsupported-capacity")
        with mock.patch.object(qfe.QFEngineHandle, "create") as create:
            with self.assertRaises(qfe.NativeContractUnavailable):
                unsupported.model_patches_models()
        create.assert_not_called()

    def test_official_prepare_sampling_cold_request_takes_comfys_estimate_as_floor_then_admits_without_early_create(self):
        """#716 (tests-07 ruling (a), 2026-09-24, over D3): with NO pipeline yet, the reserve is
        max(ComfyUI's own BaseModel estimate, the comfy side), so comfy makes room before the engine's first forward
        (measured without it: LTX-2.5 1920x1088 cold asked 127 MB and hit the admission ceiling at step 0). Never
        the weights (the Owner adapter carries the Prepared capacity), and never a create during the estimate."""
        import comfy.sampler_helpers as sampler_helpers
        import comfy.supported_models as supported_models
        from qf_loader_contract import qf_krea2_modelpatcher as krea2

        # Make persistent backing decisively larger than either request. Capacity belongs to the Owner
        # ledger, never peak demand.
        self.lib.capacity_bytes = 32 << 30
        # 32x32, not 8x8: at 8x8 ComfyUI's torch estimate (~44 MB) sits BELOW the comfy side's fixed base, so
        # max(floor, comfy side) == comfy side and the arm could not tell floor from no-floor (measured: the
        # precondition below failed on both the old and the fixed code). ~700 MB torch vs ~65 MB comfy side here.
        noise_shape = (1, 16, 32, 32)
        cfg = supported_models.Krea2({"image_model": "krea2",
                                     "disable_unet_model_creation": True})
        qfm.ensure_model_config_attrs(cfg)
        lazy = qfm.QFLazyEngine(lambda: plugin._get_engine("cold-official-chain"))
        model = krea2.QFKrea2Model(cfg, lazy, device=torch.device("cuda:0"))
        patcher = qfm.QFModelPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
        full_shape = [noise_shape[0] * 2, *noise_shape[1:]]
        minimum_shape = list(noise_shape)
        comfy_side = model._qf_comfy_side_bytes(full_shape, {})
        torch_estimate = int(super(qfm.QFSessionModelMixin, model).memory_required(full_shape, cond_shapes={}))
        torch_minimum = int(super(qfm.QFSessionModelMixin, model).memory_required(minimum_shape, cond_shapes={}))
        expected_memory = max(torch_estimate, comfy_side)
        expected_minimum = max(torch_minimum, model._qf_comfy_side_bytes(minimum_shape, {}))
        # Discriminating fixture: ComfyUI's torch estimate must exceed the comfy side, or this arm could not
        # tell "floor" from "no floor".
        self.assertGreater(torch_estimate, comfy_side)
        self.assertGreater(self.lib.capacity_bytes,
                           max(expected_memory, expected_minimum))
        observed = {}

        def create(lib, *, capacity_bytes, prepared_resource, create_params):
            key = prepared_resource._pointer.value
            self.lib.resources[key].update(held=capacity_bytes,
                                           phase=qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED)
            return qfe.QFEngineHandle(lib, ctypes.c_void_p(950 + key),
                                      resource=prepared_resource, capacity_bytes=capacity_bytes)

        def admit(models, *, memory_required, minimum_memory_required,
                  force_full_load=False, **_kwargs):
            observed.update(memory_required=memory_required,
                            minimum_memory_required=minimum_memory_required,
                            models=list(models))
            self.assertEqual(plugin._PIPELINE_CACHE, {},
                             "cold estimate must not materialize the native pipeline")
            owner = next(dependency for dependency in patcher.model_patches_models()
                         if dependency._resource.query().owner_epoch)
            observed["actual_delta"] = owner.partially_load(
                owner.load_device, self.lib.capacity_bytes)

        with mock.patch.object(qfe.QFEngineHandle, "create", side_effect=create), \
            mock.patch.object(mm, "load_models_gpu", side_effect=admit):
            # model_options as Comfy's only caller passes it (CFGGuider: the patcher's own);
            # the None default is dereferenced before _prepare_sampling is ever reached.
            real_model, conds, models = sampler_helpers.prepare_sampling(
                patcher, noise_shape, {}, model_options=patcher.model_options)

        self.assertIs(real_model, model)
        self.assertEqual(conds, {})
        self.assertEqual(observed["memory_required"], expected_memory)
        self.assertEqual(observed["minimum_memory_required"], expected_minimum)
        self.assertEqual(observed["memory_required"], torch_estimate)   # the #716 cold floor is taken
        self.assertLess(observed["memory_required"], self.lib.capacity_bytes)
        self.assertLess(observed["minimum_memory_required"], self.lib.capacity_bytes)
        self.assertEqual(observed["actual_delta"], self.lib.capacity_bytes)
        self.assertTrue(lazy.materialized)

    def test_model_config_attrs_are_ensured_without_comfys_missing_attribute_warning(self):
        # comfy's BASE answers a missing attribute through a warning __getattr__ (0.37 dropped scaled_fp8): the ensure
        # step looks up statically -- no warning, the missing default is SET so later reads stay quiet, and a present
        # attribute keeps its value. hasattr() warned here and left the attribute unset (every read warned again).
        import comfy.supported_models as supported_models
        cfg = supported_models.Krea2({"image_model": "krea2", "disable_unet_model_creation": True})
        cfg.optimizations = {"fp8": True}
        with self.assertNoLogs(level="WARNING"):
            self.assertIs(qfm.ensure_model_config_attrs(cfg), cfg)
            self.assertIsNone(cfg.scaled_fp8)
        self.assertEqual(cfg.optimizations, {"fp8": True})

        class Bare:   # a config with none of them and no __getattr__: every default is set
            pass
        bare = qfm.ensure_model_config_attrs(Bare())
        self.assertIsNone(bare.scaled_fp8)
        self.assertEqual(bare.optimizations, {})

    def test_cold_request_is_the_larger_of_comfys_estimate_and_the_engines_create_time_need(self):
        """#738: with an engine that reports its cold create-time need (quantfunc_resource_vram_need_bytes), the cold
        reserve is comfy side + that need, so ComfyUI evicts its idle models before our create (MEASURED without it:
        "engine need 0 MB cold" kept a 4.4 GB idle TE resident and Krea-2's create failed physically on a 6 GB card).
        Comfy's own estimate stays the FLOOR: the create-time need carries no forward working set and the first forward
        follows this one ask (#716: LTX-2.5 1920x1088 cold). Both ways: a large need wins, a small one leaves the floor.
        Never a create during the estimate; an engine without the entry keeps the floor alone
        (test_official_prepare_sampling_cold_request_takes_comfys_estimate_as_floor...)."""
        import comfy.sampler_helpers as sampler_helpers
        import comfy.supported_models as supported_models
        from qf_loader_contract import qf_krea2_modelpatcher as krea2

        self.lib.capacity_bytes = 32 << 30
        noise_shape = (1, 16, 32, 32)
        full_shape = [noise_shape[0] * 2, *noise_shape[1:]]
        for cold_need, floor_wins in ((3 << 30, False), (16 << 20, True)):
            with self.subTest(cold_need_mb=cold_need >> 20):
                asked = []

                def vram_need_bytes(pointer, out, cold_need=cold_need, asked=asked):
                    asked.append(pointer)
                    out._obj.value = cold_need
                    return 0
                self.lib.quantfunc_resource_vram_need_bytes = vram_need_bytes
                cfg = supported_models.Krea2({"image_model": "krea2", "disable_unet_model_creation": True})
                qfm.ensure_model_config_attrs(cfg)
                lazy = qfm.QFLazyEngine(lambda: plugin._get_engine("cold-need-chain"))
                model = krea2.QFKrea2Model(cfg, lazy, device=torch.device("cuda:0"))
                patcher = qfm.QFModelPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
                comfy_side = model._qf_comfy_side_bytes(full_shape, {})
                torch_estimate = int(super(qfm.QFSessionModelMixin, model).memory_required(full_shape, cond_shapes={}))
                # discriminating: exactly one of the two terms is the larger
                self.assertEqual(torch_estimate > comfy_side + cold_need, floor_wins)
                observed = {}

                def admit(models, *, memory_required, minimum_memory_required, force_full_load=False, **_kwargs):
                    observed.update(memory_required=memory_required)
                    self.assertEqual(plugin._PIPELINE_CACHE, {}, "cold estimate must not materialize the native pipeline")

                with mock.patch.object(mm, "load_models_gpu", side_effect=admit):
                    sampler_helpers.prepare_sampling(patcher, noise_shape, {}, model_options=patcher.model_options)
                self.assertTrue(asked, "the cold need was never asked")
                self.assertEqual(observed["memory_required"], max(torch_estimate, comfy_side + cold_need))
                self.assertFalse(lazy.materialized)

    def test_cold_need_status_split_names_why_the_engine_cannot_say(self):
        """#738 plugin CR D-V2: the cold need's status is never folded into one silent None. An older library (no entry)
        is silent by design; INVALID_ARG is a contract violation and raises; UNSUPPORTED says the layout is not
        estimable and why (the engine's own reason, plugin CR C-L2), any other status carries the engine's own error.
        Each reason is warned about ONCE per resource, and none of them reads as 0 or as an older library."""
        resource = qfe.NativeResource(self.lib, ctypes.c_void_p(1))
        self.assertEqual(resource.cold_vram_need_bytes(), qfe.ColdNeed(None, None))
        status = [qfe.QUANTFUNC_OK]

        def need(pointer, out):
            out._obj.value = (7 << 20) if status[0] == qfe.QUANTFUNC_OK else 0
            return status[0]
        self.lib.quantfunc_resource_vram_need_bytes = need
        self.lib.quantfunc_last_error = lambda: b"resource busy"
        with mock.patch.object(qfe, "say") as say:
            self.assertEqual(resource.cold_vram_need_bytes(), qfe.ColdNeed(7 << 20, None))
            status[0] = qfe.QUANTFUNC_ERROR_INVALID_ARG
            with self.assertRaisesRegex(RuntimeError, "cold VRAM need refused: resource busy"):
                resource.cold_vram_need_bytes()
            status[0] = qfe.QUANTFUNC_ERROR_UNSUPPORTED
            self.lib.quantfunc_last_error = lambda: b"no plan models this layout"
            for _ in range(2):
                self.assertEqual(resource.cold_vram_need_bytes(),
                                 qfe.ColdNeed(None, "not estimable for this layout: no plan models this layout"))
            status[0] = 5   # the engine's INTERNAL (a busy resource, an engine failure)
            self.lib.quantfunc_last_error = lambda: b"resource busy"
            for _ in range(2):
                self.assertEqual(resource.cold_vram_need_bytes(),
                                 qfe.ColdNeed(None, "unknown (engine status 5: resource busy)"))
        self.assertEqual([c.args[0] for c in say.call_args_list], [
            "[qf_native] WARNING the engine's cold VRAM need is not estimable for this layout: no plan models this "
            "layout; ComfyUI's own estimate is the floor",
            "[qf_native] WARNING the engine's cold VRAM need is unknown (engine status 5: resource busy); ComfyUI's "
            "own estimate is the floor"])

    def test_load_delta_never_sums_per_resource_residency(self):
        patcher = self.wrapper("coherent-domain")
        owner, _ = patcher.model_patches_models()
        self.lib.resources[1]["held"] = 512
        # Domain backing that no Python-side member reports: only the coherent native snapshot
        # sees it, so a policy that sums per-resource residency produces DIFFERENT numbers below.
        peer = max(self.lib.resources) + 1
        self.lib.resources[peer] = dict(epoch=peer, device=0, held=256, state=0)

        def create(lib, *, capacity_bytes, prepared_resource, create_params):
            key = prepared_resource._pointer.value
            self.lib.resources[key]["held"] = 3072
            self.lib.resources[1]["held"] = 1024
            self.lib.resources[peer]["held"] = 384
            self.lib.resources[key]["phase"] = qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED
            return qfe.QFEngineHandle(lib, ctypes.c_void_p(520 + key),
                                      resource=prepared_resource, capacity_bytes=capacity_bytes)

        before_queries = len([event for event in self.lib.events
                              if event[0] == "domain_residency"])
        with mock.patch.object(qfe.QFEngineHandle, "create", side_effect=create):
            # Coherent delta (3072 + 1024 + 384) - (0 + 512 + 256); a per-resource sum says 3584.
            self.assertEqual(owner.partially_load(owner.load_device, 4096), 3712)
        domain_queries = [event for event in self.lib.events if event[0] == "domain_residency"]
        # before-load snapshot, after-load snapshot, both through the Shared view
        self.assertEqual(len(domain_queries) - before_queries, 2)
        self.assertTrue(all(event[1] == 1 for event in domain_queries[before_queries:]))

    def test_foreign_additional_models_and_patches_are_preserved(self):
        p = self.wrapper()
        other = qfm.comfy.model_patcher.ModelPatcher(torch.nn.Module(), torch.device("cpu"), torch.device("cpu"))
        p.set_additional_models("other-plugin", [other])
        native = p.model_patches_models()
        self.assertEqual(set(p.get_nested_additional_models()), {other, *native})
        clone = p.clone()
        self.assertEqual(clone.get_additional_models_with_key("quantfunc.native_resources"), native)
        self.assertEqual(len(clone.get_additional_models_with_key("other-plugin")), 1)
        self.assertTrue(other.is_clone(clone.get_additional_models_with_key("other-plugin")[0]))

    def test_logical_patcher_uses_official_torch_ledger_without_native_double_count(self):
        engine = qfm.QFLazyEngine(lambda: plugin._get_engine("A"))
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
        model.device = torch.device("cpu")
        model._qf = engine
        patcher = qfm.QFModelPatcher(model, torch.device("cpu"), torch.device("cpu"))

        owner, shared = patcher.model_patches_models()
        self.assertEqual([patcher.model_size(), owner.model_size(), shared.model_size()],
                         [16, 4096, 0])
        self.assertEqual([patcher.loaded_size(), owner.loaded_size(), shared.loaded_size()],
                         [0, 0, 0])
        self.assertEqual(patcher.partially_load(torch.device("cpu"), 10 ** 30), 16)
        self.assertEqual([patcher.loaded_size(), owner.loaded_size(), shared.loaded_size()],
                         [16, 0, 0])
        self.assertEqual(patcher.partially_unload(torch.device("cpu"), 16), 16)
        self.assertEqual([patcher.loaded_size(), owner.loaded_size(), shared.loaded_size()],
                         [0, 0, 0])
        patcher.detach(True)

    def test_common_owner_adapter_materializes_once_with_actual_delta(self):
        p = self.wrapper()
        owner, shared = p.model_patches_models()
        self.assertEqual(owner.model_size(), 4096)
        self.assertEqual(shared.model_size(), 0)
        self.lib.resources[1]["held"] = 512
        materialized = []

        def create(lib, *, capacity_bytes, prepared_resource, create_params):
            key = prepared_resource._pointer.value
            materialized.append(key)
            self.lib.resources[key]["held"] = 3072
            self.lib.resources[1]["held"] += 512
            self.lib.resources[key]["phase"] = qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED
            return qfe.QFEngineHandle(lib, ctypes.c_void_p(77), resource=prepared_resource,
                                      capacity_bytes=capacity_bytes)

        with mock.patch.object(qfe.QFEngineHandle, "create", side_effect=create):
            self.assertEqual(shared.partially_load(shared.load_device, 4096), 0)
            self.assertEqual(owner.partially_load(owner.load_device, 4096), 3584)
            self.assertEqual(owner.partially_load(owner.load_device, 0), 0)
            self.assertEqual(owner.loaded_size(), 3072)
            self.assertEqual(shared.loaded_size(), 1024)
            self.assertIs(p.model._qf.ensure(), plugin._PIPELINE_CACHE[next(iter(plugin._PIPELINE_CACHE))])
        self.assertEqual(materialized, [2])

    def test_negative_budget_shrinks_the_model_before_the_run(self):
        """Comfy's negative budget = "shrink by this much, then run" (a stock patcher unloads that much too): the
        release_eligible walk frees it, and the load itself sets no budget."""
        patcher = self.wrapper("negative-budget")
        owner, _ = patcher.model_patches_models()
        key = owner._resource._pointer.value
        self.lib.resources[key].update(held=64, phase=qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED)
        del self.lib.events[:]
        self.assertEqual(owner.partially_load(owner.load_device, -32), 0)
        self.assertEqual(self.lib.resources[key]["held"], 32)
        self.assertEqual([event for event in self.lib.events if event[0] == "release"], [("release", key, 32)])

    def test_prepare_candidate_and_materialize_have_no_lock_order_inversion(self):
        patcher = self.wrapper("A")
        owner, _ = patcher.model_patches_models()
        candidate_ready = threading.Event()
        domain_materialize_ready = threading.Event()
        continue_both = threading.Event()
        original_domain = qfm._resource_domain
        original_materialize = qfm.QFLazyEngine._materialize_prepared
        results = {}
        errors = []

        def gated_domain(lib, device, *, prepare=False):
            if prepare and threading.current_thread().name == "qf-candidate-B":
                candidate_ready.set()
                if not continue_both.wait(3):
                    raise AssertionError("candidate barrier was not continued")
            return original_domain(lib, device, prepare=prepare)

        def gated_materialize(engine, entry):
            domain_materialize_ready.set()
            if not continue_both.wait(3):
                raise AssertionError("materialize barrier was not continued")
            return original_materialize(engine, entry)

        def create(lib, *, capacity_bytes, prepared_resource, create_params):
            key = prepared_resource._pointer.value
            self.lib.resources[key]["held"] = capacity_bytes
            self.lib.resources[key]["phase"] = qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED
            return qfe.QFEngineHandle(lib, ctypes.c_void_p(80 + key), resource=prepared_resource,
                                      capacity_bytes=capacity_bytes)

        def prepare_b():
            token = qfe.FACTORY_PREPARE_ONLY.set(True)
            try:
                results["B"] = plugin._get_engine("B")
            except BaseException as error:
                errors.append(error)
            finally:
                qfe.FACTORY_PREPARE_ONLY.reset(token)

        def load_a():
            try:
                results["A"] = owner.partially_load(owner.load_device, 4096)
            except BaseException as error:
                errors.append(error)

        with mock.patch.object(qfm, "_resource_domain", side_effect=gated_domain), \
             mock.patch.object(qfm.QFLazyEngine, "_materialize_prepared", side_effect=gated_materialize,
                               autospec=True), \
             mock.patch.object(qfe.QFEngineHandle, "create", side_effect=create):
            candidate = threading.Thread(target=prepare_b, name="qf-candidate-B", daemon=True)
            materialize = threading.Thread(target=load_a, name="qf-materialize-A", daemon=True)
            candidate.start()
            self.assertTrue(candidate_ready.wait(3), "candidate did not reach domain boundary")
            materialize.start()
            self.assertTrue(domain_materialize_ready.wait(3),
                            "materializer did not acquire the domain path")
            continue_both.set()
            candidate.join(3)
            materialize.join(3)
            self.assertFalse(candidate.is_alive() or materialize.is_alive(),
                             "prepare/materialize opposing paths deadlocked")

        self.assertEqual(errors, [])
        self.assertEqual(results["A"], 4096)
        self.assertIsInstance(results["B"][0], qfm.QFPreparedEntry)
        self.assertEqual(len(plugin._PIPELINE_CACHE), 1)

    def test_engine_identity_lock_is_short_at_all_native_boundaries(self):
        case = self
        lock = TrackingRLock()
        original_prepared = qfm.QFPreparedEntry
        observed = {"construct": 0, "create": 0}

        class ProbedPrepared(original_prepared):
            def __init__(self, *args, **kwargs):
                case.assertFalse(lock.held_by_current_thread())
                observed["construct"] += 1
                super().__init__(*args, **kwargs)

        def checked_create(lib, *, capacity_bytes, prepared_resource, create_params):
            case.assertFalse(lock.held_by_current_thread())
            observed["create"] += 1
            key = prepared_resource._pointer.value
            self.lib.resources[key]["held"] = capacity_bytes
            self.lib.resources[key]["phase"] = qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED
            return qfe.QFEngineHandle(lib, ctypes.c_void_p(90 + key), resource=prepared_resource,
                                      capacity_bytes=capacity_bytes)

        with mock.patch.object(plugin, "_ENGINE_IDENTITY_LOCK", lock), \
             mock.patch.object(qfm, "QFPreparedEntry", ProbedPrepared), \
             mock.patch.object(qfe.QFEngineHandle, "create", side_effect=checked_create):
            patcher = self.wrapper("lock-probe")
            owner, _ = patcher.model_patches_models()
            owner.partially_load(owner.load_device, 4096)

        self.assertEqual(observed["construct"], 1)
        self.assertEqual(observed["create"], 1)

    def test_hot_lookup_pin_wins_before_retire_and_retire_refuses(self):
        """The lookup winner publishes its consumer pin in the lookup transaction."""
        _, ckey, _, _ = plugin._engine_recipe("pin-wins")
        handle = qfe.QFEngineHandle(self.lib, ctypes.c_void_p(401), 4096)
        consumer = qfm.QFLazyEngine(lambda: plugin._get_engine("pin-wins"))
        plugin._PIPELINE_CACHE[ckey] = handle
        pin_published = threading.Event()
        allow_lookup_return = threading.Event()
        retire_started = threading.Event()
        original_pin = plugin._pin_pipeline_consumer_locked
        lookup = []
        retire = []
        errors = []

        def gated_pin(key):
            original_pin(key)
            pin_published.set()
            if not allow_lookup_return.wait(3):
                raise AssertionError("lookup pin barrier was not continued")

        def do_lookup():
            try:
                with qfm._engine_cache_acquisition(consumer):
                    lookup.append(plugin._get_engine("pin-wins"))
            except BaseException as error:
                errors.append(error)

        def do_retire():
            retire_started.set()
            try:
                retire.append(plugin._retire_handle(ckey, handle, None, reason="barrier lookup-wins"))
            except BaseException as error:
                errors.append(error)

        with mock.patch.object(plugin, "_pin_pipeline_consumer_locked", side_effect=gated_pin):
            lookup_thread = threading.Thread(target=do_lookup, daemon=True)
            retire_thread = threading.Thread(target=do_retire, daemon=True)
            lookup_thread.start()
            self.assertTrue(pin_published.wait(3), "lookup never published its in-flight pin")
            retire_thread.start()
            self.assertTrue(retire_started.wait(3), "retire thread did not start")
            allow_lookup_return.set()
            lookup_thread.join(3)
            retire_thread.join(3)

        self.assertFalse(lookup_thread.is_alive() or retire_thread.is_alive(), "lookup/retire deadlocked")
        self.assertEqual(errors, [])
        self.assertEqual(retire, [False])
        self.assertIs(lookup[0][0], handle)
        self.assertIs(plugin._PIPELINE_CACHE[ckey], handle)
        self.assertIn(consumer, plugin._live_pipeline_models(ckey))

    def test_hot_lookup_validation_failure_rolls_back_only_its_inflight_pin(self):
        _, ckey, _, _ = plugin._engine_recipe("pin-rollback")
        handle = qfe.QFEngineHandle(self.lib, ctypes.c_void_p(403), 4096)
        consumer = qfm.QFLazyEngine(lambda: plugin._get_engine("pin-rollback"))
        existing = torch.nn.Module()
        plugin._PIPELINE_CACHE[ckey] = handle
        plugin._bind_pipeline_model(ckey, existing)

        with self.assertRaisesRegex(RuntimeError, "retained native owner identity"):
            consumer.prepare_resource()

        self.assertEqual(plugin._live_pipeline_models(ckey), [existing])
        self.assertIs(plugin._PIPELINE_CACHE[ckey], handle)

    def test_hot_lookup_never_returns_a_bare_unpinned_handle(self):
        _, ckey, _, _ = plugin._engine_recipe("bare-hot")
        handle = qfe.QFEngineHandle(self.lib, ctypes.c_void_p(404), 4096)
        plugin._PIPELINE_CACHE[ckey] = handle
        with self.assertRaisesRegex(qfe.NativeContractUnavailable, "consumer acquisition"):
            plugin._get_engine("bare-hot")
        self.assertIs(plugin._PIPELINE_CACHE[ckey], handle)

    def test_retire_wins_then_same_recipe_gets_a_new_prepared_epoch(self):
        """A retired Attached/Closed identity remains accounting-only, never creatable."""
        retiring = self.wrapper("retire-wins")
        retiring.model_patches_models()
        entry = retiring.model._qf._prepared_entry
        _, ckey, _, _ = plugin._engine_recipe("retire-wins")
        handle = qfe.QFEngineHandle(self.lib, ctypes.c_void_p(402), 4096,
                                    resource=entry.resource)
        plugin._PIPELINE_CACHE[ckey] = handle
        plugin._PIPELINE_MODELS[ckey] = [weakref.ref(retiring.model._qf)]
        removed_before_destroy = threading.Event()
        allow_destroy = threading.Event()
        retire = []
        lookup = []
        errors = []
        original_destroy = handle.destroy

        def blocked_destroy():
            removed_before_destroy.set()
            if not allow_destroy.wait(3):
                raise AssertionError("retire destroy barrier was not continued")
            original_destroy()

        handle.destroy = blocked_destroy

        def do_retire():
            try:
                retire.append(plugin._retire_handle(
                    ckey, handle, retiring.model._qf, reason="barrier retire-wins"))
            except BaseException as error:
                errors.append(error)

        retrying = self.wrapper("retire-wins")

        def do_lookup():
            try:
                lookup.append(retrying.model._qf.prepare_resource())
            except BaseException as error:
                errors.append(error)

        retire_thread = threading.Thread(target=do_retire, daemon=True)
        lookup_thread = threading.Thread(target=do_lookup, daemon=True)
        retire_thread.start()
        self.assertTrue(removed_before_destroy.wait(3), "retire did not remove before teardown")
        self.assertNotIn(ckey, plugin._PIPELINE_CACHE)
        lookup_thread.start()
        lookup_thread.join(3)
        self.assertFalse(lookup_thread.is_alive(), "lookup waited on native teardown")
        allow_destroy.set()
        retire_thread.join(3)

        self.assertFalse(retire_thread.is_alive(), "retire did not finish")
        self.assertEqual(errors, [])
        self.assertEqual(retire, [True])
        replacement = lookup[0][0]
        self.assertIsInstance(replacement, qfm.QFPreparedEntry)
        self.assertIsNot(replacement, entry)
        self.assertIsNot(replacement.resource, entry.resource)
        self.assertNotEqual(replacement._owner_adapter._owner_epoch,
                            entry._owner_adapter._owner_epoch)
        self.assertFalse(entry._cache_usable)
        self.assertTrue(entry.resource._finalizer.alive,
                        "retired identity remains retained for residual accounting/reclaim")

    def test_concurrent_same_key_prepare_retires_loser_outside_cache_lock(self):
        first, second = self.wrapper("same-key"), self.wrapper("same-key")
        prepare_barrier = threading.Barrier(2)
        original_prepare = self.lib.quantfunc_resource_prepare
        original_close = qfe.NativeResource.close
        original_retire = qfm.QFPreparedEntry.retire_unpublished
        lock = TrackingRLock()
        closes = []
        retired = []
        close_entered = threading.Event()
        allow_close = threading.Event()

        def gated_prepare(device, version, out):
            prepare_barrier.wait(3)
            return original_prepare(device, version, out)

        def checked_retire(entry):
            self.assertFalse(lock.held_by_current_thread())
            retired.append(entry)  # Keep the loser strongly reachable after cleanup.
            return original_retire(entry)

        def checked_close(resource):
            self.assertFalse(lock.held_by_current_thread())
            key = resource._pointer.value  # close() consumes the pointer holder (value -> None)
            closes.append(key)
            close_entered.set()
            if not allow_close.wait(3):
                raise AssertionError("loser close barrier was not continued")
            result = original_close(resource)
            self.lib.resources[key]["state"] = qfe.QUANTFUNC_RESOURCE_CLOSED
            return result

        self.lib.quantfunc_resource_prepare = gated_prepare
        with mock.patch.object(plugin, "_ENGINE_IDENTITY_LOCK", lock), \
             mock.patch.object(qfm.QFPreparedEntry, "retire_unpublished", side_effect=checked_retire,
                               autospec=True), \
             mock.patch.object(qfe.NativeResource, "close", side_effect=checked_close, autospec=True):
            start = threading.Barrier(3)
            dependencies = [None, None]
            errors = []

            def prepare(index, patcher):
                try:
                    start.wait(3)
                    dependencies[index] = patcher.model_patches_models()
                except BaseException as error:
                    errors.append(error)

            threads = [threading.Thread(target=prepare, args=(0, first), daemon=True),
                       threading.Thread(target=prepare, args=(1, second), daemon=True)]
            for thread in threads:
                thread.start()
            start.wait(3)
            self.assertTrue(close_entered.wait(3), "losing prepared entry never reached close")

            with lock:
                winner = next(iter(plugin._PREPARED_CACHE.values()))
            # The loser left the canonical registry before its view closes (its close is still blocked here).
            self.assertEqual(list(winner._shared_adapter._owners.values()), [winner._owner_adapter])
            allow_close.set()
            for thread in threads:
                thread.join(3)
            self.assertFalse(any(thread.is_alive() for thread in threads), "prepare/retire deadlocked")
        self.lib.quantfunc_resource_prepare = original_prepare
        self.assertEqual(errors, [])
        self.assertIs(first.model._qf._prepared_entry, second.model._qf._prepared_entry)
        self.assertIs(dependencies[0][0], dependencies[1][0])
        prepared_ids = [event[1] for event in self.lib.events if event[0] == "prepare"]
        closed_ids = [event[1] for event in self.lib.events if event[0] == "close"]
        self.assertEqual(len(prepared_ids), 2)
        self.assertEqual(len(set(prepared_ids) & set(closed_ids)), 1)
        self.assertEqual(closes, list(set(prepared_ids) & set(closed_ids)))
        self.assertEqual(len(retired), 1)
        loser = retired[0]
        self.assertFalse(loser._cache_usable)
        self.assertEqual(list(winner._shared_adapter._owners.values()), [winner._owner_adapter])
        self.assertNotIn(loser._owner_adapter, winner._shared_adapter._owners.values())

        creates = []

        def create(lib, *, capacity_bytes, prepared_resource, create_params):
            creates.append(prepared_resource._pointer.value)
            key = prepared_resource._pointer.value
            self.lib.resources[key]["held"] = capacity_bytes
            self.lib.resources[key]["phase"] = qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED
            return qfe.QFEngineHandle(lib, ctypes.c_void_p(100 + key), resource=prepared_resource,
                                      capacity_bytes=capacity_bytes)

        owner = dependencies[0][0]
        with mock.patch.object(qfe.QFEngineHandle, "create", side_effect=create):
            _, errors = self.parallel([
                lambda: owner.partially_load(owner.load_device, 4096),
                lambda: owner.partially_load(owner.load_device, 4096),
            ])
        self.assertEqual(errors, [])
        self.assertEqual(creates, [owner._resource._pointer.value])

    def test_shared_full_detach_ready_never_escapes_comfys_unload(self):
        self.exercise_shared_full_detach_state(qfe.QUANTFUNC_RESOURCE_READY)

    def test_shared_full_detach_busy_never_escapes_comfys_unload(self):
        self.exercise_shared_full_detach_state(qfe.QUANTFUNC_RESOURCE_BUSY)

    def test_shared_full_detach_unknown_never_escapes_comfys_unload(self):
        self.exercise_shared_full_detach_state(qfe.QUANTFUNC_RESOURCE_UNKNOWN)

    def test_preflight_busy_fails_before_pop_and_ready_hot_repeat_has_one_record(self):
        patcher, owner, _ = self.warm_native_model("official-handoff")
        loaded = mm.LoadedModel(owner)
        loaded.real_model = weakref.ref(owner.model)
        loaded.model_finalizer = weakref.finalize(owner.model, lambda: None)
        self.addCleanup(lambda: loaded.model_finalizer.detach()
                        if loaded.model_finalizer is not None else None)
        old_finalizer = loaded.model_finalizer
        mm.current_loaded_models.append(loaded)
        host_patches = (
            mock.patch.object(mm, "get_free_memory", return_value=1 << 40),
            mock.patch.object(mm, "free_memory", return_value=[]),
            mock.patch.object(mm, "cleanup_models_gc", return_value=None),
        )
        for host_patch in host_patches:
            host_patch.start()
            self.addCleanup(host_patch.stop)

        self.lib.domain_state = qfe.QUANTFUNC_RESOURCE_BUSY
        self.short_busy_deadline()
        with self.assertRaisesRegex(RuntimeError, r"domain residency (unavailable|stayed BUSY)"):
            mm.load_models_gpu([owner], force_full_load=True)
        self.assertEqual(mm.current_loaded_models, [loaded])
        self.assertIs(loaded.model_finalizer, old_finalizer)
        self.assertTrue(old_finalizer.alive)

        self.lib.domain_state = qfe.QUANTFUNC_RESOURCE_READY
        for _ in range(10):
            mm.load_models_gpu([owner], force_full_load=True)
            self.assertEqual(mm.current_loaded_models, [loaded])
        self.assertFalse(old_finalizer.alive)
        self.assertTrue(loaded.model_finalizer.alive)

    def test_logical_dependency_preflight_rejects_nonready_before_host_mutation(self):
        patcher, _owner, _shared = self.warm_native_model("logical-preflight")
        sentinel = object()
        mm.current_loaded_models.append(sentinel)
        self.short_busy_deadline()
        for state in (qfe.QUANTFUNC_RESOURCE_BUSY, qfe.QUANTFUNC_RESOURCE_UNKNOWN,
                      qfe.QUANTFUNC_RESOURCE_CLOSED):
            with self.subTest(state=state):
                self.lib.domain_state = state
                with self.assertRaisesRegex(RuntimeError, r"domain residency (unavailable|stayed BUSY)"):
                    patcher.model_patches_models()
                self.assertEqual(mm.current_loaded_models, [sentinel])

        self.lib.domain_state = qfe.QUANTFUNC_RESOURCE_READY
        owner = patcher.model_patches_models()[0]
        self.lib.resources[owner._resource._pointer.value]["phase"] = \
            qfe.QUANTFUNC_RESOURCE_PHASE_CLOSED
        with self.assertRaisesRegex(RuntimeError, "Closed"):
            patcher.model_patches_models()
        self.assertEqual(mm.current_loaded_models, [sentinel])

    def test_prepare_mode_cannot_fall_through_to_native_create(self):
        token = qfe.FACTORY_PREPARE_ONLY.set(True)
        try:
            with self.assertRaisesRegex(RuntimeError, "during dependency preparation"):
                qfe.create_pipeline(self.lib, model_dir="A")
        finally:
            qfe.FACTORY_PREPARE_ONLY.reset(token)

    def test_owner_release_and_logical_detach_do_not_touch_peers(self):
        a, b = self.wrapper(), self.wrapper("B")
        owner, shared = a.model_patches_models()
        peer, _ = b.model_patches_models()
        self.lib.resources[2]["held"] = 512
        self.lib.resources[3]["held"] = 1024
        self.lib.resources[1]["held"] = 2048
        self.assertEqual(owner.partially_unload(torch.device("cpu"), 64), 64)
        self.assertEqual([x.loaded_size() for x in (owner, peer, shared)], [448, 1024, 2048])
        a.detach(False)
        a.detach(True)
        self.assertEqual([x.loaded_size() for x in (owner, peer, shared)], [448, 1024, 2048])
        self.assertEqual([e for e in self.lib.events if e[0] == "release"], [("release", 2, 64)])
        self.assertFalse(any(e[0] == "close" for e in self.lib.events))
        with self.assertRaisesRegex(RuntimeError, "full eviction"):
            owner.detach(True)

    def test_busy_and_unknown_are_not_zero(self):
        p = self.wrapper()
        owner, _ = p.model_patches_models()
        # #704: a value read that STAYS busy is refused after the bounded retry, with its own honest message.
        self.short_busy_deadline()
        refused = "unavailable|stayed BUSY"
        for state in (1, 2):
            self.lib.resources[2]["state"] = state
            if state == qfe.QUANTFUNC_RESOURCE_BUSY:  # never admitted, so nothing materialized: the logged 0
                with self.assertLogs(qfm._log.name, "WARNING"):
                    self.assertEqual(owner.loaded_size(), 0)
            else:
                with self.assertRaisesRegex(RuntimeError, refused):  # UNKNOWN still refuses, never zero
                    owner.loaded_size()
            if state == qfe.QUANTFUNC_RESOURCE_BUSY:  # a release that stays BUSY vouches for no freed bytes
                self.assertEqual(owner.partially_unload(torch.device("cpu"), 64), 0)
            else:
                with self.assertRaisesRegex(RuntimeError, refused):
                    owner.partially_unload(torch.device("cpu"), 64)
            with self.assertRaisesRegex(RuntimeError, refused):
                p.model_patches_models()

    def test_full_detach_and_an_incomplete_native_release_stays_inside_the_hook(self):
        p = self.wrapper()
        owner, shared = p.model_patches_models()
        self.lib.capabilities = 7
        self.lib.resources[2]["held"] = 512
        self.lib.resources[1]["held"] = 2048
        owner.detach(False)
        self.assertFalse(any(e[0] == "full" for e in self.lib.events))
        self.lib.full_state = 1
        owner.detach(True)  # native Busy is the ordinary answer for a live pipeline; it must not raise
        self.assertEqual(owner.loaded_size(), 512)  # and it is never reported as released
        self.assertTrue(owner._resource._finalizer.alive)
        self.lib.full_state = 0
        owner.detach(True)
        self.assertEqual([owner.loaded_size(), shared.loaded_size()], [0, 2048])
        self.assertEqual([e for e in self.lib.events if e[0] == "full"],
                         [("full", 2), ("full", 2)])
        self.assertEqual(owner.model_size(), 4096)

    def test_closed_full_release_does_not_revive_identity(self):
        p = self.wrapper()
        owner, _ = p.model_patches_models()
        self.lib.capabilities = 7
        # Destruction does not make physical counts invalid. The lifecycle
        # query, not a forged Closed physical snapshot, gates reuse.
        self.lib.resources[2]["phase"] = qfe.QUANTFUNC_RESOURCE_PHASE_CLOSED
        self.lib.resources[2]["held"] = 512
        self.assertEqual(owner.loaded_size(), 512)
        owner.detach(True)
        self.assertEqual(owner.loaded_size(), 0)
        self.assertEqual(owner.model_size(), self.lib.capacity_bytes)   # sizing never refuses; only a LOAD does
        with self.assertRaisesRegex(RuntimeError, "Closed resource identity is not reloadable"):
            owner.partially_load(owner.load_device, 4096)

    def destroy_closes_owner(self, residual=0):
        """The engine's quantfunc_destroy: the pipeline goes, its Owned identity is Closed for good, and only what
        retained aliases still pin stays backed (warm_native_model's handle is c_void_p(70 + resource key))."""
        def destroy(pipeline):
            key = pipeline.value - 70
            self.lib.resources[key].update(phase=qfe.QUANTFUNC_RESOURCE_PHASE_CLOSED, held=residual)
            self.lib.events.append(("destroy", key))
        self.lib.quantfunc_destroy = destroy

    def switch_workflows(self, residual=0):
        """The reported sequence. Workflow 1 loads model A and Comfy lists A's owner; the switch drops A's MODEL from
        Comfy's cache; workflow 2's cold create (model B) runs the host-RAM sweep, which destroys A's dead pipeline.
        A's Owned identity is then Closed while Comfy still lists its owner, which is garbage only once gc breaks its
        prepared-entry cycle (Comfy's prompt worker runs gc after a prompt: why a second run passed)."""
        self.lib.capabilities = 7
        self.destroy_closes_owner(residual)
        patcher_a, owner_a, _ = self.warm_native_model("A")
        loaded_a = self.official_loaded_model(owner_a)
        del patcher_a
        gc.collect()
        patcher_b, owner_b, _ = self.warm_native_model("B")
        self.assertIn(("destroy", owner_a._owner_epoch), self.lib.events)   # Closed by the sweep...
        self.assertIn(loaded_a, mm.current_loaded_models)                    # ...and still listed
        return loaded_a, owner_a, owner_b, patcher_b   # workflow 2's MODEL stays live (a later cold create keeps it)

    def test_workflow_switch_never_breaks_the_next_models_load(self):
        """Release blocker (the user's 4090, QI-2.1 and Krea-2 alike): the FIRST run after switching workflows died in
        VAEDecode, whose load_models_gpu -> free_memory sizes EVERY listed model, and model_size of the retired owner
        raised "Closed resource identity is not reloadable". Sizing never runs a load check: the Closed owner keeps its
        Prepared capacity as its size. Its partial unload frees nothing (the retire marked it Closed: nothing is
        eligible), so Comfy's own model_unload falls back to the full detach, whose release_all is the old owner's final
        cleanup, and free_memory drops it."""
        loaded_a, owner_a, owner_b, _patcher_b = self.switch_workflows(residual=512)
        cap, key_a = self.lib.capacity_bytes, owner_a._owner_epoch
        self.assertEqual([loaded_a.model_memory(), loaded_a.model_loaded_memory(), loaded_a.model_offloaded_memory()],
                         [cap, 512, cap - 512])
        before = len(self.lib.events)
        with mock.patch.object(mm, "cleanup_models_gc", return_value=None, create=True), \
             mock.patch.object(mm, "soft_empty_cache", return_value=None, create=True):
            mm.free_memory(256, owner_b.load_device)   # VAEDecode's load_models_gpu (model_management.py:1001)
        after = self.lib.events[before:]
        self.assertNotIn(loaded_a, mm.current_loaded_models)
        self.assertFalse([e for e in after if e[0] == "release" and e[1] == key_a])
        self.assertIn(("full", key_a), after)
        self.assertEqual(owner_a.loaded_size(), 0)

    def test_free_memory_unloads_a_retired_owner_before_the_live_models(self):
        """ComfyUI 0.37's free_memory unloads the largest offloaded first; a Closed owner sized by its capacity is
        taken before every live model (sizing it by what it holds flipped the order: the live models went first)."""
        loaded_a, owner_a, owner_b, _patcher_b = self.switch_workflows()
        _patcher_c, owner_c, _ = self.warm_native_model("C")   # a third, live model (B stays live: not swept)
        self.official_loaded_model(owner_b)
        self.official_loaded_model(owner_c)
        order, unload = [], mm.LoadedModel.model_unload

        def spy(loaded, *args, **kwargs):
            order.append(loaded.model)
            return unload(loaded, *args, **kwargs)
        with mock.patch.object(mm.LoadedModel, "model_unload", spy), \
             mock.patch.object(mm, "cleanup_models_gc", return_value=None, create=True), \
             mock.patch.object(mm, "soft_empty_cache", return_value=None, create=True):
            mm.free_memory(1 << 30, owner_b.load_device)
        self.assertEqual(len(order), 3)
        self.assertIs(order[0], owner_a)
        self.assertEqual({id(m) for m in order[1:]}, {id(owner_b), id(owner_c)})

    def test_retired_owner_is_known_closed_without_a_lifecycle_read(self):
        """The retire marks the identity it Closes, so its partial unload needs no read: with the lifecycle answering
        BUSY past the deadline it still frees nothing and issues no release (a read-only check would fall through to a
        release that a Closed identity answers CLOSED)."""
        loaded_a, owner_a, owner_b, _patcher_b = self.switch_workflows(residual=512)
        self.short_busy_deadline()
        self.busy_first("query_lifecycle")
        before = len(self.lib.events)
        self.assertEqual(owner_a.partially_unload(owner_a.offload_device, 256), 0)
        self.assertFalse([e for e in self.lib.events[before:]
                          if e[0] == "release" and e[1] == owner_a._owner_epoch])

    def test_closed_identity_is_sized_but_refuses_every_load(self):
        """Sizing a Closed identity answers (its capacity, what it holds) and never refuses; every LOAD still refuses
        loudly, before Comfy pops anything."""
        p = self.wrapper()
        owner, _ = p.model_patches_models()
        self.lib.capabilities = 7
        self.lib.resources[2]["phase"] = qfe.QUANTFUNC_RESOURCE_PHASE_CLOSED
        self.lib.resources[2]["held"] = 512
        loaded = self.official_loaded_model(owner)
        cap = self.lib.capacity_bytes
        self.assertEqual([loaded.model_memory(), loaded.model_loaded_memory(), loaded.model_offloaded_memory()],
                         [cap, 512, cap - 512])
        # Closed natively, NOT by our retire (no mark): the partial unload reads the lifecycle itself, frees nothing
        # and issues no release, and Comfy's own model_unload falls back to the full detach (release_all).
        before = len(self.lib.events)
        self.assertEqual(owner.partially_unload(owner.offload_device, 256), 0)
        self.assertFalse([e for e in self.lib.events[before:] if e[0] == "release"])
        self.assertTrue(loaded.model_unload(256))
        mm.current_loaded_models.remove(loaded)      # ...and free_memory pops what model_unload released
        self.assertEqual([owner.model_size(), owner.loaded_size()], [cap, 0])
        with self.assertRaisesRegex(RuntimeError, "Closed resource identity is not reloadable"):
            owner.partially_load(owner.load_device, 4096)
        with self.assertRaisesRegex(RuntimeError, "Closed resource identity"):
            owner.model_patches_models()
        listed = list(mm.current_loaded_models)
        with self.assertRaisesRegex(RuntimeError, "Closed resource identity"):
            mm.load_models_gpu([p])   # sampling with the MODEL that depends on it: refused at dependency expansion
        self.assertEqual(mm.current_loaded_models, listed)

    def test_retired_identity_leaves_comfys_list_without_memory_pressure(self):
        """No free_memory needed: nothing in the plugin keeps a retired owner alive, so once gc runs Comfy's own
        model finalizer (LoadedModel as model_load builds it: weak real_model + cleanup_models) drops its entry."""
        self.lib.capabilities = 7
        self.destroy_closes_owner()
        patcher_a, owner_a, _ = self.warm_native_model("A")
        loaded_a = mm.LoadedModel(owner_a)
        loaded_a.real_model = weakref.ref(owner_a.model)
        loaded_a.model_finalizer = weakref.finalize(owner_a.model, mm.cleanup_models)
        mm.current_loaded_models.append(loaded_a)
        key_a = owner_a._owner_epoch
        del patcher_a, owner_a
        gc.collect()
        self.warm_native_model("B")
        self.assertIn(("destroy", key_a), self.lib.events)
        gc.collect()
        self.assertNotIn(loaded_a, mm.current_loaded_models)

    # --- issue #704: native queries never wait. BUSY means another thread held the lock at that instant; the
    # identity stays valid on BUSY (the engine fills it before its reader runs), values do not. ---

    def answer(self, name, *, times=None, skip=0, status=None, **fields):
        """Override lib.quantfunc_resource_<name>: after `skip` normal calls, the next `times` calls (None = all)
        keep the normal output but set `fields` on it (and return `status` if given). Returns a call counter."""
        attr = "quantfunc_resource_" + name
        original = getattr(self.lib, attr)
        calls, left = [0], [times]

        def overridden(*args):
            calls[0] += 1
            result = original(*args)
            if calls[0] > skip and (left[0] is None or left[0] > 0):
                if left[0] is not None:
                    left[0] -= 1
                for key, value in fields.items():
                    setattr(args[-1]._obj, key, value)
                if status is not None:
                    return status
            return result

        setattr(self.lib, attr, overridden)
        self.addCleanup(setattr, self.lib, attr, original)
        return calls

    def short_busy_deadline(self):
        patch = mock.patch.object(qfm, "_NATIVE_BUSY_DEADLINE_S", 0.05, create=True)
        patch.start()
        self.addCleanup(patch.stop)

    def busy_first(self, name, times=None, skip=0):
        """After `skip` normal calls, lib.quantfunc_resource_<name> answers BUSY having done NOTHING (a lock miss) for
        `times` calls (None = all), then behaves normally again. Returns a call counter."""
        attr = "quantfunc_resource_" + name
        original, calls = getattr(self.lib, attr), [0]

        def busy(*args):
            calls[0] += 1
            if calls[0] <= skip or (times is not None and calls[0] > skip + times):
                return original(*args)
            args[-1]._obj.state = qfe.QUANTFUNC_RESOURCE_BUSY
            return 0

        setattr(self.lib, attr, busy)
        self.addCleanup(setattr, self.lib, attr, original)
        return calls

    def test_704_per_step_ensure_proceeds_on_busy_identity(self):
        patcher, _owner, _shared = self.warm_native_model("704-ensure")
        lazy = patcher.model._qf
        real = lazy.ensure()
        self.answer("query", state=qfe.QUANTFUNC_RESOURCE_BUSY)
        self.assertIs(lazy.ensure(), real)

    def test_704_full_detach_proceeds_on_busy_identity(self):
        """Comfy's unload hook has no failure channel: a raise here ends the server's only prompt worker."""
        _patcher, owner, _shared = self.warm_native_model("704-detach")
        self.lib.capabilities = 7
        self.answer("query", state=qfe.QUANTFUNC_RESOURCE_BUSY)
        owner.detach(True)
        self.assertIn(("full", 2), self.lib.events)

    def test_704_shared_identity_with_epoch_zero_passes_and_a_zeroed_answer_fails(self):
        _patcher, owner, shared = self.warm_native_model("704-identity")
        self.assertEqual((shared._resource.query().owner_epoch, owner._resource.query().owner_epoch), (0, 2))
        busy = self.answer("query", state=qfe.QUANTFUNC_RESOURCE_BUSY)
        shared.preflight_host_load()
        owner.preflight_host_load()
        self.assertGreater(busy[0], 0)
        # An answer the engine never populated: UNKNOWN with every identity field zero. For the Shared view on
        # device 0 that matches the expected epoch and device, so only the missing CAP_QUERY can refuse it.
        self.answer("query", state=qfe.QUANTFUNC_RESOURCE_UNKNOWN, device=0, owner_epoch=0, capabilities=0)
        for adapter in (shared, owner):
            with self.subTest(adapter="shared" if adapter is shared else "owner"):
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    adapter.preflight_host_load()

    def test_704_value_reads_busy_then_ready_proceed(self):
        _patcher, owner, _shared = self.warm_native_model("704-values")
        for name in ("query_lifecycle", "query_domain_residency"):
            with self.subTest(read=name):
                calls = self.answer(name, times=3, state=qfe.QUANTFUNC_RESOURCE_BUSY)
                owner.preflight_host_load()
                self.assertGreaterEqual(calls[0], 4)

    def test_704_value_read_busy_past_the_deadline_is_refused_honestly(self):
        _patcher, owner, _shared = self.warm_native_model("704-deadline")
        self.short_busy_deadline()
        self.answer("query_lifecycle", state=qfe.QUANTFUNC_RESOURCE_BUSY)
        with self.assertRaisesRegex(RuntimeError, r"host preflight lifecycle stayed BUSY for \d+ ms"):
            owner.preflight_host_load()

    def test_704_value_read_closed_or_unknown_is_refused_without_retry(self):
        _patcher, owner, _shared = self.warm_native_model("704-closed")
        for state in (qfe.QUANTFUNC_RESOURCE_CLOSED, qfe.QUANTFUNC_RESOURCE_UNKNOWN):
            with self.subTest(state=state):
                calls = self.answer("query_lifecycle", state=state)
                with self.assertRaisesRegex(RuntimeError, "host preflight lifecycle unavailable"):
                    owner.preflight_host_load()
                self.assertEqual(calls[0], 1)

    def test_704_query_api_error_is_refused(self):
        _patcher, owner, _shared = self.warm_native_model("704-api-error")
        self.answer("query", status=5)
        with self.assertRaisesRegex(RuntimeError, "resource query failed"):
            owner.preflight_host_load()

    def test_704_byte_reads_busy_then_ready_proceed_with_real_bytes(self):
        _patcher, owner, _shared = self.warm_native_model("704-bytes")
        held = self.lib.resources[owner._resource._pointer.value]["held"]
        domain = sum(resource["held"] for resource in self.lib.resources.values())
        for name, read, expected in (
            ("query_residency", owner.loaded_size, held),
            ("query_domain_residency", lambda: qfm._domain_loaded_size(owner), domain),
            ("query_capacity", lambda: self.wrapper("704-capacity").model_patches_models()[0].model_size(),
             self.lib.capacity_bytes),
        ):
            with self.subTest(read=name):
                calls = self.answer(name, times=3, state=qfe.QUANTFUNC_RESOURCE_BUSY)
                self.assertEqual(read(), expected)
                self.assertGreaterEqual(calls[0], 4)

    def test_704_byte_reads_busy_past_the_deadline_are_refused_never_zero(self):
        _patcher, owner, _shared = self.warm_native_model("704-bytes-deadline")
        self.short_busy_deadline()
        # Different native functions AND consumers: one override never reaches the other read. The plugin's own reads
        # are strict; ComfyUI-facing sizing answers by its declared policy instead
        # (test_704_loaded_size_busy_answers_the_last_ready_residency).
        for name, what, read in (("query_residency", "resource residency", owner._resident_bytes),
                                 ("query_domain_residency", "domain residency",
                                  lambda: qfm._domain_loaded_size(owner))):
            with self.subTest(read=name):
                self.answer(name, state=qfe.QUANTFUNC_RESOURCE_BUSY)
                with self.assertRaisesRegex(RuntimeError, what + r" stayed BUSY for \d+ ms"):
                    read()

    def test_704_release_busy_then_ready_frees_and_reports_exactly_the_request(self):
        _patcher, owner, _shared = self.warm_native_model("704-release")
        key = owner._resource._pointer.value
        held = self.lib.resources[key]["held"]
        calls = self.busy_first("release_eligible", times=2)
        self.assertEqual(owner.partially_unload(torch.device("cpu"), 1024), 1024)
        self.assertEqual(self.lib.resources[key]["held"], held - 1024)
        self.assertEqual(calls[0], 3)

    def test_704_release_partial_free_then_busy_reports_only_ready_bytes_and_comfy_falls_back(self):
        """A BUSY release may already have freed bytes it does not report; the retry frees only eligible backing,
        `freed` is only what the READY answer reports, and Comfy's own full-detach fallback then runs cleanly."""
        _patcher, owner, _shared = self.warm_native_model("704-release-partial")
        self.lib.capabilities = 7  # CAP_RELEASE_ALL for Comfy's fallback detach
        key = owner._resource._pointer.value
        held, want = self.lib.resources[key]["held"], 2048
        self.assertGreater(held, want)
        self.answer("release_eligible", times=1, state=qfe.QUANTFUNC_RESOURCE_BUSY)  # frees `want`, reports BUSY
        loaded, reported, real = self.official_loaded_model(owner), [], owner.partially_unload

        def partially_unload(*args, **kwargs):
            reported.append(real(*args, **kwargs))
            return reported[-1]

        with mock.patch.object(owner, "partially_unload", side_effect=partially_unload):
            self.assertTrue(loaded.model_unload(memory_to_free=want))  # fell short -> Comfy's full detach
        self.assertEqual(reported, [held - want])  # the READY retry's bytes only, never the BUSY attempt's
        self.assertEqual(self.lib.resources[key]["held"], 0)
        self.assertIn(("full", key), self.lib.events)

    def test_704_release_busy_past_the_deadline_returns_zero_with_a_logged_busy(self):
        """partially_unload runs inside Comfy's unload path (no failure channel): a persistent BUSY vouches for no
        freed bytes, so it returns 0 and Comfy falls back to its own full detach."""
        _patcher, owner, _shared = self.warm_native_model("704-release-deadline")
        self.short_busy_deadline()
        calls = self.busy_first("release_eligible")
        with self.assertLogs(qfm._log.name, "WARNING") as logs:
            self.assertEqual(owner.partially_unload(torch.device("cpu"), 1024), 0)
        self.assertRegex("\n".join(logs.output), r"resource release stayed BUSY for \d+ ms")
        self.assertGreater(calls[0], 1)

    def test_704_every_comfy_facing_override_declares_a_busy_policy(self):
        """ComfyUI's memory manager calls these on a loaded model, directly or through the base ModelPatcher methods it
        calls (get_nested_additional_models -> get_additional_models). The set is re-derived from the INSTALLED comfy
        source, so a ComfyUI that starts calling another override fails here until that override declares a policy."""
        def parse(path):
            with open(path, encoding="utf-8") as source:
                return ast.parse(source.read())

        def attr_calls(node, on_self=False):
            return {call.func.attr for call in ast.walk(node)
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and (not on_self or getattr(call.func.value, "id", None) == "self")}

        base = qfm.comfy.model_patcher.ModelPatcher
        called = attr_calls(parse(mm.__file__))
        patcher_class = next(node for node in parse(qfm.comfy.model_patcher.__file__).body
                             if isinstance(node, ast.ClassDef) and node.name == base.__name__)
        bodies = {node.name: node for node in patcher_class.body if isinstance(node, ast.FunctionDef)}
        frontier = list(called & bodies.keys())
        while frontier:
            reached = (attr_calls(bodies[frontier.pop()], on_self=True) & bodies.keys()) - called
            called |= reached
            frontier += reached
        declared = set(qfm._COMFY_BUSY_POLICY)
        for cls in (qfm.QFNativeResourcePatcher, qfm.QFModelPatcher):
            overrides = {name for name, value in vars(cls).items()
                         if callable(value) and not name.startswith("__") and hasattr(base, name) and name in called}
            with self.subTest(cls=cls.__name__):
                self.assertTrue(overrides)
                self.assertEqual({f"{cls.__name__}.{name}" for name in overrides} - declared, set())

    def sizing_busy(self, name):
        self.short_busy_deadline()
        return self.busy_first(name)

    def test_704_loaded_size_busy_answers_the_last_ready_residency(self):
        _patcher, owner, _shared = self.warm_native_model("704-size-seeded")
        seen = owner.loaded_size()
        self.sizing_busy("query_residency")
        with self.assertLogs(qfm._log.name, "WARNING") as logs:
            self.assertEqual(owner.loaded_size(), seen)
        self.assertRegex("\n".join(logs.output), r"resource residency stayed BUSY for \d+ ms.*last READY")

    def test_704_loaded_size_busy_never_admitted_answers_a_logged_zero(self):
        owner, _shared = self.wrapper("704-size-fresh").model_patches_models()
        self.sizing_busy("query_residency")
        with self.assertLogs(qfm._log.name, "WARNING") as logs:
            self.assertEqual(owner.loaded_size(), 0)
        self.assertRegex("\n".join(logs.output), r"no READY residency")

    def test_704_model_size_busy_owned_answers_its_prepared_capacity(self):
        """An Owned view's size is its Prepared capacity, fixed at prepare time: sizing reads nothing native, so no
        BUSY answer can reach it (it used to read the lifecycle for a load check that sizing must not run)."""
        _patcher, owner, _shared = self.warm_native_model("704-size-owned")
        self.sizing_busy("query_lifecycle")
        self.busy_first("query_residency")
        with self.assertNoLogs(qfm._log.name, "WARNING"):
            self.assertEqual(owner.model_size(), self.lib.capacity_bytes)

    def test_704_model_size_busy_shared_answers_the_last_ready_residency(self):
        _patcher, _owner, shared = self.warm_native_model("704-size-shared")
        seen = shared.loaded_size()
        self.sizing_busy("query_residency")   # Shared sizing reads only residency (it is never Closed)
        with self.assertLogs(qfm._log.name, "WARNING"):
            self.assertEqual(shared.model_size(), seen)

    def test_704_shared_detach_never_raises_on_a_persistent_busy(self):
        _patcher, _owner, shared = self.warm_native_model("704-detach-shared")
        self.lib.capabilities = 7  # CAP_RELEASE_ALL
        self.answer("query", state=qfe.QUANTFUNC_RESOURCE_BUSY)
        self.lib.full_state = qfe.QUANTFUNC_RESOURCE_BUSY
        shared.detach(True)
        self.assertIn(("full", 1), self.lib.events)

    def test_704_model_load_path_refuses_a_persistent_busy(self):
        """The load path runs only for the model a prompt asked for, and that prompt reports the failure."""
        patcher, _owner, _shared = self.warm_native_model("704-load-refuses")
        self.sizing_busy("query_lifecycle")
        with self.assertRaisesRegex(RuntimeError, "stayed BUSY"):
            patcher.partially_load(patcher.load_device, 4096)

    def test_704_capacity_busy_past_the_deadline_refuses_the_prepare(self):
        """A target still Creating answers BUSY for its whole create: a legitimate refusal, never a zero capacity."""
        self.short_busy_deadline()
        self.answer("query_capacity", state=qfe.QUANTFUNC_RESOURCE_BUSY)
        original_close, closed = qfe.NativeResource.close, []

        def close(resource):  # the explicit close; the view's GC finalizer would also emit a native destroy
            closed.append(resource._pointer.value)
            return original_close(resource)

        with mock.patch.object(qfe.NativeResource, "close", close):
            with self.assertRaisesRegex(RuntimeError, r"resource capacity stayed BUSY for \d+ ms"):
                self.wrapper("704-capacity-deadline").model_patches_models()
        key = next(event[1] for event in self.lib.events if event[0] == "prepare")
        self.assertIn(key, closed)

    def test_704_admission_refuses_a_non_ready_prepared_lifecycle(self):
        """A non-READY lifecycle carries no phase; admission read None as "not Prepared" and skipped materialize()."""
        patcher = self.wrapper("704-admission")
        owner, _shared = patcher.model_patches_models()
        # require_load_contract's read stays normal; the admission's own Prepared-phase read answers UNKNOWN.
        self.answer("query_lifecycle", skip=1, state=qfe.QUANTFUNC_RESOURCE_UNKNOWN)
        with mock.patch.object(qfe.QFEngineHandle, "create") as create:
            with self.assertRaisesRegex(RuntimeError, "admission lifecycle unavailable"):
                owner.partially_load(owner.load_device, 4096)
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
