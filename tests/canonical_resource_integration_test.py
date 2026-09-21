#!/usr/bin/env python3
"""Batch CPU integration: actual plugin factories + official host; only native ABI doubled."""
import ctypes
import os
import sys
import threading
import time
import types
import unittest
import weakref
from unittest import mock

from host_loader_contract_test import plugin, torch

qfe, qfm = plugin.qfe, plugin.qfmp
mm = qfm.comfy.model_management


class Library:
    def __init__(self):
        self.resources = {1: dict(epoch=0, device=0, held=0, enrolled=False, limit=0, state=0)}
        self.events = []
        self.capabilities = 3
        self.full_state = 0
        self.capacity_state = 0
        self.capacity_bytes = 4096
        self.domain_state = 0
        self.domain_grant_state = 0
        self.device_limit = (1 << 64) - 1
        self.quantfunc_last_error = lambda: b"native contract refusal"
        self.quantfunc_resource_acquire_shared = self.shared
        # Function objects permit ctypes signature annotations.
        for name in ("acquire_shared", "prepare", "query", "query_residency", "destroy",
                     "enroll_host", "query_grant", "set_grant", "query_device_grant", "set_device_grant",
                     "release_eligible", "release_all", "acquire", "query_lifecycle", "configure",
                     "query_capacity", "query_domain_residency", "set_domain_grants"):
            method = getattr(self, name if name != "acquire_shared" else "shared")
            setattr(self, "quantfunc_resource_" + name, lambda *a, fn=method: fn(*a))

    def shared(self, device, version, out):
        self.events.append(("shared", device))
        out._obj.value = 1
        return 0

    def prepare(self, device, version, out):
        key = max(self.resources) + 1
        self.resources[key] = dict(epoch=key, device=device, held=0, enrolled=False, limit=0, state=0)
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

    def query_grant(self, pointer, out):
        r = self.resources[pointer.value]
        out._obj.state, out._obj.enrolled = r["state"], int(r["enrolled"])
        out._obj.limit_bytes, out._obj.pending_bytes = r["limit"], 0
        return 0

    def enroll_host(self, pointer, out):
        self.events.append(("enroll", pointer.value))
        self.resources[pointer.value]["enrolled"] = True
        return self.query_grant(pointer, out)

    def set_grant(self, pointer, limit, out):
        self.events.append(("grant", pointer.value, limit))
        self.resources[pointer.value]["limit"] = limit
        return self.query_grant(pointer, out)

    def query_device_grant(self, pointer, out):
        if self.resources[pointer.value]["epoch"]:
            return 2
        out._obj.state, out._obj.enrolled = 0, int(self.resources[pointer.value]["enrolled"])
        out._obj.limit_bytes, out._obj.pending_bytes = self.device_limit, 0
        return 0

    def set_device_grant(self, pointer, limit, out):
        if self.resources[pointer.value]["epoch"]:
            return 2
        self.events.append(("device_grant", pointer.value, limit))
        self.device_limit = limit
        return self.query_device_grant(pointer, out)

    def set_domain_grants(self, shared, owned, command, out):
        request = command._obj
        owner_key = owned.value if owned else None
        self.events.append(("domain_grants", owner_key, request.mask,
                            request.owner_limit_bytes, request.shared_limit_bytes,
                            request.device_limit_bytes))
        out._obj.state = self.domain_grant_state
        out._obj.applied_mask = 0
        if self.domain_grant_state != qfe.QUANTFUNC_RESOURCE_READY:
            return 0
        if request.mask & qfe.QUANTFUNC_RESOURCE_GRANT_OWNER:
            self.resources[owner_key]["limit"] = request.owner_limit_bytes
        if request.mask & qfe.QUANTFUNC_RESOURCE_GRANT_SHARED:
            self.resources[shared.value]["limit"] = request.shared_limit_bytes
        if request.mask & qfe.QUANTFUNC_RESOURCE_GRANT_DEVICE:
            self.device_limit = request.device_limit_bytes
        out._obj.applied_mask = request.mask
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
            mock.patch.dict(os.environ, {"QF_NATIVE_CREATE_EXTRA": ""}),
            mock.patch.object(qfe, "load_lib", return_value=self.lib),
            mock.patch.object(qfe, "resolve_so_path", return_value="contract.so"),
            mock.patch.object(plugin, "_read_auth", return_value=("", "")),
            mock.patch.object(mm, "get_total_memory", return_value=64 << 30),
            # No free memory unless a test says otherwise: growth is then exactly Comfy's allowance, so
            # every number pinned before the inference-reserve term existed stays as it was.
            mock.patch.object(mm, "get_free_memory", return_value=0),
            mock.patch.object(mm, "current_loaded_models", []),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def wrapper(self, recipe="A", engine=None, shadow=False):
        if engine is None:
            # Deliberately NOT make_engine_factory: custom family factory path.
            engine = qfm.QFLazyEngine(lambda: plugin._get_engine(recipe))
        model = torch.nn.Module()
        model.device = torch.device("cuda:0")
        model._qf, model._qf_shadow = engine, shadow
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
        patcher, owner, shared = self.warm_native_model()
        lazy = patcher.model._qf
        loaded_shared = self.official_loaded_model(shared)
        self.assertIn(loaded_shared, mm.current_loaded_models)
        self.assertEqual(loaded_shared.model_loaded_memory(), 1024)
        self.lib.capabilities = 7
        self.lib.full_state = state
        release_entered = threading.Event()
        release_continue = threading.Event()
        original_release = self.lib.quantfunc_resource_release_all

        def blocked_release(pointer, out):
            release_entered.set()
            if not release_continue.wait(3):
                raise AssertionError("release barrier was not continued")
            return original_release(pointer, out)

        self.lib.quantfunc_resource_release_all = blocked_release
        errors = []

        def detach_shared():
            try:
                mm.free_memory(1 << 60, shared.load_device)
            except BaseException as error:
                errors.append(error)

        with mock.patch.object(mm, "DISABLE_SMART_MEMORY", True), \
             mock.patch.object(mm, "cleanup_models_gc", return_value=None, create=True), \
             mock.patch.object(mm, "soft_empty_cache", return_value=None, create=True):
            detach_thread = threading.Thread(target=detach_shared, daemon=True)
            detach_thread.start()
            self.assertTrue(release_entered.wait(3), "Shared detach never reached release_all")
            # First revoke, observed while release_all is still blocked: nothing but
            # production's own pre-release revoke can have produced this state.
            self.assertEqual((self.lib.resources[1]["limit"], self.lib.device_limit), (0, 0))
            self.assertTrue(shared._domain.shared_growth_fenced)
            self.assertIs(mm.current_loaded_models[0], loaded_shared)

            growth = []

            def native_growth_attempt():
                allowed = self.lib.resources[1]["limit"] >= 256 and self.lib.device_limit >= 256
                if allowed:
                    self.lib.resources[1]["held"] += 256
                growth.append(allowed)

            growth_thread = threading.Thread(target=native_growth_attempt, daemon=True)
            growth_thread.start()
            growth_thread.join(3)
            self.assertFalse(growth_thread.is_alive(), "native growth probe did not finish")
            self.assertEqual(growth, [False])

            release_continue.set()
            detach_thread.join(3)
            self.assertFalse(detach_thread.is_alive(), "Shared detach deadlocked")
        self.lib.quantfunc_resource_release_all = original_release
        # Comfy's unload hook has no failure channel: free_memory() runs unguarded on the prompt worker
        # (OOM handler, POST /free). Whatever native answered, nothing may escape it, and Comfy completes
        # its own deregistration. MEASURED (host-vram-h3 attempt 8): the old raise ended the worker thread.
        self.assertEqual(errors, [], "an incomplete native release escaped Comfy's unload hook")
        self.assertNotIn(loaded_shared, mm.current_loaded_models)
        self.assertIsNone(loaded_shared.model_finalizer)

        # The test must NOT revoke here. _revoke_domain_growth writes exactly the values
        # asserted below, so calling it would hide a detach that restored permission
        # after release_all. Observe what production's own detach left behind.
        self.assert_release_left_growth_revoked(shared, lazy)
        self.assertTrue(shared._resource._finalizer.alive)

    RESTORED_PERMISSION = "production restored growth permission after release_all"

    def assert_release_left_growth_revoked(self, shared, lazy):
        released = self.lib.events.index(("full", 1))
        regranted = [event for event in self.lib.events[released + 1:]
                     if (event[0] in ("grant", "device_grant") and event[2])
                     or (event[0] == "domain_grants" and any(event[3:]))]
        self.assertEqual(regranted, [], self.RESTORED_PERMISSION)
        self.assertEqual((self.lib.resources[1]["limit"], self.lib.device_limit), (0, 0),
                         self.RESTORED_PERMISSION)
        self.assertTrue(shared._domain.shared_growth_fenced, self.RESTORED_PERMISSION)
        with self.assertRaises(qfe.NativeContractUnavailable):
            lazy.ensure()
        self.assertEqual((self.lib.resources[1]["limit"], self.lib.device_limit), (0, 0),
                         self.RESTORED_PERMISSION)

    def exercise_restoring_detach_is_caught(self, state):
        """Death rule for the oracle above.

        Mutate production so a failing release_all re-publishes growth, and require
        exercise_shared_full_detach_state to go red for THAT reason. If this ever stops
        failing, the full-detach tests no longer see a restored grant.
        """
        production_detach = qfm.QFNativeResourcePatcher.detach
        restored = []

        def restoring_detach(adapter, unpatch_all=True):
            result = production_detach(adapter, unpatch_all)
            if unpatch_all:
                qfm._publish_domain_grants(adapter, growth_allowance=4096)
                restored.append(adapter)
            return result

        with mock.patch.object(qfm.QFNativeResourcePatcher, "detach", restoring_detach):
            with self.assertRaisesRegex(self.failureException, self.RESTORED_PERMISSION):
                self.exercise_shared_full_detach_state(state)
        # A mutation that never ran proves nothing about the oracle.
        self.assertEqual(len(restored), 1, "mutation never reached the failing release")

    def test_cold_factory_graph_is_canonical_before_any_model_create(self):
        a, same, b = self.wrapper(), self.wrapper(), self.wrapper("B")
        shadow = self.wrapper(engine=a.model._qf, shadow=True)
        groups = [p.model_patches_models() for p in (a, same, shadow, a.clone(), b)]
        self.assertTrue(all(group == groups[0] for group in groups[:4]))
        self.assertIs(groups[0][1], groups[4][1])
        self.assertIsNot(groups[0][0], groups[4][0])
        self.assertEqual(a.get_nested_additional_models(), groups[0])
        self.assertEqual([p.loaded_size() for p in (a, same, shadow, b)], [0, 0, 0, 0])
        self.assertEqual(len(self.lib.resources), 3)  # Shared + A + B, not one per clone.
        self.assertLess(self.lib.events.index(("enroll", 1)), self.lib.events.index(("prepare", 2)))
        self.assertLess(self.lib.events.index(("configure", 2)), self.lib.events.index(("capacity", 2)))
        self.assertLess(self.lib.events.index(("capacity", 2)), self.lib.events.index(("enroll", 2)))
        self.assertEqual(plugin._PIPELINE_CACHE, {})
        self.assertTrue(all(r["held"] == r["limit"] == 0 for r in self.lib.resources.values()))

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

    def test_official_prepare_sampling_uses_cold_host_floor_then_admits_without_early_create(self):
        import comfy.sampler_helpers as sampler_helpers
        import comfy.supported_models as supported_models
        from qf_loader_contract import qf_krea2_modelpatcher as krea2

        # Make persistent backing decisively larger than either request-shaped
        # cold floor. Capacity belongs to the Owner ledger, never peak demand.
        self.lib.capacity_bytes = 32 << 30
        noise_shape = (1, 16, 8, 8)
        cfg = supported_models.Krea2({"image_model": "krea2",
                                     "disable_unet_model_creation": True})
        qfm.ensure_model_config_attrs(cfg)
        lazy = qfm.QFLazyEngine(lambda: plugin._get_engine("cold-official-chain"))
        model = krea2.QFKrea2Model(cfg, lazy, device=torch.device("cuda:0"))
        patcher = qfm.QFModelPatcher(model, torch.device("cuda:0"), torch.device("cpu"))
        full_shape = [noise_shape[0] * 2, *noise_shape[1:]]
        minimum_shape = list(noise_shape)
        expected_memory = max(
            model._qf_comfy_side_bytes(full_shape, {}),
            int(super(qfm.QFSessionModelMixin, model).memory_required(
                full_shape, cond_shapes={})))
        expected_minimum = max(
            model._qf_comfy_side_bytes(minimum_shape, {}),
            int(super(qfm.QFSessionModelMixin, model).memory_required(
                minimum_shape, cond_shapes={})))
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
        self.assertLess(observed["memory_required"], self.lib.capacity_bytes)
        self.assertLess(observed["minimum_memory_required"], self.lib.capacity_bytes)
        self.assertEqual(observed["actual_delta"], self.lib.capacity_bytes)
        self.assertTrue(lazy.materialized)

    def test_domain_actual_and_load_delta_never_sum_per_resource_residency(self):
        patcher = self.wrapper("coherent-domain")
        owner, _ = patcher.model_patches_models()
        owner_key = owner._resource._pointer.value
        self.lib.resources[1]["held"] = 512
        # Domain backing that no Python-side member reports: only the coherent native snapshot
        # sees it, so a policy that sums per-resource residency produces DIFFERENT numbers below.
        # (Forbidding NativeResource.residency outright would be wrong: each resource's OWN
        # ceiling is its own residency + the allowance, and must read it.)
        peer = max(self.lib.resources) + 1
        self.lib.resources[peer] = dict(epoch=peer, device=0, held=256, enrolled=False,
                                        limit=0, state=0)

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
        # Owner 0 + 4096, Shared 512 + 4096, Device = domain actual 768 + 4096.
        # A per-resource sum would publish a Device ceiling of 512 + 4096 = 4608.
        self.assertIn(("domain_grants", owner_key, 7, 4096, 4608, 4864), self.lib.events)
        domain_queries = [event for event in self.lib.events if event[0] == "domain_residency"]
        # before-load snapshot, the admission's domain actual, after-load snapshot
        self.assertEqual(len(domain_queries) - before_queries, 3)
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
        model._qf, model._qf_shadow = engine, False
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

    def test_common_owner_adapter_uses_finite_grant_then_materializes_once_with_actual_delta(self):
        p = self.wrapper()
        owner, shared = p.model_patches_models()
        self.assertEqual(owner.model_size(), 4096)
        self.assertEqual(shared.model_size(), 0)
        self.lib.resources[1]["held"] = 512
        materialized = []

        def create(lib, *, capacity_bytes, prepared_resource, create_params):
            key = prepared_resource._pointer.value
            if self.lib.resources[key]["limit"] < capacity_bytes or self.lib.device_limit < capacity_bytes:
                raise RuntimeError("native finite grant denied cold create")
            self.lib.events.append(("materialize", key))
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
            owner_grant = self.lib.resources[2]["limit"]
            shared_grant = self.lib.resources[1]["limit"]
            # Allowance 0: every ceiling IS its residency, so Device == Owner + Shared here is the
            # true identity, not a summing defect. The no-sum property is pinned at the 4096
            # admission below: Device 4608 = domain 512 + 4096, where summed ceilings give 8704.
            self.assertEqual((owner_grant, shared_grant, self.lib.device_limit), (3072, 1024, 4096))
            self.assertIs(p.model._qf.ensure(), plugin._PIPELINE_CACHE[next(iter(plugin._PIPELINE_CACHE))])
        self.assertEqual(materialized, [2])
        materialize_index = self.lib.events.index(("materialize", 2))
        admission = ("domain_grants", 2, 7, 4096, 4608, 4608)
        self.assertLess(self.lib.events.index(admission), materialize_index)
        self.assertFalse(any(event[0] in ("grant", "device_grant")
                             for event in self.lib.events))

    def test_two_owner_permissions_never_sum_unrealized_allowance_into_device_ceiling(self):
        a, b = self.wrapper(), self.wrapper("B")
        owner_a, shared = a.model_patches_models()
        owner_b, same_shared = b.model_patches_models()
        self.assertIs(shared, same_shared)
        # Nonzero Shared residency is what makes the last assertion of this block able to fail:
        # with Shared at 0 the correct Device ceiling (domain actual + allowance) and the sum of
        # the Owner ceilings are both 8192, so "not the sum" could not be told from "the sum".
        self.lib.resources[1]["held"] = 512
        materialized = []

        def create(lib, *, capacity_bytes, prepared_resource, create_params):
            key = prepared_resource._pointer.value
            if self.lib.resources[key]["limit"] < capacity_bytes or self.lib.device_limit < capacity_bytes:
                raise RuntimeError("native finite grant denied cold create")
            materialized.append(key)
            self.lib.resources[key]["held"] = capacity_bytes
            self.lib.resources[key]["phase"] = qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED
            return qfe.QFEngineHandle(lib, ctypes.c_void_p(70 + key), resource=prepared_resource,
                                      capacity_bytes=capacity_bytes)

        with mock.patch.object(qfe.QFEngineHandle, "create", side_effect=create):
            self.assertEqual(owner_a.partially_load(owner_a.load_device, 4096), 4096)
            self.assertEqual(self.lib.device_limit, 512 + 4096)
            self.assertEqual(owner_b.partially_load(owner_b.load_device, 4096), 4096)
        owner_limits = [self.lib.resources[key]["limit"] for key in (2, 3)]
        self.assertEqual(materialized, [2, 3])
        self.assertEqual(owner_limits, [4096, 4096])
        self.assertEqual(self.lib.resources[1]["limit"], 512 + 4096)
        # domain actual (512 Shared + 4096 owner A) + this admission's allowance
        self.assertEqual(self.lib.device_limit, 4608 + 4096)
        self.assertNotEqual(self.lib.device_limit, sum(owner_limits))

        self.assertEqual(owner_a.partially_unload(torch.device("cpu"), 1024), 1024)
        owner_limits = [self.lib.resources[key]["limit"] for key in (2, 3)]
        self.assertEqual(owner_limits, [0, 4096])
        self.assertEqual(self.lib.resources[1]["limit"], 0)
        self.assertEqual(self.lib.device_limit, 0)
        self.assertTrue(shared._domain.shared_growth_fenced)

        # Delayed peer release must not turn the old aggregate occupancy into
        # realizable Owner or Shared permission after a passive release.
        self.lib.resources[3]["held"] = 0
        owner_growth_allowed = (self.lib.resources[2]["limit"] >= 256 and
                                self.lib.device_limit >= 256)
        shared_growth_allowed = (self.lib.resources[1]["limit"] >= 256 and
                                 self.lib.device_limit >= 256)
        self.assertFalse(owner_growth_allowed)
        self.assertFalse(shared_growth_allowed)

        self.lib.capabilities = 7
        owner_b.detach(True)
        owner_limits = [self.lib.resources[key]["limit"] for key in (2, 3)]
        self.assertEqual(owner_limits, [0, 0])
        self.assertEqual(self.lib.resources[3]["held"], 0)
        self.assertEqual(self.lib.resources[1]["limit"], 0)
        self.assertEqual(self.lib.device_limit, 0)

    def test_grant_is_domain_actual_plus_this_allowance_not_capacity_or_old_permission(self):
        self.lib.capacity_bytes = 20
        patcher = self.wrapper("grant-separation")
        owner, shared = patcher.model_patches_models()
        key = owner._resource._pointer.value
        self.lib.resources[key].update(held=10, limit=40,
                                       phase=qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED)
        self.lib.resources[1].update(held=0, limit=40)
        self.lib.device_limit = 40

        self.assertEqual(owner.partially_load(owner.load_device, 5), 0)
        self.assertEqual([event for event in self.lib.events if event[0] == "domain_residency"][-2:],
                         [("domain_residency", 1), ("domain_residency", 1)])
        self.assertIn(("domain_grants", key, 7, 15, 5, 15), self.lib.events)
        self.assertGreater(self.lib.resources[key]["limit"], self.lib.capacity_bytes // 2)
        self.assertEqual((self.lib.resources[1]["limit"], self.lib.device_limit), (5, 15))

        # A second owner receiving the same unused allowance replaces the
        # aggregate window; it does not add another unmaterialized reservation.
        peer = self.wrapper("grant-peer")
        peer_owner, _ = peer.model_patches_models()
        peer_key = peer_owner._resource._pointer.value
        self.lib.resources[peer_key].update(held=0, limit=99,
                                            phase=qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED)
        self.assertEqual(peer_owner.partially_load(peer_owner.load_device, 5), 0)
        self.assertEqual(self.lib.device_limit, 15)
        self.assertEqual(self.lib.resources[peer_key]["limit"], 5)
        self.assertEqual(self.lib.resources[1]["limit"], 5)

    def test_ceiling_is_what_comfy_left_free_never_below_its_own_weights_budget(self):
        """Comfy's extra_memory = free - minimum_memory_required: the inference reserve is taken OUT.

        Native activations live under the same ceiling as native weights, so the published growth is
        what Comfy left free at this admission. Both directions: free above the budget wins (the
        reserve is granted), free below it never lowers Comfy's own number.
        """
        for free, growth in ((30, 30), (3, 5)):
            with self.subTest(comfy_free=free):
                patcher = self.wrapper(f"inference-reserve-{free}")
                owner, _ = patcher.model_patches_models()
                key = owner._resource._pointer.value
                self.lib.resources[key].update(held=10, limit=0,
                                               phase=qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED)
                self.lib.resources[1].update(held=0, limit=0)
                self.lib.device_limit = 0
                with mock.patch.object(mm, "get_free_memory", return_value=free):
                    self.assertEqual(owner.partially_load(owner.load_device, 5), 0)
                # owner: own residency + growth; Shared: 0 + growth; Device: domain residency + growth
                self.assertEqual((self.lib.resources[key]["limit"], self.lib.resources[1]["limit"],
                                  self.lib.device_limit), (10 + growth, growth, 10 + growth))
                self.lib.resources[key].update(held=0)  # keep the next subTest's domain residency its own

    def test_negative_allowance_shrinks_then_admits_the_run_without_a_weights_budget(self):
        """Comfy's negative budget = "shrink by this much, THEN RUN" (a stock patcher samples in low-VRAM mode).

        MEASURED (host-vram-h3 measure-10): returning after the shrink left growth revoked, and stage 2 of the
        double-sample died seven runs in a row on "requires a positive Owned host grant". Both halves are pinned:
        the release still happens FIRST and under a revoked grant, and the call then ends in a formal admission
        whose growth is what Comfy left free - never a weights budget of its own (free = 0 => limit == residency).
        """
        for free, growth in ((0, 0), (24, 24)):
            with self.subTest(comfy_free=free):
                patcher = self.wrapper(f"negative-allowance-{free}")
                owner, _ = patcher.model_patches_models()
                key = owner._resource._pointer.value
                self.lib.resources[key].update(held=64, limit=128,
                                               phase=qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED)
                self.lib.device_limit = 128
                del self.lib.events[:]

                class OutermostCount:   # every use of the domain lock is a `with`; count depth-0 entries
                    def __init__(self, inner):
                        self.inner, self.depth, self.outermost = inner, 0, 0

                    def __enter__(self):
                        self.inner.acquire()
                        self.outermost += self.depth == 0
                        self.depth += 1

                    def __exit__(self, *exc):
                        self.depth -= 1
                        self.inner.release()

                counted = OutermostCount(owner._domain.transaction_lock)
                with mock.patch.object(owner._domain, "transaction_lock", counted), \
                        mock.patch.object(mm, "get_free_memory", return_value=free):
                    self.assertEqual(owner.partially_load(owner.load_device, -32), 0)
                # The shrink revokes growth DOMAIN-wide; a second acquisition would expose that fenced state to a
                # peer between the release and the re-admission. One Comfy operation = one critical section.
                self.assertEqual(counted.outermost, 1, "shrink and re-admission must share one domain transaction")
                self.assertEqual(self.lib.resources[key]["held"], 32)
                revoke, release = ("domain_grants", key, 7, 0, 0, 0), ("release", key, 32)
                self.assertLess(self.lib.events.index(revoke), self.lib.events.index(release),
                                "the shrink must run under a revoked grant")
                # then the formal admission: own residency + growth; Shared 0 + growth; Device = domain + growth
                self.assertEqual((self.lib.resources[key]["limit"], self.lib.resources[1]["limit"],
                                  self.lib.device_limit), (32 + growth, growth, 32 + growth))
                self.assertGreater(self.lib.resources[key]["limit"], 0, "a fenced engine cannot sample")
                self.assertFalse(owner._domain.shared_growth_fenced)
                self.lib.resources[key].update(held=0)  # keep the next subTest's domain residency its own

    def test_same_domain_lock_serializes_concurrent_owner_load_and_unload(self):
        a, b = self.wrapper(), self.wrapper("B")
        owner_a, shared = a.model_patches_models()
        owner_b, same_shared = b.model_patches_models()
        self.assertIs(shared, same_shared)

        guard = threading.Lock()
        active = 0
        max_active = 0

        def create(lib, *, capacity_bytes, prepared_resource, create_params):
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            key = prepared_resource._pointer.value
            self.lib.resources[key]["held"] += 3072
            self.lib.resources[1]["held"] += 1024
            self.lib.resources[key]["phase"] = qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED
            with guard:
                active -= 1
            return qfe.QFEngineHandle(lib, ctypes.c_void_p(80 + key), resource=prepared_resource,
                                      capacity_bytes=capacity_bytes)

        def parallel(calls):
            gate = threading.Barrier(len(calls) + 1)
            results, errors = [], []
            def run(call):
                try:
                    gate.wait()
                    results.append(call())
                except BaseException as error:
                    errors.append(error)
            threads = [threading.Thread(target=run, args=(call,)) for call in calls]
            for thread in threads:
                thread.start()
            gate.wait()
            for thread in threads:
                thread.join(3)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            return results

        with mock.patch.object(qfe.QFEngineHandle, "create", side_effect=create):
            loaded = parallel([
                lambda: owner_a.partially_load(owner_a.load_device, 4096),
                lambda: owner_b.partially_load(owner_b.load_device, 4096),
            ])
        self.assertEqual(max_active, 1)
        self.assertEqual(sorted(loaded), [4096, 4096])
        actual = sum(resource["held"] for resource in self.lib.resources.values())
        self.assertEqual(sum(loaded), actual)
        self.assertEqual(actual, 8192)
        self.assertEqual(self.lib.device_limit, 8192)

        original_release = self.lib.quantfunc_resource_release_eligible
        active = max_active = 0
        def slow_release(*args):
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            try:
                return original_release(*args)
            finally:
                with guard:
                    active -= 1
        self.lib.quantfunc_resource_release_eligible = slow_release
        unloaded = parallel([
            lambda: owner_a.partially_unload(torch.device("cpu"), 1024),
            lambda: owner_b.partially_unload(torch.device("cpu"), 1024),
        ])
        self.assertEqual(max_active, 1)
        self.assertEqual(sorted(unloaded), [1024, 1024])
        self.assertEqual([self.lib.resources[key]["limit"] for key in (1, 2, 3)], [0, 0, 0])
        self.assertEqual(self.lib.device_limit, 0)
        self.assertTrue(shared._domain.shared_growth_fenced)

    def test_prepare_candidate_and_domain_materialize_have_no_engine_domain_abba(self):
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
                             "engine/domain opposing paths deadlocked")

        self.assertEqual(errors, [])
        self.assertEqual(results["A"], 4096)
        self.assertIsInstance(results["B"][0], qfm.QFPreparedEntry)
        self.assertEqual(len(plugin._PIPELINE_CACHE), 1)

    def test_engine_identity_lock_is_short_at_all_native_boundaries(self):
        case = self
        lock = TrackingRLock()
        original_prepared = qfm.QFPreparedEntry
        original_require = qfm._require_engine_host_grants
        observed = {"construct": 0, "grant": 0, "create": 0}

        class ProbedPrepared(original_prepared):
            def __init__(self, *args, **kwargs):
                case.assertFalse(lock.held_by_current_thread())
                observed["construct"] += 1
                super().__init__(*args, **kwargs)

        def checked_require(engine):
            case.assertFalse(lock.held_by_current_thread())
            observed["grant"] += 1
            return original_require(engine)

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
             mock.patch.object(qfm, "_require_engine_host_grants", side_effect=checked_require), \
             mock.patch.object(qfe.QFEngineHandle, "create", side_effect=checked_create):
            patcher = self.wrapper("lock-probe")
            owner, _ = patcher.model_patches_models()
            owner.partially_load(owner.load_device, 4096)

        self.assertEqual(observed["construct"], 1)
        self.assertGreaterEqual(observed["grant"], 1)
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

    def test_concurrent_same_key_prepare_retires_loser_outside_cache_lock_and_domain_refresh(self):
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
            refresh_started = threading.Event()
            refresh_done = threading.Event()

            def refresh_winner():
                refresh_started.set()
                try:
                    qfm._revoke_domain_growth(winner._owner_adapter)
                except BaseException as error:
                    errors.append(error)
                finally:
                    refresh_done.set()

            refresh_thread = threading.Thread(target=refresh_winner, daemon=True)
            refresh_thread.start()
            self.assertTrue(refresh_started.wait(3), "domain refresh thread did not start")
            self.assertFalse(refresh_done.wait(0.05),
                             "refresh entered while loser retirement held the domain transaction")
            allow_close.set()
            for thread in threads:
                thread.join(3)
            refresh_thread.join(3)
            self.assertFalse(any(thread.is_alive() for thread in [*threads, refresh_thread]),
                             "prepare/retire/refresh deadlocked")
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
        domain = winner._owner_adapter._domain
        with domain.transaction_lock:
            self.assertEqual(list(domain.owners.values()), [winner._owner_adapter])
            self.assertNotIn(loser._owner_adapter, domain.owners.values())

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

    def test_shared_full_detach_ready_keeps_zero_grant(self):
        self.exercise_shared_full_detach_state(qfe.QUANTFUNC_RESOURCE_READY)

    def test_shared_full_detach_busy_keeps_zero_grant(self):
        self.exercise_shared_full_detach_state(qfe.QUANTFUNC_RESOURCE_BUSY)

    def test_shared_full_detach_unknown_keeps_zero_grant(self):
        self.exercise_shared_full_detach_state(qfe.QUANTFUNC_RESOURCE_UNKNOWN)

    # The three arms above are the controls for these two: same exercise, production
    # unmutated, must stay green.
    def test_shared_full_detach_busy_oracle_dies_if_permission_is_restored(self):
        self.exercise_restoring_detach_is_caught(qfe.QUANTFUNC_RESOURCE_BUSY)

    def test_shared_full_detach_unknown_oracle_dies_if_permission_is_restored(self):
        self.exercise_restoring_detach_is_caught(qfe.QUANTFUNC_RESOURCE_UNKNOWN)

    def test_fenced_shared_reopens_only_by_atomic_all_or_none_domain_grant(self):
        patcher, owner, shared = self.warm_native_model()
        lazy = patcher.model._qf
        self.lib.capabilities = 7
        shared.detach(True)
        # Production's detach alone must leave Shared revoked; assert BEFORE the setup
        # revoke below, which would otherwise satisfy these assertions by itself.
        self.assertEqual((self.lib.resources[1]["limit"], self.lib.device_limit), (0, 0))
        self.assertTrue(shared._domain.shared_growth_fenced)
        qfm._revoke_domain_growth(owner)  # setup only: also zero the warm Owner grant
        self.assertEqual(self.lib.resources[owner._resource._pointer.value]["limit"], 0)

        # A newly prepared owner starts at zero. Busy/Unknown from the native
        # atomic command leaves all three authorization layers unchanged.
        newcomer = self.wrapper("fenced-new-owner")
        newcomer_owner, newcomer_shared = newcomer.model_patches_models()
        self.assertIs(newcomer_shared, shared)
        before = (self.lib.resources[newcomer_owner._resource._pointer.value]["limit"],
                  self.lib.resources[1]["limit"], self.lib.device_limit)
        event_start = len(self.lib.events)
        self.lib.domain_grant_state = qfe.QUANTFUNC_RESOURCE_BUSY
        with self.assertRaisesRegex(RuntimeError, "domain grant"):
            newcomer_owner.partially_load(newcomer_owner.load_device, 4096)
        after = (self.lib.resources[newcomer_owner._resource._pointer.value]["limit"],
                 self.lib.resources[1]["limit"], self.lib.device_limit)
        self.assertEqual(after, before)
        self.assertFalse(any(event[0] in ("grant", "device_grant")
                             for event in self.lib.events[event_start:]))
        self.assertTrue(shared._domain.shared_growth_fenced)
        with self.assertRaises(qfe.NativeContractUnavailable):
            lazy.ensure()

        self.lib.domain_grant_state = qfe.QUANTFUNC_RESOURCE_READY

        def create(lib, *, capacity_bytes, prepared_resource, create_params):
            key = prepared_resource._pointer.value
            self.lib.resources[key]["held"] = capacity_bytes
            self.lib.resources[key]["phase"] = qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED
            return qfe.QFEngineHandle(lib, ctypes.c_void_p(500 + key),
                                      resource=prepared_resource, capacity_bytes=capacity_bytes)

        event_start = len(self.lib.events)
        with mock.patch.object(qfe.QFEngineHandle, "create", side_effect=create):
            self.assertEqual(newcomer_owner.partially_load(newcomer_owner.load_device, 4096), 4096)
        grants = [event for event in self.lib.events[event_start:] if event[0] == "domain_grants"]
        self.assertIn(("domain_grants", newcomer_owner._resource._pointer.value,
                       7, 4096, 4096, 7168), grants)
        self.assertFalse(shared._domain.shared_growth_fenced)

    def test_post_grant_create_failure_rolls_back_and_blocks_direct_ensure(self):
        patcher = self.wrapper("post-grant-create-failure")
        owner, shared = patcher.model_patches_models()
        peer = self.wrapper("post-grant-peer")
        peer_owner, _ = peer.model_patches_models()
        peer_key = peer_owner._resource._pointer.value
        self.lib.resources[peer_key].update(held=256, limit=777,
                                            phase=qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED)
        key = owner._resource._pointer.value
        self.lib.resources[key]["limit"] = 111
        self.lib.resources[1]["limit"] = 17
        self.lib.device_limit = 31
        shared._domain.shared_growth_fenced = True

        with mock.patch.object(qfe.QFEngineHandle, "create",
                               side_effect=RuntimeError("factory failed after grant")):
            with self.assertRaisesRegex(RuntimeError, "factory failed after grant"):
                owner.partially_load(owner.load_device, 4096)

        self.assertEqual((self.lib.resources[key]["limit"],
                          self.lib.resources[1]["limit"], self.lib.device_limit), (111, 17, 31))
        self.assertEqual((self.lib.resources[peer_key]["held"],
                          self.lib.resources[peer_key]["limit"]), (256, 777))

        # Drop the peer's occupancy without publishing a fresh host grant. If
        # rollback had raised permission to the old residency, that permission
        # would now become realizable as new Shared growth.
        self.lib.resources[peer_key]["held"] = 0
        shared_growth_allowed = (self.lib.resources[1]["limit"] >= 256 and
                                 self.lib.device_limit >= 256)
        if shared_growth_allowed:
            self.lib.resources[1]["held"] += 256
        self.assertFalse(shared_growth_allowed)
        self.assertEqual(self.lib.resources[1]["held"], 0)
        self.assertTrue(owner._admission_failed)
        self.assertFalse(peer_owner._admission_failed)
        self.assertTrue(shared._domain.shared_growth_fenced)
        with self.assertRaisesRegex(qfe.NativeContractUnavailable, "failed host admission"):
            patcher.model._qf.ensure()

    def test_post_create_domain_query_failure_restores_prior_grants_and_fences_hot_handle(self):
        patcher = self.wrapper("post-create-query-failure")
        owner, shared = patcher.model_patches_models()
        self.lib.device_limit = 0
        self.lib.resources[1]["limit"] = 0
        key = owner._resource._pointer.value

        def create(lib, *, capacity_bytes, prepared_resource, create_params):
            self.lib.resources[key].update(held=capacity_bytes,
                                           phase=qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED)
            self.lib.domain_state = qfe.QUANTFUNC_RESOURCE_BUSY
            return qfe.QFEngineHandle(lib, ctypes.c_void_p(880 + key),
                                      resource=prepared_resource, capacity_bytes=capacity_bytes)

        with mock.patch.object(qfe.QFEngineHandle, "create", side_effect=create):
            with self.assertRaisesRegex(RuntimeError, "domain residency unavailable"):
                owner.partially_load(owner.load_device, 4096)

        self.assertEqual((self.lib.resources[key]["limit"],
                          self.lib.resources[1]["limit"], self.lib.device_limit), (0, 0, 0))
        self.assertTrue(owner._admission_failed)
        self.assertFalse(shared._domain.shared_growth_fenced)
        self.assertTrue(patcher.model._qf.materialized)
        with self.assertRaisesRegex(qfe.NativeContractUnavailable, "failed host admission"):
            patcher.model._qf.ensure()

    def test_busy_rollback_keeps_python_fence_until_next_formal_admission(self):
        last = None
        for rollback_state in (qfe.QUANTFUNC_RESOURCE_BUSY, qfe.QUANTFUNC_RESOURCE_UNKNOWN):
            with self.subTest(rollback_state=rollback_state):
                self.lib.domain_grant_state = qfe.QUANTFUNC_RESOURCE_READY
                patcher = self.wrapper(f"post-grant-rollback-{rollback_state}")
                owner, shared = patcher.model_patches_models()
                # Independent lazy alias of the SAME canonical Owner: its own
                # QFLazyEngine, never prepared or materialized before the failure.
                alias = self.wrapper(f"post-grant-rollback-{rollback_state}")
                alias_engine = alias.model._qf
                self.assertIsNot(alias_engine, patcher.model._qf)
                peer = self.wrapper(f"post-grant-rollback-peer-{rollback_state}")
                peer_owner, _ = peer.model_patches_models()
                peer_key = peer_owner._resource._pointer.value
                self.lib.resources[peer_key]["limit"] = 777
                self.lib.device_limit = 0
                self.lib.resources[1]["limit"] = 0
                key = owner._resource._pointer.value
                shared._domain.shared_growth_fenced = True

                def fail_and_block_rollback(*_args, **_kwargs):
                    self.lib.domain_grant_state = rollback_state
                    raise RuntimeError("factory failed before rollback")

                with mock.patch.object(qfe.QFEngineHandle, "create",
                                       side_effect=fail_and_block_rollback):
                    with self.assertRaisesRegex(RuntimeError, "factory failed before rollback") as raised:
                        owner.partially_load(owner.load_device, 4096)

                self.assertEqual((self.lib.resources[key]["limit"],
                                  self.lib.resources[1]["limit"], self.lib.device_limit),
                                 (4096, 4096, 4096))
                self.assertTrue(any("grant rollback was unavailable" in note
                                    for note in getattr(raised.exception, "__notes__", ())))
                self.assertTrue(owner._admission_failed)
                self.assertFalse(peer_owner._admission_failed)
                self.assertTrue(shared._domain.shared_growth_fenced)
                with self.assertRaisesRegex(qfe.NativeContractUnavailable, "failed host admission"):
                    patcher.model._qf.ensure()
                with self.assertRaisesRegex(qfe.NativeContractUnavailable, "Shared host grant"):
                    peer.model._qf.ensure()
                # The alias is refused by the canonical Owner's failed-admission fence
                # (not merely the domain's Shared fence that stops the peer above), so
                # the fence is keyed by Owner identity rather than by lazy engine.
                self.assertFalse(alias_engine.materialized)
                with self.assertRaisesRegex(qfe.NativeContractUnavailable, "failed host admission"):
                    alias_engine.ensure()
                self.assertFalse(alias_engine.materialized)
                self.assertIs(alias_engine._prepared_entry, patcher.model._qf._prepared_entry)
                last = patcher, owner, shared, key, alias_engine

        patcher, owner, shared, key, alias_engine = last
        self.lib.domain_grant_state = qfe.QUANTFUNC_RESOURCE_READY

        def create(lib, *, capacity_bytes, prepared_resource, create_params):
            self.lib.resources[key].update(held=capacity_bytes,
                                           phase=qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED)
            return qfe.QFEngineHandle(lib, ctypes.c_void_p(900 + key),
                                      resource=prepared_resource, capacity_bytes=capacity_bytes)

        with mock.patch.object(qfe.QFEngineHandle, "create", side_effect=create):
            self.assertEqual(owner.partially_load(owner.load_device, 4096), 4096)
        self.assertFalse(owner._admission_failed)
        self.assertFalse(shared._domain.shared_growth_fenced)
        self.assertIsNotNone(patcher.model._qf.ensure())
        # Only that formal admission reopens the alias, and onto the same live handle.
        self.assertIs(alias_engine.ensure(), patcher.model._qf.ensure())

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
        with self.assertRaisesRegex(RuntimeError, "domain residency unavailable"):
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
        for state in (qfe.QUANTFUNC_RESOURCE_BUSY, qfe.QUANTFUNC_RESOURCE_UNKNOWN,
                      qfe.QUANTFUNC_RESOURCE_CLOSED):
            with self.subTest(state=state):
                self.lib.domain_state = state
                with self.assertRaises(RuntimeError):
                    patcher.model_patches_models()
                self.assertEqual(mm.current_loaded_models, [sentinel])

        self.lib.domain_state = qfe.QUANTFUNC_RESOURCE_READY
        owner = patcher.model_patches_models()[0]
        self.lib.resources[owner._resource._pointer.value]["phase"] = \
            qfe.QUANTFUNC_RESOURCE_PHASE_CLOSED
        with self.assertRaisesRegex(RuntimeError, "Closed"):
            patcher.model_patches_models()
        self.assertEqual(mm.current_loaded_models, [sentinel])

    def test_direct_cold_create_cannot_bypass_zero_host_grant(self):
        p = self.wrapper()
        p.model_patches_models()  # binds the exact predicted capacity
        with mock.patch.object(qfe.QFEngineHandle, "create") as create:
            with self.assertRaisesRegex(qfe.NativeContractUnavailable, "positive Owned host grant"):
                p.model._qf.ensure()
        create.assert_not_called()
        self.assertEqual(self.lib.resources[2]["held"], 0)

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
        revoke = ("domain_grants", 2, 7, 0, 0, 0)
        release = ("release", 2, 64)
        self.assertLess(self.lib.events.index(revoke), self.lib.events.index(release))
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
        for state in (1, 2):
            self.lib.resources[2]["state"] = state
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                owner.loaded_size()
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                owner.partially_unload(torch.device("cpu"), 64)
            self.assertEqual((self.lib.resources[2]["limit"],
                              self.lib.resources[1]["limit"], self.lib.device_limit), (0, 0, 0))
            self.assertTrue(owner._domain.shared_growth_fenced)
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                p.model_patches_models()

    def test_full_detach_fences_growth_and_an_incomplete_native_release_stays_inside_the_hook(self):
        p = self.wrapper()
        owner, shared = p.model_patches_models()
        self.lib.capabilities = 7
        self.lib.resources[2]["held"] = 512
        self.lib.resources[1]["held"] = 2048
        owner.detach(False)
        self.assertFalse(any(e[0] in ("grant", "full") for e in self.lib.events))
        self.lib.full_state = 1
        owner.detach(True)  # native Busy is the ordinary answer for a live pipeline; it must not raise
        self.assertEqual(owner.loaded_size(), 512)  # and it is never reported as released
        self.assertTrue(owner._resource._finalizer.alive)
        self.lib.full_state = 0
        owner.detach(True)
        self.assertEqual([owner.loaded_size(), shared.loaded_size()], [0, 2048])
        self.assertEqual([e for e in self.lib.events if e[0] == "full"],
                         [("full", 2), ("full", 2)])
        self.assertEqual([e for e in self.lib.events if e[0] == "domain_grants"],
                         [("domain_grants", 2, 7, 0, 0, 0)] * 4)  # before + after release_all, per detach
        self.assertEqual(self.lib.resources[1]["limit"], 0)
        self.assertEqual(self.lib.device_limit, 0)
        self.assertTrue(shared._domain.shared_growth_fenced)
        self.assertEqual(owner.model_size(), 4096)

    def test_closed_full_release_does_not_revive_identity_or_change_grants(self):
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
        self.assertFalse(any(e[0] in ("grant", "device_grant", "domain_grants")
                             for e in self.lib.events))
        with self.assertRaisesRegex(RuntimeError, "Closed resource identity"):
            owner.model_size()

    def test_zero_grant_blocks_warm_ensure_after_failed_and_full_release(self):
        p = self.wrapper()
        owner, _ = p.model_patches_models()
        owner._prepared = False
        self.lib.resources[2]["phase"] = qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED
        entry = p.model._qf._prepared_entry
        warm = types.SimpleNamespace(resource=entry.resource, pipeline=object())
        lazy = p.model._qf
        lazy._real = warm
        lazy._created_lora_sig = lazy._lora_sig()
        self.lib.capabilities = 7
        self.lib.resources[2]["held"] = 512
        for state in (1, 2):
            self.lib.full_state = state
            owner.detach(True)
            self.assertEqual(owner.loaded_size(), 512)
            self.assertEqual(owner.model_size(), 4096)
            self.assertEqual(self.lib.resources[2]["limit"], 0)
            with self.subTest(full_release_state=state, direct_path="warm ensure"):
                with self.assertRaises(qfe.NativeContractUnavailable):
                    lazy.ensure()
            self.assertTrue(owner._resource._finalizer.alive)
        self.lib.full_state = 0
        owner.detach(True)  # Retry remains valid; release permission is not load permission.
        self.assertEqual(owner.loaded_size(), 0)
        self.assertEqual(owner.model_size(), 4096)
        self.assertEqual(self.lib.resources[2]["limit"], 0)
        with self.assertRaises(qfe.NativeContractUnavailable):
            lazy.ensure()


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
