#!/usr/bin/env python3
"""Real Comfy scheduler/executor contracts, with only native CUDA calls doubled.

Run with COMFY_ROOT and its Python environment. This process forces CPU mode;
the CUDA device object is an identity, never an allocation target.
"""
import ctypes
import gc
import importlib
import contextlib
import io
import os
from pathlib import Path
import sys
import types
import unittest
import weakref
from unittest import mock

PROBE_CLONE_HANDOFF = "--probe-clone-handoff" in sys.argv
root = os.environ.get("COMFY_ROOT")
if not root or not (Path(root) / "comfy/model_management.py").is_file():
    print("[SKIP] host_scheduler_contract: set COMFY_ROOT to the tested host")
    raise SystemExit(77)
sys.path.insert(0, root)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import execution
import nodes
import comfy.model_management as mm

pkg = types.ModuleType("qf_host_contract")
pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[pkg.__name__] = pkg
qfe = importlib.import_module("qf_host_contract.qf_engine")
qfm = importlib.import_module("qf_host_contract.qf_modelpatcher")


class NativeLibrary:
    def __init__(self):
        self.held = 512
        self.query_status = 0
        self.query_calls = 0
        self.unload_calls = 0
        self.unload_status = 1

    def quantfunc_resident_vram_bytes(self, pipeline, out):
        self.query_calls += 1
        out._obj.value = self.held
        return self.query_status

    def quantfunc_unload_sync(self, pipeline):
        raise AssertionError("legacy status-only unload cannot report actual bytes")

    def quantfunc_unload_sync_ex(self, pipeline, out):
        self.unload_calls += 1
        if self.unload_status == 0:
            out._obj.value = self.held
            self.held = 0
        return self.unload_status

    def quantfunc_partial_unload(self, pipeline, requested, out):
        out._obj.value = min(requested.value, 64)
        self.held -= out._obj.value
        return 0

    def quantfunc_last_error(self):
        return b"contract: native refused"


class HostSchedulerContract(unittest.TestCase):
    def setUp(self):
        self.library = NativeLibrary()
        self.engine = qfe.QFEngineHandle(self.library, ctypes.c_void_p(1), footprint_bytes=1024)
        self.model = torch.nn.Module()
        self.model.device = torch.device("cuda:0")
        self.model._qf = self.engine
        self.patcher = qfm.QFModelPatcher(self.model, self.model.device, torch.device("cpu"), size=1024)

    def tearDown(self):
        # No native calls from fixture teardown.
        self.model._qf = None
        self.patcher = None
        gc.collect()

    def test_zero_residency_is_not_replaced_with_file_size(self):
        self.library.held = 0
        self.assertEqual(self.patcher.loaded_size(), 0)

    def test_query_failure_is_not_replaced_with_file_size(self):
        self.library.query_status = 1
        with self.assertRaisesRegex(RuntimeError, "native refused"):
            self.patcher.loaded_size()

    def test_native_measurement_wins_over_unloaded_flag(self):
        self.engine.unloaded = True
        self.assertEqual(self.patcher.loaded_size(), 512)

    def test_cold_lazy_handle_is_zero_without_creating_engine(self):
        creates = []
        def factory():
            creates.append(True)
            return self.engine, "key"
        self.model._qf = qfm.QFLazyEngine(factory, 1024)
        self.patcher.partially_load(self.model.device, 1)
        self.assertEqual(self.patcher.loaded_size(), 0)
        self.assertEqual(creates, [])

    def test_failed_unload_preserves_registration_and_next_prompt_runs(self):
        self._assert_failed_reclaim_preserves_registration(expected_unloads=1)

    def test_query_failure_in_eviction_sizing_preserves_registration_and_next_prompt_runs(self):
        self.library.query_status = 1
        self._assert_failed_reclaim_preserves_registration(expected_unloads=0)
        self.assertGreater(self.library.query_calls, 0)

    def _assert_failed_reclaim_preserves_registration(self, expected_unloads):
        loaded = mm.LoadedModel(self.patcher)
        loaded.real_model = weakref.ref(self.model)
        loaded.model_finalizer = weakref.finalize(self.model, lambda: None)
        previous = mm.current_loaded_models[:]
        mm.current_loaded_models[:] = [loaded]
        device = self.model.device

        class ProbeNode:
            RETURN_TYPES = ()
            FUNCTION = "run"
            OUTPUT_NODE = True

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"reclaim": ("BOOLEAN",)}}

            def run(self, reclaim):
                if reclaim:
                    mm.free_memory(1024, device)
                return ()

        events = []
        server = types.SimpleNamespace(
            client_id=None, last_node_id=None,
            send_sync=lambda event, data, sid: events.append((event, data)),
        )
        executor = execution.PromptExecutor(server, cache_type=execution.CacheType.NONE,
                                            cache_args={"ram": 0, "ram_inactive": 0})
        try:
            with mock.patch.dict(nodes.NODE_CLASS_MAPPINGS, {"QFHostContract": ProbeNode}), mock.patch.object(
                mm, "get_free_memory", return_value=0
            ):
                with self.assertLogs(level="ERROR"):
                    executor.execute({"1": {"class_type": "QFHostContract", "inputs": {"reclaim": True}}},
                                     "failed-unload", {"client_id": "contract"}, ["1"])
                self.assertFalse(executor.success)
                errors = [data for event, data in events if event == "execution_error"]
                self.assertEqual(len(errors), 1)
                self.assertIn("native refused", errors[0]["exception_message"])
                self.assertEqual(self.library.unload_calls, expected_unloads)
                self.assertFalse(self.engine.unloaded)
                self.assertEqual(mm.current_loaded_models, [loaded])
                self.assertIs(loaded.real_model(), self.model)
                self.assertTrue(loaded.model_finalizer.alive)
                executor.execute({"2": {"class_type": "QFHostContract", "inputs": {"reclaim": False}}},
                                 "next-prompt", {"client_id": "contract"}, ["2"])
                self.assertTrue(executor.success)
                self.assertTrue(any(event == "execution_success" and data["prompt_id"] == "next-prompt"
                                    for event, data in events))
        finally:
            mm.current_loaded_models[:] = previous
            if loaded.model_finalizer is not None:
                loaded.model_finalizer.detach()

    def test_partial_shortfall_is_returned_to_host_without_full_eviction(self):
        self.assertEqual(self.patcher.partially_unload(torch.device("cpu"), 128), 64)
        self.assertEqual(self.patcher.loaded_size(), 448)
        self.assertEqual(self.library.unload_calls, 0)

    def test_zero_partial_request_is_inert(self):
        self.assertEqual(self.patcher.partially_unload(torch.device("cpu"), 0), 0)
        self.assertEqual(self.library.held, 512)
        self.assertEqual(self.library.unload_calls, 0)

    def test_host_full_eviction_completes_before_record_is_removed(self):
        self.library.unload_status = 0
        self.model.contract_value = "patched"
        self.patcher.object_patches_backup["contract_value"] = "original"
        detached = []
        self.patcher.add_callback(qfm.comfy.model_patcher.CallbacksMP.ON_DETACH,
                                 lambda patcher, full: detached.append((full, self.library.held)))
        loaded = mm.LoadedModel(self.patcher)
        loaded.real_model = weakref.ref(self.model)
        loaded.model_finalizer = weakref.finalize(self.model, lambda: None)
        finalizer = loaded.model_finalizer
        try:
            with mock.patch("threading.Timer", side_effect=AssertionError("host eviction must not start a timer")):
                self.assertTrue(loaded.model_unload())
            self.assertEqual(self.library.held, 0)
            self.assertEqual(self.library.unload_calls, 1)
            self.assertEqual(detached, [(True, 0)])
            self.assertEqual(self.model.contract_value, "original")
            self.assertIsNone(loaded.real_model)
            self.assertFalse(finalizer.alive)
        finally:
            finalizer.detach()

    def test_clone_detach_preserves_resource_and_official_callback_without_timer(self):
        detached = []
        self.patcher.add_callback(qfm.comfy.model_patcher.CallbacksMP.ON_DETACH,
                                 lambda patcher, full: detached.append(full))
        with mock.patch("threading.Timer", side_effect=AssertionError("clone switch must not start a timer")):
            self.assertIs(self.patcher.detach(False), self.model)
        self.assertEqual(self.library.held, 512)
        self.assertEqual(self.library.unload_calls, 0)
        self.assertEqual(detached, [False])

    def test_residual_residency_prevents_successful_full_detach(self):
        def incomplete_release(pipeline, out):
            out._obj.value = 128
            self.library.held = 384
            return 0
        self.library.quantfunc_unload_sync_ex = incomplete_release
        loaded = mm.LoadedModel(self.patcher)
        loaded.real_model = weakref.ref(self.model)
        loaded.model_finalizer = weakref.finalize(self.model, lambda: None)
        try:
            with self.assertRaisesRegex(RuntimeError, "384.*resident"):
                loaded.model_unload()
            self.assertIs(loaded.real_model(), self.model)
            self.assertTrue(loaded.model_finalizer.alive)
            self.assertEqual(self.patcher.loaded_size(), 384)
        finally:
            if loaded.model_finalizer is not None:
                loaded.model_finalizer.detach()

    def test_ledger_log_does_not_label_failed_residency_query_as_zero(self):
        class Model(qfm.QFSessionModelMixin):
            pass
        model = Model()
        model._qf = self.engine
        self.library.query_status = 1
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            model.memory_required([1, 16, 1, 8, 8])
        self.assertIn("engine hold unknown", output.getvalue())
        self.assertNotIn("engine hold 0 MB", output.getvalue())


class NativeResourceSchedulerContract(unittest.TestCase):
    def make_resource(self, held=1536, eligible=1536, device=0):
        # Only the external C calls are doubled. Use the actual retained Python
        # view, adapter, ModelPatcher, LoadedModel and host scheduling loop.
        lib = types.SimpleNamespace(held=held, eligible=eligible, state=0,
                                    status=0, requests=[], closed=[], device=device)
        def acquire(pipeline, version, out):
            out._obj.value = 17
            return 0
        def residency(pointer, out):
            out._obj.state = lib.state
            out._obj.resident_bytes = lib.held
            return lib.status
        def release(pointer, requested, out):
            lib.requests.append(requested)
            out._obj.state = lib.state
            out._obj.freed_bytes = min(requested, lib.eligible)
            if lib.state == 0 and lib.status == 0:
                lib.held -= out._obj.freed_bytes
                lib.eligible -= out._obj.freed_bytes
            return lib.status
        def query(pointer, out):
            out._obj.state, out._obj.device = lib.state, lib.device
            out._obj.owner_epoch, out._obj.capabilities = 123, 3
            # Deliberately unlike the aggregate: these fields may identify
            # the resource's device but must never become Python byte policy.
            out._obj.cca_live, out._obj.arena_backed = 7000, 9000
            return lib.status
        lib.quantfunc_resource_acquire = acquire
        lib.quantfunc_resource_acquire_shared = acquire
        lib.quantfunc_resource_query_residency = residency
        lib.quantfunc_resource_release_eligible = release
        lib.quantfunc_resource_query = query
        lib.quantfunc_resource_destroy = lambda pointer: lib.closed.append(pointer.value)
        lib.quantfunc_last_error = lambda: b"resource contract refused"
        resource = qfe.NativeResource.acquire(lib, ctypes.c_void_p(1))
        self.addCleanup(resource.close)
        return lib, qfm.QFNativeResourcePatcher(resource)

    @contextlib.contextmanager
    def host_registry(self):
        previous = mm.current_loaded_models[:]
        mm.current_loaded_models[:] = []
        try:
            yield
        finally:
            for loaded in mm.current_loaded_models:
                if loaded.model_finalizer is not None:
                    loaded.model_finalizer.detach()
            mm.current_loaded_models[:] = previous

    def load(self, *patchers):
        with mock.patch.object(mm, "get_free_memory", side_effect=self.free_memory(1 << 40)):
            mm.load_models_gpu(list(patchers))

    @staticmethod
    def free_memory(amount):
        return lambda device, torch_free_too=False: (amount, amount) if torch_free_too else amount

    def test_native_resource_residency_reaches_real_loaded_model(self):
        # Breaking the adapter's native forwarding must change the host's
        # observable loaded bytes; no file-size or Python category sum allowed.
        self.assertTrue(hasattr(qfm, "QFNativeResourcePatcher"),
                        "native resource has no ComfyUI scheduling adapter")
        lib, patcher = self.make_resource()
        lib.quantfunc_resource_query = lambda *_: self.fail("loaded bytes require only native aggregate")
        loaded = mm.LoadedModel(patcher)
        self.assertEqual(loaded.model_loaded_memory(), 1536)

    def test_registration_device_comes_from_native_identity(self):
        lib, patcher = self.make_resource(device=2)
        self.assertEqual(patcher.load_device, torch.device("cuda:2"))
        self.assertEqual(mm.LoadedModel(patcher).device, torch.device("cuda:2"))
        self.assertEqual(patcher.current_loaded_device(), torch.device("cuda:2"))

    def test_release_count_is_not_reconstructed_from_residency_delta(self):
        lib, patcher = self.make_resource()
        def release(pointer, requested, out):
            out._obj.state, out._obj.freed_bytes = 0, 64
            lib.held = 1024  # other work changed occupancy across the call
            return 0
        lib.quantfunc_resource_release_eligible = release
        self.assertEqual(patcher.partially_unload(torch.device("cpu"), 128), 64)
        self.assertEqual(patcher.loaded_size(), 1024)

    def test_host_partial_reclaim_reports_native_confirmed_count(self):
        lib, patcher = self.make_resource(held=1536, eligible=64)
        self.assertEqual(patcher.partially_unload(torch.device("cpu"), 128), 64)
        self.assertEqual(patcher.loaded_size(), 1472)
        self.assertEqual(lib.requests, [128])

    def test_host_sentinel_saturates_and_zero_request_is_inert(self):
        lib, patcher = self.make_resource()
        self.assertEqual(patcher.partially_unload(torch.device("cpu"), 0), 0)
        self.assertEqual(lib.requests, [])
        self.assertEqual(patcher.partially_unload(torch.device("cpu"), 1e32), 1536)
        self.assertEqual(lib.requests, [(1 << 64) - 1])

    def test_canonical_dependency_clone_is_deduplicated_by_actual_host(self):
        lib, patcher = self.make_resource()
        peer_lib, peer = self.make_resource(held=2048, eligible=2048)
        with self.host_registry():
            self.load(patcher, patcher.clone(), peer, patcher)
            self.assertEqual(len(mm.current_loaded_models), 2)
            self.assertEqual(sorted(item.model_loaded_memory() for item in mm.current_loaded_models),
                             [1536, 2048])
            self.assertIs(patcher.clone(), patcher)
            original = next(item for item in mm.current_loaded_models if item.model is patcher)
            finalizer = original.model_finalizer
            self.load(patcher.clone(), peer)
            self.assertEqual(len(mm.current_loaded_models), 2)
            # This host replaces the finalizer even for a repeat load of the
            # same patcher. Its clone handoff removes the old list entry before
            # unconditional insertion; opting out would duplicate registration.
            self.assertFalse(finalizer.alive)
            self.assertTrue(original.model_finalizer.alive)
        self.assertEqual(lib.requests + peer_lib.requests, [])

    def test_even_zero_backing_cannot_authorize_full_host_detach(self):
        # A sampled zero is not a native lifetime/admission fence. Until that
        # handshake exists this adapter must keep its record, including at zero.
        for held in (0, 1536):
            with self.subTest(held=held):
                lib, patcher = self.make_resource(held=held, eligible=held)
                with self.host_registry():
                    self.load(patcher)
                    original = mm.current_loaded_models[0]
                    with mock.patch.object(mm, "get_free_memory", side_effect=self.free_memory(0)):
                        with self.assertRaisesRegex(RuntimeError, "full eviction.*unsupported"):
                            mm.free_memory(4096, patcher.load_device)
                    self.assertEqual(mm.current_loaded_models, [original])
                    self.assertTrue(original.model_finalizer.alive)
                    self.assertIs(original.real_model(), patcher.model)
                self.assertEqual(lib.requests, [])

    def test_partial_full_shortfall_cannot_remove_host_registration(self):
        lib, patcher = self.make_resource(held=1536, eligible=64)
        with self.host_registry():
            self.load(patcher)
            original = mm.current_loaded_models[0]
            with mock.patch.object(mm, "get_free_memory", side_effect=self.free_memory(0)):
                with self.assertRaisesRegex(RuntimeError, "full eviction.*unsupported"):
                    mm.free_memory(128, patcher.load_device)
            self.assertEqual(mm.current_loaded_models, [original])
            self.assertTrue(original.model_finalizer.alive)
            self.assertIs(original.real_model(), patcher.model)
        self.assertEqual(lib.requests, [128])
        self.assertEqual(patcher.loaded_size(), 1472)

    def test_nonready_and_native_error_never_become_zero_or_drop_record(self):
        for state, status in ((1, 0), (2, 0), (3, 0), (0, 1)):
            with self.subTest(state=state, status=status):
                lib, patcher = self.make_resource()
                with self.host_registry():
                    self.load(patcher)
                    original = mm.current_loaded_models[0]
                    lib.state, lib.status = state, status
                    with mock.patch.object(mm, "get_free_memory", side_effect=self.free_memory(0)):
                        with self.assertRaises(RuntimeError):
                            mm.free_memory(4096, patcher.load_device)
                    self.assertEqual(mm.current_loaded_models, [original])
                    with self.assertRaises(RuntimeError):
                        patcher.partially_unload(torch.device("cpu"), 128)

    def test_clone_handoff_does_not_request_physical_release(self):
        lib, patcher = self.make_resource()
        patcher.detach(False)
        self.assertEqual(lib.requests, [])
        self.assertEqual(patcher.loaded_size(), 1536)

    def test_resource_cannot_be_moved_or_copied_as_model_weights(self):
        lib, patcher = self.make_resource()
        with self.assertRaisesRegex(ValueError, "device"):
            patcher.partially_load(torch.device("cuda:1"), 4096)
        with self.assertRaisesRegex(ValueError, "resource"):
            patcher.clone(model_override=(torch.nn.Module(), ({}, {}, {}, set())))
        with self.assertRaisesRegex(ValueError, "resource"):
            patcher.clone(force_deepcopy=True)
        with self.assertRaisesRegex(ValueError, "resource"):
            patcher.add_patches({"weight": (torch.ones(1),)})
        with self.assertRaisesRegex(ValueError, "resource"):
            patcher.add_object_patch("device", torch.device("cuda:1"))
        self.assertEqual(lib.requests, [])

    def probe_clone_handoff_preserves_record_after_native_becomes_busy(self):
        """Separate acceptance probe: expected host invariant, currently broken.

        Run explicitly with --probe-clone-handoff. Not part of the adapter's
        passing scoped contract: correcting the host transaction needs separate
        ComfyUI-core authority. The callback models native becoming Busy in the
        interval between successful detach(False) and the next size query.
        """
        lib, patcher = self.make_resource()
        with self.host_registry():
            self.load(patcher)
            original = mm.current_loaded_models[0]
            patcher.add_callback(qfm.comfy.model_patcher.CallbacksMP.ON_DETACH,
                                 lambda p, full: setattr(lib, "state", 1))
            with self.assertRaisesRegex(RuntimeError, "residency unavailable"):
                self.load(patcher)
            self.assertEqual(mm.current_loaded_models, [original],
                             "host lost the native resource after clone handoff query failure")


if __name__ == "__main__":
    if PROBE_CLONE_HANDOFF:
        result = unittest.TextTestRunner().run(unittest.TestSuite([
            NativeResourceSchedulerContract(
                "probe_clone_handoff_preserves_record_after_native_becomes_busy")]))
        raise SystemExit(not result.wasSuccessful())
    unittest.main(argv=[sys.argv[0]])
