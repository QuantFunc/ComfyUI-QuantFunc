#!/usr/bin/env python3
"""Real Comfy scheduler/executor contracts, with only native CUDA calls doubled.

Run with COMFY_ROOT and its Python environment. This process forces CPU mode;
the CUDA device object is an identity, never an allocation target.
"""
import ctypes
import gc
import importlib
import os
from pathlib import Path
import sys
import types
import unittest
import weakref
from unittest import mock

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
        self.unload_calls = 0
        self.unload_status = 1

    def quantfunc_resident_vram_bytes(self, pipeline, out):
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
                self.assertEqual(self.library.unload_calls, 1)
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


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
