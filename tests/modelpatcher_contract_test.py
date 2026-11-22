#!/usr/bin/env python3
"""Direct QFModelPatcher VRAM coexistence contract (PC Engine Rev 4.2 §3.2.B) on ComfyUI's real ModelPatcher:
  - is_dynamic() is False;
  - loaded_size() reports qf_xfm_resident_vram_bytes;
  - partially_unload(n) delegates to qf_xfm_unload_vram(n) and returns freed bytes;
  - detach(False) preserves VRAM (clone handoff); detach(True) calls qf_xfm_unload_vram(QF_UINT64_MAX);
  - partially_load materializes lazy handle only and no retired ABI-2 quantfunc_resource_* symbol exists.

Run with COMFY_ROOT and its Python environment (CPU only)."""
import ctypes
import importlib
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

root = os.environ.get("COMFY_ROOT")
if not root or not (Path(root) / "comfy/model_management.py").is_file():
    print("[SKIP] modelpatcher_contract: set COMFY_ROOT to the tested host")
    raise SystemExit(77)
sys.path.insert(0, root)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import comfy.model_management as mm
import torch

pkg = types.ModuleType("qf_modelpatcher_contract")
pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[pkg.__name__] = pkg
qfe = importlib.import_module("qf_modelpatcher_contract.qf_engine")
qfm = importlib.import_module("qf_modelpatcher_contract.qf_modelpatcher")

XFM_PTR = 17
GIB = 1 << 30
MIB = 1 << 20


def pc_engine_library():
    """Minimal 25-symbol PC Engine stub covering VRAM residency & unload verbs."""
    lib = types.SimpleNamespace(held=1536, need=4096, unload_status=qfe.QF_OK, calls=[])

    def resident(xfm, out_ptr):
        out_ptr._obj.value = lib.held
        return qfe.QF_OK

    def need(xfm, dims, ndim, out_ptr):
        out_ptr._obj.value = lib.need
        return qfe.QF_OK

    def unload_vram(xfm, target_bytes, out_freed_ptr):
        lib.calls.append(("qf_xfm_unload_vram", xfm.value, target_bytes))
        if lib.unload_status != qfe.QF_OK:
            return lib.unload_status
        if target_bytes == qfe.QF_UINT64_MAX:
            freed = lib.held
        else:
            freed = min(target_bytes, 1000, lib.held)
        lib.held -= freed
        out_freed_ptr._obj.value = freed
        return qfe.QF_OK

    def destroy(xfm):
        lib.calls.append(("qf_xfm_destroy", xfm.value))
        lib.held = 0

    lib.qf_xfm_resident_vram_bytes = resident
    lib.qf_xfm_vram_need_bytes = need
    lib.qf_xfm_unload_vram = unload_vram
    lib.qf_xfm_destroy = destroy
    lib.qf_last_error = lambda: b"session busy"
    return lib


class _ColdTorchModel(qfm.QFSessionModelMixin, torch.nn.Module):
    def __init__(self, lazy):
        torch.nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self._qf = lazy

    def memory_required(self, input_shape, cond_shapes=None, **_kw):
        return qfm.QFSessionModelMixin.memory_required(self, input_shape, cond_shapes=cond_shapes, **_kw)


class ModelPatcherContract(unittest.TestCase):
    def setUp(self):
        self.lib = pc_engine_library()
        self.handle = qfe.QFEngineHandle(self.lib, ctypes.c_void_p(XFM_PTR), capacity_bytes=4096)
        self.model = torch.nn.Module()
        self.model.device = torch.device("cpu")
        self.model._qf = self.handle
        self.patcher = qfm.QFModelPatcher(self.model, torch.device("cpu"), torch.device("cpu"))

    def test_is_dynamic_is_false_and_no_abi2_patcher_exists(self):
        self.assertIs(self.patcher.is_dynamic(), False)
        self.assertFalse(hasattr(qfm, "QFNativeResourcePatcher"))
        self.assertFalse(hasattr(qfe, "NativeResource"))

    def test_loaded_size_and_partially_unload_delegate_to_qf_xfm_vram_verbs(self):
        self.assertEqual(self.patcher.loaded_size(), 1536)
        self.assertEqual(self.patcher.partially_unload(self.patcher.offload_device, 4096), 1000)
        self.assertEqual(self.lib.calls, [("qf_xfm_unload_vram", XFM_PTR, 4096)])
        self.assertEqual(self.patcher.loaded_size(), 536)

    def test_detach_preserves_vram_on_false_and_unloads_all_on_true(self):
        self.patcher.detach(False)
        self.assertEqual(self.lib.calls, [])
        self.assertEqual(self.patcher.loaded_size(), 1536)
        self.patcher.detach(True)
        self.assertEqual(self.lib.calls, [("qf_xfm_unload_vram", XFM_PTR, qfe.QF_UINT64_MAX)])
        self.assertEqual(self.patcher.loaded_size(), 0)

    def test_partially_load_materializes_lazy_engine_without_abi2_verbs(self):
        materialized = []
        lazy = qfm.QFLazyEngine(lambda: (materialized.append(True) or self.handle, ("k",)))
        m = torch.nn.Module()
        m.device = torch.device("cpu")
        m._qf = lazy
        p = qfm.QFModelPatcher(m, torch.device("cpu"), torch.device("cpu"))
        self.assertEqual(p.loaded_size(), 0)
        p.partially_load(p.load_device, 4096)
        self.assertEqual(materialized, [True])
        self.assertEqual(p.loaded_size(), 1536)
        self.assertEqual(self.lib.calls, [])

    def test_partially_load_full_load_clamps_no_2x_double_count(self):
        self.patcher.partially_load(self.patcher.load_device, 10 ** 9)
        self.assertLessEqual(getattr(self.model, "model_loaded_weight_memory", 0), self.patcher.size)
        self.assertEqual(self.patcher.loaded_size(), 1536)

    def test_partially_load_device_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError, "cannot migrate"):
            self.patcher.partially_load(torch.device("meta"), 4096)

    def test_partially_load_negative_extra_memory_delegates_unload(self):
        got = self.patcher.partially_load(self.patcher.load_device, -512)
        self.assertEqual(got, 0)
        self.assertEqual(self.lib.calls, [("qf_xfm_unload_vram", XFM_PTR, 512)])
        self.assertEqual(self.patcher.loaded_size(), 1536 - 512)

    def test_partially_unload_busy_logs_warning_and_returns_zero(self):
        self.lib.unload_status = qfe.QF_ERR_BUSY
        with self.assertLogs(qfm._log, level="WARNING") as logs:
            freed = self.patcher.partially_unload(self.patcher.offload_device, 4096)
        self.assertEqual(freed, 0)
        self.assertEqual(self.patcher.loaded_size(), 1536)
        self.assertTrue(any("busy" in line.lower() for line in logs.output))

    def test_cold_vram_with_real_loaded_model_is_1x_not_2x(self):
        resident_w = 14 * GIB
        min_offload = 4 * GIB
        work_floor = 1536 * MIB
        created = []
        lazy = qfm.QFLazyEngine(
            lambda: (created.append(1) or self.handle, ("k",)),
            estimate_cold=lambda: qfe.ColdNeed(resident_w, None, min_offload, work_floor),
        )
        m = _ColdTorchModel(lazy)
        p = qfm.QFModelPatcher(m, torch.device("cpu"), torch.device("cpu"))
        loaded = mm.LoadedModel(p)
        shape = [1, 16, 64, 64]
        comfy_side = m._qf_comfy_side_bytes(shape, None)

        self.assertEqual(p.model_size(), resident_w)
        self.assertEqual(p.loaded_size(), 0)
        self.assertEqual(loaded.model_memory_required(torch.device("cpu")), resident_w)
        mem_req = m.memory_required(shape)
        self.assertEqual(mem_req, comfy_side + work_floor)
        # ComfyUI load_models_gpu requests 1.1 * model_memory_required + memory_required: must be ~1.2x, never ~2.1x.
        free_target = int(loaded.model_memory_required(torch.device("cpu")) * 1.1) + mem_req
        self.assertGreater(free_target, resident_w)
        self.assertLess(free_target, int(resident_w * 1.4))
        self.assertEqual(created, [])

    def test_cold_estimate_passes_ui_api_key_and_preserves_all_three_fields(self):
        plugin_pkg = "qf_modelpatcher_contract_plugin"
        spec = importlib.util.spec_from_file_location(
            plugin_pkg,
            Path(__file__).resolve().parents[1] / "__init__.py",
            submodule_search_locations=[str(Path(__file__).resolve().parents[1])],
        )
        plugin = importlib.util.module_from_spec(spec)
        sys.modules[plugin_pkg] = plugin
        spec.loader.exec_module(plugin)

        captured = []

        def fake_estimate(params_ptr, est_ptr):
            p = params_ptr._obj
            captured.append(json.loads(p.config_json.decode("utf-8")))
            est_ptr._obj.resident_weight_bytes = 12 * GIB
            est_ptr._obj.min_offload_bytes = 3 * GIB
            est_ptr._obj.working_set_floor_bytes = 768 * MIB
            return plugin.qfe.QF_OK

        fake_lib = types.SimpleNamespace(qf_xfm_estimate_vram=fake_estimate, qf_last_error=lambda: b"")
        with mock.patch.object(plugin.qfe, "_LIB", fake_lib), \
             mock.patch.object(plugin, "_read_auth", side_effect=lambda k=None: (k or "", "https://auth.example")):
            cold = plugin._estimate_engine("/m.safetensors", {"model_arch": "krea2"}, device_id=0,
                                           api_key="qf_ui_secret_key_12345")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["api_key"], "qf_ui_secret_key_12345")
        self.assertEqual(captured[0]["server_url"], "https://auth.example")
        self.assertEqual(cold.bytes, 12 * GIB)
        self.assertEqual(cold.resident_weight_bytes, 12 * GIB)
        self.assertEqual(cold.min_offload_bytes, 3 * GIB)
        self.assertEqual(cold.working_set_floor_bytes, 768 * MIB)
        self.assertIsNone(cold.note)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
