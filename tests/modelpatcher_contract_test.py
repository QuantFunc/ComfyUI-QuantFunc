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
    def __init__(self, lazy, with_weight=False):
        torch.nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self._qf = lazy
        if with_weight:
            self.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))

    def memory_required(self, input_shape, cond_shapes=None, **_kw):
        return qfm.QFSessionModelMixin.memory_required(self, input_shape, cond_shapes=cond_shapes, **_kw)


def _load_plugin_module():
    plugin_pkg = "qf_modelpatcher_contract_plugin"
    if plugin_pkg in sys.modules:
        return sys.modules[plugin_pkg]
    spec = importlib.util.spec_from_file_location(
        plugin_pkg,
        Path(__file__).resolve().parents[1] / "__init__.py",
        submodule_search_locations=[str(Path(__file__).resolve().parents[1])],
    )
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[plugin_pkg] = plugin
    spec.loader.exec_module(plugin)
    return plugin


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
        plugin = _load_plugin_module()
        captured = []

        def fake_estimate(params_ptr, est_ptr):
            p = params_ptr._obj
            captured.append(json.loads(p.config_json.decode("utf-8")))
            est_ptr._obj.resident_weight_bytes = 12 * GIB
            est_ptr._obj.min_offload_bytes = 3 * GIB
            est_ptr._obj.working_set_floor_bytes = 768 * MIB
            est_ptr._obj.recommended_offload_mode = 1
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
        self.assertEqual(cold.recommended_offload_mode, 1)
        self.assertIsNone(cold.note)

    def test_t6_clone_chain_adopt_and_partially_load_never_compound_native_vram(self):
        cpu = torch.device("cpu")
        h14 = qfe.QFEngineHandle(self.lib, ctypes.c_void_p(XFM_PTR), capacity_bytes=14 * GIB)
        self.lib.held = 14 * GIB
        m = _ColdTorchModel(h14, with_weight=True)
        p0 = qfm.QFModelPatcher(m, cpu, cpu)
        p1 = p0.clone()
        p2 = p1.clone()
        p3 = p2.clone()
        for px in (p0, p1, p2, p3):
            self.assertEqual(px.model_size(), 16 + 14 * GIB)
            self.assertEqual(px.size, 16)

        p2.partially_load(cpu, 10 ** 12)
        self.assertEqual(m.model_loaded_weight_memory, 16)
        self.assertEqual(p2.loaded_size(), 16 + 14 * GIB)

        m_reb = _ColdTorchModel(h14, with_weight=True)
        rebuilt = qfm.QFModelPatcher(m_reb, cpu, cpu).adopt_comfy_state_from(p2)
        self.assertEqual(rebuilt.model_size(), 16 + 14 * GIB)
        self.assertEqual(rebuilt.size, 16)

        p_over = qfm.QFModelPatcher(m, cpu, cpu, size=999_999_999)
        self.assertEqual(p_over.size, 0)
        self.assertEqual(p_over.model_size(), 16 + 14 * GIB)
        self.assertEqual(p_over.size, 16)

        cuda0 = torch.device("cuda", 0)
        p_off = qfm.QFModelPatcher(m, cuda0, cpu)
        with mock.patch.object(qfm.comfy.model_patcher.ModelPatcher, "partially_unload", return_value=0) as pu:
            got = p_off.partially_load(cuda0, -512)
        self.assertEqual(got, 0)
        self.assertEqual(pu.call_args[0][0], cpu)

    def test_t8_should_autoload_for_estimate_and_cold_estimate_engine(self):
        plugin = _load_plugin_module()
        for dev, want in (
            (torch.device("cuda", 0), True),
            (torch.device("cpu"), False),
            (torch.device("mps"), False),
        ):
            with self.subTest(dev=dev):
                with mock.patch.object(mm, "get_torch_device", return_value=dev):
                    self.assertIs(plugin._should_autoload_for_estimate(), want)
        with mock.patch.object(mm, "get_torch_device", side_effect=RuntimeError("no dev")):
            self.assertIs(plugin._should_autoload_for_estimate(), False)

        def fake_est(params_ptr, est_ptr):
            est_ptr._obj.resident_weight_bytes = 23 * GIB
            est_ptr._obj.min_offload_bytes = 3 * GIB
            est_ptr._obj.working_set_floor_bytes = 1 * GIB
            est_ptr._obj.recommended_offload_mode = 1
            return plugin.qfe.QF_OK

        fake_lib = types.SimpleNamespace(qf_xfm_estimate_vram=fake_est, qf_last_error=lambda: b"")
        with mock.patch.object(plugin.qfe, "_LIB", None), \
             mock.patch.object(plugin.qfe, "_refuse_second_arch", lambda *_a, **_kw: None), \
             mock.patch.object(plugin, "_should_autoload_for_estimate", return_value=True), \
             mock.patch.object(plugin.qfe, "has_ready_so", return_value=True) as hrs, \
             mock.patch.object(plugin.qfe, "load_lib", return_value=fake_lib) as ll, \
             mock.patch.object(plugin, "_read_auth", return_value=("", "https://auth.example")):
            cold = plugin._estimate_engine("/m.safetensors", {"model_arch": "minimax_h3"}, device_id=1)
        hrs.assert_called_once_with(device_id=1)
        ll.assert_called_once_with(device_id=1)
        self.assertEqual(cold, plugin.qfe.ColdNeed(23 * GIB, None, 3 * GIB, 1 * GIB, 1))

        for auto, ready in ((False, True), (True, False)):
            with mock.patch.object(plugin.qfe, "_LIB", None), \
                 mock.patch.object(plugin, "_should_autoload_for_estimate", return_value=auto), \
                 mock.patch.object(plugin.qfe, "has_ready_so", return_value=ready), \
                 mock.patch.object(plugin.qfe, "load_lib") as ll2:
                cold2 = plugin._estimate_engine("/m.safetensors", device_id=0)
            ll2.assert_not_called()
            self.assertEqual(cold2, plugin.qfe.ColdNeed(None, None))

        with mock.patch.object(plugin.qfe, "_LIB", None), \
             mock.patch.object(plugin, "_should_autoload_for_estimate", return_value=True), \
             mock.patch.object(plugin.qfe, "has_ready_so", return_value=True), \
             mock.patch.object(plugin.qfe, "load_lib", side_effect=RuntimeError("not ready")):
            cold3 = plugin._estimate_engine("/m.safetensors", device_id=0)
        self.assertIsNone(cold3.bytes)
        self.assertIn("engine library not ready (RuntimeError: not ready)", cold3.note)

    def test_t9_model_size_cuda_budget_and_get_total_memory_fallback(self):
        cuda0 = torch.device("cuda", 0)
        cpu = torch.device("cpu")
        cold = qfe.ColdNeed(23 * GIB, None, 3 * GIB, 1 * GIB, 1)
        lazy_b = qfm.QFLazyEngine(lambda: (self.handle, ("k",)), estimate_cold=lambda: cold,
                                  vram_budget_bytes=8 * GIB)
        m = _ColdTorchModel(lazy_b, with_weight=True)
        p_cuda = qfm.QFModelPatcher(m, cuda0, cpu)
        with mock.patch.object(qfm.comfy.model_management, "get_total_memory", return_value=24 * GIB) as gtm:
            sz = p_cuda.model_size()
        gtm.assert_called_once_with(cuda0)
        self.assertEqual(sz, 16 + ((8 * GIB * 17) // 20 - 1 * GIB))

        lazy_nb = qfm.QFLazyEngine(lambda: (self.handle, ("k",)), estimate_cold=lambda: cold, vram_budget_bytes=0)
        m2 = _ColdTorchModel(lazy_nb, with_weight=True)
        p_cuda2 = qfm.QFModelPatcher(m2, cuda0, cpu)
        with mock.patch.object(qfm.comfy.model_management, "get_total_memory", side_effect=RuntimeError("cuda err")):
            sz_fb = p_cuda2.model_size()
        self.assertEqual(sz_fb, 16 + 3 * GIB)

        p_cpu = qfm.QFModelPatcher(m2, cpu, cpu)
        with mock.patch.object(qfm.comfy.model_management, "get_total_memory") as gtm_cpu:
            sz_cpu = p_cpu.model_size()
        gtm_cpu.assert_not_called()
        self.assertEqual(sz_cpu, 16 + 3 * GIB)

    def test_t10_and_t15_lazy_destroy_clears_ledger_and_detach_controls_scratch(self):
        h1 = qfe.QFEngineHandle(self.lib, ctypes.c_void_p(XFM_PTR), capacity_bytes=4096)
        h2 = qfe.QFEngineHandle(self.lib, ctypes.c_void_p(XFM_PTR + 1), capacity_bytes=8192)
        handles = [h1, h2]
        lazy = qfm.QFLazyEngine(lambda: (handles.pop(0), ("k",)), vram_budget_bytes=8 * GIB)
        self.assertEqual(lazy.vram_budget_bytes, 8 * GIB)
        lazy.ensure()
        self.assertEqual(lazy.capacity_bytes, 4096)
        self.assertEqual(lazy.host_weight_bytes, 4096)

        h1.destroy()
        self.assertIsNone(lazy.ensure_if_cached())
        self.assertIsNone(lazy.capacity_bytes)
        self.assertIsNone(lazy.host_weight_bytes)

        lazy.capacity_bytes = 9999
        lazy.host_weight_bytes = 9999
        lazy.ensure()
        self.assertEqual(lazy.capacity_bytes, 8192)
        self.assertEqual(lazy.host_weight_bytes, 8192)

        m = _ColdTorchModel(lazy)
        m._out = torch.ones(2, 2)
        m._out_video = torch.ones(2, 2)
        m._out_audio = torch.ones(2, 2)
        p = qfm.QFModelPatcher(m, torch.device("cpu"), torch.device("cpu"))
        p.detach(False)
        self.assertIsNotNone(m._out)
        self.assertIsNotNone(m._out_video)
        self.assertIsNotNone(m._out_audio)
        p.detach(True)
        self.assertIsNone(m._out)
        self.assertIsNone(m._out_video)
        self.assertIsNone(m._out_audio)

    def test_t8b_headless_multi_gpu_engine_recipe_device_id(self):
        import hashlib
        import tempfile

        plugin = _load_plugin_module()
        pqfe = plugin.qfe
        plat = pqfe._engine_platform()
        self.assertIsNotNone(plat)
        host_name = plat["hosts"][13]
        host_bytes = b"HOST86"
        sha_map = {host_name: hashlib.sha256(host_bytes).hexdigest()}

        with tempfile.TemporaryDirectory() as td:
            pair = Path(td) / "0.0.14-sm86-cu13"
            pair.mkdir(parents=True, exist_ok=True)
            (pair / host_name).write_bytes(host_bytes)
            kern_name = None
            if plat["kernel"]:
                kern_name = pqfe._kernel_filename(13)
                kern_bytes = b"KERN86"
                (pair / kern_name).write_bytes(kern_bytes)
                sha_map[kern_name] = hashlib.sha256(kern_bytes).hexdigest()
            marker = {
                "version": "0.0.14",
                "set": "sm86",
                "cuda": 13,
                "host": host_name,
                "kernel": kern_name,
                "sms": [86],
                "sha256": sha_map,
            }
            (Path(td) / ".engine-sm86-cu13.json").write_text(json.dumps(marker), encoding="utf-8")

            loaded = []
            raw_cdll = object()
            bound_lib = types.SimpleNamespace(qf_set_log_level=lambda *_: None)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(pqfe._ENV_SO_OVERRIDE, None)
                with mock.patch.multiple(
                    pqfe,
                    _LIB=None,
                    _LIB_PATH=None,
                    _ENGINE_DEVICE=0,
                    _FINGERPRINT_PENDING=None,
                    _engine_bin_dir=lambda: td,
                    _torch_cuda_major=lambda: 13,
                    _gpu_sm=lambda dev=0: {0: 89, 1: 86}[int(dev)],
                    assert_toolchain_compatible=lambda p: None,
                    _engine_cuda_needs=lambda p: set(),
                    _bind=lambda lib: bound_lib,
                    _engine_load_ok=lambda p: None,
                    _log_lib_fingerprint=lambda *_: None,
                ), mock.patch.object(
                    pqfe.ctypes, "CDLL",
                    side_effect=lambda p, mode=0: (loaded.append(p), raw_cdll)[1],
                ):
                    lib, ckey, cfg = plugin._engine_recipe("/m.safetensors", {"model_arch": "flux"}, device_id=1)
                    loaded_hosts = [p for p in loaded if isinstance(p, str) and os.path.basename(p) == host_name]
                    self.assertIs(lib, bound_lib)
                    self.assertEqual(len(loaded_hosts), 1)
                    self.assertEqual(os.path.basename(os.path.dirname(loaded_hosts[0])), "0.0.14-sm86-cu13")
                    self.assertEqual(pqfe._ENGINE_DEVICE, 1)
                    self.assertEqual(ckey[2], 1)
                    self.assertEqual(cfg, {"model_arch": "flux"})
                    lib0, ckey0, _ = plugin._engine_recipe("/m.safetensors", {"model_arch": "flux"}, device_id=0)
                    self.assertIs(lib0, bound_lib)
                    self.assertEqual(ckey0[2], 0)

    def test_t10b_lazy_ensure_directly_after_destroy_drops_dead_real(self):
        h1 = qfe.QFEngineHandle(self.lib, ctypes.c_void_p(XFM_PTR), capacity_bytes=4096)
        h2 = qfe.QFEngineHandle(self.lib, ctypes.c_void_p(XFM_PTR + 1), capacity_bytes=8192)
        handles = [h1, h2]
        seen_before_factory = []

        def _factory():
            seen_before_factory.append((lazy.capacity_bytes, lazy.host_weight_bytes))
            return handles.pop(0), ("k",)

        lazy = qfm.QFLazyEngine(_factory)
        self.assertIs(lazy.ensure(), h1)
        self.assertEqual(lazy.capacity_bytes, 4096)
        self.assertEqual(lazy.host_weight_bytes, 4096)

        h1.destroy()
        self.assertIs(lazy.ensure(), h2)
        self.assertEqual(seen_before_factory, [(None, None), (None, None)])
        self.assertEqual(lazy.capacity_bytes, 8192)
        self.assertEqual(lazy.host_weight_bytes, 8192)

    def test_t9_family_build_vram_budget_mb_plumbing_and_legacy_cold_fallback(self):
        deps = {
            "get_engine": lambda *a, **k: (self.handle, ("k",)),
            "bind_pipeline_model": lambda m: None,
            "read_auth": lambda k=None: ("", ""),
        }
        with mock.patch.object(qfm, "current_torch_device", return_value=(torch.device("cpu"), 0)):
            p_pos = qfm.family_build(
                deps, "/w.safetensors", {"vram_budget_mb": 8192},
                lambda cfg: types.SimpleNamespace(**cfg), {}, lambda mc, eng, dev: _ColdTorchModel(eng),
                "test-budget",
            )([])
            p_neg = qfm.family_build(
                deps, "/w.safetensors", {"vram_budget_mb": -10},
                lambda cfg: types.SimpleNamespace(**cfg), {}, lambda mc, eng, dev: _ColdTorchModel(eng),
                "test-neg-budget",
            )([])
        self.assertEqual(p_pos.model._qf.vram_budget_bytes, 8192 << 20)
        self.assertEqual(p_neg.model._qf.vram_budget_bytes, 0)

        lazy_leg = qfm.QFLazyEngine(lambda: (self.handle, ("k",)), estimate_cold=lambda: types.SimpleNamespace(bytes=12345))
        m_leg = _ColdTorchModel(lazy_leg, with_weight=True)
        p_leg = qfm.QFModelPatcher(m_leg, torch.device("cpu"), torch.device("cpu"))
        self.assertEqual(p_leg.model_size(), 16 + 12345)

        lazy_none = qfm.QFLazyEngine(lambda: (self.handle, ("k",)), estimate_cold=lambda: None)
        m_none = _ColdTorchModel(lazy_none, with_weight=True)
        p_none = qfm.QFModelPatcher(m_none, torch.device("cpu"), torch.device("cpu"))
        self.assertEqual(p_none.model_size(), 16)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
