#!/usr/bin/env python3
"""Real loader cache reuse must not evict another host-managed live engine."""
import ctypes
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

root = os.environ.get("COMFY_ROOT")
if not root or not (Path(root) / "comfy/model_management.py").is_file():
    print("[SKIP] host_loader_contract: set COMFY_ROOT to the tested host")
    raise SystemExit(77)
sys.path.insert(0, root)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch

plugin_root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qf_loader_contract", plugin_root / "__init__.py",
                                            submodule_search_locations=[str(plugin_root)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)


class NativeLibrary:
    def __init__(self):
        self.held = 512
        self.unloads = 0

    def quantfunc_unload_sync_ex(self, pipeline, out):
        self.unloads += 1
        out._obj.value = self.held
        self.held = 0
        return 0


class HostLoaderContract(unittest.TestCase):
    def test_every_family_has_exited_the_legacy_capacity_estimator_path(self):
        self.assertNotIn("estimate_footprint", plugin._register_families.__code__.co_consts)
        for name in plugin._FAMILY_MODULES:
            source = (plugin_root / f"{name}.py").read_text(encoding="utf-8")
            self.assertNotIn('deps["estimate_footprint"]', source, name)
            self.assertNotIn("estimate_footprint(model_dir)", source, name)
            self.assertNotIn("footprint~", source, name)
        substrate = (plugin_root / "qf_modelpatcher.py").read_text(encoding="utf-8")
        self.assertNotIn("._resource.set_grant(", substrate)
        self.assertNotIn("._resource.set_device_grant(", substrate)

    def test_no_grant_call_is_left_in_the_plugin_sources(self):
        # #751 deleted the host-grant protocol on both sides: ComfyUI arranges room through load_models_gpu/free_memory.
        sources = [path for path in plugin_root.rglob("*.py")
                   if "tests" not in path.relative_to(plugin_root).parts]
        self.assertIn("qf_modelpatcher.py", {path.name for path in sources})
        for path in sources:
            text = path.read_text(encoding="utf-8")
            for banned in ("set_domain_grants(", "_publish_domain_grants("):
                self.assertNotIn(banned, text, f"{path.relative_to(plugin_root)} still calls {banned}")

    def test_cache_selection_does_not_evict_a_live_other_engine(self):
        other_lib, selected_lib = NativeLibrary(), NativeLibrary()
        other = plugin.qfe.QFEngineHandle(other_lib, ctypes.c_void_p(1), 512)
        selected = plugin.qfe.QFEngineHandle(selected_lib, ctypes.c_void_p(2), 512)
        # the cache key: (library, package, transformer file, backend, device, create config)
        selected_key = ("contract.so", "selected-package", None, "svdq", 0, "{}")
        other_key = ("contract.so", "other-package", None, "svdq", 0, "{}")
        consumer = torch.nn.Module()
        consumer._qf = other
        with mock.patch.dict(plugin._PIPELINE_CACHE, {other_key: other, selected_key: selected}, clear=True), \
             mock.patch.dict(plugin._PIPELINE_MODELS, {}, clear=True), \
             mock.patch.object(plugin.qfe, "loaded_so_path", return_value="contract.so"), \
             mock.patch.object(plugin.qfe, "load_lib", return_value=selected_lib), \
             mock.patch.object(plugin.qfe, "create_pipeline", side_effect=AssertionError("cache hit must not create")):
            plugin._bind_pipeline_model(other_key, consumer)
            selected_consumer = plugin.qfmp.QFLazyEngine(
                lambda: plugin._get_engine("selected-package"))
            with plugin.qfmp._engine_cache_acquisition(selected_consumer):
                engine, key = plugin._get_engine("selected-package")
            self.assertIs(engine, selected)
            self.assertEqual(key, selected_key)
            self.assertIn(selected_consumer, plugin._live_pipeline_models(selected_key))
            self.assertEqual(other_lib.held, 512)
            self.assertEqual(other_lib.unloads, 0)
            self.assertIs(plugin._PIPELINE_CACHE[other_key], other)
            self.assertIs(consumer._qf, other)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
