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
    def test_cache_selection_does_not_evict_a_live_other_engine(self):
        other_lib, selected_lib = NativeLibrary(), NativeLibrary()
        other = plugin.qfe.QFEngineHandle(other_lib, ctypes.c_void_p(1), 512)
        selected = plugin.qfe.QFEngineHandle(selected_lib, ctypes.c_void_p(2), 512)
        selected_key = ("contract.so", "selected-package", "svdq", 0, "{}")
        other_key = ("contract.so", "other-package", "svdq", 0, "{}")
        consumer = torch.nn.Module()
        consumer._qf = other
        with mock.patch.dict(plugin._PIPELINE_CACHE, {other_key: other, selected_key: selected}, clear=True), \
             mock.patch.dict(plugin._PIPELINE_MODELS, {}, clear=True), \
             mock.patch.dict(os.environ, {"QF_NATIVE_CREATE_EXTRA": ""}), \
             mock.patch.object(plugin.qfe, "resolve_so_path", return_value="contract.so"), \
             mock.patch.object(plugin.qfe, "load_lib", return_value=selected_lib), \
             mock.patch.object(plugin.qfe, "create_pipeline", side_effect=AssertionError("cache hit must not create")):
            plugin._bind_pipeline_model(other_key, consumer)
            engine, key = plugin._get_engine("selected-package")
            self.assertIs(engine, selected)
            self.assertEqual(key, selected_key)
            self.assertEqual(other_lib.held, 512)
            self.assertEqual(other_lib.unloads, 0)
            self.assertIs(plugin._PIPELINE_CACHE[other_key], other)
            self.assertIs(consumer._qf, other)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
