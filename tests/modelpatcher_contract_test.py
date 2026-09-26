#!/usr/bin/env python3
"""The native resource adapter's ModelPatcher contract (#751 design §6.1) on ComfyUI's real ModelPatcher, with only
the native C ABI doubled: is_dynamic() is False; partially_unload(n) asks release_eligible(n) and returns what the
engine freed; detach() asks release_all; partially_load materializes only and never calls a grant verb.

Run with COMFY_ROOT and its Python environment (CPU only)."""
import ctypes
import importlib
import os
from pathlib import Path
import sys
import types
import unittest

root = os.environ.get("COMFY_ROOT")
if not root or not (Path(root) / "comfy/model_management.py").is_file():
    print("[SKIP] modelpatcher_contract: set COMFY_ROOT to the tested host")
    raise SystemExit(77)
sys.path.insert(0, root)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch

pkg = types.ModuleType("qf_modelpatcher_contract")
pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[pkg.__name__] = pkg
qfe = importlib.import_module("qf_modelpatcher_contract.qf_engine")
qfm = importlib.import_module("qf_modelpatcher_contract.qf_modelpatcher")

OWNED, SHARED = 17, 18
GRANT_VERBS = ("set_grant", "query_grant", "set_device_grant", "query_device_grant", "set_domain_grants")


def native_library():
    """The Owned view (epoch 123) and the device's Shared view. The deleted grant verbs are exported as traps."""
    lib = types.SimpleNamespace(held=1536, phase=qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED, calls=[])

    def acquire(pipeline, version, out):
        out._obj.value = OWNED
        return 0

    def acquire_shared(device, version, out):
        out._obj.value = SHARED
        return 0

    def query(pointer, out):
        out._obj.state, out._obj.device, out._obj.capabilities = 0, 0, 7   # QUERY | RELEASE_ELIGIBLE | RELEASE_ALL
        out._obj.owner_epoch = 123 if pointer.value == OWNED else 0
        return 0

    def residency(pointer, out):
        out._obj.state, out._obj.resident_bytes = 0, lib.held if pointer.value == OWNED else 0
        return 0

    def domain(pointer, out):
        out._obj.state, out._obj.resident_bytes = 0, lib.held
        return 0

    def lifecycle(pointer, out):
        out._obj.state = 0
        out._obj.phase = lib.phase if pointer.value == OWNED else qfe.QUANTFUNC_RESOURCE_PHASE_SHARED
        return 0

    def release_eligible(pointer, requested, out):
        lib.calls.append(("release_eligible", pointer.value, requested))
        freed = min(requested, 1000)   # the engine's walk frees what is eligible, not what was asked
        lib.held -= freed
        out._obj.state, out._obj.freed_bytes = 0, freed
        return 0

    def release_all(pointer, out):
        lib.calls.append(("release_all", pointer.value))
        out._obj.state, out._obj.freed_bytes, lib.held = 0, lib.held, 0
        return 0

    def trap(name):
        def grant_verb(*_args):
            lib.calls.append((name,))
            raise AssertionError(f"quantfunc_resource_{name} is not part of the engine ABI")
        return grant_verb

    for name, function in (("acquire", acquire), ("acquire_shared", acquire_shared), ("query", query),
                           ("query_residency", residency), ("query_domain_residency", domain),
                           ("query_lifecycle", lifecycle), ("release_eligible", release_eligible),
                           ("release_all", release_all), ("destroy", lambda pointer: None),
                           *((name, trap(name)) for name in GRANT_VERBS)):
        setattr(lib, "quantfunc_resource_" + name, function)
    lib.quantfunc_last_error = lambda: b"contract refusal"
    return lib


class ModelPatcherContract(unittest.TestCase):
    def setUp(self):
        self.lib = native_library()
        owned = qfe.NativeResource.acquire(self.lib, ctypes.c_void_p(1))
        shared = qfe.NativeResource.shared(self.lib, 0)
        self.addCleanup(owned.close)
        self.addCleanup(shared.close)
        self.shared = qfm.QFNativeResourcePatcher(shared)
        self.shared._shared_adapter = self.shared
        self.owner = qfm.QFNativeResourcePatcher(owned)
        self.owner._shared_adapter = self.shared

    def test_is_dynamic_is_false(self):
        model = torch.nn.Module()
        model.device = torch.device("cpu")
        logical = qfm.QFModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        for patcher in (self.owner, self.shared, logical):
            self.assertIs(patcher.is_dynamic(), False)

    def test_partially_unload_asks_release_eligible_and_returns_what_it_freed(self):
        self.assertEqual(self.owner.partially_unload(self.owner.offload_device, 4096), 1000)
        self.assertEqual(self.lib.calls, [("release_eligible", OWNED, 4096)])
        self.assertEqual(self.owner.loaded_size(), 536)

    def test_detach_asks_release_all(self):
        self.owner.detach(False)   # a clone handoff keeps the pages
        self.assertEqual(self.lib.calls, [])
        self.owner.detach(True)
        self.assertEqual(self.lib.calls, [("release_all", OWNED)])
        self.assertEqual(self.owner.loaded_size(), 0)

    def test_partially_load_materializes_only_and_never_calls_a_grant_verb(self):
        for budget in (4096, 0, 10 ** 30):   # a loaded identity has nothing to load, whatever ComfyUI's budget
            self.assertEqual(self.owner.partially_load(self.owner.load_device, budget), 0)
        self.assertEqual(self.shared.partially_load(self.shared.load_device, 4096), 0)
        materialized = []
        self.lib.phase = qfe.QUANTFUNC_RESOURCE_PHASE_PREPARED
        self.owner.bind_prepared(types.SimpleNamespace(materialize=lambda: materialized.append(True)), 4096)
        self.assertEqual(self.owner.partially_load(self.owner.load_device, 4096), 0)
        self.assertEqual(materialized, [True])
        self.assertEqual(self.lib.calls, [])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
