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

PROBE_OWNER_LEDGER = "--probe-owner-ledger" in sys.argv
root = os.environ.get("COMFY_ROOT")
if not root or not (Path(root) / "comfy/model_management.py").is_file():
    print("[SKIP] host_scheduler_contract: set COMFY_ROOT to the tested host")
    raise SystemExit(77)
sys.path.insert(0, root)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
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

    def quantfunc_vram_need_bytes(self, pipeline, dims, ndim, out):
        out._obj.value = 0
        return 0


class HostSchedulerContract(unittest.TestCase):
    def setUp(self):
        self.library = NativeLibrary()
        self.engine = qfe.QFEngineHandle(self.library, ctypes.c_void_p(1), footprint_bytes=1024)
        self.model = torch.nn.Module()
        self.model.contract_weight = torch.nn.Parameter(torch.ones(32, dtype=torch.float32))
        self.model.device = torch.device("cuda:0")
        self.model._qf = self.engine
        self.patcher = qfm.QFModelPatcher(self.model, self.model.device, torch.device("cpu"))
        self.model.model_loaded_weight_memory = mm.module_size(self.model)

    def tearDown(self):
        # No native calls from fixture teardown.
        self.model._qf = None
        self.patcher = None
        gc.collect()

    def probe_owned_residency_excludes_peers_and_shared_cache(self):
        """Production-base acceptance RED, separate from the passing bridge suite.

        Two real handles/patchers share a device. Only the native ABI is doubled;
        each owner has a different measured count and the device includes Shared.
        Querying an owner must not report its peer or the Shared dependency.
        """
        counts = {11: 512, 22: 1024, 33: 2048}
        self.library.held = 512 + 1024 + 2048
        def residency(pointer, out):
            out._obj.state = qfe.QUANTFUNC_RESOURCE_READY
            out._obj.resident_bytes = counts[pointer.value]
            return 0
        def query(pointer, out):
            out._obj.state = qfe.QUANTFUNC_RESOURCE_READY
            out._obj.device = 0
            out._obj.owner_epoch = 0 if pointer.value == 33 else pointer.value
            out._obj.capabilities = 3
            return 0
        def lifecycle(pointer, out):
            out._obj.state = qfe.QUANTFUNC_RESOURCE_READY
            out._obj.phase = (qfe.QUANTFUNC_RESOURCE_PHASE_SHARED
                              if pointer.value == 33 else qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED)
            return 0
        def domain(pointer, out):
            out._obj.state = qfe.QUANTFUNC_RESOURCE_READY
            out._obj.resident_bytes = sum(counts.values())
            return 0
        def grant(pointer, out):
            out._obj.state = qfe.QUANTFUNC_RESOURCE_READY
            out._obj.enrolled = 1
            out._obj.limit_bytes = counts.get(pointer.value, sum(counts.values()))
            out._obj.pending_bytes = 0
            return 0
        def shared(device, version, out):
            out._obj.value = 33
            return 0
        def unexpected(*args):
            raise AssertionError("read-only graph expansion must not acquire an attached owner or release backing")
        self.library.quantfunc_resource_acquire = unexpected
        self.library.quantfunc_resource_release_eligible = unexpected
        self.library.quantfunc_last_error = lambda: b"owner graph boundary refused"
        self.library.quantfunc_resource_query = query
        self.library.quantfunc_resource_query_lifecycle = lifecycle
        self.library.quantfunc_resource_query_residency = residency
        self.library.quantfunc_resource_query_domain_residency = domain
        self.library.quantfunc_resource_query_grant = grant
        self.library.quantfunc_resource_query_device_grant = grant
        self.library.quantfunc_resource_acquire_shared = shared
        self.library.quantfunc_resource_destroy = lambda _: None
        # Validate the full production binder before the intended missing-graph
        # assertion, so future wiring cannot be blocked by an incomplete double.
        with qfe.NativeResource.shared(self.library, 0) as shared_view:
            self.assertEqual(shared_view.residency().resident_bytes, 2048)
        resources = [qfe.NativeResource(self.library, ctypes.c_void_p(key))
                     for key in (11, 22)]
        try:
            models, patchers = [], []
            for index, resource in enumerate(resources):
                model = torch.nn.Module()
                model.device = torch.device("cuda:0")
                model._qf = qfe.QFEngineHandle(
                    self.library, ctypes.c_void_p(index + 1), 4096, resource=resource)
                models.append(model)
                patchers.append(qfm.QFModelPatcher(model, model.device, torch.device("cpu")))
            # Physical records belong to canonical dependencies, not logical
            # MODEL wrappers. Both official host expansion paths must see them.
            published = [p.model_patches_models() for p in patchers]
            self.assertEqual([len(group) for group in published], [2, 2])
            dependencies = [set(group) for group in published]
            self.assertEqual([sorted(d.loaded_size() for d in group) for group in dependencies],
                             [[512, 2048], [1024, 2048]],
                             "each MODEL must expose its owner plus canonical Shared")
            for patcher, expected in zip(patchers, dependencies):
                self.assertEqual(set(patcher.get_nested_additional_models()), expected)
                self.assertEqual(set(patcher.clone().model_patches_models()), expected)
            common = dependencies[0] & dependencies[1]
            self.assertEqual(len(common), 1)
            self.assertEqual(next(iter(common)).loaded_size(), 2048)
            self.assertEqual([p.loaded_size() for p in patchers], [0, 0],
                             "logical wrappers must not double-charge dependencies")
            self.assertEqual(sum(d.loaded_size() for d in set.union(*dependencies)), 3584)
            counts[11] = 0
            self.assertEqual([sorted(d.loaded_size() for d in group) for group in dependencies],
                             [[0, 2048], [1024, 2048]])
        finally:
            for resource in resources:
                resource.close()

    def test_logical_patcher_reports_only_official_torch_bytes(self):
        torch_bytes = self.model.contract_weight.nbytes
        self.library.held = 8192
        self.library.query_status = 1
        self.engine.unloaded = True
        self.assertEqual(self.patcher.model_size(), torch_bytes)
        self.assertEqual(self.patcher.loaded_size(), torch_bytes)
        self.assertEqual(mm.LoadedModel(self.patcher).model_loaded_memory(), torch_bytes)
        self.assertEqual(self.library.query_calls, 0)

    def test_common_model_demand_failure_reaches_host(self):
        class Model(qfm.QFSessionModelMixin):
            pass
        model = Model()
        model._qf = self.engine
        def demand(*args):
            raise RuntimeError("native demand unavailable")
        self.library.quantfunc_vram_need_bytes = demand
        with self.assertRaisesRegex(RuntimeError, "native demand unavailable"):
            model.memory_required([1, 16, 64, 64])

    def test_real_family_models_share_demand_failure_and_torch_only_reclaim(self):
        # Real constructors/MRO, not extracted methods or renamed dummy models.
        # Only the native CUDA library is replaced, as in the other host tests.
        import comfy.supported_models as supported
        cases = (
            ("qf_wan_modelpatcher", "QFWanModel", "WAN21_I2V", "wan2.1", {"start_image": None}),
            ("qf_h3_modelpatcher", "QFH3Model", "MiniMaxH3", "minimax_h3", {}),
            ("qf_krea2_modelpatcher", "QFKrea2Model", "Krea2", "krea2", {}),
            ("qf_ltx_modelpatcher", "QFLTXModel", "LTXV", "ltxv", {"connector": None}),
            ("qf_ltx_modelpatcher", "QFLTXAVModel", "LTXAV", "ltxav", {}),
        )
        for module_name, class_name, config_name, image_model, kwargs in cases:
            with self.subTest(model=class_name):
                module = importlib.import_module(f"qf_host_contract.{module_name}")
                cfg = getattr(supported, config_name)({
                    "image_model": image_model, "model_type": "i2v",
                    "disable_unet_model_creation": True,
                })
                qfm.ensure_model_config_attrs(cfg)
                model = getattr(module, class_name)(cfg, self.engine, device=torch.device("cpu"), **kwargs)
                patcher = module.QFModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
                # In a LEAF module: Comfy's load list manages only leaf-module parameters. One set
                # directly on the root is skipped as "default weights in a non-leaf module" once
                # the root has a parameterized child - Wan/Krea2 arm the stub's 0-byte concat-shape
                # carrier, so a root-level weight there is never loaded and never reclaimed.
                model.contract = torch.nn.Module()
                model.contract.weight = torch.nn.Parameter(torch.ones(8, dtype=torch.float32))
                torch_bytes = mm.module_size(model)
                # Comfy's own loader, not a hand-written byte counter: it also marks each
                # module comfy_patched_weights, the only state its partial unload frees.
                patcher.load(torch.device("cpu"), full_load=True)
                self.library.held = 512
                self.assertEqual(mm.LoadedModel(patcher).model_loaded_memory(), torch_bytes)
                freed = patcher.partially_unload(torch.device("cpu"), 1)
                self.assertGreater(freed, 0)
                self.assertEqual(patcher.loaded_size(), torch_bytes - freed)
                self.assertEqual(self.library.held, 512)
                self.assertEqual(self.library.unload_calls, 0)
                def demand(*args):
                    raise RuntimeError("family demand unavailable")
                self.library.quantfunc_vram_need_bytes = demand
                with self.assertRaisesRegex(RuntimeError, "family demand unavailable"):
                    model.memory_required([1, 16, 64, 64])

    def test_official_outer_sample_supplies_current_av_geometry_before_admission(self):
        import comfy.samplers
        import comfy.nested_tensor
        import comfy.supported_models as supported

        # Use the real sample/pack/wrapper dispatch. Only the GPU sampling body
        # is replaced with an admission probe, before inner_sample writes shapes.
        class AdmissionProbe(comfy.samplers.CFGGuider):
            cancel = False
            def outer_sample(self, noise, latent_image, *args, **kwargs):
                self.model_patcher.model.memory_required(list(noise.shape))
                if self.cancel:
                    raise KeyboardInterrupt("cancel admission probe")
                return latent_image

        for modname, clsname, cfgname, image_model, channels in (
            ("qf_h3_modelpatcher", "QFH3Model", "MiniMaxH3", "minimax_h3", 24),
            ("qf_ltx_modelpatcher", "QFLTXAVModel", "LTXAV", "ltxav", 128),
        ):
            with self.subTest(model=clsname):
                cfg = getattr(supported, cfgname)({"image_model": image_model, "disable_unet_model_creation": True})
                qfm.ensure_model_config_attrs(cfg)
                cls = getattr(importlib.import_module(f"qf_host_contract.{modname}"), clsname)
                model = cls(cfg, self.engine, device=torch.device("cpu"))
                patcher = qfm.QFModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
                observed = []
                def native_demand(pipeline, dims, ndim, out):
                    observed.append(list(dims))
                    if any(d <= 0 or d > 32768 for d in dims):
                        return 1  # actual native API's extent validation
                    out._obj.value = 4096
                    return 0
                self.library.quantfunc_vram_need_bytes = native_demand
                # Missing metadata; changed N; same N but a different aspect.
                for height, width in ((28, 48), (32, 48), (24, 64)):
                    video = torch.zeros((1, channels, 37, height, width))
                    audio = torch.zeros((1, 8, 16, 16))
                    latent = comfy.nested_tensor.NestedTensor((video, audio))
                    # Clone must retain exactly the same common hook semantics.
                    guider = AdmissionProbe(patcher.clone())
                    result = guider.sample(latent, latent, None, torch.tensor([1.0, 0.0]))
                    self.assertEqual(observed[-1], [1, channels, 37, height, width])
                    self.assertEqual(tuple(result.unbind()[0].shape), tuple(video.shape))
                    # A direct caller's metadata is visible again after scope
                    # exit; leaking the request context makes this return H/W.
                    alternate = (1, channels, 37, 1, height * width)
                    model.latent_shapes = [alternate, audio.shape]
                    packed = [1, 1, video.numel() + audio.numel()]
                    self.assertEqual(model._qf_engine_latent_dims(packed), list(alternate))
                guider.cancel = True
                with self.assertRaisesRegex(KeyboardInterrupt, "cancel admission probe"):
                    guider.sample(latent, latent, None, torch.tensor([1.0, 0.0]))
                self.assertEqual(model._qf_engine_latent_dims(packed), list(alternate))

    def test_logical_partial_unload_reclaims_only_torch_weights(self):
        # Reach "loaded" through Comfy's own loader (see the family test): setUp's byte
        # counter alone describes a model Comfy never produces, and its unload frees 0 of it.
        self.patcher.load(torch.device("cpu"), full_load=True)
        torch_bytes = self.patcher.loaded_size()
        self.assertEqual(torch_bytes, self.model.contract_weight.nbytes)  # never a vacuous 0 == 0
        self.assertEqual(self.patcher.partially_unload(torch.device("cpu"), 1), torch_bytes)
        self.assertEqual(self.patcher.loaded_size(), 0)
        self.assertEqual(self.library.held, 512)
        self.assertEqual(self.library.query_calls, 0)
        self.assertEqual(self.library.unload_calls, 0)

    def test_logical_full_unload_leaves_native_reclaim_to_dependencies(self):
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
            self.assertEqual(self.library.held, 512)
            self.assertEqual(self.library.query_calls, 0)
            self.assertEqual(self.library.unload_calls, 0)
            self.assertEqual(detached, [(True, 512)])
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
        lib = types.SimpleNamespace(held=held, eligible=eligible, state=0, capabilities=3,
                                    status=0, requests=[], closed=[], device=device)
        def acquire(pipeline, version, out):
            out._obj.value = 17
            return 0
        def shared(device_index, version, out):
            out._obj.value = 18
            return 0
        def residency(pointer, out):
            out._obj.state = lib.state
            out._obj.resident_bytes = lib.held if pointer.value == 17 else 0
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
            out._obj.owner_epoch = 123 if pointer.value == 17 else 0
            out._obj.capabilities = lib.capabilities
            # Deliberately unlike the aggregate: these fields may identify
            # the resource's device but must never become Python byte policy.
            out._obj.cca_live, out._obj.arena_backed = 7000, 9000
            return lib.status
        def lifecycle(pointer, out):
            out._obj.state = lib.state
            out._obj.phase = (qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED
                              if pointer.value == 17 else qfe.QUANTFUNC_RESOURCE_PHASE_SHARED)
            return lib.status
        def domain(pointer, out):
            out._obj.state = lib.state
            out._obj.resident_bytes = lib.held
            return lib.status
        def grant(pointer, out):
            out._obj.state, out._obj.enrolled = lib.state, 1
            out._obj.limit_bytes, out._obj.pending_bytes = lib.held, 0
            return lib.status
        lib.quantfunc_resource_acquire = acquire
        lib.quantfunc_resource_acquire_shared = shared
        lib.quantfunc_resource_query_residency = residency
        lib.quantfunc_resource_query_domain_residency = domain
        lib.quantfunc_resource_release_eligible = release
        lib.quantfunc_resource_query = query
        lib.quantfunc_resource_query_lifecycle = lifecycle
        lib.quantfunc_resource_query_grant = grant
        lib.quantfunc_resource_query_device_grant = grant
        lib.quantfunc_resource_destroy = lambda pointer: lib.closed.append(pointer.value)
        lib.quantfunc_last_error = lambda: b"resource contract refused"
        resource = qfe.NativeResource.acquire(lib, ctypes.c_void_p(1))
        shared_resource = qfe.NativeResource.shared(lib, device)
        self.addCleanup(resource.close)
        self.addCleanup(shared_resource.close)
        patcher = qfm.QFNativeResourcePatcher(resource)
        shared_patcher = qfm.QFNativeResourcePatcher(shared_resource)
        domain_state = qfm._CanonicalResourceDomain(shared_patcher)
        shared_patcher._domain = domain_state
        shared_patcher._shared_adapter = shared_patcher
        shared_patcher._domain_key = (qfe.library_identity(lib), device)
        patcher._domain = domain_state
        patcher._shared_adapter = shared_patcher
        patcher._domain_key = shared_patcher._domain_key
        patcher._owner_epoch = 123
        domain_state.owners[123] = patcher
        return lib, patcher

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
            for _ in range(10):
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
        for state, status in ((2, 0), (3, 0), (0, 1)):
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

    def test_704_free_memory_completes_past_a_persistent_busy(self):
        """#704: Comfy's own free_memory sizes and unloads this model while native stays BUSY. It runs on every
        load_models_gpu, and in unload_all_models behind POST /free and the OOM handler. Each method answers by its
        declared policy (qfm._COMFY_BUSY_POLICY), so nothing escapes:
        - sizing answers the last READY residency, never 0;
        - no release is issued;
        - the eviction stops with growth fenced;
        - the owner is re-admitted formally on its next load."""
        lib, patcher = self.make_resource()
        lib.capabilities |= qfe.QUANTFUNC_RESOURCE_CAP_RELEASE_ALL  # without it detach refuses on the capability
        with self.host_registry(), mock.patch.object(qfm, "_NATIVE_BUSY_DEADLINE_S", 0.05, create=True):
            self.load(patcher)
            requests = list(lib.requests)
            self.assertFalse(patcher._domain.shared_growth_fenced or patcher._needs_readmission)
            lib.state = qfe.QUANTFUNC_RESOURCE_BUSY
            with mock.patch.object(mm, "get_free_memory", side_effect=self.free_memory(0)):
                mm.free_memory(4096, patcher.load_device)
            self.assertEqual(mm.current_loaded_models, [])
            self.assertEqual(lib.requests, requests)
            self.assertTrue(patcher._domain.shared_growth_fenced and patcher._needs_readmission)
            self.assertEqual(patcher.loaded_size(), 1536)

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

    def test_clone_handoff_preflight_busy_preserves_record_before_detach(self):
        """Observable QF failure is rejected during dependency expansion."""
        lib, patcher = self.make_resource()
        with self.host_registry():
            self.load(patcher)
            original = mm.current_loaded_models[0]
            finalizer = original.model_finalizer
            lib.state = qfe.QUANTFUNC_RESOURCE_BUSY
            # Identity survives BUSY (#704); a value read that STAYS busy is refused after the bounded retry.
            with mock.patch.object(qfm, "_NATIVE_BUSY_DEADLINE_S", 0.05, create=True), \
                    self.assertRaisesRegex(RuntimeError, "preflight.*unavailable|stayed BUSY"):
                self.load(patcher)
            self.assertEqual(mm.current_loaded_models, [original])
            self.assertIs(original.model_finalizer, finalizer)
            self.assertTrue(finalizer.alive)


if __name__ == "__main__":
    if PROBE_OWNER_LEDGER:
        result = unittest.TextTestRunner().run(unittest.TestSuite([
            HostSchedulerContract("probe_owned_residency_excludes_peers_and_shared_cache")]))
        raise SystemExit(not result.wasSuccessful())
    unittest.main(argv=[sys.argv[0]])
