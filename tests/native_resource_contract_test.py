#!/usr/bin/env python3
"""Exercise the Python resource boundary without loading a model or CUDA."""
import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys
import gc
import threading
import contextlib
import io
import os
import json

spec = importlib.util.spec_from_file_location("qf_resource_test_engine", Path(__file__).resolve().parents[1] / "qf_engine.py")
qfe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qfe)


def library():
    lib = SimpleNamespace(state=0, status=0, destroyed=[], requests=[], acquired=[], prepared=[], created=[], configured=[])
    def acquire(pipeline, version, out):
        lib.acquired.append((pipeline.value, version))
        out._obj.value = 17 if not lib.status else None
        return lib.status
    def shared(device, version, out):
        lib.acquired.append((device, version))
        out._obj.value = 18 if not lib.status else None
        return lib.status
    def query(handle, out):
        value = out._obj
        assert value.struct_size == 72 and value.abi_version == 1
        value.state, value.device, value.owner_epoch, value.capabilities = lib.state, 2, 123, 3
        value.cca_live, value.cca_cached, value.cca_deferred = 4096, 512, 256
        value.arena_backed, value.arena_pinned = 8192, 4096
        return lib.status
    def release(handle, requested, out):
        value = out._obj
        assert value.struct_size == 24 and value.abi_version == 1
        lib.requests.append(requested)
        value.state, value.freed_bytes = lib.state, 512 if requested else 0
        return lib.status
    lib.quantfunc_resource_acquire = acquire
    lib.quantfunc_resource_acquire_shared = shared
    def prepare(device, version, out):
        lib.prepared.append((device, version))
        out._obj.value = 19 if not lib.status else None
        return lib.status
    def create(params, out):
        lib.created.append((None, params._obj.device_idx, params._obj.model_dir))
        out._obj.value = 91 if not lib.status else None
        return lib.status
    def create_with_resource(params, resource, out):
        lib.created.append((resource.value, params._obj.device_idx, params._obj.model_dir))
        out._obj.value = 92 if not lib.status else None
        return lib.status
    lib.quantfunc_resource_prepare = prepare
    def configure(params, resource):
        p = params._obj
        lib.configured.append((resource.value, p.device_idx, p.model_dir, p.config_json))
        return lib.status
    lib.quantfunc_resource_configure = configure
    lib.quantfunc_create = create
    lib.quantfunc_create_with_resource = create_with_resource
    lib.quantfunc_resource_query = query
    lib.quantfunc_resource_release_eligible = release
    lib.quantfunc_resource_destroy = lambda ptr: lib.destroyed.append(ptr.value)
    lib.quantfunc_last_error = lambda: b"native resource failure"
    return lib


class EngineResourceContract(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(qfe.QFEngineHandle, "create"), "common engine factory lacks prepared identity")

    def test_one_recipe_is_retained_across_configure_and_create(self):
        lib = library()
        lib.quantfunc_destroy = lambda _: None
        cfg = {"api_key": "fixture-only-not-a-secret", "bf16": True}
        params = qfe.make_create_params(model_dir="fixture", device_idx=2, config_json=cfg)
        self.assertNotIn("_cache_dir", cfg)
        cfg["bf16"] = False
        self.assertTrue(json.loads(params.config_json)["bf16"])
        with qfe.NativeResource.prepare(lib, 2) as resource:
            resource.configure(params)
            self.assertEqual(lib.created, [])
            self.assertEqual(lib.configured, [(19, 2, b"fixture", params.config_json)])
            engine = qfe.QFEngineHandle.create(lib, create_params=params, prepared_resource=resource)
            self.assertEqual(lib.created, [(19, 2, b"fixture")])
            engine.destroy()

    def test_retained_recipe_cannot_be_combined_with_new_keywords(self):
        lib = library()
        params = qfe.make_create_params(model_dir="fixture", device_idx=2)
        with self.assertRaisesRegex(ValueError, "not both"):
            qfe.QFEngineHandle.create(lib, create_params=params, model_dir="different")
        self.assertEqual(lib.prepared, [])
        self.assertEqual(lib.created, [])

    def test_explicit_params_do_not_bypass_session_knob_guard(self):
        lib = library()
        params = qfe.make_create_params(model_dir="fixture", device_idx=2)
        params.config_json = b'{"nested":{"step_cache":true}}'
        with self.assertRaisesRegex(RuntimeError, "SESSION knob"):
            qfe.create_pipeline(lib, create_params=params)
        self.assertEqual(lib.created, [])

    def test_configure_failure_and_close_do_not_materialize(self):
        lib = library()
        params = qfe.make_create_params(model_dir="fixture", device_idx=2)
        resource = qfe.NativeResource.prepare(lib, 2)
        lib.status = 1
        with self.assertRaisesRegex(RuntimeError, "configuration failed"):
            resource.configure(params)
        self.assertEqual(lib.created, [])
        self.assertEqual(lib.destroyed, [])
        resource.close()
        with self.assertRaisesRegex(RuntimeError, "view is closed"):
            resource.configure(params)

    def test_destroy_does_not_close_a_view_retained_by_another_consumer(self):
        lib = library()
        lib.quantfunc_destroy = lambda _: None
        engine = qfe.QFEngineHandle.create(lib, model_dir="fixture", device_idx=2)
        retained = engine.resource
        engine.destroy()
        self.assertEqual(lib.destroyed, [], "destroying a pipeline must not close another consumer's view")
        self.assertEqual(retained.query().owner_epoch, 123)
        retained.close()
        self.assertEqual(lib.destroyed, [19])

    def test_factory_consumes_the_callers_prepared_identity_without_replacing_it(self):
        lib = library()
        lib.quantfunc_destroy = lambda _: None
        with qfe.NativeResource.prepare(lib, 2) as prepared:
            engine = qfe.QFEngineHandle.create(
                lib, model_dir="fixture", device_idx=2, prepared_resource=prepared)
            self.assertEqual(lib.prepared, [(2, 1)])
            self.assertEqual(lib.created, [(19, 2, b"fixture")])
            self.assertIs(engine.resource, prepared)
            engine.destroy()
            self.assertEqual(lib.destroyed, [])
            self.assertEqual(prepared.query().owner_epoch, 123)
        self.assertEqual(lib.destroyed, [19])

    def test_failed_create_keeps_callers_prepared_view_open(self):
        lib = library()
        with qfe.NativeResource.prepare(lib, 2) as prepared:
            lib.quantfunc_create_with_resource = lambda *_: 1
            with self.assertRaisesRegex(RuntimeError, "native resource failure"):
                qfe.QFEngineHandle.create(
                    lib, model_dir="fixture", device_idx=2, prepared_resource=prepared)
            self.assertEqual(lib.prepared, [(2, 1)])
            self.assertEqual(lib.destroyed, [])
            self.assertEqual(prepared.query().owner_epoch, 123)
        self.assertEqual(lib.destroyed, [19])

    def test_actual_engine_factory_retains_the_identity_used_for_creation(self):
        lib = library()
        destroyed = []
        lib.quantfunc_destroy = lambda pipeline: destroyed.append(pipeline.value)
        engine = qfe.QFEngineHandle.create(lib, model_dir="fixture", device_idx=2, footprint_bytes=4096)
        self.assertEqual(lib.prepared, [(2, 1)])
        self.assertEqual(lib.created, [(19, 2, b"fixture")])
        self.assertEqual(engine.pipeline.value, 92)
        self.assertEqual(engine.resource.query().owner_epoch, 123)
        self.assertEqual(engine.footprint_bytes, 4096)
        self.assertEqual(lib.destroyed, [])
        engine.destroy()
        self.assertEqual(destroyed, [92])
        self.assertEqual(lib.destroyed, [19])

    def test_failed_creation_closes_prepared_view_without_publishing_an_engine(self):
        lib = library()
        def failed_create(*_):
            return 1
        lib.quantfunc_create_with_resource = failed_create
        with self.assertRaisesRegex(RuntimeError, "native resource failure"):
            qfe.QFEngineHandle.create(lib, model_dir="fixture", device_idx=2)
        self.assertEqual(lib.destroyed, [19])

    def test_python_handle_adoption_failure_destroys_pipeline_and_closes_view(self):
        lib = library()
        destroyed = []
        lib.quantfunc_destroy = lambda pipeline: destroyed.append(pipeline.value)
        class CannotAdopt(qfe.QFEngineHandle):
            def __init__(self, *_args, **_kwargs):
                raise MemoryError("handle adoption failure")
        with self.assertRaises(MemoryError):
            CannotAdopt.create(lib, model_dir="fixture", device_idx=2)
        self.assertEqual(destroyed, [92])
        self.assertEqual(lib.destroyed, [19])

    def test_python_handle_adoption_failure_preserves_callers_view(self):
        lib = library()
        destroyed = []
        lib.quantfunc_destroy = lambda pipeline: destroyed.append(pipeline.value)
        class CannotAdopt(qfe.QFEngineHandle):
            def __init__(self, *_args, **_kwargs):
                raise MemoryError("handle adoption failure")
        with qfe.NativeResource.prepare(lib, 2) as prepared:
            with self.assertRaisesRegex(MemoryError, "handle adoption failure"):
                CannotAdopt.create(lib, model_dir="fixture", device_idx=2,
                                   prepared_resource=prepared)
            self.assertEqual(lib.prepared, [(2, 1)])
            self.assertEqual(lib.created, [(19, 2, b"fixture")])
            self.assertEqual(destroyed, [92])
            self.assertEqual(lib.destroyed, [])
            # Only view ownership is tested here. A real destroyed native
            # identity is Closed, not promised reusable for another create.
        self.assertEqual(lib.destroyed, [19])

    def test_unload_diagnostic_reports_native_owner_and_shared_without_changing_freed(self):
        lib = library()
        def unload(_pipeline, out):
            out._obj.value = 4096
            return 0
        lib.quantfunc_unload_sync_ex = unload
        lib.quantfunc_destroy = lambda _: None
        engine = qfe.QFEngineHandle.create(lib, model_dir="fixture", device_idx=2)
        output = io.StringIO()
        with patch.dict(os.environ, {"QF_NATIVE_PROF": "1"}), contextlib.redirect_stdout(output):
            self.assertEqual(engine.unload_vram(), 4096)
        self.assertIn("unload resource owner=ResourceSnapshot", output.getvalue())
        self.assertIn("shared=ResourceSnapshot", output.getvalue())
        self.assertIn("cca_live=4096", output.getvalue())
        self.assertIn("arena_backed=8192", output.getvalue())
        self.assertEqual(lib.destroyed, [18])
        engine.destroy()
        self.assertEqual(lib.destroyed, [18, 19])


class GrantContract(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(qfe.NativeResource, "enroll_host"), "finite grant Python bridge missing")

    @staticmethod
    def grant_library():
        lib = library()
        lib.grant_calls = []
        lib.enrolled = 1
        def call(name, handle, limit, out):
            value = out._obj
            assert value.struct_size == 32 and value.abi_version == 1
            lib.grant_calls.append((name, handle.value, limit))
            value.state, value.enrolled = lib.state, lib.enrolled
            value.limit_bytes, value.pending_bytes = (1 << 63) + 512, 1024
            return lib.status
        lib.quantfunc_resource_enroll_host = lambda h, o: call("enroll", h, None, o)
        lib.quantfunc_resource_query_grant = lambda h, o: call("query", h, None, o)
        lib.quantfunc_resource_set_grant = lambda h, n, o: call("set", h, n, o)
        lib.quantfunc_resource_query_device_grant = lambda h, o: call("query_device", h, None, o)
        lib.quantfunc_resource_set_device_grant = lambda h, n, o: call("set_device", h, n, o)
        return lib

    def test_native_fields_are_forwarded_without_estimation(self):
        lib = self.grant_library()
        with qfe.NativeResource.prepare(lib, 2) as resource:
            for operation in (resource.enroll_host, resource.query_grant, lambda: resource.set_grant((1 << 64) - 1)):
                self.assertEqual(operation(), (0, True, (1 << 63) + 512, 1024))
            self.assertEqual(lib.grant_calls, [("enroll", 19, None), ("query", 19, None), ("set", 19, (1 << 64) - 1)])
            self.assertEqual(lib.requests, [])
            self.assertEqual(lib.created, [])

    def test_device_grant_fields_are_forwarded_without_occupancy_arithmetic(self):
        lib = self.grant_library()
        with qfe.NativeResource.shared(lib, 2) as resource:
            self.assertEqual(resource.query_device_grant(), (0, True, (1 << 63) + 512, 1024))
            self.assertEqual(resource.set_device_grant((1 << 64) - 1),
                             (0, True, (1 << 63) + 512, 1024))
        self.assertEqual(lib.grant_calls,
                         [("query_device", 18, None), ("set_device", 18, (1 << 64) - 1)])
        self.assertEqual(lib.quantfunc_resource_query_device_grant.restype, ctypes.c_int)
        self.assertEqual(lib.quantfunc_resource_query_device_grant.argtypes,
                         [ctypes.c_void_p, ctypes.POINTER(qfe._ResourceGrant)])
        self.assertEqual(lib.quantfunc_resource_set_device_grant.restype, ctypes.c_int)
        self.assertEqual(lib.quantfunc_resource_set_device_grant.argtypes,
                         [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(qfe._ResourceGrant)])

    def test_device_grant_nonready_and_unenrolled_never_become_numeric_permission(self):
        lib = self.grant_library()
        with qfe.NativeResource.shared(lib, 2) as resource:
            for state in (1, 2, 3):
                lib.state = state
                self.assertEqual(resource.query_device_grant(), (state, None, None, None))
                self.assertEqual(resource.set_device_grant(512), (state, None, None, None))
            lib.state = 0
            lib.enrolled = 0
            self.assertEqual(resource.query_device_grant(), (0, False, None, None))
            self.assertEqual(resource.set_device_grant(512), (0, False, None, None))

    def test_device_grant_rejects_invalid_missing_error_and_closed_calls(self):
        lib = self.grant_library()
        resource = qfe.NativeResource.shared(lib, 2)
        for value in (-1, 1 << 64, 2.5):
            with self.assertRaises((ValueError, TypeError)):
                resource.set_device_grant(value)
        self.assertEqual(lib.grant_calls, [])

        lib.status = 1
        for operation in (resource.query_device_grant, lambda: resource.set_device_grant(512)):
            with self.assertRaisesRegex(RuntimeError, "native resource failure"):
                operation()
        lib.status = 0
        calls_before_missing = list(lib.grant_calls)
        for name, operation in (("query_device_grant", resource.query_device_grant),
                                ("set_device_grant", lambda: resource.set_device_grant(512))):
            delattr(lib, "quantfunc_resource_" + name)
            with self.assertRaisesRegex(RuntimeError, "quantfunc_resource_" + name):
                operation()
        self.assertEqual(lib.grant_calls, calls_before_missing)

        resource.close()
        calls_before_closed = list(lib.grant_calls)
        for operation in (resource.query_device_grant, lambda: resource.set_device_grant(0)):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                operation()
        self.assertEqual(lib.grant_calls, calls_before_closed)
        self.assertEqual(lib.destroyed, [18])

    def test_nonready_and_unenrolled_never_become_numeric_permission(self):
        lib = self.grant_library()
        with qfe.NativeResource.prepare(lib, 2) as resource:
            for state in (1, 2, 3):
                lib.state = state
                for operation in (resource.enroll_host, resource.query_grant, lambda: resource.set_grant(512)):
                    self.assertEqual(operation(), (state, None, None, None))
            lib.state = 0; lib.enrolled = 0
            self.assertEqual(resource.query_grant(), (0, False, None, None))

    def test_invalid_integers_errors_missing_symbols_and_closed_view(self):
        lib = self.grant_library()
        resource = qfe.NativeResource.prepare(lib, 2)
        for value in (-1, 1 << 64, 2.5):
            with self.assertRaises((ValueError, TypeError)):
                resource.set_grant(value)
        self.assertEqual(lib.grant_calls, [])
        lib.status = 1
        for operation in (resource.enroll_host, resource.query_grant, lambda: resource.set_grant(512)):
            with self.assertRaisesRegex(RuntimeError, "native resource failure"):
                operation()
        lib.status = 0
        for name, operation in (("enroll_host", resource.enroll_host), ("query_grant", resource.query_grant),
                                ("set_grant", lambda: resource.set_grant(512))):
            delattr(lib, "quantfunc_resource_" + name)
            with self.assertRaisesRegex(RuntimeError, "quantfunc_resource_" + name):
                operation()
        self.assertEqual(lib.created, [])
        resource.close()
        for operation in (resource.enroll_host, resource.query_grant, lambda: resource.set_grant(0)):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                operation()

    def test_each_grant_call_holds_the_view_lock_until_native_returns(self):
        operations = (("enroll_host", False), ("query_grant", False), ("set_grant", False),
                      ("query_device_grant", True), ("set_device_grant", True))
        for name, device_grant in operations:
            with self.subTest(operation=name):
                lib = self.grant_library()
                resource = (qfe.NativeResource.shared if device_grant else qfe.NativeResource.prepare)(lib, 2)
                original = getattr(lib, "quantfunc_resource_" + name)
                observed = []
                def checked(*args):
                    acquired = resource._lock.acquire(blocking=False)
                    observed.append(acquired)
                    if acquired:
                        resource._lock.release()
                    return original(*args)
                setattr(lib, "quantfunc_resource_" + name, checked)
                getattr(resource, name)(*([512] if name.startswith("set_") else []))
                self.assertEqual(observed, [False])
                resource.close()
                self.assertEqual(lib.destroyed, [18 if device_grant else 19])

    def test_concurrent_close_waits_for_grant_success_and_failure(self):
        operations = (("enroll_host", False), ("query_grant", False), ("set_grant", False),
                      ("query_device_grant", True), ("set_device_grant", True))
        for name, device_grant in operations:
            for status in (0, 1):
                with self.subTest(operation=name, status=status):
                    lib = self.grant_library()
                    resource = (qfe.NativeResource.shared if device_grant else qfe.NativeResource.prepare)(lib, 2)
                    lib.status = status
                    entered, finish, contended = (threading.Event() for _ in range(3))
                    lock = resource._lock
                    class ObservedLock:
                        def __enter__(self):
                            if not lock.acquire(blocking=False):
                                contended.set()
                                lock.acquire()
                            return self
                        def __exit__(self, *_):
                            lock.release()
                    resource._lock = ObservedLock()
                    original = getattr(lib, "quantfunc_resource_" + name)
                    def waiting(*args):
                        entered.set()
                        if not finish.wait(3):
                            raise AssertionError("grant call barrier timed out")
                        if lib.destroyed:
                            raise AssertionError("view destroyed during native grant call")
                        return original(*args)
                    setattr(lib, "quantfunc_resource_" + name, waiting)
                    errors = []
                    def calling():
                        try:
                            getattr(resource, name)(*([512] if name.startswith("set_") else []))
                        except BaseException as error:
                            errors.append(error)
                    caller = threading.Thread(target=calling)
                    closer = threading.Thread(target=resource.close)
                    caller.start()
                    reached = entered.wait(3)
                    closer.start()
                    blocked = contended.wait(2)
                    destroyed_early = bool(lib.destroyed)
                    finish.set()
                    caller.join(3); closer.join(3)
                    self.assertTrue(reached and blocked)
                    self.assertFalse(caller.is_alive() or closer.is_alive() or destroyed_early)
                    self.assertEqual(len(errors), status)
                    if status:
                        self.assertIsInstance(errors[0], RuntimeError)
                        self.assertIn("native resource failure", str(errors[0]))
                    self.assertEqual(lib.destroyed, [18 if device_grant else 19])


class FrozenCapacityDomainContract(unittest.TestCase):
    @staticmethod
    def authority_library():
        lib = library()
        lib.capacity_state = qfe.QUANTFUNC_RESOURCE_READY
        lib.capacity_bytes = (1 << 63) + 4096
        lib.capacity_components = 3
        lib.capacity_status = qfe.QUANTFUNC_OK
        lib.domain_state = qfe.QUANTFUNC_RESOURCE_READY
        lib.domain_bytes = (1 << 62) + 2048
        lib.domain_status = qfe.QUANTFUNC_OK
        lib.domain_grant_state = qfe.QUANTFUNC_RESOURCE_READY
        lib.domain_grant_status = qfe.QUANTFUNC_OK
        lib.domain_grant_calls = []

        def capacity(handle, out):
            value = out._obj
            assert (value.struct_size, value.abi_version) == (24, 1)
            value.state = lib.capacity_state
            value.component_count = lib.capacity_components
            value.required_persistent_bytes = lib.capacity_bytes
            return lib.capacity_status

        def domain(handle, out):
            value = out._obj
            assert (value.struct_size, value.abi_version) == (24, 1)
            value.state = lib.domain_state
            value.resident_bytes = lib.domain_bytes
            return lib.domain_status

        def domain_grants(shared, owned, command, out):
            request, result = command._obj, out._obj
            assert (request.struct_size, request.abi_version) == (40, 1)
            assert (result.struct_size, result.abi_version) == (16, 1)
            lib.domain_grant_calls.append(
                (shared.value, owned.value if owned else None, request.mask,
                 request.owner_limit_bytes, request.shared_limit_bytes,
                 request.device_limit_bytes))
            result.state = lib.domain_grant_state
            result.applied_mask = request.mask if result.state == qfe.QUANTFUNC_RESOURCE_READY else 0
            return lib.domain_grant_status

        lib.quantfunc_resource_query_capacity = capacity
        lib.quantfunc_resource_query_domain_residency = domain
        lib.quantfunc_resource_set_domain_grants = domain_grants
        return lib

    def test_frozen_ctypes_layouts_match_the_header(self):
        self.assertEqual(ctypes.sizeof(qfe._ResourceCapacity), 24)
        self.assertEqual(ctypes.sizeof(qfe._ResourceDomain), 24)
        self.assertEqual(ctypes.sizeof(qfe._ResourceDomainGrants), 40)
        self.assertEqual(ctypes.sizeof(qfe._ResourceDomainGrantsResult), 16)
        self.assertEqual(qfe._ResourceCapacity.required_persistent_bytes.offset, 16)
        self.assertEqual(qfe._ResourceDomain.resident_bytes.offset, 16)
        self.assertEqual(qfe._ResourceDomainGrants.owner_limit_bytes.offset, 16)
        self.assertEqual(qfe._ResourceDomainGrants.shared_limit_bytes.offset, 24)
        self.assertEqual(qfe._ResourceDomainGrants.device_limit_bytes.offset, 32)
        self.assertEqual(qfe._ResourceDomainGrantsResult.applied_mask.offset, 12)

    def test_capacity_is_ready_positive_and_never_calls_create(self):
        lib = self.authority_library()
        lib.quantfunc_create = lambda *_: self.fail("capacity query must not create")
        lib.quantfunc_create_with_resource = lambda *_: self.fail("capacity query must not create")
        params = qfe.make_create_params(model_dir="fixture", device_idx=2)
        with qfe.NativeResource.prepare(lib, 2) as resource:
            resource.configure(params)
            result = resource.query_capacity()
        self.assertEqual(result.component_count, 3)
        self.assertEqual(result.required_persistent_bytes, (1 << 63) + 4096)
        self.assertEqual(lib.created, [])
        self.assertEqual(lib.quantfunc_resource_query_capacity.restype, ctypes.c_int)
        self.assertEqual(lib.quantfunc_resource_query_capacity.argtypes,
                         [ctypes.c_void_p, ctypes.POINTER(qfe._ResourceCapacity)])

    def test_capacity_missing_unsupported_nonready_closed_and_zero_fail_closed(self):
        lib = self.authority_library()
        resource = qfe.NativeResource.prepare(lib, 2)
        params = qfe.make_create_params(model_dir="fixture", device_idx=2)
        resource.configure(params)
        del lib.quantfunc_resource_query_capacity
        with self.assertRaises(qfe.NativeContractUnavailable):
            resource.query_capacity()
        resource.close()
        lib = self.authority_library()
        resource = qfe.NativeResource.prepare(lib, 2)
        resource.configure(params)
        for status, state, byte_count in (
            (qfe.QUANTFUNC_ERROR_UNSUPPORTED, qfe.QUANTFUNC_RESOURCE_CAPACITY_UNSUPPORTED, 123),
            (qfe.QUANTFUNC_OK, qfe.QUANTFUNC_RESOURCE_BUSY, 123),
            (qfe.QUANTFUNC_OK, qfe.QUANTFUNC_RESOURCE_UNKNOWN, 123),
            (qfe.QUANTFUNC_OK, qfe.QUANTFUNC_RESOURCE_CLOSED, 123),
            (qfe.QUANTFUNC_OK, qfe.QUANTFUNC_RESOURCE_READY, 0),
        ):
            with self.subTest(status=status, state=state, bytes=byte_count):
                lib.capacity_status, lib.capacity_state, lib.capacity_bytes = status, state, byte_count
                with self.assertRaises((qfe.NativeContractUnavailable, RuntimeError)):
                    resource.query_capacity()
        resource.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            resource.query_capacity()

    def test_domain_snapshot_accepts_only_ready_positive_or_zero_numeric_result(self):
        lib = self.authority_library()
        with qfe.NativeResource.shared(lib, 2) as shared:
            self.assertEqual(shared.query_domain_residency().resident_bytes, (1 << 62) + 2048)
            lib.domain_bytes = 0
            self.assertEqual(shared.query_domain_residency().resident_bytes, 0)
            for state in (qfe.QUANTFUNC_RESOURCE_BUSY, qfe.QUANTFUNC_RESOURCE_UNKNOWN,
                          qfe.QUANTFUNC_RESOURCE_CLOSED):
                lib.domain_state = state
                with self.assertRaises(RuntimeError):
                    shared.query_domain_residency()
            lib.domain_state, lib.domain_status = qfe.QUANTFUNC_RESOURCE_READY, 1
            with self.assertRaisesRegex(RuntimeError, "native resource failure"):
                shared.query_domain_residency()

        lib = self.authority_library()
        shared = qfe.NativeResource.shared(lib, 2)
        del lib.quantfunc_resource_query_domain_residency
        with self.assertRaises(qfe.NativeContractUnavailable):
            shared.query_domain_residency()
        shared.close()

    def test_atomic_domain_grants_forward_mask_and_limits_in_one_call(self):
        lib = self.authority_library()
        with qfe.NativeResource.shared(lib, 2) as shared, qfe.NativeResource.prepare(lib, 2) as owner:
            result = shared.set_domain_grants(
                owner, qfe.QUANTFUNC_RESOURCE_GRANT_OWNER |
                qfe.QUANTFUNC_RESOURCE_GRANT_SHARED |
                qfe.QUANTFUNC_RESOURCE_GRANT_DEVICE,
                owner_limit_bytes=101, shared_limit_bytes=202, device_limit_bytes=303)
            self.assertEqual((result.state, result.applied_mask), (qfe.QUANTFUNC_RESOURCE_READY, 7))
            shared.set_domain_grants(None,
                                     qfe.QUANTFUNC_RESOURCE_GRANT_SHARED |
                                     qfe.QUANTFUNC_RESOURCE_GRANT_DEVICE,
                                     shared_limit_bytes=11, device_limit_bytes=22)
        self.assertEqual(lib.domain_grant_calls,
                         [(18, 19, 7, 101, 202, 303), (18, None, 6, 0, 11, 22)])

    def test_atomic_domain_grants_nonready_error_and_invalid_inputs_never_succeed(self):
        lib = self.authority_library()
        with qfe.NativeResource.shared(lib, 2) as shared, qfe.NativeResource.prepare(lib, 2) as owner:
            for state in (qfe.QUANTFUNC_RESOURCE_BUSY, qfe.QUANTFUNC_RESOURCE_UNKNOWN,
                          qfe.QUANTFUNC_RESOURCE_CLOSED):
                lib.domain_grant_state = state
                with self.assertRaises(RuntimeError):
                    shared.set_domain_grants(owner, 7, owner_limit_bytes=1,
                                             shared_limit_bytes=2, device_limit_bytes=3)
            lib.domain_grant_state, lib.domain_grant_status = qfe.QUANTFUNC_RESOURCE_READY, 1
            with self.assertRaisesRegex(RuntimeError, "native resource failure"):
                shared.set_domain_grants(owner, 7, owner_limit_bytes=1,
                                         shared_limit_bytes=2, device_limit_bytes=3)
            for mask, owned, owner_limit, shared_limit, device_limit in (
                (0, None, 0, 0, 0), (8, None, 0, 0, 0),
                (7, None, 1, 2, 3), (6, owner, 0, 2, 3),
                (6, None, 1, 2, 3), (1, owner, 1, 2, 0),
            ):
                with self.assertRaises((TypeError, ValueError)):
                    shared.set_domain_grants(owned, mask, owner_limit_bytes=owner_limit,
                                             shared_limit_bytes=shared_limit,
                                             device_limit_bytes=device_limit)

    def test_atomic_domain_call_holds_both_views_against_close(self):
        lib = self.authority_library()
        shared = qfe.NativeResource.shared(lib, 2)
        owner = qfe.NativeResource.prepare(lib, 2)
        original = lib.quantfunc_resource_set_domain_grants
        observed = []

        def checked(*args):
            acquired = []
            for resource in (shared, owner):
                got = resource._lock.acquire(blocking=False)
                acquired.append(got)
                if got:
                    resource._lock.release()
            observed.append(tuple(acquired))
            return original(*args)

        lib.quantfunc_resource_set_domain_grants = checked
        shared.set_domain_grants(owner, 7, owner_limit_bytes=1,
                                 shared_limit_bytes=2, device_limit_bytes=3)
        self.assertEqual(observed, [(False, False)])
        owner.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            shared.set_domain_grants(owner, 7, owner_limit_bytes=1,
                                     shared_limit_bytes=2, device_limit_bytes=3)
        shared.close()

    def test_atomic_domain_call_serializes_concurrent_owner_close(self):
        lib = self.authority_library()
        shared = qfe.NativeResource.shared(lib, 2)
        owner = qfe.NativeResource.prepare(lib, 2)
        entered, finish = threading.Event(), threading.Event()
        original = lib.quantfunc_resource_set_domain_grants

        def waiting(*args):
            entered.set()
            if not finish.wait(3):
                raise AssertionError("atomic grant barrier timed out")
            self.assertNotIn(19, lib.destroyed)
            return original(*args)

        lib.quantfunc_resource_set_domain_grants = waiting
        errors = []

        def publish():
            try:
                shared.set_domain_grants(owner, 7, owner_limit_bytes=1,
                                         shared_limit_bytes=2, device_limit_bytes=3)
            except BaseException as error:
                errors.append(error)

        caller = threading.Thread(target=publish, daemon=True)
        closer = threading.Thread(target=owner.close, daemon=True)
        caller.start()
        self.assertTrue(entered.wait(3))
        closer.start()
        self.assertNotIn(19, lib.destroyed)
        finish.set()
        caller.join(3)
        closer.join(3)
        self.assertFalse(caller.is_alive() or closer.is_alive())
        self.assertEqual(errors, [])
        self.assertIn(19, lib.destroyed)
        shared.close()


class PreparedResourceContract(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(qfe.NativeResource, "prepare"), "pre-load resource identity bridge missing")

    def test_prepare_does_not_load_and_create_forwards_the_same_native_view(self):
        lib = library()
        with qfe.NativeResource.prepare(lib, 2) as resource:
            self.assertEqual(lib.prepared, [(2, 1)])
            self.assertEqual(lib.created, [])
            handle = qfe.create_pipeline(lib, model_dir="fixture", device_idx=2, prepared_resource=resource)
            self.assertEqual(handle.value, 92)
            self.assertEqual(lib.created, [(19, 2, b"fixture")])
        self.assertEqual(lib.destroyed, [19])
        self.assertEqual(lib.requests, [])
        self.assertEqual(qfe.create_pipeline(lib, model_dir="legacy").value, 91)
        self.assertEqual(lib.created[-1], (None, 0, b"legacy"))

    def test_prepare_validates_devices_and_cleans_up_failed_python_adoption(self):
        lib = library()
        for device in (-1, 1 << 31, 2.5):
            with self.assertRaises((TypeError, ValueError)):
                qfe.NativeResource.prepare(lib, device)
        self.assertEqual(lib.prepared, [])
        with patch.object(qfe.weakref, "finalize", side_effect=MemoryError("finalizer allocation")):
            with self.assertRaises(MemoryError):
                qfe.NativeResource.prepare(lib, 2)
        self.assertEqual(lib.destroyed, [19])
        class CannotAllocate(qfe.NativeResource):
            def __new__(cls, *args):
                raise MemoryError("object allocation")
        with self.assertRaises(MemoryError):
            CannotAllocate.prepare(lib, 2)
        self.assertEqual(lib.destroyed, [19, 19])
        lib.status = 1
        with self.assertRaisesRegex(RuntimeError, "native resource failure"):
            qfe.NativeResource.prepare(lib, 2)

    def test_missing_extensions_never_fall_back_to_unregistered_creation(self):
        lib = library()
        del lib.quantfunc_resource_prepare
        with self.assertRaisesRegex(RuntimeError, "quantfunc_resource_prepare"):
            qfe.NativeResource.prepare(lib, 2)
        self.assertEqual(qfe.create_pipeline(lib, model_dir="legacy").value, 91)
        lib = library()
        with qfe.NativeResource.prepare(lib, 2) as resource:
            del lib.quantfunc_create_with_resource
            with self.assertRaisesRegex(RuntimeError, "quantfunc_create_with_resource"):
                qfe.create_pipeline(lib, model_dir="fixture", device_idx=2, prepared_resource=resource)
            self.assertEqual(lib.created, [])
            self.assertEqual(resource.query().owner_epoch, 123)

    def test_closed_cross_library_and_narrowing_inputs_are_refused_before_native_create(self):
        lib = library()
        with qfe.NativeResource.prepare(lib, 2) as resource:
            other = library()
            with self.assertRaises((ValueError, RuntimeError)):
                qfe.create_pipeline(other, model_dir="fixture", device_idx=2, prepared_resource=resource)
            self.assertEqual(other.created, [])
            for device in (-1, 1 << 31, 1 << 32, 2.5):
                with self.assertRaises((TypeError, ValueError)):
                    qfe.create_pipeline(lib, model_dir="fixture", device_idx=device, prepared_resource=resource)
            self.assertEqual(lib.created, [])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            qfe.create_pipeline(lib, model_dir="fixture", device_idx=2, prepared_resource=resource)
        self.assertEqual(lib.created, [])

    def test_create_failure_preserves_view_and_retries_through_native_authority(self):
        lib = library()
        with qfe.NativeResource.prepare(lib, 2) as resource:
            lib.status = 1
            with self.assertRaisesRegex(RuntimeError, "native resource failure"):
                qfe.create_pipeline(lib, model_dir="fixture", device_idx=2, prepared_resource=resource)
            lib.status = 0
            self.assertEqual(qfe.create_pipeline(lib, model_dir="fixture", device_idx=2,
                                                 prepared_resource=resource).value, 92)
            self.assertEqual(lib.destroyed, [])
            def broken(*_):
                raise OSError("create ABI failure")
            lib.quantfunc_create_with_resource = broken
            with self.assertRaisesRegex(OSError, "create ABI failure"):
                qfe.create_pipeline(lib, model_dir="fixture", device_idx=2, prepared_resource=resource)
            self.assertEqual(resource.query().owner_epoch, 123)
            lib.quantfunc_create_with_resource = lambda *_: 0  # invalid success with NULL output
            with self.assertRaisesRegex(RuntimeError, "quantfunc_create_with_resource"):
                qfe.create_pipeline(lib, model_dir="fixture", device_idx=2, prepared_resource=resource)
        self.assertEqual(lib.destroyed, [19])

    def test_close_cannot_free_a_view_borrowed_by_native_create(self):
        for status in (0, 1):
            with self.subTest(native_status=status):
                self._check_close_during_create(status)

    def _check_close_during_create(self, status):
        lib = library()
        resource = qfe.NativeResource.prepare(lib, 2)
        lib.status = status
        entered, finish, contended, closed = (threading.Event() for _ in range(4))
        view_lock = resource._lock
        class ObservedLock:
            def __enter__(self):
                if not view_lock.acquire(blocking=False):
                    contended.set()  # actual acquisition failed, not merely a thread-start signal
                    view_lock.acquire()
                return self
            def __exit__(self, *_):
                view_lock.release()
        resource._lock = ObservedLock()
        errors, handles = [], []
        create = lib.quantfunc_create_with_resource
        def waiting(*args):
            entered.set()
            if not finish.wait(3):
                raise AssertionError("create barrier timed out")
            if lib.destroyed:
                raise AssertionError("native view destroyed during create")
            return create(*args)
        lib.quantfunc_create_with_resource = waiting
        def creating():
            try:
                handles.append(qfe.create_pipeline(lib, model_dir="fixture", device_idx=2,
                                                  prepared_resource=resource).value)
            except BaseException as error:
                errors.append(error)
        def closing_view():
            resource.close()
            closed.set()
        creator = threading.Thread(target=creating)
        closer = threading.Thread(target=closing_view)
        creator.start()
        self.assertTrue(entered.wait(3))
        closer.start()
        observed_contention = contended.wait(2)
        premature = closed.is_set()
        finish.set()
        creator.join(3)
        closer.join(3)
        self.assertFalse(creator.is_alive() or closer.is_alive())
        self.assertTrue(observed_contention, "close must reach the held native-view lock")
        self.assertFalse(premature)
        if status:
            self.assertEqual(handles, [])
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeError)
            self.assertIn("native resource failure", str(errors[0]))
        else:
            self.assertEqual(errors, [])
            self.assertEqual(handles, [92])
        self.assertEqual(lib.destroyed, [19])


class ResourceContract(unittest.TestCase):
    def test_residency_uses_native_aggregate_and_preserves_unknown(self):
        lib = library()
        self.assertTrue(hasattr(qfe.NativeResource, "residency"), "native aggregate bridge missing")
        calls = []
        def residency(handle, out):
            value = out._obj
            self.assertEqual((value.struct_size, value.abi_version), (24, 1))
            calls.append(handle.value)
            value.state, value.resident_bytes = lib.state, 123456789
            return lib.status
        lib.quantfunc_resource_query_residency = residency
        lib.quantfunc_resource_query = lambda *_: self.fail("must not sum category snapshots in Python")
        with qfe.NativeResource.shared(lib, 0) as resource:
            self.assertEqual(resource.residency().resident_bytes, 123456789)
            for state in (1, 2, 3):
                lib.state = state
                result = resource.residency()
                self.assertEqual(result.state, state)
                self.assertIsNone(result.resident_bytes)
            lib.status = 1
            with self.assertRaisesRegex(RuntimeError, "native resource failure"):
                resource.residency()
            def broken(*_):
                raise OSError("residency ABI call failed")
            lib.quantfunc_resource_query_residency = broken
            with self.assertRaisesRegex(OSError, "residency ABI call failed"):
                resource.residency()
        self.assertEqual(calls, [18] * 5)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            resource.residency()

    def test_missing_residency_extension_keeps_v1_queries_usable(self):
        self.assertTrue(hasattr(qfe.NativeResource, "residency"), "native aggregate bridge missing")
        lib = library()
        with qfe.NativeResource.shared(lib, 0) as resource:
            with self.assertRaisesRegex(RuntimeError, "quantfunc_resource_query_residency"):
                resource.residency()
            self.assertEqual(resource.query().cca_live, 4096)
            self.assertEqual(resource.release_eligible(0).freed_bytes, 0)

    def test_python_adoption_failure_releases_the_acquired_native_view(self):
        lib = library()
        with patch.object(qfe.weakref, "finalize", side_effect=MemoryError("finalizer allocation")):
            for acquire in (lambda: qfe.NativeResource.shared(lib, 0),
                            lambda: qfe.NativeResource.acquire(lib, ctypes.c_void_p(9))):
                with self.assertRaises(MemoryError):
                    acquire()
        self.assertEqual(lib.destroyed, [18, 17])

    def test_python_object_allocation_failure_releases_native_view(self):
        lib = library()
        class CannotAllocate(qfe.NativeResource):
            def __new__(cls, *args):
                raise MemoryError("object allocation")
        with self.assertRaises(MemoryError):
            CannotAllocate.shared(lib, 0)
        self.assertEqual(lib.destroyed, [18])

    def test_interrupted_finalizer_registration_cannot_double_destroy(self):
        lib = library()
        finalize = qfe.weakref.finalize
        def interrupted(*args):
            finalize(*args)
            raise KeyboardInterrupt("interrupted after registration")
        with patch.object(qfe.weakref, "finalize", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                qfe.NativeResource.shared(lib, 0)
        gc.collect()
        self.assertEqual(lib.destroyed, [18])

    def test_abandoned_view_is_destroyed_once_without_reclaim(self):
        lib = library()
        resource = qfe.NativeResource.shared(lib, 0)
        del resource
        gc.collect()
        self.assertEqual(lib.destroyed, [18])
        self.assertEqual(lib.requests, [])

    def test_close_waits_for_an_inflight_view_query(self):
        for operation, native_name in (("query", "quantfunc_resource_query"),
                                       ("residency", "quantfunc_resource_query_residency")):
            with self.subTest(operation=operation):
                self._check_close_waits(operation, native_name)

    def _check_close_waits(self, operation, native_name):
        lib = library()
        resource = qfe.NativeResource.shared(lib, 0)
        entered, finish, closing, closed = (threading.Event() for _ in range(4))
        errors = []
        query = getattr(lib, native_name, lambda *_: 0)
        def waiting_query(*args):
            entered.set()
            if not finish.wait(3):
                raise AssertionError("query barrier timed out")
            if lib.destroyed:
                raise AssertionError("view freed during query")
            return query(*args)
        setattr(lib, native_name, waiting_query)
        def querying():
            try:
                getattr(resource, operation)()
            except BaseException as error:
                errors.append(error)
        def closing_view():
            closing.set()
            resource.close()
            closed.set()
        reader = threading.Thread(target=querying)
        closer = threading.Thread(target=closing_view)
        reader.start()
        self.assertTrue(entered.wait(3))
        closer.start()
        self.assertTrue(closing.wait(3))
        premature = closed.wait(0.05)
        finish.set()
        reader.join(3)
        closer.join(3)
        self.assertFalse(premature)
        self.assertFalse(reader.is_alive() or closer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(lib.destroyed, [18])

    def test_ready_preserves_native_categories_without_double_counting_pins(self):
        lib = library()
        with qfe.NativeResource.acquire(lib, ctypes.c_void_p(9)) as resource:
            value = resource.query()
            self.assertEqual((value.device, value.owner_epoch, value.capabilities), (2, 123, 3))
            self.assertEqual((value.cca_live, value.cca_cached, value.cca_deferred,
                              value.arena_backed, value.arena_pinned), (4096, 512, 256, 8192, 4096))
            self.assertEqual(lib.acquired, [(9, 1)])
        self.assertEqual(lib.destroyed, [17])

    def test_nonready_never_exposes_invalid_native_counts(self):
        lib = library()
        with qfe.NativeResource.shared(lib, 2) as resource:
            for state in (1, 2, 3):
                lib.state = state
                value = resource.query()
                self.assertEqual((value.state, value.owner_epoch), (state, 123))
                self.assertIsNone(value.cca_live)
                self.assertIsNone(value.cca_cached)
                self.assertIsNone(value.cca_deferred)
                self.assertIsNone(value.arena_backed)
                self.assertIsNone(value.arena_pinned)
                freed = resource.release_eligible(99)
                self.assertEqual(freed.state, state)
                self.assertIsNone(freed.freed_bytes)

    def test_native_errors_and_ffi_exceptions_propagate(self):
        lib = library()
        with qfe.NativeResource.shared(lib, 0) as resource:
            lib.status = 1
            for operation in (resource.query, lambda: resource.release_eligible(1)):
                with self.assertRaisesRegex(RuntimeError, "native resource failure"):
                    operation()
            def broken(*args):
                raise OSError("broken ABI")
            lib.quantfunc_resource_query = broken
            with self.assertRaisesRegex(OSError, "broken ABI"):
                resource.query()
        with self.assertRaisesRegex(RuntimeError, "native resource failure"):
            qfe.NativeResource.acquire(lib, ctypes.c_void_p(9))

    def test_close_is_idempotent_and_never_reclaims(self):
        lib = library()
        resource = qfe.NativeResource.shared(lib, 0)
        resource.close()
        resource.close()
        self.assertEqual(lib.destroyed, [18])
        self.assertEqual(lib.requests, [])
        for operation in (resource.query, lambda: resource.release_eligible(0)):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                operation()

    def test_unsigned_requests_do_not_wrap_or_truncate(self):
        lib = library()
        with qfe.NativeResource.shared(lib, 0) as resource:
            for invalid in (-1, 1 << 64, 1.5):
                with self.assertRaises((TypeError, ValueError)):
                    resource.release_eligible(invalid)
            self.assertEqual(lib.requests, [])
            self.assertEqual(resource.release_eligible(0).freed_bytes, 0)
            resource.release_eligible((1 << 64) - 1)
            self.assertEqual(lib.requests, [0, (1 << 64) - 1])

    def test_older_library_and_invalid_device_fail_before_native_call(self):
        lib = library()
        for device in (-1, 1 << 31, 2.5):
            with self.assertRaises((TypeError, ValueError)):
                qfe.NativeResource.shared(lib, device)
        self.assertEqual(lib.acquired, [])
        del lib.quantfunc_resource_release_eligible
        with self.assertRaisesRegex(RuntimeError, "quantfunc_resource_release_eligible"):
            qfe.NativeResource.shared(lib, 0)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--actual-so":
        lib = ctypes.CDLL(sys.argv[2], mode=ctypes.RTLD_GLOBAL)
        with qfe.NativeResource.shared(lib, 0) as resource:
            snapshot = resource.query()
            assert snapshot.state == 0 and snapshot.device == 0 and snapshot.owner_epoch == 0
            assert snapshot.capabilities == 3
            assert all(v == 0 for v in snapshot[4:]), snapshot
            residency = resource.residency()
            assert residency == qfe.ResourceResidency(0, 0), residency
            released = resource.release_eligible(0)
            assert released == qfe.ResourceRelease(0, 0), released
        try:
            qfe.NativeResource.acquire(lib, None)
        except RuntimeError as error:
            assert "acquisition failed" in str(error)
        else:
            raise AssertionError("null model accepted")
        with qfe.NativeResource.prepare(lib, 0) as resource:
            snapshot = resource.query()
            assert snapshot.state == 0 and snapshot.device == 0 and snapshot.owner_epoch != 0
            assert resource.residency() == qfe.ResourceResidency(0, 0)
            try:
                qfe.create_pipeline(lib, model_dir="", device_idx=0, prepared_resource=resource)
            except RuntimeError as error:
                assert "model_dir is required" in str(error), error
            else:
                raise AssertionError("empty model directory accepted")
            assert resource.query().owner_epoch == snapshot.owner_epoch
            assert resource.release_eligible(1) == qfe.ResourceRelease(0, 0)
        with qfe.NativeResource.shared(lib, 0) as shared, qfe.NativeResource.prepare(lib, 0) as owned:
            assert owned.query_grant() == qfe.ResourceGrant(0, False, None, None)
            assert shared.enroll_host() == qfe.ResourceGrant(0, True, 0, 0)
            assert owned.enroll_host() == qfe.ResourceGrant(0, True, 0, 0)
            assert owned.set_grant((1 << 64) - 1) == qfe.ResourceGrant(0, True, (1 << 64) - 1, 0)
            assert owned.enroll_host() == qfe.ResourceGrant(0, True, (1 << 64) - 1, 0)
            assert owned.query_grant() == qfe.ResourceGrant(0, True, (1 << 64) - 1, 0)
            assert owned.set_grant(0) == qfe.ResourceGrant(0, True, 0, 0)
            assert owned.residency() == qfe.ResourceResidency(0, 0)
        print("NATIVE_RESOURCE_PYTHON_ACTUAL_ABI_PASS")
    else:
        unittest.main()
