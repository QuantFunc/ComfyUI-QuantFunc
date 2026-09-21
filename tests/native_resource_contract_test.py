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

spec = importlib.util.spec_from_file_location("qf_resource_test_engine", Path(__file__).resolve().parents[1] / "qf_engine.py")
qfe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qfe)


def library():
    lib = SimpleNamespace(state=0, status=0, destroyed=[], requests=[], acquired=[], prepared=[], created=[])
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
    lib.quantfunc_create = create
    lib.quantfunc_create_with_resource = create_with_resource
    lib.quantfunc_resource_query = query
    lib.quantfunc_resource_release_eligible = release
    lib.quantfunc_resource_destroy = lambda ptr: lib.destroyed.append(ptr.value)
    lib.quantfunc_last_error = lambda: b"native resource failure"
    return lib


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
        print("NATIVE_RESOURCE_PYTHON_ACTUAL_ABI_PASS")
    else:
        unittest.main()
