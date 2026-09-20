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
    lib = SimpleNamespace(state=0, status=0, destroyed=[], requests=[], acquired=[])
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
    lib.quantfunc_resource_query = query
    lib.quantfunc_resource_release_eligible = release
    lib.quantfunc_resource_destroy = lambda ptr: lib.destroyed.append(ptr.value)
    lib.quantfunc_last_error = lambda: b"native resource failure"
    return lib


class ResourceContract(unittest.TestCase):
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
        lib = library()
        resource = qfe.NativeResource.shared(lib, 0)
        entered, finish, closing, closed = (threading.Event() for _ in range(4))
        errors = []
        query = lib.quantfunc_resource_query
        def waiting_query(*args):
            entered.set()
            if not finish.wait(3):
                raise AssertionError("query barrier timed out")
            if lib.destroyed:
                raise AssertionError("view freed during query")
            return query(*args)
        lib.quantfunc_resource_query = waiting_query
        def querying():
            try:
                resource.query()
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
            released = resource.release_eligible(0)
            assert released == qfe.ResourceRelease(0, 0), released
        try:
            qfe.NativeResource.acquire(lib, None)
        except RuntimeError as error:
            assert "acquisition failed" in str(error)
        else:
            raise AssertionError("null model accepted")
        print("NATIVE_RESOURCE_PYTHON_ACTUAL_ABI_PASS")
    else:
        unittest.main()
