#!/usr/bin/env python3
"""New release-all FFI contract, no Comfy import/native DSO/GPU required."""
import unittest
from native_resource_contract_test import library, qfe, ctypes


class FullReleaseContract(unittest.TestCase):
    def test_native_status_and_exact_count_without_residency_arithmetic(self):
        lib = library()
        calls = []
        def release(pointer, out):
            self.assertEqual((out._obj.struct_size, out._obj.abi_version), (24, 1))
            calls.append(pointer.value)
            out._obj.state, out._obj.freed_bytes = lib.state, 12345
            return lib.status
        lib.quantfunc_resource_release_all = release
        with qfe.NativeResource.acquire(lib, ctypes.c_void_p(7)) as resource:
            self.assertEqual(resource.release_all(), qfe.ResourceRelease(0, 12345))
            for state in (1, 2, 3):
                lib.state = state
                self.assertEqual(resource.release_all(), qfe.ResourceRelease(state, None))
            lib.status = 1
            with self.assertRaisesRegex(RuntimeError, "full resource release failed"):
                resource.release_all()
            self.assertTrue(resource._finalizer.alive)
        self.assertEqual(calls, [17] * 5)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            resource.release_all()

    def test_closed_native_target_is_forwarded_but_missing_abi_never_falls_back(self):
        lib = library()
        with qfe.NativeResource.acquire(lib, ctypes.c_void_p(7)) as resource:
            with self.assertRaises(qfe.NativeContractUnavailable):
                resource.release_all()
            self.assertEqual(lib.requests, [])
            lib.state = qfe.QUANTFUNC_RESOURCE_CLOSED
            self.assertEqual(resource.query().state, qfe.QUANTFUNC_RESOURCE_CLOSED)
            def release(pointer, out):
                out._obj.state, out._obj.freed_bytes = 0, 512
                return 0
            lib.quantfunc_resource_release_all = release
            self.assertEqual(resource.release_all(), qfe.ResourceRelease(0, 512))
            self.assertEqual(resource.query().state, qfe.QUANTFUNC_RESOURCE_CLOSED)


if __name__ == "__main__":
    unittest.main()
