#!/usr/bin/env python3
"""Lifecycle and physical validity are separate native observations."""
import unittest
from native_resource_contract_test import library, qfe, ctypes


class LifecycleContract(unittest.TestCase):
    def test_closed_target_can_still_have_ready_physical_occupancy(self):
        lib = library()
        def lifecycle(pointer, out):
            self.assertEqual((out._obj.struct_size, out._obj.abi_version), (16, 1))
            out._obj.state = qfe.QUANTFUNC_RESOURCE_READY
            out._obj.phase = qfe.QUANTFUNC_RESOURCE_PHASE_CLOSED
            return 0
        lib.quantfunc_resource_query_lifecycle = lifecycle
        with qfe.NativeResource.acquire(lib, ctypes.c_void_p(7)) as resource:
            self.assertEqual(resource.query().state, qfe.QUANTFUNC_RESOURCE_READY)
            self.assertEqual(resource.lifecycle(), qfe.ResourceLifecycle(0, qfe.QUANTFUNC_RESOURCE_PHASE_CLOSED))

    def test_unknown_busy_and_missing_api_are_not_reload_permission(self):
        lib = library()
        with qfe.NativeResource.acquire(lib, ctypes.c_void_p(7)) as resource:
            with self.assertRaises(qfe.NativeContractUnavailable):
                resource.lifecycle()
            def lifecycle(pointer, out):
                out._obj.state = lib.state
                out._obj.phase = qfe.QUANTFUNC_RESOURCE_PHASE_ATTACHED
                return lib.status
            lib.quantfunc_resource_query_lifecycle = lifecycle
            for state in (qfe.QUANTFUNC_RESOURCE_BUSY, qfe.QUANTFUNC_RESOURCE_UNKNOWN):
                lib.state = state
                self.assertEqual(resource.lifecycle(), qfe.ResourceLifecycle(state, None))
            lib.status = 1
            with self.assertRaisesRegex(RuntimeError, "lifecycle query failed"):
                resource.lifecycle()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            resource.lifecycle()


if __name__ == "__main__":
    unittest.main()
