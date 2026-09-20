#!/usr/bin/env python3
"""Native-bridge regression: a refused unload must not look like success.

No GPU/Comfy dependency: execute the real bridge with a deterministic C-ABI
double. It substitutes the external library, not the Python behavior under test.
"""
import ctypes
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("qf_engine_contract", Path(__file__).resolve().parents[1] / "qf_engine.py")
qfe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qfe)


class NativeLibrary:
    def __init__(self, status=0, error=None):
        self.status = status
        self.error = error
        self.unload_calls = 0
        self.end_status = 0
        self.calls = []
        self.freed = 384

    def quantfunc_last_error(self):
        return b"pipeline busy"

    def quantfunc_denoise_end(self, session):
        self.calls.append("end")
        return self.end_status

    def quantfunc_unload_sync(self, pipeline):
        raise AssertionError("legacy status-only unload cannot report actual bytes")

    def quantfunc_unload_sync_ex(self, pipeline, out):
        self.calls.append("unload")
        self.unload_calls += 1
        if self.error:
            raise self.error
        out._obj.value = self.freed
        if self.status == 0:
            self.freed = 0
        return self.status


class HostReclaimContract(unittest.TestCase):
    def test_missing_or_refused_capacity_estimate_is_not_zero(self):
        with self.assertRaisesRegex(RuntimeError, "quantfunc_estimate_resident_bytes"):
            qfe.estimate_resident_bytes(NativeLibrary(), "model")
        for status in (1, 8):
            library = NativeLibrary()
            library.quantfunc_estimate_resident_bytes = lambda params, out: status
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, "pipeline busy"):
                qfe.estimate_resident_bytes(library, "model")

    def test_capacity_estimate_preserves_native_bytes_and_device(self):
        library = NativeLibrary()
        def estimate(params, out):
            self.assertEqual(params._obj.model_dir, b"model")
            self.assertEqual(params._obj.device_idx, 1)
            out._obj.value = 123456789
            return 0
        library.quantfunc_estimate_resident_bytes = estimate
        self.assertEqual(qfe.estimate_resident_bytes(library, "model", device_idx=1), 123456789)

    def test_capacity_ffi_failure_and_missing_input_are_explicit(self):
        library = NativeLibrary()
        def estimate(params, out):
            raise OSError("estimate ABI failed")
        library.quantfunc_estimate_resident_bytes = estimate
        with self.assertRaisesRegex(OSError, "estimate ABI failed"):
            qfe.estimate_resident_bytes(library, "model")
        with self.assertRaises(ValueError):
            qfe.estimate_resident_bytes(library, None)

    def test_missing_residency_query_is_not_zero(self):
        engine = qfe.QFEngineHandle(NativeLibrary(), ctypes.c_void_p(1))
        with self.assertRaisesRegex(RuntimeError, "quantfunc_resident_vram_bytes"):
            engine.resident_vram_bytes()

    def test_residency_refusal_and_ffi_errors_are_not_zero(self):
        for error in (None, OSError("query ABI failed")):
            library = NativeLibrary()
            def query(pipeline, out):
                if error is not None:
                    raise error
                return 1
            library.quantfunc_resident_vram_bytes = query
            engine = qfe.QFEngineHandle(library, ctypes.c_void_p(1))
            with self.subTest(error=error), self.assertRaises((RuntimeError, OSError)):
                engine.resident_vram_bytes()

    def test_residency_queries_zero_and_held_bytes_even_after_unload_flag(self):
        library = NativeLibrary()
        values = iter((0, 4096))
        def query(pipeline, out):
            self.assertEqual(pipeline.value, 1)
            out._obj.value = next(values)
            return 0
        library.quantfunc_resident_vram_bytes = query
        engine = qfe.QFEngineHandle(library, ctypes.c_void_p(1))
        self.assertEqual(engine.resident_vram_bytes(), 0)
        engine.unloaded = True
        self.assertEqual(engine.resident_vram_bytes(), 4096)

    def test_unmaterialized_residency_is_a_known_zero_without_native_query(self):
        engine = qfe.QFEngineHandle(NativeLibrary(), None)
        self.assertEqual(engine.resident_vram_bytes(), 0)

    def test_native_refusal_propagates_without_changing_residency_state(self):
        engine = qfe.QFEngineHandle(NativeLibrary(status=1), ctypes.c_void_p(1), footprint_bytes=1024)
        with self.assertRaisesRegex(RuntimeError, "pipeline busy"):
            engine.unload_vram()
        self.assertFalse(engine.unloaded)

    def test_bridge_failure_is_not_a_successful_zero_byte_unload(self):
        engine = qfe.QFEngineHandle(NativeLibrary(error=OSError("ABI call failed")), ctypes.c_void_p(1))
        with self.assertRaisesRegex(OSError, "ABI call failed"):
            engine.unload_vram()
        self.assertFalse(engine.unloaded)

    def test_session_that_cannot_end_prevents_physical_unload(self):
        library = NativeLibrary()
        library.end_status = 1
        engine = qfe.QFEngineHandle(library, ctypes.c_void_p(1))
        engine.current_session = ctypes.c_void_p(2)
        with self.assertRaisesRegex(RuntimeError, "session"):
            engine.unload_vram()
        self.assertIsNotNone(engine.current_session)
        self.assertFalse(engine.unloaded)
        self.assertEqual(library.unload_calls, 0)

    def test_unmaterialized_handle_needs_no_native_call(self):
        library = NativeLibrary()
        engine = qfe.QFEngineHandle(library, None)
        self.assertEqual(engine.unload_vram(), 0)
        self.assertEqual(library.unload_calls, 0)

    def test_success_is_idempotent_and_ends_the_session_first(self):
        library = NativeLibrary()
        engine = qfe.QFEngineHandle(library, ctypes.c_void_p(1), footprint_bytes=1024)
        engine.current_session = ctypes.c_void_p(2)
        self.assertEqual(engine.unload_vram(), 384)
        self.assertTrue(engine.unloaded)
        self.assertIsNone(engine.current_session)
        self.assertEqual(engine.unload_vram(), 0)
        self.assertEqual(library.unload_calls, 2)
        self.assertEqual(library.calls, ["end", "unload", "unload"])

    def test_full_release_after_previous_unload_still_observes_native_result(self):
        library = NativeLibrary()
        engine = qfe.QFEngineHandle(library, ctypes.c_void_p(1), footprint_bytes=1024)
        engine.unloaded = True
        self.assertEqual(engine.unload_vram(), 384)

    def test_status_only_library_cannot_claim_released_bytes(self):
        library = type("OldLibrary", (), {"quantfunc_unload_sync": lambda self, p: 0})()
        engine = qfe.QFEngineHandle(library, ctypes.c_void_p(1), footprint_bytes=1024)
        with self.assertRaisesRegex(RuntimeError, "quantfunc_unload_sync_ex"):
            engine.unload_vram()
        self.assertFalse(engine.unloaded)

    def test_partial_failure_is_not_zero_or_an_implicit_full_release(self):
        for error in (None, OSError("partial ABI failed")):
            library = NativeLibrary()
            def partial(pipeline, requested, out):
                if error:
                    raise error
                return 1
            library.quantfunc_partial_unload = partial
            engine = qfe.QFEngineHandle(library, ctypes.c_void_p(1))
            with self.subTest(error=error), self.assertRaises((RuntimeError, OSError)):
                engine.partial_unload_vram(128)
            self.assertEqual(library.unload_calls, 0)

    def test_partial_preserves_zero_and_rejects_negative_native_result(self):
        library = NativeLibrary()
        values = iter((0, -1))
        def partial(pipeline, requested, out):
            out._obj.value = next(values)
            return 0
        library.quantfunc_partial_unload = partial
        engine = qfe.QFEngineHandle(library, ctypes.c_void_p(1))
        self.assertEqual(engine.partial_unload_vram(128), 0)
        with self.assertRaisesRegex(RuntimeError, "negative"):
            engine.partial_unload_vram(128)


if __name__ == "__main__":
    unittest.main()
