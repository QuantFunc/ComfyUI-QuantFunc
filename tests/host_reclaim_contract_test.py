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

    def quantfunc_last_error(self):
        return b"pipeline busy"

    def quantfunc_denoise_end(self, session):
        self.calls.append("end")
        return self.end_status

    def quantfunc_unload_sync(self, pipeline):
        self.calls.append("unload")
        self.unload_calls += 1
        if self.error:
            raise self.error
        return self.status


class HostReclaimContract(unittest.TestCase):
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
        engine.unload_vram()
        self.assertTrue(engine.unloaded)
        self.assertIsNone(engine.current_session)
        self.assertEqual(engine.unload_vram(), 0)
        self.assertEqual(library.unload_calls, 1)
        self.assertEqual(library.calls, ["end", "unload"])


if __name__ == "__main__":
    unittest.main()
