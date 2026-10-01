#!/usr/bin/env python3
"""Native-bridge regression: a refused or failed native query must not look like a successful zero.

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
        self.end_status = 0

    def quantfunc_last_error(self):
        return b"pipeline busy"

    def quantfunc_denoise_end(self, session):
        return self.end_status


class HostReclaimContract(unittest.TestCase):
    def test_demand_query_failure_is_not_successful_zero(self):
        for status in (1, 8):
            library = NativeLibrary()
            library.quantfunc_vram_need_bytes = lambda *args: status
            engine = qfe.QFEngineHandle(library, ctypes.c_void_p(1))
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, "pipeline busy"):
                engine.vram_need_bytes([1, 24, 37, 28, 48])

    def test_demand_ffi_failure_is_not_successful_zero(self):
        library = NativeLibrary()
        def demand(*args):
            raise OSError("demand ABI failed")
        library.quantfunc_vram_need_bytes = demand
        engine = qfe.QFEngineHandle(library, ctypes.c_void_p(1))
        with self.assertRaisesRegex(OSError, "demand ABI failed"):
            engine.vram_need_bytes([1, 16, 64, 64])

    def test_missing_demand_query_is_not_successful_zero(self):
        engine = qfe.QFEngineHandle(NativeLibrary(), ctypes.c_void_p(1))
        with self.assertRaisesRegex(RuntimeError, "quantfunc_vram_need_bytes"):
            engine.vram_need_bytes([1, 16, 64, 64])

    def test_successful_demand_preserves_native_zero_and_positive_count(self):
        library = NativeLibrary()
        values = iter((0, 987654321))
        def demand(pipeline, dims, ndim, out):
            self.assertEqual(pipeline.value, 1)
            self.assertEqual(list(dims), [1, 16, 64, 64])
            self.assertEqual(ndim, 4)
            out._obj.value = next(values)
            return 0
        library.quantfunc_vram_need_bytes = demand
        engine = qfe.QFEngineHandle(library, ctypes.c_void_p(1))
        self.assertEqual(engine.vram_need_bytes([1, 16, 64, 64]), 0)
        self.assertEqual(engine.vram_need_bytes([1, 16, 64, 64]), 987654321)

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

    def test_residency_queries_zero_and_held_bytes(self):
        library = NativeLibrary()
        values = iter((0, 4096))
        def query(pipeline, out):
            self.assertEqual(pipeline.value, 1)
            out._obj.value = next(values)
            return 0
        library.quantfunc_resident_vram_bytes = query
        engine = qfe.QFEngineHandle(library, ctypes.c_void_p(1))
        self.assertEqual(engine.resident_vram_bytes(), 0)
        self.assertEqual(engine.resident_vram_bytes(), 4096)

    def test_unmaterialized_residency_is_a_known_zero_without_native_query(self):
        engine = qfe.QFEngineHandle(NativeLibrary(), None)
        self.assertEqual(engine.resident_vram_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
