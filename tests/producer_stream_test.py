#!/usr/bin/env python3
"""Contract tests for producer_stream passthrough and zero host stream synchronize() calls (U-C08).

Verifies:
  1. _resolve_producer_stream_ptr maps default stream 0 -> 1 (cudaStreamLegacy), preserves non-zero
     stream handles, falls back cleanly on None/CPU/negative/bool devices, and returns 0 without CUDA.
  2. _open_native_session, QFImageSessionModel._apply_model (including QFQwenImage21Model._denoise_group
     QFStepRefsParams), and QFAVSessionModel._execute_av_group_step pass producer_stream through and
     perform 0 stream.synchronize() calls.
  3. AST-extracted methods running in isolated namespaces without _resolve_producer_stream_ptr fall back
     to 0 without NameError or AttributeError.
"""
import ast
import ctypes
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types
import unittest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PKG_NAME = "qf_native_ps_test"

_COMFY_ROOT = os.environ.get("COMFY_ROOT")
if not _COMFY_ROOT or not (Path(_COMFY_ROOT) / "comfy/model_management.py").is_file():
    print("[SKIP] producer_stream_test: set COMFY_ROOT to the tested host")
    raise SystemExit(77)
if _COMFY_ROOT not in sys.path:
    sys.path.insert(0, _COMFY_ROOT)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import torch  # noqa: E402


def _load_pkg():
    if PKG_NAME in sys.modules:
        return (
            sys.modules[f"{PKG_NAME}.qf_engine"],
            sys.modules[f"{PKG_NAME}.qf_modelpatcher"],
            sys.modules[f"{PKG_NAME}.qf_qwenimage21_modelpatcher"],
        )
    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [str(PLUGIN_DIR)]
    sys.modules[PKG_NAME] = pkg
    loaded = []
    for mod_name in ("qf_capi", "qf_engine", "qf_sol", "qf_modelpatcher", "qf_qwenimage21_modelpatcher"):
        spec = importlib.util.spec_from_file_location(f"{PKG_NAME}.{mod_name}", PLUGIN_DIR / f"{mod_name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        loaded.append(mod)
    return loaded[1], loaded[3], loaded[4]


qfe, qfm, qf_qi = _load_pkg()


class _FakeStream:
    def __init__(self, ptr, sync_log):
        self.cuda_stream = ptr
        self._sync_log = sync_log

    def synchronize(self):
        self._sync_log.append(self.cuda_stream)


class _RecordingLib:
    def __init__(self):
        self.calls = []

    def qf_last_error(self):
        return b""

    def qf_xfm_session_begin(self, _xfm, bpx_p, sess_p):
        self.calls.append(("begin", int(bpx_p._obj.producer_stream)))
        sess_p._obj.value = 0x1010
        return qfe.QF_OK

    def qf_xfm_step(self, _sess, p_p):
        self.calls.append(("step", int(p_p._obj.producer_stream)))
        return qfe.QF_OK

    def qf_xfm_step_refs(self, _sess, rp_p):
        self.calls.append(("step_refs", int(rp_p._obj.base.producer_stream), int(rp_p._obj.num_refs)))
        return qfe.QF_OK

    def qf_xfm_step_av(self, _sess, mp_p):
        self.calls.append(("step_av", int(mp_p._obj.base.producer_stream)))
        return qfe.QF_OK


class _FakeEngine:
    def __init__(self, device_id=0):
        self.lib = _RecordingLib()
        self.xfm = ctypes.c_void_p(0x9001)
        self.current_session = ctypes.c_void_p(0x1010)
        self.device_id = device_id
        self.step_count = 0
        self.sampler_step_count = 0

    def end_session_if_open(self):
        return (False, True)


class _TestImgModel(qfm.QFImageSessionModel):
    _TAG = "IMG"
    _NO_COND_HINT = "missing"
    _REFUSED_HINT = "refused"

    def __init__(self, device, device_id=0):
        self.device = device
        self._qf = _FakeEngine(device_id=device_id)
        self._qf_needs_begin = False
        self._ctx_key_assigner = qfm._CtxKeyAssigner()
        self._num_steps = 4
        self._step_i = 0
        self._sess_denoise = 0
        self._out = None
        self.model_sampling = types.SimpleNamespace(calculate_denoised=lambda s, o, x: o)


class _TestQi21Model(qf_qi.QFQwenImage21Model):
    def __init__(self, device, device_id=0):
        self.device = device
        self._qf = _FakeEngine(device_id=device_id)
        self._qf_needs_begin = False
        self._ctx_key_assigner = qfm._CtxKeyAssigner()
        self._num_steps = 4
        self._step_i = 0
        self._sess_denoise = 0
        self._out = None
        self.model_sampling = types.SimpleNamespace(calculate_denoised=lambda s, o, x: o)


class _TestAVModel(qfm.QFAVSessionModel):
    _TAG = "AV"

    def __init__(self, device, device_id=0):
        self.device = device
        self._qf = _FakeEngine(device_id=device_id)
        self._ctx_key_assigner = qfm._CtxKeyAssigner()
        self._num_steps = 4
        self._step_i = 0
        self._sess_denoise = 0


class ProducerStreamTests(unittest.TestCase):
    def setUp(self):
        self._orig_avail = torch.cuda.is_available
        self._orig_cur = torch.cuda.current_stream

    def tearDown(self):
        torch.cuda.is_available = self._orig_avail
        torch.cuda.current_stream = self._orig_cur

    def test_resolve_producer_stream_ptr_branches(self):
        syncs = []
        queried = []

        # 1) CUDA unavailable -> 0 and never queries current_stream
        torch.cuda.is_available = lambda: False
        torch.cuda.current_stream = lambda *a: (_ for _ in ()).throw(AssertionError("should not query"))
        self.assertEqual(qfm._resolve_producer_stream_ptr(torch.device("cuda:0")), 0)
        self.assertEqual(qfm._resolve_producer_stream_ptr(None), 0)

        # 2) Default stream (0) -> legacy sentinel 1
        torch.cuda.is_available = lambda: True

        def fake_cur_zero(*args):
            queried.append(args)
            return _FakeStream(0, syncs)

        torch.cuda.current_stream = fake_cur_zero
        self.assertEqual(qfm._resolve_producer_stream_ptr(torch.device("cuda:1")), 1)
        self.assertEqual(qfm._resolve_producer_stream_ptr(2), 1)
        self.assertEqual(qfm._resolve_producer_stream_ptr(None), 1)
        self.assertEqual(qfm._resolve_producer_stream_ptr(torch.device("cpu")), 1)
        self.assertEqual(qfm._resolve_producer_stream_ptr(False), 1)
        self.assertEqual(qfm._resolve_producer_stream_ptr(-1), 1)
        self.assertEqual(queried, [(torch.device("cuda:1"),), (2,), (), (), (), ()])
        self.assertEqual(syncs, [])

        # 3) Non-zero stream pointer preserved verbatim
        torch.cuda.current_stream = lambda *args: _FakeStream(0xCAFEBABE, syncs)
        self.assertEqual(qfm._resolve_producer_stream_ptr(torch.device("cuda:0")), 0xCAFEBABE)
        self.assertEqual(syncs, [])

    def test_begin_step_refs_and_av_pass_producer_stream_with_zero_sync(self):
        x = torch.zeros(1, 4, 2, 2, dtype=torch.bfloat16)
        ctx = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
        ref = torch.zeros(1, 4, 2, 2, dtype=torch.bfloat16)
        vid = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
        vout = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
        aud = torch.zeros(1, 2, 4, dtype=torch.float32)
        aout = torch.zeros(1, 2, 4, dtype=torch.float32)

        for stream_ptr, expected_ps in ((0, 1), (0xDEAD1234, 0xDEAD1234)):
            with self.subTest(stream_ptr=stream_ptr):
                syncs = []
                torch.cuda.is_available = lambda: True
                torch.cuda.current_stream = lambda *args: _FakeStream(stream_ptr, syncs)

                img = _TestImgModel(torch.device("cuda:0"))
                img._open_native_session("IMG", (1, 2, 4), qfe.QF_DTYPE_BF16, 64, 64, {}, skip_end_if_open=True)
                img._apply_model(x, torch.tensor([1.0]), c_crossattn=ctx)
                self.assertEqual(img._qf.lib.calls, [("begin", expected_ps), ("step", expected_ps)])

                qi = _TestQi21Model(torch.device("cuda:0"))
                qi._apply_model(x, torch.tensor([1.0]), c_crossattn=ctx, ref_latents=[ref])
                self.assertEqual(qi._qf.lib.calls, [("step_refs", expected_ps, 1)])

                av = _TestAVModel(torch.device("cuda:0"))
                av._open_native_session("AV", (1, 2, 4), qfe.QF_DTYPE_BF16, 64, 64, {}, skip_end_if_open=True)
                av._execute_av_group_step(
                    vid_in=vid, vid_dims=(1, 2, 4), vid_out=vout,
                    aud_in=aud, aud_dims=(1, 2, 4), aud_out=aout,
                    vi=ctx, sig_i=1.0, step_index=0, ctx_key=1,
                    audio_scale=1.0, ref_active=False, group_i=0,
                )
                self.assertEqual(av._qf.lib.calls, [("begin", expected_ps), ("step_av", expected_ps)])
                self.assertEqual(syncs, [], "must never call stream.synchronize() on host")

    def test_cuda_unavailable_passes_zero_producer_stream(self):
        torch.cuda.is_available = lambda: False
        torch.cuda.current_stream = lambda *args: (_ for _ in ()).throw(AssertionError("must not query stream"))

        x = torch.zeros(1, 4, 2, 2, dtype=torch.bfloat16)
        ctx = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
        vid = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
        vout = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
        aud = torch.zeros(1, 2, 4, dtype=torch.float32)
        aout = torch.zeros(1, 2, 4, dtype=torch.float32)

        img = _TestImgModel(torch.device("cpu"))
        img._open_native_session("IMG", (1, 2, 4), qfe.QF_DTYPE_BF16, 64, 64, {}, skip_end_if_open=True)
        img._apply_model(x, torch.tensor([1.0]), c_crossattn=ctx)
        self.assertEqual(img._qf.lib.calls, [("begin", 0), ("step", 0)])

        av = _TestAVModel(torch.device("cpu"))
        av._execute_av_group_step(
            vid_in=vid, vid_dims=(1, 2, 4), vid_out=vout,
            aud_in=aud, aud_dims=(1, 2, 4), aud_out=aout,
            vi=ctx, sig_i=1.0, step_index=0, ctx_key=1,
            audio_scale=1.0, ref_active=False, group_i=0,
        )
        self.assertEqual(av._qf.lib.calls, [("step_av", 0)])

    def test_ast_extracted_methods_without_helper_in_globals_do_not_raise(self):
        """AST-extracted methods must degrade to producer_stream=0 without NameError/AttributeError."""
        src = (PLUGIN_DIR / "qf_modelpatcher.py").read_text(encoding="utf-8")
        tree = ast.parse(src)

        def _extract_method(cls_name, meth_name):
            cls_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
            meth_node = next(m for m in cls_node.body if isinstance(m, ast.FunctionDef) and m.name == meth_name)
            return ast.Module(body=[meth_node], type_ignores=[])

        ns = {
            "ctypes": ctypes,
            "json": json,
            "torch": torch,
            "qfe": qfe,
            "_qf_dtype": qfm._qf_dtype,
            "_interrupt_poll_end_session_on_raise": lambda _eng: None,
            "_BUSY_BEGIN_RETRY_S": 0.0,
        }
        for cls_name, meth_name in (
            ("QFSessionModelMixin", "_open_native_session"),
            ("QFSessionModelMixin", "_call_denoise_step"),
            ("QFSessionModelMixin", "_call_denoise_step_av"),
            ("QFImageSessionModel", "_apply_model"),
            ("QFAVSessionModel", "_execute_av_group_step"),
        ):
            exec(compile(ast.fix_missing_locations(_extract_method(cls_name, meth_name)), "<ast>", "exec"), ns)

        self.assertNotIn("_resolve_producer_stream_ptr", ns)

        class _Iso:
            _TAG = "ISO"
            _NO_COND_HINT = "missing"
            _REFUSED_HINT = "refused"

            def __init__(self):
                self.device = torch.device("cpu")
                self._qf = _FakeEngine()
                self._qf_needs_begin = False
                self._ctx_key_assigner = qfm._CtxKeyAssigner()
                self._num_steps = 4
                self._step_i = 0
                self._sess_denoise = 0
                self._out = None
                self.model_sampling = types.SimpleNamespace(calculate_denoised=lambda s, o, x: o)

            _ensure_scratch = qfm.QFSessionModelMixin._ensure_scratch
            _sigma_step_index = qfm.QFSessionModelMixin._sigma_step_index
            _denoise_group = qfm.QFImageSessionModel._denoise_group

        _Iso._open_native_session = ns["_open_native_session"]
        _Iso._call_denoise_step = ns["_call_denoise_step"]
        _Iso._call_denoise_step_av = ns["_call_denoise_step_av"]
        _Iso._apply_model = ns["_apply_model"]
        _Iso._execute_av_group_step = ns["_execute_av_group_step"]

        iso = _Iso()
        x = torch.zeros(1, 4, 2, 2, dtype=torch.bfloat16)
        ctx = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
        vid = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
        vout = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
        aud = torch.zeros(1, 2, 4, dtype=torch.float32)
        aout = torch.zeros(1, 2, 4, dtype=torch.float32)

        iso._open_native_session("ISO", (1, 2, 4), qfe.QF_DTYPE_BF16, 64, 64, {}, skip_end_if_open=True)
        iso._apply_model(x, torch.tensor([1.0]), c_crossattn=ctx)
        iso._execute_av_group_step(
            vid_in=vid, vid_dims=(1, 2, 4), vid_out=vout,
            aud_in=aud, aud_dims=(1, 2, 4), aud_out=aout,
            vi=ctx, sig_i=1.0, step_index=0, ctx_key=1,
            audio_scale=1.0, ref_active=False, group_i=0,
        )
        self.assertEqual(iso._qf.lib.calls, [("begin", 0), ("step", 0), ("step_av", 0)])

    def test_input_tensor_cuda_device_takes_priority_over_self_device(self):
        class _CudaStubTensor:
            def __init__(self, shape, dtype, device, ptr):
                self.shape = tuple(shape)
                self.dtype = dtype
                self.device = device
                self._ptr = ptr

            def dim(self):
                return len(self.shape)

            def numel(self):
                return math.prod(self.shape)

            def element_size(self):
                return 2 if self.dtype in (torch.bfloat16, torch.float16) else 4

            def data_ptr(self):
                return self._ptr

            def __getitem__(self, _idx):
                return self

            def contiguous(self):
                return self

            def float(self):
                return self

        for model_dev in (torch.device("cuda:0"), torch.device("cpu")):
            with self.subTest(model_dev=str(model_dev)):
                syncs = []
                queried = []
                torch.cuda.is_available = lambda: True

                def fake_cur(dev=None):
                    queried.append(dev)
                    ptr = 0xBEEF0001 if dev == torch.device("cuda:1") else 0xBEEF0002
                    return _FakeStream(ptr, syncs)

                torch.cuda.current_stream = fake_cur

                img = _TestImgModel(model_dev, device_id=3)
                dev_img = torch.device("cuda:1")
                x_stub = _CudaStubTensor((1, 4, 2, 2), torch.bfloat16, dev_img, 0x3000)
                ctx_stub = _CudaStubTensor((1, 2, 4), torch.bfloat16, dev_img, 0x4000)
                img._out = _CudaStubTensor((1, 4, 2, 2), torch.bfloat16, dev_img, 0x5000)
                img._apply_model(x_stub, torch.tensor([1.0]), c_crossattn=ctx_stub)
                self.assertEqual(queried, [torch.device("cuda:1")])
                self.assertEqual(img._qf.lib.calls, [("step", 0xBEEF0001)])

                queried.clear()
                av = _TestAVModel(model_dev, device_id=3)
                vid_stub = _CudaStubTensor((1, 2, 4), torch.bfloat16, torch.device("cuda:2"), 0x6000)
                vout = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
                aud = torch.zeros(1, 2, 4, dtype=torch.float32)
                aout = torch.zeros(1, 2, 4, dtype=torch.float32)
                vi = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
                av._execute_av_group_step(
                    vid_in=vid_stub, vid_dims=(1, 2, 4), vid_out=vout,
                    aud_in=aud, aud_dims=(1, 2, 4), aud_out=aout,
                    vi=vi, sig_i=1.0, step_index=0, ctx_key=1,
                    audio_scale=1.0, ref_active=False, group_i=0,
                )
                self.assertEqual(queried, [torch.device("cuda:2")])
                self.assertEqual(av._qf.lib.calls, [("step_av", 0xBEEF0002)])
                self.assertEqual(syncs, [])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
