#!/usr/bin/env python3
"""CPU-only actual-method contracts for LTX-AV reference audio, masks, and frozen-audio restoration."""
import ast
import ctypes
import gc
import json
import math
import operator
from pathlib import Path
import types
import unittest
import weakref
from contextvars import ContextVar
import torch

ROOT = Path(__file__).resolve().parents[1]


def production_namespace(source=None, engine_source=None, shared_source=None):
    ns = dict(torch=torch, ctypes=ctypes, math=math, json=json, operator=operator, ContextVar=ContextVar,
              time=types.SimpleNamespace(sleep=lambda _: None))
    for filename, names in [("qf_engine.py", {"QFSessionBeginParams", "QFStepParams", "QFStepAvParams", "QFEngineHandle"}),
                            ("qf_modelpatcher.py", {"_CtxKeyAssigner", "_qf_dtype"})]:
        override = engine_source if filename == "qf_engine.py" else shared_source
        text = override if override is not None else (ROOT / filename).read_text(encoding="utf-8")
        tree = ast.parse(text)
        body = [n for n in tree.body if getattr(n, "name", None) in names]
        if filename == "qf_engine.py":
            assigns = [n for n in tree.body if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {
                    "_REFERENCE_AUDIO_PENDING_OWNERS", "QF_OK", "QF_UINT64_MAX",
                } for t in n.targets)]
            body = assigns + body
        if filename == "qf_modelpatcher.py":
            body += [n for n in tree.body if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {
                    "_QF_SAMPLING_GEOMETRY", "_OUTER_SAMPLE_LATENT_SHAPES_ARG",
                } for t in n.targets)]
            patcher = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QFModelPatcher")
            patcher.decorator_list = []
            patcher.bases = []
            patcher.body = [n for n in patcher.body if getattr(n, "name", None) == "_sampling_geometry"]
            body.append(patcher)
            mixin = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QFSessionModelMixin")
            mixin.decorator_list = []
            mixin.body = [n for n in mixin.body if getattr(n, "name", None) == "_call_denoise_step_av"]
            body.append(mixin)
            lazy = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QFLazyEngine")
            lazy.decorator_list = []
            lazy.body = [n for n in lazy.body if getattr(n, "name", None) in {
                "lib", "xfm", "current_session", "retain_reference_audio", "reference_audio_pending",
                "reference_audio_copy_completed", "end_session_if_open", "unload_vram", "unload_sync"}]
            body.append(lazy)
        exec(compile(ast.Module(body=body, type_ignores=[]), filename, "exec"), ns)
        if filename == "qf_engine.py":
            ns["qfe"] = types.SimpleNamespace(
                **{n: ns[n] for n in ["QFSessionBeginParams", "QFStepParams", "QFStepAvParams"]},
                QF_DTYPE_FP32=0, QF_DTYPE_FP16=1, QF_DTYPE_BF16=2,
                QF_OK=0, QF_UINT64_MAX=ns["QF_UINT64_MAX"],
                last_err=lambda lib: lib.error, info=lambda *a, **k: None)
    ns["_KNO_CTX_KEY"] = 0
    ns["last_err"] = lambda lib: lib.error
    ns["say"] = lambda *a, **k: None

    class Cond:
        def __init__(self, cond):
            self.cond = cond

    ns["comfy"] = types.SimpleNamespace(conds=types.SimpleNamespace(CONDConstant=Cond, CONDRegular=Cond))
    ns["qfmp"] = types.SimpleNamespace(refuse_all_zero_initial_latent=lambda *a: None)
    ns["_interrupt_poll_end_session_on_raise"] = lambda _: None
    ltx_text = source if source is not None else (ROOT / "qf_ltx_modelpatcher.py").read_text(encoding="utf-8")
    tree = ast.parse(ltx_text)
    consts = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {
            "_LTXAV_AUDIO_CH", "_LTXAV_AUDIO_MEL", "_LTXAV_AUDIO_PACK",
            "_REF_AUDIO_MAX_NUMEL", "_VIDEO_LATENT_NDIM", "_AUDIO_LATENT_NDIM", "_BUSY_BEGIN_RETRY_S",
        } for t in n.targets)]
    classes = []
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name in {"QFLTXModel", "QFLTXAVModel"}:
            n.decorator_list = []
            n.bases = [ast.Name(id="object" if n.name == "QFLTXModel" else "QFLTXModel", ctx=ast.Load())]
            n.body = [x for x in n.body if getattr(x, "name", None) != "__init__"]
            classes.append(n)
    exec(compile(ast.fix_missing_locations(ast.Module(body=consts + classes, type_ignores=[])), "actual-ltx", "exec"), ns)
    return ns


class FakeDLL:
    def __init__(self):
        self.status = 0
        self.error = ""
        self.calls = []
        self.copy_pending = False

        class ExplicitSyncOnly:
            calls = 0

            def __call__(self, _xfm, _req, _out):
                self.calls += 1
                raise RuntimeError("synchronous unload must be explicitly requested")

        self.qf_xfm_unload_vram = ExplicitSyncOnly()

    def qf_xfm_session_begin(self, xfm, ptr, out):
        p = ptr._obj
        self.calls.append({
            "options_utf8": p.options_json.decode(),
            "width": p.width,
            "height": p.height,
            "num_steps": p.num_steps,
            "max_context_dims": list(p.max_context_dims),
            "cond_dtype": p.cond_dtype,
            "struct_size": p.struct_size,
        })
        opts = json.loads(p.options_json)
        if opts.get("av_conds"):
            c = opts["av_conds"][0]
            dtype = {0: torch.float32, 1: torch.float16, 2: torch.bfloat16}[c["audio_latent_dtype"]]
            size = math.prod(c["audio_latent_dims"]) * torch.empty((), dtype=dtype).element_size()
            raw = bytearray(ctypes.string_at(int(c["audio_latent_ptr"], 16), size))
            self.upload_values = torch.frombuffer(raw, dtype=dtype).clone().reshape(c["audio_latent_dims"])
        if self.status == 0:
            self.copy_pending = False
            out._obj.value = 123
        return self.status


class Engine:
    def __init__(self):
        self.lib = FakeDLL()
        self.xfm = 1
        self.current_session = None
        self.close_ok = True
        self.step_count = self.sampler_step_count = 0

    def qf_xfm_session_end(self, session):
        return 0 if self.close_ok else 8


def model(ns, av=True):
    m = ns["QFLTXAVModel" if av else "QFLTXModel"]()
    m._qf = Engine()
    m._qf.lib.qf_xfm_session_end = m._qf.qf_xfm_session_end
    for name in ("retain_reference_audio", "reference_audio_pending", "reference_audio_copy_completed",
                 "end_session_if_open", "destroy", "unload_vram", "unload_sync"):
        setattr(m._qf, name, types.MethodType(getattr(ns["QFEngineHandle"], name), m._qf))
    m._ctx_key_assigner = ns["_CtxKeyAssigner"]()
    m._call_denoise_step_av = types.MethodType(ns["QFSessionModelMixin"]._call_denoise_step_av, m)
    m._ctx_unprocessed = True
    m._audio_rows = 101
    m._audio_pack = 128
    m._max_ctx_seq = 1024
    m._num_frames = 97
    m._num_steps = 8
    m._fps = 24.0
    m.latent_format = types.SimpleNamespace(spacial_downscale_ratio=32)
    m._out = m._out_audio = None
    m._step_i = m._sess_denoise = 0
    m._native_frame_rate = lambda kw: ns["comfy"].conds.CONDConstant(kw.get("frame_rate", 24))
    m._post_connector_seq = lambda n: n
    m.residency_opts = lambda: {}
    m._derive_geometry = lambda *a: None
    m._call_fps = lambda *a: None
    m._sigma_step_index = lambda *a: 0
    m.model_sampling = types.SimpleNamespace(calculate_denoised=lambda t, v, x: v)
    return m


class ReferenceAudioTests(unittest.TestCase):
    def setUp(self):
        self.ns = production_namespace()

    def test_body_activation_contract(self):
        m = model(self.ns)
        self.assertTrue(hasattr(self.ns["QFLTXAVModel"], "_reference_activation_dtype"),
                        "LTX body dtype contract resolver is missing")
        self.assertEqual(self.ns["QFLTXAVModel"]._reference_activation_dtype(torch.device("cpu")), torch.bfloat16)
        self.assertEqual(m._reference_activation_dtype(torch.device("cpu")), torch.bfloat16)

    def test_video_only_hook_and_av_emission(self):
        p = {"tokens": torch.ones(1, 3, 128)}
        av = model(self.ns)
        try:
            out = av.extra_conds(ref_audio=p, unprocessed_ltxav_embeds=True)
        except RuntimeError as exc:
            self.fail(f"dedicated AV ref_audio hook missing: {exc}")
        self.assertIn("ref_audio", out)
        self.assertIs(out["ref_audio"].cond, p)
        with self.assertRaisesRegex(RuntimeError, "AV"):
            model(self.ns, False).extra_conds(ref_audio=p)
        with self.assertRaises(RuntimeError):
            av.extra_conds(generated_keyframes=object())

    def test_schema_and_post_cast_finite(self):
        m = model(self.ns)
        count = 0
        for source in (torch.float16, torch.bfloat16, torch.float32):
            for act in (torch.float16, torch.bfloat16):
                for n in (1, 3, 4, 5):
                    for inference in (False, True):
                        for batch in (1, 2):
                            with self.subTest(source=source, act=act, n=n, inference=inference, batch=batch):
                                with torch.inference_mode(inference):
                                    p = {"tokens": torch.ones(1, n, 128, dtype=source)}
                                m = model(self.ns)
                                m._prepare_audio_reference(p, torch.device("cpu"), act, True)
                                self.assertEqual(m._audio_reference[-1].dtype, act)
                                calls = []
                                m._reference_activation_dtype = lambda _, act=act: act
                                m._call_denoise_step_av = lambda mp, *_: calls.append((int(mp.audio_reference_active), list(mp.audio_dims)))
                                m.latent_shapes = [(batch, 128, 1, 1, 1), (batch, 8, 4, 16)]
                                m._apply_model(torch.ones(batch, 1, 640), torch.ones(batch),
                                               c_crossattn=torch.ones(batch, 3, 6144), ref_audio=p,
                                               transformer_options={"cond_or_uncond": list(range(batch))})
                                self.assertEqual(len(calls), batch)
                                opts = json.loads(m._qf.lib.calls[-1]["options_utf8"])
                                self.assertIn("av_conds", opts)
                                carrier = opts["av_conds"][0]
                                self.assertEqual(carrier["audio_latent_dims"], [1, n, 128])
                                self.assertEqual(carrier["audio_latent_dtype"], self.ns["_qf_dtype"](act))
                                self.assertEqual(carrier["audio_latent_ptr"], hex(m._audio_reference[-1].data_ptr()))
                                self.assertTrue(all(v[0] == 1 for v in calls))
                                count += 1
        print("SCHEMA_MATRIX executed=", count)
        bad = [{}, {"tokens": torch.ones(1, 2, 128), "extra": 1}, {"tokens": None},
               *[{"tokens": t} for t in [
                   torch.ones(2, 2, 128), torch.ones(1, 0, 128), torch.ones(1, 2, 127),
                   torch.ones(2, 128), torch.ones(1, 2, 128, dtype=torch.int32),
                   torch.full((1, 2, 128), float("nan")), torch.full((1, 2, 128), float("inf"))]]]
        for p in bad:
            with self.subTest(bad=str(p)[:60]), self.assertRaises(RuntimeError):
                model(self.ns)._prepare_audio_reference(p, torch.device("cpu"), torch.float16, True)
        for act, value in [(torch.float16, 1e10), (torch.bfloat16, torch.finfo(torch.float32).max)]:
            with self.subTest(act=act), self.assertRaisesRegex(RuntimeError, "finite"):
                model(self.ns)._prepare_audio_reference({"tokens": torch.full((1, 2, 128), value)},
                                                        torch.device("cpu"), act, True)

    def test_actual_body_overflow_precedes_native_borrow(self):
        m = model(self.ns)
        m.latent_shapes = [(1, 128, 1, 1, 1), (1, 8, 4, 16)]
        with self.assertRaisesRegex(RuntimeError, "finite"):
            m._apply_model(torch.ones(1, 1, 640), torch.ones(1), c_crossattn=torch.ones(1, 3, 6144),
                           ref_audio={"tokens": torch.full((1, 3, 128), torch.finfo(torch.float32).max)},
                           transformer_options={"cond_or_uncond": [0]})
        self.assertEqual(m._qf.lib.calls, [])
        self.assertFalse(m._qf.reference_audio_pending())
        self.assertIsNone(getattr(m._qf, "_reference_audio_owner", None))

    def test_no_ref_and_mask_transport_fixture(self):
        frozen = json.loads((ROOT / "tests/fixtures/ltx_ref_legacy_transport.json").read_text(encoding="utf-8"))
        for f in frozen["fixtures"]:
            m = model(self.ns)
            opts = json.loads(f["options_utf8"])
            m._pending_frame_t_scale = opts.get("frame_t_scales")
            m._pending_audio_frame_t_scale = opts.get("audio_frame_t_scales")
            m._begin(torch.ones(1, 128, 13, 9, 16), torch.ones(1, 1024, 6144, dtype=torch.bfloat16))
            self.assertEqual(m._qf.lib.calls[-1], {k: v for k, v in f.items() if k != "name"})
        self.assertEqual(ctypes.sizeof(self.ns["QFSessionBeginParams"]), 56)
        self.assertEqual(ctypes.sizeof(self.ns["QFStepParams"]), 112)
        self.assertEqual(ctypes.sizeof(self.ns["QFStepAvParams"]), 176)

    def test_snapshot_normal_inference_and_alias(self):
        for inference in (False, True):
            for alias in ("normal", "data", "view"):
                with self.subTest(inference=inference, alias=alias):
                    with torch.inference_mode(inference):
                        t = torch.ones(1, 3, 128)
                    p = {"tokens": t}
                    m = model(self.ns)
                    m._prepare_audio_reference(p, torch.device("cpu"), torch.float16, True)
                    m._audio_reference_bound = True
                    m._prepare_audio_reference(p, torch.device("cpu"), torch.float16, False)
                    m._prepare_audio_reference(None, torch.device("cpu"), torch.float16, False)
                    m._prepare_audio_reference(p, torch.device("cpu"), torch.float16, False)
                    with torch.inference_mode(inference):
                        target = t if alias == "normal" else (t.data if alias == "data" else t.view_as(t))
                        target.add_(1)
                    self.assertTrue(torch.equal(m._audio_reference[-1], torch.ones(1, 3, 128, dtype=torch.float16)))
                    with self.assertRaisesRegex(RuntimeError, "changed"):
                        m._prepare_audio_reference(p, torch.device("cpu"), torch.float16, False)
        m = model(self.ns)
        p = {"tokens": torch.ones(1, 3, 128)}
        m._prepare_audio_reference(p, torch.device("cpu"), torch.float16, True)
        m._audio_reference_bound = True
        with self.assertRaisesRegex(RuntimeError, "changed"):
            m._prepare_audio_reference({"tokens": p["tokens"].clone()}, torch.device("cpu"), torch.float16, False)
        with self.assertRaisesRegex(RuntimeError, "new session"):
            model(self.ns)._prepare_audio_reference(p, torch.device("cpu"), torch.float16, False)

    def test_snapshot_is_only_upload_source(self):
        m = model(self.ns)
        t = torch.ones(1, 3, 128)
        original = torch.Tensor.to

        def mutate(source, *args, **kwargs):
            if kwargs.get("dtype") == torch.float16:
                t.data.fill_(9)
            return original(source, *args, **kwargs)

        from unittest.mock import patch
        with patch.object(torch.Tensor, "to", mutate):
            m._prepare_audio_reference({"tokens": t}, torch.device("cpu"), torch.float16, True)
        self.assertTrue(torch.equal(m._audio_reference[-1], torch.ones(1, 3, 128, dtype=torch.float16)))
        m._begin(torch.ones(1, 128, 13, 9, 16), torch.ones(1, 3, 6144, dtype=torch.bfloat16))
        self.assertTrue(torch.equal(m._qf.lib.upload_values, torch.ones(1, 3, 128, dtype=torch.float16)))
        m._audio_reference_bound = True
        with self.assertRaisesRegex(RuntimeError, "changed"):
            m._prepare_audio_reference({"tokens": t}, torch.device("cpu"), torch.float16, False)

    def test_original_metadata_captured_before_normalization(self):
        from unittest.mock import patch
        m = model(self.ns)
        t = torch.ones(1, 3, 128, dtype=torch.float32)
        original = torch.Tensor.to

        def mutate(source, *args, **kwargs):
            if kwargs.get("dtype") == torch.float16:
                t.data = original(t.data, dtype=torch.float16)
            return original(source, *args, **kwargs)

        with patch.object(torch.Tensor, "to", mutate):
            m._prepare_audio_reference({"tokens": t}, torch.device("cpu"), torch.float16, True)
        m._audio_reference_bound = True
        with self.assertRaisesRegex(RuntimeError, "changed"):
            m._prepare_audio_reference({"tokens": t}, torch.device("cpu"), torch.float16, False)
        for change in ("shape", "dtype"):
            m = model(self.ns)
            t = torch.ones(1, 3, 128)
            m._prepare_audio_reference({"tokens": t}, torch.device("cpu"), torch.float16, True)
            m._audio_reference_bound = True
            t.data = torch.ones(1, 2, 128) if change == "shape" else t.data.to(torch.bfloat16)
            with self.assertRaisesRegex(RuntimeError, "changed"):
                m._prepare_audio_reference({"tokens": t}, torch.device("cpu"), torch.float16, False)

    def test_grouped_av_loop_uuid_and_selector(self):
        m = model(self.ns)
        calls = []
        actual = self.ns["_CtxKeyAssigner"]

        class Keys(actual):
            def __init__(self):
                super().__init__()
                self.arguments = []

            def key(self, uuid):
                self.arguments.append(uuid)
                return super().key(uuid)

        m._ctx_key_assigner = Keys()
        m._call_denoise_step_av = lambda mp, *a: calls.append((mp.base.cfg_context_key, int(mp.audio_reference_active), list(mp.audio_dims)))
        m.latent_shapes = [(2, 128, 1, 1, 1), (2, 8, 4, 16)]
        x = torch.ones(2, 1, 128 + 8 * 4 * 16)
        ctx = torch.ones(2, 3, 6144)
        p = {"tokens": torch.ones(1, 3, 128)}

        def run(uuids, ref=p):
            m._apply_model(x, torch.tensor([1., 1.]), c_crossattn=ctx,
                           transformer_options={"cond_or_uncond": [0, 1], "uuids": uuids}, ref_audio=ref)

        run(["a", "b"])
        first = calls[-2:]
        self.assertEqual([v[0] for v in first], [1, 2])
        self.assertTrue(all(v[1] == 1 for v in first))
        self.assertTrue(all(v[2] == [1, 4, 128, 0] for v in first))
        upload = m._audio_reference[-1]
        self.assertEqual(len(m._qf.lib.calls), 1)
        run(["b", "a"], None)
        self.assertEqual([v[0] for v in calls[-2:]], [2, 1])
        self.assertTrue(all(v[1] == 0 for v in calls[-2:]))
        run([None, "a"])
        self.assertEqual([v[0] for v in calls[-2:]], [0, 1])
        self.assertIs(m._audio_reference[-1], upload)
        self.assertEqual(m._ctx_key_assigner.arguments[-2:], [None, "a"])
        m.latent_shapes = [(1, 128, 1, 1, 1), (1, 8, 4, 16)]
        for uuid, role, key in [("b", 1, 2), ("a", 0, 1)]:
            m._apply_model(x[:1], torch.tensor([1.]), c_crossattn=ctx[:1], ref_audio=p,
                           transformer_options={"cond_or_uncond": [role], "uuids": [uuid]})
            self.assertEqual(calls[-1][0], key)
        m.latent_shapes = [(2, 128, 1, 1, 1), (2, 8, 4, 16)]
        m._apply_model(x, torch.ones(2), c_crossattn=ctx, ref_audio=p,
                       transformer_options={"cond_or_uncond": [1, 0], "uuids": ["b", "a"]})
        self.assertEqual([v[0] for v in calls[-2:]], [2, 1])
        m2 = model(self.ns)
        m2._call_denoise_step_av = m._call_denoise_step_av
        m2.latent_shapes = m.latent_shapes
        m2._apply_model(x, torch.tensor([1., 1.]), c_crossattn=ctx, transformer_options={"cond_or_uncond": [0, 1]})
        self.assertTrue(all(v[1] == 0 for v in calls[-2:]))

    def test_step_av_error_propagates_without_fallback(self):
        m = model(self.ns)
        calls = []
        m._qf.current_session = ctypes.c_void_p(123)
        m._qf.lib.error = "step_av refused"

        def fail(*args):
            calls.append(args)
            return 1

        m._qf.lib.qf_xfm_step_av = fail
        mp = self.ns["QFStepAvParams"]()
        mp.audio_reference_active = 1
        with self.assertRaisesRegex(RuntimeError, "step_av refused"):
            m._call_denoise_step_av(mp, "LTX-AV step")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(m._qf.lib.calls), 0)
        self.assertEqual(mp.audio_reference_active, 1)

    def test_pending_borrow_holds_owner_after_all_python_callers_drop(self):
        m = model(self.ns)
        p = {"tokens": torch.ones(1, 3, 128)}
        m._prepare_audio_reference(p, torch.device("cpu"), torch.bfloat16, True)
        owner = m._qf
        owner.lib.status = 8
        owner.lib.error = "begin copy pending"
        with self.assertRaises(RuntimeError):
            m._begin(torch.ones(1, 128, 13, 9, 16), torch.ones(1, 3, 6144, dtype=torch.bfloat16))
        wo = weakref.ref(owner)
        wu = weakref.ref(m._audio_reference[-1])
        del m, owner
        gc.collect()
        self.assertIsNotNone(wo(), "pending borrow must strongly retain actual engine owner")
        self.assertIsNotNone(wu(), "pending borrow must retain upload after model AND owner GC")
        owner = wo()
        owner.lib.qf_xfm_destroy = lambda *_: self.fail("pending destroy reached void native")
        with self.assertRaisesRegex(RuntimeError, "reference.*pending"):
            owner.destroy()
        self.assertEqual(owner.xfm, 1)
        self.assertIsNotNone(wu())
        owner.current_session = ctypes.c_void_p(123)
        owner.close_ok = True
        owner.end_session_if_open()
        del owner
        gc.collect()
        self.assertIsNone(wo())
        self.assertIsNone(wu())

    def test_explicit_sync_cleanup_status_only_and_next_session(self):
        m = model(self.ns)
        p = {"tokens": torch.ones(1, 3, 128)}
        m._prepare_audio_reference(p, torch.device("cpu"), torch.bfloat16, True)
        owner = m._qf
        owner.lib.status = 8
        owner.lib.error = "busy"
        with self.assertRaises(RuntimeError):
            m._begin(torch.ones(1, 128, 13, 9, 16), torch.ones(1, 3, 6144, dtype=torch.bfloat16))
        self.assertTrue(hasattr(owner, "unload_sync"))
        self.assertEqual(owner.lib.qf_xfm_unload_vram.calls, 0, "failed begin must not auto-sync")
        wu = weakref.ref(m._audio_reference[-1])

        class SyncCall:
            status = 8
            raised = False
            calls = 0

            def __call__(self, _xfm, req, out):
                self.calls += 1
                if self.raised:
                    raise RuntimeError("sync transport failure")
                out._obj.value = 0
                return self.status

        sync = SyncCall()
        owner.lib.qf_xfm_unload_vram = sync
        lazy = self.ns["QFLazyEngine"]()
        lazy._real = owner
        lazy.ensure = lambda: owner
        m._qf = lazy
        self.assertEqual(sync.calls, 0)
        for status, raised in [(8, False), (1, False), (0, True)]:
            sync.status = status
            sync.raised = raised
            with self.assertRaises(RuntimeError):
                lazy.unload_sync()
            self.assertTrue(owner.reference_audio_pending())
            self.assertIsNotNone(wu())
            with self.assertRaises(RuntimeError):
                m._prepare_audio_reference(None, torch.device("cpu"), torch.bfloat16, True)
        sync.status = 0
        sync.raised = False
        self.assertEqual(lazy.unload_sync(), 0)
        self.assertFalse(owner.reference_audio_pending())
        self.assertIsNone(owner._reference_audio_owner)
        m._prepare_audio_reference(None, torch.device("cpu"), torch.bfloat16, True)
        gc.collect()
        self.assertIsNone(wu())
        owner.lib.status = 0
        m._begin(torch.ones(1, 128, 13, 9, 16), torch.ones(1, 3, 6144, dtype=torch.bfloat16))
        self.assertNotIn("av_conds", json.loads(owner.lib.calls[-1]["options_utf8"]))
        m._prepare_audio_reference(p, torch.device("cpu"), torch.bfloat16, True)
        m._begin(torch.ones(1, 128, 13, 9, 16), torch.ones(1, 3, 6144, dtype=torch.bfloat16))
        self.assertTrue(m._audio_reference_bound)
        self.assertFalse(owner.reference_audio_pending())

    def test_sync_failure_keeps_gc_hold_and_ok_releases_it(self):
        m = model(self.ns)
        m._prepare_audio_reference({"tokens": torch.ones(1, 3, 128)}, torch.device("cpu"), torch.bfloat16, True)
        owner = m._qf
        owner.lib.status = 8
        owner.lib.error = "busy"
        with self.assertRaises(RuntimeError):
            m._begin(torch.ones(1, 128, 13, 9, 16), torch.ones(1, 3, 6144, dtype=torch.bfloat16))
        wo = weakref.ref(owner)
        wu = weakref.ref(m._audio_reference[-1])

        class Sync:
            status = 8
            raised = False

            def __call__(self, _xfm, _req, out):
                if self.raised:
                    raise RuntimeError("sync raised")
                out._obj.value = 0
                return self.status

        sync = Sync()
        owner.lib.qf_xfm_unload_vram = sync
        del m, owner
        gc.collect()
        self.assertIsNotNone(wo())
        for status, raised in [(8, False), (1, False), (0, True)]:
            owner = wo()
            sync.status = status
            sync.raised = raised
            with self.assertRaises(RuntimeError):
                owner.unload_sync()
            del owner
            gc.collect()
            self.assertIsNotNone(wo(), "non-OK sync must retain engine despite all caller GC")
            self.assertIsNotNone(wu())
        owner = wo()
        sync.status = 0
        sync.raised = False
        self.assertEqual(owner.unload_sync(), 0)
        del owner
        gc.collect()
        self.assertIsNone(wo())
        self.assertIsNone(wu())

    def test_actual_lazy_wrapper_pins_cached_owner(self):
        m = model(self.ns)
        real = m._qf
        lazy = self.ns["QFLazyEngine"]()
        lazy._real = real
        lazy.ensure = lambda: real
        m._qf = lazy
        m._prepare_audio_reference({"tokens": torch.ones(1, 3, 128)}, torch.device("cpu"), torch.bfloat16, True)
        m._begin(torch.ones(1, 128, 13, 9, 16), torch.ones(1, 3, 6144, dtype=torch.bfloat16))
        self.assertIs(getattr(real, "_reference_audio_owner", None), m._audio_reference)
        self.assertFalse(lazy.reference_audio_pending())
        self.assertNotIn(real, self.ns["_REFERENCE_AUDIO_PENDING_OWNERS"])
        self.assertEqual(real.current_session.value, 123)
        real.close_ok = False
        m.extra_conds(unprocessed_ltxav_embeds=True)
        self.assertIs(getattr(real, "_reference_audio_owner", None), m._audio_reference)
        real.close_ok = True
        m.extra_conds(unprocessed_ltxav_embeds=True)
        self.assertIsNone(real._reference_audio_owner)
        self.assertIsNone(m._audio_reference)

    def test_failed_upload_owned_by_cached_engine(self):
        m = model(self.ns)
        p = {"tokens": torch.ones(1, 3, 128)}
        m._prepare_audio_reference(p, torch.device("cpu"), torch.float16, True)
        engine = m._qf
        engine.lib.status = 8
        engine.lib.error = "busy"
        engine.lib.copy_pending = True
        with self.assertRaises(RuntimeError):
            m._begin(torch.ones(1, 128, 13, 9, 16), torch.ones(1, 3, 6144, dtype=torch.bfloat16))
        w = weakref.ref(m._audio_reference[-1])
        del m
        gc.collect()
        self.assertIsNotNone(w(), "cached engine must own upload after failed model is collected")
        other = model(self.ns)
        other._qf = engine
        with self.assertRaisesRegex(RuntimeError, "copy completion"):
            other._begin(torch.ones(1, 128, 13, 9, 16), torch.ones(1, 3, 6144, dtype=torch.bfloat16))
        self.assertEqual(len(engine.lib.calls), 1)
        engine.current_session = ctypes.c_void_p(123)
        engine.close_ok = False
        engine.end_session_if_open()
        gc.collect()
        self.assertIsNotNone(w())
        engine.close_ok = True
        engine.end_session_if_open()
        gc.collect()
        self.assertIsNone(w())

    def test_upload_pin_completion_and_failure(self):
        m = model(self.ns)
        p = {"tokens": torch.ones(1, 3, 128)}
        m._prepare_audio_reference(p, torch.device("cpu"), torch.float16, True)
        upload = m._audio_reference[-1]
        w = weakref.ref(upload)
        del upload
        lib = m._qf.lib
        lib.status = 8
        lib.error = "busy permanent refusal"
        lib.copy_pending = True
        with self.assertRaises(RuntimeError):
            m._begin(torch.ones(1, 128, 13, 9, 16), torch.ones(1, 3, 6144, dtype=torch.bfloat16))
        self.assertEqual(len(lib.calls), 1)
        gc.collect()
        self.assertIsNotNone(w())
        m._qf.close_ok = False
        with self.assertRaises(RuntimeError):
            m._prepare_audio_reference(p, torch.device("cpu"), torch.float16, True)
        self.assertEqual(len(lib.calls), 1)
        self.assertIsNotNone(w())
        m._qf.close_ok = True
        with self.assertRaises(RuntimeError):
            m._prepare_audio_reference(p, torch.device("cpu"), torch.float16, True)
        self.assertIsNotNone(w())
        lib.copy_pending = False
        lib.status = 0
        with self.assertRaises(RuntimeError):
            m._prepare_audio_reference(None, torch.device("cpu"), torch.float16, True)
        self.assertIsNotNone(w())
        m._qf.current_session = ctypes.c_void_p(123)
        m._prepare_audio_reference(None, torch.device("cpu"), torch.float16, True)
        gc.collect()
        self.assertIsNone(w())
        m._prepare_audio_reference(p, torch.device("cpu"), torch.float16, True)
        m._begin(torch.ones(1, 128, 13, 9, 16), torch.ones(1, 3, 6144, dtype=torch.bfloat16))
        self.assertTrue(m._audio_reference_bound)
        m._qf.close_ok = False
        m.extra_conds(unprocessed_ltxav_embeds=True)
        self.assertIsNotNone(m._audio_reference)
        m._qf.close_ok = True
        m.extra_conds(unprocessed_ltxav_embeds=True)
        self.assertIsNone(m._audio_reference)

    def _target_fixture(self, flat_batch):
        m = model(self.ns)
        m.latent_shapes = [(2, 128, 1, 1, 1), (2, 8, 4, 16)]
        calls = []

        def step(mp, *args):
            calls.append((mp.base.cfg_context_key, int(mp.audio_reference_active),
                          list(mp.base.dims), list(mp.audio_dims)))
            m._out.fill_(len(calls))
            m._out_audio.fill_(100 + len(calls))

        m._call_denoise_step_av = step
        return m, torch.ones(flat_batch, 1, 640), torch.ones(flat_batch, 3, 6144), calls

    def test_target_batch2_single_condition(self):
        m, x, ctx, calls = self._target_fixture(2)
        before = x.clone()
        shapes = list(m.latent_shapes)
        payload = {"tokens": torch.ones(1, 3, 128)}
        result = m._apply_model(x, torch.ones(2), c_crossattn=ctx, ref_audio=payload,
                                transformer_options={"cond_or_uncond": [0], "uuids": ["a"]})
        self.assertEqual([c[0] for c in calls], [1, 2])
        self.assertTrue(all(c[1] == 1 for c in calls))
        self.assertTrue(all(c[2] == [1, 1, 128, 0, 0] and c[3] == [1, 4, 128, 0] for c in calls))
        for i in range(2):
            self.assertTrue(torch.all(result[i, :, :128] == i + 1))
            self.assertTrue(torch.all(result[i, :, 128:] == 101 + i))
        self.assertTrue(torch.equal(x, before))
        self.assertEqual(m.latent_shapes, shapes)
        upload = m._audio_reference[-1]
        m._apply_model(x, torch.ones(2), c_crossattn=ctx, ref_audio=None,
                       transformer_options={"cond_or_uncond": [0], "uuids": ["a"]})
        self.assertEqual([c[0] for c in calls[-2:]], [1, 2])
        self.assertTrue(all(c[1] == 0 for c in calls[-2:]))
        self.assertIs(m._audio_reference[-1], upload)
        self.assertEqual(len(m._qf.lib.calls), 1)

    def test_target_batch2_group_composition_keys(self):
        m, x, ctx, calls = self._target_fixture(4)
        payload = {"tokens": torch.ones(1, 3, 128)}

        def run(roles, ids, batch):
            return m._apply_model(x[:batch], torch.ones(batch), c_crossattn=ctx[:batch],
                                  ref_audio=payload, transformer_options={"cond_or_uncond": roles, "uuids": ids})

        result = run([0, 1], ["a", "b"], 4)
        self.assertEqual([c[0] for c in calls], [1, 2, 3, 4])
        for i in range(4):
            self.assertTrue(torch.all(result[i, :, :128] == i + 1))
        run([1], ["b"], 2)
        self.assertEqual([c[0] for c in calls[-2:]], [3, 4])
        run([1, 0], ["b", "a"], 4)
        self.assertEqual([c[0] for c in calls[-4:]], [3, 4, 1, 2])
        run([0, 0], ["a", "c"], 4)
        self.assertEqual([c[0] for c in calls[-4:]], [1, 2, 5, 6])
        run([0], [None], 2)
        self.assertEqual([c[0] for c in calls[-2:]], [0, 0])
        run([0], ["a"], 2)
        self.assertEqual([c[0] for c in calls[-2:]], [1, 2])

    def test_packed_cfg_expands_actual_batch(self):
        m, x, ctx, calls = self._target_fixture(2)
        m.latent_shapes = [(1, 128, 1, 1, 1), (1, 8, 4, 16)]
        shapes = list(m.latent_shapes)
        result = m._apply_model(x, torch.ones(2), c_crossattn=ctx,
                                transformer_options={"cond_or_uncond": [0, 1], "uuids": ["a", "b"]})
        self.assertEqual(tuple(result.shape), tuple(x.shape))
        self.assertEqual([c[0] for c in calls], [1, 2])
        self.assertTrue(all(c[1] == 0 for c in calls))
        self.assertEqual(m.latent_shapes, shapes)

    def test_nested_target_batch2_missing_uuid_is_zero(self):
        m, x, ctx, calls = self._target_fixture(2)

        class Pair:
            is_nested = True

            def __init__(self, values):
                self.values = values

            def unbind(self):
                return self.values

        self.ns["comfy"].nested_tensor = types.SimpleNamespace(NestedTensor=Pair)
        source = Pair((x[:, :, :128].reshape(2, 128, 1, 1, 1), x[:, :, 128:].reshape(2, 8, 4, 16)))
        result = m._apply_model(source, torch.ones(2), c_crossattn=ctx,
                                ref_audio={"tokens": torch.ones(1, 3, 128)},
                                transformer_options={"cond_or_uncond": [0], "uuids": [None]})
        self.assertEqual([c[0] for c in calls], [0, 0])
        video, audio = result.unbind()
        self.assertEqual(tuple(video.shape), (2, 128, 1, 1, 1))
        self.assertEqual(tuple(audio.shape), (2, 8, 4, 16))
        self.assertTrue(torch.all(video[0] == 1) and torch.all(video[1] == 2))
        self.assertTrue(torch.all(audio[0] == 101) and torch.all(audio[1] == 102))

    def test_target_batch2_invalid_groups_and_context_recover(self):
        m, x, ctx, calls = self._target_fixture(2)
        payload = {"tokens": torch.ones(1, 3, 128)}
        for groups in [None, [], [0, 1, 0]]:
            with self.assertRaises(RuntimeError):
                m._apply_model(x, torch.ones(2), c_crossattn=ctx, ref_audio=payload,
                               transformer_options={"cond_or_uncond": groups, "uuids": ["a"]})
        with self.assertRaisesRegex(RuntimeError, "context batch"):
            m._apply_model(x, torch.ones(2), c_crossattn=ctx[:1], ref_audio=payload,
                           transformer_options={"cond_or_uncond": [0], "uuids": ["a"]})
        self.assertEqual(calls, [])
        self.assertEqual(m._qf.lib.calls, [])
        self.assertIsNone(getattr(m, "_audio_reference", None))
        m._apply_model(x, torch.ones(2), c_crossattn=ctx, ref_audio=payload,
                       transformer_options={"cond_or_uncond": [0], "uuids": ["a"]})
        self.assertEqual(len(calls), 2)

    def _target_mask_fixture(self):
        m = model(self.ns)
        m.latent_shapes = [(2, 128, 1, 1, 1), (2, 8, 4, 16)]
        audio = torch.ones(8, 4, 16)
        audio[:, 1] = 0.0
        audio[:, 2] = 0.5
        row = torch.cat((torch.full((128,), 0.25), audio.reshape(-1)))
        return m, row.reshape(1, 1, -1).repeat(2, 1, 1)

    def test_target_batch2_shared_mask_scales(self):
        m, mask = self._target_mask_fixture()
        source = mask.clone()
        expected = ([0.25], [1., 0., 0.5, 1.])
        self.assertEqual(m._frame_scales_from_mask(mask), expected)
        m.latent_shapes = [(1, 128, 1, 1, 1), (1, 8, 4, 16)]
        self.assertEqual(m._frame_scales_from_mask(mask[:1]), expected)
        self.assertTrue(torch.equal(mask, source))
        video = torch.ones(2, 128, 2, 1, 1)
        video[:, :, 0] = 0.25
        self.assertEqual(m._frame_scales_from_mask(video), ([0.25, 1.], []))

    def test_target_batch2_distinct_masks_refuse_and_recover(self):
        m, mask = self._target_mask_fixture()
        bad = mask.clone()
        bad[1, 0, :128] = 0.5
        with self.assertRaisesRegex(RuntimeError, "same per-frame/audio-row"):
            m._frame_scales_from_mask(bad)
        bad = mask.clone()
        bad[1, 0, 128:].reshape(8, 4, 16)[:, 1] = 0.25
        with self.assertRaisesRegex(RuntimeError, "same per-frame/audio-row"):
            m._frame_scales_from_mask(bad)
        video = torch.ones(2, 128, 2, 1, 1)
        video[1, :, 0] = 0.5
        with self.assertRaisesRegex(RuntimeError, "same per-frame/audio-row"):
            m._frame_scales_from_mask(video)
        self.assertEqual(m._frame_scales_from_mask(mask), ([0.25], [1., 0., 0.5, 1.]))

    def test_mask_nonfinite_refuses_before_clamp(self):
        m, mask = self._target_mask_fixture()
        for value in [float("nan"), float("inf"), -float("inf")]:
            for offset in [0, 128]:
                bad = mask.clone()
                bad[0, 0, offset] = value
                with self.assertRaisesRegex(RuntimeError, "finite"):
                    m._frame_scales_from_mask(bad)


class FrozenAudioTests(unittest.TestCase):
    def fixture(self, batch=1, dtype=torch.float32, av=True):
        ns = production_namespace()
        m = model(ns, av=av)
        shapes = [(batch, 128, 1, 1, 1), (batch, 8, 2, 16)]
        clean = torch.linspace(-1.23, 1.17, batch * 384, dtype=dtype).reshape(batch, 1, 384)
        result = clean.float() + 0.123
        mask = torch.ones_like(result)
        mask[:, :, 128:] = 0
        guider = types.SimpleNamespace(model_patcher=types.SimpleNamespace(model=m), model_options={})
        executor = types.SimpleNamespace(class_obj=guider)
        return ns, m, shapes, clean, result, mask, executor

    def invoke(self, fixture, *, result=None, mask="fixture", shapes="fixture", keyword=False,
               all_positional=False, action=None, sigmas=None):
        ns, m, ls, clean, sampled, default_mask, holder = fixture
        result = sampled if result is None else result
        mask = default_mask if isinstance(mask, str) else mask
        shapes = ls if isinstance(shapes, str) else shapes
        sigmas = torch.tensor([1., 0.]) if sigmas is None else sigmas
        args = (torch.zeros_like(clean), clean, object(), sigmas, mask, None, False, 42)
        outer = self

        class Executor:
            class_obj = holder.class_obj

            def __call__(self, *a, **kw):
                current = ns["_QF_SAMPLING_GEOMETRY"].get()
                outer.assertIs(current[0], m)
                outer.assertEqual(current[1], shapes)
                if action:
                    action(self.class_obj)
                return result

        sentinel = object()
        token = ns["_QF_SAMPLING_GEOMETRY"].set(sentinel)
        try:
            wrapper = ns["QFModelPatcher"]._sampling_geometry
            if keyword:
                names = ("noise", "latent_image", "sampler", "sigmas", "denoise_mask", "callback", "disable_pbar", "seed")
                return wrapper(Executor(), **dict(zip(names, args)), latent_shapes=shapes)
            if all_positional:
                return wrapper(Executor(), *args, shapes)
            return wrapper(Executor(), *args, latent_shapes=shapes)
        finally:
            self.assertIs(ns["_QF_SAMPLING_GEOMETRY"].get(), sentinel)
            ns["_QF_SAMPLING_GEOMETRY"].reset(token)

    def test_exact_frozen_audio_after_euler_cancellation(self):
        f = self.fixture()
        clean = f[3]
        x = torch.randn(clean.shape, generator=torch.Generator().manual_seed(42))
        sampled = x + (x - clean) * -1.
        self.assertFalse(torch.equal(sampled[:, :, 128:], clean[:, :, 128:]))
        output = self.invoke(f, result=sampled)
        self.assertTrue(torch.equal(output[:, :, 128:], clean[:, :, 128:]))
        self.assertTrue(torch.equal(output[:, :, :128], sampled[:, :, :128]))

    def test_mixed_mask_restores_only_exact_zero_audio(self):
        f = self.fixture()
        mask = torch.ones_like(f[5])
        mask[:, :, :128] = 0
        mask[:, :, 128:133] = torch.tensor([0., .5, 1., 1e-12, -0.])
        output = self.invoke(f, mask=mask)
        expected = f[4].clone()
        expected[:, :, 128] = f[3][:, :, 128]
        expected[:, :, 132] = f[3][:, :, 132]
        self.assertTrue(torch.equal(output, expected))

    def test_noop_masks_and_borrowed_tensors_are_unchanged(self):
        f = self.fixture()
        clean, result, mask = f[3:6]
        copies = [t.clone() for t in (clean, result, mask)]
        for candidate in [None, torch.ones_like(mask),
                          torch.cat([torch.zeros_like(mask[:, :, :128]), torch.ones_like(mask[:, :, 128:])], dim=-1)]:
            with self.subTest(mask=candidate is None):
                self.assertIs(self.invoke(f, mask=candidate), result)
        output = self.invoke(f)
        self.assertIsNot(output, result)
        for a, b in zip((clean, result, mask), copies):
            self.assertTrue(torch.equal(a, b))

    def test_batches_noncontiguous_result_and_dtype(self):
        f = self.fixture(batch=2, dtype=torch.bfloat16)
        sampled = f[4].reshape(2, -1).t().contiguous().t().unsqueeze(1)
        self.assertFalse(sampled.is_contiguous())
        output = self.invoke(f, result=sampled)
        self.assertEqual(output.shape, sampled.shape)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.equal(output[:, :, 128:], f[3][:, :, 128:].float()))
        self.assertTrue(torch.equal(output[:, :, :128], sampled[:, :, :128]))

    def test_keyword_and_all_positional_geometry(self):
        for key in ["keyword", "all_positional"]:
            with self.subTest(key=key):
                f = self.fixture()
                output = self.invoke(f, **{key: True})
                self.assertTrue(torch.equal(output[:, :, 128:], f[3][:, :, 128:]))

    def test_other_models_keep_executor_result(self):
        f = self.fixture(av=False)
        self.assertFalse(hasattr(f[1], "_qf_restore_frozen_audio"))
        self.assertIs(self.invoke(f), f[4])

    def test_dynamic_mask_registration_before_and_after(self):
        for initial, action in [
            (True, lambda g: None),
            (True, lambda g: g.model_options.pop("denoise_mask_function")),
            (False, lambda g: g.model_options.update(denoise_mask_function=object())),
        ]:
            with self.subTest(initial=initial):
                f = self.fixture()
                if initial:
                    f[6].class_obj.model_options["denoise_mask_function"] = object()
                self.assertIs(self.invoke(f, action=action), f[4])

    def test_executor_error_skips_restoration(self):
        f = self.fixture()
        calls = []
        f[1]._qf_restore_frozen_audio = lambda *a, **kw: calls.append(True)

        def fail(g):
            raise ValueError("executor witness")

        with self.assertRaisesRegex(ValueError, "executor witness"):
            self.invoke(f, action=fail)
        self.assertEqual(calls, [])

    def test_finalizer_error_resets_context(self):
        f = self.fixture()
        with self.assertRaises(RuntimeError):
            self.invoke(f, result=f[4][:, :, :-1])

    def test_bad_geometry_rejects_before_audio_slice(self):
        variants = [
            None, [], [(1, 128, 1, 1, 1)], [(1, 128, 1, 1, 1), (1, 8, 2, 16), (1, 1)],
            [(1, 128, 1, 1), (1, 8, 2, 16)], [(1, 128, 1, 1, 1), (1, 8, 2, 1, 16)],
            [(2, 128, 1, 1, 1), (1, 8, 2, 16)], [(0, 128, 1, 1, 1), (0, 8, 2, 16)],
            [(1, 128, 1, 1, 1), (1, 8, -2, 16)],
        ]
        for shapes in variants:
            with self.subTest(shapes=shapes):
                with self.assertRaises(RuntimeError):
                    self.invoke(self.fixture(), shapes=shapes)
        f = self.fixture()
        for index in [3, 4, 5]:
            for replacement in [f[index][:, :, :-1], f[index].reshape(2, 1, 192)]:
                with self.subTest(index=index, shape=replacement.shape):
                    changed = list(f)
                    changed[index] = replacement
                    with self.assertRaises(RuntimeError):
                        self.invoke(changed)
        self.assertIs(self.invoke(f, mask=None, shapes=None), f[4])

    def test_no_mask_state_retained_between_calls(self):
        f = self.fixture()
        keys = set(vars(f[1]))
        self.invoke(f)
        self.assertEqual(set(vars(f[1])), keys)
        changed = list(f)
        changed[3] = f[3] + 4
        changed[4] = f[4] + 7
        self.assertIs(self.invoke(changed, mask=torch.ones_like(f[5])), changed[4])

    def test_partial_end_sigma_keeps_frozen_audio_clean(self):
        f = self.fixture()
        output = self.invoke(f, sigmas=torch.tensor([1., .5]))
        self.assertTrue(torch.equal(output[:, :, 128:], f[3][:, :, 128:]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
