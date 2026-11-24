#!/usr/bin/env python3
"""CPU-only runtime contract for QFQwenImage21Model (t2i + multi-reference edit):
  - extra_conds tracks reference presence and cond-group count, emits refs/slots via ComfyUI's
    QwenImage21.extra_conds, and refuses _ENGINE_IGNORED_COND_KEYS loudly;
  - _check_conds refuses reference_latents when loaded engine lacks qf_xfm_step_refs;
  - _denoise_group marshals QFStepRefsParams (struct_size, num_refs, ref_latents, ref_dims,
    image_slots with slot padding and per-group ref slicing) into qf_xfm_step_refs;
  - > QF_MAX_STEP_REFS (16) refs and non-4D refs fail loud and close the open session;
  - single-cond batch=1 t2i calls qf_xfm_step (not qf_xfm_step_refs) with 0.85 ratio, whereas
    edit / CFG>1 / batch>1 always emit 1.0;
  - process_latent_out resets per-run state and closes the open session.

Run with COMFY_ROOT and its Python environment (CPU only)."""
import ctypes
import importlib
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

root = os.environ.get("COMFY_ROOT")
if not root or not (Path(root) / "comfy/model_management.py").is_file():
    print("[SKIP] qwenimage21_refs_contract: set COMFY_ROOT to the tested host")
    raise SystemExit(77)
sys.path.insert(0, root)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import comfy.supported_models
import torch

pkg = types.ModuleType("qf_qi21_contract")
pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[pkg.__name__] = pkg
qfe = importlib.import_module("qf_qi21_contract.qf_engine")
qfm = importlib.import_module("qf_qi21_contract.qf_modelpatcher")
qi21 = importlib.import_module("qf_qi21_contract.qf_qwenimage21_modelpatcher")
krea2 = importlib.import_module("qf_qi21_contract.qf_krea2_modelpatcher")


class _FakeLib:
    def __init__(self, with_step_refs=True):
        self.events = []
        self._next_tok = 0x2000
        if with_step_refs:
            self.qf_xfm_step_refs = self._step_refs

    def qf_last_error(self):
        return b"stub error"

    def qf_xfm_session_begin(self, _xfm, params_ref, session_ref):
        p = params_ref._obj
        tok = self._next_tok
        self._next_tok += 1
        session_ref._obj.value = tok
        self.events.append(("begin", {
            "token": tok,
            "width": int(p.width),
            "height": int(p.height),
            "num_steps": int(p.num_steps),
            "max_context_dims": tuple(int(v) for v in p.max_context_dims),
            "options": json.loads(p.options_json.decode("utf-8")),
        }))
        return qfe.QF_OK

    def qf_xfm_step(self, session, params_ref):
        p = params_ref._obj
        self.events.append(("step", {
            "token": int(session.value),
            "struct_size": int(p.struct_size),
            "dims": tuple(int(d) for d in p.dims),
            "step_index": int(p.step_index),
            "ctx_key": int(p.cfg_context_key),
        }))
        return qfe.QF_OK

    def _step_refs(self, session, params_ref):
        rp = params_ref._obj
        n = int(rp.num_refs)
        ptrs = [int(rp.ref_latents[k]) for k in range(n)]
        dims = [tuple(int(rp.ref_dims[4 * k + j]) for j in range(4)) for k in range(n)]
        slots = [int(rp.image_slots[k]) for k in range(n)]
        self.events.append(("step_refs", {
            "token": int(session.value),
            "struct_size": int(rp.struct_size),
            "base_struct_size": int(rp.base.struct_size),
            "base_dims": tuple(int(d) for d in rp.base.dims),
            "num_refs": n,
            "ref_ptrs": ptrs,
            "ref_dims": dims,
            "image_slots": slots,
            "ctx_key": int(rp.base.cfg_context_key),
        }))
        return qfe.QF_OK

    def qf_xfm_session_end(self, session):
        self.events.append(("end", {"token": int(session.value)}))
        return qfe.QF_OK


class _FakeEngine:
    def __init__(self, lib):
        self.lib = lib
        self.xfm = ctypes.c_void_p(0xBEEF)
        self.current_session = None
        self.step_count = 0
        self.sampler_step_count = 0

    def end_session_if_open(self):
        if self.current_session is None:
            return False, True
        st = self.lib.qf_xfm_session_end(self.current_session)
        if st == qfe.QF_OK:
            self.current_session = None
            return True, True
        return True, False


def _make_model(with_step_refs=True):
    lib = _FakeLib(with_step_refs=with_step_refs)
    eng = _FakeEngine(lib)
    cfg = comfy.supported_models.QwenImage21({"image_model": "qwen_image21", "disable_unet_model_creation": True})
    qfm.ensure_model_config_attrs(cfg)
    model = qi21.QFQwenImage21Model(cfg, eng, device=torch.device("cpu"))
    model.set_video_enhance(False)
    return model, lib, eng


def _ratio_value(begin_opts):
    return next(v for k, v in begin_opts.items() if isinstance(v, float) and k.endswith("_ratio"))


class QwenImage21RefsContract(unittest.TestCase):
    def test_extra_conds_tracks_refs_cond_groups_and_refuses_ignored_keys(self):
        model, lib, _ = _make_model()
        c_ch = int(model.latent_format.latent_channels)
        ref = torch.ones((1, c_ch, 4, 4), dtype=torch.float32)
        ctx = torch.ones((1, 8, 16), dtype=torch.bfloat16)

        with mock.patch.object(qfe, "load_lib", return_value=lib):
            out1 = model.extra_conds(cross_attn=ctx)
            self.assertFalse(model._run_has_refs)
            self.assertEqual(model._run_cond_groups, 1)
            self.assertIn("c_crossattn", out1)

            # Second extra_conds before _apply_model simulates CFG (pos + neg).
            out2 = model.extra_conds(cross_attn=ctx, reference_latents=[ref], image_slots=[3])
            self.assertTrue(model._run_has_refs)
            self.assertEqual(model._run_cond_groups, 2)
            self.assertIn("ref_latents", out2)
            self.assertIn("image_slots", out2)

            for bad_key in qi21.QFQwenImage21Model._ENGINE_IGNORED_COND_KEYS:
                with self.subTest(ignored=bad_key):
                    with self.assertRaisesRegex(RuntimeError, bad_key):
                        model.extra_conds(cross_attn=ctx, **{bad_key: torch.ones((1, 1))})

    def test_check_conds_requires_qf_xfm_step_refs_when_refs_wired(self):
        model_no, lib_no, _ = _make_model(with_step_refs=False)
        model_yes, lib_yes, _ = _make_model(with_step_refs=True)
        ref = torch.ones((1, 16, 4, 4), dtype=torch.float32)

        with mock.patch.object(qfe, "load_lib", return_value=lib_no):
            model_no._check_conds({})  # no refs -> ok
            with self.assertRaisesRegex(RuntimeError, "qf_xfm_step_refs"):
                model_no._check_conds({"reference_latents": [ref]})

        with mock.patch.object(qfe, "load_lib", return_value=lib_yes):
            model_yes._check_conds({"reference_latents": [ref]})

    def test_denoise_group_marshals_qf_step_refs_params_and_slots(self):
        model, lib, _ = _make_model()
        c_ch = int(model.latent_format.latent_channels)
        x = torch.ones((2, c_ch, 4, 4), dtype=torch.bfloat16)
        ctx = torch.ones((2, 11, 16), dtype=torch.bfloat16)
        ref_batched = torch.ones((2, c_ch, 4, 4), dtype=torch.float32)
        ref_single = torch.ones((1, c_ch, 6, 8), dtype=torch.float32)
        sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
        opts = {"sample_sigmas": sigmas, "cond_or_uncond": [0, 1], "uuids": ["pos", "neg"]}

        out = model._apply_model(
            x, torch.tensor([1.0, 1.0], dtype=torch.float32),
            c_crossattn=ctx, transformer_options=opts,
            ref_latents=[ref_batched, ref_single], image_slots=[5],
        )
        self.assertEqual(tuple(out.shape), tuple(x.shape))

        begins = [ev for kind, ev in lib.events if kind == "begin"]
        self.assertEqual(len(begins), 1)
        self.assertEqual(_ratio_value(begins[0]["options"]), 1.0)

        ref_steps = [ev for kind, ev in lib.events if kind == "step_refs"]
        plain_steps = [ev for kind, ev in lib.events if kind == "step"]
        self.assertEqual(len(plain_steps), 0)
        self.assertEqual(len(ref_steps), 2)
        for st in ref_steps:
            self.assertEqual(st["struct_size"], ctypes.sizeof(qfe.QFStepRefsParams))
            self.assertEqual(st["base_struct_size"], ctypes.sizeof(qfe.QFStepParams))
            self.assertEqual(st["num_refs"], 2)
            self.assertTrue(all(ptr != 0 for ptr in st["ref_ptrs"]))
            self.assertEqual(st["ref_dims"], [(1, c_ch, 4, 4), (1, c_ch, 6, 8)])
            self.assertEqual(st["image_slots"], [5, 11])  # missing 2nd slot defaults to ctx seq_len (11)
        self.assertNotEqual(ref_steps[0]["ctx_key"], ref_steps[1]["ctx_key"])

    def test_denoise_group_refuses_more_than_16_refs_and_closes_session(self):
        model, lib, eng = _make_model()
        c_ch = int(model.latent_format.latent_channels)
        x = torch.ones((1, c_ch, 4, 4), dtype=torch.bfloat16)
        ctx = torch.ones((1, 8, 16), dtype=torch.bfloat16)
        too_many = [torch.ones((1, c_ch, 2, 2), dtype=torch.bfloat16) for _ in range(qfe.QF_MAX_STEP_REFS + 1)]
        opts = {"sample_sigmas": torch.tensor([1.0, 0.0]), "cond_or_uncond": [0], "uuids": ["c"]}

        with self.assertRaisesRegex(RuntimeError, f"at most {qfe.QF_MAX_STEP_REFS}"):
            model._apply_model(x, torch.tensor([1.0]), c_crossattn=ctx, transformer_options=opts, ref_latents=too_many)
        self.assertIsNone(eng.current_session)
        self.assertEqual([k for k, _ in lib.events], ["begin", "end"])

    def test_denoise_group_refuses_non_4d_ref_and_closes_session(self):
        c_ch = None
        for bad_shape in ((1, 16, 1, 4, 4), (16, 4, 4)):
            with self.subTest(bad_shape=bad_shape):
                model, lib, eng = _make_model()
                c_ch = int(model.latent_format.latent_channels)
                x = torch.ones((1, c_ch, 4, 4), dtype=torch.bfloat16)
                ctx = torch.ones((1, 8, 16), dtype=torch.bfloat16)
                bad_ref = torch.ones(bad_shape, dtype=torch.bfloat16)
                opts = {"sample_sigmas": torch.tensor([1.0, 0.0]), "cond_or_uncond": [0], "uuids": ["c"]}
                with self.assertRaisesRegex(RuntimeError, "4-D"):
                    model._apply_model(
                        x, torch.tensor([1.0]), c_crossattn=ctx, transformer_options=opts, ref_latents=[bad_ref])
                self.assertIsNone(eng.current_session)
                self.assertEqual([k for k, _ in lib.events], ["begin", "end"])

    def test_single_cond_t2i_uses_qf_xfm_step_and_process_latent_out_resets_state(self):
        model, lib, eng = _make_model()
        c_ch = int(model.latent_format.latent_channels)
        x = torch.ones((1, c_ch, 4, 4), dtype=torch.bfloat16)
        ctx = torch.ones((1, 8, 16), dtype=torch.bfloat16)
        opts = {"sample_sigmas": torch.tensor([1.0, 0.0]), "cond_or_uncond": [0], "uuids": ["c"]}

        model._apply_model(x, torch.tensor([1.0]), c_crossattn=ctx, transformer_options=opts)
        begins = [ev for kind, ev in lib.events if kind == "begin"]
        self.assertEqual(_ratio_value(begins[0]["options"]), 0.85)
        self.assertEqual(sum(k == "step" for k, _ in lib.events), 1)
        self.assertEqual(sum(k == "step_refs" for k, _ in lib.events), 0)
        self.assertIsNotNone(eng.current_session)

        # Simulate prior edit/CFG state, then verify process_latent_out closes session and resets all flags.
        model._run_has_refs = True
        model._run_batch = 2
        model._run_single_cond_group = False
        model._run_cond_groups = 2
        out = model.process_latent_out(x)
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertIsNone(eng.current_session)
        self.assertFalse(model._run_has_refs)
        self.assertEqual(model._run_batch, 1)
        self.assertTrue(model._run_single_cond_group)
        self.assertEqual(model._run_cond_groups, 0)
        self.assertEqual(sum(k == "end" for k, _ in lib.events), 1)

    def test_t12_qi21_cfg_siblings_nonweakref_noise_outer_sample_reset_and_presession_abort(self):
        model, lib, _ = _make_model()
        c_ch = int(model.latent_format.latent_channels)
        ref = torch.ones((1, c_ch, 4, 4), dtype=torch.float32)
        ctx_long = torch.ones((1, 512, 16), dtype=torch.bfloat16)
        ctx_short = torch.ones((1, 64, 16), dtype=torch.bfloat16)
        ctx_t2i = torch.ones((1, 32, 16), dtype=torch.bfloat16)

        # (1) Same noise (including non-weakref dict/object): Cond 1 + Cond 2 accumulate without wiping Cond 1
        with mock.patch.object(qfe, "load_lib", return_value=lib):
            for noise_obj in ({"stub": 1}, object(), torch.zeros((1, c_ch, 4, 4))):
                model._qf_conds_done = True
                model.extra_conds(cross_attn=ctx_long, reference_latents=[ref], noise=noise_obj)
                model.extra_conds(cross_attn=ctx_short, noise=noise_obj)
                self.assertTrue(model._run_has_refs)
                self.assertEqual(model._run_cond_groups, 2)
                self.assertEqual(model._max_ctx_seq, 512)

            # (2) Same OUTER_SAMPLE run_id accumulates; exception inside executor resets _QF_RUN_ID to None
            exec_ok = types.SimpleNamespace(
                class_obj=types.SimpleNamespace(
                    model_patcher=types.SimpleNamespace(model=model), model_options={}),
            )
            def _run_outer():
                model.extra_conds(cross_attn=ctx_long, reference_latents=[ref])
                model.extra_conds(cross_attn=ctx_short)
                return "done"
            class _Exec:
                class_obj = exec_ok.class_obj
                def __call__(self, *a, **kw):
                    return _run_outer()
            self.assertIsNone(qfm._QF_RUN_ID.get())
            self.assertEqual(qfm.QFModelPatcher._sampling_geometry(_Exec()), "done")
            self.assertTrue(model._run_has_refs)
            self.assertEqual(model._run_cond_groups, 2)
            self.assertEqual(model._max_ctx_seq, 512)
            self.assertIsNone(qfm._QF_RUN_ID.get())

            class _BoomExec:
                class_obj = exec_ok.class_obj
                def __call__(self, *a, **kw):
                    raise RuntimeError("sampler boom")
            with self.assertRaisesRegex(RuntimeError, "sampler boom"):
                qfm.QFModelPatcher._sampling_geometry(_BoomExec())
            self.assertIsNone(qfm._QF_RUN_ID.get())

            # (3) Pre-session _apply_model aborts (control / missing sigmas / missing c_crossattn) reset next naked extra_conds
            x = torch.ones((1, c_ch, 4, 4), dtype=torch.bfloat16)
            for abort_kw in (
                {"c_crossattn": ctx_long, "transformer_options": {"sample_sigmas": torch.tensor([1.0, 0.0])}, "control": object()},
                {"c_crossattn": ctx_long, "transformer_options": {}},
                {"c_crossattn": None, "transformer_options": {"sample_sigmas": torch.tensor([1.0, 0.0])}},
            ):
                model.extra_conds(cross_attn=ctx_long, reference_latents=[ref])
                self.assertTrue(model._run_has_refs)
                with self.assertRaises(RuntimeError):
                    model._apply_model(x, torch.tensor([1.0]), **abort_kw)
                model.extra_conds(cross_attn=ctx_t2i)
                self.assertFalse(model._run_has_refs)
                self.assertEqual(model._run_cond_groups, 1)
                self.assertEqual(model._max_ctx_seq, 32)

        opts_t2i = {"sample_sigmas": torch.tensor([1.0, 0.0]), "cond_or_uncond": [0], "uuids": ["t2i"]}
        begin_n = len(lib.events)
        model._apply_model(x, torch.tensor([1.0]), c_crossattn=ctx_t2i, transformer_options=opts_t2i)
        begins = [ev for kind, ev in lib.events[begin_n:] if kind == "begin"]
        self.assertEqual(_ratio_value(begins[0]["options"]), 0.85)
        self.assertEqual(begins[0]["max_context_dims"][1], 32)
        model.process_latent_out(x)

    def test_t12b_extra_conds_reset_on_prev_noise_change_and_gc(self):
        import gc
        model, _lib, eng = _make_model()
        ctx_long = torch.ones((1, 512, 16), dtype=torch.bfloat16)
        ctx_short = torch.ones((1, 32, 16), dtype=torch.bfloat16)
        ref = torch.ones((1, 16, 1, 2, 2), dtype=torch.bfloat16)
        n1 = torch.ones((1, 1, 1, 1))
        n2 = torch.ones((1, 1, 1, 1))
        n3 = torch.ones((1, 1, 1, 1))

        # Distinct live noise with _qf_conds_done=False and was_open=False triggers prev is not noise reset
        self.assertFalse(model._qf_conds_done)
        self.assertIsNone(eng.current_session)
        with mock.patch.object(qfe, "load_lib", return_value=_lib):
            model.extra_conds(cross_attn=ctx_long, reference_latents=[ref], noise=n1)
            self.assertTrue(model._run_has_refs)
            self.assertEqual(model._max_ctx_seq, 512)
            self.assertFalse(model._qf_conds_done)

            model.extra_conds(cross_attn=ctx_short, noise=n2)
            self.assertFalse(model._run_has_refs)
            self.assertEqual(model._run_cond_groups, 1)
            self.assertEqual(model._max_ctx_seq, 32)

            # Garbage-collected prior noise (prev is None while prev_ref is not None) also triggers fresh reset
            model.extra_conds(cross_attn=ctx_long, reference_latents=[ref], noise=n2)
            self.assertTrue(model._run_has_refs)
            del n2
            gc.collect()
            model.extra_conds(cross_attn=ctx_short, noise=n3)
            self.assertFalse(model._run_has_refs)
            self.assertEqual(model._run_cond_groups, 1)
            self.assertEqual(model._max_ctx_seq, 32)

    def test_t13_krea2_extra_conds_rollback_and_presession_abort(self):
        lib = _FakeLib()
        eng = _FakeEngine(lib)
        k_cfg = comfy.supported_models.Krea2({"image_model": "krea2", "disable_unet_model_creation": True})
        qfm.ensure_model_config_attrs(k_cfg)
        km = krea2.QFKrea2Model(k_cfg, eng, device=torch.device("cpu"))
        km.set_video_enhance(False)
        c_ch = int(km.latent_format.latent_channels)
        ctx_512 = torch.ones((1, 512, 16), dtype=torch.bfloat16)
        ctx_32 = torch.ones((1, 32, 16), dtype=torch.bfloat16)

        km.extra_conds(cross_attn=ctx_512)
        self.assertEqual(km._max_ctx_seq, 512)
        with self.assertRaisesRegex(RuntimeError, "reference_latents"):
            km.extra_conds(cross_attn=ctx_32, reference_latents=object())
        self.assertEqual(km._max_ctx_seq, 0)

        km.extra_conds(cross_attn=ctx_512)
        x = torch.ones((1, c_ch, 1, 4, 4), dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "sigma schedule"):
            km._apply_model(x, torch.tensor([1.0]), c_crossattn=ctx_32, transformer_options={})
        km.extra_conds(cross_attn=ctx_32)
        self.assertEqual(km._max_ctx_seq, 32)

    def test_t15_and_t16_image_session_busy_schedule_finally_and_scratch_realloc(self):
        model, lib, eng = _make_model()
        c_ch = int(model.latent_format.latent_channels)
        x = torch.ones((1, c_ch, 4, 4), dtype=torch.bfloat16)
        ctx = torch.ones((1, 8, 16), dtype=torch.bfloat16)

        # Strict _stage_schedule refuses missing/short/non-decreasing before begin
        for sched in (None, [1.0], [0.4, 0.8], [0.5, 0.5]):
            to = {} if sched is None else {"sample_sigmas": torch.tensor(sched, dtype=torch.float32)}
            with self.assertRaises(RuntimeError):
                model._apply_model(x, torch.tensor([1.0]), c_crossattn=ctx, transformer_options=to)
        self.assertFalse(any(k == "begin" for k, _ in lib.events))

        # Busy retry 1x succeeds; 2x fails loud
        orig_begin = lib.qf_xfm_session_begin
        statuses = [qfe.QF_ERR_BUSY, qfe.QF_OK]

        def flaky_begin(xfm, params_ref, session_ref):
            st = statuses.pop(0)
            if st != qfe.QF_OK:
                lib.qf_last_error = lambda: b"busy draining"
                return st
            return orig_begin(xfm, params_ref, session_ref)

        lib.qf_xfm_session_begin = flaky_begin
        opts = {"sample_sigmas": torch.tensor([1.0, 0.0]), "cond_or_uncond": [0], "uuids": ["c"]}
        with mock.patch.object(qfm.time, "sleep") as slp:
            model._apply_model(x, torch.tensor([1.0]), c_crossattn=ctx, transformer_options=opts)
        slp.assert_called_once_with(qfm._BUSY_BEGIN_RETRY_S)
        self.assertIsNotNone(eng.current_session)

        # Pre-seeded wrong dtype & wrong device scratch buffers must be reallocated cleanly
        for bad_buf in (
            torch.zeros(x.shape, dtype=torch.float16, device="cpu"),
            torch.empty(x.shape, dtype=torch.bfloat16, device="meta"),
        ):
            model._out = bad_buf
            model._apply_model(x, torch.tensor([1.0]), c_crossattn=ctx, transformer_options=opts)
            self.assertEqual(model._out.dtype, torch.bfloat16)
            self.assertEqual(model._out.device, x.device)
            self.assertIsNot(model._out, bad_buf)

        # process_latent_out closes session & clears _out even when qfe.info raises
        with mock.patch.object(qfe, "info", side_effect=RuntimeError("console boom")):
            with self.assertRaisesRegex(RuntimeError, "console boom"):
                model.process_latent_out(x)
        self.assertIsNone(eng.current_session)
        self.assertIsNone(model._out)

        # 2x consecutive QF_ERR_BUSY exhausts retry and leaves current_session None
        model2, lib2, eng2 = _make_model()
        lib2.qf_xfm_session_begin = lambda *_a: qfe.QF_ERR_BUSY
        with mock.patch.object(qfm.time, "sleep") as slp2:
            with self.assertRaisesRegex(RuntimeError, "qf_xfm_session_begin"):
                model2._apply_model(x, torch.tensor([1.0]), c_crossattn=ctx, transformer_options=opts)
        slp2.assert_called_once_with(qfm._BUSY_BEGIN_RETRY_S)
        self.assertIsNone(eng2.current_session)

        # Krea2 scratch wrong dtype/device realloc + detach(False/True)
        k_cfg = comfy.supported_models.Krea2({"image_model": "krea2", "disable_unet_model_creation": True})
        qfm.ensure_model_config_attrs(k_cfg)
        k_lib = _FakeLib()
        km = krea2.QFKrea2Model(k_cfg, _FakeEngine(k_lib), device=torch.device("cpu"))
        km.set_video_enhance(False)
        kx = torch.ones((1, int(km.latent_format.latent_channels), 1, 4, 4), dtype=torch.bfloat16)
        kctx = torch.ones((1, 8, 16), dtype=torch.bfloat16)
        km._out = torch.zeros(kx.shape, dtype=torch.float16, device="cpu")
        km._apply_model(kx, torch.tensor([1.0]), c_crossattn=kctx, transformer_options=opts)
        self.assertEqual(km._out.dtype, torch.bfloat16)
        self.assertEqual(km._out.device, kx.device)
        km._out = torch.empty(kx.shape, dtype=torch.bfloat16, device="meta")
        km._apply_model(kx, torch.tensor([1.0]), c_crossattn=kctx, transformer_options=opts)
        self.assertEqual(km._out.device, kx.device)
        km.process_latent_out(kx)
        self.assertIsNone(km._out)

    def test_t12c_check_conds_forwards_cuda_device_id(self):
        model, lib, _ = _make_model()
        c_ch = int(model.latent_format.latent_channels)
        ref = torch.ones((1, c_ch, 4, 4), dtype=torch.bfloat16)
        ctx = torch.ones((1, 8, 16), dtype=torch.bfloat16)

        model.device = torch.device("cuda", 1)
        with mock.patch.object(qfe, "load_lib", return_value=lib) as ll_cuda:
            model.extra_conds(cross_attn=ctx, reference_latents=[ref])
        ll_cuda.assert_called_once_with(device_id=1)

        model.device = torch.device("cpu")
        with mock.patch.object(qfe, "load_lib", return_value=lib) as ll_cpu:
            model.extra_conds(cross_attn=ctx, reference_latents=[ref])
        ll_cpu.assert_called_once_with(device_id=None)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
