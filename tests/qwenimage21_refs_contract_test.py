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


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
