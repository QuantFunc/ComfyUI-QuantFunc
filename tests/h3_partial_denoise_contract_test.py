#!/usr/bin/env python3
"""CPU-only behavioral contract for H3 two-stage (double-sampling) sessions and deprecated audio_enhance.

  P1 a full-range stage runs with its own step count and sigmas; audio_enhance is NEVER sent to PC Engine
  P2 an end-trimmed stage and P3 a start-trimmed stage run with their stage-local schedule verbatim
  P4 set_audio_enhance(True) emits a log-once deprecation warning and never transmits audio_enhance to engine
  P6 no schedule, or fewer than 2 sigmas, and P7 non-decreasing schedules fail loud before qf_xfm_session_begin
  P8 two stages: the first closes before the second begins with its own geometry and stage-local counters
  P9 audio_shift follows comfy's ModelSamplingAV (explicit or None -> video shift; <=0 refused)
  P10 begin width/height follow the model's latent_format spatial scale
  L1/L2 loader has no allow_partial_denoise; keeps audio_enhance as positional compat switch; builds real QFH3Model
"""
from contextvars import ContextVar
import ast
import copy
import ctypes
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import torch

_COMFY_ROOT = os.environ.get("COMFY_ROOT")
if _COMFY_ROOT and (Path(_COMFY_ROOT) / "comfy/cli_args.py").is_file():
    sys.path.insert(0, _COMFY_ROOT)
    _argv, sys.argv = sys.argv, [sys.argv[0], "--cpu"]
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        import comfy.cli_args  # noqa: F401
    finally:
        sys.argv = _argv

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
H3_SOURCE = (PLUGIN_ROOT / "qf_h3_modelpatcher.py").read_text(encoding="utf-8")
MIXIN_SOURCE = (PLUGIN_ROOT / "qf_modelpatcher.py").read_text(encoding="utf-8")
LOADER_SOURCE = (PLUGIN_ROOT / "__init__.py").read_text(encoding="utf-8")


def _class_node(source, name):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method_node(source, class_name, method_name):
    for node in _class_node(source, class_name).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return copy.deepcopy(node)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == method_name for t in node.targets):
            return copy.deepcopy(node)
    raise AssertionError(f"method {class_name}.{method_name} not found")


def _module_function(source, name, namespace):
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
            ast.fix_missing_locations(module)
            exec(compile(module, str(PLUGIN_ROOT / "qf_h3_modelpatcher.py"), "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"function {name} not found")


def _standalone_method(source, class_name, method_name, namespace):
    node = _method_node(source, class_name, method_name)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(PLUGIN_ROOT / "qf_modelpatcher.py"), "exec"), namespace)
    return namespace[method_name]


def _subset_class(source, source_name, output_name, base_name, method_names, namespace):
    body = [_method_node(source, source_name, name) for name in method_names]
    node = ast.ClassDef(
        name=output_name,
        bases=[ast.Name(id=base_name, ctx=ast.Load())],
        keywords=[],
        body=body,
        decorator_list=[],
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(PLUGIN_ROOT / "qf_h3_modelpatcher.py"), "exec"), namespace)
    return namespace[output_name]


def _load_qf_engine():
    spec = importlib.util.spec_from_file_location("qfe_h3_partial_contract", PLUGIN_ROOT / "qf_engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_qf_sol():
    spec = importlib.util.spec_from_file_location("qf_sol_h3_partial_contract", PLUGIN_ROOT / "qf_sol.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qfe = _load_qf_engine()
qf_sol = _load_qf_sol()


def _load_real_plugin(test_case):
    comfy_root = os.environ.get("COMFY_ROOT")
    if not comfy_root:
        print("[SKIP] production-path arm requires COMFY_ROOT")
        test_case.skipTest("[SKIP] production-path arm requires COMFY_ROOT")
    comfy_root = Path(comfy_root)
    if not (comfy_root / "comfy/model_management.py").is_file():
        test_case.fail(f"COMFY_ROOT does not contain comfy/model_management.py: {comfy_root}")
    sys.path.insert(0, str(comfy_root))
    package_name = "qf_h3_partial_production_contract"
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_ROOT / "__init__.py",
        submodule_search_locations=[str(PLUGIN_ROOT)],
    )
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = plugin
    spec.loader.exec_module(plugin)
    test_case.assertTrue(getattr(plugin, "_IMPORT_OK", False))
    return plugin


def _module_assign(source, name, namespace):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            module = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
            ast.fix_missing_locations(module)
            exec(compile(module, str(PLUGIN_ROOT / "qf_modelpatcher.py"), "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"assign {name} not found")


def _module_class(source, name, namespace, *, bases=()):
    node = copy.deepcopy(_class_node(source, name))
    node.decorator_list = []
    if bases:
        node.bases = [ast.Name(id=b, ctx=ast.Load()) for b in bases]
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(PLUGIN_ROOT / "qf_modelpatcher.py"), "exec"), namespace)
    return namespace[name]


class _HarnessBase:
    def process_latent_out(self, latent):
        self._base_process_calls += 1
        return latent


class _CondStub:
    def __init__(self, cond):
        self.cond = cond


class _NestedStub:
    is_nested = True

    def __init__(self, values):
        self.values = values

    def unbind(self):
        return self.values


_comfy_stub = SimpleNamespace(
    conds=SimpleNamespace(CONDRegular=_CondStub, CONDConstant=_CondStub),
    nested_tensor=SimpleNamespace(NestedTensor=_NestedStub),
)
_mixin_namespace = {
    "ctypes": ctypes, "json": json, "math": math, "os": os, "torch": torch, "weakref": weakref,
    "ContextVar": ContextVar, "NamedTuple": __import__("typing").NamedTuple,
    "time": SimpleNamespace(sleep=lambda _seconds: None),
    "comfy": _comfy_stub, "qfe": qfe, "qf_sol": qf_sol,
    "_CAT_KEEP_DEFAULT": 0.80,
    "_QF_RUN_ID": ContextVar("quantfunc_run_id", default=None),
    "_QF_SAMPLING_GEOMETRY": ContextVar("quantfunc_sampling_geometry", default=None),
    "_interrupt_poll_end_session_on_raise": lambda _engine: None,
}
_module_assign(MIXIN_SOURCE, "_BUSY_BEGIN_RETRY_S", _mixin_namespace)
_module_function(MIXIN_SOURCE, "_qf_dtype", _mixin_namespace)
_module_class(MIXIN_SOURCE, "_AVPackedMeta", _mixin_namespace)
for _cattr in ("_SCRATCH_ATTRS", "_ENGINE_IGNORED_COND_KEYS", "_TAG", "_REFUSED_HINT"):
    setattr(_HarnessBase, _cattr, ast.literal_eval(
        next(n.value for n in _class_node(MIXIN_SOURCE, "QFSessionModelMixin").body
             if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == _cattr for t in n.targets))))
for _mname in ("set_attn_backend", "set_video_enhance", "_effective_cat_ratio", "dial_opts", "residency_opts",
               "_sigma_step_index", "_call_denoise_step_av", "_stage_schedule",
               "_reset_run_cond_state", "_begin_extra_conds_guard", "_reject_ignored_conds",
               "_ensure_scratch", "_release_scratch_buffers", "_open_native_session",
               "_close_session_on_latent_out"):
    setattr(_HarnessBase, _mname, _standalone_method(
        MIXIN_SOURCE, "QFSessionModelMixin", _mname, _mixin_namespace))

_mixin_namespace["_HarnessBase"] = _HarnessBase
_HarnessAVBase = _subset_class(
    MIXIN_SOURCE, "QFAVSessionModel", "_HarnessAVBase", "_HarnessBase",
    ("_unpack_av_latent", "_execute_av_group_step", "_av_repack_output", "_apply_model", "process_latent_out"),
    _mixin_namespace,
)


class _ContextKeys:
    def reset(self):
        self._keys = {}

    def key(self, value):
        if value not in self._keys:
            self._keys[value] = len(self._keys) + 1
        return self._keys[value]


class _FakeLib:
    def __init__(self):
        self.events = []
        self._next_token = 0x1000

    @property
    def begin_calls(self):
        return sum(kind == "begin" for kind, _ in self.events)

    def qf_xfm_session_begin(self, _xfm, params_ref, session_ref):
        params = params_ref._obj
        token = self._next_token
        self._next_token += 1
        session_ref._obj.value = token
        options = json.loads(params.options_json.decode("utf-8"))
        self.events.append(("begin", {
            "token": token,
            "width": int(params.width),
            "height": int(params.height),
            "num_steps": int(params.num_steps),
            "max_context_dims": tuple(params.max_context_dims),
            "cond_dtype": int(params.cond_dtype),
            "options": options,
        }))
        return qfe.QF_OK

    def qf_xfm_step_av(self, session, params_ref):
        params = params_ref._obj
        base = params.base
        video_value = float(base.step_index) + 0.25
        audio_value = float(base.step_index) + 0.75
        self._write_float_buffer(base.velocity_out, base.velocity_out_capacity, video_value)
        self._write_float_buffer(params.audio_velocity_out, params.audio_velocity_out_capacity, audio_value)
        self.events.append(("step", {
            "token": int(session.value),
            "sigma": float(base.sigma),
            "step_index": int(base.step_index),
            "total_steps": int(base.total_steps),
            "dims": tuple(base.dims),
            "audio_dims": tuple(params.audio_dims),
            "audio_velocity_out_capacity": int(params.audio_velocity_out_capacity),
            "audio_reference_active": int(params.audio_reference_active),
            "video_value": video_value,
            "audio_value": audio_value,
        }))
        return qfe.QF_OK

    def qf_xfm_session_end(self, session):
        self.events.append(("close", {"token": int(session.value)}))
        return qfe.QF_OK

    @staticmethod
    def _write_float_buffer(address, capacity, value):
        item_size = ctypes.sizeof(ctypes.c_float)
        if not address or capacity <= 0 or capacity % item_size:
            raise AssertionError(f"invalid float output buffer address={address} capacity={capacity}")
        output = (ctypes.c_float * (capacity // item_size)).from_address(int(address))
        for index in range(len(output)):
            output[index] = value


class _FakeEngine:
    def __init__(self, lib):
        self.lib = lib
        self.xfm = ctypes.c_void_p(0xCAFE)
        self.current_session = None
        self.step_count = 0
        self.sampler_step_count = 0

    def end_session_if_open(self):
        if self.current_session is None:
            return False, True
        status = self.lib.qf_xfm_session_end(self.current_session)
        if status == qfe.QF_OK:
            self.current_session = None
            return True, True
        return True, False


def _patchify_video(value):
    return value.reshape(-1)


def _unpatchify_video(value, time, height, width, channels, patch_size=(1, 2, 2)):
    pt, ph, pw = patch_size
    return value.reshape(1, channels, time * pt, height * ph, width * pw)


def _pack_audio(value):
    return value[0].permute(1, 2, 0).reshape(-1, value.shape[1])


def _unpack_audio(value):
    time = value.shape[0] // 2
    return value.reshape(2, time, value.shape[1]).permute(2, 0, 1).unsqueeze(0)


class _Log:
    def __init__(self):
        self.warnings, self.infos = [], []

    def warning(self, message, *args):
        self.warnings.append(message % args if args else message)

    def info(self, message, *args):
        self.infos.append(message % args if args else message)


_h3_namespace = {
    "_HarnessAVBase": _HarnessAVBase,
    "ctypes": ctypes,
    "json": json,
    "math": math,
    "time": SimpleNamespace(sleep=lambda _seconds: None),
    "torch": torch,
    "comfy": _comfy_stub,
    "qfe": qfe,
    "qfmp": SimpleNamespace(
        refuse_all_zero_initial_latent=_module_function(MIXIN_SOURCE, "refuse_all_zero_initial_latent", {"RuntimeError": RuntimeError}),
        _AVPackedMeta=_mixin_namespace["_AVPackedMeta"],
    ),
    "_interrupt_poll_end_session_on_raise": lambda _engine: None,
    "_qf_dtype": _mixin_namespace["_qf_dtype"],
    "patchify_video": _patchify_video,
    "unpatchify_video": _unpatchify_video,
    "pack_audio": _pack_audio,
    "unpack_audio": _unpack_audio,
    "_AUDIO_ENHANCE_DEPRECATION_WARNED": False,
    "_log": _Log(),
}
try:
    from comfy_extras import nodes_minimax_h3 as _H3_NODES
except ImportError:
    def _align(n):
        while n % 17 != 5:
            n += 1
        return n
    _H3_NODES = SimpleNamespace(FPS=24, align_frame_count=_align,
                                video_latent_t=lambda fc: 2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2)
_h3_namespace["_h3_nodes"] = lambda: _H3_NODES
_h3_namespace["_H3_PATCH"] = (1, 2, 2)
_h3_namespace["_h3_frames_from_latent_t"] = _module_function(H3_SOURCE, "_h3_frames_from_latent_t", _h3_namespace)

H3ContractModel = _subset_class(
    H3_SOURCE, "QFH3Model", "H3ContractModel", "_HarnessAVBase",
    ("_TAG", "_NO_COND_HINT", "_ENGINE_IGNORED_COND_KEYS",
     "set_audio_enhance", "extra_conds", "_derive_geometry", "_begin",
     "_av_unpack_and_validate", "_av_prepare_session", "_av_step_tensors"),
    _h3_namespace,
)


class _Sampling:
    sigma_max = 1.0
    shift = 1.2
    audio_shift = 3.0
    audio_scale = 1.0

    @staticmethod
    def calculate_denoised(_sigma, velocity, _latent):
        return velocity


def _new_model(audio_enhance=False):
    lib = _FakeLib()
    model = H3ContractModel.__new__(H3ContractModel)
    model._qf = _FakeEngine(lib)
    model.model_sampling = _Sampling()
    model._ctx_key_assigner = _ContextKeys()
    model._ctx_key_assigner.reset()
    model._num_steps = 0
    model._num_frames = 0
    model._fps = float(_H3_NODES.FPS)
    model.latent_format = SimpleNamespace(spacial_downscale_ratio=16, latent_channels=32)
    model._audio_enhance = False
    model._step_i = 0
    model._sess_denoise = 0
    model._out = None
    model._out_video = None
    model._out_audio = None
    model._max_ctx_seq = 0
    model._base_process_calls = 0
    model._qf_needs_begin = False
    model._qf_conds_done = False
    model._qf_conds_run_id = None
    model._qf_conds_noise_ref = None
    model._qf_av_geom = None
    _h3_namespace["_log"].warnings.clear()
    _h3_namespace["_log"].infos.clear()
    _h3_namespace["_AUDIO_ENHANCE_DEPRECATION_WARNED"] = False
    model.set_video_enhance(False)
    model.set_audio_enhance(audio_enhance)
    return model, lib


def _stage_inputs(model, video_shape):
    audio_shape = (1, 32, 2, 3)
    model.latent_shapes = [list(video_shape), list(audio_shape)]
    count = int(torch.tensor(video_shape[1:]).prod()) + int(torch.tensor(audio_shape[1:]).prod())
    packed = torch.arange(1, count + 1, dtype=torch.float32).reshape(1, 1, count)
    context = torch.ones((1, 3, 4), dtype=torch.float32)
    return packed, context


def _run_stage(model, sigmas, video_shape=(1, 24, 2, 2, 2), outputs=None):
    packed, context = _stage_inputs(model, video_shape)
    schedule = torch.tensor(sigmas, dtype=torch.float32)
    options = {"sample_sigmas": schedule, "cond_or_uncond": [0], "uuids": ["cond"]}
    for sigma in sigmas[:-1]:
        packed = model._apply_model(
            packed, torch.tensor([sigma], dtype=torch.float32),
            c_crossattn=context, transformer_options=options)
        if outputs is not None:
            outputs.append(packed.detach().clone())
    return packed


def _events(lib, kind):
    return [payload for event_kind, payload in lib.events if event_kind == kind]


def _warnings():
    return list(_h3_namespace["_log"].warnings)


FULL = [1.0, 0.7, 0.2, 0.0]
END_TRIM = [1.0, 0.8, 0.4]
START_TRIM = [0.8, 0.3, 0.0]


class H3TwoStageContract(unittest.TestCase):
    def assert_step_trace(self, lib, sigmas):
        steps = _events(lib, "step")
        self.assertEqual([step["step_index"] for step in steps], list(range(len(sigmas) - 1)))
        self.assertEqual([step["total_steps"] for step in steps], [len(sigmas) - 1] * (len(sigmas) - 1))
        for actual, expected in zip((step["sigma"] for step in steps), sigmas[:-1]):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_p1_full_range_never_sends_deprecated_audio_enhance(self):
        for audio in (False, True):
            with self.subTest(audio_enhance=audio):
                model, lib = _new_model(audio)
                _run_stage(model, FULL)
                begin = _events(lib, "begin")[0]
                self.assertEqual(begin["num_steps"], len(FULL) - 1)
                self.assert_step_trace(lib, FULL)
                self.assertNotIn("audio_enhance", begin["options"])
                if audio:
                    self.assertEqual(len(_warnings()), 1)
                    self.assertIn("deprecated", _warnings()[0])
                    model.set_audio_enhance(True)
                    self.assertEqual(len(_warnings()), 1)  # log-once deprecation
                else:
                    self.assertEqual(_warnings(), [])

    def test_p2_p3_trimmed_stages_run_verbatim(self):
        for sigmas in (END_TRIM, START_TRIM):
            with self.subTest(sigmas=sigmas):
                model, lib = _new_model()
                _run_stage(model, sigmas)
                self.assertEqual(_events(lib, "begin")[0]["num_steps"], len(sigmas) - 1)
                self.assert_step_trace(lib, sigmas)
                self.assertEqual(_warnings(), [])

    def test_p4_audio_enhance_on_matches_off_in_partial_stages(self):
        for sigmas in (END_TRIM, START_TRIM):
            with self.subTest(sigmas=sigmas):
                runs = []
                for audio in (False, True):
                    model, lib = _new_model(audio)
                    outputs = []
                    packed = _run_stage(model, sigmas, outputs=outputs)
                    runs.append((lib.events, outputs, model.process_latent_out(packed)))
                (off_events, off_outputs, off_latent), (on_events, on_outputs, on_latent) = runs
                self.assertEqual(on_events, off_events)
                for off, on in zip(off_outputs, on_outputs):
                    self.assertTrue(torch.equal(off, on))
                self.assertTrue(torch.equal(off_latent, on_latent))

    def test_p6_missing_or_short_schedule_fails_loud_without_begin(self):
        for schedule in (None, [], [1.0]):
            with self.subTest(schedule=schedule):
                model, lib = _new_model()
                packed, context = _stage_inputs(model, (1, 24, 2, 2, 2))
                options = {} if schedule is None else {"sample_sigmas": torch.tensor(schedule)}
                with self.assertRaisesRegex(RuntimeError, "qf_native H3: the sampler did not publish a sigma schedule"):
                    model._apply_model(packed, torch.tensor([1.0]), c_crossattn=context, transformer_options=options)
                self.assertEqual(lib.begin_calls, 0)

    def test_p7_non_decreasing_schedule_fails_loud_without_begin(self):
        for schedule in ([0.4, 0.8], [0.5, 0.5]):
            with self.subTest(schedule=schedule):
                model, lib = _new_model()
                packed, context = _stage_inputs(model, (1, 24, 2, 2, 2))
                with self.assertRaisesRegex(RuntimeError, "qf_native H3: sigma schedule must be strictly DECREASING"):
                    model._apply_model(packed, torch.tensor([schedule[0]]), c_crossattn=context,
                                       transformer_options={"sample_sigmas": torch.tensor(schedule)})
                self.assertEqual(lib.begin_calls, 0)

    def test_p9_audio_shift_follows_comfy_model_sampling(self):
        for audio_shift, want in ((3.0, 3.0), (None, _Sampling.shift)):
            with self.subTest(audio_shift=audio_shift):
                model, lib = _new_model()
                model.model_sampling.audio_shift = audio_shift
                _run_stage(model, FULL)
                options = _events(lib, "begin")[0]["options"]
                self.assertEqual(options["av_sigma_shift_video"], _Sampling.shift)
                self.assertEqual(options["av_sigma_shift_audio"], want)
        for shift, audio_shift in ((1.2, 0.0), (0.0, 3.0), (1.2, -1.0)):
            with self.subTest(shift=shift, audio_shift=audio_shift):
                model, lib = _new_model()
                model.model_sampling.shift, model.model_sampling.audio_shift = shift, audio_shift
                packed, context = _stage_inputs(model, (1, 24, 2, 2, 2))
                with self.assertRaisesRegex(RuntimeError, "qf_native H3: shift="):
                    model._apply_model(packed, torch.tensor([1.0]), c_crossattn=context,
                                       transformer_options={"sample_sigmas": torch.tensor(FULL)})
                self.assertEqual(lib.begin_calls, 0)

    def test_p10_begin_size_follows_latent_format(self):
        model, lib = _new_model()
        model.latent_format = SimpleNamespace(spacial_downscale_ratio=7, latent_channels=32)
        _run_stage(model, FULL, video_shape=(1, 24, 2, 4, 6))
        begin = _events(lib, "begin")[0]
        self.assertEqual((begin["width"], begin["height"]), (6 * 7, 4 * 7))

    def test_p11_attention_backend_passes_through_without_legacy_auto_rewrite(self):
        for chosen in ("auto", "int8", "i4xi8", "int4", "fp16"):
            with self.subTest(chosen=chosen):
                model, lib = _new_model()
                model.set_attn_backend(chosen)
                _run_stage(model, FULL)
                self.assertEqual([b["options"]["attention_backend"] for b in _events(lib, "begin")], [chosen])

    def test_p8_two_stages_close_then_rebegin(self):
        model, lib = _new_model(True)
        stage_a = _run_stage(model, END_TRIM, (1, 24, 2, 2, 2))
        model.process_latent_out(stage_a)
        stage_b = _run_stage(model, START_TRIM, (1, 24, 2, 4, 6))
        begins = _events(lib, "begin")
        self.assertEqual([(item["width"], item["height"]) for item in begins], [(32, 32), (96, 64)])
        self.assertEqual([item["options"].get("audio_enhance") for item in begins], [None, None])
        self.assertNotEqual(begins[0]["token"], begins[1]["token"])
        close_a = next(i for i, e in enumerate(lib.events) if e[0] == "close" and e[1]["token"] == begins[0]["token"])
        begin_b = next(i for i, e in enumerate(lib.events) if e[0] == "begin" and e[1]["token"] == begins[1]["token"])
        self.assertLess(close_a, begin_b)
        model.process_latent_out(stage_b)
        self.assertEqual([item["token"] for item in _events(lib, "close")], [begins[0]["token"], begins[1]["token"]])

    def test_p12_cfg_batch2_sends_per_slice_audio_dims(self):
        model, lib = _new_model()
        video_shape = (1, 24, 2, 2, 2)
        audio_shape = (1, 32, 2, 3)
        model.latent_shapes = [list(video_shape), list(audio_shape)]
        count = int(torch.tensor(video_shape[1:]).prod()) + int(torch.tensor(audio_shape[1:]).prod())
        packed = torch.arange(1, 2 * count + 1, dtype=torch.float32).reshape(2, 1, count)
        context = torch.ones((2, 3, 4), dtype=torch.float32)
        schedule = torch.tensor([1.0, 0.5], dtype=torch.float32)
        options = {"sample_sigmas": schedule, "cond_or_uncond": [0, 1], "uuids": ["pos", "neg"]}
        out = model._apply_model(
            packed, torch.tensor([1.0, 1.0], dtype=torch.float32),
            c_crossattn=context, transformer_options=options)
        self.assertEqual(tuple(out.shape), tuple(packed.shape))
        steps = _events(lib, "step")
        self.assertEqual(len(steps), 2)
        want_audio_cap = 1 * (audio_shape[2] * audio_shape[3]) * audio_shape[1] * ctypes.sizeof(ctypes.c_float)
        for step in steps:
            self.assertEqual(step["dims"][0], 1)
            self.assertEqual(step["audio_dims"], (1, audio_shape[1], audio_shape[2], audio_shape[3]))
            self.assertEqual(step["audio_velocity_out_capacity"], want_audio_cap)

    def test_t13_extra_conds_error_rollback_and_presession_abort(self):
        model, lib = _new_model()
        ctx_512 = torch.ones((1, 512, 4), dtype=torch.float32)
        ctx_32 = torch.ones((1, 32, 4), dtype=torch.float32)

        model.extra_conds(cross_attn=ctx_512)
        self.assertEqual(model._max_ctx_seq, 512)
        with self.assertRaisesRegex(RuntimeError, "unknown minimax_refs"):
            model.extra_conds(cross_attn=ctx_32, minimax_refs=[{"kind": "bad"}])
        self.assertEqual(model._max_ctx_seq, 0)

        model.extra_conds(cross_attn=ctx_512)
        self.assertEqual(model._max_ctx_seq, 512)
        packed_ok, _ = _stage_inputs(model, (1, 24, 2, 2, 2))
        packed_zero = torch.zeros_like(packed_ok)
        opts = {"sample_sigmas": torch.tensor(FULL, dtype=torch.float32), "cond_or_uncond": [0], "uuids": ["c"]}
        with self.assertRaisesRegex(RuntimeError, "ALL ZEROS"):
            model._apply_model(packed_zero, torch.tensor([1.0]), c_crossattn=ctx_32, transformer_options=opts)
        self.assertEqual(lib.begin_calls, 0)
        model.extra_conds(cross_attn=ctx_32)
        self.assertEqual(model._max_ctx_seq, 32)

    def test_t16_h3_scratch_dtype_device_realloc_release_and_av_guards(self):
        model, _ = _new_model()
        packed, ctx = _stage_inputs(model, (1, 24, 2, 2, 2))
        opts = {"sample_sigmas": torch.tensor(FULL, dtype=torch.float32), "cond_or_uncond": [0], "uuids": ["c"]}
        v_real = (1, 2, 24 * 1 * 2 * 2)
        a_real = (1, 2 * 3, 32)

        for bad_v, bad_a in (
            (torch.zeros(v_real, dtype=torch.float16), torch.zeros(a_real, dtype=torch.float16)),
            (torch.empty(v_real, dtype=torch.float32, device="meta"),
             torch.empty(a_real, dtype=torch.float32, device="meta")),
        ):
            model._out_video = bad_v
            model._out_audio = bad_a
            model._apply_model(packed, torch.tensor([1.0]), c_crossattn=ctx, transformer_options=opts)
            self.assertEqual(model._out_video.dtype, torch.float32)
            self.assertEqual(model._out_audio.dtype, torch.float32)
            self.assertEqual(model._out_video.device, packed.device)
            self.assertEqual(model._out_audio.device, packed.device)
            self.assertIsNot(model._out_video, bad_v)
            self.assertIsNot(model._out_audio, bad_a)

        with self.assertRaisesRegex(RuntimeError, "no c_crossattn"):
            model._apply_model(packed, torch.tensor([1.0]), c_crossattn=None, transformer_options=opts)
        with self.assertRaisesRegex(RuntimeError, "ControlNet"):
            model._apply_model(packed, torch.tensor([1.0]), c_crossattn=ctx, control=object(),
                               transformer_options=opts)
        with self.assertRaisesRegex(RuntimeError, "av_conds bind session-wide"):
            model._apply_model(
                packed, torch.tensor([1.0]), c_crossattn=ctx, transformer_options=opts,
                minimax_payload={"keyframes": [{"latent": torch.ones(1, 24, 1, 2, 2)}], "refs": []})

        model.process_latent_out(packed)
        self.assertIsNone(model._out_video)
        self.assertIsNone(model._out_audio)
        self.assertIsNone(model._out)


class H3LoaderContract(unittest.TestCase):
    def test_l1_loader_has_no_partial_denoise_switch(self):
        cls = _class_node(LOADER_SOURCE, "QuantFuncH3Loader")
        load = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load")
        self.assertNotIn("allow_partial_denoise", [a.arg for a in load.args.args + load.args.kwonlyargs])
        self.assertNotIn("allow_partial_denoise", LOADER_SOURCE)
        self.assertNotIn("allow_partial_denoise", H3_SOURCE)


class H3ProductionPathContract(unittest.TestCase):
    def test_l2_old_prompt_input_dropped_and_real_loader_builds_qfh3(self):
        plugin = _load_real_plugin(self)
        loader_class = plugin.NODE_CLASS_MAPPINGS.get("QuantFuncH3Loader")
        self.assertIs(loader_class, plugin.QuantFuncH3Loader)
        self.assertNotIn("allow_partial_denoise", loader_class.INPUT_TYPES().get("optional", {}))
        self.assertIn("audio_enhance", loader_class.INPUT_TYPES().get("optional", {}))
        import execution
        got = execution.get_input_data({"transformer": "t.safetensors", "allow_partial_denoise": True}, loader_class, "1")
        self.assertNotIn("allow_partial_denoise", got[0])

        h3_module = sys.modules[f"{plugin.__name__}.qf_h3_modelpatcher"]
        real_bundle = str(PLUGIN_ROOT / "configs" / "minimax-h3-fl2va")
        engine_create_attempts = []

        def forbid_engine_create(*args, **kwargs):
            engine_create_attempts.append((args, kwargs))
            raise AssertionError("production-path contract must not materialize engine")

        cpu = plugin.qfmp.torch.device("cpu")
        with mock.patch.object(plugin, "_load_model_config", return_value=(real_bundle, {"family": "minimax-h3"})), \
             mock.patch.object(plugin, "_model_config_choices", return_value=["minimax-h3-fl2va"]), \
             mock.patch.object(plugin, "_resolve_transformer",
                               return_value="/contract/minimax-h3-transformer.safetensors"), \
             mock.patch.object(plugin, "_get_engine", side_effect=forbid_engine_create), \
             mock.patch.object(plugin, "_FAMILY_BUILDERS", {}), \
             mock.patch.object(plugin, "_FAMILY_MATCHERS", []), \
             mock.patch.object(plugin.qfmp, "current_torch_device", return_value=(cpu, 0)), \
             mock.patch.object(plugin.qfmp.comfy.model_management, "unet_offload_device", return_value=cpu):
            plugin._register_families()
            patcher = loader_class().load("contract-transformer", "minimax-h3-fl2va", audio_enhance=True)[0]
            self.assertIs(type(patcher), plugin.qfmp.QFModelPatcher)
            self.assertIs(type(patcher.model), h3_module.QFH3Model)
            self.assertIs(patcher.model._audio_enhance, True)

            patcher.model._qf_av_geom = ((24, 1, 2, 2), (32, 1, 3))
            xv = torch.ones((1, 24, 2, 2, 2), dtype=torch.float32)
            xa = torch.ones((1, 32, 2, 3), dtype=torch.float32)
            vemb = torch.ones((1, 3, 4), dtype=torch.float32)
            to = {"sample_sigmas": torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32),
                  "cond_or_uncond": [0], "uuids": ["c"]}
            patcher.model._av_prepare_session(xv, xa, vemb, to, {}, opening=False)
            self.assertEqual(patcher.model._qf_av_geom, ((24, 2, 2, 2), (32, 2, 3)))
            self.assertGreater(patcher.model._num_frames, 0)
        self.assertEqual(engine_create_attempts, [])


if __name__ == "__main__":
    unittest.main()
