#!/usr/bin/env python3
"""CPU-only behavioral contract for H3 two-stage (double-sampling) sessions (P1-P8, L1-L2).

User 2026-09-28: double sampling works with no switch, as on LTX (「双采样的开关后 我要还能支持双采样啊 好像LTX一样」), and
audio_enhance does not support it (「audio_enhance不支持双采」).

  P1 a full-range stage runs with its own step count and sigmas; audio_enhance ON is sent, with no warning
  P2 an end-trimmed stage and P3 a start-trimmed stage run with their stage-local schedule, verbatim
  P4 in either partial stage audio_enhance ON is not sent: one warning per session, and the begin, the step trace and
     the output equal the audio_enhance OFF run
  P5 the full-range tolerance: a start at >= 0.98 sigma_max and an end at <= 1e-3 are full range
  P6 no schedule, or fewer than 2 sigmas, and P7 a schedule that is not strictly decreasing fail loud, before any
     native begin (the rule the mixin shares with LTX)
  P8 two stages: the first closes before the second begins with its own geometry and stage-local counters
  L1 the loader has no allow_partial_denoise input or parameter; L2 (COMFY_ROOT) ComfyUI drops that input from an old
     API prompt before the loader runs, and the registered loader builds the real QFH3Model

MUTATION (each goes RED): refuse a partial stage again -> P2/P3/P8; send audio_enhance in a partial stage, or warn
per step / never -> P4; tighten or drop the full-range tolerance -> P5; drop the schedule checks -> P6/P7; re-add the
input -> L1.
"""
import ast
import copy
import ctypes
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


# ComfyUI's own --cpu mode (the contract tests' idiom) for the production-path arm: it needs comfy importable, not a
# GPU, and the CPU suite hides CUDA. comfy parses its args ONCE, when comfy.cli_args is first imported, so force that
# parse here, before any test can import comfy, and give unittest back its own argv.
_COMFY_ROOT = os.environ.get("COMFY_ROOT")
if _COMFY_ROOT and (Path(_COMFY_ROOT) / "comfy/cli_args.py").is_file():
    sys.path.insert(0, _COMFY_ROOT)
    _argv, sys.argv = sys.argv, [sys.argv[0], "--cpu"]
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        import comfy.cli_args  # noqa: F401 — the one parse, with --cpu
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
            return copy.deepcopy(node)   # a class attribute a method reads (the loader's QF_FAMILY)
    raise AssertionError(f"method {class_name}.{method_name} not found")


def _module_function(source, name, namespace):
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
            ast.fix_missing_locations(module)
            exec(compile(module, str(PLUGIN_ROOT / "qf_h3_modelpatcher.py"), "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"function {name} not found")


def _literal_assignment(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"assignment {name} not found")


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


qfe = _load_qf_engine()


def _load_real_plugin(test_case):
    """Import the shipped package in the remote Comfy test environment.

    A missing local Comfy checkout is a disclosed strict-run skip.  Once COMFY_ROOT is
    supplied, every import/registration failure is a hard failure: the AST harness below
    must never stand in for production-path reachability.
    """
    comfy_root = os.environ.get("COMFY_ROOT")
    if not comfy_root:
        print("[SKIP] production-path arm requires COMFY_ROOT")
        test_case.skipTest("[SKIP] production-path arm requires COMFY_ROOT")
    comfy_root = Path(comfy_root)
    if not (comfy_root / "comfy/model_management.py").is_file():
        test_case.fail(f"COMFY_ROOT does not contain comfy/model_management.py: {comfy_root}")
    sys.path.insert(0, str(comfy_root))   # comfy runs in --cpu mode: forced at module import (top of file)
    package_name = "qf_h3_partial_production_contract"
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_ROOT / "__init__.py",
        submodule_search_locations=[str(PLUGIN_ROOT)],
    )
    test_case.assertIsNotNone(spec)
    test_case.assertIsNotNone(spec.loader)
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = plugin
    try:
        spec.loader.exec_module(plugin)
    except Exception as exc:  # noqa: BLE001 - import reachability is the contract under test
        test_case.fail(f"real plugin package import failed: {type(exc).__name__}: {exc}")
    test_case.assertTrue(
        getattr(plugin, "_IMPORT_OK", False),
        "real plugin package imported with _IMPORT_OK=False",
    )
    return plugin


class _HarnessBase:
    def residency_opts(self):
        return {"sparse_cdf": 1.0, "attention_backend": "auto", "video_enhance": False}

    def process_latent_out(self, latent):
        self._base_process_calls += 1
        return latent


_mixin_namespace = {"ctypes": ctypes, "qfe": qfe, "os": os}
_HarnessBase._sigma_step_index = _standalone_method(
    MIXIN_SOURCE, "QFSessionModelMixin", "_sigma_step_index", _mixin_namespace)
_HarnessBase._call_denoise_step_multi = _standalone_method(
    MIXIN_SOURCE, "QFSessionModelMixin", "_call_denoise_step_multi", _mixin_namespace)


class _ContextKeys:
    def reset(self):
        self._keys = {}

    def key(self, value):
        if value not in self._keys:
            self._keys[value] = len(self._keys) + 1
        return self._keys[value]


class _FakeLib:
    """Python-bridge probe, not an oracle for the native C header ABI.

    The fake deliberately reads only the ctypes fields selected by the current Python
    bridge.  Header layout and real native handle generation remain remote native-fixture
    responsibilities.
    """

    def __init__(self):
        self.events = []
        self._next_token = 0x1000

    @property
    def begin_calls(self):
        return sum(kind == "begin" for kind, _ in self.events)

    def quantfunc_denoise_begin(self, _pipeline, params_ref, session_ref):
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
        return qfe.QUANTFUNC_OK

    def quantfunc_denoise_step_multi(self, session, params_ref):
        params = params_ref._obj
        base = params.base
        video_value = float(base.step_index) + 0.25
        audio_value = float(base.step_index) + 0.75
        self._write_float_buffer(base.velocity_out, base.velocity_out_capacity, video_value)
        self._write_float_buffer(
            params.audio_velocity_out,
            params.audio_velocity_out_capacity,
            audio_value,
        )
        self.events.append(("step", {
            "token": int(session.value),
            "sigma": float(base.sigma),
            "step_index": int(base.step_index),
            "total_steps": int(base.total_steps),
            "dims": tuple(base.dims),
            "audio_dims": tuple(params.audio_dims),
            "video_value": video_value,
            "audio_value": audio_value,
        }))
        return qfe.QUANTFUNC_OK

    def quantfunc_denoise_end(self, session):
        self.events.append(("close", {"token": int(session.value)}))
        return qfe.QUANTFUNC_OK

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
        self.pipeline = ctypes.c_void_p(0xCAFE)
        self.current_session = None
        self.step_count = 0
        self.sampler_step_count = 0

    def end_session_if_open(self):
        if self.current_session is None:
            return False, True
        status = self.lib.quantfunc_denoise_end(self.current_session)
        if status == qfe.QUANTFUNC_OK:
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


_h3_namespace = {
    "_HarnessBase": _HarnessBase,
    "ctypes": ctypes,
    "json": json,
    "math": __import__("math"),
    "time": SimpleNamespace(sleep=lambda _seconds: None),
    "torch": torch,
    "qfe": qfe,
    "qfmp": SimpleNamespace(refuse_all_zero_initial_latent=lambda *_args: None),
    "_interrupt_poll_end_session_on_raise": lambda _engine: None,
    "_qf_dtype": lambda _dtype: 0,
    "patchify_video": _patchify_video,
    "unpatchify_video": _unpatchify_video,
    "pack_audio": _pack_audio,
    "unpack_audio": _unpack_audio,
}
try:   # ComfyUI's stock H3 nodes: the frame grid and its FPS (the plugin reads them from there, never restates them)
    from comfy_extras import nodes_minimax_h3 as _H3_NODES
except ImportError:   # no ComfyUI on this box: a stub with comfy's behaviour, only so the CPU-only arms can run
    def _align(n):
        while n % 17 != 5:
            n += 1
        return n
    _H3_NODES = SimpleNamespace(FPS=24, align_frame_count=_align,
                                video_latent_t=lambda fc: 2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2)
_h3_namespace["_h3_nodes"] = lambda: _H3_NODES
_h3_namespace["_H3_PATCH"] = (1, 2, 2)   # comfy's patchify_video default (the plugin reads it from comfy's signature)
_h3_namespace["_h3_frames_from_latent_t"] = _module_function(
    H3_SOURCE, "_h3_frames_from_latent_t", _h3_namespace)

class _Log:
    def __init__(self):
        self.warnings = []

    def warning(self, message, *args):
        self.warnings.append(message % args if args else message)


_HarnessBase._stage_schedule = _standalone_method(
    MIXIN_SOURCE, "QFSessionModelMixin", "_stage_schedule", _mixin_namespace)
for _name in ("_H3_DENOISED_SIGMA", "_H3_FULL_START", "_AUDIO_ENHANCE_TWO_STAGE"):
    _h3_namespace[_name] = _literal_assignment(H3_SOURCE, _name)
_h3_namespace["_log"] = _Log()
WARNING = _h3_namespace["_AUDIO_ENHANCE_TWO_STAGE"]

H3ContractModel = _subset_class(
    H3_SOURCE,
    "QFH3Model",
    "H3ContractModel",
    "_HarnessBase",
    ("_derive_geometry", "_begin", "_apply_model", "process_latent_out"),
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
    model.latent_format = SimpleNamespace(spacial_downscale_ratio=16, latent_channels=32)   # comfy's MiniMaxH3AV format
    model._audio_enhance = audio_enhance
    model._stage_partial = False
    model._step_i = 0
    model._sess_denoise = 0
    model._out_video = None
    model._out_audio = None
    model._max_ctx_seq = 0
    model._base_process_calls = 0
    model._qf_needs_begin = False
    _h3_namespace["_log"].warnings.clear()
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
            packed,
            torch.tensor([sigma], dtype=torch.float32),
            c_crossattn=context,
            transformer_options=options,
        )
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

    def test_p1_full_range_runs_with_its_steps_and_sends_audio_enhance(self):
        for audio in (False, True):
            with self.subTest(audio_enhance=audio):
                model, lib = _new_model(audio)
                _run_stage(model, FULL)
                begin = _events(lib, "begin")[0]
                self.assertEqual(begin["num_steps"], len(FULL) - 1)
                self.assert_step_trace(lib, FULL)
                self.assertIs(begin["options"].get("audio_enhance"), True if audio else None)
                self.assertEqual(_warnings(), [])

    def test_p2_p3_a_trimmed_stage_runs_its_own_schedule_verbatim(self):
        for sigmas in (END_TRIM, START_TRIM):
            with self.subTest(sigmas=sigmas):
                model, lib = _new_model()
                _run_stage(model, sigmas)
                self.assertEqual(_events(lib, "begin")[0]["num_steps"], len(sigmas) - 1)
                self.assert_step_trace(lib, sigmas)
                self.assertEqual(_warnings(), [])

    def test_p4_audio_enhance_is_ignored_in_a_partial_stage_with_one_warning(self):
        for sigmas in (END_TRIM, START_TRIM):
            with self.subTest(sigmas=sigmas):
                runs = []
                for audio in (False, True):
                    model, lib = _new_model(audio)
                    outputs = []
                    packed = _run_stage(model, sigmas, outputs=outputs)
                    runs.append((lib.events, outputs, model.process_latent_out(packed), _warnings()))
                (off_events, off_outputs, off_latent, off_warn), (on_events, on_outputs, on_latent, on_warn) = runs
                self.assertNotIn("audio_enhance", _events_of(on_events, "begin")[0]["options"])
                self.assertEqual(on_events, off_events)
                self.assertEqual(len(on_outputs), len(sigmas) - 1)
                for off, on in zip(off_outputs, on_outputs):
                    self.assertTrue(torch.equal(off, on))
                self.assertTrue(torch.equal(off_latent, on_latent))
                self.assertEqual(off_warn, [])
                self.assertEqual(on_warn, [WARNING])            # once per session, not per step
        self.assertIn("audio_enhance is not supported with two-stage (double-sampling) workflows", WARNING)

    def test_p5_full_range_tolerance(self):
        cases = ((([0.985, 0.5, 0.0]), True), (([1.0, 0.5, 5e-4]), True),
                 (([0.97, 0.5, 0.0]), False), (([1.0, 0.5, 2e-3]), False))
        for sigmas, full in cases:
            with self.subTest(sigmas=sigmas):
                model, lib = _new_model(True)
                _run_stage(model, sigmas)
                self.assertIs(_events(lib, "begin")[0]["options"].get("audio_enhance"), True if full else None)
                self.assertEqual(_warnings(), [] if full else [WARNING])

    def test_p6_missing_or_short_schedule_fails_loud_without_native_begin(self):
        for schedule in (None, [], [1.0]):
            with self.subTest(schedule=schedule):
                model, lib = _new_model()
                packed, context = _stage_inputs(model, (1, 24, 2, 2, 2))
                options = {} if schedule is None else {"sample_sigmas": torch.tensor(schedule)}
                with self.assertRaisesRegex(RuntimeError, "qf_native H3: the sampler did not publish a sigma schedule"):
                    model._apply_model(packed, torch.tensor([1.0]), c_crossattn=context, transformer_options=options)
                self.assertEqual(lib.begin_calls, 0)

    def test_p7_a_schedule_that_is_not_decreasing_fails_loud_without_native_begin(self):
        for schedule in ([0.4, 0.8], [0.5, 0.5]):
            with self.subTest(schedule=schedule):
                model, lib = _new_model()
                packed, context = _stage_inputs(model, (1, 24, 2, 2, 2))
                with self.assertRaisesRegex(RuntimeError, "qf_native H3: sigma schedule must be strictly DECREASING"):
                    model._apply_model(packed, torch.tensor([schedule[0]]), c_crossattn=context,
                                       transformer_options={"sample_sigmas": torch.tensor(schedule)})
                self.assertEqual(lib.begin_calls, 0)

    def test_p8_two_stages_close_then_rebegin_with_new_geometry_tokens_and_counters(self):
        model, lib = _new_model(True)
        stage_a = _run_stage(model, END_TRIM, (1, 24, 2, 2, 2))
        model.process_latent_out(stage_a)
        stage_b = _run_stage(model, START_TRIM, (1, 24, 2, 4, 6))
        self.assertEqual(_warnings(), [WARNING, WARNING])       # each stage says it once

        begins = _events(lib, "begin")
        self.assertEqual([(item["width"], item["height"]) for item in begins], [(32, 32), (96, 64)])
        self.assertEqual([item["options"].get("audio_enhance") for item in begins], [None, None])
        # These are deterministic fake-issued begin tokens, not evidence about native handles.
        self.assertNotEqual(begins[0]["token"], begins[1]["token"])

        close_a = next(i for i, event in enumerate(lib.events)
                       if event[0] == "close" and event[1]["token"] == begins[0]["token"])
        begin_b = next(i for i, event in enumerate(lib.events)
                       if event[0] == "begin" and event[1]["token"] == begins[1]["token"])
        self.assertLess(close_a, begin_b)

        stage_b_steps = [item for item in _events(lib, "step") if item["token"] == begins[1]["token"]]
        self.assertEqual([item["step_index"] for item in stage_b_steps], [0, 1])
        self.assertEqual([item["total_steps"] for item in stage_b_steps], [2, 2])
        model.process_latent_out(stage_b)
        self.assertEqual(model._step_i, 2)
        self.assertEqual(model._sess_denoise, 2)

        closes = _events(lib, "close")
        self.assertEqual([item["token"] for item in closes], [begins[0]["token"], begins[1]["token"]])


def _events_of(events, kind):
    return [payload for event_kind, payload in events if event_kind == kind]


class H3LoaderContract(unittest.TestCase):
    def test_l1_the_loader_has_no_partial_denoise_switch(self):
        cls = _class_node(LOADER_SOURCE, "QuantFuncH3Loader")
        load = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load")
        self.assertNotIn("allow_partial_denoise", [a.arg for a in load.args.args + load.args.kwonlyargs])
        self.assertNotIn("allow_partial_denoise", LOADER_SOURCE)
        self.assertNotIn("allow_partial_denoise", H3_SOURCE)


class H3ProductionPathContract(unittest.TestCase):
    def test_l2_an_old_prompt_input_is_dropped_and_the_real_loader_builds_qfh3(self):
        plugin = _load_real_plugin(self)
        loader_class = plugin.NODE_CLASS_MAPPINGS.get("QuantFuncH3Loader")
        self.assertIs(loader_class, plugin.QuantFuncH3Loader)
        self.assertNotIn("allow_partial_denoise", loader_class.INPUT_TYPES().get("optional", {}))
        import execution    # ComfyUI's own input gathering: an input the node does not declare never reaches load()
        got = execution.get_input_data({"transformer": "t.safetensors", "allow_partial_denoise": True}, loader_class, "1")
        self.assertNotIn("allow_partial_denoise", got[0])

        h3_module = sys.modules[f"{plugin.__name__}.qf_h3_modelpatcher"]
        engine_create_attempts = []

        def forbid_engine_create(*args, **kwargs):
            engine_create_attempts.append((args, kwargs))
            raise AssertionError("production-path contract must not create/materialize a native engine")

        cpu = plugin.qfmp.torch.device("cpu")
        with mock.patch.object(plugin, "_load_model_config", return_value=(
                "/contract/minimax-h3", {"family": "minimax-h3"})), \
             mock.patch.object(plugin, "_model_config_choices", return_value=["contract-config"]), \
             mock.patch.object(plugin, "_resolve_transformer",
                               return_value="/contract/minimax-h3-transformer.safetensors"), \
             mock.patch.object(plugin, "_get_engine", side_effect=forbid_engine_create), \
             mock.patch.object(plugin, "_FAMILY_BUILDERS", {}), \
             mock.patch.object(plugin, "_FAMILY_MATCHERS", []), \
             mock.patch.object(plugin.qfmp, "stage_denoise_only_package",
                               return_value="/contract/staged-minimax-h3"), \
             mock.patch.object(plugin.qfmp, "current_torch_device", return_value=(cpu, 0)), \
             mock.patch.object(plugin.qfmp.comfy.model_management,
                               "unet_offload_device", return_value=cpu):
            plugin._register_families()
            builder = plugin._FAMILY_BUILDERS.get("minimax-h3")
            self.assertIsNotNone(builder)
            self.assertEqual(builder.__module__, h3_module.__name__)
            patcher = loader_class().load("contract-transformer", "contract-config", audio_enhance=True)[0]
            self.assertIs(type(patcher), plugin.qfmp.QFModelPatcher)
            self.assertIs(type(patcher.model), h3_module.QFH3Model)
            self.assertIs(patcher.model._audio_enhance, True)
            self.assertIs(patcher.model._stage_partial, False)

        self.assertEqual(engine_create_attempts, [])


if __name__ == "__main__":
    unittest.main()
