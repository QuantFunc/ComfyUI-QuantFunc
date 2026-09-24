#!/usr/bin/env python3
"""CPU-only behavioral contract for H3 partial-denoise sessions (P1-P8)."""
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
    sys.path.insert(0, str(comfy_root))
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
        return {"sparse_cdf": 1.0, "video_enhance": False}

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
        self.unloaded = False
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


def _unpatchify_video(value, time, height, width, channels):
    return value.reshape(1, channels, time, height * 2, width * 2)


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
for _name in ("_H3_SPATIAL", "_H3_FPS", "_H3_VIDEO_CHANNELS", "_H3_AUDIO_CHANNELS", "_H3_AUDIO_STEREO"):
    _h3_namespace[_name] = _literal_assignment(H3_SOURCE, _name)
_h3_namespace["_h3_frames_from_latent_t"] = _module_function(
    H3_SOURCE, "_h3_frames_from_latent_t", _h3_namespace)

H3ContractModel = _subset_class(
    H3_SOURCE,
    "QFH3Model",
    "H3ContractModel",
    "_HarnessBase",
    ("set_allow_partial_denoise", "_derive_geometry", "_begin", "_apply_model", "process_latent_out"),
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


def _new_model(allow_partial):
    lib = _FakeLib()
    model = H3ContractModel.__new__(H3ContractModel)
    model._qf = _FakeEngine(lib)
    model.model_sampling = _Sampling()
    model._ctx_key_assigner = _ContextKeys()
    model._ctx_key_assigner.reset()
    model._num_steps = 0
    model._num_frames = 0
    model._fps = _h3_namespace["_H3_FPS"]
    model._audio_enhance = False
    model._allow_partial_denoise = False
    model._step_i = 0
    model._sess_denoise = 0
    model._out_video = None
    model._out_audio = None
    model._max_ctx_seq = 0
    model._base_process_calls = 0
    model._qf_needs_begin = False
    model.set_allow_partial_denoise(allow_partial)
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


class H3PartialDenoiseContract(unittest.TestCase):
    def assert_step_trace(self, lib, sigmas):
        steps = _events(lib, "step")
        self.assertEqual([step["step_index"] for step in steps], list(range(len(sigmas) - 1)))
        self.assertEqual([step["total_steps"] for step in steps], [len(sigmas) - 1] * (len(sigmas) - 1))
        for actual, expected in zip((step["sigma"] for step in steps), sigmas[:-1]):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_p1_flag_false_full_range_passes_with_full_step_count(self):
        sigmas = [1.0, 0.7, 0.2, 0.0]
        model, lib = _new_model(False)
        _run_stage(model, sigmas)
        self.assertEqual(_events(lib, "begin")[0]["num_steps"], len(sigmas) - 1)
        self.assert_step_trace(lib, sigmas)

    def test_p2_flag_true_does_not_change_full_range_trace_or_returned_output(self):
        sigmas = [1.0, 0.7, 0.2, 0.0]
        traces, step_outputs, returned_latents = [], [], []
        for allow_partial in (False, True):
            model, lib = _new_model(allow_partial)
            outputs = []
            packed = _run_stage(model, sigmas, outputs=outputs)
            returned = model.process_latent_out(packed)
            traces.append(lib.events)
            step_outputs.append(outputs)
            returned_latents.append(returned.detach().clone())
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(len(step_outputs[0]), len(sigmas) - 1)
        for false_output, true_output in zip(step_outputs[0], step_outputs[1]):
            self.assertTrue(torch.equal(false_output, true_output))
            self.assertTrue(torch.isfinite(false_output).all())
        self.assertTrue(torch.equal(returned_latents[0], returned_latents[1]))
        self.assertTrue(torch.equal(returned_latents[0][..., :192],
                                    torch.full_like(returned_latents[0][..., :192], 2.25)))
        self.assertTrue(torch.equal(returned_latents[0][..., 192:],
                                    torch.full_like(returned_latents[0][..., 192:], 2.75)))

    def test_p3_flag_false_rejects_end_trim_before_native_begin(self):
        model, lib = _new_model(False)
        with self.assertRaisesRegex(RuntimeError, "partial / trimmed denoise is disabled"):
            _run_stage(model, [1.0, 0.8, 0.4])
        self.assertEqual(lib.begin_calls, 0)
        self.assertEqual(_events(lib, "step"), [])

    def test_p4_flag_false_rejects_start_trim_before_native_begin(self):
        model, lib = _new_model(False)
        with self.assertRaisesRegex(RuntimeError, "partial / trimmed denoise is disabled"):
            _run_stage(model, [0.8, 0.3, 0.0])
        self.assertEqual(lib.begin_calls, 0)
        self.assertEqual(_events(lib, "step"), [])

    def test_p5_flag_true_end_trim_uses_stage_local_schedule_verbatim(self):
        sigmas = [1.0, 0.8, 0.4]
        model, lib = _new_model(True)
        _run_stage(model, sigmas)
        self.assertEqual(_events(lib, "begin")[0]["num_steps"], 2)
        self.assert_step_trace(lib, sigmas)

    def test_p6_flag_true_start_trim_uses_stage_local_schedule_verbatim(self):
        sigmas = [0.8, 0.3, 0.0]
        model, lib = _new_model(True)
        _run_stage(model, sigmas)
        self.assertEqual(_events(lib, "begin")[0]["num_steps"], 2)
        self.assert_step_trace(lib, sigmas)

    def test_p7_missing_or_short_schedule_fails_loud_without_native_begin(self):
        for allow_partial in (False, True):
            for schedule in (None, [], [1.0]):
                with self.subTest(allow_partial=allow_partial, schedule=schedule):
                    model, lib = _new_model(allow_partial)
                    packed, context = _stage_inputs(model, (1, 24, 2, 2, 2))
                    options = {} if schedule is None else {"sample_sigmas": torch.tensor(schedule)}
                    with self.assertRaisesRegex(RuntimeError, "did not publish a sigma schedule"):
                        model._apply_model(
                            packed,
                            torch.tensor([1.0]),
                            c_crossattn=context,
                            transformer_options=options,
                        )
                    self.assertEqual(lib.begin_calls, 0)

    def test_p8_two_stages_close_then_rebegin_with_new_geometry_tokens_and_counters(self):
        model, lib = _new_model(True)
        stage_a = _run_stage(model, [1.0, 0.8, 0.4], (1, 24, 2, 2, 2))
        model.process_latent_out(stage_a)
        stage_b = _run_stage(model, [0.8, 0.3, 0.0], (1, 24, 2, 4, 6))

        begins = _events(lib, "begin")
        self.assertEqual([(item["width"], item["height"]) for item in begins], [(32, 32), (96, 64)])
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
        self.assertEqual([item["token"] for item in closes],
                         [begins[0]["token"], begins[1]["token"]])
        begin_a = next(i for i, event in enumerate(lib.events)
                       if event[0] == "begin" and event[1]["token"] == begins[0]["token"])
        close_b = next(i for i, event in enumerate(lib.events)
                       if event[0] == "close" and event[1]["token"] == begins[1]["token"])
        self.assertLess(begin_a, close_a)
        self.assertLess(begin_b, close_b)


class _LoaderModel:
    def __init__(self):
        self.partial_values = []

    def set_attn_backend(self, _value):
        pass

    def set_sol_tau(self, _value):
        pass

    def set_audio_enhance(self, _value):
        pass

    def set_quality(self, _value):   # the ONE quality switch every loader must reach (mandatory, unguarded)
        pass

    def set_allow_partial_denoise(self, value):
        self.partial_values.append(value)


_loader_model = _LoaderModel()
_loader_patcher = SimpleNamespace(model=_loader_model)
_loader_namespace = {
    "object": object,
    "_transformer_choices": lambda: ["transformer"],
    "_model_config_choices": lambda **_kwargs: ["config"],
    "_attn_backend_input": lambda default: ([default], {"default": default}),
    "_SOL_TAU_INPUT": ("FLOAT", {"default": 1.0}),
    "_quality_input": lambda: (["balance", "best_quality"], {"default": "balance"}),
    "_QUALITY_LEGACY_HIDDEN": {"quality_enhance": ("BOOLEAN", {})},
    "_loaded_device_index": lambda _patcher: 0,
    "_resolve_quality": lambda *_args: "balance",
    "_apply_quality": lambda model, q, _dev=None: model.set_quality(q),
    "_AUDIO_ENHANCE_INPUT": ("BOOLEAN", {"default": False}),
    "_STEP_CACHE_INPUT": ("FLOAT", {"default": 0.0}),
    "_BLOCK_CACHE_INPUT": ("FLOAT", {"default": 0.0}),
    "_run_family_load": lambda *_args, **_kwargs: _loader_patcher,
    "_attn_backend_to_engine": lambda value: value,
    "_arm_session_caches": lambda *_args: None,
}
QuantFuncH3LoaderContract = _subset_class(
    LOADER_SOURCE,
    "QuantFuncH3Loader",
    "QuantFuncH3LoaderContract",
    "object",
    ("INPUT_TYPES", "load"),
    _loader_namespace,
)


class H3LoaderPartialDenoiseContract(unittest.TestCase):
    def test_extracted_loader_distinguishes_default_false_explicit_false_and_true(self):
        schema = QuantFuncH3LoaderContract.INPUT_TYPES()
        field_type, field_options = schema["optional"]["allow_partial_denoise"]
        self.assertEqual(field_type, "BOOLEAN")
        self.assertIs(field_options["default"], False)
        self.assertIs(
            inspect.signature(QuantFuncH3LoaderContract.load)
            .parameters["allow_partial_denoise"].default,
            False,
        )

        loader = QuantFuncH3LoaderContract()
        cases = (
            ("omitted", {}, False),
            ("explicit_false", {"allow_partial_denoise": False}, False),
            ("explicit_true", {"allow_partial_denoise": True}, True),
        )
        for label, kwargs, expected in cases:
            with self.subTest(label=label):
                _loader_model.partial_values.clear()
                result = loader.load("transformer", "config", **kwargs)
                self.assertEqual(result, (_loader_patcher,))
                self.assertEqual(_loader_model.partial_values, [expected])

    def test_extracted_loader_fails_loud_when_returned_model_lacks_setter(self):
        with mock.patch.object(_loader_patcher, "model", SimpleNamespace()):
            with self.assertRaises(AttributeError):
                QuantFuncH3LoaderContract().load("transformer", "config")


class H3ProductionPathContract(unittest.TestCase):
    def test_registered_loader_builds_real_qfh3_and_forwards_all_boolean_cases(self):
        plugin = _load_real_plugin(self)
        loader_class = plugin.NODE_CLASS_MAPPINGS.get("QuantFuncH3Loader")
        self.assertIs(loader_class, plugin.QuantFuncH3Loader)
        self.assertIs(
            inspect.signature(loader_class.load).parameters["allow_partial_denoise"].default,
            False,
        )

        h3_module = sys.modules[f"{plugin.__name__}.qf_h3_modelpatcher"]
        engine_create_attempts = []

        def forbid_engine_create(*args, **kwargs):
            engine_create_attempts.append((args, kwargs))
            raise AssertionError("production-path contract must not create/materialize a native engine")

        cpu = plugin.qfmp.torch.device("cpu")
        with mock.patch.object(plugin, "_load_model_config", return_value=(
                "/contract/minimax-h3", {"family": "minimax-h3", "dual_expert": False})), \
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

            loader = loader_class()
            cases = (
                ("omitted", {}, False),
                ("explicit_false", {"allow_partial_denoise": False}, False),
                ("explicit_true", {"allow_partial_denoise": True}, True),
            )
            for label, kwargs, expected in cases:
                with self.subTest(label=label):
                    result = loader.load("contract-transformer", "contract-config", **kwargs)
                    self.assertEqual(len(result), 1)
                    patcher = result[0]
                    self.assertIs(type(patcher), plugin.qfmp.QFModelPatcher)
                    self.assertIs(type(patcher.model), h3_module.QFH3Model)
                    self.assertTrue(callable(getattr(patcher.model, "set_allow_partial_denoise", None)))
                    self.assertIs(patcher.model._allow_partial_denoise, expected)

        self.assertEqual(engine_create_attempts, [])


if __name__ == "__main__":
    unittest.main()
