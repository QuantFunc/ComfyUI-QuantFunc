#!/usr/bin/env python3
"""Unified in-place LoRA hot-swap contract for QuantFunc PC Engine (qf_xfm_set_loras).

PC Engine retires the legacy BF16 load_time rebuild path: every LoRA stack shares ONE pure
transformer handle and hot-swaps in place via qf_xfm_set_loras(xfm, json_array).
"""
import ast
import contextlib
import contextvars
import copy
import json
import os
import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace as NS

Q = Path(__file__).resolve().parents[1]


def function(path, name):
    return next(n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(n, ast.FunctionDef) and n.name == name)


def install(nodes, env):
    exec(compile(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]), "actual_candidate_ast", "exec"), env)


class Lazy:
    def __init__(self, factory, **kwargs):
        self.factory = factory
        self._lora = []


class Model:
    def __init__(self, engine):
        self._qf = engine


class Patcher:
    def __init__(self, model, **kwargs):
        self.model = model
        self.adopted = None

    def adopt_comfy_state_from(self, old):
        self.adopted = old
        return self


def make_actual_fixture(testcase):
    """Extract actual PC Engine _get_engine + QFLazyEngine lifecycle over synthetic C seams."""
    acquisition = contextvars.ContextVar("acquisition", default=None)

    @contextlib.contextmanager
    def acquiring(consumer):
        token = acquisition.set(NS(consumer=consumer))
        try:
            yield
        finally:
            acquisition.reset(token)

    class Handle:
        calls = 0

        @classmethod
        def create(cls, lib, *, create_params, capacity_bytes=0):
            cls.calls += 1
            if cls.calls == 1:
                raise RuntimeError("injected first create refusal")
            result = cls()
            result.lib = lib
            result.xfm = object()
            result.capacity_bytes = capacity_bytes
            result.current_session = None
            result.step_count = 0
            result.sampler_step_count = 0
            result.applied_lora_sig = "[]"
            result.applied_api_key = None
            result.loras_applied = []
            result.set_loras = lambda loras: result.loras_applied.append(copy.deepcopy(loras))
            return result

    qfe = NS(
        load_lib=lambda *_a, **_k: "lib",
        loaded_so_path=lambda: "engine.so",
        library_identity=lambda lib: "image-id",
        estimate_vram=lambda lib, params: NS(bytes=64, reason=None),
        QFEngineHandle=Handle,
        info=lambda *a, **k: None,
        console_safe=str,
    )
    qfmp = NS(_current_engine_cache_acquisition=lambda: acquisition.get())
    create_locks = {}
    engine_env = dict(
        os=os, json=json, threading=threading, qfe=qfe, qfmp=qfmp,
        _ENGINE_IDENTITY_LOCK=threading.RLock(),
        _CREATE_LOCKS=create_locks,
        _PIPELINE_CACHE={},
        _prepare_params=lambda weights_path, cfg, dev, api_key=None: NS(
            weights_path=weights_path, device_id=dev, observed_cfg=copy.deepcopy(cfg)),
        _pin_pipeline_consumer_locked=lambda k: None,
        _sweep_dead_pipelines=lambda k: None,
    )
    install([function(Q / "__init__.py", name) for name in ("_create_lock_for", "_engine_recipe", "_get_engine")], engine_env)

    class ActualLazy:
        pass

    cls_node = next(n for n in ast.walk(ast.parse((Q / "qf_modelpatcher.py").read_text(encoding="utf-8")))
                    if isinstance(n, ast.ClassDef) and n.name == "QFLazyEngine")
    for name in ("__init__", "_adopt_real", "_drop_dead_real", "ensure", "_apply_runtime_lora", "lora_set", "_lora_sig"):
        method = next(c for c in cls_node.body if isinstance(c, ast.FunctionDef) and c.name == name)
        env = dict(qfe=qfe, _engine_cache_acquisition=acquiring, json=json,
                   _LORA_DRAIN_RETRY_S=0.0, warn_lora_not_loaded=lambda: None)
        install([method], env)
        setattr(ActualLazy, name, env[name])
    ActualLazy._apply_runtime_api_key = lambda self: False
    return locals()


class UnifiedLoRATests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "adapter.safetensors"
        self.path.write_bytes(b"original lora bytes")
        self.captured = []
        self.env = dict(
            weakref=weakref, QFLazyEngine=Lazy, QFModelPatcher=Patcher,
            QF_LORA_STACK_ATTR="_stack", QF_LORA_REBUILD_ATTR="_rebuild",
            LOADER_SIG=NS(get=lambda: "sig"), LOADER_API_KEY=NS(get=lambda: None),
            current_torch_device=lambda: ("cpu", 0), ensure_model_config_attrs=lambda m: None,
            comfy=NS(model_management=NS(unet_offload_device=lambda: "cpu")),
            qfe=NS(info=lambda *a, **k: None),
        )
        install([function(Q / "qf_modelpatcher.py", n) for n in ("make_engine_factory", "tag_lora_rebuild", "family_build")],
                self.env)
        self.deps = {
            "get_engine": lambda *a, **k: self.captured.append((a, k)) or (NS(), "key"),
            "bind_pipeline_model": lambda *a: None,
            "read_auth": lambda *a: None,
        }
        self.entry = {"path": str(self.path), "scale": 1.0}

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        return self.env["family_build"](
            self.deps, "model.safetensors", {"model_arch": "krea2", "config_path": "/cfg.json"},
            lambda c: NS(), {}, lambda c, e, d: Model(e), "label")

    def test_no_load_time_widget_and_unified_in_place_apply(self):
        src = ast.parse((Q / "__init__.py").read_text(encoding="utf-8"))
        cls = next(n for n in ast.walk(src) if isinstance(n, ast.ClassDef) and n.name == "QuantFuncNativeLoRA")
        env = {
            "qfmp": NS(rebuild_of=lambda p: p.model._rebuild, lora_stack_of=lambda p: list(p.model._stack)),
            "_resolve_lora": lambda n: str(self.path),
            "_lora_choices": lambda: ["adapter"],
        }
        install([cls], env)
        Node = env["QuantFuncNativeLoRA"]
        spec = Node.INPUT_TYPES()
        self.assertNotIn("application", spec.get("required", {}))
        self.assertNotIn("application", spec.get("optional", {}))
        node = Node()
        node._refuse_foreign_lora_format = lambda p: None
        base = self.build()([])
        out = node.apply(base, "adapter", 1.0)[0]
        self.assertIs(out.adopted, base)
        out.model._qf.factory()
        self.assertNotIn("lora", self.captured[-1][1]["create_cfg"])
        self.assertEqual(out.model._qf._lora, [self.entry])
        self.assertEqual(out.model._stack, [self.entry])
        second = node.apply(out, "adapter", 0.5)[0]
        self.assertIs(second.adopted, out)
        self.assertEqual(second.model._stack, [self.entry, {"path": str(self.path), "scale": 0.5}])

    def test_recipe_cache_key_is_pure_weight_recipe(self):
        env = dict(os=os, json=json, qfe=NS(load_lib=lambda *_a, **_k: "lib", loaded_so_path=lambda: "engine.so"))
        install([function(Q / "__init__.py", "_engine_recipe")], env)
        recipe = env["_engine_recipe"]
        for cfg in (None, {}, {"model_arch": "ltx25", "pinned_memory": True}):
            want = ("engine.so", "model.safetensors", 0, json.dumps(cfg or {}, sort_keys=True))
            self.assertEqual(recipe("model.safetensors", cfg), ("lib", want, cfg))

    def test_failed_create_does_not_publish_and_retry_publishes_once(self):
        f = make_actual_fixture(self)
        env, LazyCls, Handle = f["engine_env"], f["ActualLazy"], f["Handle"]
        lazy = LazyCls(lambda: env["_get_engine"]("model.safetensors", {"model_arch": "krea2"}))
        with self.assertRaisesRegex(RuntimeError, "injected first create refusal"):
            lazy.ensure()
        self.assertEqual(len(env["_PIPELINE_CACHE"]), 0)
        self.assertIsNone(lazy._real)

        real = lazy.ensure()
        self.assertEqual(Handle.calls, 2)
        self.assertEqual(len(env["_PIPELINE_CACHE"]), 1)
        self.assertIs(lazy._real, real)
        self.assertEqual(real.loras_applied, [])

        second = LazyCls(lazy._factory)
        second._lora = [self.entry]
        self.assertIs(second.ensure(), real)
        self.assertEqual(Handle.calls, 2)
        self.assertEqual(real.loras_applied, [[self.entry]])
        self.assertIs(second.ensure(), real)
        self.assertEqual(real.loras_applied, [[self.entry]])
        self.assertIs(lazy.ensure(), real)
        self.assertEqual(real.loras_applied, [[self.entry], []])


if __name__ == "__main__":
    unittest.main(verbosity=2)
