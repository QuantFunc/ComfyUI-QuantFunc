#!/usr/bin/env python3
"""Public QuantFuncNativeLoRA chain cache contract under QuantFunc PC Engine.

Verifies end-to-end across actual QuantFuncNativeLoRA.apply, family_build, _get_engine, and
QFLazyEngine that all LoRA stacks share ONE transformer handle and hot-swap in place via set_loras.
"""
import ast
import copy
import json
import pathlib
import sys
import unittest
from types import SimpleNamespace as NS

Q = pathlib.Path(__file__).resolve().parents[1]
foundation = {"__name__": "frozen_fixture_only", "__file__": str(Q / "tests/load_time_lora_test.py")}
exec(compile((Q / "tests/load_time_lora_test.py").read_text(encoding="utf-8"),
             str(Q / "tests/load_time_lora_test.py"), "exec"), foundation)
install = foundation["install"]
function = foundation["function"]


class PublicChain(foundation["UnifiedLoRATests"]):
    def setUp(self):
        super().setUp()
        self.fixture = foundation["make_actual_fixture"](self)
        f = self.fixture
        qfe, handle, cache = f["qfe"], f["Handle"], f["engine_env"]
        handle.calls = 1
        self.creates, self.updates = [], []
        original_create = handle.create

        def create(cls, lib, **kwargs):
            result = original_create(lib, **kwargs)
            cfg = copy.deepcopy(kwargs["create_params"].observed_cfg)
            self.creates.append(cfg)
            result.observed_cfg = cfg
            result.set_loras = lambda payload: self.updates.append(copy.deepcopy(payload))
            return result

        handle.create = classmethod(create)
        self.env["QFLazyEngine"] = f["ActualLazy"]
        self.env["qfe"] = qfe
        self.deps["get_engine"] = cache["_get_engine"]
        for name in ("family_build", "rebuild_of", "lora_stack_of"):
            install([function(Q / "qf_modelpatcher.py", name)], self.env)
        source = ast.parse((Q / "__init__.py").read_text(encoding="utf-8"))
        cls = next(n for n in ast.walk(source) if isinstance(n, ast.ClassDef) and n.name == "QuantFuncNativeLoRA")
        env = {
            "qfmp": NS(rebuild_of=self.env["rebuild_of"], lora_stack_of=self.env["lora_stack_of"]),
            "_resolve_lora": lambda name: str(self.path),
            "_lora_choices": lambda: ["adapter"],
        }
        install([cls], env)
        self.node = env["QuantFuncNativeLoRA"]()
        self.node._refuse_foreign_lora_format = lambda path: None
        self.base = self.build()([])

    def apply(self, patcher, strength=1.0):
        return self.node.apply(patcher, "adapter", strength)[0]

    def test_public_identical_lora_stacks_reuse_one_handle_and_apply_once(self):
        first = self.apply(self.base)
        second = self.apply(self.base)
        third = self.apply(self.base)
        self.assertEqual(self.creates, [])
        real = first.model._qf.ensure()
        self.assertIs(second.model._qf.ensure(), real)
        self.assertIs(third.model._qf.ensure(), real)
        self.assertEqual(len(self.creates), 1)
        self.assertNotIn("lora", self.creates[0])
        self.assertEqual(self.updates, [[self.entry]])

    def test_public_two_chained_nodes_share_one_handle_and_swap_stack_in_place(self):
        first = self.apply(self.base)
        final = self.apply(first, 0.5)
        self.assertIs(final.adopted, first)
        self.assertEqual(self.creates, [])
        real = final.model._qf.ensure()
        expected = [self.entry, dict(self.entry, scale=0.5)]
        self.assertEqual(len(self.creates), 1)
        self.assertEqual(final.model._stack, expected)
        self.assertEqual(self.updates, [expected])
        self.assertIs(self.base.model._qf.ensure(), real)
        self.assertEqual(self.updates, [expected, []])

    def test_public_strength_and_order_swap_in_place_on_single_handle(self):
        one = self.apply(self.base, 1.0)
        half = self.apply(self.base, 0.5)
        ab = self.apply(one, 0.5)
        ba = self.apply(half, 1.0)
        handles = [x.model._qf.ensure() for x in (one, half, ab, ba)]
        self.assertEqual(len({id(x) for x in handles}), 1)
        self.assertEqual(len(self.creates), 1)
        self.assertEqual(self.updates, [
            [self.entry],
            [dict(self.entry, scale=0.5)],
            [self.entry, dict(self.entry, scale=0.5)],
            [dict(self.entry, scale=0.5), self.entry],
        ])


if __name__ == "__main__":
    names = sorted(n for n in PublicChain.__dict__ if n.startswith("test_"))
    suite = unittest.TestSuite(PublicChain(n) for n in names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)}))
    sys.exit(not result.wasSuccessful())
