"""CPU-only Sol/sparse option and migration checks; no Comfy, Torch or engine import."""
import ast
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import qf_sol  # noqa: E402


def mixin():
    tree = ast.parse((ROOT / "qf_modelpatcher.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QFSessionModelMixin")
    cls.decorator_list = []
    wanted = {"set_sol_tau", "set_sol_options", "residency_opts", "dial_opts",
              "adopt_session_dials_from", "_effective_cat_ratio"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted or
                isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_SESSION_DIALS" for t in n.targets)]
    ns = {"qf_sol": qf_sol, "_CAT_KEEP_DEFAULT": 0.80}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "mixin", "exec"), ns)
    model = ns["QFSessionModelMixin"]()
    model._video_enhance = False
    return model


class SolOptionsTest(unittest.TestCase):
    def test_existing_legacy_setter_migrates_to_canonical_sparse_object(self):
        model = mixin()
        model.set_sol_tau(0.25)
        for opts in (model.dial_opts(), model.residency_opts()):
            self.assertIn("sparse", opts)
            self.assertNotIn("sol", opts)
            self.assertNotIn("sparse_cdf", opts)
            self.assertNotIn("sol_tau", opts)
            self.assertEqual(opts["sparse"]["selector"], "svg2")
            self.assertAlmostEqual(opts["sparse"]["tau"], 0.67448975, places=7)
            self.assertIsNone(opts["sparse"]["window"])
            self.assertEqual(opts["sparse"]["min_tokens"], 0)

    def test_marker_defaults_zero_and_negative_conversion(self):
        self.assertFalse(qf_sol.from_loader().enabled)
        self.assertFalse(qf_sol.from_loader(sol_tau=0).enabled)
        self.assertFalse(qf_sol.from_loader(sol_tau=1 - 0.5e-6).enabled)
        self.assertAlmostEqual(qf_sol.from_loader(sol_tau=0.75).tau, -0.67448975, places=7)
        self.assertAlmostEqual(qf_sol.from_loader(sol_tau=0.99999).tau, -3.71901655, places=7)
        new = qf_sol.from_loader(sol_version=2, sol_enabled=True, sol_tau=0)
        self.assertTrue(new.enabled)
        self.assertEqual(new.tau, 0)
        self.assertEqual(qf_sol.from_loader(sol_version=2, sol_enabled=True).tau, qf_sol.f32(1.3))
        self.assertFalse(qf_sol.from_loader(sol_version=2).enabled)

    def test_actual_sigma_mapping_unbounded_and_reset(self):
        calls = []

        class Sampling:
            def percent_to_sigma(self, p):
                calls.append(p)
                return 2 / (1 + 3 * p)

        model = mixin()
        model.model_sampling = Sampling()
        model.set_sol_options(qf_sol.from_loader(sol_version=2, sol_enabled=True))
        opts = model.dial_opts()["sparse"]
        self.assertEqual(calls, [0.2, 0.9])
        self.assertEqual(opts["selector"], "svg2")
        self.assertEqual(opts["window"], {"sigma_start": 1.25, "sigma_end": 2 / 3.7})
        model.set_sol_options(qf_sol.from_loader(sol_version=2, sol_enabled=True, sol_window_enabled=False))
        self.assertIsNone(model.dial_opts()["sparse"]["window"])
        model.set_sol_options(qf_sol.from_loader(sol_version=2, sol_enabled=False))
        self.assertNotIn("sparse", model.dial_opts())
        self.assertNotIn("sol", model.dial_opts())
        model.set_sol_tau(0.25)
        self.assertIn("sparse", model.dial_opts())
        model.set_sol_tau(1)
        self.assertNotIn("sparse", model.dial_opts())

    def test_invalid_or_ambiguous_input_is_rejected(self):
        for kwargs in ({"sol_enabled": True}, {"sol_version": 3}, {"sol_version": True},
                       {"sol_tau": math.nan}, {"sol_version": 2, "sol_tau": math.inf},
                       {"sol_version": 2, "sol_start_percent": 0.8, "sol_end_percent": 0.2},
                       {"sol_version": 2, "sol_min_tokens": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                qf_sol.from_loader(**kwargs)

    def test_lora_clone_keeps_options_and_independent_return_dicts(self):
        original, clone = mixin(), mixin()
        original.set_sol_options(qf_sol.from_loader(sol_version=2, sol_enabled=True, sol_window_enabled=False))
        clone.adopt_session_dials_from(original)
        first = clone.dial_opts()
        self.assertEqual(first, original.dial_opts())
        first["sparse"]["tau"] = -4
        self.assertEqual(clone.dial_opts()["sparse"]["tau"], qf_sol.f32(1.3))
        clone.set_sol_options(qf_sol.from_loader(sol_version=2))
        self.assertNotIn("sparse", clone.dial_opts())
        self.assertIn("sparse", original.dial_opts())

    def test_legacy_payload_matches_canonical_pc_engine_sparse_schema(self):
        self.assertEqual(qf_sol.from_loader(sol_tau=0.25).native(), {
            "enabled": True, "selector": "svg2", "tau": 0.6744897365570068, "window": None,
            "min_tokens": 0, "sink_conditioning": "off"})
        self.assertEqual(qf_sol.from_loader(sol_version=2, sol_enabled=True).sink_conditioning,
                         "exact_kv_and_rows")


if __name__ == "__main__":
    unittest.main()
