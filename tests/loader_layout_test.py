#!/usr/bin/env python3
"""A workflow saved with an earlier loader layout loads with every value in the right widget (user 2026-09-28:
「要输入的都在上方 开关的统一在下方 所有控件都这样」: value inputs first, switches last, on every QuantFunc loader).

ComfyUI restores a saved node's widget values by POSITION, so web/quantfunc_loader_layout.js remaps an earlier saved
order on load. This drives the real script under Node with a stub `app`; each loader's CURRENT order is read from
__init__.py's INPUT_TYPES (required, then optional), so the test follows the node as it is.

  L1 each earlier saved order is remapped by name into the current one, for all four loaders: LTX-2 and MiniMax-H3 as
     published (quality_enhance before the caches; H3's removed allow_partial_denoise is dropped) and as the release
     candidate saved them (a `quality` dropdown: best_quality is ON, anything else OFF; a widget with no saved value
     keeps its default); Krea-2 and Qwen-Image-2.1 as the release candidate saved them
  L2 a node saved by name (widgets_values_named) is placed by name
  L3 nothing is remapped when the node was saved with the current layout (the qf_layout property), when the values
     already have the current order's shape, or when they match no earlier order; the API key row (never saved) takes
     no saved value
  L4 every loader node is stamped with the current layout when created and when loaded, so it is never remapped twice;
     other nodes are left alone

MUTATION (each goes RED): drop an earlier order or its type check -> L1/L3; map `quality` by truthiness -> L1; ignore
the saved names -> L2; ignore the layout property -> L3; count the API key row -> L3; skip the stamp -> L4.

Run:  python tests/loader_layout_test.py      (needs node)
"""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

PLUGIN = Path(__file__).resolve().parents[1]
LOADERS = ("QuantFuncLTXLoader", "QuantFuncH3Loader", "QuantFuncKrea2Loader", "QuantFuncQwenImage21Loader")


def current_orders():
    """Each loader's widget order as its INPUT_TYPES declares it: required keys, then optional keys."""
    orders = {}
    for node in ast.walk(ast.parse((PLUGIN / "__init__.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.ClassDef) and node.name in LOADERS:
            fn = next(m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == "INPUT_TYPES")
            ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return)).value
            orders[node.name] = [k.value for part, spec in zip(ret.keys, ret.values)
                                 if isinstance(part, ast.Constant) and part.value in ("required", "optional")
                                 for k in spec.keys]
    return orders


ORDERS = current_orders()
BOOLEANS = {"quality_enhance", "audio_enhance", "pinned_memory"}
NUMBERS = {"sol_tau", "step_cache", "block_cache"}

# Saved exactly as the earlier plugins saved them (the orders of 863c2ac, published, and 61f7076, the release
# candidate), with values that differ per widget so a misplaced value shows.
SAVED = {
    "ltx_published": ("QuantFuncLTXLoader",
                      ["ltx.safetensors", "ltx2", "sage", 0.7, True, 0.11, 0.22, False]),
    "ltx_rc": ("QuantFuncLTXLoader", ["ltx.safetensors", "ltx2", "sage", 0.7, "best_quality", 0.11, 0.22]),
    "h3_published": ("QuantFuncH3Loader",
                     ["h3.safetensors", "minimax-h3", "flash", 0.6, True, True, 0.33, 0.44, True, False]),
    "h3_rc": ("QuantFuncH3Loader", ["h3.safetensors", "minimax-h3", "flash", 0.6, "fast", True, 0.33, 0.44, False]),
    "krea2_rc": ("QuantFuncKrea2Loader", ["k.safetensors", "krea2", "auto", "best_quality"]),
    "qi21_rc": ("QuantFuncQwenImage21Loader", ["q.safetensors", "qwenimage21", "auto", "balanced"]),
}
# ...and what each widget must hold after loading (a widget the earlier node did not have keeps its default).
DEFAULTS = {"pinned_memory": "DEFAULT_PINNED"}
WANT = {
    "ltx_published": {"transformer": "ltx.safetensors", "model_config": "ltx2", "attention_backend": "sage",
                      "sol_tau": 0.7, "step_cache": 0.11, "block_cache": 0.22, "quality_enhance": True,
                      "pinned_memory": False},
    "ltx_rc": {"transformer": "ltx.safetensors", "model_config": "ltx2", "attention_backend": "sage", "sol_tau": 0.7,
               "step_cache": 0.11, "block_cache": 0.22, "quality_enhance": True, "pinned_memory": "DEFAULT_PINNED"},
    "h3_published": {"transformer": "h3.safetensors", "model_config": "minimax-h3", "attention_backend": "flash",
                     "sol_tau": 0.6, "step_cache": 0.33, "block_cache": 0.44, "quality_enhance": True,
                     "audio_enhance": True, "pinned_memory": False},
    "h3_rc": {"transformer": "h3.safetensors", "model_config": "minimax-h3", "attention_backend": "flash",
              "sol_tau": 0.6, "step_cache": 0.33, "block_cache": 0.44, "quality_enhance": False, "audio_enhance": True,
              "pinned_memory": "DEFAULT_PINNED"},
    "krea2_rc": {"transformer": "k.safetensors", "model_config": "krea2", "attention_backend": "auto",
                 "quality_enhance": True, "pinned_memory": "DEFAULT_PINNED"},
    "qi21_rc": {"transformer": "q.safetensors", "model_config": "qwenimage21", "attention_backend": "auto",
                "quality_enhance": False, "pinned_memory": "DEFAULT_PINNED"},
}

_APP_STUB = "export const app = { extensions: [], registerExtension(e) { this.extensions.push(e); } };\n"
_DRIVER = """const { app } = await import("./scripts/app.js");
const { remappedValues } = await import("./extensions/ComfyUI-QuantFunc/quantfunc_loader_layout.js");
const ORDERS = %s, SAVED = %s, DEFAULTS = %s;
const out = { remapped: {}, loaded: {}, stamped: {}, untouched: {} };
// the loader node as the frontend builds it: one widget per input, the API key row (never saved) above the switches
function widgets(cls) {
  const ws = ORDERS[cls].map(name => ({ name, value: DEFAULTS[name] ?? "DEFAULT_" + name }));
  const first = ws.findIndex(w => typeof w.value === "string" && %s.includes(w.name));
  ws.splice(first < 0 ? ws.length : first, 0, { name: "api_key", value: "qf_0123\\u2026cdef", serialize: false });
  return ws;
}
const types = {};
for (const cls of [...Object.keys(ORDERS), "KSampler"]) {
  const got = [];
  types[cls] = { prototype: { configure(info) { got.push(info.widgets_values); } } };
  types[cls].got = got;
  for (const e of app.extensions) e.beforeRegisterNodeDef?.(types[cls], { name: cls });
}
const values = (cls, ws, v) => Object.fromEntries(ws.filter(w => w.serialize !== false).map((w, i) => [w.name, v[i]]));
for (const [label, [cls, saved]] of Object.entries(SAVED)) {
  const ws = widgets(cls);
  const node = { comfyClass: cls, widgets: ws, properties: {} };
  types[cls].prototype.configure.call(node, { widgets_values: saved, properties: {} });
  const got = types[cls].got.at(-1);
  out.loaded[label] = { values: values(cls, ws, got), length: got.length, stamp: node.properties.qf_layout };
}
// L2: saved by name, in any order
const byName = { widgets_values: [], widgets_values_named: { block_cache: 0.5, transformer: "n.safetensors",
                  quality_enhance: true, sol_tau: 0.9 } };
out.named = remappedValues("QuantFuncLTXLoader", widgets("QuantFuncLTXLoader"), byName);
// L3
const current = ORDERS.QuantFuncLTXLoader.map(n => (%s.includes(n) ? false : %s.includes(n) ? 0.5 : "x"));
out.untouched.marked = remappedValues("QuantFuncLTXLoader", widgets("QuantFuncLTXLoader"),
                                      { widgets_values: SAVED.ltx_published[1], properties: { qf_layout: 2 } });
out.untouched.currentShape = remappedValues("QuantFuncLTXLoader", widgets("QuantFuncLTXLoader"),
                                            { widgets_values: current, properties: {} });
out.untouched.unknownShape = remappedValues("QuantFuncH3Loader", widgets("QuantFuncH3Loader"),
                                            { widgets_values: ["h3.safetensors", "minimax-h3"], properties: {} });
out.untouched.otherNode = remappedValues("KSampler", [{ name: "seed", value: 1 }], { widgets_values: [5] });
const k = { comfyClass: "KSampler", widgets: [], properties: {} };
types.KSampler.prototype.configure.call(k, { widgets_values: [5], properties: {} });
out.untouched.otherNodeStamp = k.properties.qf_layout ?? null;
out.untouched.otherNodeGot = types.KSampler.got.at(-1);
// L4
for (const cls of [...Object.keys(ORDERS), "KSampler"]) {
  const n = { comfyClass: cls, properties: {} };
  for (const e of app.extensions) e.nodeCreated?.(n);
  out.stamped[cls] = n.properties.qf_layout ?? null;
}
console.log(JSON.stringify(out));
"""


class LoaderLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = None
        node = shutil.which("node")
        if node is None:
            return
        d = tempfile.mkdtemp(prefix="qf_layout_js_")
        try:
            (Path(d) / "package.json").write_text('{"type": "module"}', encoding="utf-8")
            (Path(d) / "scripts").mkdir()
            (Path(d) / "scripts" / "app.js").write_text(_APP_STUB, encoding="utf-8")
            ext = Path(d) / "extensions" / "ComfyUI-QuantFunc"
            ext.mkdir(parents=True)
            shutil.copy(PLUGIN / "web" / "quantfunc_loader_layout.js", ext)
            switches = json.dumps(sorted(BOOLEANS))
            (Path(d) / "driver.mjs").write_text(
                _DRIVER % (json.dumps(ORDERS), json.dumps(SAVED), json.dumps(DEFAULTS), switches, switches,
                           json.dumps(sorted(NUMBERS))), encoding="utf-8")
            r = subprocess.run([node, "driver.mjs"], cwd=d, capture_output=True, encoding="utf-8", timeout=60)
            if r.returncode != 0:
                raise AssertionError(f"node driver failed: {r.stderr[-2000:]}")
            cls.out = json.loads(r.stdout.strip().splitlines()[-1])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def setUp(self):
        if self.out is None:
            self.skipTest("[SKIP] node is not installed")

    def test_the_current_orders_put_the_switches_last(self):
        self.assertEqual(sorted(ORDERS), sorted(LOADERS))
        for cls, order in ORDERS.items():
            kinds = ["switch" if n in BOOLEANS else "value" for n in order]
            self.assertEqual(kinds, sorted(kinds, key=lambda k: k == "switch"), cls)
            self.assertNotIn("allow_partial_denoise", order, cls)

    def test_l1_every_earlier_saved_order_loads_by_name(self):
        for label, want in WANT.items():
            with self.subTest(label):
                cls = SAVED[label][0]
                got = self.out["loaded"][label]
                self.assertEqual(got["values"], want)
                self.assertEqual(got["length"], len(ORDERS[cls]))
                self.assertEqual(got.get("stamp"), 2)

    def test_l2_a_node_saved_by_name_is_placed_by_name(self):
        self.assertIsNotNone(self.out["named"])
        by_name = dict(zip(ORDERS["QuantFuncLTXLoader"], self.out["named"]))
        self.assertEqual(by_name["transformer"], "n.safetensors")
        self.assertEqual(by_name["sol_tau"], 0.9)
        self.assertEqual(by_name["block_cache"], 0.5)
        self.assertIs(by_name["quality_enhance"], True)
        self.assertEqual(by_name["step_cache"], "DEFAULT_step_cache")

    def test_l3_nothing_else_is_remapped(self):
        self.assertEqual(self.out["untouched"], {"marked": None, "currentShape": None, "unknownShape": None,
                                                 "otherNode": None, "otherNodeStamp": None, "otherNodeGot": [5]})

    def test_l4_loader_nodes_are_stamped_when_created(self):
        self.assertEqual(self.out["stamped"], {**{cls: 2 for cls in LOADERS}, "KSampler": None})


if __name__ == "__main__":
    result = unittest.main(argv=[__file__], exit=False, verbosity=2).result
    print(f"LOADER_LAYOUT: {'PASS' if result.wasSuccessful() else 'FAIL'} ({result.testsRun} arms)")
    raise SystemExit(0 if result.wasSuccessful() else 1)
