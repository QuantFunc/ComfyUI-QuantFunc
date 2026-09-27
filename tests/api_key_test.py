#!/usr/bin/env python3
"""The API key field of the four QuantFunc loaders (user 2026-09-27: a valid key typed there wins; config.json is then
not read).

ComfyUI saves every widget value into the workflow and every prompt input into each image and video it saves, and it
has no secret widget a custom node can use. So the key is never a widget value or a prompt value: the field
(web/quantfunc_api_key.js) is a password input that is never saved, and a queued prompt carries a reference
(POST /quantfunc/api_key) that only this ComfyUI process turns back into the key.

  A  qf_api_key, pure Python
    A1 a field key is what the QuantFunc service mints (qf_ + 64 lowercase hex); anything else is refused loud and
       never echoed
    A2 an empty field (or none: the script is absent) is "no field key"
    A3 a key written into a prompt directly is refused, never echoed; so is a non-string value
    A4 a reference this process never issued is refused (never read as empty)
    A5 a reference holds no key material; it is stable within a process (ComfyUI's node cache) and differs across
       processes
    A6 the route: {"key": k} -> {"ref": r} resolving to k (surrounding whitespace dropped); empty -> ""; anything else
       -> 400 that does not echo the value
    A7 add_api_key_input gives ANY loader class the hidden input: the key is published only while the loader runs (reset
       after, also on an exception), the loader never receives the api_key argument, an unusable value fails before
       the loader runs, the class's own spec dicts are untouched and the signature gains a keyword-only api_key
  B  the plugin (needs COMFY_ROOT)
    B1 every registered QuantFunc*Loader (derived from NODE_CLASS_MAPPINGS, not a list) takes api_key as a HIDDEN input
       (ComfyUI never makes a plain, saved widget for it), and ComfyUI hands a hidden input's prompt value to it
    B2 precedence: a field key wins, and neither QUANTFUNC_API_KEY nor config.json is read; no field key = today's path
    B3 each loader hands its field key to the models its build makes (a LoRA rebuild too); an unusable field fails the
       loader before its builder runs
    B4 the key reaches the create, never the pipeline cache key: two keys share one prepared pipeline
    B5 another key on a live pipeline switches it in place (quantfunc_set_api_key) once, without a create; an emptied
       field switches back to the config.json key; with no key there it is refused, never set_api_key(""); a switch the
       engine refuses fails the run without the key in the message, and the next run retries it
    B6 no field key anywhere: config.json is read once, at the create, and nothing is ever switched
    B7 no key reaches a log line or an error message
    B8 ComfyUI can serve the field's script (WEB_DIRECTORY), and a serving ComfyUI gets the route
  C  web/quantfunc_api_key.js under Node (skipped without node)
    C1 exactly the QuantFunc nodes that declare the hidden api_key input (the four loaders, and a loader added later)
       get one api_key field, never saved into a workflow; another pack's node never does
    C2 an empty field sends nothing and queues ""; a key is POSTed to /quantfunc/api_key and only the reference is queued,
       also when the prompt is queued while the field is still being edited
    C3 a refused POST fails the queue; the plain key is never queued
    C4 the key is remembered in this browser (localStorage) and shown again in every new field; emptying the field forgets it
    C5 (user 2026-09-27: 部分明文) a field not being edited shows the key shortened, qf_ + its first and last 4 characters,
       never more (short text shows only the prefix); being edited, it holds the whole key masked

MUTATION (each goes RED): accept any non-empty field key -> A1; read "" as a key -> A2/B3; return the plain key for a
non-reference -> A3; read an unknown reference as empty -> A4; make the reference the key or a slice of it -> A5; echo
the value in a 400 -> A6; never reset the published key, or pass api_key on to the loader -> A7; skip the attach for the
loaders -> B1; let QUANTFUNC_API_KEY beat the field key, or read config.json with one -> B2; drop the ContextVar hand-off
in family_build -> B3; put the key into create_cfg -> B4; drop the switch in ensure(), send set_api_key(""), or record a
switch before the engine accepted it -> B5; read config.json in ensure() for a default consumer -> B6; print the key in
the switch line -> B7; drop WEB_DIRECTORY -> B8; draw the field on every node or on another pack's node, save it
(serialize true) or queue the plain key, or queue a stale key while the field is edited -> C1/C2; fall back to the plain
key on a refused POST -> C3; drop the localStorage write -> C4; show the whole key while not edited, or unmasked while
edited -> C5.

Run:  python tests/api_key_test.py      (B needs COMFY_ROOT; C needs node)
"""
import asyncio
import contextlib
import ctypes
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

PLUGIN = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qf_api_key_under_test", PLUGIN / "qf_api_key.py")
ak = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ak)

HEX = "0123456789abcdef"
KEY = "qf_" + HEX * 4                  # well formed; built at run time, never a real key
KEY2 = "qf_" + HEX[::-1] * 4
DEFAULT_KEY = "qf_" + "7" * 64         # stands in for the key of config.json
SHORT = "qf_0123…cdef"                 # KEY shortened: qf_ + its first and last 4 characters
DEFAULT_URL = "https://service.quantfunc.com"
MALFORMED = ("qf_" + "a" * 63, "qf_" + "a" * 65, "QF_" + "a" * 64, "qf_" + "A" * 64, "qf-" + "a" * 64,
             "qf_" + "a" * 62 + "zz", "a" * 67, "qf_" + "a" * 32 + " " + "a" * 31)
LOADERS = ("QuantFuncLTXLoader", "QuantFuncH3Loader", "QuantFuncKrea2Loader", "QuantFuncQwenImage21Loader")


def skip(case, why):
    print(f"[SKIP] {case.id()}: {why}")     # tests/run_plugin_tests.py counts these
    case.skipTest(why)


# ── A: qf_api_key ─────────────────────────────────────────────────────────────────────────────────────────────────────
class FieldKey(unittest.TestCase):
    def test_a_well_formed_key_behind_its_reference_is_the_field_key(self):
        self.assertEqual(ak.field_key(ak.remember(KEY)), KEY)

    def test_a_malformed_key_is_refused_loud_and_never_echoed(self):
        for bad in MALFORMED:
            with self.subTest(bad=bad), self.assertRaises(RuntimeError) as cm:
                ak.field_key(ak.remember(bad))
            self.assertNotIn(bad, str(cm.exception))
            self.assertIn("qf_", str(cm.exception))     # it names the shape a key has

    def test_an_empty_field_is_no_field_key(self):
        for empty in ("", "   \t\n", None):
            self.assertIsNone(ak.field_key(empty))

    def test_a_key_written_into_the_prompt_is_refused_and_never_echoed(self):
        for plain in (KEY, "  " + KEY, "hello"):
            with self.subTest(plain=plain), self.assertRaises(RuntimeError) as cm:
                ak.field_key(plain)
            self.assertNotIn(KEY, str(cm.exception))

    def test_a_non_string_field_is_refused(self):       # ComfyUI does not validate hidden inputs
        for value in (5, ["qfk:" + "0" * 32], {"ref": 1}):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                ak.field_key(value)

    def test_a_reference_this_process_never_issued_is_refused(self):
        with self.assertRaises(RuntimeError):
            ak.field_key("qfk:" + "0" * 32)


class Reference(unittest.TestCase):
    def test_a_reference_holds_no_key_material(self):
        ref = ak.remember(KEY)
        for i in range(len(KEY) - 8):
            self.assertNotIn(KEY[i:i + 8], ref)

    def test_a_reference_is_stable_within_a_process_and_distinct_per_key(self):
        self.assertEqual(ak.remember(KEY), ak.remember(KEY))
        self.assertNotEqual(ak.remember(KEY), ak.remember(KEY2))

    def test_another_process_issues_another_reference(self):
        code = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('m', sys.argv[1]); "
                "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.remember(sys.argv[2]))")
        out = subprocess.run([sys.executable, "-c", code, str(PLUGIN / "qf_api_key.py"), KEY],
                             capture_output=True, encoding="utf-8", check=True).stdout.strip()
        self.assertTrue(out.startswith("qfk:"), out)
        self.assertNotEqual(out, ak.remember(KEY))


class Request:
    """The part of an aiohttp request the route reads."""
    def __init__(self, body=None, broken=False):
        self.body, self.broken = body, broken

    async def json(self):
        if self.broken:
            raise json.JSONDecodeError("not JSON", "", 0)
        return self.body


class Route(unittest.TestCase):
    def setUp(self):
        try:
            from aiohttp import web
        except ImportError:
            skip(self, "aiohttp is not installed")
        routes = web.RouteTableDef()
        ak.register_route(routes)
        posts = [r for r in routes if r.method == "POST" and r.path == "/quantfunc/api_key"]
        self.assertEqual(len(posts), 1, [(r.method, r.path) for r in routes])
        self.handler = posts[0].handler

    def call(self, request):
        response = asyncio.run(self.handler(request))
        return response.status, response.text

    def test_a_key_comes_back_as_a_reference_to_it(self):
        status, text = self.call(Request({"key": "  " + KEY + "\n"}))
        self.assertEqual(status, 200)
        self.assertNotIn(KEY, text)
        self.assertEqual(ak.field_key(json.loads(text)["ref"]), KEY)

    def test_an_empty_key_comes_back_empty(self):
        for empty in ("", "  \t"):
            status, text = self.call(Request({"key": empty}))
            self.assertEqual((status, json.loads(text)), (200, {"ref": ""}))

    def test_anything_else_is_a_400_that_does_not_echo_it(self):
        long = "qf_" + "a" * 5000
        for request in (Request(broken=True), Request([KEY]), Request({"key": 5}), Request({"token": KEY}),
                        Request({"key": long})):
            status, text = self.call(request)
            self.assertEqual(status, 400, text)
            self.assertNotIn(KEY, text)
            self.assertNotIn(long, text)


class Retrofit(unittest.TestCase):
    """A7: add_api_key_input on a loader class this plugin has never seen (a loader added later)."""
    def setUp(self):
        from contextvars import ContextVar
        self.published = ContextVar("published_under_test", default=None)
        self.seen = []
        published, seen = self.published, self.seen
        self.spec = {"required": {"transformer": (["a.safetensors"],)}, "hidden": {"quality": ("STRING", {})}}
        spec = self.spec

        class Loader:
            FUNCTION = "load"

            @classmethod
            def INPUT_TYPES(cls):
                return spec

            def load(self, transformer, pinned_memory=False, **extra):
                """Load the model."""
                seen.append((transformer, published.get(), dict(extra)))
                if transformer == "boom":
                    raise ValueError("loader failed")
                return ("model",)
        self.Loader = ak.add_api_key_input(Loader, published)

    def test_the_hidden_input_is_added_and_the_class_spec_is_untouched(self):
        hidden = self.Loader.INPUT_TYPES()["hidden"]
        self.assertEqual(sorted(hidden), ["api_key", "quality"])
        self.assertEqual(self.spec["hidden"], {"quality": ("STRING", {})})

    def test_the_key_is_published_only_while_the_loader_runs(self):
        self.assertEqual(self.Loader().load(transformer="a", api_key=ak.remember(KEY)), ("model",))
        self.assertEqual(self.seen, [("a", KEY, {})])            # published during the run; api_key not passed on
        self.assertIsNone(self.published.get())
        self.Loader().load(transformer="a")
        self.assertEqual(self.seen[-1], ("a", None, {}))
        with self.assertRaises(ValueError):
            self.Loader().load(transformer="boom", api_key=ak.remember(KEY))
        self.assertIsNone(self.published.get())                 # reset on an exception too

    def test_an_unusable_value_fails_before_the_loader_runs(self):
        for field in (ak.remember(MALFORMED[0]), KEY, "qfk:" + "0" * 32):
            with self.subTest(field=field[:6]), self.assertRaises(RuntimeError):
                self.Loader().load(transformer="a", api_key=field)
        self.assertEqual(self.seen, [])

    def test_the_signature_gains_a_keyword_only_api_key(self):
        import inspect
        params = inspect.signature(self.Loader.load).parameters
        self.assertEqual(list(params), ["self", "transformer", "pinned_memory", "api_key", "extra"])
        self.assertEqual((params["api_key"].kind, params["api_key"].default), (inspect.Parameter.KEYWORD_ONLY, ""))
        self.assertIs(params["pinned_memory"].default, False)
        self.assertEqual((self.Loader.load.__name__, self.Loader.load.__doc__), ("load", "Load the model."))


# ── B: the plugin ─────────────────────────────────────────────────────────────────────────────────────────────────────
_COMFY = os.environ.get("COMFY_ROOT")
if _COMFY and (Path(_COMFY) / "comfy/model_management.py").is_file():
    from host_loader_contract_test import plugin, torch   # imports ComfyUI (--cpu) and the plugin package
    from canonical_resource_integration_test import Library

    qfe, qfmp = plugin.qfe, plugin.qfmp
    mm = qfmp.comfy.model_management
    pak = plugin.qf_api_key          # the plugin's own instance: its secret and store are the ones its loaders read
    # every registered QuantFunc loader, derived from the registry (a loader added later is included by construction)
    REGISTERED = sorted(n for n in plugin.NODE_CLASS_MAPPINGS if n.startswith("QuantFunc") and n.endswith("Loader"))

    def clean_env(**values):
        """os.environ without the plugin's auth variables, plus `values`."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("QUANTFUNC_API_KEY", "QF_API_KEY", "QF_SERVER_URL", qfe._ENV_KEYFILE_OVERRIDE)}
        env.update(values)
        return mock.patch.dict(os.environ, env, clear=True)

    class Surface(unittest.TestCase):
        def test_every_quantfunc_loader_takes_the_key_as_a_hidden_input(self):
            self.assertTrue(set(LOADERS) <= set(REGISTERED), REGISTERED)
            for name in REGISTERED:
                spec = plugin.NODE_CLASS_MAPPINGS[name].INPUT_TYPES()
                self.assertIn("api_key", spec.get("hidden", {}), name)
                self.assertNotIn("api_key", spec.get("required", {}), name)
                self.assertNotIn("api_key", spec.get("optional", {}), name)
            lora = plugin.NODE_CLASS_MAPPINGS["QuantFuncNativeLoRA"].INPUT_TYPES()
            self.assertNotIn("api_key", {**lora.get("required", {}), **lora.get("hidden", {})})

        def test_comfyui_hands_the_hidden_input_to_the_loader(self):
            import execution    # the ComfyUI executor's own input gathering, the one assumption this design rides on
            for name in REGISTERED:
                got = execution.get_input_data({"transformer": "t.safetensors", "api_key": "qfk:x"},
                                               plugin.NODE_CLASS_MAPPINGS[name], "1")
                self.assertEqual(got[0].get("api_key"), ["qfk:x"], name)

    class Precedence(unittest.TestCase):
        def keyfile(self, text):
            d = tempfile.mkdtemp(prefix="qf_apikey_")
            self.addCleanup(shutil.rmtree, d)
            path = os.path.join(d, "config.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            return path

        def test_a_field_key_wins_and_neither_the_env_key_nor_config_json_is_read(self):
            unreadable = self.keyfile("{ not json")          # reading it raises
            with clean_env(QUANTFUNC_API_KEY=KEY2, **{qfe._ENV_KEYFILE_OVERRIDE: unreadable}):
                self.assertEqual(plugin._read_auth(KEY), (KEY, DEFAULT_URL))
            with clean_env(QF_SERVER_URL="https://example.invalid", **{qfe._ENV_KEYFILE_OVERRIDE: unreadable}):
                self.assertEqual(plugin._read_auth(KEY), (KEY, "https://example.invalid"))

        def test_no_field_key_is_todays_path(self):
            good = self.keyfile(json.dumps({"api_key": DEFAULT_KEY, "server_url": "https://from-config.invalid"}))
            with clean_env(**{qfe._ENV_KEYFILE_OVERRIDE: good}):
                self.assertEqual(plugin._read_auth(None), (DEFAULT_KEY, "https://from-config.invalid"))
                self.assertEqual(plugin._read_auth(), (DEFAULT_KEY, "https://from-config.invalid"))
            with clean_env(QUANTFUNC_API_KEY=KEY2, **{qfe._ENV_KEYFILE_OVERRIDE: good}):
                self.assertEqual(plugin._read_auth(None), (KEY2, DEFAULT_URL))

    class LoaderHandOff(unittest.TestCase):
        """The loader node -> _run_family_load -> the models its build makes."""
        def model(self):
            calls = []
            m = types.SimpleNamespace()
            for setter in ("set_attn_backend", "set_sol_tau", "set_video_enhance", "set_audio_enhance",
                           "set_allow_partial_denoise", "set_step_cache", "set_block_cache"):
                setattr(m, setter, lambda *a, _s=setter: calls.append(_s))
            return m

        def run_loader(self, name, api_key):
            seen = []

            def builder(transformer1_path, bundle_dir, pinned_memory):
                seen.append(qfmp.LOADER_API_KEY.get())
                return types.SimpleNamespace(model=self.model())
            family = plugin.NODE_CLASS_MAPPINGS[name].QF_FAMILY
            with mock.patch.dict(plugin._FAMILY_BUILDERS, {family: builder}), \
                    mock.patch.object(plugin, "_family_preset", return_value="preset"), \
                    mock.patch.object(plugin, "_load_model_config", return_value=("/bundle", {"family": family})), \
                    mock.patch.object(plugin, "_resolve_transformer", side_effect=lambda n: "/models/" + n):
                try:
                    plugin.NODE_CLASS_MAPPINGS[name]().load(transformer="t.safetensors", api_key=api_key)
                except RuntimeError as exc:
                    return seen, exc
            return seen, None

        def test_every_loader_hands_its_field_key_to_its_build(self):
            for name in REGISTERED:
                with self.subTest(loader=name):
                    self.assertEqual(self.run_loader(name, pak.remember(KEY)), ([KEY], None))
                    self.assertIsNone(qfmp.LOADER_API_KEY.get())      # published only while the build runs
                    self.assertEqual(self.run_loader(name, ""), ([None], None))

        def test_an_unusable_field_fails_the_loader_before_its_build(self):
            for name in REGISTERED:
                for field in (pak.remember(MALFORMED[0]), KEY, "qfk:" + "0" * 32):
                    with self.subTest(loader=name, field=field[:6]):
                        seen, exc = self.run_loader(name, field)
                        self.assertIsInstance(exc, RuntimeError)
                        self.assertEqual(seen, [])                # the build never ran

        def test_the_field_key_rides_every_model_of_the_build_and_its_lora_rebuild(self):
            import comfy.supported_models as supported
            krea2 = sys.modules[plugin.__name__ + ".qf_krea2_modelpatcher"]
            asked, lib = [], PipelineLib()
            handle = qfe.QFEngineHandle(lib, ctypes.c_void_p(9), resource=object(), capacity_bytes=4096)

            def get_engine(model_dir, create_cfg=None, device_idx=0, api_key=None):
                asked.append((api_key, create_cfg))
                return handle, "ck"
            deps = {"get_engine": get_engine, "bind_pipeline_model": lambda *a: None, "read_auth": None}

            def builder(transformer1_path, bundle_dir, pinned_memory):
                return qfmp.family_build(deps, "/nowhere", {"denoise_only": True}, supported.Krea2,
                                         {"image_model": "krea2", "disable_unet_model_creation": True},
                                         krea2.QFKrea2Model, "[test] krea2", pinned_memory)([])
            with mock.patch.dict(plugin._FAMILY_BUILDERS, {"krea2": builder}), \
                    mock.patch.object(plugin, "_family_preset", return_value="preset"), \
                    mock.patch.object(plugin, "_load_model_config", return_value=("/bundle", {"family": "krea2"})), \
                    mock.patch.object(plugin, "_resolve_transformer", side_effect=lambda n: "/models/" + n):
                patcher, = plugin.NODE_CLASS_MAPPINGS["QuantFuncKrea2Loader"]().load(transformer="k2.safetensors",
                                                                                     api_key=pak.remember(KEY))
            rebuilt = qfmp.rebuild_of(patcher)([{"path": "a.safetensors", "scale": 1.0}])   # outside the loader
            patcher.model._qf.ensure()
            rebuilt.model._qf.ensure()
            self.assertEqual(asked, [(KEY, {"denoise_only": True})] * 2)   # beside the create keys, never among them
            self.assertEqual(lib.keys, [KEY])        # the first run signed the pipeline in; the second found it so

    class PipelineLib(Library):
        """The native double plus the two runtime mutations a live pipeline takes."""
        def __init__(self):
            super().__init__()
            self.keys, self.updates, self.key_status = [], [], []
            self.quantfunc_set_api_key = lambda pipeline, key: self._set_key(key)
            self.quantfunc_pipeline_update = lambda pipeline, payload: self.updates.append(payload) or 0

        def _set_key(self, key):
            self.keys.append(key.decode())
            return self.key_status.pop(0) if self.key_status else 0

    class Engines(unittest.TestCase):
        """The real _get_engine / QFLazyEngine over the native double."""
        def setUp(self):
            self.lib = PipelineLib()
            d = tempfile.mkdtemp(prefix="qf_apikey_")
            self.addCleanup(shutil.rmtree, d)
            keyfile = os.path.join(d, "config.json")
            with open(keyfile, "w", encoding="utf-8") as fh:
                json.dump({"api_key": DEFAULT_KEY, "server_url": DEFAULT_URL}, fh)
            self.read_auth = mock.Mock(wraps=plugin._read_auth)
            for patch in (
                    clean_env(**{qfe._ENV_KEYFILE_OVERRIDE: keyfile}),
                    mock.patch.dict(qfmp._RESOURCE_DOMAINS, {}, clear=True),
                    mock.patch.dict(plugin._PIPELINE_CACHE, {}, clear=True),
                    mock.patch.dict(plugin._PIPELINE_MODELS, {}, clear=True),
                    mock.patch.dict(plugin._PREPARED_CACHE, {}, clear=True),
                    mock.patch.object(qfe, "load_lib", return_value=self.lib),
                    mock.patch.object(qfe, "loaded_so_path", return_value="contract.so"),
                    mock.patch.object(plugin, "_read_auth", self.read_auth),
                    mock.patch.object(mm, "get_free_memory", return_value=0),
                    mock.patch.object(mm, "current_loaded_models", [])):
                patch.start()
                self.addCleanup(patch.stop)
            self.creates = []

            def create(lib, *, capacity_bytes, prepared_resource, create_params):
                self.creates.append(json.loads(create_params.config_json))
                return qfe.QFEngineHandle(lib, ctypes.c_void_p(70 + len(self.creates)), resource=prepared_resource,
                                          capacity_bytes=capacity_bytes)
            patch = mock.patch.object(qfe.QFEngineHandle, "create", side_effect=create)
            patch.start()
            self.addCleanup(patch.stop)

        def config_reads(self):
            return sum(1 for c in self.read_auth.call_args_list if not (c.args and c.args[0]))

        def consumer(self, key, read_auth=None):
            return qfmp.QFLazyEngine(lambda: plugin._get_engine("pkg", {"denoise_only": True}, api_key=key),
                                     api_key=key, read_auth=read_auth or (lambda: plugin._read_auth()))

        def test_the_key_reaches_the_create_and_never_the_pipeline_cache_key(self):
            a, b = self.consumer(KEY), self.consumer(KEY2)
            entry_a, ck_a = a.prepare_resource()
            entry_b, ck_b = b.prepare_resource()
            self.assertIs(entry_a, entry_b)                      # one prepared pipeline for both keys
            self.assertEqual(ck_a, ck_b)
            self.assertNotIn(KEY, json.dumps(ck_a))
            recipe = json.loads(entry_a.create_params.config_json)
            self.assertEqual((recipe["api_key"], recipe["server_url"]), (KEY, DEFAULT_URL))
            self.assertEqual(self.config_reads(), 0)             # a field key never reads config.json

        def test_another_key_switches_the_live_pipeline_in_place(self):
            first = self.consumer(KEY)
            first.ensure()                                       # created, signed in with KEY
            self.assertEqual((len(self.creates), self.lib.keys), (1, []))
            second = self.consumer(KEY2)
            second.ensure()
            second.ensure()
            self.assertEqual((len(self.creates), self.lib.keys), (1, [KEY2]))    # once, no create
            emptied = self.consumer(None)
            emptied.ensure()
            emptied.ensure()
            self.assertEqual(self.lib.keys, [KEY2, DEFAULT_KEY])  # back to the key of config.json
            first.ensure()
            self.assertEqual(self.lib.keys, [KEY2, DEFAULT_KEY, KEY])
            self.assertEqual(len(self.creates), 1)

        def test_an_emptied_field_with_no_key_to_fall_back_to_is_refused(self):
            self.consumer(KEY).ensure()
            lonely = self.consumer(None, read_auth=lambda: ("", DEFAULT_URL))
            with self.assertRaises(RuntimeError) as cm:
                lonely.ensure()
            self.assertEqual(self.lib.keys, [])                  # never set_api_key(""): that turns the key checks off
            self.assertIn("config.json", str(cm.exception))

        def test_a_refused_switch_fails_the_run_and_the_next_run_retries(self):
            self.consumer(KEY).ensure()
            other = self.consumer(KEY2)
            self.lib.key_status = [5]                            # the engine refuses the switch once
            with self.assertRaises(RuntimeError) as cm:
                other.ensure()
            self.assertNotIn(KEY2, str(cm.exception))
            other.ensure()                                       # not recorded as switched: the next run tries again
            other.ensure()
            self.assertEqual(self.lib.keys, [KEY2, KEY2])

        def test_no_field_key_anywhere_reads_config_json_once_and_switches_nothing(self):
            for _ in range(3):
                self.consumer(None).ensure()
            self.assertEqual(self.config_reads(), 1)             # at the create, as before this field existed
            self.assertEqual((self.creates[0]["api_key"], self.lib.keys), (DEFAULT_KEY, []))

        def test_no_key_reaches_a_log_line_or_an_error(self):
            out, err = io.StringIO(), io.StringIO()
            records = []
            handler = logging.Handler(logging.DEBUG)
            handler.emit = lambda record: records.append(record.getMessage())
            root = logging.getLogger()
            root.addHandler(handler)
            level = root.level
            root.setLevel(logging.DEBUG)
            qfe.set_log_level(2)                                 # the plugin's info lines print too
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    self.consumer(KEY).ensure()
                    self.consumer(KEY2).ensure()
                    self.consumer(None).ensure()
                    try:
                        self.consumer(None, read_auth=lambda: ("", DEFAULT_URL)).ensure()
                    except RuntimeError as exc:
                        print(exc)
                    for field in (pak.remember(MALFORMED[0]), KEY, "qfk:" + "0" * 32):
                        try:
                            pak.field_key(field)
                        except RuntimeError as exc:
                            print(exc)
            finally:
                qfe.set_log_level(3)
                root.removeHandler(handler)
                root.setLevel(level)
            text = out.getvalue() + err.getvalue() + "\n".join(records)
            self.assertIn("API key", text)                       # the lines ran
            for key in (KEY, KEY2, DEFAULT_KEY, MALFORMED[0]):
                self.assertNotIn(key, text)

    class Serving(unittest.TestCase):
        def test_comfyui_can_serve_the_field_script(self):
            self.assertTrue((PLUGIN / plugin.WEB_DIRECTORY / "quantfunc_api_key.js").is_file())

        def test_a_serving_comfyui_gets_the_route(self):
            from aiohttp import web
            routes = web.RouteTableDef()
            server = types.SimpleNamespace(PromptServer=types.SimpleNamespace(
                instance=types.SimpleNamespace(routes=routes)))
            with mock.patch.dict(sys.modules, {"server": server}):
                plugin._serve_api_key_route()
            self.assertEqual([(r.method, r.path) for r in routes], [("POST", "/quantfunc/api_key")])
else:
    print("[SKIP] api_key_test part B (plugin wiring): set COMFY_ROOT to a ComfyUI checkout")


# ── C: the field script under Node ────────────────────────────────────────────────────────────────────────────────────
_APP_STUB = "export const app = { extensions: [], registerExtension(e) { this.extensions.push(e); } };\n"
_API_STUB = """export const api = {
  calls: [], reply: { ok: true, status: 200, body: { ref: "qfk:" + "1".repeat(32) } },
  async fetchApi(route, options) {
    this.calls.push({ route, method: options.method, body: JSON.parse(options.body) });
    const r = this.reply;
    return { ok: r.ok, status: r.status, json: async () => r.body, text: async () => JSON.stringify(r.body) };
  },
};
"""
_DRIVER = """const store = new Map();
globalThis.localStorage = { getItem: k => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)),
                            removeItem: k => store.delete(k) };
globalThis.document = { createElement(tag) {
  return { tagName: tag.toUpperCase(), type: "text", value: "", listeners: {}, selected: false,
           addEventListener(ev, fn) { (this.listeners[ev] ??= []).push(fn); }, setAttribute(k, v) { this[k] = v; },
           select() { this.selected = true; }, fire(ev) { for (const fn of this.listeners[ev] ?? []) fn({ target: this }); } };
} };
const { app } = await import("./scripts/app.js");
const { api } = await import("./scripts/api.js");
await import("./extensions/ComfyUI-QuantFunc/quantfunc_api_key.js");
// the frontend's addDOMWidget(name, type, element, options): the value is options.getValue()/setValue()
function node(comfyClass) {
  return { comfyClass, widgets: [], addDOMWidget(name, type, element, options) {
    const w = { name, type, element, options };
    Object.defineProperty(w, "value", { get() { return options.getValue(); }, set(v) { options.setValue(v); } });
    this.widgets.push(w);
    return w;
  } };
}
async function created(comfyClass) {
  const n = node(comfyClass);
  for (const e of app.extensions) await e.nodeCreated?.(n);
  return n;
}
// what /object_info declares: the loaders' hidden inputs, a QuantFunc loader added later, and nodes that must not get one
const LOADERS = %s;
const withKey = { api_key: ["STRING", {}] };
const defs = [...LOADERS, "QuantFuncFutureLoader"].map(name => ({ name, input: { required: {}, hidden: { ...withKey, log_level: ["STRING", {}] } } }))
  .concat([{ name: "QuantFuncNativeLoRA", input: { required: {} } }, { name: "KSampler", input: { required: {} } },
           { name: "OtherPackApiNode", input: { required: {}, hidden: withKey } }]);
for (const d of defs) for (const e of app.extensions) await e.beforeRegisterNodeDef?.(function () {}, d, app);
const out = { loaders: {}, others: {} };
for (const c of [...LOADERS, "QuantFuncFutureLoader"]) {
  const n = await created(c);
  out.loaders[c] = n.widgets.map(w => ({ name: w.name, serialize: w.serialize }));
}
// a user typing into the field: focus (the whole key, masked, all selected), type, and later leave it
function type(w, text) { w.element.fire("focus"); w.element.value = text; w.element.fire("input"); }
function leave(w) { w.element.fire("blur"); }
const look = w => ({ type: w.element.type, shows: w.element.value, key: w.value });
for (const c of ["KSampler", "QuantFuncNativeLoRA", "OtherPackApiNode"]) out.others[c] = (await created(c)).widgets.length;
const w = (await created(LOADERS[0])).widgets[0];
out.fresh = look(w);
type(w, "  \\t"); leave(w);
out.empty = { queued: await w.serializeValue(), calls: api.calls.length };
type(w, "  %s\\n");
out.editing = { ...look(w), selected: w.element.selected };
leave(w);
out.idle = look(w);
out.key = { queued: await w.serializeValue(), calls: api.calls.slice() };
w.element.fire("focus");
out.refocused = look(w);
type(w, "%s");                                   // queued while still being edited (e.g. Ctrl+Enter): the typed key goes
out.queuedWhileEditing = { queued: await w.serializeValue(), sent: api.calls.at(-1).body.key };
leave(w);
out.short = {};
for (const text of ["abc", "qf_1234", "qf_" + "9".repeat(8), "qf_" + "9".repeat(9)]) { type(w, text); leave(w); out.short[text] = w.element.value; }
type(w, "%s"); leave(w);
api.reply = { ok: false, status: 404, body: {} };
try { out.refused = { queued: await w.serializeValue() }; } catch (e) { out.refused = { threw: true }; }
out.remembered = look((await created(LOADERS[1])).widgets[0]);
type(w, ""); leave(w);
out.forgotten = look((await created(LOADERS[2])).widgets[0]);
out.stored = localStorage.getItem("QuantFunc.api_key");
console.log(JSON.stringify(out));
"""


class FieldScript(unittest.TestCase):
    """C1-C4: the real web/quantfunc_api_key.js against the frontend surface it uses (stubbed)."""
    @classmethod
    def setUpClass(cls):
        cls.out = None
        node = shutil.which("node")
        if node is None:
            return
        d = tempfile.mkdtemp(prefix="qf_apikey_js_")
        try:
            (Path(d) / "package.json").write_text('{"type": "module"}', encoding="utf-8")
            (Path(d) / "scripts").mkdir()
            (Path(d) / "scripts" / "app.js").write_text(_APP_STUB, encoding="utf-8")
            (Path(d) / "scripts" / "api.js").write_text(_API_STUB, encoding="utf-8")
            ext = Path(d) / "extensions" / "ComfyUI-QuantFunc"
            ext.mkdir(parents=True)
            shutil.copy(PLUGIN / "web" / "quantfunc_api_key.js", ext)
            (Path(d) / "driver.mjs").write_text(_DRIVER % (json.dumps(list(LOADERS)), KEY, KEY2, KEY), encoding="utf-8")
            r = subprocess.run([node, "driver.mjs"], cwd=d, capture_output=True, encoding="utf-8", timeout=60)
            if r.returncode != 0:
                raise AssertionError(f"node driver failed: {r.stderr[-2000:]}")
            cls.out = json.loads(r.stdout.strip().splitlines()[-1])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def setUp(self):
        if self.out is None:
            skip(self, "node is not installed")

    def test_exactly_the_declaring_quantfunc_nodes_get_one_field_that_is_never_saved(self):
        for c in (*LOADERS, "QuantFuncFutureLoader"):
            self.assertEqual(self.out["loaders"][c], [{"name": "api_key", "serialize": False}], c)
        self.assertEqual(self.out["others"], {"KSampler": 0, "QuantFuncNativeLoRA": 0, "OtherPackApiNode": 0})

    def test_only_a_reference_is_queued(self):
        self.assertEqual(self.out["empty"], {"queued": "", "calls": 0})
        self.assertEqual(self.out["key"]["queued"], "qfk:" + "1" * 32)
        self.assertEqual(self.out["key"]["calls"], [{"route": "/quantfunc/api_key", "method": "POST", "body": {"key": KEY}}])
        self.assertEqual(self.out["queuedWhileEditing"], {"queued": "qfk:" + "1" * 32, "sent": KEY2})

    def test_a_refused_post_fails_the_queue(self):
        self.assertEqual(self.out["refused"], {"threw": True})

    def test_the_key_is_remembered_in_this_browser_until_the_field_is_emptied(self):
        self.assertEqual(self.out["remembered"], {"type": "text", "shows": SHORT, "key": KEY})   # visibly set
        self.assertEqual(self.out["forgotten"], {"type": "text", "shows": "", "key": ""})
        self.assertIsNone(self.out["stored"])

    def test_the_field_shows_the_key_shortened_unless_it_is_being_edited(self):
        self.assertEqual(self.out["fresh"], {"type": "text", "shows": "", "key": ""})
        self.assertEqual(self.out["editing"], {"type": "password", "shows": "  " + KEY + "\n", "key": "  " + KEY + "\n",
                                              "selected": True})
        self.assertEqual(self.out["idle"], {"type": "text", "shows": SHORT, "key": KEY})
        self.assertEqual(self.out["refocused"], {"type": "password", "shows": KEY, "key": KEY})
        self.assertEqual(self.out["short"], {"abc": "…", "qf_1234": "qf_…", "qf_" + "9" * 8: "qf_…",
                                             "qf_" + "9" * 9: "qf_9999…9999"})


if __name__ == "__main__":
    result = unittest.main(argv=[__file__], exit=False, verbosity=2).result   # sys.argv may carry comfy's --cpu
    print(f"API_KEY: {'PASS' if result.wasSuccessful() else 'FAIL'} ({result.testsRun} arms)")
    raise SystemExit(0 if result.wasSuccessful() else 1)
