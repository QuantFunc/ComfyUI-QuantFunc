#!/usr/bin/env python3
"""Contract tests for blinded _ESTIMATE_CACHE (U-C10), depth-1 safetensors key scan (U-C11),
and background zero-GPU OS page-cache prefetch (U-C12).
"""
import ast
import contextlib
import copy
import ctypes
import errno
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PKG_NAME = "qf_native_cache_pf_test"

_COMFY_ROOT = os.environ.get("COMFY_ROOT")
if not _COMFY_ROOT or not (Path(_COMFY_ROOT) / "comfy/model_management.py").is_file():
    print("[SKIP] estimate_cache_and_prefetch_test: set COMFY_ROOT to the tested host")
    raise SystemExit(77)
if _COMFY_ROOT not in sys.path:
    sys.path.insert(0, _COMFY_ROOT)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()


def _load_plugin():
    if PKG_NAME in sys.modules:
        return sys.modules[PKG_NAME], sys.modules[f"{PKG_NAME}.qf_ltx_modelpatcher"]
    spec = importlib.util.spec_from_file_location(
        PKG_NAME, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PKG_NAME] = mod
    spec.loader.exec_module(mod)
    ltx_mod = importlib.import_module(f"{PKG_NAME}.qf_ltx_modelpatcher")
    return mod, ltx_mod


plugin, qf_ltx = _load_plugin()


def _write_safetensors(path, header_dict, trailing=b""):
    hdr_bytes = json.dumps(header_dict).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        fh.write(trailing)


class EstimateCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.weights = Path(self.tmp.name) / "model.safetensors"
        self.weights.write_bytes(b"weights-v1")
        self.conn = Path(self.tmp.name) / "connectors.safetensors"
        self.conn.write_bytes(b"conn-v1")
        with plugin._ESTIMATE_CACHE_LOCK:
            plugin._ESTIMATE_CACHE.clear()
        with plugin._ENGINE_IDENTITY_LOCK:
            plugin._PIPELINE_CACHE.clear()
            plugin._PIPELINE_MODELS.clear()
            plugin._PIPELINE_SIGS.clear()

    def tearDown(self):
        with plugin._ESTIMATE_CACHE_LOCK:
            plugin._ESTIMATE_CACHE.clear()
        with plugin._ENGINE_IDENTITY_LOCK:
            plugin._PIPELINE_CACHE.clear()
            plugin._PIPELINE_MODELS.clear()
            plugin._PIPELINE_SIGS.clear()
        self.tmp.cleanup()

    def test_estimate_then_get_engine_calls_estimate_vram_once(self):
        est_calls = []

        def fake_estimate(params_ptr, est_ptr):
            est_calls.append(json.loads(params_ptr._obj.config_json.decode("utf-8")))
            est_ptr._obj.resident_weight_bytes = 4 * 1024 * 1024 * 1024
            est_ptr._obj.min_offload_bytes = 1 * 1024 * 1024 * 1024
            est_ptr._obj.working_set_floor_bytes = 512 * 1024 * 1024
            est_ptr._obj.recommended_offload_mode = 0
            return plugin.qfe.QF_OK

        fake_lib = types.SimpleNamespace(
            qf_xfm_estimate_vram=fake_estimate,
            qf_last_error=lambda: b"",
        )
        created = []

        def fake_create(lib, *, create_params, capacity_bytes=0):
            created.append(capacity_bytes)
            h = plugin.qfe.QFEngineHandle(lib, ctypes.c_void_p(0x7001), capacity_bytes=capacity_bytes)
            h.resident_vram_bytes = lambda: capacity_bytes
            return h

        cfg = {"model_arch": "krea2", "connectors_path": str(self.conn)}
        with mock.patch.object(plugin.qfe, "_LIB", fake_lib), \
             mock.patch.object(plugin.qfe, "load_lib", return_value=fake_lib), \
             mock.patch.object(plugin.qfe, "loaded_so_path", return_value="/fake/libquantfunc.so"), \
             mock.patch.object(plugin.qfe, "library_identity", return_value="img-1"), \
             mock.patch.object(plugin.qfe, "_refuse_second_arch", lambda *_a, **_kw: None), \
             mock.patch.object(plugin, "_read_auth", side_effect=lambda k=None: (k or "qf_env_default_1234", "https://srv.example")), \
             mock.patch.object(plugin.qfe.QFEngineHandle, "create", side_effect=fake_create):
            cold = plugin._estimate_engine(str(self.weights), cfg, device_id=0, api_key=None)
            self.assertEqual(cold.bytes, 4 * 1024 * 1024 * 1024)
            self.assertEqual(len(est_calls), 1)
            eng, _ckey = plugin._get_engine(str(self.weights), cfg, device_id=0, api_key=None)
            self.assertEqual(len(est_calls), 1, "_get_engine must hit _ESTIMATE_CACHE after _estimate_engine")
            self.assertEqual(created, [4 * 1024 * 1024 * 1024])
            self.assertIsNotNone(eng.xfm)

    def test_invalidation_on_mtime_connectors_backend_credentials_and_no_plaintext(self):
        est_calls = []

        def fake_estimate(params_ptr, est_ptr):
            est_calls.append(json.loads(params_ptr._obj.config_json.decode("utf-8")))
            est_ptr._obj.resident_weight_bytes = 2 * 1024 * 1024 * 1024
            est_ptr._obj.min_offload_bytes = 512 * 1024 * 1024
            est_ptr._obj.working_set_floor_bytes = 256 * 1024 * 1024
            est_ptr._obj.recommended_offload_mode = 0
            return plugin.qfe.QF_OK

        fake_lib = types.SimpleNamespace(qf_xfm_estimate_vram=fake_estimate, qf_last_error=lambda: b"")
        cfg = {"model_arch": "ltx25", "connectors_path": str(self.conn)}
        key_a = "qf_secret_alpha_1111111111111111"
        key_b = "qf_secret_beta_2222222222222222"
        env_auth = ["qf_env_key_3333333333333333", "https://a.example"]

        with mock.patch.object(plugin.qfe, "_LIB", fake_lib), \
             mock.patch.object(plugin.qfe, "_refuse_second_arch", lambda *_a, **_kw: None), \
             mock.patch.object(plugin, "_read_auth", side_effect=lambda k=None: (k, env_auth[1]) if k else tuple(env_auth)), \
             mock.patch.dict(os.environ, {"QF_BACKEND": "cuda"}, clear=False):
            # 1) Initial call with key_a -> miss (1)
            plugin._estimate_engine(str(self.weights), cfg, device_id=0, api_key=key_a)
            self.assertEqual(len(est_calls), 1)
            # Repeat same inputs -> hit (1)
            plugin._estimate_engine(str(self.weights), cfg, device_id=0, api_key=key_a)
            self.assertEqual(len(est_calls), 1)

            # 2) Different api_key -> MUST miss (2)
            plugin._estimate_engine(str(self.weights), cfg, device_id=0, api_key=key_b)
            self.assertEqual(len(est_calls), 2)

            # 3) api_key=None resolved from env/keyfile -> MUST miss (3), repeat -> hit (3)
            plugin._estimate_engine(str(self.weights), cfg, device_id=0, api_key=None)
            self.assertEqual(len(est_calls), 3)
            plugin._estimate_engine(str(self.weights), cfg, device_id=0, api_key=None)
            self.assertEqual(len(est_calls), 3)

            # 4) Revoke env/keyfile key (now empty) -> MUST miss (4)
            env_auth[0] = ""
            plugin._estimate_engine(str(self.weights), cfg, device_id=0, api_key=None)
            self.assertEqual(len(est_calls), 4)

            # 5) Change server_url with same api_key -> MUST miss (5)
            env_auth[1] = "https://b.example"
            plugin._estimate_engine(str(self.weights), cfg, device_id=0, api_key=key_a)
            self.assertEqual(len(est_calls), 5)

            # 6) Change weights mtime_ns -> MUST miss (6)
            st = os.stat(self.weights)
            os.utime(self.weights, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
            plugin._estimate_engine(str(self.weights), cfg, device_id=0, api_key=key_a)
            self.assertEqual(len(est_calls), 6)

            # 7) Change connectors_path mtime_ns -> MUST miss (7)
            cst = os.stat(self.conn)
            os.utime(self.conn, ns=(cst.st_atime_ns, cst.st_mtime_ns + 1_000_000))
            plugin._estimate_engine(str(self.weights), cfg, device_id=0, api_key=key_a)
            self.assertEqual(len(est_calls), 7)

            # 8) Change QF_BACKEND -> MUST miss (8)
            os.environ["QF_BACKEND"] = "rocm"
            plugin._estimate_engine(str(self.weights), cfg, device_id=0, api_key=key_a)
            self.assertEqual(len(est_calls), 8)

            # 9) Zero plaintext secrets anywhere in _ESTIMATE_CACHE keys
            keys_dump = repr(list(plugin._ESTIMATE_CACHE.keys()))
            for secret in (key_a, key_b, "qf_env_key_3333333333333333", "qf_secret", "https://a.example", "https://b.example"):
                self.assertNotIn(secret, keys_dump)

    def test_failed_estimate_not_cached_and_virtual_path_or_none_lib_bypass(self):
        calls = [0]

        def fake_estimate_fail(_params_ptr, _est_ptr):
            calls[0] += 1
            return plugin.qfe.QF_ERR_AUTH

        fake_lib_fail = types.SimpleNamespace(
            qf_xfm_estimate_vram=fake_estimate_fail,
            qf_last_error=lambda: b"auth rejected",
        )
        with mock.patch.object(plugin.qfe, "_LIB", fake_lib_fail), \
             mock.patch.object(plugin.qfe, "_refuse_second_arch", lambda *_a, **_kw: None), \
             mock.patch.object(plugin, "_read_auth", return_value=("qf_bad_key_1234", "https://srv.example")):
            c1 = plugin._estimate_engine(str(self.weights), {"model_arch": "krea2"}, device_id=0)
            c2 = plugin._estimate_engine(str(self.weights), {"model_arch": "krea2"}, device_id=0)
            self.assertIsNone(c1.bytes)
            self.assertIsNone(c2.bytes)
            self.assertEqual(calls[0], 2, "failed ColdNeed(None, err) must never be cached")
            self.assertEqual(len(plugin._ESTIMATE_CACHE), 0)

        # Virtual / non-existent path skips cache without raising FileNotFoundError
        ok_calls = [0]

        def fake_estimate_ok(_params_ptr, est_ptr):
            ok_calls[0] += 1
            est_ptr._obj.resident_weight_bytes = 1024
            return plugin.qfe.QF_OK

        fake_lib_ok = types.SimpleNamespace(qf_xfm_estimate_vram=fake_estimate_ok, qf_last_error=lambda: b"")
        with mock.patch.object(plugin.qfe, "_LIB", fake_lib_ok), \
             mock.patch.object(plugin.qfe, "_refuse_second_arch", lambda *_a, **_kw: None), \
             mock.patch.object(plugin, "_read_auth", return_value=("", "https://srv.example")):
            c_virt1 = plugin._estimate_engine("/virtual/does_not_exist.safetensors", {"model_arch": "krea2"})
            c_virt2 = plugin._estimate_engine("/virtual/does_not_exist.safetensors", {"model_arch": "krea2"})
            self.assertEqual(c_virt1.bytes, 1024)
            self.assertEqual(c_virt2.bytes, 1024)
            self.assertEqual(ok_calls[0], 2)
            self.assertEqual(len(plugin._ESTIMATE_CACHE), 0)

        # qfe._LIB is None bypasses cache cleanly
        ok_calls[0] = 0
        with mock.patch.object(plugin.qfe, "_LIB", None), \
             mock.patch.object(plugin.qfe, "_refuse_second_arch", lambda *_a, **_kw: None), \
             mock.patch.object(plugin, "_should_autoload_for_estimate", return_value=True), \
             mock.patch.object(plugin.qfe, "has_ready_so", return_value=True), \
             mock.patch.object(plugin.qfe, "load_lib", return_value=fake_lib_ok), \
             mock.patch.object(plugin, "_read_auth", return_value=("", "https://srv.example")):
            plugin._estimate_engine(str(self.weights), {"model_arch": "krea2"})
            plugin._estimate_engine(str(self.weights), {"model_arch": "krea2"})
            self.assertEqual(ok_calls[0], 2)
            self.assertEqual(len(plugin._ESTIMATE_CACHE), 0)

    def test_estimate_cache_bounded_capacity(self):
        def fake_estimate_ok(_params_ptr, est_ptr):
            est_ptr._obj.resident_weight_bytes = 2048
            return plugin.qfe.QF_OK

        fake_lib = types.SimpleNamespace(qf_xfm_estimate_vram=fake_estimate_ok, qf_last_error=lambda: b"")
        with mock.patch.object(plugin.qfe, "_LIB", fake_lib), \
             mock.patch.object(plugin.qfe, "_refuse_second_arch", lambda *_a, **_kw: None), \
             mock.patch.object(plugin, "_read_auth", side_effect=lambda k=None: (k or "", "https://srv.example")):
            for i in range(plugin._ESTIMATE_CACHE_MAX_ENTRIES + 5):
                plugin._estimate_engine(str(self.weights), {"model_arch": "krea2"}, api_key=f"qf_cap_{i:04d}")
            self.assertLessEqual(len(plugin._ESTIMATE_CACHE), plugin._ESTIMATE_CACHE_MAX_ENTRIES)

    def test_device_id_isolation_nonsecret_cfg_invalidation_and_missing_connectors_bypass(self):
        est_calls = []

        def fake_estimate(params_ptr, est_ptr):
            est_calls.append((params_ptr._obj.device_id, json.loads(params_ptr._obj.config_json.decode("utf-8"))))
            est_ptr._obj.resident_weight_bytes = 3 * 1024 * 1024 * 1024
            est_ptr._obj.min_offload_bytes = 512 * 1024 * 1024
            est_ptr._obj.working_set_floor_bytes = 256 * 1024 * 1024
            est_ptr._obj.recommended_offload_mode = 0
            return plugin.qfe.QF_OK

        fake_lib = types.SimpleNamespace(qf_xfm_estimate_vram=fake_estimate, qf_last_error=lambda: b"")
        with mock.patch.object(plugin.qfe, "_LIB", fake_lib), \
             mock.patch.object(plugin.qfe, "_refuse_second_arch", lambda *_a, **_kw: None), \
             mock.patch.object(plugin, "_read_auth", return_value=("", "https://srv.example")):
            base_cfg = {"model_arch": "krea2", "offload_mode": 0, "vram_budget_mb": 8192}
            plugin._estimate_engine(str(self.weights), base_cfg, device_id=0)
            self.assertEqual(len(est_calls), 1)

            # Reordered identical non-secret cfg on same device_id -> hit (1)
            plugin._estimate_engine(
                str(self.weights), {"vram_budget_mb": 8192, "offload_mode": 0, "model_arch": "krea2"}, device_id=0)
            self.assertEqual(len(est_calls), 1)

            # Different device_id != 0 -> MUST miss (2), repeat -> hit (2)
            plugin._estimate_engine(str(self.weights), base_cfg, device_id=1)
            self.assertEqual(len(est_calls), 2)
            plugin._estimate_engine(str(self.weights), base_cfg, device_id=1)
            self.assertEqual(len(est_calls), 2)

            p0 = plugin._prepare_params(str(self.weights), base_cfg, 0)
            p1 = plugin._prepare_params(str(self.weights), base_cfg, 1)
            self.assertNotEqual(
                plugin._estimate_cache_key(str(self.weights), p0, 0),
                plugin._estimate_cache_key(str(self.weights), p1, 1),
            )

            # Non-secret create_cfg mutations (offload_mode / vram_budget_mb) -> MUST miss (3, 4)
            plugin._estimate_engine(
                str(self.weights), {"model_arch": "krea2", "offload_mode": 1, "vram_budget_mb": 8192}, device_id=0)
            self.assertEqual(len(est_calls), 3)
            plugin._estimate_engine(
                str(self.weights), {"model_arch": "krea2", "offload_mode": 1, "vram_budget_mb": 4096}, device_id=0)
            self.assertEqual(len(est_calls), 4)

            # Non-existent connectors_path -> key returns None and bypasses cache without populating it
            cache_count_before = len(plugin._ESTIMATE_CACHE)
            missing_conn_cfg = {
                "model_arch": "ltx25",
                "connectors_path": str(Path(self.tmp.name) / "missing_connectors.safetensors"),
            }
            p_missing = plugin._prepare_params(str(self.weights), missing_conn_cfg, 0)
            self.assertIsNone(plugin._estimate_cache_key(str(self.weights), p_missing, 0))
            plugin._estimate_engine(str(self.weights), missing_conn_cfg, device_id=0)
            plugin._estimate_engine(str(self.weights), missing_conn_cfg, device_id=0)
            self.assertEqual(len(est_calls), 6)
            self.assertEqual(len(plugin._ESTIMATE_CACHE), cache_count_before)

            # Invalid weights_path and str/None/corrupt/non-dict config_json variants
            self.assertIsNone(plugin._estimate_cache_key(None, p0, 0))
            self.assertIsNone(plugin._estimate_cache_key(123, p0, 0))
            self.assertIsNotNone(plugin._estimate_cache_key(str(self.weights), types.SimpleNamespace(config_json=None), 0))
            self.assertIsNotNone(plugin._estimate_cache_key(str(self.weights), types.SimpleNamespace(config_json='{"a":1}'), 0))
            self.assertIsNone(plugin._estimate_cache_key(str(self.weights), types.SimpleNamespace(config_json=123), 0))
            self.assertIsNone(plugin._estimate_cache_key(str(self.weights), types.SimpleNamespace(config_json="{bad"), 0))
            self.assertIsNone(plugin._estimate_cache_key(str(self.weights), types.SimpleNamespace(config_json="[1,2]"), 0))

    def test_ast_extracted_get_engine_works_with_only_11_keys(self):
        """Regression guard for load_time_lora_test / public_chain_cache_test isolated 11-key engine_env."""
        src = (PLUGIN_DIR / "__init__.py").read_text(encoding="utf-8")
        tree = ast.parse(src)

        def _fn(name):
            return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)

        class _H:
            @classmethod
            def create(cls, lib, *, create_params, capacity_bytes=0):
                return types.SimpleNamespace(lib=lib, xfm=object(), capacity_bytes=capacity_bytes)

        qfe_ns = types.SimpleNamespace(
            load_lib=lambda *_a, **_k: "lib",
            loaded_so_path=lambda: "engine.so",
            library_identity=lambda _lib: "img",
            estimate_vram=lambda _lib, _params: types.SimpleNamespace(bytes=64),
            QFEngineHandle=_H,
        )
        qfmp_ns = types.SimpleNamespace(_current_engine_cache_acquisition=lambda: None)
        env = {
            "os": os,
            "json": json,
            "threading": threading,
            "qfe": qfe_ns,
            "qfmp": qfmp_ns,
            "_ENGINE_IDENTITY_LOCK": threading.RLock(),
            "_CREATE_LOCKS": {},
            "_PIPELINE_CACHE": {},
            "_prepare_params": lambda w, cfg, dev, api_key=None: types.SimpleNamespace(
                weights_path=w, device_id=dev, observed_cfg=copy.deepcopy(cfg)),
            "_pin_pipeline_consumer_locked": lambda _k: None,
            "_sweep_dead_pipelines": lambda _k: None,
        }
        exec(compile(ast.Module(body=[copy.deepcopy(_fn(n)) for n in ("_create_lock_for", "_engine_recipe", "_get_engine")],
                                type_ignores=[]), "<ast>", "exec"), env)
        eng, ckey = env["_get_engine"](str(self.weights), {"model_arch": "krea2"})
        self.assertIsNotNone(eng.xfm)
        eng2, ckey2 = env["_get_engine"](str(self.weights), {"model_arch": "krea2"})
        self.assertIs(eng, eng2)
        self.assertEqual(ckey, ckey2)


class Depth1SafetensorsScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_depth1_key_scan_zero_json_loads_and_spoof_rejection(self):
        p_wrapped = Path(self.tmp.name) / "wrapped.safetensors"
        _write_safetensors(p_wrapped, {
            "__metadata__": {"note": "escaped \\\" } { ["},
            "model.diffusion_model.video_embeddings_connector.blocks.0.weight": {
                "dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8],
            },
        })

        p_spoof = Path(self.tmp.name) / "spoof.safetensors"
        _write_safetensors(p_spoof, {
            "__metadata__": {
                "model.diffusion_model.video_embeddings_connector.fake": "1",
                "tricky": 'a\\"}{"model.diffusion_model.video_embeddings_connector.x": {}',
            },
            "other.weight": {
                "dtype": "model.diffusion_model.video_embeddings_connector.dtype_spoof",
                "shape": [1],
                "data_offsets": [0, 2],
            },
        })

        p_after_meta = Path(self.tmp.name) / "after_meta.safetensors"
        _write_safetensors(p_after_meta, {
            "__metadata__": {
                "escaped_quote_and_brace": 'x\\"}\\"][{\\"video_embeddings_connector.fake\\":1}',
                "nested": {"deep": "video_embeddings_connector.fake"},
            },
            "unrelated.bias": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "video_embeddings_connector.learnable_registers": {"dtype": "BF16", "shape": [1], "data_offsets": [16, 18]},
        })

        p_many = Path(self.tmp.name) / "many.safetensors"
        big_hdr = {f"blocks.{i}.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]} for i in range(2000)}
        big_hdr["model.diffusion_model.audio_embeddings_connector.proj.weight"] = {
            "dtype": "BF16", "shape": [1], "data_offsets": [0, 2],
        }
        _write_safetensors(p_many, big_hdr)

        p_trunc_hdr = Path(self.tmp.name) / "trunc_hdr.safetensors"
        p_trunc_hdr.write_bytes(struct.pack("<Q", 200) + b'{"model.diffusion_model.video_embeddings_connector.w":')

        p_trunc_len = Path(self.tmp.name) / "trunc_len.safetensors"
        p_trunc_len.write_bytes(b"\x01\x02\x03")

        p_huge = Path(self.tmp.name) / "huge.safetensors"
        p_huge.write_bytes(struct.pack("<Q", qf_ltx._ST_HEADER_MAX_BYTES + 1) + b"{}")

        def forbid_json_loads(*_a, **_k):
            raise AssertionError("json.loads must never be called by _file_has_prefix")

        with mock.patch.object(json, "loads", side_effect=forbid_json_loads):
            # Wrapped key matches both full prefix and unwrapped prefix
            self.assertTrue(qf_ltx._file_has_prefix(str(p_wrapped), qf_ltx._CONN_VID_PREFIX))
            self.assertTrue(qf_ltx._file_has_prefix(str(p_wrapped), "video_embeddings_connector."))
            self.assertFalse(qf_ltx._file_has_prefix(str(p_wrapped), qf_ltx._CONN_AUD_PREFIX))

            # Metadata and value-object spoofs are strictly rejected
            self.assertFalse(qf_ltx._file_has_prefix(str(p_spoof), qf_ltx._CONN_VID_PREFIX))
            self.assertFalse(qf_ltx._file_has_prefix(str(p_spoof), "video_embeddings_connector."))

            # Escaped quotes/braces inside __metadata__ do not derail subsequent real depth-1 keys
            self.assertTrue(qf_ltx._file_has_prefix(str(p_after_meta), "video_embeddings_connector."))
            self.assertFalse(qf_ltx._file_has_prefix(str(p_after_meta), qf_ltx._CONN_VID_PREFIX))

            # Large header match at tail
            self.assertTrue(qf_ltx._file_has_prefix(str(p_many), qf_ltx._CONN_AUD_PREFIX))
            self.assertTrue(qf_ltx._file_has_prefix(str(p_many), "audio_embeddings_connector."))

            # Truncated, oversized, zero-length, missing, or empty prefix all return False
            p_zero_hdr = Path(self.tmp.name) / "zero_hdr.safetensors"
            p_zero_hdr.write_bytes(struct.pack("<Q", 0) + b'{"video_embeddings_connector.w": {}}')
            p_zero_only = Path(self.tmp.name) / "zero_only.safetensors"
            p_zero_only.write_bytes(struct.pack("<Q", 0))

            self.assertFalse(qf_ltx._file_has_prefix(str(p_trunc_hdr), qf_ltx._CONN_VID_PREFIX))
            self.assertFalse(qf_ltx._file_has_prefix(str(p_trunc_len), qf_ltx._CONN_VID_PREFIX))
            self.assertFalse(qf_ltx._file_has_prefix(str(p_huge), qf_ltx._CONN_VID_PREFIX))
            self.assertFalse(qf_ltx._file_has_prefix(str(p_zero_hdr), qf_ltx._CONN_VID_PREFIX))
            self.assertFalse(qf_ltx._file_has_prefix(str(p_zero_only), "video_embeddings_connector."))
            self.assertFalse(qf_ltx._file_has_prefix(str(Path(self.tmp.name) / "missing.safetensors"), qf_ltx._CONN_VID_PREFIX))
            self.assertFalse(qf_ltx._file_has_prefix(str(p_wrapped), ""))

    def test_depth1_scalars_strings_and_double_backslash_quote_escapes(self):
        p_scalar_match = Path(self.tmp.name) / "scalar_match.safetensors"
        _write_safetensors(p_scalar_match, {
            "__metadata__": {"even_bs": "a\\", "spoof": 'x\\\\"video_embeddings_connector.fake'},
            "k": 123,
            "s": "abc\\",
            "b": True,
            "flt": -1.5e2,
            "nul": None,
            "spoof_str": "video_embeddings_connector.str_spoof",
            "key_bs\\": 456,
            "video_embeddings_connector.real": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
        })

        p_scalar_spoof_only = Path(self.tmp.name) / "scalar_spoof_only.safetensors"
        _write_safetensors(p_scalar_spoof_only, {
            "__metadata__": {"even_bs": "a\\", "video_embeddings_connector.meta": "1"},
            "k": 123,
            "s": "video_embeddings_connector.in_str_value",
            "s_esc": "abc\\",
            "b": False,
            "nul": None,
        })

        with mock.patch.object(json, "loads", side_effect=AssertionError("json.loads must never be called")):
            self.assertTrue(qf_ltx._file_has_prefix(str(p_scalar_match), "video_embeddings_connector."))
            self.assertFalse(qf_ltx._file_has_prefix(str(p_scalar_match), qf_ltx._CONN_AUD_PREFIX))
            self.assertFalse(qf_ltx._file_has_prefix(str(p_scalar_spoof_only), "video_embeddings_connector."))
            self.assertFalse(qf_ltx._file_has_prefix(str(p_scalar_spoof_only), qf_ltx._CONN_VID_PREFIX))

        # Direct check on _iter_safetensors_top_keys unescaping of trailing escaped backslash and other escapes
        raw = json.dumps({"k": 123, "s": "abc\\", "b": True, "key_bs\\": 7, "line\nkey": 8, "ok": {}}).encode("utf-8")
        self.assertEqual(
            list(qf_ltx._iter_safetensors_top_keys(raw)),
            [b"k", b"s", b"b", b"key_bs\\", b"line\\nkey", b"ok"],
        )
        # Malformed/truncated header bodies return without error
        for bad_raw in (b"", b"[]", b"{123: 1}", b'{"a', b'{"a": "unterm', b'{"a": {"b": [1}', b'{"a" "b"}'):
            self.assertIsInstance(list(qf_ltx._iter_safetensors_top_keys(bad_raw)), list)


class OsPageCachePrefetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.small = Path(self.tmp.name) / "small.safetensors"
        self.small.write_bytes(b"x" * 1024)
        self.medium = Path(self.tmp.name) / "medium.safetensors"
        self.medium.write_bytes(b"A" * (128 * 1024))

    def tearDown(self):
        for p in (self.small, self.medium):
            plugin._cancel_os_pagecache_prefetch(str(p))
        self.tmp.cleanup()

    def test_prefetch_reads_with_zero_gpu_and_respects_ram_size_platform_guards(self):
        # 1) Small file under default 16 MiB threshold -> skipped
        with mock.patch.object(plugin, "_linux_mem_available_bytes", return_value=64 * 1024 ** 3):
            self.assertFalse(plugin._start_os_pagecache_prefetch(str(self.small)))
            self.assertFalse(plugin._start_os_pagecache_prefetch(str(Path(self.tmp.name) / "missing.bin")))
            self.assertFalse(plugin._start_os_pagecache_prefetch(self.tmp.name))

        # 2) Low MemAvailable (< 2 * file_size + reserve) or None -> skipped
        with mock.patch.object(plugin, "_linux_mem_available_bytes", return_value=1024):
            self.assertFalse(plugin._start_os_pagecache_prefetch(str(self.medium), min_bytes=64 * 1024))
        with mock.patch.object(plugin, "_linux_mem_available_bytes", return_value=None), \
             mock.patch.object(os, "open", side_effect=AssertionError("os.open must not run when mem_avail is None")):
            self.assertFalse(plugin._start_os_pagecache_prefetch(str(self.medium), min_bytes=64 * 1024))

        # 3) Non-Linux platform -> skipped
        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch.object(plugin, "_linux_mem_available_bytes", return_value=64 * 1024 ** 3):
            self.assertFalse(plugin._start_os_pagecache_prefetch(str(self.medium), min_bytes=64 * 1024))

        # 4) Positive run on Linux with sufficient RAM: reads full file with 0 CUDA calls and cleans up
        if sys.platform.startswith("linux"):
            pread_bytes = [0]
            orig_pread = os.pread

            def spy_pread(fd, length, offset):
                data = orig_pread(fd, length, offset)
                pread_bytes[0] += len(data)
                return data

            fadv_calls = []
            orig_fadv = getattr(os, "posix_fadvise", None)

            def spy_fadv(fd, off, length, advice):
                fadv_calls.append((off, length, advice))
                if orig_fadv is not None:
                    return orig_fadv(fd, off, length, advice)
                return None

            import torch
            with mock.patch.object(plugin, "_linux_mem_available_bytes", return_value=64 * 1024 ** 3), \
                 mock.patch.object(os, "pread", side_effect=spy_pread), \
                 mock.patch.object(os, "posix_fadvise", side_effect=spy_fadv), \
                 mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("prefetch must not touch CUDA")):
                started = plugin._start_os_pagecache_prefetch(
                    str(self.medium), min_bytes=64 * 1024, ram_reserve_bytes=1024, chunk_bytes=16 * 1024, num_threads=4)
                self.assertTrue(started)
                # Wait for completion
                st = os.stat(self.medium)
                key = (int(st.st_dev), int(st.st_ino))
                with plugin._OS_PREFETCH_LOCK:
                    entry = plugin._OS_PREFETCH_ACTIVE.get(key)
                if entry is not None:
                    self.assertTrue(entry["done"].wait(timeout=5.0))
                    for t in entry["threads"]:
                        t.join(timeout=5.0)
            self.assertEqual(pread_bytes[0], 128 * 1024)
            self.assertEqual(len(fadv_calls), 1)
            self.assertEqual(fadv_calls[0][2], os.POSIX_FADV_SEQUENTIAL)
            with plugin._OS_PREFETCH_LOCK:
                self.assertNotIn(key, plugin._OS_PREFETCH_ACTIVE)

    def test_prefetch_inode_dedup_and_clean_cancel(self):
        if not sys.platform.startswith("linux"):
            return
        sym = Path(self.tmp.name) / "symlink.safetensors"
        os.symlink(self.medium, sym)

        entered = threading.Event()
        release = threading.Event()
        orig_pread = os.pread

        def gated_pread(fd, length, offset):
            entered.set()
            release.wait(timeout=5.0)
            return orig_pread(fd, length, offset)

        with mock.patch.object(plugin, "_linux_mem_available_bytes", return_value=64 * 1024 ** 3), \
             mock.patch.object(os, "pread", side_effect=gated_pread):
            ok1 = plugin._start_os_pagecache_prefetch(
                str(self.medium), min_bytes=64 * 1024, ram_reserve_bytes=1024, chunk_bytes=16 * 1024, num_threads=4)
            self.assertTrue(ok1)
            self.assertTrue(entered.wait(timeout=5.0))
            # Duplicate start on same path AND on symlink to same (st_dev, st_ino) must dedup
            self.assertFalse(plugin._start_os_pagecache_prefetch(
                str(self.medium), min_bytes=64 * 1024, ram_reserve_bytes=1024, chunk_bytes=16 * 1024, num_threads=4))
            self.assertFalse(plugin._start_os_pagecache_prefetch(
                str(sym), min_bytes=64 * 1024, ram_reserve_bytes=1024, chunk_bytes=16 * 1024, num_threads=4))

            st = os.stat(self.medium)
            key = (int(st.st_dev), int(st.st_ino))
            with plugin._OS_PREFETCH_LOCK:
                entry = plugin._OS_PREFETCH_ACTIVE[key]
                threads = list(entry["threads"])
                orig_stop_set = entry["stop"].set
                entry["stop"].set = lambda: (orig_stop_set(), release.set())[0]
            try:
                cancelled = plugin._cancel_os_pagecache_prefetch(str(self.medium), wait=True, timeout=5.0)
            finally:
                release.set()
            self.assertTrue(cancelled)
            self.assertTrue(all(not t.is_alive() for t in threads))
            with plugin._OS_PREFETCH_LOCK:
                self.assertNotIn(key, plugin._OS_PREFETCH_ACTIVE)
            # Second cancel is a clean no-op
            self.assertFalse(plugin._cancel_os_pagecache_prefetch(str(self.medium)))

    def test_helpers_available_when_import_ok_is_false(self):
        """Prefetch and estimate-cache helpers live outside `if _IMPORT_OK:` (text_encoding_test compat)."""
        script = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('qf_no_comfy', {str(PLUGIN_DIR / '__init__.py')!r}, "
            f"submodule_search_locations=[{str(PLUGIN_DIR)!r}])\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "sys.modules['qf_no_comfy'] = mod\n"
            "spec.loader.exec_module(mod)\n"
            "assert mod._IMPORT_OK is False, mod._IMPORT_OK\n"
            "assert isinstance(mod._ESTIMATE_CACHE, dict)\n"
            "assert callable(mod._start_os_pagecache_prefetch)\n"
            "assert callable(mod._cancel_os_pagecache_prefetch)\n"
            "assert callable(mod._linux_mem_available_bytes)\n"
            "print('IMPORT_OK_FALSE_PASS')\n"
        )
        env = dict(os.environ)
        env.pop("COMFY_ROOT", None)
        env["PYTHONPATH"] = ""
        r = subprocess.run([sys.executable, "-S", "-c", script], env=env, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(r.returncode, 0, f"stderr={r.stderr}\nstdout={r.stdout}")
        self.assertIn("IMPORT_OK_FALSE_PASS", r.stdout)

    def test_linux_mem_available_bytes_proc_meminfo_branches(self):
        if sys.platform.startswith("linux"):
            real_val = plugin._linux_mem_available_bytes()
            self.assertIsInstance(real_val, int)
            self.assertGreater(real_val, 0)

        import io
        orig_open = open

        def fake_open_factory(payload=None, exc=None):
            def _open(path, *args, **kwargs):
                if path == "/proc/meminfo":
                    if exc is not None:
                        raise exc
                    return io.StringIO(payload)
                return orig_open(path, *args, **kwargs)
            return _open

        with mock.patch.object(sys, "platform", "linux"):
            with mock.patch("builtins.open", side_effect=fake_open_factory(
                    "MemTotal:       65536 kB\nMemAvailable:   12345 kB\nMemFree:         1024 kB\n")):
                self.assertEqual(plugin._linux_mem_available_bytes(), 12345 * 1024)
            with mock.patch("builtins.open", side_effect=fake_open_factory(
                    "MemTotal:       65536 kB\nMemFree:         1024 kB\n")):
                self.assertIsNone(plugin._linux_mem_available_bytes())
            with mock.patch("builtins.open", side_effect=fake_open_factory(exc=OSError(errno.ENOENT, "missing"))):
                self.assertIsNone(plugin._linux_mem_available_bytes())
        with mock.patch.object(sys, "platform", "darwin"):
            self.assertIsNone(plugin._linux_mem_available_bytes())

    def test_prefetch_concurrent_start_cancel_case_a(self):
        if not sys.platform.startswith("linux"):
            return
        opened_fds = []
        closed_fds = []
        orig_open = os.open
        orig_close = os.close
        orig_pread = os.pread
        orig_start = threading.Thread.start

        def spy_open(path, flags, *args):
            fd = orig_open(path, flags, *args)
            if os.fspath(path) == str(self.medium):
                opened_fds.append(fd)
            return fd

        def spy_close(fd):
            if fd in opened_fds:
                closed_fds.append(fd)
            return orig_close(fd)

        st = os.stat(self.medium)
        key = (int(st.st_dev), int(st.st_ino))
        first_start_entered = threading.Event()
        cancel_done = threading.Event()
        release = threading.Event()
        lock_held_at_start = []
        created_threads = []
        cancel_results = []
        cancel_errors = []
        stop_wrapped = [False]

        def gated_pread(fd, length, offset):
            release.wait(timeout=5.0)
            return orig_pread(fd, length, offset)

        def spy_thread_start(thread_self):
            if thread_self.name.startswith("qf-os-prefetch-"):
                created_threads.append(thread_self)
                held = plugin._OS_PREFETCH_LOCK.locked()
                lock_held_at_start.append(held)
                if thread_self.name == "qf-os-prefetch-0":
                    entry = plugin._OS_PREFETCH_ACTIVE.get(key)
                    if entry is not None and not stop_wrapped[0]:
                        stop_wrapped[0] = True
                        orig_stop_set = entry["stop"].set
                        entry["stop"].set = lambda: (orig_stop_set(), release.set())[0]
                    first_start_entered.set()
                    if not held:
                        cancel_done.wait(timeout=1.0)
            return orig_start(thread_self)

        def canceller():
            try:
                self.assertTrue(first_start_entered.wait(timeout=5.0))
                res = plugin._cancel_os_pagecache_prefetch(str(self.medium), wait=True, timeout=5.0)
                cancel_results.append(res)
            except BaseException as exc:  # noqa: BLE001
                cancel_errors.append(exc)
            finally:
                release.set()
                cancel_done.set()

        with mock.patch.object(plugin, "_linux_mem_available_bytes", return_value=64 * 1024 ** 3), \
             mock.patch.object(os, "open", side_effect=spy_open), \
             mock.patch.object(os, "close", side_effect=spy_close), \
             mock.patch.object(os, "pread", side_effect=gated_pread), \
             mock.patch.object(threading.Thread, "start", spy_thread_start):
            ct = threading.Thread(target=canceller)
            ct.start()
            try:
                started = plugin._start_os_pagecache_prefetch(
                    str(self.medium), min_bytes=64 * 1024, ram_reserve_bytes=1024, chunk_bytes=16 * 1024, num_threads=4)
            finally:
                release.set()
                ct.join(timeout=5.0)

        self.assertTrue(started)
        self.assertEqual(cancel_errors, [])
        self.assertEqual(cancel_results, [True])
        self.assertEqual(len(created_threads), 4)
        self.assertTrue(all(lock_held_at_start), "all worker Thread.start() calls must hold _OS_PREFETCH_LOCK")
        for t in created_threads:
            t.join(timeout=5.0)
            self.assertFalse(t.is_alive())
        with plugin._OS_PREFETCH_LOCK:
            self.assertNotIn(key, plugin._OS_PREFETCH_ACTIVE)
        self.assertTrue(opened_fds)
        self.assertTrue(set(opened_fds).issubset(set(closed_fds)))

        # Stress rounds with Barrier(2) racing ungated start vs cancel
        opened_fds.clear()
        closed_fds.clear()
        for _ in range(12):
            bar = threading.Barrier(2)
            round_errors = []
            cancel_round_res = []

            def run_start():
                try:
                    bar.wait(timeout=5.0)
                    plugin._start_os_pagecache_prefetch(
                        str(self.medium), min_bytes=64 * 1024, ram_reserve_bytes=1024, chunk_bytes=16 * 1024, num_threads=4)
                except BaseException as exc:  # noqa: BLE001
                    round_errors.append(exc)

            def run_cancel():
                try:
                    bar.wait(timeout=5.0)
                    cancel_round_res.append(
                        plugin._cancel_os_pagecache_prefetch(str(self.medium), wait=True, timeout=5.0))
                except BaseException as exc:  # noqa: BLE001
                    round_errors.append(exc)

            with mock.patch.object(plugin, "_linux_mem_available_bytes", return_value=64 * 1024 ** 3), \
                 mock.patch.object(os, "open", side_effect=spy_open), \
                 mock.patch.object(os, "close", side_effect=spy_close):
                t_a = threading.Thread(target=run_start)
                t_b = threading.Thread(target=run_cancel)
                t_a.start()
                t_b.start()
                t_a.join(timeout=5.0)
                t_b.join(timeout=5.0)
                plugin._cancel_os_pagecache_prefetch(str(self.medium), wait=True, timeout=5.0)
                with plugin._OS_PREFETCH_LOCK:
                    rem = plugin._OS_PREFETCH_ACTIVE.get(key)
                if rem is not None:
                    rem["done"].wait(timeout=5.0)
                    for wt in rem["threads"]:
                        wt.join(timeout=5.0)
            self.assertEqual(round_errors, [])
            self.assertIn(cancel_round_res[0], (True, False))
        self.assertTrue(set(opened_fds).issubset(set(closed_fds)))

    def test_prefetch_partial_thread_start_failure_case_b(self):
        if not sys.platform.startswith("linux"):
            return
        opened_fds = []
        closed_fds = []
        orig_open = os.open
        orig_close = os.close
        orig_pread = os.pread
        orig_start = threading.Thread.start

        def spy_open(path, flags, *args):
            fd = orig_open(path, flags, *args)
            if os.fspath(path) == str(self.medium):
                opened_fds.append(fd)
            return fd

        def spy_close(fd):
            if fd in opened_fds:
                closed_fds.append(fd)
            return orig_close(fd)

        st = os.stat(self.medium)
        key = (int(st.st_dev), int(st.st_ino))

        # (B1) 2nd Thread.start() raises while worker 0 is still active -> cancel joins only worker 0, 0 RuntimeError, fd closed
        worker0_entered = threading.Event()
        release_worker0 = threading.Event()

        def gated_pread(fd, length, offset):
            worker0_entered.set()
            release_worker0.wait(timeout=5.0)
            return orig_pread(fd, length, offset)

        def fail_second_start(thread_self):
            if thread_self.name == "qf-os-prefetch-1":
                raise RuntimeError("injected thread start failure on worker 1")
            return orig_start(thread_self)

        with mock.patch.object(plugin, "_linux_mem_available_bytes", return_value=64 * 1024 ** 3), \
             mock.patch.object(os, "open", side_effect=spy_open), \
             mock.patch.object(os, "close", side_effect=spy_close), \
             mock.patch.object(os, "pread", side_effect=gated_pread), \
             mock.patch.object(threading.Thread, "start", fail_second_start):
            res = plugin._start_os_pagecache_prefetch(
                str(self.medium), min_bytes=64 * 1024, ram_reserve_bytes=1024, chunk_bytes=16 * 1024, num_threads=4)
            self.assertTrue(res)
            self.assertTrue(worker0_entered.wait(timeout=5.0))
            with plugin._OS_PREFETCH_LOCK:
                entry = plugin._OS_PREFETCH_ACTIVE.get(key)
                self.assertIsNotNone(entry)
                self.assertEqual(len(entry["threads"]), 1)
                self.assertTrue(entry["stop"].is_set())
                orig_stop_set = entry["stop"].set
                entry["stop"].set = lambda: (orig_stop_set(), release_worker0.set())[0]
            try:
                cancelled = plugin._cancel_os_pagecache_prefetch(str(self.medium), wait=True, timeout=5.0)
            finally:
                release_worker0.set()
            self.assertTrue(cancelled)
            entry["done"].wait(timeout=5.0)
            for t in entry["threads"]:
                t.join(timeout=5.0)
                self.assertFalse(t.is_alive())
        self.assertEqual(len(opened_fds), 1)
        self.assertEqual(set(opened_fds), set(closed_fds))
        with self.assertRaises(OSError):
            os.fstat(opened_fds[0])
        with plugin._OS_PREFETCH_LOCK:
            self.assertNotIn(key, plugin._OS_PREFETCH_ACTIVE)

        # (B2) 2nd Thread.start() raises after worker 0 already finished -> except closes fd immediately
        opened_fds.clear()
        closed_fds.clear()
        started_w0 = []

        def fail_second_after_w0_done(thread_self):
            if thread_self.name == "qf-os-prefetch-0":
                started_w0.append(thread_self)
                return orig_start(thread_self)
            if thread_self.name == "qf-os-prefetch-1":
                started_w0[0].join(timeout=5.0)
                raise RuntimeError("injected start failure after worker 0 finished")
            return orig_start(thread_self)

        with mock.patch.object(plugin, "_linux_mem_available_bytes", return_value=64 * 1024 ** 3), \
             mock.patch.object(os, "open", side_effect=spy_open), \
             mock.patch.object(os, "close", side_effect=spy_close), \
             mock.patch.object(threading.Thread, "start", fail_second_after_w0_done):
            res2 = plugin._start_os_pagecache_prefetch(
                str(self.medium), min_bytes=64 * 1024, ram_reserve_bytes=1024, chunk_bytes=16 * 1024, num_threads=4)
            self.assertTrue(res2)
        self.assertEqual(len(opened_fds), 1)
        self.assertEqual(set(opened_fds), set(closed_fds))
        self.assertFalse(plugin._cancel_os_pagecache_prefetch(str(self.medium), wait=True, timeout=5.0))
        with plugin._OS_PREFETCH_LOCK:
            self.assertNotIn(key, plugin._OS_PREFETCH_ACTIVE)

        # (B3) 1st Thread.start() raises (started == 0) -> returns False, pops entry, closes fd immediately
        opened_fds.clear()
        closed_fds.clear()

        def fail_first_start(thread_self):
            if thread_self.name == "qf-os-prefetch-0":
                raise RuntimeError("injected thread start failure on worker 0")
            return orig_start(thread_self)

        with mock.patch.object(plugin, "_linux_mem_available_bytes", return_value=64 * 1024 ** 3), \
             mock.patch.object(os, "open", side_effect=spy_open), \
             mock.patch.object(os, "close", side_effect=spy_close), \
             mock.patch.object(threading.Thread, "start", fail_first_start):
            res0 = plugin._start_os_pagecache_prefetch(
                str(self.medium), min_bytes=64 * 1024, ram_reserve_bytes=1024, chunk_bytes=16 * 1024, num_threads=4)
            self.assertFalse(res0)
        self.assertEqual(len(opened_fds), 1)
        self.assertEqual(set(opened_fds), set(closed_fds))
        self.assertFalse(plugin._cancel_os_pagecache_prefetch(str(self.medium), wait=True, timeout=5.0))
        with plugin._OS_PREFETCH_LOCK:
            self.assertNotIn(key, plugin._OS_PREFETCH_ACTIVE)

    def test_prefetch_pread_oserror_and_eof_safely_close_fd(self):
        if not sys.platform.startswith("linux"):
            return
        opened_fds = []
        closed_fds = []
        orig_open = os.open
        orig_close = os.close

        def spy_open(path, flags, *args):
            fd = orig_open(path, flags, *args)
            if os.fspath(path) == str(self.medium):
                opened_fds.append(fd)
            return fd

        def spy_close(fd):
            if fd in opened_fds:
                closed_fds.append(fd)
            return orig_close(fd)

        st = os.stat(self.medium)
        key = (int(st.st_dev), int(st.st_ino))

        for pread_mode in ("oserror", "eof"):
            opened_fds.clear()
            closed_fds.clear()
            thread_excs = []
            orig_excepthook = threading.excepthook
            threading.excepthook = lambda args: thread_excs.append(args.exc_value)
            try:
                if pread_mode == "oserror":
                    def fake_pread(_fd, _len, _off):
                        raise OSError(errno.EIO, "simulated pread error")
                else:
                    def fake_pread(_fd, _len, _off):
                        return b""

                created_threads = []
                orig_start = threading.Thread.start

                def track_start(t_self):
                    if t_self.name.startswith("qf-os-prefetch-"):
                        created_threads.append(t_self)
                    return orig_start(t_self)

                with mock.patch.object(plugin, "_linux_mem_available_bytes", return_value=64 * 1024 ** 3), \
                     mock.patch.object(os, "open", side_effect=spy_open), \
                     mock.patch.object(os, "close", side_effect=spy_close), \
                     mock.patch.object(os, "pread", side_effect=fake_pread), \
                     mock.patch.object(threading.Thread, "start", track_start):
                    ok = plugin._start_os_pagecache_prefetch(
                        str(self.medium), min_bytes=64 * 1024, ram_reserve_bytes=1024, chunk_bytes=16 * 1024, num_threads=4)
                    self.assertTrue(ok)
                    for t in created_threads:
                        t.join(timeout=5.0)
                        self.assertFalse(t.is_alive())
                self.assertEqual(thread_excs, [])
                self.assertEqual(len(opened_fds), 1)
                self.assertEqual(set(opened_fds), set(closed_fds))
                with self.assertRaises(OSError):
                    os.fstat(opened_fds[0])
                with plugin._OS_PREFETCH_LOCK:
                    self.assertNotIn(key, plugin._OS_PREFETCH_ACTIVE)
            finally:
                threading.excepthook = orig_excepthook

    def test_run_family_load_starts_prefetch_and_get_engine_cancels_on_hit_create_and_error(self):
        if not sys.platform.startswith("linux"):
            return
        opened_fds = []
        closed_fds = []
        orig_open = os.open
        orig_close = os.close
        orig_pread = os.pread

        def spy_open(path, flags, *args):
            fd = orig_open(path, flags, *args)
            if os.fspath(path) == str(self.medium):
                opened_fds.append(fd)
            return fd

        def spy_close(fd):
            if fd in opened_fds:
                closed_fds.append(fd)
            return orig_close(fd)

        st = os.stat(self.medium)
        key = (int(st.st_dev), int(st.st_ino))
        fake_lib = types.SimpleNamespace(
            qf_xfm_estimate_vram=lambda _p, _e: plugin.qfe.QF_OK,
            qf_last_error=lambda: b"",
        )

        with plugin._ENGINE_IDENTITY_LOCK:
            plugin._PIPELINE_CACHE.clear()
            plugin._PIPELINE_MODELS.clear()
            plugin._PIPELINE_SIGS.clear()
            plugin._CREATE_LOCKS.clear()
        try:
            release = threading.Event()
            entered = threading.Event()

            def gated_pread(fd, length, offset):
                entered.set()
                release.wait(timeout=5.0)
                return orig_pread(fd, length, offset)

            def _arm_stop_releases_pread():
                with plugin._OS_PREFETCH_LOCK:
                    self.assertIn(key, plugin._OS_PREFETCH_ACTIVE)
                    entry = plugin._OS_PREFETCH_ACTIVE[key]
                    threads = list(entry["threads"])
                    orig_stop_set = entry["stop"].set
                    entry["stop"].set = lambda: (orig_stop_set(), release.set())[0]
                return threads

            cancel_log = []
            orig_cancel = plugin._cancel_os_pagecache_prefetch

            def spy_cancel(path, *, wait=True, timeout=5.0):
                ret = orig_cancel(path, wait=wait, timeout=timeout)
                cancel_log.append((path, ret))
                return ret

            def fake_builder(*, transformer1_path, bundle_dir, pinned_memory):
                return types.SimpleNamespace(model=None, path=transformer1_path, bundle=bundle_dir, pinned=pinned_memory)

            with contextlib.ExitStack() as stack:
                for ctx_mgr in (
                    mock.patch.dict(plugin._FAMILY_BUILDERS, {"krea2": fake_builder}, clear=False),
                    mock.patch.object(plugin, "_resolve_transformer", return_value=str(self.medium)),
                    mock.patch.object(plugin, "_cancel_os_pagecache_prefetch", side_effect=spy_cancel),
                    mock.patch.object(plugin, "_OS_PREFETCH_MIN_BYTES", 64 * 1024),
                    mock.patch.object(plugin, "_OS_PREFETCH_RAM_RESERVE_BYTES", 1024),
                    mock.patch.object(plugin, "_OS_PREFETCH_CHUNK_BYTES", 16 * 1024),
                    mock.patch.object(plugin, "_linux_mem_available_bytes", return_value=64 * 1024 ** 3),
                    mock.patch.object(os, "open", side_effect=spy_open),
                    mock.patch.object(os, "close", side_effect=spy_close),
                    mock.patch.object(os, "pread", side_effect=gated_pread),
                    mock.patch.object(plugin.qfe, "_LIB", fake_lib),
                    mock.patch.object(plugin.qfe, "load_lib", return_value=fake_lib),
                    mock.patch.object(plugin.qfe, "loaded_so_path", return_value="/fake/libquantfunc.so"),
                    mock.patch.object(plugin.qfe, "library_identity", return_value="img-1"),
                    mock.patch.object(plugin.qfe, "_refuse_second_arch", lambda *_a, **_kw: None),
                    mock.patch.object(plugin, "_read_auth", return_value=("", "https://srv.example")),
                    mock.patch.object(plugin, "_estimate_vram_cached", return_value=plugin.qfe.ColdNeed(1024, None)),
                ):
                    stack.enter_context(ctx_mgr)

                # 1) _run_family_load starts OS prefetch -> _get_engine cancels on create success
                res = plugin._run_family_load(
                    "krea2", "krea2-quantfunc-int4.safetensors", model_config="krea2-turbo-int4", pinned_memory=False)
                self.assertEqual(res.path, str(self.medium))
                self.assertTrue(entered.wait(timeout=5.0))
                threads_1 = _arm_stop_releases_pread()

                def create_ok(lib, *, create_params, capacity_bytes=0):
                    h = plugin.qfe.QFEngineHandle(lib, ctypes.c_void_p(0x8001), capacity_bytes=capacity_bytes)
                    h.resident_vram_bytes = lambda: capacity_bytes
                    return h

                try:
                    with mock.patch.object(plugin.qfe.QFEngineHandle, "create", side_effect=create_ok):
                        eng, _ckey = plugin._get_engine(str(self.medium), {"model_arch": "krea2"}, device_id=0)
                finally:
                    release.set()
                self.assertIsNotNone(eng.xfm)
                self.assertEqual(cancel_log[-1], (str(self.medium), True))
                self.assertTrue(all(not t.is_alive() for t in threads_1))
                with plugin._OS_PREFETCH_LOCK:
                    self.assertNotIn(key, plugin._OS_PREFETCH_ACTIVE)

                # 2) _run_family_load restarts prefetch -> _get_engine cancels on initial cache hit
                entered.clear()
                release.clear()
                plugin._run_family_load(
                    "krea2", "krea2-quantfunc-int4.safetensors", model_config="krea2-turbo-int4", pinned_memory=False)
                self.assertTrue(entered.wait(timeout=5.0))
                threads_2 = _arm_stop_releases_pread()
                try:
                    eng2, _ = plugin._get_engine(str(self.medium), {"model_arch": "krea2"}, device_id=0)
                finally:
                    release.set()
                self.assertIs(eng2, eng)
                self.assertEqual(cancel_log[-1], (str(self.medium), True))
                self.assertTrue(all(not t.is_alive() for t in threads_2))
                with plugin._OS_PREFETCH_LOCK:
                    self.assertNotIn(key, plugin._OS_PREFETCH_ACTIVE)

                # 3) Double-checked locked cache hit path inside _create_lock_for(ckey) cancels prefetch
                with plugin._ENGINE_IDENTITY_LOCK:
                    plugin._PIPELINE_CACHE.clear()
                t1_in_create = threading.Event()
                t2_waiting_lock = threading.Event()
                t1_pin_hooked = [False]
                threads_3 = []

                def create_wait(lib, *, create_params, capacity_bytes=0):
                    t1_in_create.set()
                    t2_waiting_lock.wait(timeout=5.0)
                    h = plugin.qfe.QFEngineHandle(lib, ctypes.c_void_p(0x8002), capacity_bytes=capacity_bytes)
                    h.resident_vram_bytes = lambda: capacity_bytes
                    return h

                orig_create_lock_for = plugin._create_lock_for
                lock_calls = [0]

                def spy_create_lock_for(ckey_arg):
                    lk = orig_create_lock_for(ckey_arg)
                    lock_calls[0] += 1
                    if lock_calls[0] == 2:
                        t2_waiting_lock.set()
                    return lk

                orig_pin = plugin._pin_pipeline_consumer_locked

                def spy_pin(ckey_arg):
                    orig_pin(ckey_arg)
                    if not t1_pin_hooked[0]:
                        t1_pin_hooked[0] = True
                        release.clear()
                        entered.clear()
                        plugin._start_os_pagecache_prefetch(str(self.medium))
                        threads_3.extend(_arm_stop_releases_pread())

                t2_results = []

                def run_t2():
                    t2_results.append(plugin._get_engine(str(self.medium), {"model_arch": "krea2"}, device_id=0))

                with mock.patch.object(plugin.qfe.QFEngineHandle, "create", side_effect=create_wait), \
                     mock.patch.object(plugin, "_create_lock_for", side_effect=spy_create_lock_for), \
                     mock.patch.object(plugin, "_pin_pipeline_consumer_locked", side_effect=spy_pin):
                    t1 = threading.Thread(target=lambda: plugin._get_engine(str(self.medium), {"model_arch": "krea2"}, device_id=0))
                    t1.start()
                    self.assertTrue(t1_in_create.wait(timeout=5.0))
                    t2 = threading.Thread(target=run_t2)
                    t2.start()
                    try:
                        t1.join(timeout=5.0)
                        t2.join(timeout=5.0)
                    finally:
                        release.set()
                self.assertEqual(len(t2_results), 1)
                self.assertEqual(lock_calls[0], 2)
                self.assertEqual(cancel_log[-1], (str(self.medium), True))
                self.assertTrue(threads_3)
                self.assertTrue(all(not t.is_alive() for t in threads_3))
                with plugin._OS_PREFETCH_LOCK:
                    self.assertNotIn(key, plugin._OS_PREFETCH_ACTIVE)

                # 4) _get_engine create exception -> finally: still cancels active prefetch
                with plugin._ENGINE_IDENTITY_LOCK:
                    plugin._PIPELINE_CACHE.clear()
                entered.clear()
                release.clear()
                plugin._run_family_load(
                    "krea2", "krea2-quantfunc-int4.safetensors", model_config="krea2-turbo-int4", pinned_memory=False)
                self.assertTrue(entered.wait(timeout=5.0))
                threads_4 = _arm_stop_releases_pread()
                try:
                    with mock.patch.object(plugin.qfe.QFEngineHandle, "create", side_effect=RuntimeError("create exploded")):
                        with self.assertRaisesRegex(RuntimeError, "create exploded"):
                            plugin._get_engine(str(self.medium), {"model_arch": "krea2"}, device_id=0)
                finally:
                    release.set()
                self.assertEqual(cancel_log[-1], (str(self.medium), True))
                self.assertTrue(all(not t.is_alive() for t in threads_4))
                with plugin._OS_PREFETCH_LOCK:
                    self.assertNotIn(key, plugin._OS_PREFETCH_ACTIVE)

                # 5) _run_family_load swallows errors raised by _start_os_pagecache_prefetch
                with mock.patch.object(plugin, "_start_os_pagecache_prefetch", side_effect=OSError("pf broken")):
                    res_ok = plugin._run_family_load(
                        "krea2", "krea2-quantfunc-int4.safetensors", model_config="krea2-turbo-int4", pinned_memory=False)
                self.assertEqual(res_ok.path, str(self.medium))

            self.assertEqual(len(opened_fds), 4)
            self.assertEqual(set(opened_fds), set(closed_fds))
        finally:
            with plugin._ENGINE_IDENTITY_LOCK:
                plugin._PIPELINE_CACHE.clear()
                plugin._PIPELINE_MODELS.clear()
                plugin._PIPELINE_SIGS.clear()
                plugin._CREATE_LOCKS.clear()


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
