#!/usr/bin/env python3
"""Lifecycle contracts for ComfyUI Plugin 0.0.09 pre-finalization atexit & GC handle cleanup (C01-C11)."""
import atexit
import contextlib
import ctypes
import gc
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
import weakref

_PLUGIN_DIR = Path(__file__).resolve().parents[1]
_COMFY_ROOT = os.environ.get("COMFY_ROOT")
_PKG_NAME = "qf_native_atexit_test_pkg"


def _load_plugin():
    if _PKG_NAME in sys.modules:
        return sys.modules[_PKG_NAME]
    saved_argv = list(sys.argv)
    try:
        if _COMFY_ROOT and (Path(_COMFY_ROOT) / "comfy/model_management.py").is_file():
            if _COMFY_ROOT not in sys.path:
                sys.path.insert(0, _COMFY_ROOT)
            sys.argv = [sys.argv[0], "--cpu"]
            import comfy.options  # noqa: E402
            comfy.options.enable_args_parsing()
        spec = importlib.util.spec_from_file_location(
            _PKG_NAME,
            _PLUGIN_DIR / "__init__.py",
            submodule_search_locations=[str(_PLUGIN_DIR)],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_PKG_NAME] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = saved_argv


plugin = _load_plugin()
qfe = plugin.qfe


class _RecordingLib:
    """Minimal PC Engine C-API double tracking destroy / session / unload calls."""

    def __init__(self, *, unload_rc=0, end_rc=0, raise_on_destroy=False,
                 raise_on_unload=False, on_destroy=None):
        self.unload_rc = unload_rc
        self.end_rc = end_rc
        self.raise_on_destroy = raise_on_destroy
        self.raise_on_unload = raise_on_unload
        self.on_destroy = on_destroy
        self.destroy_calls = []
        self.session_end_calls = []
        self.unload_calls = []
        self._lock = threading.Lock()

    def qf_xfm_destroy(self, xfm):
        with self._lock:
            self.destroy_calls.append(xfm)
        if self.on_destroy is not None:
            self.on_destroy(xfm)
        if self.raise_on_destroy:
            raise RuntimeError("simulated qf_xfm_destroy failure")

    def qf_xfm_session_end(self, sess):
        with self._lock:
            self.session_end_calls.append(sess)
        return self.end_rc

    def qf_xfm_unload_vram(self, xfm, req, out_ptr):
        with self._lock:
            self.unload_calls.append((xfm, int(req)))
        if self.raise_on_unload:
            raise RuntimeError("simulated qf_xfm_unload_vram failure")
        out_ptr._obj.value = 0
        return self.unload_rc

    def qf_last_error(self):
        return b"stub error"


class _MockCacheHandle:
    """Mock cache handle matching host_ram_release_test.Handle (0-arg destroy, no _destroy_for_shutdown)."""

    def __init__(self, *, raise_on_destroy=False):
        self.xfm = object()
        self.capacity_bytes = 1024
        self.destroy_count = 0
        self.raise_on_destroy = raise_on_destroy

    def destroy(self):
        self.destroy_count += 1
        self.xfm = None
        if self.raise_on_destroy:
            raise RuntimeError("simulated mock Handle.destroy failure")


class _Consumer:
    pass


class _AudioOwner:
    pass


class AtexitCleanupTests(unittest.TestCase):
    def setUp(self):
        self.saved_cache = dict(plugin._PIPELINE_CACHE)
        self.saved_models = {k: list(v) for k, v in plugin._PIPELINE_MODELS.items()}
        self.saved_sigs = {k: set(v) for k, v in plugin._PIPELINE_SIGS.items()}
        with qfe._PENDING_OWNERS_LOCK:
            self.saved_pending = set(qfe._REFERENCE_AUDIO_PENDING_OWNERS)
            qfe._REFERENCE_AUDIO_PENDING_OWNERS.clear()
        with plugin._ENGINE_IDENTITY_LOCK:
            plugin._PIPELINE_CACHE.clear()
            plugin._PIPELINE_MODELS.clear()
            plugin._PIPELINE_SIGS.clear()
        self._handles_to_disarm = []

    def tearDown(self):
        for h in self._handles_to_disarm:
            try:
                h._reference_audio_pending = False
                h._reference_audio_owner = None
                h.xfm = None
            except Exception:
                pass
        with qfe._PENDING_OWNERS_LOCK:
            qfe._REFERENCE_AUDIO_PENDING_OWNERS.clear()
            qfe._REFERENCE_AUDIO_PENDING_OWNERS.update(self.saved_pending)
        with plugin._ENGINE_IDENTITY_LOCK:
            plugin._PIPELINE_CACHE.clear()
            plugin._PIPELINE_CACHE.update(self.saved_cache)
            plugin._PIPELINE_MODELS.clear()
            plugin._PIPELINE_MODELS.update(self.saved_models)
            plugin._PIPELINE_SIGS.clear()
            plugin._PIPELINE_SIGS.update(self.saved_sigs)

    def _make_handle(self, lib, xfm_val=0x1001, capacity=4096):
        h = qfe.QFEngineHandle(lib, ctypes.c_void_p(xfm_val), capacity_bytes=capacity)
        self._handles_to_disarm.append(h)
        return h

    # ── C01 ────────────────────────────────────────────────────────────────────
    def test_c01_shutdown_drains_cached_handles_even_with_live_foreign_consumers(self):
        lib1, lib2 = _RecordingLib(), _RecordingLib()
        h1 = self._make_handle(lib1, 0xA01)
        h2 = self._make_handle(lib2, 0xA02)
        h3 = _MockCacheHandle()
        c1, c2, c3 = _Consumer(), _Consumer(), _Consumer()
        c1._qf, c2._qf, c3._qf = h1, h2, h3
        plugin._PIPELINE_CACHE[("k1", "{}")] = h1
        plugin._PIPELINE_CACHE[("k2", "{}")] = h2
        plugin._PIPELINE_CACHE[("k3", "{}")] = h3
        plugin._PIPELINE_MODELS[("k1", "{}")] = [weakref.ref(c1)]
        plugin._PIPELINE_MODELS[("k2", "{}")] = [weakref.ref(c2)]
        plugin._PIPELINE_MODELS[("k3", "{}")] = [weakref.ref(c3)]
        plugin._PIPELINE_SIGS[("k1", "{}")] = {("sig", 1)}

        plugin._shutdown_all_pipelines()

        self.assertEqual(len(lib1.destroy_calls), 1)
        self.assertEqual(len(lib2.destroy_calls), 1)
        self.assertEqual(h3.destroy_count, 1)
        self.assertIsNone(h1.xfm)
        self.assertIsNone(h2.xfm)
        self.assertIsNone(h3.xfm)
        self.assertEqual(plugin._PIPELINE_CACHE, {})
        self.assertEqual(plugin._PIPELINE_MODELS, {})
        self.assertEqual(plugin._PIPELINE_SIGS, {})

    # ── C02 ────────────────────────────────────────────────────────────────────
    def test_c02_runtime_retire_handle_still_refuses_live_foreign_consumer(self):
        lib = _RecordingLib()
        h = self._make_handle(lib, 0xB01)
        foreign = _Consumer()
        foreign._qf = h
        ckey = ("k_live", "{}")
        plugin._PIPELINE_CACHE[ckey] = h
        plugin._PIPELINE_MODELS[ckey] = [weakref.ref(foreign)]

        retired = plugin._retire_handle(ckey, h, requester=None, reason="runtime test")

        self.assertFalse(retired)
        self.assertIs(plugin._PIPELINE_CACHE[ckey], h)
        self.assertIsNotNone(h.xfm)
        self.assertEqual(lib.destroy_calls, [])

    # ── C03 ────────────────────────────────────────────────────────────────────
    def test_c03_per_handle_exception_isolation_for_cached_and_orphan_handles(self):
        lib_bad = _RecordingLib(raise_on_destroy=True)
        lib_ok = _RecordingLib()
        h_bad = self._make_handle(lib_bad, 0xC01)
        h_ok = self._make_handle(lib_ok, 0xC02)
        m_bad = _MockCacheHandle(raise_on_destroy=True)
        m_ok = _MockCacheHandle()
        plugin._PIPELINE_CACHE[("c_bad", "{}")] = h_bad
        plugin._PIPELINE_CACHE[("c_m_bad", "{}")] = m_bad
        plugin._PIPELINE_CACHE[("c_ok", "{}")] = h_ok
        plugin._PIPELINE_CACHE[("c_m_ok", "{}")] = m_ok

        # Multiple orphaned pending-audio handles (not in _PIPELINE_CACHE) where first raises
        lib_orp1 = _RecordingLib(unload_rc=1, end_rc=1, raise_on_destroy=True)
        lib_orp2 = _RecordingLib(unload_rc=1, end_rc=1)
        orp1 = self._make_handle(lib_orp1, 0xD01)
        orp2 = self._make_handle(lib_orp2, 0xD02)
        orp1.retain_reference_audio(_AudioOwner())
        orp2.retain_reference_audio(_AudioOwner())

        plugin._shutdown_all_pipelines()

        self.assertEqual(len(lib_bad.destroy_calls), 1)
        self.assertEqual(len(lib_ok.destroy_calls), 1)
        self.assertEqual(m_bad.destroy_count, 1)
        self.assertEqual(m_ok.destroy_count, 1)
        self.assertEqual(len(lib_orp1.destroy_calls), 1)
        self.assertEqual(len(lib_orp2.destroy_calls), 1)
        self.assertIsNone(h_bad.xfm)
        self.assertIsNone(h_ok.xfm)
        self.assertIsNone(orp1.xfm)
        self.assertIsNone(orp2.xfm)
        self.assertEqual(plugin._PIPELINE_CACHE, {})
        self.assertEqual(plugin._PIPELINE_MODELS, {})
        self.assertEqual(plugin._PIPELINE_SIGS, {})
        self.assertEqual(qfe._snapshot_pending_reference_audio_owners(), [])

        # Direct _retire_handle exceptions exercise BOTH per-handle try/except blocks inside _shutdown_all_pipelines
        lib_sc1, lib_sc2 = _RecordingLib(), _RecordingLib()
        spy_c1 = self._make_handle(lib_sc1, 0xC11)
        spy_c2 = self._make_handle(lib_sc2, 0xC12)
        plugin._PIPELINE_CACHE[("spy_c1", "{}")] = spy_c1
        plugin._PIPELINE_CACHE[("spy_c2", "{}")] = spy_c2

        lib_so1, lib_so2 = _RecordingLib(unload_rc=1, end_rc=1), _RecordingLib(unload_rc=1, end_rc=1)
        spy_o1 = self._make_handle(lib_so1, 0xD11)
        spy_o2 = self._make_handle(lib_so2, 0xD12)
        spy_o1.retain_reference_audio(_AudioOwner())
        spy_o2.retain_reference_audio(_AudioOwner())

        real_retire = plugin._retire_handle

        def _retire_raising_first(ckey, eng, requester=None, **kwargs):
            if ckey == ("spy_c1", "{}"):
                raise RuntimeError("injected _retire_handle cache failure")
            if ckey is None and eng is spy_o1:
                raise RuntimeError("injected _retire_handle orphan failure")
            return real_retire(ckey, eng, requester, **kwargs)

        with mock.patch.object(plugin, "_retire_handle", side_effect=_retire_raising_first):
            plugin._shutdown_all_pipelines()

        try:
            self.assertEqual(len(lib_sc1.destroy_calls), 0)
            self.assertEqual(len(lib_sc2.destroy_calls), 1)
            self.assertIsNone(spy_c2.xfm)
            self.assertEqual(len(lib_so1.destroy_calls), 0)
            self.assertEqual(len(lib_so2.destroy_calls), 1)
            self.assertIsNone(spy_o2.xfm)
            self.assertEqual(plugin._PIPELINE_CACHE, {})
            self.assertEqual(plugin._PIPELINE_MODELS, {})
            self.assertEqual(plugin._PIPELINE_SIGS, {})
        finally:
            spy_o1.reference_audio_copy_completed()

    # ── C04 ────────────────────────────────────────────────────────────────────
    def test_c04_idempotency_8thread_barrier_destroy_and_raising_destroy(self):
        lib1 = _RecordingLib()
        h1 = self._make_handle(lib1, 0xE01)
        plugin._PIPELINE_CACHE[("idem", "{}")] = h1
        plugin._shutdown_all_pipelines()
        plugin._shutdown_all_pipelines()
        self.assertEqual(len(lib1.destroy_calls), 1)

        # 8-thread concurrent destroy()
        lib_race = _RecordingLib()
        h_race = self._make_handle(lib_race, 0xE02)
        barrier = threading.Barrier(8)
        errs = []

        def _worker():
            try:
                barrier.wait(timeout=5.0)
                h_race.destroy()
            except Exception as exc:  # noqa: BLE001
                errs.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        self.assertEqual(errs, [])
        self.assertEqual(len(lib_race.destroy_calls), 1)
        self.assertIsNone(h_race.xfm)

        # Raising qf_xfm_destroy still clears xfm and subsequent destroy() is a no-op
        lib_err = _RecordingLib(raise_on_destroy=True)
        h_err = self._make_handle(lib_err, 0xE03)
        h_err.destroy()
        self.assertIsNone(h_err.xfm)
        self.assertEqual(len(lib_err.destroy_calls), 1)
        h_err.destroy()
        self.assertEqual(len(lib_err.destroy_calls), 1)

    # ── C05 ────────────────────────────────────────────────────────────────────
    def test_c05_pending_reference_audio_shutdown_all_paths_and_concurrency(self):
        # (a) unload_sync succeeds
        lib_a = _RecordingLib(unload_rc=0)
        h_a = self._make_handle(lib_a, 0xF01)
        h_a.retain_reference_audio(_AudioOwner())
        plugin._PIPELINE_CACHE[("pa_a", "{}")] = h_a
        plugin._shutdown_all_pipelines()
        self.assertEqual(len(lib_a.unload_calls), 1)
        self.assertEqual(len(lib_a.destroy_calls), 1)
        self.assertFalse(h_a.reference_audio_pending())
        self.assertEqual(qfe._snapshot_pending_reference_audio_owners(), [])

        # (b) unload_sync fails, end_session_if_open succeeds
        lib_b = _RecordingLib(unload_rc=1, end_rc=0, raise_on_unload=True)
        h_b = self._make_handle(lib_b, 0xF02)
        h_b.current_session = ctypes.c_void_p(0x9002)
        h_b.retain_reference_audio(_AudioOwner())
        plugin._PIPELINE_CACHE[("pa_b", "{}")] = h_b
        plugin._shutdown_all_pipelines()
        self.assertEqual(len(lib_b.session_end_calls), 1)
        self.assertEqual(len(lib_b.destroy_calls), 1)
        self.assertIsNone(h_b.current_session)
        self.assertFalse(h_b.reference_audio_pending())
        self.assertEqual(qfe._snapshot_pending_reference_audio_owners(), [])

        # (c) both unload_sync and end_session_if_open fail -> owner_keep stays alive across qf_xfm_destroy
        alive_inside_destroy = []
        owner_c = _AudioOwner()
        owner_ref = weakref.ref(owner_c)

        def _on_destroy(_xfm):
            gc.collect()
            alive_inside_destroy.append(owner_ref() is not None)

        lib_c = _RecordingLib(unload_rc=1, end_rc=1, raise_on_unload=True, on_destroy=_on_destroy)
        h_c = self._make_handle(lib_c, 0xF03)
        h_c.current_session = ctypes.c_void_p(0x9003)
        h_c.retain_reference_audio(owner_c)
        del owner_c
        gc.collect()
        plugin._PIPELINE_CACHE[("pa_c", "{}")] = h_c
        plugin._shutdown_all_pipelines()
        gc.collect()

        self.assertEqual(alive_inside_destroy, [True])
        self.assertIsNone(owner_ref(), "owner_keep must be released after _shutdown_all_pipelines returns")
        self.assertEqual(len(lib_c.destroy_calls), 1)
        self.assertFalse(h_c.reference_audio_pending())
        self.assertIsNone(h_c.xfm)
        self.assertEqual(qfe._snapshot_pending_reference_audio_owners(), [])

        # (d) orphaned handle already popped from _PIPELINE_CACHE but still in _REFERENCE_AUDIO_PENDING_OWNERS
        lib_d = _RecordingLib(unload_rc=1, end_rc=1)
        h_d = self._make_handle(lib_d, 0xF04)
        h_d.retain_reference_audio(_AudioOwner())
        self.assertNotIn(h_d, plugin._PIPELINE_CACHE.values())
        self.assertIn(h_d, qfe._snapshot_pending_reference_audio_owners())
        plugin._shutdown_all_pipelines()
        self.assertEqual(len(lib_d.destroy_calls), 1)
        self.assertIsNone(h_d.xfm)
        self.assertEqual(qfe._snapshot_pending_reference_audio_owners(), [])

        # (e) concurrent retain_reference_audio / reference_audio_copy_completed vs snapshot iteration
        stress_handles = [self._make_handle(_RecordingLib(), 0xF10 + i) for i in range(6)]
        stop_ev = threading.Event()
        stress_errs = []

        def _mutator(idx):
            h_m = stress_handles[idx]
            try:
                while not stop_ev.is_set():
                    h_m.retain_reference_audio(_AudioOwner())
                    h_m.reference_audio_copy_completed()
            except Exception as exc:  # noqa: BLE001
                stress_errs.append(exc)

        def _reader():
            try:
                for _ in range(400):
                    _ = qfe._snapshot_pending_reference_audio_owners()
            except Exception as exc:  # noqa: BLE001
                stress_errs.append(exc)

        mutators = [threading.Thread(target=_mutator, args=(i,)) for i in range(len(stress_handles))]
        readers = [threading.Thread(target=_reader) for _ in range(3)]
        for t in mutators + readers:
            t.start()
        for t in readers:
            t.join(timeout=10.0)
        stop_ev.set()
        for t in mutators:
            t.join(timeout=5.0)
        self.assertEqual(stress_errs, [])
        for h_m in stress_handles:
            h_m.reference_audio_copy_completed()

    # ── C06 ────────────────────────────────────────────────────────────────────
    def test_c06_runtime_destroy_with_pending_reference_audio_still_refuses(self):
        lib = _RecordingLib()
        h = self._make_handle(lib, 0x1601)
        h.retain_reference_audio(_AudioOwner())
        with self.assertRaisesRegex(RuntimeError, "reference copy pending"):
            h.destroy()
        self.assertIsNotNone(h.xfm)
        self.assertEqual(lib.destroy_calls, [])

    # ── C07 ────────────────────────────────────────────────────────────────────
    def test_c07_del_guards_and_object_new_safety(self):
        # (a) live uncached handle GC'd while is_finalizing() == False -> destroys native once
        lib_a = _RecordingLib()
        h_a = qfe.QFEngineHandle(lib_a, ctypes.c_void_p(0x7001))
        del h_a
        gc.collect()
        self.assertEqual(len(lib_a.destroy_calls), 1)

        # (b) is_finalizing() == True or sys is None -> 0 calls
        lib_b1 = _RecordingLib()
        h_b1 = self._make_handle(lib_b1, 0x7002)
        with mock.patch.object(sys, "is_finalizing", return_value=True):
            h_b1.__del__()
        self.assertEqual(len(lib_b1.destroy_calls), 0)
        self.assertIsNotNone(h_b1.xfm)

        lib_b2 = _RecordingLib()
        h_b2 = self._make_handle(lib_b2, 0x7003)
        with mock.patch.object(qfe, "sys", None):
            h_b2.__del__()
        self.assertEqual(len(lib_b2.destroy_calls), 0)
        self.assertIsNotNone(h_b2.xfm)

        lib_b3 = _RecordingLib()
        h_b3 = self._make_handle(lib_b3, 0x7004)
        h_b3._sys = None
        h_b3.__del__()
        self.assertEqual(len(lib_b3.destroy_calls), 0)
        self.assertIsNotNone(h_b3.xfm)

        # (c) xfm is None -> 0 calls
        lib_c = _RecordingLib()
        h_c = self._make_handle(lib_c, 0x7005)
        h_c.xfm = None
        h_c.__del__()
        self.assertEqual(len(lib_c.destroy_calls), 0)

        # (d) _reference_audio_pending == True -> 0 calls
        lib_d = _RecordingLib()
        h_d = self._make_handle(lib_d, 0x7006)
        h_d.retain_reference_audio(_AudioOwner())
        h_d.__del__()
        self.assertEqual(len(lib_d.destroy_calls), 0)
        self.assertIsNotNone(h_d.xfm)

        # (e) object.__new__(QFEngineHandle) with ONLY lib & xfm across all teardown/query methods
        stderr_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf):
            for method_name in ("destroy", "_destroy_for_shutdown", "__del__"):
                lib_e = _RecordingLib()
                h_e = object.__new__(qfe.QFEngineHandle)
                h_e.lib = lib_e
                h_e.xfm = ctypes.c_void_p(0x7010)
                self.assertFalse(h_e.reference_audio_pending())
                self.assertEqual(h_e.end_session_if_open(), (False, True))
                getattr(h_e, method_name)()
                self.assertIsNone(h_e.xfm)
                self.assertEqual(len(lib_e.destroy_calls), 1)

            # object.__new__ + normal QFEngineHandle with stub lib lacking qf_xfm_destroy / qf_xfm_session_end
            class _NoDestroyLib:
                pass

            h_stub = object.__new__(qfe.QFEngineHandle)
            h_stub.lib = _NoDestroyLib()
            h_stub.xfm = ctypes.c_void_p(0x7011)
            h_stub.current_session = ctypes.c_void_p(0x7012)
            self.assertEqual(h_stub.end_session_if_open(), (True, False))
            h_stub.__del__()
            self.assertIsNone(h_stub.xfm)

            for ctor in (
                lambda: qfe.QFEngineHandle(_NoDestroyLib(), ctypes.c_void_p(0x7014)),
                lambda: (lambda x: (setattr(x, "lib", _NoDestroyLib()), setattr(x, "xfm", ctypes.c_void_p(0x7015)), x)[2])(
                    object.__new__(qfe.QFEngineHandle)
                ),
                lambda: qfe.QFEngineHandle(None, ctypes.c_void_p(0x7016)),
            ):
                h_nodestroy = ctor()
                h_nodestroy.destroy()
                self.assertIsNone(h_nodestroy.xfm)

            lib_raise = _RecordingLib(raise_on_destroy=True)
            h_raise = object.__new__(qfe.QFEngineHandle)
            h_raise.lib = lib_raise
            h_raise.xfm = ctypes.c_void_p(0x7013)
            h_raise.__del__()
            self.assertIsNone(h_raise.xfm)
            self.assertEqual(len(lib_raise.destroy_calls), 1)
        self.assertEqual(stderr_buf.getvalue(), "")

        # (f) _destroy_lock held by another thread -> non-blocking early return without deadlock;
        #     plus same-thread reentrant _destroying guard inside qf_xfm_destroy callback
        lib_f = _RecordingLib()
        h_f = self._make_handle(lib_f, 0x7020)
        held_ev = threading.Event()
        release_ev = threading.Event()

        def _hold_destroy_lock():
            with h_f._destroy_lock:
                held_ev.set()
                release_ev.wait(timeout=5.0)

        holder = threading.Thread(target=_hold_destroy_lock)
        holder.start()
        try:
            self.assertTrue(held_ev.wait(timeout=5.0))
            h_f.__del__()
            self.assertEqual(len(lib_f.destroy_calls), 0)
            self.assertIsNotNone(h_f.xfm)
        finally:
            release_ev.set()
            holder.join(timeout=5.0)

        # Same-thread reentrancy during destroy(): RLock + _destroying guard skip recursive destroy/__del__
        reentrant_holder = []

        def _reenter_cb(_xfm):
            h_re = reentrant_holder[0]
            h_re.__del__()
            h_re.destroy()

        lib_re = _RecordingLib(on_destroy=_reenter_cb)
        h_re = self._make_handle(lib_re, 0x7021)
        reentrant_holder.append(h_re)
        h_re.destroy()
        self.assertEqual(len(lib_re.destroy_calls), 1)
        self.assertIsNone(h_re.xfm)

    # ── C08 ────────────────────────────────────────────────────────────────────
    def test_c08_subprocess_atexit_destroys_surviving_handles_before_finalizing(self):
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "atexit_result.json")
            script = os.path.join(td, "run_subproc.py")
            with open(script, "w", encoding="utf-8") as fh:
                fh.write(
                    "import ctypes, importlib.util, json, os, sys\n"
                    "comfy_root = os.environ.get('COMFY_ROOT')\n"
                    "if comfy_root and os.path.isfile(os.path.join(comfy_root, 'comfy', 'model_management.py')):\n"
                    "    sys.path.insert(0, comfy_root)\n"
                    "    sys.argv = [sys.argv[0], '--cpu']\n"
                    "    import comfy.options\n"
                    "    comfy.options.enable_args_parsing()\n"
                    "plugin_dir = os.environ['QF_PLUGIN_DIR']\n"
                    "spec = importlib.util.spec_from_file_location(\n"
                    "    'qf_sub_pkg', os.path.join(plugin_dir, '__init__.py'),\n"
                    "    submodule_search_locations=[plugin_dir])\n"
                    "mod = importlib.util.module_from_spec(spec)\n"
                    "sys.modules['qf_sub_pkg'] = mod\n"
                    "spec.loader.exec_module(mod)\n"
                    "events = []\n"
                    "with open(os.environ['QF_OUT'], 'w', encoding='utf-8') as out:\n"
                    "    json.dump(events, out)\n"
                    "class _Lib:\n"
                    "    def qf_xfm_destroy(self, xfm):\n"
                    "        events.append({'xfm': int(xfm.value), 'finalizing': bool(sys.is_finalizing())})\n"
                    "        with open(os.environ['QF_OUT'], 'w', encoding='utf-8') as out:\n"
                    "            json.dump(events, out)\n"
                    "lib = _Lib()\n"
                    "mod._PIPELINE_CACHE[('sub', 1)] = mod.qfe.QFEngineHandle(lib, ctypes.c_void_p(101))\n"
                    "mod._PIPELINE_CACHE[('sub', 2)] = mod.qfe.QFEngineHandle(lib, ctypes.c_void_p(102))\n"
                )
            env = dict(os.environ, QF_PLUGIN_DIR=str(_PLUGIN_DIR), QF_OUT=out_path, PYTHONIOENCODING="utf-8")
            cmd = [sys.executable, "-W", "ignore::FutureWarning", script]
            proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", env=env, timeout=30)
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr!r}")
            if _COMFY_ROOT and (Path(_COMFY_ROOT) / "comfy/model_management.py").is_file():
                self.assertEqual(proc.stderr, "")
            else:
                self.assertNotIn("Traceback", proc.stderr)
                self.assertNotIn("Exception ignored", proc.stderr)
            with open(out_path, encoding="utf-8") as fh:
                events = json.load(fh)
            self.assertEqual(len(events), 2)
            self.assertEqual([e["finalizing"] for e in events], [False, False])
            self.assertEqual(sorted(e["xfm"] for e in events), [101, 102])

    # ── C10 ────────────────────────────────────────────────────────────────────
    def test_c10_negative_degraded_and_concurrent_replacement_boundaries(self):
        # (a) _retire_handle(None, eng, force_shutdown=False) -> False, and eng is None -> False
        lib_a = _RecordingLib()
        h_a = self._make_handle(lib_a, 0x10A1)
        self.assertFalse(plugin._retire_handle(None, h_a, force_shutdown=False))
        self.assertFalse(plugin._retire_handle(("none_eng", "{}"), None, force_shutdown=False))
        self.assertFalse(plugin._retire_handle(None, None, force_shutdown=True))
        self.assertIsNotNone(h_a.xfm)
        self.assertEqual(lib_a.destroy_calls, [])

        # (b) _retire_handle(ckey, old_eng, force_shutdown=True) when replaced by new_eng -> False, new_eng untouched
        lib_old, lib_new = _RecordingLib(), _RecordingLib()
        old_eng = self._make_handle(lib_old, 0x10B1)
        new_eng = self._make_handle(lib_new, 0x10B2)
        ckey = ("swap_key", "{}")
        plugin._PIPELINE_CACHE[ckey] = new_eng
        self.assertFalse(plugin._retire_handle(ckey, old_eng, None, reason="stale", force_shutdown=True))
        self.assertIs(plugin._PIPELINE_CACHE.get(ckey), new_eng)
        self.assertIsNotNone(new_eng.xfm)
        self.assertEqual(lib_new.destroy_calls, [])
        self.assertEqual(lib_old.destroy_calls, [])

        # (c) concurrent _shutdown_all_pipelines() vs _sweep_dead_pipelines() race
        race_libs = []
        for idx in range(24):
            lib_r = _RecordingLib()
            race_libs.append(lib_r)
            h_r = self._make_handle(lib_r, 0x10C0 + idx)
            k_r = ("race", idx)
            plugin._PIPELINE_CACHE[k_r] = h_r
            # dead weakref so _sweep_dead_pipelines sees k_r as dead
            dead_consumer = _Consumer()
            plugin._PIPELINE_MODELS[k_r] = [weakref.ref(dead_consumer)]
            del dead_consumer
        gc.collect()

        barrier = threading.Barrier(4)
        race_errs = []

        def _sweeper():
            try:
                barrier.wait(timeout=5.0)
                plugin._sweep_dead_pipelines(("keep_none", -1))
            except Exception as exc:  # noqa: BLE001
                race_errs.append(exc)

        def _shutter():
            try:
                barrier.wait(timeout=5.0)
                plugin._shutdown_all_pipelines()
            except Exception as exc:  # noqa: BLE001
                race_errs.append(exc)

        workers = [threading.Thread(target=_sweeper), threading.Thread(target=_sweeper),
                   threading.Thread(target=_shutter), threading.Thread(target=_shutter)]
        for t in workers:
            t.start()
        for t in workers:
            t.join(timeout=10.0)
        self.assertEqual(race_errs, [])
        self.assertTrue(all(len(lb.destroy_calls) <= 1 for lb in race_libs))
        self.assertEqual(plugin._PIPELINE_CACHE, {})
        self.assertEqual(plugin._PIPELINE_MODELS, {})
        self.assertEqual(plugin._PIPELINE_SIGS, {})

        # (d) degraded import (plugin.qfe is None AND real degraded import with failing qf_engine)
        with mock.patch.object(plugin, "qfe", None):
            plugin._shutdown_all_pipelines()
        self.assertEqual(plugin._PIPELINE_CACHE, {})

        with tempfile.TemporaryDirectory() as td:
            sub = os.path.join(td, "degraded_pkg")
            os.makedirs(sub)
            with open(os.path.join(sub, "__init__.py"), "w", encoding="utf-8") as fh:
                fh.write((_PLUGIN_DIR / "__init__.py").read_text(encoding="utf-8"))
            with open(os.path.join(sub, "qf_engine.py"), "w", encoding="utf-8") as fh:
                fh.write("raise RuntimeError('simulated degraded qf_engine import')\n")
            spec_deg = importlib.util.spec_from_file_location(
                "qf_degraded_test_pkg",
                os.path.join(sub, "__init__.py"),
                submodule_search_locations=[sub],
            )
            mod_deg = importlib.util.module_from_spec(spec_deg)
            sys.modules["qf_degraded_test_pkg"] = mod_deg
            try:
                with mock.patch("logging.Logger.warning"):
                    spec_deg.loader.exec_module(mod_deg)
                self.assertIsNone(mod_deg.qfe)
                mod_deg._shutdown_all_pipelines()
            finally:
                atexit.unregister(mod_deg._shutdown_all_pipelines)
                for k in [name for name in sys.modules if name == "qf_degraded_test_pkg" or name.startswith("qf_degraded_test_pkg.")]:
                    sys.modules.pop(k, None)

        # (e) concurrent single-entry replacement during _shutdown_all_pipelines():
        # pass 1 sees old_eng replaced by new_eng -> returns False (retired_any=False) -> pass 2 retires new_eng
        lib_old_e, lib_new_e = _RecordingLib(), _RecordingLib()
        old_e = self._make_handle(lib_old_e, 0x10E1)
        new_e = self._make_handle(lib_new_e, 0x10E2)
        ck_e = ("single_replace", "{}")
        plugin._PIPELINE_CACHE[ck_e] = old_e
        real_retire = plugin._retire_handle
        calls_e = []

        def _spy_retire(ckey, eng, requester=None, **kwargs):
            calls_e.append((ckey, eng))
            if len(calls_e) == 1:
                with plugin._ENGINE_IDENTITY_LOCK:
                    plugin._PIPELINE_CACHE[ck_e] = new_e
            return real_retire(ckey, eng, requester, **kwargs)

        with mock.patch.object(plugin, "_retire_handle", side_effect=_spy_retire):
            plugin._shutdown_all_pipelines()

        self.assertEqual(len(calls_e), 2)
        self.assertIs(calls_e[0][1], old_e)
        self.assertIs(calls_e[1][1], new_e)
        self.assertEqual(len(lib_old_e.destroy_calls), 0)
        self.assertEqual(len(lib_new_e.destroy_calls), 1)
        self.assertIsNone(new_e.xfm)
        self.assertEqual(plugin._PIPELINE_CACHE, {})
        self.assertEqual(plugin._PIPELINE_MODELS, {})
        self.assertEqual(plugin._PIPELINE_SIGS, {})

        # (f) unchanged cache when _retire_handle returns False -> breaks after 1 pass and clears all dicts
        h_unch = self._make_handle(_RecordingLib(), 0x10F1)
        c_unch = _Consumer()
        plugin._PIPELINE_CACHE[("unchanged", "{}")] = h_unch
        plugin._PIPELINE_MODELS[("unchanged", "{}")] = [weakref.ref(c_unch)]
        plugin._PIPELINE_SIGS[("unchanged", "{}")] = {("sig", 1)}
        unch_calls = []
        with mock.patch.object(plugin, "_retire_handle", side_effect=lambda *a, **kw: unch_calls.append(a) or False):
            plugin._shutdown_all_pipelines()
        self.assertEqual(len(unch_calls), 1)
        self.assertEqual(plugin._PIPELINE_CACHE, {})
        self.assertEqual(plugin._PIPELINE_MODELS, {})
        self.assertEqual(plugin._PIPELINE_SIGS, {})

        # (g) 8-pass exhaustion fallback when cache keeps changing >= 8 times -> exits in 8 passes and clears
        thrash_handles = [self._make_handle(_RecordingLib(), 0x10F2 + i) for i in range(10)]
        plugin._PIPELINE_CACHE[("thrash", "{}")] = thrash_handles[0]
        plugin._PIPELINE_MODELS[("thrash", "{}")] = [weakref.ref(c_unch)]
        plugin._PIPELINE_SIGS[("thrash", "{}")] = {("sig", 2)}
        thrash_idx = [0]

        def _thrash_retire(ckey, eng, requester=None, **kwargs):
            thrash_idx[0] += 1
            with plugin._ENGINE_IDENTITY_LOCK:
                plugin._PIPELINE_CACHE[("thrash", "{}")] = thrash_handles[thrash_idx[0]]
            return False

        with mock.patch.object(plugin, "_retire_handle", side_effect=_thrash_retire):
            plugin._shutdown_all_pipelines()
        self.assertEqual(thrash_idx[0], 8)
        self.assertEqual(plugin._PIPELINE_CACHE, {})
        self.assertEqual(plugin._PIPELINE_MODELS, {})
        self.assertEqual(plugin._PIPELINE_SIGS, {})

        # (h) _snapshot_pending_reference_audio_owners() raising is caught cleanly (pending = ())
        lib_snap_ok = _RecordingLib()
        h_snap_ok = self._make_handle(lib_snap_ok, 0x10F9)
        plugin._PIPELINE_CACHE[("snap_err", "{}")] = h_snap_ok
        with mock.patch.object(qfe, "_snapshot_pending_reference_audio_owners",
                               side_effect=RuntimeError("snapshot failure")):
            plugin._shutdown_all_pipelines()
        self.assertEqual(len(lib_snap_ok.destroy_calls), 1)
        self.assertIsNone(h_snap_ok.xfm)
        self.assertEqual(plugin._PIPELINE_CACHE, {})

    # ── C12 ────────────────────────────────────────────────────────────────────
    def test_c12_cyclic_gc_reentrancy_and_zero_lock_nesting(self):
        self.assertIsInstance(qfe._PENDING_OWNERS_LOCK, type(threading.RLock()))
        probe_h = self._make_handle(_RecordingLib(), 0x1200)
        self.assertIsInstance(probe_h._destroy_lock, type(threading.RLock()))

        # (a) Zero lock nesting: whenever _PENDING_OWNERS_LOCK is acquired via _flush_pending_discards,
        # _destroy_lock must already be released (depth == 0 and acquirable from another thread)
        nested_observations = []
        lib_nest = _RecordingLib()
        h_nest = self._make_handle(lib_nest, 0x1201)
        h_nest.current_session = ctypes.c_void_p(0x9201)
        h_nest.retain_reference_audio(_AudioOwner())
        real_flush = qfe._flush_pending_discards

        def _spy_flush(items):
            other_acquired = []

            def _probe():
                got = h_nest._destroy_lock.acquire(blocking=False)
                other_acquired.append(got)
                if got:
                    h_nest._destroy_lock.release()

            t = threading.Thread(target=_probe)
            t.start()
            t.join(timeout=5.0)
            nested_observations.append((getattr(qfe._THREAD_DESTROY_STATE, "depth", 0), other_acquired[0]))
            return real_flush(items)

        with mock.patch.object(qfe, "_flush_pending_discards", side_effect=_spy_flush):
            h_nest._destroy_for_shutdown()
        self.assertEqual(nested_observations, [(0, True)])
        self.assertIsNone(h_nest.xfm)

        # (b) Aggressive cyclic GC (gc.set_threshold(1, 0, 0)) while _PENDING_OWNERS_LOCK is held
        # and during outer destroy() / _destroy_for_shutdown()
        saved_thresh = gc.get_threshold()
        gc_destroy_calls = []
        try:
            gc.set_threshold(1, 0, 0)

            class _CycleLib(_RecordingLib):
                def qf_xfm_destroy(self, xfm):
                    gc_destroy_calls.append(int(xfm.value))
                    super().qf_xfm_destroy(xfm)

            with qfe._PENDING_OWNERS_LOCK:
                c1 = qfe.QFEngineHandle(_CycleLib(), ctypes.c_void_p(0x1211))
                c2 = qfe.QFEngineHandle(_CycleLib(), ctypes.c_void_p(0x1212))
                c1.current_session = ctypes.c_void_p(0x9211)
                c2.current_session = ctypes.c_void_p(0x9212)
                c1._cycle_ref = c2
                c2._cycle_ref = c1
                del c1, c2
                # Trigger allocations/cyclic GC while _PENDING_OWNERS_LOCK is held on this thread
                for _ in range(16):
                    _ = [[i] for i in range(32)]
                gc.collect()

            self.assertEqual(sorted(gc_destroy_calls), [0x1211, 0x1212])

            # Nested cyclic GC inside outer destroy() callback: inner handle's discard waits for outer unlock
            inner_lib = _CycleLib()
            outer_holder = []

            def _spawn_and_collect_cycle(_xfm):
                h_in = qfe.QFEngineHandle(inner_lib, ctypes.c_void_p(0x1213))
                h_in.current_session = ctypes.c_void_p(0x9213)
                h_in._cycle_self = h_in
                del h_in
                gc.collect()

            outer_lib = _CycleLib(on_destroy=_spawn_and_collect_cycle)
            h_out = self._make_handle(outer_lib, 0x1214)
            h_out.current_session = ctypes.c_void_p(0x9214)
            outer_holder.append(h_out)
            h_out.destroy()
            self.assertIn(0x1213, gc_destroy_calls)
            self.assertIn(0x1214, gc_destroy_calls)
            self.assertEqual(getattr(qfe._THREAD_DESTROY_STATE, "depth", 0), 0)

            # (c) Deferred discard while _snapshot_pending_reference_audio_owners() iterates
            h_snap_d = self._make_handle(_RecordingLib(), 0x1215)
            h_snap_d.retain_reference_audio(_AudioOwner())
            with qfe._PENDING_OWNERS_LOCK:
                qfe._PENDING_SNAPSHOT_STACK.append(None)
                try:
                    h_snap_d.reference_audio_copy_completed()
                    self.assertIn(h_snap_d, qfe._REFERENCE_AUDIO_PENDING_OWNERS)
                finally:
                    qfe._PENDING_SNAPSHOT_STACK.pop()
            # First snapshot drains _DEFERRED_PENDING_DISCARDS in finally; subsequent read sees it gone
            _ = qfe._snapshot_pending_reference_audio_owners()
            self.assertNotIn(h_snap_d, qfe._REFERENCE_AUDIO_PENDING_OWNERS)
            self.assertNotIn(h_snap_d, qfe._snapshot_pending_reference_audio_owners())
            self.assertEqual(qfe._DEFERRED_PENDING_DISCARDS, [])
        finally:
            gc.set_threshold(*saved_thresh)

    # ── C11 ────────────────────────────────────────────────────────────────────
    def test_c11_readme_recent_updates_and_no_datacenter_gpu_names(self):
        readme = (_PLUGIN_DIR / "README.md").read_text(encoding="utf-8")
        section = readme.split("## Recent Updates", 1)[1].split("\n## ", 1)[0]
        headings = re.findall(r"^### Engine .+$", section, flags=re.M)
        self.assertLessEqual(len(headings), 3, f"Recent Updates headings={headings}")
        self.assertTrue(headings and headings[0].startswith("### Engine 0.0.19 · Plugin 0.0.09"), headings)
        user_guide = (_PLUGIN_DIR / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8")
        self.assertIn("(0.0.19, Linux & Windows)", user_guide)
        forbidden_gpu_pat = re.compile(
            r"\b(?:" + "|".join(p + n for p, n in (("A", "100"), ("H", "100"), ("H", "200"),
                                                   ("B", "100"), ("B", "200"), ("GB", "300"))) + r")\b"
        )
        for rel in ("README.md", "docs/USER_GUIDE.md", "tests/atexit_cleanup_test.py"):
            txt = (_PLUGIN_DIR / rel).read_text(encoding="utf-8")
            self.assertEqual(forbidden_gpu_pat.findall(txt), [], f"{rel} contains forbidden GPU name")


if __name__ == "__main__":
    unittest.main()
