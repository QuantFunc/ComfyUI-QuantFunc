#!/usr/bin/env python3
"""#738: a dropped model's pipeline hands its host copy back before holding it puts ComfyUI under its RAM headroom.

A 32 GB PC froze while ComfyUI staged QI-2.1's 16.7 GB TE on top of the idle Krea-2 pipeline's 8.85 GB host backup,
which the plugin swept only at its next cold create. plugin._install_host_ram_release runs that same sweep from
ComfyUI's own RAM-pressure hook (comfy.memory_management.extra_ram_release, called before every staging pin), after
ComfyUI's own release, once the available RAM is less than ComfyUI's headroom plus what the dead pipelines hold (their
capacity_bytes), and only for handles whose every consumer is gone. Run with COMFY_ROOT and its Python environment (CPU
mode).
"""
import gc
import os
import re
import unittest
import weakref
from unittest import mock

from host_loader_contract_test import plugin

import comfy.memory_management as cmm
import comfy.model_management as cmodel

GiB = 1 << 30


class Handle:
    """A cached engine handle: only what _retire_handle and the release read."""

    def __init__(self, events, capacity):
        self.lib, self.resource, self.pipeline, self.events = object(), None, object(), events
        self.capacity_bytes = capacity

    def destroy(self):
        self.events.append("destroy")


class Consumer:
    pass


class HostRamRelease(unittest.TestCase):
    """ComfyUI's headroom 3 GiB; a dropped model's pipeline holding 10 GiB is released once available < 3 + 10 GiB."""

    def setUp(self):
        self.saved = dict(plugin._PIPELINE_CACHE), {k: list(v) for k, v in plugin._PIPELINE_MODELS.items()}
        plugin._PIPELINE_CACHE.clear()
        plugin._PIPELINE_MODELS.clear()
        self.events = []
        cmm.set_ram_cache_release_state(lambda target, free_active=False: self.events.append(("comfy", target)) or 0,
                                        3 * GiB)

    def tearDown(self):
        cmm.set_ram_cache_release_state(None, 0)
        plugin._PIPELINE_CACHE.clear()
        plugin._PIPELINE_CACHE.update(self.saved[0])
        plugin._PIPELINE_MODELS.clear()
        plugin._PIPELINE_MODELS.update(self.saved[1])

    def cached(self, key, alive, capacity=10 * GiB):
        consumer, handle = Consumer(), Handle(self.events, capacity)
        plugin._PIPELINE_CACHE[key] = handle
        plugin._PIPELINE_MODELS[key] = [weakref.ref(consumer)]
        if not alive:   # ComfyUI dropped the model: its consumer is gone
            del consumer
            gc.collect()
            consumer = None
        return handle, consumer

    def stage(self, available):
        """One staging pin: ComfyUI asks its RAM cache for its headroom first."""
        with mock.patch.object(cmodel, "get_free_memory", return_value=available) as avail:
            cmm.extra_ram_release(cmm.RAM_CACHE_HEADROOM)
        return avail

    def test_installed_at_plugin_import(self):
        self.assertTrue(getattr(cmm.extra_ram_release, "_qf_host_ram_release", False))

    def test_fires_when_the_dead_holding_exceeds_the_room_above_the_headroom(self):
        self.cached("dead", alive=False)
        self.stage(12 * GiB)   # above ComfyUI's 3 GiB headroom, but less than 3 + the 10 GiB held dead
        self.assertNotIn("dead", plugin._PIPELINE_CACHE)
        self.assertEqual(self.events, [("comfy", 3 * GiB), "destroy"])   # ComfyUI's own release first, then ours

    def test_fires_under_deep_pressure(self):
        self.cached("dead", alive=False)
        self.stage(1 * GiB)
        self.assertNotIn("dead", plugin._PIPELINE_CACHE)

    def test_keeps_it_while_ram_is_ample(self):
        self.cached("dead", alive=False)
        self.stage(13 * GiB)   # exactly the headroom plus the holding: keeping it leaves ComfyUI its headroom
        self.assertIn("dead", plugin._PIPELINE_CACHE)
        self.assertEqual(self.events, [("comfy", 3 * GiB)])

    def test_no_dead_pipeline_no_read(self):
        _, keep = self.cached("live", alive=True)
        avail = self.stage(1 * GiB)
        self.assertIn("live", plugin._PIPELINE_CACHE)
        self.assertNotIn("destroy", self.events)
        self.assertFalse(avail.called)   # nothing dead: not even an availability read on this per-pin path
        del keep

    def test_one_live_consumer_keeps_the_handle(self):
        _, keep = self.cached("shared", alive=True)   # two loaders on one package: one dropped, one still live
        gone = Consumer()
        plugin._PIPELINE_MODELS["shared"].append(weakref.ref(gone))
        del gone
        gc.collect()
        avail = self.stage(1 * GiB)
        self.assertIn("shared", plugin._PIPELINE_CACHE)
        self.assertFalse(avail.called)   # not dead while any consumer lives
        del keep

    def test_unknown_capacity_falls_back_to_the_headroom(self):
        self.cached("dead", alive=False, capacity=0)   # no capacity figure: ComfyUI's headroom alone decides
        self.stage(4 * GiB)
        self.assertIn("dead", plugin._PIPELINE_CACHE)
        self.stage(2 * GiB)
        self.assertNotIn("dead", plugin._PIPELINE_CACHE)

    def test_passes_every_argument_through_and_contains_its_own_failure(self):
        """A ComfyUI that forwards more arguments (its cache's ram_release already takes min_entry_size) must reach its
        own function unchanged, get its own answer back, and never see our failure."""
        seen, installed = [], cmm.extra_ram_release

        def comfy_release(target, *args, **kwargs):
            seen.append((target, args, kwargs))
            return 7
        cmm.extra_ram_release = comfy_release
        try:
            plugin._install_host_ram_release()   # wraps this ComfyUI function
            self.cached("dead", alive=False)
            with mock.patch.object(plugin, "_sweep_dead_pipelines", side_effect=RuntimeError("native refused")), \
                    mock.patch.object(cmodel, "get_free_memory", return_value=0):
                got = cmm.extra_ram_release(3 * GiB, True, min_entry_size=5)
        finally:
            cmm.extra_ram_release = installed
        self.assertEqual((got, seen), (7, [(3 * GiB, (True,), {"min_entry_size": 5})]))

    def test_comfyui_reaches_the_hook_through_the_module_attribute(self):
        """The hook works only while ComfyUI looks it up on comfy.memory_management at call time: a `from ... import`
        would bind the original and bypass it silently. Derived from the installed ComfyUI core source."""
        root = os.environ["COMFY_ROOT"]
        files = [os.path.join(root, f) for f in ("execution.py", "main.py", "nodes.py", "server.py")]
        for sub in ("comfy", "comfy_execution", "comfy_extras"):
            for dirpath, _, names in os.walk(os.path.join(root, sub)):
                files += [os.path.join(dirpath, n) for n in names if n.endswith(".py")]
        calls, bad = 0, []
        for path in files:
            if not os.path.isfile(path):
                continue
            src = open(path, encoding="utf-8", errors="replace").read()
            for line in src.splitlines():
                if "extra_ram_release" not in line or line.lstrip().startswith(("def extra_ram_release", "#")):
                    continue
                if re.search(r"\bimport\b.*\bextra_ram_release\b", line) or re.search(r"(?<!memory_management\.)\bextra_ram_release\(", line):
                    bad.append(f"{os.path.relpath(path, root)}: {line.strip()}")
                elif "memory_management.extra_ram_release(" in line:
                    calls += 1
        self.assertEqual(bad, [])
        self.assertGreater(calls, 0)   # ComfyUI still calls it (else the hook never runs)


if __name__ == "__main__":
    result = unittest.main(argv=[__file__], exit=False, verbosity=2).result   # sys.argv carries comfy's --cpu
    print(f"HOST_RAM_RELEASE: {'PASS' if result.wasSuccessful() else 'FAIL'} ({result.testsRun} arms)")
    raise SystemExit(0 if result.wasSuccessful() else 1)
