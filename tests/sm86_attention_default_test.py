#!/usr/bin/env python3
"""SM86∧Windows-safe attention default (cluster plugin-sm86-safe-default): the (cc, platform) ->
widget-default policy is a pure data row behind one platform gate; an explicit user selection
still wins; a saved/omitted "auto" re-decides at the resolve face.

  1. PURE (any env) — the real source, not a copy: `_attn_backend_default` / `_attn_backend_choices`
     and the `_ATTN_BACKEND_*` constants are AST-extracted from __init__.py and executed in a fresh
     namespace (they live behind the plugin's ComfyUI import gate; the policy must not need ComfyUI
     to be testable). Arms (the key is cc ∧ platform): A (8,6)+win32 -> fp16_native; A-linux (8,6)
     +linux -> auto (Linux sage is measured clean and is NOT downgraded); B/C/D/G75 -> auto;
     E-win32 None -> fp16_native (FAIL-CLOSED on the crashing platform); E-linux None -> auto.
     Plus: choices(cc) identity for the whole cc table (the SM86 dropdown must still be the ADDON
     list, sage included), and the tooltip literals of both branches.
  2. MUTATION (any env) — flip the SM86 row's value to "auto" in the extracted source (unique match
     enforced) -> arm A must FAIL; the unmutated control must PASS. A syntax error or any other
     crash is not a kill.
  3. WIRING (needs the ComfyUI env; SKIPs without it) — fake torch injected via sys.modules (the
     import path the production code actually resolves; never a module attribute, which would mask
     a NameError) and a fake device through qfmp.comfy.model_management.get_torch_device:
     SM86 default/tooltip/log-once, probe counts (one device query + at most one capability call per
     INPUT_TYPES build), non-SM86 byte-identical widget, probe-failure and non-CUDA-device fallbacks,
     arm F (explicit selections pass through _attn_backend_to_engine untouched), one real loader's
     INPUT_TYPES carrying the new default.
  4. REAL (opt-in, QF_ATTN_REAL_PROBE=1) — no fakes: the box's real cc and platform drive the real
     probe; on a real LINUX SM86 device this asserts auto (sage kept — the crash is measured only
     on Windows); on a real Windows SM86 device it would assert fp16_native.

Run:  python tests/sm86_attention_default_test.py
"""
import ast
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.dirname(_HERE)
_INIT = os.path.join(_PLUGIN, "__init__.py")

_ORIG_AUTO_TOOLTIP = ("auto (default) picks the best setting for your GPU. Try another setting only if a result "
                      "looks wrong or a run fails on your GPU. Takes effect on the next run.")

# (arm, (major, minor) or None, platform, expected default value).
# The key is cc ∧ platform: the crash is measured only on SM86 under Windows, so Linux SM86 keeps
# auto (sage) and an unprobeable capability is fail-closed to fp16_native on Windows only.
_DEFAULT_TABLE = [
    ("A", (8, 6), "win32", "fp16_native"),
    ("A-linux", (8, 6), "linux", "auto"),
    ("B", (8, 0), "win32", "auto"),
    ("B-linux", (8, 0), "linux", "auto"),
    ("C", (8, 9), "win32", "auto"),
    ("D90", (9, 0), "win32", "auto"),
    ("D90-linux", (9, 0), "linux", "auto"),
    ("D100", (10, 0), "win32", "auto"),
    ("D120", (12, 0), "win32", "auto"),
    ("E-win32", None, "win32", "fp16_native"),
    ("E-linux", None, "linux", "auto"),
    ("G75", (7, 5), "win32", "auto"),
    ("G75-linux", (7, 5), "linux", "auto"),
]

# (major, minor) -> which module constant the dropdown must be (identity, not equality)
_CHOICES_TABLE = [
    ((7, 5), "_ATTN_BACKEND_SM75"),
    ((8, 0), "_ATTN_BACKEND_SM80PLUS"),
    ((8, 6), "_ATTN_BACKEND_ADDON"),
    ((8, 9), "_ATTN_BACKEND_ADDON"),
    ((9, 0), "_ATTN_BACKEND_SM80PLUS"),
    ((10, 0), "_ATTN_BACKEND_SM80PLUS"),
    ((12, 0), "_ATTN_BACKEND_ADDON"),
    (None, "_ATTN_BACKEND_SM80PLUS"),
]


def _source():
    with open(_INIT, encoding="utf-8") as f:
        return f.read()


def _extract_fn_src(src, name):
    """The real FunctionDef source of `name`, wherever it sits in __init__.py (the helpers are
    indented inside the plugin's import gate)."""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def _extract_assign(src, name):
    """The evaluated right-hand side of a module-level `name = ...` assignment (the choice lists)."""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def _policy_namespace(src):
    """A fresh namespace holding the extracted functions + the real choice-list objects."""
    ns = {}
    for const in ("_ATTN_BACKEND_SM80PLUS", "_ATTN_BACKEND_ADDON", "_ATTN_BACKEND_SM75", "_ATTN_ADDON_SMS"):
        ns[const] = _extract_assign(src, const)
        if ns[const] is None:
            raise AssertionError(f"{const} not found in __init__.py")
    for fn in ("_attn_backend_default", "_attn_backend_choices"):
        code = _extract_fn_src(src, fn)
        if not code:
            raise AssertionError(f"{fn} not found in __init__.py")
        exec(compile(code, _INIT, "exec"), ns)  # noqa: S102 - the repo's own source, test-only namespace
    return ns


def _default_rows(src):
    """The data rows of _attn_backend_default, extracted from the real source (the class
    invariant must hold for rows added tomorrow, not only for today's)."""
    fn_src = _extract_fn_src(src, "_attn_backend_default") or ""
    for node in ast.walk(ast.parse(fn_src)):
        if isinstance(node, ast.Dict):
            return {ast.literal_eval(k): ast.literal_eval(v) for k, v in zip(node.keys, node.values)}
    return {}


def run_pure(src, check):
    ns = _policy_namespace(src)
    default_fn = ns["_attn_backend_default"]
    for arm, cc, platform, want in _DEFAULT_TABLE:
        got = default_fn(cc, platform)
        check(f"{arm}: cc={cc} platform={platform} -> default {want} (got {got[0]})", got[0] == want)
    for arm in ("A", "E-win32"):
        cc, platform = next((c, pl) for a, c, pl, _ in _DEFAULT_TABLE if a == arm)
        check(f"{arm}: the fail/row entry carries a reason", bool(default_fn(cc, platform)[1]))
    for cc, const in _CHOICES_TABLE:
        got = ns["_attn_backend_choices"](cc)
        check(f"choices({cc}) is {const} (identity)", got is ns[const])

    # class invariants over the data rows themselves (a future row cannot ship half-armed):
    rows = _default_rows(src)
    table_ccs = {r for _, r, _, _ in _DEFAULT_TABLE}
    check(f"row-class: every data row {sorted(str(r) for r in rows)} has a table arm", set(rows) <= table_ccs)
    for cc, (dflt, why) in sorted(rows.items()):
        check(f"row-class: {cc} default {dflt!r} is offered by the dropdown and carries a reason",
              dflt in ns["_attn_backend_choices"](cc) and bool(why))

    # the tooltip literals of both branches (AST over the real function source) — scanned with the
    # FULL banned-terms machinery, not a sampled word list: the SM86 branch is user-visible text
    import _banned_terms as _bt
    input_src = _extract_fn_src(src, "_attn_backend_input") or ""
    strings = "".join(v.value for v in ast.walk(ast.parse(input_src))
                      if isinstance(v, ast.Constant) and isinstance(v.value, str))
    check("non-SM86 tooltip keeps the original auto-default copy", _ORIG_AUTO_TOOLTIP in strings)
    check("'auto (default)' survives only in the auto branch", strings.count("auto (default)") == 1)
    check("SM86 tooltip names the new default per GPU", "is the default on this GPU (SM" in strings
          and "capability could not be probed" in strings)
    _frag_hits = [x for x in _bt.desc_hits(strings) + _bt.term_hits(strings) + _bt.mode_id_hits(strings)
                  if x != "sage"]   # the [SM86] tooltip's 'sage' is sanctioned (user order 2026-10-29)
    check(f"tooltip strings carry no technique word beyond the sanctioned sage "
          f"(full DESC_* machinery; hits {_frag_hits})", not _frag_hits)

    # W4: exactly one _attn_backend_input() call inside each of the four loaders, none elsewhere
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_attn_backend_input"]
    loaders = ("QuantFuncLTXLoader", "QuantFuncKrea2Loader", "QuantFuncQwenImage21Loader", "QuantFuncH3Loader")
    per = {c: 0 for c in loaders}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name in per]:
        per[cls.name] = sum(1 for n in ast.walk(cls) if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Name) and n.func.id == "_attn_backend_input")
    check(f"W4: exactly 4 call sites, one per loader ({per})", len(calls) == 4 and all(v == 1 for v in per.values()))
    # resolve call-site arm (round-6): every loader's load() must route attention_backend through
    # _attn_backend_to_engine — a loader that skips the resolve would show fp16_native in the
    # widget while the engine receives raw auto (Windows+SM86 crash re-opens).
    tree_full = ast.parse(src)
    resolve_arms = {}
    for cls in [n for n in ast.walk(tree_full) if isinstance(n, ast.ClassDef) and n.name in per]:
        loads = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load"]
        if not loads:
            resolve_arms[cls.name] = "NO load()"
            continue
        n_uses = sum(1 for n in ast.walk(loads[0])
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                     and n.func.id == "_attn_backend_to_engine")
        resolve_arms[cls.name] = n_uses
    check(f"resolve call sites: all four loaders route attention_backend through "
          f"_attn_backend_to_engine in load() ({resolve_arms})",
          all(v == 1 for v in resolve_arms.values()) and len(resolve_arms) == 4)
    # (the check() helper is 2-arg; the detail lives in the label above)


def _mutated(src, needle, repl, what):
    code = _extract_fn_src(src, "_attn_backend_default")
    n = code.count(needle)
    if n != 1:
        raise AssertionError(f"mutation needle {needle!r} matched {n} times (need exactly 1): {what}")
    ns = {}
    exec(compile(code.replace(needle, repl), _INIT + "<mutated>", "exec"), ns)  # noqa: S102
    return ns["_attn_backend_default"]


def run_mutation(src, check):
    control = _policy_namespace(src)["_attn_backend_default"]
    check("MUT control: unmutated arm A passes (win32)", control((8, 6), "win32")[0] == "fp16_native")
    row_flip = _mutated(src, '("fp16_native", "sage crashes on Windows")', '("auto", "sage crashes on Windows")',
                        "row flip")
    check("MUT row-flip: SM86 row -> auto, arm A FAILS on win32", row_flip((8, 6), "win32")[0] != "fp16_native")
    check("MUT row-flip: only the value moved (arm B still auto, A-linux still auto)",
          row_flip((8, 0), "win32")[0] == "auto" and row_flip((8, 6), "linux")[0] == "auto")
    plat_flip = _mutated(src, '"win32"', '"linux32"', "platform flip")
    check("MUT platform-flip: the win32 gate broken -> arm A FAILS", plat_flip((8, 6), "win32")[0] != "fp16_native")
    check("MUT platform-flip: non-win32 stays auto", plat_flip((8, 6), "linux")[0] == "auto")


def _load_plugin():
    """Import the plugin package the way ComfyUI does (by file path), with a real comfy + the
    plugin dir on sys.path (the plugin imports its sibling qf_* modules by bare name)."""
    comfy_root = os.environ.get("COMFY_ROOT")
    if not comfy_root:
        guess = os.path.dirname(os.path.dirname(_PLUGIN))
        comfy_root = guess if os.path.isfile(os.path.join(guess, "folder_paths.py")) else None
    if not comfy_root:
        return None, "no COMFY_ROOT and no folder_paths.py next to the plugin"
    for p in (comfy_root, _PLUGIN):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        # The contract tests' idiom: run ComfyUI's import in its own --cpu mode so a box with no
        # visible GPU still imports comfy (and no CUDA context is created here). Must precede the
        # first comfy import. QF_ATTN_REAL_PROBE=1 opts OUT of --cpu: that section needs the real
        # device query (a real CUDA context, tiny).
        sys.argv = [sys.argv[0]] + ([] if os.environ.get("QF_ATTN_REAL_PROBE") == "1" else ["--cpu"])
        import comfy.options
        comfy.options.enable_args_parsing()
        import importlib.util
        import folder_paths  # noqa: F401 — proves the env is real
        spec = importlib.util.spec_from_file_location("qf_plugin_sm86_default_under_test", _INIT)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["qf_plugin_sm86_default_under_test"] = mod
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 - a broken import in a live env is a FAIL, not a skip
        return None, f"plugin import raised: {type(e).__name__}: {e}"
    if not getattr(mod, "_IMPORT_OK", False):
        return None, "plugin imported but _IMPORT_OK is False (comfy imports failed)"
    return mod, None


class _FakeTorch(types.ModuleType):
    def __init__(self, cc):
        super().__init__("torch")
        self.cap_calls = []
        self.cuda = types.SimpleNamespace(
            get_device_capability=lambda dev: (self.cap_calls.append(dev), cc)[1])


def run_wiring(plugin, check):
    import _banned_terms as _bt
    mm = plugin.qfmp.comfy.model_management
    real_get_dev, real_log = mm.get_torch_device, plugin._log
    real_torch, real_platform = sys.modules.get("torch"), sys.platform
    try:
        def _set_platform(p):
            sys.platform = p
        # --- SM86: default, choices identity, tooltip, probe counts, log-once -------------------
        fake_torch = _FakeTorch((8, 6))
        sys.modules["torch"] = fake_torch
        dev_queries = []
        mm.get_torch_device = lambda: (dev_queries.append(1), types.SimpleNamespace(type="cuda"))[1]
        infos = []

        def _note(*a):  # render the %-format the way the real logger would
            infos.append(a[0] % a[1:] if len(a) > 1 else a[0])
        plugin._log = types.SimpleNamespace(info=_note, warning=lambda *a, **k: None)
        plugin._SM86_ATTENTION_DEFAULT_NOTED = False
        _set_platform("win32")   # the SM86 branch exists only under Windows (the measured platform)
        choices, opts = plugin._attn_backend_input()
        check("W1 SM86: default fp16_native", opts["default"] == "fp16_native")
        check("W1 SM86: dropdown is still the ADDON list, sage included",
              choices is plugin._ATTN_BACKEND_ADDON and "sage" in choices)
        hits = _bt.desc_hits(opts["tooltip"]) + _bt.term_hits(opts["tooltip"]) + _bt.mode_id_hits(opts["tooltip"])
        check("W1 SM86: tooltip names fp16_native + SM 8.6 + the measured crash (sage/Windows — sanctioned by "
              "user order 2026-10-29); no OTHER technique word; never 'auto (default)' — hits " + str(hits),
              opts["tooltip"].startswith("fp16_native is the default on this GPU (SM 8.6): sage crashes on Windows")
              and "auto (default)" not in opts["tooltip"]
              and [x for x in hits if x != "sage"] == [] and hits.count("sage") == 2)
        check("W1 exactly one device query + one capability call per widget build",
              len(dev_queries) == 1 and len(fake_torch.cap_calls) == 1)
        plugin._attn_backend_input()
        plugin._attn_backend_input()   # a second and third loader on the same process
        check("W2 the SM86 notice logs exactly once across three widget builds",
              len(infos) == 1 and "fp16_native" in infos[0] and "SM 8.6" in infos[0])

        # the SAVED-AUTO arm (ruling #2, the heaviest item): a stored "auto" — and an omitted
        # input, which reaches load() as the signature default "auto" — re-decides at the resolve
        # face instead of sailing to the engine's sage. Explicit names are never rewritten.
        plugin._ATTN_AUTO_REDECIDE_NOTED = False
        check("W-saved-auto: to_engine('auto') re-decides to native on Windows+SM86",
              plugin._attn_backend_to_engine("auto") == "native")
        check("W-saved-auto: the re-decision announces itself exactly once",
              sum("re-decided" in s for s in infos) == 1)
        check("W-saved-auto: to_engine(None) (omitted input) re-decides too",
              plugin._attn_backend_to_engine(None) == "native")
        check("W-saved-auto: the re-decision notice stays once-per-process",
              sum("re-decided" in s for s in infos) == 1)
        plugin._ATTN_AUTO_REDECIDE_NOTED = False   # separate face: the explicit-sage warning
        warns = []
        plugin._log = types.SimpleNamespace(info=_note, warning=lambda *a, **k: warns.append(
            a[0] % a[1:] if len(a) > 1 else a[0]))
        check("W-explicit-sage: an explicitly chosen sage is preserved (never rewritten)",
              plugin._attn_backend_to_engine("sage") == "sage")
        check("W-explicit-sage: the measured reason is announced at WARNING before it happens",
              len(warns) == 1 and "sage crashes on Windows" in warns[0] and "pick a different backend" in warns[0])
        # the unprobeable sibling must NOT claim the crash (seat A round-4): the reason travels
        plugin._SM86_ATTENTION_DEFAULT_NOTED = False
        warns.clear()

        _gd = mm.get_torch_device
        mm.get_torch_device = lambda: (_ for _ in ()).throw(RuntimeError("no probe"))
        sys.modules["torch"] = _FakeTorch((8, 6))
        _set_platform("win32")
        check("W-explicit-sage unprobed: sage preserved, reason = probe fact only "
              "(no crash claim, no false default-claim)",
              plugin._attn_backend_to_engine("sage") == "sage"
              and len(warns) == 1 and "could not be probed" in warns[0]
              and "crashes" not in warns[0] and "keeping the safe default" not in warns[0])
        mm.get_torch_device = _gd   # restore: the later arms need a working probe

        # the PLATFORM gate, live: the same SM86 device on Linux keeps auto + the original tooltip
        plugin._SM86_ATTENTION_DEFAULT_NOTED = False
        infos.clear()
        _set_platform("linux")
        choices, opts = plugin._attn_backend_input()
        check("W-lin Linux SM86: default stays auto (sage kept), original tooltip byte-exact",
              opts["default"] == "auto" and opts["tooltip"] == _ORIG_AUTO_TOOLTIP)
        check("W-lin Linux SM86: no SM86 notice, to_engine('auto') stays auto",
              not infos and plugin._attn_backend_to_engine("auto") == "auto")
        _set_platform("win32")

        # --- non-SM86: byte-identical widget, silent -------------------------------------------
        for label, cc, const in (("W3", (8, 9), "_ATTN_BACKEND_ADDON"), ("W4", (7, 5), "_ATTN_BACKEND_SM75")):
            plugin._SM86_ATTENTION_DEFAULT_NOTED = False
            infos.clear()
            sys.modules["torch"] = _FakeTorch(cc)
            choices, opts = plugin._attn_backend_input()
            check(f"{label} cc={cc}: default auto, original tooltip byte-exact",
                  opts["default"] == "auto" and opts["tooltip"] == _ORIG_AUTO_TOOLTIP)
            check(f"{label} cc={cc}: dropdown is {const}", choices is getattr(plugin, const))
            check(f"{label} cc={cc}: no SM86 notice", not infos)

        # --- probe failure / non-CUDA device: Linux falls back to auto (original); Windows is
        # --- FAIL-CLOSED to fp16_native (ruling #3: the crashing platform must not re-open)
        for label, dev, platform, want in (("W5-lin probe raises", None, "linux", "auto"),
                                           ("W5-win probe raises", None, "win32", "fp16_native"),
                                           ("W6-lin non-CUDA device", "cpu", "linux", "auto"),
                                           ("W6-win non-CUDA device", "cpu", "win32", "fp16_native")):
            plugin._SM86_ATTENTION_DEFAULT_NOTED = False
            infos.clear()
            fake_torch = _FakeTorch((8, 6))

            def _boom(dev=dev):
                if dev is None:
                    raise RuntimeError("no device")
                return dev
            mm.get_torch_device = _boom
            _set_platform(platform)
            sys.modules["torch"] = fake_torch
            choices, opts = plugin._attn_backend_input()
            fallback = plugin._ATTN_BACKEND_SM80PLUS
            check(f"{label}: default {want}, dropdown falls back to SM80PLUS",
                  opts["default"] == want and choices is plugin._ATTN_BACKEND_SM80PLUS)
            check(f"{label}: capability never called", not fake_torch.cap_calls)
            if platform == "win32":   # fail-closed IS the notice-worthy event: one INFO, then quiet
                check(f"{label}: the fail-closed default announces itself once (neutral probe fact)",
                      len(infos) == 1 and "capability unavailable" in infos[0]
                      and "crash" not in infos[0])
                check(f"{label}: the fail-closed tooltip states the probe fact, never a crash claim",
                      "could not be probed" in opts["tooltip"] and "crash" not in opts["tooltip"])
            else:
                check(f"{label}: no SM86 notice on Linux", not infos)

        # --- arm F: an explicit user selection is never rewritten (on ANY platform) ------------
        # re-arm a WORKING SM86 probe: the W5/W6 loop left a raising stub + the platform unset,
        # which silently degraded these arms to the fail-closed row (seat B round-5, MEASURED)
        mm.get_torch_device = lambda: (dev_queries.append(1), types.SimpleNamespace(type="cuda"))[1]
        sys.modules["torch"] = _FakeTorch((8, 6))
        _set_platform("win32")
        plugin._ATTN_AUTO_REDECIDE_NOTED = False
        for plat in ("win32", "linux"):
            _set_platform(plat)
            check(f"F[{plat}] explicit sage/flash stay themselves",
                  plugin._attn_backend_to_engine("sage") == "sage"
                  and plugin._attn_backend_to_engine("flash") == "flash")
            check(f"F[{plat}] explicit fp16_native maps to the engine's native",
                  plugin._attn_backend_to_engine("fp16_native") == "native")
        _set_platform("win32")
        check("F[win32+SM86] UNSET auto is the one value that re-decides (-> native)",
              plugin._attn_backend_to_engine("auto") == "native")
        _set_platform("linux")
        check("F[linux+SM86] UNSET auto re-decides to auto (Linux keeps sage)",
              plugin._attn_backend_to_engine("auto") == "auto")

        # --- one real loader carries the new default end to end --------------------------------
        _set_platform("win32")
        sys.modules["torch"] = _FakeTorch((8, 6))
        mm.get_torch_device = lambda: types.SimpleNamespace(type="cuda")
        plugin._SM86_ATTENTION_DEFAULT_NOTED = False
        spec = plugin.QuantFuncLTXLoader.INPUT_TYPES()
        attn = spec["optional"]["attention_backend"]
        check("W8 QuantFuncLTXLoader.INPUT_TYPES on SM86 offers the ADDON list with default fp16_native",
              attn[0] is plugin._ATTN_BACKEND_ADDON and attn[1]["default"] == "fp16_native")
    finally:
        mm.get_torch_device = real_get_dev
        plugin._log = real_log
        sys.platform = real_platform
        if real_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = real_torch


def run_real(check):
    """No fakes: the box's own torch + device. On a real SM86 device this asserts the fix."""
    plugin, reason = _load_plugin()
    if plugin is None:
        check(f"REAL skipped ({reason})", True)
        return
    cc = plugin._attn_device_cc()
    default = plugin._attn_backend_default(cc, sys.platform)[0]
    if cc == (8, 6) and sys.platform == "win32":
        check(f"REAL Windows SM86 device: default is fp16_native (cc={cc}, default={default})",
              default == "fp16_native")
    else:
        want = "auto"
        check(f"REAL device (this box: Linux): default stays auto (cc={cc}, default={default}; "
              "sage kept — the crash is measured only on Windows)", default == want)


def main():
    failures = []

    passed = [0]

    def check(label, ok):
        print(f"{'PASS' if ok else 'FAIL'} {label}")
        if ok:
            passed[0] += 1
        else:
            failures.append(label)

    src = _source()
    run_pure(src, check)
    run_mutation(src, check)

    skips = 0
    plugin, reason = _load_plugin()
    if plugin is None:
        if "no COMFY_ROOT" in reason:
            # bare machine: a legal skip, in the runner's literal [SKIP] form so run_plugin_tests
            # counts it — never a silent exit-0 pass.
            skips += 1
            print(f"[SKIP] wiring: {reason} (pure + mutation sections above still ran)")
        else:
            # a LIVE env whose plugin import is broken means every ComfyUI user just lost the
            # nodes: that is a FAIL with a nonzero exit, never a skip (orchestrator round-6).
            failures.append(f"wiring import failed in a live env: {reason}")
            print(f"FAIL wiring: {reason}")
    else:
        run_wiring(plugin, check)
    if os.environ.get("QF_ATTN_REAL_PROBE") == "1":
        run_real(check)

    print(f"SM86_ATTENTION_DEFAULT: {'PASS' if not failures else 'FAIL'} "
          f"({passed[0]} pass, {len(failures)} failure(s), {skips} skipped section)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
