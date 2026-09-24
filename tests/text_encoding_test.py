#!/usr/bin/env python3
"""Death-rule test: the plugin reads its files the same way whatever the OS preferred encoding is (#738).

WHY THIS EXISTS: Python decodes a text file with the OS preferred encoding unless the caller names one. On Windows that
is the ANSI code page (cp936 on Chinese Windows). The Windows E2E on 远程-windows-4090 failed every native loader:
`json.load(open(mf))` read a shipped qf_native.json (UTF-8, with an em dash) as GBK and raised "'gbk' codec can't
decode byte 0x94". The keyfile (_read_auth) and the LTX connectors config (_connector_config_heads) read files the same
way, and SWALLOWED the error: no API key, or no head count, with nothing said.

ARMS. Each runs in a child Python whose preferred encoding is FORCED, and the child runs the REAL plugin code:
  static      every text-mode open / read_text / write_text / text subprocess in the repo names its encoding
  manifest    every shipped configs/*/qf_native.json loads through _load_model_config
  family      every shipped family resolves its one preset through _family_preset
  keyfile     every shipped bin/*/config.json yields an API key through _read_auth (the key is never printed)
  connectors  every shipped configs/*/connectors/config.json yields its head count through _connector_config_heads
  malformed   a manifest / keyfile / connectors config that is invalid JSON, not UTF-8, not a JSON object, or (connectors)
              declares an unusable head count raises a RuntimeError that names the file. A family whose only manifest
              is broken names that manifest instead of reporting "no model config shipped", and so does a saved
              workflow that names it (the real _run_family_load). Never a silent default.
MODES (the forced preferred encoding):
  cp936       a GBK locale compiled with localedef into a temp LOCPATH (Linux); the ANSI code page itself on Windows
  c-locale    LC_ALL=C (ascii), POSIX only
  warn        -X warn_default_encoding -W error::EncodingWarning: an encoding-less open anywhere on the path raises,
              whatever the bytes (so an ASCII-only file cannot hide the defect)
A mode this OS cannot force is a disclosed [SKIP] (run_plugin_tests.py counts it; --strict fails it), never a pass.

Run:  python3 tests/text_encoding_test.py       stdlib only: no ComfyUI, no torch, no GPU
      QF_TEXTIO_TEST_ROOT=<plugin tree> points it at another tree (e.g. the unfixed base, to show the arms go RED).
"""
import ast
import codecs
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.environ.get("QF_TEXTIO_TEST_ROOT") or os.path.join(_HERE, ".."))
_KEY_ENVS = ("QUANTFUNC_API_KEY", "QF_API_KEY")  # _read_auth prefers these: the child must not inherit them
_LOCALE_ENVS = ("PYTHONUTF8", "PYTHONIOENCODING", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "LOCPATH")


# ---------------------------------------------------------------------------------------------------- static arm --
_TEXT_MODE_CHARS = set("rwxat+")
_SUBPROCESS_TEXT_CALLS = {"run", "Popen", "check_output", "call", "check_call"}


def _kw(call, name):
    return next((k for k in call.keywords if k.arg == name), None)


def _mode_literal(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _dotted(func):
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _text_io_without_encoding(call):
    """The call's name when it opens / decodes text with the OS preferred encoding because it names none, else None."""
    name, has_enc = _dotted(call.func), _kw(call, "encoding") is not None
    last = name.rsplit(".", 1)[-1]
    if has_enc:
        return None
    if name in ("open", "io.open", "codecs.open", "os.fdopen"):
        mode = _kw(call, "mode") or (call.args[1] if len(call.args) > 1 else None)
        literal = "r" if mode is None else _mode_literal(mode)
        positional_enc = name in ("open", "io.open") and len(call.args) >= 4
        return None if positional_enc or (literal is not None and "b" in literal) else name
    if isinstance(call.func, ast.Attribute) and last == "open":
        # Path.open(): text unless the mode says "b". Another receiver's .open(x) (urllib's opener, zipfile) takes no
        # mode string first, so only a no-argument call or a mode-string literal counts.
        mode = _kw(call, "mode") or (call.args[0] if call.args else None)
        literal = _mode_literal(mode) if mode is not None else "r"
        if literal is None or not literal or not set(literal) <= _TEXT_MODE_CHARS | {"b"} or "b" in literal:
            return None
        return name
    if last == "read_text":
        return None if call.args else name
    if last == "write_text":
        return None if len(call.args) >= 2 else name
    if name.startswith("subprocess.") and last in _SUBPROCESS_TEXT_CALLS:
        text = [k for k in call.keywords if k.arg in ("text", "universal_newlines")]
        if any(not (isinstance(k.value, ast.Constant) and not k.value.value) for k in text):
            return name
        return None
    if last in ("NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"):
        mode = _kw(call, "mode") or (call.args[0] if call.args else None)
        literal = _mode_literal(mode) if mode is not None else "w+b"
        return None if literal is not None and "b" in literal else name
    if last in ("FileHandler", "RotatingFileHandler", "TimedRotatingFileHandler", "WatchedFileHandler",
                "TextIOWrapper") or name == "os.popen":
        return name
    return None


def _static_sites(root):
    sites = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in (".git", "__pycache__")]
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dp, fn)
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = _text_io_without_encoding(node)
                    if name:
                        sites.append(f"{os.path.relpath(path, root)}:{node.lineno} {name}(...)")
    return sorted(sites)


# ----------------------------------------------------------------------------------------------------- the child --
def _import_plugin(root):
    """The plugin package the way ComfyUI imports it (by file path), with torch BLOCKED and comfy stubbed: the readers
    under test are stdlib code, so no ComfyUI, torch or GPU is needed (__init__ degrades to _IMPORT_OK=False, with qfe
    bound)."""
    import importlib.util
    import types
    sys.modules["torch"] = None
    for name in ("comfy", "comfy.model_management", "comfy.supported_models"):
        sys.modules[name] = types.ModuleType(name)
    spec = importlib.util.spec_from_file_location("qfn_textio_pkg", os.path.join(root, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qfn_textio_pkg"] = mod
    spec.loader.exec_module(mod)
    return mod


def _extract(path, name, ns, consts=()):
    """The function `name` from the REAL source file, defined in `ns` (with the module constants named in `consts`), or
    None when the source lacks it. For code that cannot be imported light: qf_ltx_modelpatcher imports torch + comfy
    (connector_arch_derivation_test.py drives _derive_connector_arch the same way), and __init__ defines
    _run_family_load only once comfy imported."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    fn = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in consts:
            ns[node.targets[0].id] = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(textwrap.dedent(ast.get_source_segment(src, node, padded=True)), ns)  # noqa: S102 — the repo's own source
            fn = ns[name]
    return fn


def _outcome(fn, *args):
    try:
        return ["ok", fn(*args)]
    except Exception as exc:  # noqa: BLE001 — every failure is data for the parent's verdict
        return ["error", type(exc).__name__, str(exc)]


def _child(job):
    import locale
    # getencoding (3.11+): getpreferredencoding itself raises EncodingWarning in the warn mode. utf8_mode is reported apart.
    enc = locale.getencoding() if hasattr(locale, "getencoding") else locale.getpreferredencoding(False)
    out = {"encoding": codecs.lookup(enc).name, "utf8_mode": sys.flags.utf8_mode}
    mod = _import_plugin(job["root"])
    for d in job["presets"]:
        out[f"manifest {d}"] = _outcome(lambda n: mod._load_model_config(n)[1]["family"], d)
    for fam in job["families"]:
        out[f"family {fam}"] = _outcome(mod._family_preset, fam)
    for rel in job["keyfiles"]:
        os.environ["QF_NATIVE_KEYFILE"] = os.path.join(job["root"], rel)
        out[f"keyfile {rel}"] = _outcome(lambda: bool(mod._read_auth()[0]))   # the key itself never leaves the child
    heads = _extract(os.path.join(job["root"], "qf_ltx_modelpatcher.py"), "_connector_config_heads",
                     {"os": os, "json": json}, consts=("_MAX_CONNECTOR_HEADS",))
    for label, model_dir in job["connectors"]:
        out[f"connectors {label}"] = (_outcome(heads, model_dir) if heads
                                      else ["error", "Missing", "no _connector_config_heads in qf_ltx_modelpatcher.py"])
    mod._CONFIGS_DIR = job["mal_configs"]
    for d in job["mal_presets"]:
        out[f"malformed-manifest {d}"] = _outcome(lambda n: mod._load_model_config(n)[1]["family"], d)
    out["malformed-family ltx2"] = _outcome(mod._family_preset, "ltx2")
    # a saved workflow naming a preset whose manifest is broken: the same refusal, through the real loader core
    run_load = _extract(os.path.join(job["root"], "__init__.py"), "_run_family_load", dict(vars(mod)))
    out["malformed-saved-workflow ltx2"] = _outcome(run_load, "ltx2", "x.safetensors", "notutf8")
    for path in job["mal_keyfiles"]:
        os.environ["QF_NATIVE_KEYFILE"] = path
        out[f"malformed-keyfile {os.path.basename(path)}"] = _outcome(lambda: bool(mod._read_auth()[0]))
    print(json.dumps(out))   # ASCII (ensure_ascii): printable under any forced stdout encoding


# ---------------------------------------------------------------------------------------------------- the parent --
def _shipped(root):
    presets, families, keyfiles, connector_dirs = [], {}, [], []
    cfg = os.path.join(root, "configs")
    for d in sorted(os.listdir(cfg)):
        mp = os.path.join(cfg, d, "qf_native.json")
        if os.path.isfile(mp):
            presets.append(d)
            with open(mp, encoding="utf-8") as f:
                families[d] = json.load(f)["family"]
        cc = os.path.join(cfg, d, "connectors", "config.json")
        if os.path.isfile(cc):
            with open(cc, encoding="utf-8") as f:
                connector_dirs.append((f"configs/{d}", json.load(f).get("video_connector_num_attention_heads")))
    for plat in sorted(os.listdir(os.path.join(root, "bin"))):
        kf = os.path.join(root, "bin", plat, "config.json")
        if os.path.isfile(kf):
            with open(kf, encoding="utf-8") as f:
                if json.load(f).get("api_key"):
                    keyfiles.append(f"bin/{plat}/config.json")
    return presets, families, keyfiles, connector_dirs


_BROKEN = {  # name -> bytes: each is malformed in its own way
    "badjson": b'{"family": "ltx2",',
    "notutf8": '{"family": "ltx2", "_note": "编码"}'.encode("gbk"),
    "notobject": b'["ltx2"]',
}
_BROKEN_HEADS = {"strheads": b'{"video_connector_num_attention_heads": "32"}',
                 "zeroheads": b'{"video_connector_num_attention_heads": 0}',
                 "boolheads": b'{"video_connector_num_attention_heads": true}'}


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _malformed(tmp):
    """The broken fixtures + what each must do. Returns (job fields, expectations)."""
    mal_configs, keys, conn = (os.path.join(tmp, n) for n in ("configs", "keys", "conn"))
    job = {"mal_configs": mal_configs, "mal_presets": [], "mal_keyfiles": [], "connectors": []}
    expect = {}
    for name, data in _BROKEN.items():
        _write(os.path.join(mal_configs, name, "qf_native.json"), data)
        job["mal_presets"].append(name)
        expect[f"malformed-manifest {name}"] = ("raises", name)
        _write(os.path.join(keys, f"{name}.json"), data)
        job["mal_keyfiles"].append(os.path.join(keys, f"{name}.json"))
        expect[f"malformed-keyfile {name}.json"] = ("raises", f"{name}.json")
        _write(os.path.join(conn, name, "connectors", "config.json"), data)
    for name, data in _BROKEN_HEADS.items():
        _write(os.path.join(conn, name, "connectors", "config.json"), data)
    for name in list(_BROKEN) + list(_BROKEN_HEADS):
        job["connectors"].append([f"conn/{name}", os.path.join(conn, name)])
        expect[f"connectors conn/{name}"] = ("raises", os.path.join(name, "connectors", "config.json"))
    _write(os.path.join(conn, "keyless", "connectors", "config.json"), b"{}")   # gated checkpoints declare no heads
    os.makedirs(os.path.join(conn, "absent"))                                    # no connectors/ at all
    for name in ("keyless", "absent"):
        job["connectors"].append([f"conn/{name}", os.path.join(conn, name)])
        expect[f"connectors conn/{name}"] = ("ok", None)
    # a family whose ONLY manifest is broken: the refusal names the broken file, not "no model config shipped"
    expect["malformed-family ltx2"] = ("raises", "manifest unreadable")
    expect["malformed-saved-workflow ltx2"] = ("raises", "manifest unreadable")
    return job, expect


def _gbk_locale(tmp):
    """(env, label) forcing a GBK preferred encoding, or (None, why)."""
    if os.name == "nt":
        return {"PYTHONUTF8": "0"}, "ansi-code-page"
    try:
        have = subprocess.run(["locale", "-a"], capture_output=True, encoding="utf-8", errors="replace").stdout.split()
    except OSError:
        have = []
    for name in have:
        if name.lower().replace("-", "") in ("zh_cn.gbk", "zh_cn.gb18030"):
            return {"LC_ALL": name, "PYTHONUTF8": "0"}, "cp936"
    locpath = os.path.join(tmp, "locale")
    os.makedirs(locpath)
    try:
        subprocess.run(["localedef", "-c", "-i", "zh_CN", "-f", "GBK", os.path.join(locpath, "zh_CN.GBK")],
                       capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"no GBK locale and localedef failed ({exc})"
    if not os.path.isdir(os.path.join(locpath, "zh_CN.GBK")):
        return None, "no GBK locale and localedef produced none"
    return {"LOCPATH": locpath, "LC_ALL": "zh_CN.GBK", "PYTHONUTF8": "0"}, "cp936"


def _modes(tmp):
    """[(label, python flags, env overrides, required preferred encoding or None)] + [(label, skip reason)]."""
    modes, skips = [], []
    env, label = _gbk_locale(tmp)
    if env is None:
        skips.append(("cp936", label))
    else:
        modes.append((label, [], env, "gbk" if label == "cp936" else "non-utf-8"))
    if os.name == "nt":
        skips.append(("c-locale", "no C locale on Windows"))
    else:
        modes.append(("c-locale", [], {"LC_ALL": "C", "PYTHONUTF8": "0"}, "ascii"))
    if sys.version_info >= (3, 10):
        modes.append(("warn", ["-X", "warn_default_encoding", "-W", "error::EncodingWarning"], {"PYTHONUTF8": "0"},
                      None))
    else:
        skips.append(("warn", "EncodingWarning needs Python 3.10+"))
    return modes, skips


def _verdict(expect, got):
    if expect[0] == "ok":
        return got[0] == "ok" and got[1] == expect[1]
    return got[0] == "error" and got[1] == "RuntimeError" and expect[1] in got[2]


def _show(got):
    s = got[1] if got[0] == "ok" else f"{got[1]}: {got[2]}"
    return str(s).encode("ascii", "backslashreplace").decode()[:150]


def main():
    presets, families, keyfiles, connector_dirs = _shipped(_ROOT)
    fails, rows, skipped = [], [], []
    sites = _static_sites(_ROOT)
    rows.append(("static", "-", "0 encoding-less text-I/O sites", f"{len(sites)} site(s)", not sites))
    for s in sites:
        print(f"  encoding-less: {s}")
    tmp = tempfile.mkdtemp(prefix="qf-textio-")
    try:
        mal_job, mal_expect = _malformed(tmp)
        job = dict(root=_ROOT, presets=presets, families=sorted(set(families.values())), keyfiles=keyfiles, **mal_job)
        expect = {f"manifest {d}": ("ok", families[d]) for d in presets}
        expect.update({f"family {f}": ("ok", next(d for d in presets if families[d] == f)) for f in job["families"]})
        expect.update({f"keyfile {k}": ("ok", True) for k in keyfiles})
        for rel, heads in connector_dirs:
            job["connectors"].append([rel, os.path.join(_ROOT, rel)])
            expect[f"connectors {rel}"] = ("ok", heads)
        expect.update(mal_expect)
        modes, skipped = _modes(tmp)
        for label, flags, overrides, need in modes:
            env = {k: v for k, v in os.environ.items() if k not in _KEY_ENVS + _LOCALE_ENVS}
            env.update(overrides)
            r = subprocess.run([sys.executable, *flags, os.path.abspath(__file__), "--child", json.dumps(job)],
                               env=env, capture_output=True, encoding="utf-8", errors="replace")
            try:
                got = json.loads(r.stdout.strip().splitlines()[-1])
            except (IndexError, ValueError):
                rows.append((label, "child", "a result line", f"rc={r.returncode}: {r.stderr.strip()[-400:]}", False))
                continue
            real = need is None or (got["encoding"] != "utf-8" if need == "non-utf-8" else got["encoding"] == need)
            if not real or got["utf8_mode"]:
                skipped.append((label, f"the child ran with {got['encoding']} (utf8_mode={got['utf8_mode']})"))
                continue
            for arm, exp in expect.items():
                g = got.get(arm, ["error", "Missing", "the child did not run this arm"])
                rows.append((label, arm, f"{exp[0]} {exp[1]}", _show(g), _verdict(exp, g)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{'mode':<14} {'PASS':<5} {'arm':<44} expected -> actual")
    for label, arm, exp, act, ok in rows:
        print(f"{label:<14} {'ok' if ok else 'FAIL':<5} {arm:<44} {exp} -> {act}")
        if not ok:
            fails.append(f"{label}/{arm}")
    for label, why in skipped:
        print(f"[SKIP] mode {label}: {why}")
    print(f"TEXT_ENCODING: {len(rows) - len(fails)}/{len(rows)} arms pass; {len(skipped)} mode(s) skipped"
          + (f"; FAIL: {', '.join(fails[:12])}{' ...' if len(fails) > 12 else ''}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        _child(json.loads(sys.argv[2]))
        sys.exit(0)
    sys.exit(main())
