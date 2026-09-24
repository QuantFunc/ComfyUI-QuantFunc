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

CONSOLE ARMS (the plugin's own OUTPUT must never raise on the console's code page). ComfyUI keeps the OS encoding on
stdout/stderr with errors='strict' (its LogInterceptor), so on a redirected Windows console one character the code page
cannot hold raises UnicodeEncodeError out of the print() / log call, and out of the loader:
  static      no print() outside qf_engine.say, no root-logger call, no bare getLogger where qfe.logger() exists, and
              no non-ASCII on any line a traceback can print (a code line: its literals and its trailing comment).
              ComfyUI logs an uncaught node exception and its traceback; on cp932/cp949 one unencodable character
              there makes that logging raise
  runtime     per code page cp932 / cp949 / cp1252 / cp936 (PYTHONIOENCODING, streams wrapped like ComfyUI's): importing
              the plugin, qf_engine.say / info / logger, and the LTX audio-fix warning never raise and never lose a line
              ("--- Logging error ---"); every plugin logger carries the console-safe filter; the bytes on stdout/stderr
              are valid in the code page, with what it cannot hold backslash-escaped; qf_lora_convert --help runs;
              an ENGINE error (last_err) raised through the plugin's real create_pipeline and logged the way
              ComfyUI's execution.py logs it (the message, then the traceback) never raises a second exception;
              an error naming a user path (a Chinese username, chained from the OSError) raised through a node
              FUNCTION, its VALIDATE_INPUTS, a denoise method, a property, a constructor and the ComfyUI sampler the
              LTX audio fix wraps never raises a second exception either, keeps its type and its original text
              (qf_console_original), and is unchanged where the console holds it (cp936, UTF-8); one whose text
              comes from its __str__ becomes a RuntimeError carrying that text escaped; the loaders' VALIDATE_INPUTS
              message, logged the way execution.py logs it, is never lost. An OOM or a user cancel raised while
              handling (or from) an exception whose text cannot be rewritten keeps its type, which is how
              execution.py tells them apart, and the chained text stays in the traceback, escaped; so does the chain
              of a replaced top.
  boundary    static: every class ComfyUI calls into (its base named by module path or by a name imported from comfy)
              carries @qfe.console_safe_methods, every registered node's entry points are wrapped by
              qfe.console_safe_nodes after the last registration, and no wrapped method is async or a generator.
              A planted tree proves the check finds each kind of escape.

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
    """`a.b.c` for a call's target. A chain rooted in an expression (`Path(x).open`) gets a `?` root, so it is never
    mistaken for the builtin of the same name."""
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    parts.append(func.id if isinstance(func, ast.Name) else "?")
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


_LOG_LEVELS = {"debug", "info", "warning", "error", "exception", "critical", "log"}


def _console_sites(root):
    """Console-output paths in the plugin's RUNTIME modules (the .py files at its root) that can raise on a code page:
    a print() outside qf_engine.say; a call on the ROOT logger (logging.<level>, whatever `logging` is imported as);
    logging.getLogger() in a module that has qf_engine (qfe.logger() gives the console-safe one there)."""
    sites = []
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(root, fn)
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), path)
        aliases = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                   for a in n.names if a.name == "logging"}
        has_qfe = any(isinstance(n, ast.ImportFrom) and any(a.name == "qf_engine" for a in n.names)
                      for n in ast.walk(tree))
        say_body, fallback = set(), set()
        for n in ast.walk(tree):
            if fn == "qf_engine.py" and isinstance(n, ast.FunctionDef) and n.name == "say":
                say_body = {id(x) for x in ast.walk(n)}
            # the handler of a failed `from . import qf_engine`: no qfe.logger() exists there, by construction
            if isinstance(n, ast.Try) and any(isinstance(b, ast.ImportFrom) and any(a.name == "qf_engine" for a in b.names)
                                              for b in n.body):
                fallback |= {id(x) for h in n.handlers for x in ast.walk(h)}
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if isinstance(f, ast.Name) and f.id == "print" and id(n) not in say_body:
                sites.append(f"{fn}:{n.lineno} print(...)")
            elif (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in aliases
                  and f.attr in _LOG_LEVELS):
                sites.append(f"{fn}:{n.lineno} {f.value.id}.{f.attr}(...) on the root logger")
            elif (has_qfe and isinstance(f, ast.Attribute) and f.attr == "getLogger" and isinstance(f.value, ast.Name)
                  and f.value.id in aliases and id(n) not in fallback):
                sites.append(f"{fn}:{n.lineno} logging.getLogger(...) where qfe.logger(...) exists")
        sites += [f"{fn}:{row} non-ASCII on a traceback-visible line ({', '.join(f'U+{ord(c):04X}' for c in chars)})"
                  for row, chars in _non_ascii_code_lines(tree, path)]
    return sites


def _non_ascii_code_lines(tree, path):
    """(line, chars) for each CODE line (outside a docstring, not comment-only) carrying a non-ASCII character: its
    string literals and its trailing comment alike. ComfyUI logs an uncaught node exception AND its traceback, which
    prints the source line of every frame (execution.py: logging.error(traceback.format_exc())). On a code page that
    lacks one character there, logging's handleError re-prints the exception chain to the same strict stream and raises
    (cp932/cp949, measured), so every line a traceback can print stays ASCII (#738). Comment-only lines and docstrings
    never appear in a traceback."""
    import io
    import tokenize
    doc = set()
    for n in ast.walk(tree):
        if (isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
                and isinstance(n.body[0].value.value, str)):
            doc.update(range(n.body[0].lineno, n.body[0].end_lineno + 1))
    with open(path, encoding="utf-8") as f:
        src = f.read()
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    code = set()
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type not in skip:
            code.update(range(t.start[0], t.end[0] + 1))
    lines = src.split("\n")
    return [(ln, sorted({c for c in lines[ln - 1] if ord(c) > 127})) for ln in sorted(code - doc)
            if any(ord(c) > 127 for c in lines[ln - 1])]


# ComfyUI's entry points on a node class besides its FUNCTION (execution.py / server.py call each one when present).
_COMFY_NODE_ENTRY_POINTS = ("INPUT_TYPES", "VALIDATE_INPUTS", "IS_CHANGED", "check_lazy_status")


def _suspends(f):
    """Whether the ast function `f` is async or a generator: its body runs after the call returned, where a call-time
    wrapper (console_safe_errors) no longer catches what it raises."""
    if isinstance(f, ast.AsyncFunctionDef):
        return True
    stack = list(f.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.Yield, ast.YieldFrom)):
            return True
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(n))
    return False


def _boundary_sites(root):
    """Where an exception leaves the plugin for ComfyUI, it must pass the console-safe boundary (#738): every class
    ComfyUI calls into (a comfy base, named by module path or by a name imported from comfy; a plugin subclass of one;
    a mixin combined with one) carries @qfe.console_safe_methods; __init__ wraps every registered node's entry points
    with qfe.console_safe_nodes(NODE_CLASS_MAPPINGS) after the last registration, so a new node cannot skip it; and no
    wrapped method is async or a generator (it would raise past the wrapper)."""
    classes, sites, trees, comfy = {}, [], {}, {}
    for fn in sorted(os.listdir(root)):
        if fn.endswith(".py"):
            with open(os.path.join(root, fn), encoding="utf-8") as f:
                trees[fn] = ast.parse(f.read())
            comfy[fn] = {"comfy"}   # the names this file binds to a comfy module or to a name imported from one
            for n in ast.walk(trees[fn]):
                if isinstance(n, ast.ClassDef):
                    classes[n.name] = (fn, n, [ast.unparse(b) for b in n.bases])
                elif isinstance(n, ast.ImportFrom) and not n.level and (n.module or "").split(".")[0] == "comfy":
                    comfy[fn] |= {a.asname or a.name for a in n.names}
                elif isinstance(n, ast.Import):
                    comfy[fn] |= {a.asname for a in n.names if a.asname and a.name.split(".")[0] == "comfy"}
    facing = {c for c, (fn, _, bases) in classes.items() if any(b.split(".")[0] in comfy[fn] for b in bases)}
    grown = True
    while grown:   # plugin subclasses of a comfy-facing class, and the mixins combined with one
        grown = False
        for c, (_, _, bases) in classes.items():
            names = {b.rsplit(".", 1)[-1] for b in bases}
            add = ({c} if names & facing else set()) | ({b for b in names if b in classes} if c in facing else set())
            if add - facing:
                facing |= add
                grown = True
    for c in sorted(facing):
        fn, node, _ = classes[c]
        if not any(ast.unparse(d).endswith("console_safe_methods") for d in node.decorator_list):
            sites.append(f"{fn}:{node.lineno} class {c}: ComfyUI calls into it, but it lacks @qfe.console_safe_methods")
    for c, (fn, node, _) in sorted(classes.items()):
        wrapped = {s.value.value for s in node.body if isinstance(s, ast.Assign) and isinstance(s.value, ast.Constant)
                   and any(getattr(t, "id", None) == "FUNCTION" for t in s.targets)}
        wrapped |= set(_COMFY_NODE_ENTRY_POINTS) if wrapped else set()
        for m in node.body:
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and _suspends(m) and (m.name in wrapped or (
                    c in facing and (m.name == "__init__" or not (m.name.startswith("__") and m.name.endswith("__"))))):
                sites.append(f"{fn}:{m.lineno} {c}.{m.name}: async or a generator, it raises past console_safe_errors")
    init = trees.get("__init__.py")
    wraps = [n.lineno for n in ast.walk(init) if isinstance(n, ast.Call)
             and ast.unparse(n.func).endswith("console_safe_nodes") and "NODE_CLASS_MAPPINGS" in ast.unparse(n)]
    if not wraps:
        sites.append("__init__.py: no qfe.console_safe_nodes(NODE_CLASS_MAPPINGS): the node FUNCTIONs are unwrapped")
    else:
        for n in ast.walk(init):
            registers = ((isinstance(n, (ast.Assign, ast.AugAssign)) and "NODE_CLASS_MAPPINGS" in ast.unparse(
                n.targets[0] if isinstance(n, ast.Assign) else n.target)) or (
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("update", "setdefault")
                and "NODE_CLASS_MAPPINGS" in ast.unparse(n.func.value)))
            if registers and n.lineno > max(wraps):
                sites.append(f"__init__.py:{n.lineno} registers a node after console_safe_nodes: it would skip it")
    return sites


def _introspection_arm(root):
    """The boundary is invisible to introspection: ComfyUI passes VALIDATE_INPUTS only the inputs that
    inspect.getfullargspec names, and skips its own list check for those (execution.py:891-893, 1082-1086), and
    getfullargspec ignores __wrapped__. So every entry point console_safe_nodes / console_safe_methods wraps keeps
    its argspec."""
    import importlib.util
    import inspect
    spec = importlib.util.spec_from_file_location("qf_textio_engine", os.path.join(root, "qf_engine.py"))
    qfe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qfe)
    row = ("static", "boundary introspection", "getfullargspec of every wrapped entry point unchanged")
    if not (hasattr(qfe, "console_safe_nodes") and hasattr(qfe, "console_safe_methods")):
        return row + ("no console_safe_nodes / console_safe_methods", False)

    class Node:
        FUNCTION = "load"

        @classmethod
        def INPUT_TYPES(cls):
            return {}

        @classmethod
        def VALIDATE_INPUTS(cls, quality=None):
            return True

        @classmethod
        def IS_CHANGED(cls, model, quality=None):
            return ""

        def load(self, model, quality=None):
            return (model,)

    class Model:
        def __init__(self, model, device=None):
            self.model = model

        def apply_model(self, x, t, **kwargs):
            return x

    def specs():
        return {n: inspect.getfullargspec(getattr(c, a)) for n, c, a in (
            ("INPUT_TYPES", Node, "INPUT_TYPES"), ("VALIDATE_INPUTS", Node, "VALIDATE_INPUTS"),
            ("IS_CHANGED", Node, "IS_CHANGED"), ("FUNCTION", Node, "load"), ("__init__", Model, "__init__"),
            ("method", Model, "apply_model"))}
    before = specs()
    qfe.console_safe_nodes({"QuantFuncTextioNode": Node})
    qfe.console_safe_methods(Model)
    changed = [n for n, s in specs().items() if s != before[n]]
    return row + (f"changed: {', '.join(changed)}" if changed else "unchanged", not changed)


def _boundary_selftest():
    """_boundary_sites on a planted tree finds exactly its three escapes: a class on a comfy base imported under an
    alias, without the decorator; an async method of a decorated class on a comfy module alias; a generator FUNCTION."""
    planted = {
        "facing.py": "from comfy.model_base import BaseModel as Aliased\nimport comfy.model_patcher as mp\n\n\n"
                     "class Plain(Aliased):\n    pass\n\n\n@qfe.console_safe_methods\nclass Patcher(mp.ModelPatcher):\n"
                     "    async def load(self):\n        pass\n",
        "__init__.py": "class Node:\n    FUNCTION = 'run'\n\n    def run(self):\n        yield 1\n\n\n"
                       "NODE_CLASS_MAPPINGS = {'Node': Node}\nqfe.console_safe_nodes(NODE_CLASS_MAPPINGS)\n",
    }
    with tempfile.TemporaryDirectory(prefix="qf-boundary-") as d:
        for name, src in planted.items():
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(src)
        got = _boundary_sites(d)
    want = ("class Plain:", "Patcher.load:", "Node.run:")
    found = [w for w in want if any(w in s for s in got)]
    return ("static", "boundary selftest", "the 3 planted escapes, nothing else",
            f"{len(got)} site(s), planted found: {', '.join(found) or 'none'}", len(got) == 3 and found == list(want))


# The plugin's own non-ASCII punctuation (— – →), CJK, Hangul, Latin-1 and an emoji: no single code page holds them all.
_CONSOLE_TEXT = "— – → 中文 한국어 日本語 éü \U0001f600"
_CONSOLE_PAGES = ("cp932", "cp949", "cp1252", "cp936", "utf-8")   # redirected Windows consoles, and a UTF-8 one
# A Chinese username + directory in a path: GBK holds it, cp932/cp949/cp1252 do not (English Windows, Chinese user).
_CONSOLE_PATH = "C:\\Users\\\u5f20\u4e09\\\u4e2d\u6587\u76ee\u5f55\\model.safetensors"
_BOUNDARY_MSG = "qf_native: cannot stage the weights: " + _CONSOLE_PATH
_LOG_ERROR = b"--- Logging error ---"
_LOST = []   # the records the console child's logging could not print (its handler's handleError): each a lost line
_LOST_ROW = ["error", "LoggingError", "logging lost a line: '--- Logging error ---'"]


def _console_child(job):
    """Runs under PYTHONIOENCODING=<code page>. Wraps stdout/stderr the way ComfyUI's LogInterceptor does (the stream's
    own encoding, errors='strict') and logs to stderr through a root StreamHandler like ComfyUI's, then drives the
    plugin's output paths. Results go to a file: stdout/stderr ARE the thing under test."""
    import io
    import logging
    for name in ("stdout", "stderr"):
        s = getattr(sys, name)
        setattr(sys, name, io.TextIOWrapper(s.buffer, encoding=s.encoding, line_buffering=True))
    class Handler(logging.StreamHandler):   # ComfyUI's root handler, noting each line logging could not print
        def handleError(self, record):
            _LOST.append(record)
            super().handleError(record)
    handler = Handler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)

    def fail():   # a plugin error naming a user path, chained from the OSError that named it first
        try:
            raise OSError(2, "No such file or directory", _CONSOLE_PATH)
        except OSError as e:
            raise RuntimeError(_BOUNDARY_MSG) from e

    def opaque():
        raise _Opaque()
    import types
    sampling = types.ModuleType("comfy.k_diffusion.sampling")   # a ComfyUI sampler, for the audio fix to wrap

    def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0,
                               noise_sampler=None):
        fail()
    sampling.sample_euler_ancestral = sample_euler_ancestral
    sys.modules.update({"comfy.k_diffusion": types.ModuleType("comfy.k_diffusion"),
                        "comfy.k_diffusion.sampling": sampling})
    out = {}
    try:
        mod = _import_plugin(job["root"])   # the guarded comfy import fails here (torch blocked): its warning prints
        out["console import"] = ["ok", None]
    except BaseException as exc:  # noqa: BLE001 — a crashed import is the result
        out["console import"] = ["error", type(exc).__name__, str(exc)]
        mod = None
    if job.get("mode") == "broken-engine":   # qf_engine itself raises on import: the plugin must still degrade
        degraded = mod is not None and mod.qfe is None and not mod._IMPORT_OK and not mod.NODE_CLASS_MAPPINGS
        out["console broken qf_engine"] = (["ok", None] if degraded else
                                           out["console import"] if mod is None else ["error", "NotDegraded", "-"])
        with open(job["result"], "w", encoding="utf-8") as f:
            json.dump(out, f)
        return
    qfe = getattr(mod, "qfe", None)
    missing = ["error", "Missing", "qf_engine has no console-safe output"]
    if qfe is not None and hasattr(qfe, "say") and hasattr(qfe, "logger"):
        out["console say"] = _outcome(qfe.say, "[say] " + _CONSOLE_TEXT)
        qfe._LOG_LEVEL = qfe._LOG_INFO
        out["console info"] = _outcome(qfe.info, "[info] " + _CONSOLE_TEXT)
        out["console logger"] = _outcome(qfe.logger("qf_textio").warning, "%s", "[log] " + _CONSOLE_TEXT)

        def logged_exception(log):   # a traceback (exc_info) carrying text the code page cannot hold
            try:
                raise RuntimeError("[exc] " + _CONSOLE_TEXT)
            except RuntimeError:
                log.exception("[exception] logged")
        out["console exception"] = _outcome(logged_exception, qfe.logger("qf_textio"))
    elif qfe is not None:   # the unfixed plugin: its info() line is the output path that exists
        qfe._LOG_LEVEL = qfe._LOG_INFO
        out["console say"] = out["console logger"] = out["console exception"] = missing
        out["console info"] = _outcome(qfe.info, "[info] " + _CONSOLE_TEXT)
    if qfe is not None and hasattr(qfe, "create_pipeline"):
        import types
        engine = types.SimpleNamespace(   # an engine whose create fails, with an error text the code page may not hold
            quantfunc_create=lambda params, handle: 3,
            quantfunc_last_error=lambda: ("[engine] " + _CONSOLE_TEXT).encode("utf-8"))

        def comfy_logs_an_engine_error():   # ComfyUI execution.py:637-638, around the plugin's real raise of engine text
            import traceback
            try:
                qfe.create_pipeline(engine, create_params=qfe.InitParams())
            except Exception as ex:  # noqa: BLE001
                logging.getLogger().error(f"!!! Exception during processing !!! {ex}")
                logging.getLogger().error(traceback.format_exc())
        out["console engine error"] = _outcome(comfy_logs_an_engine_error)
    arms = ("console node error", "console validate error", "console denoise error", "console init error",
            "console property error")
    if qfe is not None and hasattr(qfe, "console_safe_nodes") and hasattr(qfe, "console_safe_methods"):
        class Node:   # stands in for a QuantFunc node class, wrapped the way __init__ wraps every registered node
            FUNCTION = "load"

            @classmethod
            def VALIDATE_INPUTS(cls, quality=None):
                fail()

            def load(self):
                fail()

        class OpaqueNode:
            FUNCTION = "load"

            def load(self):
                opaque()
        qfe.console_safe_nodes({"QuantFuncTextioNode": Node, "QuantFuncTextioOpaque": OpaqueNode})

        @qfe.console_safe_methods
        class Model:   # stands in for a class ComfyUI calls into while it samples
            def _apply_model(self):
                fail()

            @property
            def patcher(self):
                fail()

        @qfe.console_safe_methods
        class Patcher:   # stands in for a class ComfyUI constructs (ModelPatcher.clone calls self.__class__)
            def __init__(self):
                fail()
        calls = (lambda: Node().load(), lambda: Node.VALIDATE_INPUTS(), lambda: Model()._apply_model(), Patcher,
                 lambda: Model().patcher)
        out.update({arm: _boundary_outcome(qfe, call) for arm, call in zip(arms, calls)})
        out["console fallback error"] = _fallback_outcome(qfe, lambda: OpaqueNode().load())
    else:   # no boundary: the same error reaches ComfyUI's logging as it is
        out.update({arm: _boundary_outcome(qfe, fail) for arm in arms})
        out["console fallback error"] = _fallback_outcome(qfe, opaque)
    import subprocess
    process_error = subprocess.CalledProcessError(1, ["nvidia-smi", _CONSOLE_PATH])   # its text is computed from cmd
    key_error = KeyError((_CONSOLE_PATH,))   # its text reprs a non-str key: neither can be rewritten in place

    def oom_while_handling():   # an OOM raised while handling one of them (the chain's context)
        try:
            raise process_error
        except subprocess.CalledProcessError:
            raise _OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    def cancel_from():   # a user cancel raised from one (cause and context both point at it)
        try:
            raise key_error
        except KeyError as e:
            raise _InterruptProcessingException() from e

    opaque = _Opaque(_CONSOLE_PATH)   # a str arg its __str__ ignores: rewriting it cannot help, so it must stay as is

    def opaque_while_handling():   # the top's own text cannot be rewritten, and it has a chain
        try:
            raise OSError(2, "No such file or directory", _CONSOLE_PATH)
        except OSError:
            raise opaque
    safe = getattr(qfe, "console_safe", lambda t: t)
    chains = [("console oom keeps type", oom_while_handling, _OutOfMemoryError, safe(str(process_error))),
              ("console cancel keeps type", cancel_from, _InterruptProcessingException, safe(str(key_error))),
              ("console fallback keeps chain", opaque_while_handling,
               _Opaque if safe(_BOUNDARY_MSG) == _BOUNDARY_MSG else RuntimeError, "No such file or directory")]
    if hasattr(qfe, "console_safe_errors"):   # the boundary every node FUNCTION and comfy-facing method runs under
        chains = [(arm, qfe.console_safe_errors(f), want, text) for arm, f, want, text in chains]
    out.update({arm: _chain_outcome(qfe, f, want, text) for arm, f, want, text in chains})
    if out["console fallback keeps chain"][0] == "ok" and opaque.args != (_CONSOLE_PATH,):   # the stand-in carries it
        out["console fallback keeps chain"] = ["error", "Mutated", f"the replaced exception now has {opaque.args!r}"]
    validate = _extract(os.path.join(job["root"], "__init__.py"), "_validate_quality", {"qfe": qfe},
                        consts=("_QUALITY_FAST_OPTIONS",))

    def comfy_logs_a_validation_failure():   # ComfyUI execution.py:1098 + :1227 around the loaders' VALIDATE_INPUTS
        logging.getLogger().error(f"  - Custom validation failed for node: quality - {validate(_CONSOLE_TEXT)}")
    out["console validate message"] = (_outcome(comfy_logs_a_validation_failure) if validate
                                       else ["error", "Missing", "no _validate_quality in __init__.py"])
    afix = sys.modules.get("qfn_textio_pkg.qf_ltx_ancestral_audio_fix")
    out["console audio-fix"] = (_outcome(afix._log_once, "sampler " + _CONSOLE_TEXT, "euler") if afix
                                else ["error", "Missing", "the audio-fix module was not imported"])
    if afix is not None and getattr(sampling.sample_euler_ancestral, "_qf_av_wrapped", False):
        afix._resolve_qf_av = lambda model: model   # every model stands in for a QF LTX-2.5 AV one
        out["console sampler error"] = _boundary_outcome(
            qfe, lambda: sampling.sample_euler_ancestral(object(), None, None))
    else:
        out["console sampler error"] = ["error", "Missing", "the audio fix did not wrap ComfyUI's sampler"]
    import logging as _lg
    bare = sorted(n for n, lg in _lg.Logger.manager.loggerDict.items()
                  if n.startswith("qfn_textio_pkg") and isinstance(lg, _lg.Logger)
                  and not any(type(f).__name__ == "_ConsoleSafe" for f in lg.filters))
    out["console loggers"] = ["ok", None] if not bare else ["error", "Unsafe", ", ".join(bare)]
    with open(job["result"], "w", encoding="utf-8") as f:
        json.dump(out, f)


def _broken_engine_copy(root, tmp):
    """The plugin's runtime modules, with a qf_engine.py that raises on import (once per run)."""
    dst = os.path.join(tmp, "broken-engine")
    if not os.path.isdir(dst):
        os.makedirs(dst)
        for fn in os.listdir(root):
            if fn.endswith(".py"):
                shutil.copy(os.path.join(root, fn), dst)
        path = os.path.join(dst, "qf_engine.py")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        with open(path, "w", encoding="utf-8") as f:
            f.write('raise RuntimeError("qf_engine is broken \\u2014 on purpose")\n' + src)
    return dst


class _Opaque(Exception):
    """An exception whose text comes from other state (its __str__), not its args: no rewrite in place reaches it."""

    def __str__(self):
        return _BOUNDARY_MSG


class _OutOfMemoryError(RuntimeError):
    """Stands in for torch's OutOfMemoryError: ComfyUI's is_oom() is isinstance(e, OOM_EXCEPTION)."""


class _InterruptProcessingException(Exception):
    """Stands in for comfy.model_management's: execution.py tells a user cancel from an error by its type."""


def _chain_outcome(qfe, call, want, chained):
    """ComfyUI's logging around one plugin call whose exception has a chain: logged without a second exception or a lost
    line; the exception ComfyUI sees is of type `want` (execution.py tells an OOM and a user cancel by type); and the
    chained exception's text (`chained`) is still in its traceback, escaped where the console cannot hold it."""
    import traceback
    caught, err = _comfy_logs(call)
    if err:
        return err
    if type(caught) is not want:
        return ["error", "TypeLost", f"ComfyUI sees {type(caught).__name__}, not {want.__name__}"]
    text = "".join(traceback.format_exception(type(caught), caught, caught.__traceback__))
    if chained not in text:
        return ["error", "ChainLost", f"the traceback lacks the chained text {chained[:60]!r}"]
    return ["ok", f"ComfyUI sees {want.__name__}, chain logged"]


def _comfy_logs(call):
    """ComfyUI's execution.py:637-638 around one plugin call: (the exception it logged, None), or (None, an error row)
    when the call did not raise or the logging raised a second exception."""
    import logging
    import traceback
    lost = len(_LOST)
    try:
        try:
            call()
        except Exception as ex:  # noqa: BLE001
            logging.getLogger().error(f"!!! Exception during processing !!! {ex}")
            logging.getLogger().error(traceback.format_exc())
            caught = ex
        else:
            return None, ["error", "NoRaise", "the call did not raise"]
    except Exception as second:  # noqa: BLE001 - the secondary exception this guards against
        return None, ["error", type(second).__name__, str(second)]
    return (caught, None) if len(_LOST) == lost else (None, _LOST_ROW)


def _fallback_outcome(qfe, call):
    """For an _Opaque exception: logged without a second exception; where the console holds its text it passes as it
    is, elsewhere it becomes a RuntimeError carrying that text escaped, the original object kept
    (qf_console_original)."""
    caught, err = _comfy_logs(call)
    if err:
        return err
    safe = qfe.console_safe(_BOUNDARY_MSG) if hasattr(qfe, "console_safe") else _BOUNDARY_MSG
    if safe == _BOUNDARY_MSG and type(caught) is _Opaque:
        return ["ok", "unchanged"]
    if (type(caught) is RuntimeError and safe in str(caught)
            and type(getattr(caught, "qf_console_original", None)) is _Opaque):
        return ["ok", "replaced: its text escaped, the original kept"]
    return ["error", "Lost", f"type={type(caught).__name__} now={str(caught)[:60]!r}"]


def _boundary_outcome(qfe, call):
    """ComfyUI's execution.py:637-638 around one plugin call. The logging must not raise a second exception; the
    exception keeps its type, its original message is recoverable, and it is unchanged where the console holds it."""
    caught, err = _comfy_logs(call)
    if err:
        return err
    fits = qfe.console_safe(_BOUNDARY_MSG) == _BOUNDARY_MSG if hasattr(qfe, "console_safe") else True
    original = getattr(caught, "qf_console_original", str(caught))
    kept = type(caught) is RuntimeError and original == _BOUNDARY_MSG and isinstance(caught.__cause__, OSError)
    if kept and (str(caught) == _BOUNDARY_MSG or not fits):
        return ["ok", "unchanged" if fits else "escaped; type and original recoverable"]
    return ["error", "Lost", f"type={type(caught).__name__} original={original[:50]!r} now={str(caught)[:50]!r}"]


def _console_arms(root, tmp):
    """rows for every code page: each output path, and the bytes the child wrote to stdout/stderr."""
    rows = []
    for cp in _CONSOLE_PAGES:
        env = {k: v for k, v in os.environ.items() if k not in _KEY_ENVS + _LOCALE_ENVS}
        env.update(PYTHONIOENCODING=cp, PYTHONUTF8="0")
        res = os.path.join(tmp, f"console-{cp}.json")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--console-child",
                            json.dumps({"root": root, "result": res})], env=env, capture_output=True)
        try:
            with open(res, encoding="utf-8") as f:
                got = json.load(f)
        except (OSError, ValueError):
            tail = r.stderr.decode("ascii", "backslashreplace")[-400:]
            rows.append((cp, "console child", "a result", f"rc={r.returncode}: {tail}", False))
            continue
        want = {"console say": ("stdout", "[say] "), "console info": ("stdout", "[info] "),
                "console logger": ("stderr", "[log] "), "console exception": ("stderr", "[exc] "),
                "console engine error": ("stderr", "[engine] ")}
        for arm in ("console import", "console say", "console info", "console logger", "console exception",
                    "console engine error", "console node error", "console validate error", "console denoise error",
                    "console init error", "console property error", "console fallback error",
                    "console oom keeps type", "console cancel keeps type", "console fallback keeps chain",
                    "console validate message", "console sampler error", "console audio-fix", "console loggers"):
            g = got.get(arm, ["error", "Missing", "the child did not run this arm"])
            ok, act = g[0] == "ok", _show(g)
            if ok and arm in want:   # the line reached the console, with what the code page cannot hold escaped
                stream, tag = want[arm]
                line = (tag + _CONSOLE_TEXT).encode(cp, "backslashreplace")
                ok = line in (r.stdout if stream == "stdout" else r.stderr)
                act = f"written as {line[:60]!r}..." if ok else f"not found on {stream}"
            rows.append((cp, arm, "ok, never raises", act, ok))
        for stream in ("stdout", "stderr"):
            data = getattr(r, stream)
            try:
                data.decode(cp)
                clean = _LOG_ERROR not in data
                act = "decodes, no logging error" if clean else "a '--- Logging error ---' (a message was lost)"
            except UnicodeDecodeError as exc:
                clean, act = False, f"undecodable: {exc}"
            rows.append((cp, f"console {stream} bytes", f"valid {cp}", act, clean))
        res = os.path.join(tmp, f"console-broken-{cp}.json")
        b = subprocess.run([sys.executable, os.path.abspath(__file__), "--console-child",
                            json.dumps({"root": _broken_engine_copy(root, tmp), "result": res, "mode": "broken-engine"})],
                           env=env, capture_output=True)
        try:
            with open(res, encoding="utf-8") as f:
                g = json.load(f)["console broken qf_engine"]
        except (OSError, ValueError, KeyError):
            g = ["error", "Crashed", f"rc={b.returncode}: {b.stderr.decode('ascii', 'backslashreplace')[-200:]}"]
        warned = b"[qf_native] disabled - qf_engine failed to import" in b.stderr and _LOG_ERROR not in b.stderr
        rows.append((cp, "console broken qf_engine", "degrades: zero nodes + a warning",
                     _show(g) if g[0] != "ok" else ("ok" if warned else "no warning line"), g[0] == "ok" and warned))
        h = subprocess.run([sys.executable, os.path.join(root, "scripts", "qf_lora_convert.py"), "--help"],
                           env=env, capture_output=True)
        help_ok = h.returncode == 0 and b"--model" in h.stdout
        rows.append((cp, "console qf_lora_convert --help", "rc 0 + the help text",
                     "ok" if help_ok else f"rc={h.returncode}: {h.stderr.decode('ascii', 'backslashreplace')[-200:]}",
                     help_ok))
    return rows


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
    lost = len(_LOST)
    try:
        got = fn(*args)
    except Exception as exc:  # noqa: BLE001 — every failure is data for the parent's verdict
        return ["error", type(exc).__name__, str(exc)]
    return ["ok", got] if len(_LOST) == lost else _LOST_ROW


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
        os.environ[mod.qfe._ENV_KEYFILE_OVERRIDE] = os.path.join(job["root"], rel)
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
        os.environ[mod.qfe._ENV_KEYFILE_OVERRIDE] = path
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
        expect[f"malformed-manifest {name}"] = ("raises", (name, "reinstall"))   # names it and says how to fix it
        _write(os.path.join(keys, f"{name}.json"), data)
        job["mal_keyfiles"].append(os.path.join(keys, f"{name}.json"))
        expect[f"malformed-keyfile {name}.json"] = ("raises", (f"{name}.json", "QUANTFUNC_API_KEY", "reinstall"))
        _write(os.path.join(conn, name, "connectors", "config.json"), data)
    for name, data in _BROKEN_HEADS.items():
        _write(os.path.join(conn, name, "connectors", "config.json"), data)
    for name in list(_BROKEN) + list(_BROKEN_HEADS):
        job["connectors"].append([f"conn/{name}", os.path.join(conn, name)])
        expect[f"connectors conn/{name}"] = ("raises", (os.path.join(name, "connectors", "config.json"), "download"))
    _write(os.path.join(conn, "keyless", "connectors", "config.json"), b"{}")   # gated checkpoints declare no heads
    os.makedirs(os.path.join(conn, "absent"))                                    # no connectors/ at all
    for name in ("keyless", "absent"):
        job["connectors"].append([f"conn/{name}", os.path.join(conn, name)])
        expect[f"connectors conn/{name}"] = ("ok", None)
    # a family whose manifest is broken: the refusal names EVERY broken manifest (it cannot tell whose each one was)
    # and the family it was resolving - never "no model config shipped"
    expect["malformed-family ltx2"] = ("raises", ("ltx2", "'badjson'", "'notutf8'", "'notobject'", "reinstall"))
    expect["malformed-saved-workflow ltx2"] = ("raises", ("ltx2", "'badjson'", "'notutf8'", "'notobject'",
                                                        "reinstall"))
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
    need = expect[1] if isinstance(expect[1], tuple) else (expect[1],)
    return got[0] == "error" and got[1] == "RuntimeError" and all(s in got[2] for s in need)


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
    csites = _console_sites(_ROOT)
    rows.append(("static", "console", "0 unsafe console-output sites", f"{len(csites)} site(s)", not csites))
    for s in csites:
        print(f"  console-unsafe: {s}")
    bsites = _boundary_sites(_ROOT)
    rows.append(("static", "boundary", "every node and comfy-facing class wrapped", f"{len(bsites)} site(s)", not bsites))
    for s in bsites:
        print(f"  boundary: {s}")
    rows.append(_boundary_selftest())
    rows.append(_introspection_arm(_ROOT))
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
        rows += _console_arms(_ROOT, tmp)
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
    if len(sys.argv) >= 3 and sys.argv[1] == "--console-child":
        _console_child(json.loads(sys.argv[2]))
        sys.exit(0)
    sys.exit(main())
