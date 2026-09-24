"""Engine log detail for the QuantFunc loader nodes.

The engine prints only warnings and errors by default. Every family loader takes one HIDDEN
`log_level` input: ComfyUI never shows it (users do not choose the log level), but a prompt that
carries it (the test harness's) still reaches the loader. The setting is process-wide: it stays in
force for the whole ComfyUI session until a loader run changes it.
"""

import functools
import inspect

# Value -> engine level (the engine's scale: 2 = info, 3 = warnings and errors). No "debug" on purpose: the
# engine keeps some diagnostics at debug so ordinary users never see them; developers raise it outside ComfyUI.
LOG_LEVELS = {"warning": 3, "info": 2}


def _with_log_level(sig):
    """`sig` plus a keyword-only `log_level="warning"` (before any **kwargs, where Python requires it)."""
    params = list(sig.parameters.values())
    at = next((i for i, p in enumerate(params) if p.kind is inspect.Parameter.VAR_KEYWORD), len(params))
    params.insert(at, inspect.Parameter("log_level", inspect.Parameter.KEYWORD_ONLY, default="warning"))
    return sig.replace(parameters=params)


def add_log_level_input(cls, set_level):
    """Give a loader node the hidden `log_level` input. Its value is handed to
    `set_level(engine_level)` before the node's own function runs; a prompt without it gets warning.
    The node function keeps its own name, docstring and parameters (plus `log_level`) for anything
    that inspects it."""
    base_inputs = cls.INPUT_TYPES      # bound to cls
    run = getattr(cls, cls.FUNCTION)

    def INPUT_TYPES(_cls):
        spec = dict(base_inputs())
        # Hidden, in ComfyUI's (type, options) input form; new dicts, so the node's own spec is never modified
        # (the node may already declare hidden inputs of its own).
        spec["hidden"] = {**spec.get("hidden", {}), "log_level": ("STRING", {})}
        return spec

    @functools.wraps(run)
    def _run(self, *args, log_level="warning", **kwargs):
        # ComfyUI does not validate hidden inputs, so the value is checked here.
        level = LOG_LEVELS.get(log_level) if isinstance(log_level, str) else None
        if level is None:
            raise ValueError(f"log_level must be one of {list(LOG_LEVELS)}, got {log_level!r}")
        set_level(level)
        return run(self, *args, **kwargs)

    _run.__signature__ = _with_log_level(inspect.signature(run))
    cls.INPUT_TYPES = classmethod(INPUT_TYPES)
    setattr(cls, cls.FUNCTION, _run)
    return cls
