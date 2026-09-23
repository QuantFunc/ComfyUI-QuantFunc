"""Engine log detail for the QuantFunc loader nodes.

The engine prints only warnings and errors by default. Every family loader gets one optional
"log level" input so a user (or the test harness) can ask for more. The setting is process-wide:
it stays in force for the whole ComfyUI session until a loader run changes it.
"""

import functools
import inspect

# Choice -> engine level (the engine's scale: 2 = info, 3 = warnings and errors). No "debug" on purpose: the
# engine keeps some diagnostics at debug so ordinary users never see them; developers raise it outside the UI.
LOG_LEVELS = {"warning": 3, "info": 2}

LOG_LEVEL_INPUT = (list(LOG_LEVELS), {
    "default": "warning",
    "tooltip": "How much the QuantFunc engine prints to the console. warning (default): only "
               "warnings and errors. info: also loading and progress details. Applies to the whole "
               "ComfyUI session until changed.",
})


def _with_log_level(sig):
    """`sig` plus a keyword-only `log_level="warning"` (before any **kwargs, where Python requires it)."""
    params = list(sig.parameters.values())
    at = next((i for i, p in enumerate(params) if p.kind is inspect.Parameter.VAR_KEYWORD), len(params))
    params.insert(at, inspect.Parameter("log_level", inspect.Parameter.KEYWORD_ONLY, default="warning"))
    return sig.replace(parameters=params)


def add_log_level_input(cls, set_level):
    """Give a loader node the optional `log_level` input. Its value is handed to
    `set_level(engine_level)` before the node's own function runs. The node function keeps its
    own name, docstring and parameters (plus `log_level`) for anything that inspects it."""
    base_inputs = cls.INPUT_TYPES      # bound to cls
    run = getattr(cls, cls.FUNCTION)

    def INPUT_TYPES(_cls):
        spec = base_inputs()
        spec.setdefault("optional", {})["log_level"] = LOG_LEVEL_INPUT
        return spec

    @functools.wraps(run)
    def _run(self, *args, log_level="warning", **kwargs):
        set_level(LOG_LEVELS[log_level])
        return run(self, *args, **kwargs)

    _run.__signature__ = _with_log_level(inspect.signature(run))
    cls.INPUT_TYPES = classmethod(INPUT_TYPES)
    setattr(cls, cls.FUNCTION, _run)
    return cls
