"""Session-level test bootstrap.

Import REAL torch (when installed) BEFORE any test module loads, so the per-file
"stub heavy optional deps" loops (`sys.modules.setdefault("torch", ModuleType(...))`
in test_ideogram4_autodetect / test_precision_config_autoload /
test_qwen_layered_autodetect, and test_video_nodes_wan_ltx's fallback stub) all
no-op against the real module instead of installing a bare stub session-wide.

Without this, the suite was COLLECTION-ORDER-FRAGILE: alphabetical order silently
SKIPPED every torch-gated test collected after the first stubbing file (a green
run that never executed the swap/rollback/concurrency centerpieces), while a
reversed order turned the same suite into real failures. On a genuinely torch-less
box this import fails harmlessly and the per-file stubs apply as designed.
"""
try:
    import torch  # noqa: F401
except Exception:  # noqa: BLE001 — torch not installed: per-file stubs take over
    pass
