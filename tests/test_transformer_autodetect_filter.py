"""Tests for the transformer [auto-detect] default + GPU-SM compatibility filter.

The transformer dropdown must (a) default to '[auto-detect]', which resolves to
the HIGHEST-tier weight the target GPU can run, and (b) HIDE weights whose
minimum SM exceeds the target GPU (a 50x FP4 weight must not be offered on an
SM86 card — it __trap()s at runtime). On a mixed-SM multi-GPU box with no
pinned device the filter/auto-detect key on the MIN SM (safest); a pinned
device uses that device's SM.

Tier tokens (confirmed from real weight names — klein-9b-50x-lighting.safetensors,
qwen 50x-above / 30x-below — and the engine precision rules):
  50x / 50x-above  → FP4              → min SM 120
  40x              → INT4 + FP8       → min SM 89
  30x-below / 50x-below → INT4/INT8   → min SM 75 (BF16 fallback on Turing)

Run:  python3 tests/test_transformer_autodetect_filter.py     (also pytest-compatible)
"""
import os
import sys
import types
import importlib

_PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PLUGIN)
_PKG = os.path.basename(_PLUGIN)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
for _n in ("comfy", "torch", "folder_paths", "comfy.model_management"):
    sys.modules.setdefault(_n, types.ModuleType(_n))

mal = importlib.import_module(f"{_PKG}.model_auto_loader")

KLEIN = "QuantFunc/Klein-4B-Series"
QWEN = "QuantFunc/Qwen-Image-Series"

KLEIN_50X = "klein-4b-50x-lighting.safetensors"      # FP4  → SM120
KLEIN_40X = "klein-4b-40x-lighting.safetensors"      # FP8  → SM89
KLEIN_30X = "klein-4b-30x-below-lighting.safetensors"  # INT4 → SM75
QWEN_50X = "qwen-image-50x-above.safetensors"        # FP4  → SM120
QWEN_30X = "qwen-image-30x-below.safetensors"        # INT4 → SM75


class _Env:
    """Install test doubles for cache + GPU enumeration; restore on exit."""

    def __init__(self, cache=None, all_sms=None, local=None):
        self.cache = cache or {}
        self.all_sms = all_sms if all_sms is not None else []
        self.local = local or {}            # {(short, resource_type): [names]}

    def __enter__(self):
        self._orig = (mal._resource_cache, mal._all_gpu_sms,
                      mal._list_local_resource_names)
        mal._resource_cache = self.cache
        mal._all_gpu_sms = lambda: list(self.all_sms)
        mal._list_local_resource_names = \
            lambda short, rtype: list(self.local.get((short, rtype), []))
        return self

    def __exit__(self, *exc):
        (mal._resource_cache, mal._all_gpu_sms,
         mal._list_local_resource_names) = self._orig
        return False


def _klein_cache():
    return {KLEIN: {"transformer": [KLEIN_50X, KLEIN_40X, KLEIN_30X]}}


def _opt(series, name):
    return "{}/{}".format(series.split("/")[-1], name)


# ----------------------------- tier → min-SM -----------------------------
def test_min_sm_fp4_tokens():
    assert mal._transformer_min_sm(KLEIN_50X) == 120
    assert mal._transformer_min_sm(QWEN_50X) == 120           # 50x-above is FP4, not below
    assert mal._transformer_min_sm("foo-50x.safetensors") == 120


def test_min_sm_fp8_token():
    assert mal._transformer_min_sm(KLEIN_40X) == 89


def test_min_sm_int4_tokens():
    assert mal._transformer_min_sm(KLEIN_30X) == 75
    assert mal._transformer_min_sm(QWEN_30X) == 75
    # '50x-below' names the GPUs it runs BELOW → INT4 tier, NOT FP4
    assert mal._transformer_min_sm("z-image-50x-below-int4.safetensors") == 75


def test_min_sm_unknown_token_is_zero():
    # no recognized tier token → 0 → never filtered out
    assert mal._transformer_min_sm("mystery-model.safetensors") == 0
    assert mal._transformer_min_sm("") == 0


# --------------------------- auto-detect tier ---------------------------
def test_autodetect_sm120_picks_fp4():
    with _Env(cache=_klein_cache(), all_sms=[120]):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, KLEIN_50X), (s, n)


def test_autodetect_sm89_picks_fp8():
    with _Env(cache=_klein_cache(), all_sms=[89]):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, KLEIN_40X), (s, n)


def test_autodetect_sm86_picks_int4():
    with _Env(cache=_klein_cache(), all_sms=[86]):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, KLEIN_30X), (s, n)


def test_autodetect_sm75_picks_int4():
    with _Env(cache=_klein_cache(), all_sms=[75]):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, KLEIN_30X), (s, n)


def test_autodetect_unknown_gpu_picks_safest_lowest_tier():
    # SM undetectable (no GPUs) → safest = lowest tier (INT4/30x)
    with _Env(cache=_klein_cache(), all_sms=[]):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, KLEIN_30X), (s, n)


def test_autodetect_unknown_gpu_prefers_known_tier_over_unrecognized():
    # SM undetectable + catalog has an UNRECOGNIZED-token file (min-SM 0) next to
    # a real 30x (min-SM 75): the known-safe 30x wins — an unknown token means
    # "unknown compatibility", not "runs everywhere", so it must NOT be preferred.
    cache = {KLEIN: {"transformer": ["klein-4b-mystery.safetensors", KLEIN_30X]}}
    with _Env(cache=cache, all_sms=[]):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, KLEIN_30X), (s, n)


def test_autodetect_unknown_gpu_all_unrecognized_still_picks_one():
    cache = {KLEIN: {"transformer": ["klein-4b-aaa.safetensors", "klein-4b-bbb.safetensors"]}}
    with _Env(cache=cache, all_sms=[]):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, "klein-4b-aaa.safetensors"), (s, n)


def test_autodetect_no_transformers_returns_none():
    # a series shipping no separate transformer weights → base model's default
    with _Env(cache={KLEIN: {"transformer": []}}, all_sms=[120]):
        assert mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN) == (None, None)


# ------------------------------ filtering ------------------------------
def test_filter_excludes_incompatible_sm86():
    with _Env(cache=_klein_cache(), all_sms=[86]):
        opts = mal.get_transformer_options()
    assert opts[0] == mal.AUTO_DETECT               # auto-detect is first
    assert "None" in opts
    assert _opt(KLEIN, KLEIN_30X) in opts           # INT4 runs on SM86
    assert _opt(KLEIN, KLEIN_50X) not in opts       # FP4 hidden on SM86
    assert _opt(KLEIN, KLEIN_40X) not in opts       # FP8 hidden on SM86


def test_filter_shows_all_on_sm120():
    with _Env(cache=_klein_cache(), all_sms=[120]):
        opts = mal.get_transformer_options()
    for name in (KLEIN_50X, KLEIN_40X, KLEIN_30X):
        assert _opt(KLEIN, name) in opts, name


def test_filter_unknown_sm_shows_all():
    # SM undetectable → conservative: do not filter anything
    with _Env(cache=_klein_cache(), all_sms=[]):
        opts = mal.get_transformer_options()
    for name in (KLEIN_50X, KLEIN_40X, KLEIN_30X):
        assert _opt(KLEIN, name) in opts, name


# ----------------------------- multi-GPU -----------------------------
def test_multigpu_mixed_sm_uses_min_for_filter():
    # a 5090(120)+3060(86) box, no pinned device → key on MIN SM (86)
    with _Env(cache=_klein_cache(), all_sms=[120, 86]):
        opts = mal.get_transformer_options()
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
    assert _opt(KLEIN, KLEIN_50X) not in opts       # 50x can't run on the 3060
    assert _opt(KLEIN, KLEIN_30X) in opts
    assert (s, n) == (KLEIN, KLEIN_30X), (s, n)      # auto-detect also uses MIN SM


def test_target_gpu_sm_min_across_mixed():
    with _Env(all_sms=[89, 120, 86]):
        assert mal._target_gpu_sm() == 86


# ---------------------- fail-safe: no compatible weight ----------------------
def test_autodetect_all_incompatible_returns_none():
    # a series shipping ONLY 40x+50x (no SM75/80 floor weight) on an SM75/SM86
    # GPU: auto-detect must return (None, None) — defer to the base model's own
    # (GPU-tier-matched) transformer — and NEVER silently pick an incompatible
    # weight (which would reproduce the __trap() this feature prevents).
    cache = {KLEIN: {"transformer": [KLEIN_40X, KLEIN_50X]}}
    for sm in (75, 86):
        with _Env(cache=cache, all_sms=[sm]):
            assert mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN) == (None, None), sm
            opts = mal.get_transformer_options()
            assert _opt(KLEIN, KLEIN_40X) not in opts, sm   # both hidden from dropdown
            assert _opt(KLEIN, KLEIN_50X) not in opts, sm


def test_autodetect_partial_incompatible_picks_compatible():
    # only the incompatible tier is skipped; the compatible one is still picked
    cache = {KLEIN: {"transformer": [KLEIN_50X, KLEIN_40X]}}
    with _Env(cache=cache, all_sms=[89]):   # SM89 → 40x ok, 50x not
        assert mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN) == (KLEIN, KLEIN_40X)


# ------------------- token boundary (no substring collision) -------------------
def test_min_sm_token_boundary_no_collision():
    # a resolution-tagged '1440x' must NOT be misread as the '40x' FP8 tier
    assert mal._transformer_min_sm("qwen-image-1440x-fp4-50x-above.safetensors") == 120
    assert mal._transformer_min_sm("model-1440x-preview.safetensors") == 0
    assert mal._transformer_min_sm("foo-2540x-thing.safetensors") == 0
    assert mal._transformer_min_sm("bar_1230x_baz.safetensors") == 0
    # bare start/end delimited tokens still match
    assert mal._transformer_min_sm("50x-lighting.safetensors") == 120
    assert mal._transformer_min_sm("klein-4b-40x.safetensors") == 89


# --------------------------- backward compat ---------------------------
def test_explicit_selection_still_resolves():
    with _Env(cache=_klein_cache(), all_sms=[86]):
        # an explicit compatible pick resolves unchanged
        s, n = mal.resolve_transformer_selection(_opt(KLEIN, KLEIN_30X), KLEIN)
        assert (s, n) == (KLEIN, KLEIN_30X), (s, n)


def test_explicit_none_still_resolves_to_none():
    with _Env(cache=_klein_cache(), all_sms=[120]):
        assert mal.resolve_transformer_selection("None", KLEIN) == (None, None)


def test_explicit_wrong_series_still_raises():
    # an explicit pick from a DIFFERENT series must still error (unchanged)
    cache = {KLEIN: {"transformer": [KLEIN_50X]},
             QWEN: {"transformer": [QWEN_50X]}}
    with _Env(cache=cache, all_sms=[120]):
        try:
            mal.resolve_transformer_selection(_opt(QWEN, QWEN_50X), KLEIN)
        except ValueError:
            return
        raise AssertionError("expected ValueError for cross-series selection")


# ------------------------- disk-only merged names -------------------------
def test_disk_only_weight_is_also_filtered():
    # a weight present ONLY on disk (empty remote cache) must be filtered too
    with _Env(cache={}, all_sms=[86],
              local={("Klein-4B-Series", "transformer"): [KLEIN_50X, KLEIN_30X]}):
        opts = mal.get_transformer_options()
    assert _opt(KLEIN, KLEIN_50X) not in opts       # disk-only FP4 hidden on SM86
    assert _opt(KLEIN, KLEIN_30X) in opts            # disk-only INT4 shown


def test_disk_only_weight_autodetect():
    with _Env(cache={}, all_sms=[120],
              local={("Klein-4B-Series", "transformer"): [KLEIN_50X, KLEIN_30X]}):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
    assert (s, n) == (KLEIN, KLEIN_50X), (s, n)


if __name__ == "__main__":
    _fns = [v for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)]
    _passed = 0
    for _fn in _fns:
        try:
            _fn(); print(f"  PASS  {_fn.__name__}"); _passed += 1
        except AssertionError as _e:
            print(f"  FAIL  {_fn.__name__}: {_e}")
        except Exception as _e:  # noqa: BLE001
            print(f"  ERROR {_fn.__name__}: {type(_e).__name__}: {_e}")
    print(f"\n{_passed}/{len(_fns)} passed")
    sys.exit(0 if _passed == len(_fns) else 1)
