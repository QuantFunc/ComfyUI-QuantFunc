"""Tests for the transformer [auto-detect] default + GPU-SM compatibility filter.

The transformer dropdown must (a) default to '[auto-detect]', which resolves to
the HIGHEST-tier weight the target GPU can run, and (b) HIDE weights whose
minimum SM exceeds the target GPU (a 50x FP4 weight must not be offered on an
SM86 card — it __trap()s at runtime). The TARGET GPU is the DEFAULT CUDA device
(torch device 0) — the device BuildPipeline runs the transformer on by default —
so a weight offered/auto-picked is ALWAYS runnable on the actual run-device under
any CUDA_VISIBLE_DEVICES mask or CUDA_DEVICE_ORDER (torch "device 0" == the index
BuildPipeline defaults to). Under the default FASTEST_FIRST ordering device 0 is
the best GPU (e.g. the 4090 on a 3060+4090 box → the 40x/FP8 tier shows).

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
    """Stub the resource cache + the DEFAULT CUDA device SM (torch device 0) +
    optional per-index device SMs (dev_sm={idx: sm}, for the SELECTED-device
    auto-pick path) + on-disk listing; restore on exit."""

    def __init__(self, cache=None, device_sm=0, dev_sm=None, local=None):
        self.cache = cache or {}
        self.device_sm = device_sm          # SM of torch device 0 (the default run-device)
        self.dev_sm = dev_sm or {}          # {device_idx: sm} for _device_sm_by_index
        self.local = local or {}            # {(short, resource_type): [names]}

    def __enter__(self):
        self._orig = (mal._resource_cache, mal._default_device_sm,
                      mal._device_sm_by_index, mal._list_local_resource_names)
        mal._resource_cache = self.cache
        mal._default_device_sm = lambda: self.device_sm
        mal._device_sm_by_index = lambda idx: self.dev_sm.get(idx, 0)
        mal._list_local_resource_names = \
            lambda short, rtype: list(self.local.get((short, rtype), []))
        return self

    def __exit__(self, *exc):
        (mal._resource_cache, mal._default_device_sm,
         mal._device_sm_by_index, mal._list_local_resource_names) = self._orig
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
# device_sm = the SM of the DEFAULT CUDA device (torch device 0), the GPU the
# transformer actually runs on.
def test_autodetect_sm120_picks_fp4():
    with _Env(cache=_klein_cache(), device_sm=120):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, KLEIN_50X), (s, n)


def test_autodetect_sm89_picks_fp8():
    with _Env(cache=_klein_cache(), device_sm=89):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, KLEIN_40X), (s, n)


def test_autodetect_sm86_picks_int4():
    with _Env(cache=_klein_cache(), device_sm=86):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, KLEIN_30X), (s, n)


def test_autodetect_sm75_picks_int4():
    with _Env(cache=_klein_cache(), device_sm=75):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, KLEIN_30X), (s, n)


def test_autodetect_datacenter_sm_mapping():
    # DATACENTER GPUs are mapped purely by NUMERIC min-SM (the tokens are consumer
    # names but the comparison is on SM values): A100(80)->INT4/30x, H100/H200(90)->
    # FP8/40x, B200/GB200(100)->FP8/40x. Critically SM100 must pick 40x, NOT 50x —
    # the engine's FP4 is sm_120a-only, so a B200 (sm_100a) has NO FP4 path and must
    # fall to FP8. Locks against a future accidental FP4-on-SM100 regression.
    for sm, expect in ((80, KLEIN_30X), (90, KLEIN_40X), (100, KLEIN_40X)):
        with _Env(cache=_klein_cache(), device_sm=sm):
            s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
            assert (s, n) == (KLEIN, expect), (sm, s, n)
    # explicit proof B200 is NOT handed FP4:
    assert mal._transformer_min_sm(KLEIN_50X) > 100, "FP4 tier must stay > SM100"


def test_autodetect_known_gpu_ignores_unrecognized_token():
    # On a KNOWN GPU, an unrecognized-token weight (min-SM 0 = "unknown
    # compatibility") must NOT be auto-picked when NO recognized tier is
    # compatible — it might need a higher SM (→ __trap). Auto-detect defers to
    # the base model (None); the dropdown still SHOWS it for a manual pick.
    unk = "klein-4b-nextgen-mystery.safetensors"   # no tier token → min-SM 0
    cache = {KLEIN: {"transformer": [KLEIN_50X, unk]}}  # 50x needs SM120 (incompat on 86)
    with _Env(cache=cache, device_sm=86):
        assert mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN) == (None, None)
    # a RECOGNIZED-compatible tier still wins over the unrecognized one:
    cache2 = {KLEIN: {"transformer": [KLEIN_30X, unk]}}
    with _Env(cache=cache2, device_sm=86):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, KLEIN_30X), (s, n)
    # and the unrecognized weight is NOT filtered from the dropdown (manual pick):
    with _Env(cache=cache, device_sm=86):
        assert _opt(KLEIN, unk) in mal.get_transformer_options()


def test_autodetect_unknown_gpu_picks_safest_lowest_tier():
    # SM undetectable (CPU-only / no CUDA) → safest = lowest tier (INT4/30x)
    with _Env(cache=_klein_cache(), device_sm=0):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, KLEIN_30X), (s, n)


def test_autodetect_unknown_gpu_prefers_known_tier_over_unrecognized():
    # SM undetectable + catalog has an UNRECOGNIZED-token file (min-SM 0) next to
    # a real 30x (min-SM 75): the known-safe 30x wins — an unknown token means
    # "unknown compatibility", not "runs everywhere", so it must NOT be preferred.
    cache = {KLEIN: {"transformer": ["klein-4b-mystery.safetensors", KLEIN_30X]}}
    with _Env(cache=cache, device_sm=0):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, KLEIN_30X), (s, n)


def test_autodetect_unknown_gpu_all_unrecognized_still_picks_one():
    cache = {KLEIN: {"transformer": ["klein-4b-aaa.safetensors", "klein-4b-bbb.safetensors"]}}
    with _Env(cache=cache, device_sm=0):
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, "klein-4b-aaa.safetensors"), (s, n)


def test_autodetect_no_transformers_returns_none():
    # a series shipping no separate transformer weights → base model's default
    with _Env(cache={KLEIN: {"transformer": []}}, device_sm=120):
        assert mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN) == (None, None)


# ------------------------------ filtering ------------------------------
def test_filter_excludes_incompatible_sm86():
    with _Env(cache=_klein_cache(), device_sm=86):
        opts = mal.get_transformer_options()
    assert opts[0] == mal.AUTO_DETECT               # auto-detect is first
    assert "None" in opts
    assert _opt(KLEIN, KLEIN_30X) in opts           # INT4 runs on SM86
    assert _opt(KLEIN, KLEIN_50X) not in opts       # FP4 hidden on SM86
    assert _opt(KLEIN, KLEIN_40X) not in opts       # FP8 hidden on SM86


def test_filter_shows_all_on_sm120():
    with _Env(cache=_klein_cache(), device_sm=120):
        opts = mal.get_transformer_options()
    for name in (KLEIN_50X, KLEIN_40X, KLEIN_30X):
        assert _opt(KLEIN, name) in opts, name


def test_filter_unknown_sm_shows_all():
    # SM undetectable → conservative: do not filter anything
    with _Env(cache=_klein_cache(), device_sm=0):
        opts = mal.get_transformer_options()
    for name in (KLEIN_50X, KLEIN_40X, KLEIN_30X):
        assert _opt(KLEIN, name) in opts, name


# ---------------- default CUDA device (torch device 0) keying ----------------
def test_target_gpu_sm_is_default_device0():
    # keys on the DEFAULT CUDA device (device 0) SM — NOT max/min over all GPUs.
    with _Env(device_sm=89):
        assert mal._target_gpu_sm() == 89
    with _Env(device_sm=0):
        assert mal._target_gpu_sm() == 0            # CPU-only / no CUDA → don't filter


def test_default_device0_best_shows_all():
    # device 0 is a 5090 (SM120): every tier shows, auto-pick 50x
    with _Env(cache=_klein_cache(), device_sm=120):
        opts = mal.get_transformer_options()
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
    for name in (KLEIN_50X, KLEIN_40X, KLEIN_30X):
        assert _opt(KLEIN, name) in opts, name
    assert (s, n) == (KLEIN, KLEIN_50X), (s, n)


def test_user_scenario_3060_plus_4090_shows_40x():
    # The user's REAL box: RTX 3060 (SM86) + RTX 4090 (SM89). Under the default
    # FASTEST_FIRST CUDA ordering, device 0 = the 4090 (SM89) — the GPU BuildPipeline
    # runs on by default. So the target is SM89 → the 40x/FP8 weight (min-SM 89) MUST
    # show + auto-pick ("我有 40x 的模型却没展示" — fixed); 50x/FP4 (min-SM 120) stays
    # hidden (device 0 can't run it). Keying on device 0 (the run-device) means the
    # 40x is guaranteed runnable where it dispatches — no __trap.
    with _Env(cache=_klein_cache(), device_sm=89):
        assert mal._target_gpu_sm() == 89
        opts = mal.get_transformer_options()
        assert _opt(KLEIN, KLEIN_40X) in opts        # 40x now SHOWS (the user's ask)
        assert _opt(KLEIN, KLEIN_30X) in opts
        assert _opt(KLEIN, KLEIN_50X) not in opts    # 50x FP4: device 0 can't run it → hidden
        s, n = mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN)
        assert (s, n) == (KLEIN, KLEIN_40X), (s, n)  # auto-detect picks 40x (highest ≤ 89)


# -------- SELECTED-device keying (BuildPipeline device switch, the user's blocker) --------
def test_resolve_keys_on_selected_device_idx():
    # The user's REAL repro: device 0 = 4090 (SM89), device 1 = 3060 (SM86).
    # resolve_transformer_selection with an explicit device_idx keys on THAT device
    # (what BuildPipeline passes), NOT device 0. Switching device 0↔1 flips the pick.
    with _Env(cache=_klein_cache(), device_sm=89, dev_sm={0: 89, 1: 86}):
        # device 0 (4090) → 40x (highest ≤ 89)
        assert mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN, device_idx=0) == (KLEIN, KLEIN_40X)
        # SWITCH to device 1 (3060, SM86) → 40x is NOT runnable → picks 30x
        assert mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN, device_idx=1) == (KLEIN, KLEIN_30X)
        # switch BACK to device 0 → 40x again
        assert mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN, device_idx=0) == (KLEIN, KLEIN_40X)


def test_resolve_device_idx_none_uses_default_device0():
    # device_idx omitted (dropdown-populate path, device not yet known) → keys on
    # the DEFAULT device (device 0) via _target_gpu_sm — backward compatible.
    with _Env(cache=_klein_cache(), device_sm=86, dev_sm={0: 89, 1: 86}):
        assert mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN) == (KLEIN, KLEIN_30X)  # device0 stub=86


def test_resolve_selected_device_no_compat_returns_none():
    # series ships only 40x+50x; device 1 (SM86) can run neither → (None,None) so the
    # build()-time backstop raises a clean error (never a device __trap).
    cache = {KLEIN: {"transformer": [KLEIN_40X, KLEIN_50X]}}
    with _Env(cache=cache, dev_sm={1: 86}):
        assert mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN, device_idx=1) == (None, None)


def test_detect_device_sm_reads_torch_per_index():
    # lib_setup._detect_device_sm(idx) reads torch device `idx` (CVD-aware); out-of-range → 0.
    ns = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_capability=lambda i: (8, 9) if i == 0 else (8, 6),
    )
    assert _with_fake_torch(ns, lambda ls: ls._detect_device_sm(0)) == 89
    assert _with_fake_torch(ns, lambda ls: ls._detect_device_sm(1)) == 86
    assert _with_fake_torch(ns, lambda ls: ls._detect_device_sm(5)) == 0   # out of range


# ---------------------- fail-safe: no compatible weight ----------------------
def test_autodetect_all_incompatible_returns_none():
    # a series shipping ONLY 40x+50x (no SM75/80 floor weight) on an SM75/SM86
    # GPU: auto-detect must return (None, None) — defer to the base model's own
    # (GPU-tier-matched) transformer — and NEVER silently pick an incompatible
    # weight (which would reproduce the __trap() this feature prevents).
    cache = {KLEIN: {"transformer": [KLEIN_40X, KLEIN_50X]}}
    for sm in (75, 86):
        with _Env(cache=cache, device_sm=sm):
            assert mal.resolve_transformer_selection(mal.AUTO_DETECT, KLEIN) == (None, None), sm
            opts = mal.get_transformer_options()
            assert _opt(KLEIN, KLEIN_40X) not in opts, sm   # both hidden from dropdown
            assert _opt(KLEIN, KLEIN_50X) not in opts, sm


def test_autodetect_partial_incompatible_picks_compatible():
    # only the incompatible tier is skipped; the compatible one is still picked
    cache = {KLEIN: {"transformer": [KLEIN_50X, KLEIN_40X]}}
    with _Env(cache=cache, device_sm=89):   # SM89 → 40x ok, 50x not
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
    with _Env(cache=_klein_cache(), device_sm=86):
        # an explicit compatible pick resolves unchanged
        s, n = mal.resolve_transformer_selection(_opt(KLEIN, KLEIN_30X), KLEIN)
        assert (s, n) == (KLEIN, KLEIN_30X), (s, n)


def test_explicit_none_still_resolves_to_none():
    with _Env(cache=_klein_cache(), device_sm=120):
        assert mal.resolve_transformer_selection("None", KLEIN) == (None, None)


def test_explicit_incompatible_selection_still_resolves():
    # An explicit pick the FILTER would HIDE (a 40x weight on an SM86 box) must
    # STILL resolve — a saved-workflow explicit value stays backward-compatible
    # even when the dropdown no longer offers it. Filtering hides; it never blocks
    # an explicit resolve (the __trap protection is at selection/auto-detect time,
    # and an explicit incompatible pick is the user's deliberate override).
    with _Env(cache=_klein_cache(), device_sm=86):
        s, n = mal.resolve_transformer_selection(_opt(KLEIN, KLEIN_40X), KLEIN)
        assert (s, n) == (KLEIN, KLEIN_40X), (s, n)


def test_explicit_wrong_series_still_raises():
    # an explicit pick from a DIFFERENT series must still error (unchanged)
    cache = {KLEIN: {"transformer": [KLEIN_50X]},
             QWEN: {"transformer": [QWEN_50X]}}
    with _Env(cache=cache, device_sm=120):
        try:
            mal.resolve_transformer_selection(_opt(QWEN, QWEN_50X), KLEIN)
        except ValueError:
            return
        raise AssertionError("expected ValueError for cross-series selection")


# ------------------------- disk-only merged names -------------------------
def test_disk_only_weight_is_also_filtered():
    # a weight present ONLY on disk (empty remote cache) must be filtered too
    with _Env(cache={}, device_sm=86,
              local={("Klein-4B-Series", "transformer"): [KLEIN_50X, KLEIN_30X]}):
        opts = mal.get_transformer_options()
    assert _opt(KLEIN, KLEIN_50X) not in opts       # disk-only FP4 hidden on SM86
    assert _opt(KLEIN, KLEIN_30X) in opts            # disk-only INT4 shown


# --------- lib_setup: default-device SM is torch device 0 (CVD-aware) ---------
def _with_fake_torch(cuda_ns, fn):
    ls = importlib.import_module(f"{_PKG}.lib_setup")
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = cuda_ns
    orig = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        return fn(ls)
    finally:
        if orig is not None:
            sys.modules["torch"] = orig
        else:
            del sys.modules["torch"]


def test_detect_default_device_sm_reads_torch_device0():
    # _detect_default_device_sm keys on torch DEVICE 0 (CVD-aware — torch honors
    # CUDA_VISIBLE_DEVICES), NOT nvidia-smi. device0=SM89, device1=SM86 → returns 89.
    ns = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_capability=lambda i: (8, 9) if i == 0 else (8, 6),
    )
    assert _with_fake_torch(ns, lambda ls: ls._detect_default_device_sm()) == 89


def test_detect_default_device_sm_cpu_only_returns_zero():
    ns = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        get_device_capability=lambda i: (0, 0),
    )
    assert _with_fake_torch(ns, lambda ls: ls._detect_default_device_sm()) == 0


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
