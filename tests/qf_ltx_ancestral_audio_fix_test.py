#!/usr/bin/env python3
"""Unit + regression for the Variant-A euler-force LTX-2.5 AV audio fix (no comfy / no GPU / no torch).

Relocated to tests/ with the *_test.py suffix so run_plugin_tests.py DISCOVERS + RUNS it — a repo-root
test_*.py was ORPHANED (the runner globs tests/*_test.py only, so it would never re-run: the exact
tests-written-but-never-run trap run_plugin_tests.py exists to close). Covers:
  (1) eta-gated  -> eta=0 + s_churn=0, sampler's own (now deterministic) body runs;
  (2) the iter1 CRASH regression — comfy delegates to *_RF POSITIONALLY (sampling.py:218) so the wrapper
      must normalise via Signature.bind and NOT double-bind eta ("multiple values for argument 'eta'");
  (3) non-QF-AV -> byte-unchanged;
  (4) the NO-ETA+ns branch -> SUBSTITUTES sample_euler (zeroing lcm/ddpm/er_sde's own noise integrator
      degenerates — see the cluster dossier's measured A/B — so we run the proven euler trajectory);
  (5) churn-only (heun/dpm_2/heunpp2: s_churn, NO eta, NO noise_sampler) -> s_churn=0 KEEPS its own
      2nd-order body, NOT euler-subbed (the iter3 over-substitution regression guard);
  (6) the _FORCE_EULER override -> a has_eta sampler MEASURED broken (dpm_2_ancestral) routes to euler-sub,
      NOT eta=0 (empirical override beats the has_eta proxy), and BEFORE the has_eta branch.
Exit 0 = pass (run_plugin_tests.py convention).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # plugin root importable from tests/
import qf_ltx_ancestral_audio_fix as fix


class _X:  # minimal stand-in for a latent tensor (the fix never calls tensor methods on it)
    shape = (1, 100)


def _run():
    x = _X()
    cap = {}

    def eta_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                    eta=1.0, s_noise=1.0, noise_sampler=None, s_churn=0.0):
        cap.clear(); cap.update(which="eta_body", eta=eta, s_churn=s_churn); return "ETA_ORIG"

    sig = fix.inspect.signature(eta_sampler)
    w_eta = fix._wrap(eta_sampler, "sample_x_eta", sig, sig.parameters)

    # (1) QF-AV eta-gated -> eta=0 + s_churn=0, runs the sampler's own (now deterministic) body
    fix._resolve_qf_av = lambda m: m
    assert w_eta("M", x, [1.0, 0.0], extra_args={}, callback=None, disable=True) == "ETA_ORIG"
    assert cap["eta"] == 0.0 and cap["s_churn"] == 0.0, cap

    # (2) REGRESSION — eta+s_churn passed POSITIONALLY (as the _RF delegation does) must NOT raise
    #     'multiple values for argument eta'; bind normalises and forces them to 0.
    out = w_eta("M", x, [1.0, 0.0], {}, None, False, 0.7, 1.0, None, 3.0)  # eta=0.7, s_churn=3.0 positional
    assert out == "ETA_ORIG" and cap["eta"] == 0.0 and cap["s_churn"] == 0.0, cap

    # (3) non-QF -> byte-unchanged (defaults forwarded, nothing overridden)
    fix._resolve_qf_av = lambda m: None
    w_eta("PLAIN", x, [1.0, 0.0])
    assert cap["eta"] == 1.0 and cap["s_churn"] == 0.0, cap

    # (4) NO-ETA sampler (lcm/ddpm/er_sde) -> SUBSTITUTES sample_euler, NOT the sampler's own body.
    euler_calls = {}
    fix._ORIG_EULER = lambda model, x, sigmas, extra_args=None, callback=None, disable=None: (
        euler_calls.update(model=model, sigmas=sigmas, extra_args=extra_args, disable=disable) or "EULER_SUB")

    def lcm_like(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, s_noise=1.0):
        cap.clear(); cap.update(which="lcm_body_RAN"); return "LCM_ORIG"   # must NOT be reached on QF-AV

    sig2 = fix.inspect.signature(lcm_like)
    w_lcm = fix._wrap(lcm_like, "sample_lcm", sig2, sig2.parameters)
    fix._resolve_qf_av = lambda m: m
    cap.clear()
    r = w_lcm("M", x, [1.0, 0.0], extra_args={"seed": 7}, callback=None, disable=True)
    assert r == "EULER_SUB", r                                     # euler was substituted
    assert "which" not in cap, "lcm body must NOT run on QF-AV (it degenerates)"
    assert euler_calls["model"] == "M" and euler_calls["extra_args"] == {"seed": 7}, euler_calls

    # (4b) NO-ETA+ns on non-QF -> the sampler's own body runs (byte-unchanged)
    fix._resolve_qf_av = lambda m: None
    assert w_lcm("PLAIN", x, [1.0, 0.0]) == "LCM_ORIG"

    # (5) CHURN-ONLY no-eta (heun/dpm_2/heunpp2: s_churn, NO eta, NO noise_sampler) -> s_churn=0 and runs
    #     its OWN 2nd-order body, NOT euler-substituted. (iter3 REGRESSION guard: these were wrongly
    #     euler-subbed, silently downgrading their 2nd-order predictor-corrector to 1st-order euler.)
    def heun_like(model, x, sigmas, extra_args=None, callback=None, disable=None,
                  s_churn=0.0, s_tmin=0.0, s_tmax=1e30, s_noise=1.0):
        cap.clear(); cap.update(which="heun_body_RAN", s_churn=s_churn); return "HEUN_ORIG"
    sig3 = fix.inspect.signature(heun_like)
    w_heun = fix._wrap(heun_like, "sample_heun", sig3, sig3.parameters)
    fix._resolve_qf_av = lambda m: m       # _ORIG_EULER is a lambda from arm (4) — must NOT be used here
    cap.clear()
    assert w_heun("M", x, [1.0, 0.0], extra_args={}, callback=None, disable=True) == "HEUN_ORIG"  # OWN body
    assert cap.get("which") == "heun_body_RAN" and cap["s_churn"] == 0.0, cap   # churn neutralized, body kept

    # (6) _FORCE_EULER override: a has_eta sampler whose body-preserving form (eta=0) was MEASURED broken
    #     (dpm_2_ancestral: 27KB no-audio) -> euler-SUBSTITUTED, NOT eta=0. Empirical override beats the proxy.
    assert "sample_dpm_2_ancestral" in fix._FORCE_EULER, "the measured breaker must be listed"
    def dpm2anc_like(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
        cap.clear(); cap.update(which="dpm2anc_body_RAN"); return "DPM2ANC_ORIG"    # must NOT run on QF-AV
    sig4 = fix.inspect.signature(dpm2anc_like)
    w_force = fix._wrap(dpm2anc_like, "sample_dpm_2_ancestral", sig4, sig4.parameters)  # name IS in _FORCE_EULER
    fix._resolve_qf_av = lambda m: m
    cap.clear()
    assert w_force("M", x, [1.0, 0.0], extra_args={}, callback=None, disable=True) == "EULER_SUB"  # euler-subbed
    assert "which" not in cap, "a _FORCE_EULER sampler must NOT run its own (broken) body on QF-AV"

    print("PASS: eta-gated->eta0; no-eta+ns->euler-sub; churn-only->s_churn0 (own body); "
          "_FORCE_EULER->euler-sub; POSITIONAL no-crash; non-QF unchanged")


if __name__ == "__main__":
    _run()
