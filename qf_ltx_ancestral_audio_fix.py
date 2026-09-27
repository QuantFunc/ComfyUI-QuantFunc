"""QuantFunc LTX-2.5 AV — force a NON-RE-NOISED (deterministic) trajectory under ANY re-noising sampler.

ROOT CAUSE (machine-confirmed): comfy's ancestral/SDE/consistency samplers RE-NOISE x every step. The
QuantFunc LTX-2.5 AV plugin drives a STATELESS flow-match ENGINE forward ("latents in / velocity out;
the sampler owns x") that is only valid on a NON-re-noised (deterministic) trajectory, so the per-step
re-noise collapses the low-dim AUDIO lane to silence (video tolerates it). Plain `euler` (no re-noise) is
audible.

FIX (user mandate "无论什么采样器都要输出正常语音"): for QF LTX-2.5 AV models ONLY, run every
USER-SELECTABLE re-noising sampler on a deterministic trajectory that keeps its AUDIO lane clean. The BAR
is clean audio for EVERY sampler (mandate); preserving each sampler's exact solver-order is a bonus we keep
only where MEASURED clean. Routing (checked in order):

  0. `_FORCE_EULER` — samplers whose body-preserving neutralization was MEASURED to break on the QF AV
     model → SUBSTITUTE sample_euler. EMPIRICAL ground truth, not a signature proxy. Two measured failure
     modes: (i) SILENT/DEGENERATE audio (e.g. dpm_2_ancestral at eta=0 → a 27 KB no-audio output: its
     2nd-order interior-sigma eval lands off-schedule and the engine's nearest-by-value step_index
     duplicates a step); (ii) a comfy cfg++ sampler-body view/reshape to a single-lane shape that is
     invalid for the 2x (video+audio) AV latent → a hard RuntimeError regardless of this fix (euler-sub
     runs plain euler instead, bypassing the broken body). Clean audio wins over solver order.
  1. has `noise_sampler` and NO `eta` (lcm, ddpm, er_sde, res_multistep, sa_solver, ...): the noise draw is
     (or, for the gated ones, may be) the integrator — zeroing it does not give a valid step; for the
     consistency/posterior ones (lcm/ddpm/er_sde) it DEGENERATES (measured). SUBSTITUTE sample_euler
     outright — the one proven-clean trajectory. (A few of these, e.g. res_multistep, are deterministic by
     default and are thereby downgraded to euler — an ACCEPTED tradeoff: guaranteed-clean audio for the
     whole class > preserving a niche sampler's solver, and the alternative risks silence for a future
     integrator sampler. Every substitution is LOGGED, so the downgrade is disclosed, not silent.)
  2. has `eta` (euler_ancestral, dpmpp_*_sde, dpmpp_2s_ancestral, seeds_2, ...): set `eta=0` (+ `s_churn=0`
     if present). eta=0 recovers the sampler's OWN deterministic ODE by construction (the `if eta>0`
     re-noise block is skipped). Measured clean; keeps its solver (e.g. dpmpp 2nd-order).
  3. has `s_churn` only (euler, heun, dpm_2, heunpp2): s_churn=0 skips the ONLY noise source (the
     gamma-gated randn), KEEPING the sampler's genuine (2nd-order predictor-corrector) body. Do NOT
     substitute euler here — that would silently downgrade heun/dpm_2 to 1st-order euler.

Per-sampler dB validation (euler-reference A/B on the real QF LTX-2.5 AV int4 model): see the fix commit
message + cluster dossier. Non-QF-AV models are BYTE-UNCHANGED (the wrapper early-returns to the original).

We wrap only comfy.samplers.KSampler.SAMPLERS (the UI-exposed names), NOT the internal `*_RF` delegates —
euler_ancestral etc. delegate to their `_RF` sibling POSITIONALLY (sampling.py:218/342/718); wrapping BOTH
double-bound eta ("got multiple values for argument 'eta'"). The parent's forced eta=0 propagates into the
UNwrapped `_RF` on its own. Args are normalised via inspect.Signature.bind so forcing eta/s_churn never
collides with a positionally-passed value.

TRADEOFF (user-ACCEPTED): the VIDEO becomes deterministic (loses ancestral stochasticity) — the price for
clean audio + full generality + safety (decoupling the lanes is unsafe: video<->audio cross-attention).
"""
import inspect
import logging

# The plugin makes this logger console-safe (qf_engine.logger) when it imports the module (#738).
_log = logging.getLogger(__name__)


_installed = False
_logged_samplers = set()
_ORIG_EULER = None   # the UNWRAPPED sample_euler, captured at install() before wrapping (substitution target)

# EMPIRICAL: samplers whose body-preserving neutralization (eta=0 / s_churn=0) was MEASURED to produce
# silent/degenerate audio through the engine's stateless forward → force euler-substitution instead. Add a
# name here ONLY on a measured break; keep the measured evidence in the commit/dossier. The _gpu/_cfg_pp
# variants of a listed base inherit its eta=0 trajectory (the wrapper removes the extra noise), so they are
# listed alongside the base.
_FORCE_EULER = {
    # (i) eta=0 gives SILENT/DEGENERATE audio through the engine's stateless forward — 2nd-order/ancestral
    #     samplers whose interior-sigma eval lands off-schedule and the engine's nearest-by-value
    #     step_index duplicates a step:
    "sample_dpm_2_ancestral",              # eta=0 -> 27 KB no-audio output (measured)
    "sample_res_multistep_ancestral",      # eta=0 -> -13.6 dB clipping + over-saturated video, diverges
    # (ii) cfg++ samplers whose OWN body view/reshapes the latent to a SINGLE-lane shape that is invalid
    #      for the 2x (video+audio) AV latent → they RuntimeError ("shape [1,128,16,9,16] invalid for
    #      input of size 589824") regardless of this fix; euler-sub runs plain euler instead, bypassing
    #      the broken cfg++ body → clean audio. (Measured on 远程-linux; the comfy source LINE NUMBERS
    #      for these samplers vary by ComfyUI version, so we key on the sampler NAME, not a line.)
    "sample_euler_ancestral_cfg_pp",       # cfg++ view/reshape crash on the AV latent (measured)
    "sample_dpmpp_2s_ancestral_cfg_pp",    # cfg++ view/reshape crash on the AV latent (measured)
    "sample_res_multistep_ancestral_cfg_pp",  # ancestral AND cfg++ (both classes above)
}


def _resolve_qf_av(model):
    """Return the QFLTXAVModel if `model` (a comfy sampler wrapper) wraps one, else None.
    Walks the .inner_model chain (depth-robust: Guider_Basic/CFGGuider nest differently)."""
    try:
        from .qf_ltx_modelpatcher import QFLTXAVModel
    except Exception:
        return None
    node = model
    for _ in range(6):
        if isinstance(node, QFLTXAVModel):
            return node
        node = getattr(node, "inner_model", None)
        if node is None:
            break
    return None


def _log_once(name, how):
    if name in _logged_samplers:
        return
    _logged_samplers.add(name)
    _log.warning("[qf_native] LTX-2.5 AV: sampler '%s' -> %s for the QF AV model - the engine's stateless "
                 "flow-match forward needs a non-re-noised (deterministic) trajectory or the audio lane is "
                 "silenced. Video is deterministic here (accepted tradeoff).", name, how)


def _wrap(orig, name, sig, params, guard=None):
    """Wrap ONE re-noising sampler. sig/params/guard are threaded from install() (computed once)."""
    has_eta = "eta" in params
    has_ns = "noise_sampler" in params
    has_churn = "s_churn" in params

    def wrapped(model, x, sigmas, *args, **kwargs):
        m = _resolve_qf_av(model)
        if m is None:
            return orig(model, x, sigmas, *args, **kwargs)   # non-QF-AV: byte-unchanged
        try:
            ba = sig.bind(model, x, sigmas, *args, **kwargs)  # normalise positional+kw -> no collision
            forced = name in _FORCE_EULER                     # measured-broken body-preserving form
            if (forced or (has_ns and not has_eta)) and _ORIG_EULER is not None:
                # SUBSTITUTE the proven euler trajectory: either a measured breaker, or a no-eta
                # noise-integrator (lcm/ddpm/er_sde/...). Run euler with the sampler's own universal args.
                a = ba.arguments
                target = _ORIG_EULER
                call_args = (a["model"], a["x"], a["sigmas"])
                call_kwargs = {"extra_args": a.get("extra_args"),
                               "callback": a.get("callback"),
                               "disable": a.get("disable")}
                how = ("euler-sub (its body-preserving form measured broken)" if forced
                       else "euler-sub (no-eta, noise is the integrator)")
            elif has_eta:
                # eta=0 recovers the sampler's OWN deterministic ODE (the `if eta>0` re-noise is skipped).
                ba.arguments["eta"] = 0.0
                if has_churn:
                    ba.arguments["s_churn"] = 0.0
                target, call_args, call_kwargs = orig, ba.args, ba.kwargs
                how = "eta=0 (its deterministic ODE, body kept)"
            else:
                # churn-only (euler/heun/dpm_2/heunpp2): s_churn=0 skips the only noise source, KEEPING
                # the sampler's genuine (2nd-order) body. (Also the euler-unavailable fallback.)
                if has_churn:
                    ba.arguments["s_churn"] = 0.0
                target, call_args, call_kwargs = orig, ba.args, ba.kwargs
                how = "s_churn=0 (keeps its own deterministic body)"
            _log_once(name, how)
        except Exception as e:  # never break sampling - degrade to the original (unfixed) behavior
            _log.warning("[qf_native] LTX-2.5 AV euler-force setup failed for %s (%r); sampler unchanged.", name, e)
            return orig(model, x, sigmas, *args, **kwargs)
        # outside the try: a sampler-internal error propagates, through the plugin's boundary (install's guard)
        return (guard(target) if guard else target)(*call_args, **call_kwargs)

    return wrapped


def install(guard=None):
    """Wrap the USER-SELECTABLE re-noising samplers (comfy.samplers.KSampler.SAMPLERS with an
    eta/s_churn/noise_sampler param). Idempotent; transparent for non-QF-AV models; safe no-op if comfy
    isn't importable. Internal `*_RF` delegates are deliberately NOT wrapped (their parent threads the
    forced eta down positionally). `guard` wraps the sampler a QF-AV run calls (the plugin passes its
    console-safe boundary, qf_engine.console_safe_errors: this module loads without the package)."""
    global _installed, _ORIG_EULER
    if _installed:
        return
    try:
        import comfy.k_diffusion.sampling as S
    except Exception as e:
        _log.warning("[qf_native] LTX-2.5 AV audio-fix: comfy sampling import failed (%r); not installed.", e)
        return
    _ORIG_EULER = getattr(S, "sample_euler", None)   # capture BEFORE wrapping (substitution target)
    try:
        import comfy.samplers
        names = list(comfy.samplers.KSampler.SAMPLERS)          # exactly the UI-exposed sampler names
    except Exception:
        # fallback: sample_* funcs but SKIP internal *_RF delegates (they collide on positional re-entry)
        names = [n[len("sample_"):] for n in dir(S)
                 if n.startswith("sample_") and not n.endswith("_RF")]
    wrapped_names, skipped_renoising = [], []
    for sname in names:
        fn_name = "sample_" + sname
        fn = getattr(S, fn_name, None)
        if not callable(fn) or getattr(fn, "_qf_av_wrapped", False):
            continue
        try:
            sig = inspect.signature(fn)
            params = sig.parameters
        except (TypeError, ValueError):
            continue
        if "eta" in params or "s_churn" in params or "noise_sampler" in params:   # re-noising samplers
            w = _wrap(fn, fn_name, sig, params, guard)
            w._qf_av_wrapped = True
            setattr(S, fn_name, w)
            wrapped_names.append(fn_name)
    # coverage-completeness: a UI sampler that re-noises but isn't a sample_<name> module attr would escape
    # silently (none today: ddim->sample_euler already wrapped; uni_pc is a deterministic solver). Surface it.
    for sname in names:
        fn = getattr(S, "sample_" + sname, None)
        if fn is None:
            skipped_renoising.append(sname)
    _installed = True
    if wrapped_names:
        _log.debug("[qf_native] LTX-2.5 AV euler-force audio-fix installed (%d user-selectable re-noising "
                   "samplers wrapped: %s). QF-AV models run these on a deterministic trajectory so the "
                   "audio lane is not silenced; non-QF-AV models are unaffected.%s",
                   len(wrapped_names), ", ".join(sorted(wrapped_names)),
                   ("" if not skipped_renoising else
                    " NOTE: %d UI sampler(s) have no sample_<name> module attr (not wrapped, verified "
                    "deterministic/aliased today): %s" % (len(skipped_renoising), ", ".join(sorted(skipped_renoising)))))
