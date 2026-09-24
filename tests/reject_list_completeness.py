#!/usr/bin/env python3
"""STALENESS MECHANISM for the native seams' _ENGINE_IGNORED_COND_KEYS (mechanism, not memory).

Each QF*Model.extra_conds bypasses comfy's <Base>.extra_conds wholesale, so every conditioning channel it
would consume must be either emitted by us (cross_attn), guarded by the reject-list, handled by a dedicated
hook (concat_latent_image loud-fail; denoise_mask via the scale_latent_inpaint override), documented-accepted
infra (frame_rate/attention_mask), or provably impossible to populate — otherwise it is SILENTLY dropped.
That silent-discard class recurred across CR rounds when a reject-list was maintained by hand against the
visible leaf body only, AND it escaped ENTIRELY for a WHOLE MODEL when this audit was hardcoded to one class.

★ MODEL-AGNOSTIC (the generality fix): this audit is driven by the `_AUDITED_MODELS` table, NOT a literal
"WAN21". Every native seam registers (comfy_class, plugin_file, covered_elsewhere) and is scanned. Adding a
new seam = adding a table row; forgetting to audit a new seam is a table omission, not a silent miss inside
this file. It parses, from the INSTALLED comfy, for EACH model: <class>.extra_conds + the `super()`
BaseModel.extra_conds it opens with, AND the two methods that BaseModel.extra_conds INVOKES — concat_cond +
encode_adm (resolved to the model's override if any). SCOPE (stated honestly, not "FULL chain"): it does NOT
recurse arbitrarily — if a FUTURE comfy makes extra_conds call a THIRD helper, add it to `callees`. It fails
(exit 1) if any consumable key for any model is neither guarded nor covered by a dedicated hook / documented
acceptance. Run it on every comfy upgrade.

Usage: python3 reject_list_completeness.py [<comfy_root>]   # audits every model in _AUDITED_MODELS
       python3 reject_list_completeness.py --selftest        # negative control: prove a comfy-absent run is LOUD
Exit 0 = complete for all models; exit 1 = a straggler exists (names it + the model); exit 77 = SKIPPED
(comfy not found / layout changed) — a LOUD skip (a bracketed [SKIP] token uniform with the sibling tests +
a non-zero exit), never a silent exit-0 that a harness reads as PASS (the empty-verdict class this file is
an instance of).
"""
import collections
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_DIR = os.path.dirname(_HERE)   # tests/ lives under the plugin package root

_SKIP_EXIT = 77   # automake SKIP convention — distinct from pass(0)/fail(1) so a comfy-absent run is LOUD

# Shared infra keys never counted as "consumable conditioning": cross_attn is EMITTED by every seam's
# extra_conds; noise/device/latent_image are sampler infrastructure encode_model_conds passes to EVERY
# model.extra_conds (comfy samplers.py:1042). latent_image is the BASE latent the engine already receives via
# x in _apply_model (dropping the redundant cond is correct; an inpaint that also wires a mask is rejected at
# the mask); it surfaces in a scan only for a model whose concat_cond falls back to BaseModel's
# `kwargs.get("latent_image")` — e.g. LTXV, which does not override concat_cond (WAN21 does, so it never sees it).
_INFRA_KEYS = {"cross_attn", "noise", "device", "latent_image"}

# ── the MODEL-AGNOSTIC audit table — one row per native seam (the generality fix) ─────────────────────────
# comfy_class     : the comfy.model_base class the QF*Model subclasses (its extra_conds is what we bypass).
# plugin          : the plugin file (relative to the plugin package root) carrying _ENGINE_IGNORED_COND_KEYS.
# covered_elsewhere: consumable keys NOT in the reject-list, each covered by a VERIFIED means (see per-key
#                    reason). A key here must have a documented reason — it is not a dumping ground.
# callees         : the BaseModel.extra_conds callees to also scan for consumed kwargs (resolved to the
#                    model's override if present). concat_cond + encode_adm today (see the docstring SCOPE).
_Model = collections.namedtuple("_Model", "comfy_class plugin covered_elsewhere callees")
_AUDITED_MODELS = (
    # LTX-2 t2v (QFLTXModel) — the seam the CR generality NO-GO found was NEVER audited. Its reject-list is a
    # defensive superset (mask/keyframe/guide + noise_concat/cross_attn_controlnet/concat_latent_image). Two
    # keys are documented-ACCEPTED (not hook-handled), each with a VERIFIED reason:
    #  • frame_rate — the engine takes fps from the loader's fps widget (_begin options_json); comfy's
    #    frame_rate cond is metadata the external session never reads, and it is ALWAYS emitted (default 25),
    #    so it must NOT be rejected (that would break every run).
    #  • attention_mask — the TE PADDING mask. The plugin runs comfy's connector on the FULL cross_attn
    #    sequence (the shipped _run_connector); it is infra present on normal runs, NOT user-wired
    #    conditioning, so rejecting it would break normal generation. HONEST RESIDUAL: if a future validation
    #    shows the reference connector masks padding, attention_mask must become a HANDLED channel (an LTX
    #    follow-up) — it is NOT a NEW silent-drop introduced here (the shipped _run_connector already ignored
    #    it). Listing it here makes the acceptance EXPLICIT + auditable, not silent.
    #  • denoise_mask — ACCEPTED BY INHERITANCE (wan-align 2026-08-22 "只关注latent"): the Inplace
    #    i2v latent route rides it. It reaches the SAMPLER (KSamplerX0Inpaint), not extra_conds, and
    #    QFLTXModel deliberately INHERITS BaseModel.scale_latent_inpaint (no loud-fail override) so
    #    comfy blends x against the clean latent per step OUTSIDE the model — exact for this seam
    #    because the engine step is stateless in x. Unlike WAN (whose override loud-fails), LTX
    #    consumes the mask through comfy's own machinery.
    _Model("LTXV", "qf_ltx_modelpatcher.py",
           {"attention_mask", "frame_rate", "denoise_mask"},
           ("concat_cond", "encode_adm")),
    # MiniMax-H3 joint-AV (QFH3Model) — this seam was UNAUDITED until the roster scan was repaired
    # (the class pattern stopped matching once the shared mixin was introduced, so the check
    # silently certified nothing; MEASURED). Its reject-list is the same defensive superset shape.
    # Documented-ACCEPTED keys, each CONSUMED by the seam rather than dropped:
    #  • minimax_keyframes / minimax_refs / minimax_token_tags — read in extra_conds and forwarded
    #    to the engine through the av_conds bridge (the fl2va keyframe + ref2va reference paths).
    #  • minimax_payload — the OFFICIAL model_base.MiniMaxH3 channel; the seam re-emits it as a
    #    CONDConstant and binds it session-wide at _begin (with the C2 ordering guard).
    #  • cross_attn — emitted as c_crossattn (the one channel the engine takes).
    #  • latent_shapes — NOT dropped: comfy sets model.latent_shapes and _apply_model uses it to
    #    split the PACKED [B,1,N] AV latent into its video/audio halves (a dedicated mechanism).
    #  • seed — the engine denoises the latent comfy's sampler already noised, so the seam needs no
    #    seed of its own; comfy's own use of it (noise/unclip) is upstream of this model.
    _Model("MiniMaxH3", "qf_h3_modelpatcher.py",
           {"minimax_keyframes", "minimax_refs", "minimax_token_tags", "minimax_payload",
            "latent_shapes", "seed"},
           ("concat_cond", "encode_adm")),
    # Krea-2 t2i (QFKrea2Model) — was UNAUDITED (the roster scan flagged it red on every run, blocking the
    # per-model report for every other row). Its reject-list = reference_latents/reference_latents_method
    # (the krea2 ref2img channel) + the BaseModel controlnet/noise_concat channels; nothing covered elsewhere.
    _Model("Krea2", "qf_krea2_modelpatcher.py", set(), ("concat_cond", "encode_adm")),
    # Qwen-Image-2.1 t2i + edit (QFQwenImage21Model). The rule: a key is accepted only if the engine CONSUMES it
    # or the SAMPLER honours it. Its reject-list = attention_mask (the engine forward is mask-blind),
    # reference_latents_method (QI2.1 has one splice rule — a method would do nothing) + the BaseModel
    # controlnet/noise_concat/concat channels. Documented-ACCEPTED keys:
    #  • image_slots + reference_latents — CONSUMED: the edit channel. extra_conds keeps comfy's own processing
    #    (process_latent_in on the references, CONDConstant slots) and _apply_model forwards both per step
    #    through quantfunc_denoise_step_refs (the engine's forwardEdit = comfy build_sequence).
    #  • denoise_mask — SAMPLER-honoured: KSamplerX0Inpaint blends outside the model; comfy's QwenImage21 has no
    #    concat keys, so its concat_cond never reads it — SetLatentNoiseMask inpainting behaves as in comfy, with
    #    the engine's own (switchable) token-prune applied to the model output exactly as on plain t2i.
    _Model("QwenImage21", "qf_qwenimage21_modelpatcher.py",
           {"image_slots", "reference_latents", "denoise_mask"},
           ("concat_cond", "encode_adm")),
)


def _emit_skip(reason):
    """Single source of the comfy-absent / layout-changed SKIP: a VISIBLE bracketed [SKIP] token (uniform
    with the sibling tests, so any SKIP scanner catches it) + the LOUD _SKIP_EXIT code. Never a silent
    exit-0 — that is the empty-verdict class this file is itself an instance of (a comfy-gated test whose
    silent skip read as a PASS)."""
    print("[SKIP] reject_list_completeness: " + reason)
    return _SKIP_EXIT


def _comfy_model_base(comfy_root):
    return os.path.join(comfy_root, "comfy", "model_base.py")


def _resolve_plugin_path(model):
    return os.path.join(_PLUGIN_DIR, model.plugin)


def _method_body(src, cls, meth):
    i = src.find(f"class {cls}")
    if i < 0:
        return ""
    nxt = re.search(r"\nclass \w", src[i + 5:])
    b = src[i: i + 5 + nxt.start()] if nxt else src[i:]
    k = b.find(f"def {meth}")
    if k < 0:
        return ""
    nd = re.search(r"\n    def \w", b[k + 5:])
    return b[k: k + 5 + nd.start()] if nd else b[k: k + 3000]


def _parent_class(src, cls):
    """The single base class of `cls` in model_base (e.g. QwenImage21 -> QwenImage), module prefix stripped."""
    m = re.search(r"^class\s+" + re.escape(cls) + r"\s*\(\s*([A-Za-z_][\w.]*)", src, re.M)
    return m.group(1).split(".")[-1] if m else None


def _mro(src, cls):
    """`cls`, then its model_base parents in order, ending at BaseModel (model_base uses single inheritance). A
    leaf-only scan missed an INTERMEDIATE parent's extra_conds (QwenImage21 -> QwenImage reads reference_latents /
    reference_latents_method / attention_mask) — the reclassified edit keys were invisible to this audit."""
    chain = [cls]
    while chain[-1] != "BaseModel":
        parent = _parent_class(src, chain[-1])
        if not parent or parent in chain:
            break
        chain.append(parent)
    if chain[-1] != "BaseModel":
        chain.append("BaseModel")
    return chain


def _effective_body(src, cls, meth):
    """Body of the FIRST class in cls's parent chain that defines `meth` (the effective MRO method)."""
    for c in _mro(src, cls):
        b = _method_body(src, c, meth)
        if b:
            return b
    return ""


def _compute(comfy_root, model):
    """Pure scan for ONE model. Returns (consumed, guarded, covered_present, uncovered) as sorted lists, or
    None on a parse miss (comfy layout changed / plugin reject-list not found). No printing — the callers
    decide how to surface it."""
    src = open(_comfy_model_base(comfy_root), encoding="utf-8", errors="replace").read()
    own = _method_body(src, model.comfy_class, "extra_conds")
    base = _method_body(src, "BaseModel", "extra_conds")
    if not own or not base:
        return None  # the model's own OR BaseModel.extra_conds not found — comfy layout changed
    # Scan extra_conds along the WHOLE parent chain (<class> -> ... -> BaseModel: each super() hop) + each
    # BaseModel.extra_conds callee resolved to the nearest override. Scanning ONLY the extra_conds bodies missed
    # callee kwargs (the CR6 blind spot: a future <class>.encode_adm override would have slipped through), and
    # scanning only the leaf + BaseModel missed an intermediate parent (QwenImage21 -> QwenImage).
    chain = "".join(_method_body(src, c, "extra_conds") for c in _mro(src, model.comfy_class))
    for callee in model.callees:
        chain += _effective_body(src, model.comfy_class, callee)
    consumed = set(re.findall(r'kwargs\.get\(\s*["\']([a-z_]+)["\']', chain))
    consumed -= _INFRA_KEYS
    plugin_path = _resolve_plugin_path(model)
    if not os.path.isfile(plugin_path):
        return None  # plugin file not found — parse miss, not a false PASS
    pm = open(plugin_path, encoding="utf-8", errors="replace").read()
    m = re.search(r"_ENGINE_IGNORED_COND_KEYS\s*=\s*\(([^)]*)\)", pm, re.S)
    guarded = set(re.findall(r'["\']([a-z_]+)["\']', m.group(1))) if m else set()
    uncovered = consumed - guarded - model.covered_elsewhere
    return (sorted(consumed), sorted(guarded), sorted(model.covered_elsewhere & consumed), sorted(uncovered))


def _row_defect(model):
    """OUR-side table-row validity (distinct from comfy-external drift): a missing plugin file or a plugin with
    no _ENGINE_IGNORED_COND_KEYS is a bug in THIS table (e.g. a typo'd basename in a future row), NOT comfy
    drift — so it must HARD-FAIL, not degrade to a parse-miss SKIP/PASS. Returns a defect string or None."""
    pp = _resolve_plugin_path(model)
    if not os.path.isfile(pp):
        return f"{model.comfy_class}: plugin file not found: {pp}"
    if "_ENGINE_IGNORED_COND_KEYS" not in open(pp, encoding="utf-8", errors="replace").read():
        return f"{model.comfy_class}: plugin {model.plugin} has no _ENGINE_IGNORED_COND_KEYS"
    return None


def _derived_roster_defects():
    """MODEL-AGNOSTIC COMPLETENESS — DERIVE the seam set from code, don't trust the hand-written table.

    The _AUDITED_MODELS table is authored by hand, so a NEW native seam (a new `class QF<X>Model(
    comfy.model_base.<Base>)` whose extra_conds bypasses comfy) could be added to the plugin WITHOUT a row —
    and its bypass would silently escape this audit (the exact silent-discard class this file exists to catch).
    So we DERIVE the seam set by scanning the plugin package for every QF*Model subclass of comfy.model_base.*
    and assert each such <Base> has an _AUDITED_MODELS row. Adding a seam without registering it HARD-FAILS.

    Returns a list of defect strings (empty = every derived seam is audited). A scan that finds NO seam is
    itself a defect — a zero-reporting ruler must prove it can report non-zero, else its silence is meaningless
    (the QF*Model pattern or _PLUGIN_DIR drifted)."""
    audited = {m.comfy_class for m in _AUDITED_MODELS}
    defects = []
    seen = set()
    for fn in sorted(os.listdir(_PLUGIN_DIR)):
        if not fn.endswith(".py"):
            continue
        try:
            src = open(os.path.join(_PLUGIN_DIR, fn), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        # `class QF<Name>Model(comfy.model_base.<Base>):` — the native-seam pattern. <Base> (WAN21/LTXV/…) is
        # the comfy class whose extra_conds the subclass bypasses, i.e. exactly the _AUDITED_MODELS.comfy_class.
        # The base list may carry OTHER bases before the comfy one (the seams are
        # `class QFKrea2Model(QFSessionModelMixin, comfy.model_base.Krea2)`), so match the whole
        # base list and find the comfy base inside it. Pinning comfy.model_base to the FIRST
        # position is what made this scan match NOTHING once the shared mixin was introduced —
        # i.e. the completeness check silently certified nothing (MEASURED, pre-existing).
        file_matches = 0
        for m in re.finditer(r"class\s+(QF\w*Model)\s*\(([^)]*)\)", src):
            qf_cls, bases = m.group(1), m.group(2)
            file_matches += 1
            bm = re.search(r"comfy\.model_base\.(\w+)", bases)
            if bm is None:
                continue                 # QFLTXAVModel(QFLTXModel): inherits an already-audited seam
            base = bm.group(1)
            seen.add(base)
            if base not in audited:
                defects.append(
                    f"{qf_cls} ({fn}) subclasses comfy.model_base.{base} but there is NO _AUDITED_MODELS row for "
                    f"'{base}' — its extra_conds bypass is UNAUDITED. Add a _Model row for it.")
        # FAMILY-MODULE completeness (2nd derivation arm): every family module — the qf_*_modelpatcher.py
        # naming the registry itself walks, MINUS the shared substrate — must contribute at least one
        # scanned QF*Model seam class. Without this arm, a 4th family whose model class drifts from the
        # QF*Model naming (or that defines none) registers and runs while contributing NOTHING to this
        # audit — the hand-maintained-roster escape a generality review measured.
        if (fn.startswith("qf_") and fn.endswith("_modelpatcher.py")
                and fn != "qf_modelpatcher.py" and file_matches == 0):
            defects.append(
                f"family module {fn} contributes NO recognized `class QF<X>Model(...)` seam to this audit — "
                f"either its model class naming drifted from the scan pattern (rename it QF<X>Model) or the "
                f"family genuinely bypasses the extra_conds audit (add its seam + an _AUDITED_MODELS row).")
    if not seen:
        defects.append(
            "the QF*Model derivation scan matched NO seam in the plugin package (expected e.g. QFKrea2Model / "
            "QFLTXModel). The class pattern or _PLUGIN_DIR drifted — a completeness check that scans nothing "
            "cannot certify completeness.")
    return defects


def check(comfy_root):
    """CLI/CR path: audit EVERY model in _AUDITED_MODELS. Print a per-model report; return 0 (all complete) /
    1 (a straggler OR a malformed table row) / _SKIP_EXIT (every model comfy-parse-missed — comfy layout
    changed). An OUR-side row defect (missing plugin / no reject-list) HARD-FAILS here, NOT just in --selftest,
    so a bad future row can't silently exit-0 in the run_plugin_tests sweep (which calls check(), not selftest)."""
    roster_defects = _derived_roster_defects()
    if roster_defects:
        for d in roster_defects:
            print("REJECT_LIST_COMPLETENESS: FAIL — derived-roster completeness (a QF*Model seam is UNAUDITED, "
                  "or the derivation scanned nothing): " + d)
        return 1
    row_defects = [d for d in (_row_defect(m) for m in _AUDITED_MODELS) if d]
    if row_defects:
        for d in row_defects:
            print("REJECT_LIST_COMPLETENESS: FAIL — malformed audit-table row (OUR bug, not comfy drift): " + d)
        return 1
    any_pass = False
    failed = []           # (comfy_class, [uncovered]) tuples
    parse_missed = []     # comfy_class names
    for model in _AUDITED_MODELS:
        res = _compute(comfy_root, model)
        print(f"--- model {model.comfy_class} ({model.plugin}) ---")
        if res is None:
            print(f"  parse miss (could not locate {model.comfy_class}/BaseModel extra_conds/callees or the "
                  f"plugin reject-list — comfy layout changed?)")
            parse_missed.append(model.comfy_class)
            continue
        consumed, guarded, covered, uncovered = res
        print("  consumed (extra_conds + callees):", consumed)
        print("  guarded reject-list             :", guarded)
        print("  covered elsewhere (hook/accepted):", covered)
        print("  UNCOVERED (must be empty)        :", uncovered)
        if uncovered:
            failed.append((model.comfy_class, uncovered))
        else:
            any_pass = True
    if failed:
        for cls, unc in failed:
            print(f"REJECT_LIST_COMPLETENESS: FAIL — {cls} has a NEW unguarded consumable key: " + ",".join(unc))
        return 1
    if not any_pass and parse_missed:
        return _emit_skip("every audited model parse-missed (" + ",".join(parse_missed) + ")")
    if parse_missed:
        print("REJECT_LIST_COMPLETENESS: note — parse-missed (not audited this run): " + ",".join(parse_missed))
    print("REJECT_LIST_COMPLETENESS: PASS — every consumable key of every audited model is guarded "
          "(reject-list), emitted (cross_attn), or covered (dedicated hook / documented acceptance)")
    return 0


def _resolve_comfy_root(comfy_root, _use_fallback=True):
    """Locate the ComfyUI root: explicit value → COMFY_ROOT env → known install candidates → walk up from
    this file (custom_nodes/qf_native/tests). Mirrors the sibling cfg_context_key_collision_test's
    _find_comfy_root so a comfy-gated test is not SILENTLY SKIPPED when comfy IS locatable (a plain
    `python3 reject_list_completeness.py` from a checkout that is not installed under ComfyUI used to skip).
    _use_fallback=False honors ONLY an explicit valid arg — used by the negative-control selftest to force
    the comfy-absent path regardless of what is installed on this box."""
    if comfy_root and os.path.isfile(_comfy_model_base(comfy_root)):
        return comfy_root
    if not _use_fallback:
        return None
    for c in (os.environ.get("COMFY_ROOT", ""), "/root/qf-comfy-nat/ComfyUI", "/media/jonathan/Data/ComfyUI"):
        if c and os.path.isfile(_comfy_model_base(c)):
            return c
    p = _HERE
    for _ in range(6):
        p = os.path.dirname(p)
        if os.path.isfile(_comfy_model_base(p)):
            return p
    return None


def warn_if_stale(comfy_root=None, logger=None):
    """AUTOMATION: run the scan for EVERY audited model at plugin import and emit ONE logging.warning per
    model if comfy exposes a consumable conditioning key that model's reject-list neither guards nor handles
    (comfy drifted since the list was last audited → those keys would be SILENTLY DROPPED). This is what
    makes the completeness claim STRUCTURAL rather than "someone remembers to run the CLI on every comfy
    upgrade", AND model-agnostic (it audits every seam in _AUDITED_MODELS, not just WAN — the exact gap the
    CR generality NO-GO named). NEVER raises — a tripwire must not break plugin load. Returns a dict
    {comfy_class: [uncovered]} for models that scanned (empty list = clean), or None if NOTHING scanned.

    SCOPE (precise, not "any drift"): it detects a new key added WITHIN the scanned chain _compute follows —
    <class>/BaseModel.extra_conds + concat_cond + encode_adm. A comfy that routes a NEW key through a NEW
    helper method extra_conds calls is NOT auto-detected until that helper is added to a model's `callees`."""
    import logging as _logging
    lg = logger or _logging.getLogger("qf_native")
    try:
        root = _resolve_comfy_root(comfy_root)
        if root is None:
            return None
        results = {}
        for model in _AUDITED_MODELS:
            res = _compute(root, model)
            if res is None:
                continue
            uncovered = res[3]
            results[model.comfy_class] = uncovered
            if uncovered:
                lg.warning(
                    "[qf_native] %s conditioning reject-list is STALE — this comfy exposes cond key(s) %s that "
                    "%s.extra_conds neither guards (_ENGINE_IGNORED_COND_KEYS in %s) nor handles by a dedicated "
                    "hook; a native run would SILENTLY DROP them. Add them to the reject-list or a hook. Audit "
                    "tool: qf_native/tests/reject_list_completeness.py",
                    model.comfy_class, ",".join(uncovered), model.comfy_class, model.plugin)
        return results if results else None
    except Exception:   # noqa: BLE001 — a tripwire must never break plugin import
        return None


def _selftest():
    """NEGATIVE CONTROL (b2, same spec as cfg_context arm10's discrimination leg): prove a comfy-ABSENT run
    is LOUD — a VISIBLE bracketed [SKIP] token + a non-zero exit distinct from pass(0)/fail(1) — never a
    silent exit-0 that a harness reads as PASS. Also asserts the audit is MODEL-AGNOSTIC (the table carries
    LTXV, not just WAN21 — the generality property the CR NO-GO required). Runs WITHOUT comfy, so it is
    always executable and never itself skips."""
    import io
    import contextlib
    bad = 0
    if _SKIP_EXIT in (0, 1):
        print("[FAIL] selftest: _SKIP_EXIT=%d collides with pass(0)/fail(1)" % _SKIP_EXIT); bad += 1
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _emit_skip("selftest probe")
    out = buf.getvalue()
    if "[SKIP]" not in out or "PASS" in out or rc != _SKIP_EXIT:
        print("[FAIL] selftest: skip is not loud (out=%r rc=%r)" % (out.strip(), rc)); bad += 1
    if _resolve_comfy_root("/nonexistent-selftest-root", _use_fallback=False) is not None:
        print("[FAIL] selftest: a bogus comfy root was not detected as absent"); bad += 1
    # MODEL-AGNOSTIC property: the table must carry >1 model and INCLUDE LTXV (the seam that used to escape).
    classes = {m.comfy_class for m in _AUDITED_MODELS}
    if len(_AUDITED_MODELS) < 2 or "LTXV" not in classes or "Krea2" not in classes:
        print("[FAIL] selftest: audit table is not model-agnostic (classes=%r)" % sorted(classes)); bad += 1
    # Every model row must point at an existing plugin file + a real reject-list tuple in it (catch a typo'd
    # plugin basename that would silently parse-miss forever).
    for m in _AUDITED_MODELS:
        pp = _resolve_plugin_path(m)
        if not os.path.isfile(pp):
            print("[FAIL] selftest: %s plugin file missing: %s" % (m.comfy_class, pp)); bad += 1
        elif "_ENGINE_IGNORED_COND_KEYS" not in open(pp, encoding="utf-8", errors="replace").read():
            print("[FAIL] selftest: %s plugin %s has no _ENGINE_IGNORED_COND_KEYS" % (m.comfy_class, m.plugin)); bad += 1
    # DERIVED-ROSTER gate (_derived_roster_defects — the generality fix), proven BOTH directions here because it
    # is env-INDEPENDENT (scans the plugin package, needs no comfy) whereas check()'s call to it is comfy-gated
    # and SKIPs on this box. Without this, the derivation would be a code-only assertion no run exercises.
    import tempfile
    import shutil
    global _PLUGIN_DIR
    _real_plugin_dir = _PLUGIN_DIR
    # (pos) the REAL, fully-audited plugin dir → NO defect (the derivation must not false-positive on good state).
    if _derived_roster_defects():
        print("[FAIL] selftest: derived-roster reported a defect on the real fully-audited plugin dir"); bad += 1
    # (neg-1) a synthetic UNAUDITED seam MUST be reported (the silent-new-seam class this generality fix closes).
    _d1 = tempfile.mkdtemp(prefix="qfrl_sel1_")
    try:
        with open(os.path.join(_d1, "qf_foo_modelpatcher.py"), "w", encoding="utf-8") as _fh:
            _fh.write("class QFFooModel(comfy.model_base.FooBaseXYZ):\n    pass\n")
        _PLUGIN_DIR = _d1
        _defs1 = _derived_roster_defects()
        if not any("FooBaseXYZ" in x for x in _defs1):
            print("[FAIL] selftest: derived-roster did NOT flag an unaudited QFFooModel(FooBaseXYZ) (defs=%r)" % _defs1); bad += 1
    finally:
        _PLUGIN_DIR = _real_plugin_dir
        shutil.rmtree(_d1, ignore_errors=True)
    # (neg-2) a scan that matches NO seam is ITSELF a defect (a zero-reporting ruler must prove it can report).
    _d2 = tempfile.mkdtemp(prefix="qfrl_sel0_")
    try:
        _PLUGIN_DIR = _d2
        _defs2 = _derived_roster_defects()
        if not any("matched NO seam" in x for x in _defs2):
            print("[FAIL] selftest: derived-roster did NOT flag a vacuous (no-seam) scan (defs=%r)" % _defs2); bad += 1
    finally:
        _PLUGIN_DIR = _real_plugin_dir
        shutil.rmtree(_d2, ignore_errors=True)
    # (neg-3) FAMILY-MODULE completeness arm: a qf_<fam>_modelpatcher.py contributing NO recognized
    # QF*Model seam class MUST be reported (round-3 CR: the arm existed but had no committed
    # positive-trip proof — a guard that is never shown to fire proves nothing). The dir also
    # carries one VALID seam so the vacuous-scan defect (neg-2) cannot mask this arm's message.
    _d3 = tempfile.mkdtemp(prefix="qfrl_sel3_")
    try:
        with open(os.path.join(_d3, "qf_ok_modelpatcher.py"), "w", encoding="utf-8") as _fh:
            _fh.write("class QFOkModel(comfy.model_base.Krea2):\n    pass\n")
        with open(os.path.join(_d3, "qf_bogus_modelpatcher.py"), "w", encoding="utf-8") as _fh:
            _fh.write("FAMILY = 'bogus'\n\nclass BogusSeam:\n    pass\n")   # drifted naming: no QF*Model
        _PLUGIN_DIR = _d3
        _defs3 = _derived_roster_defects()
        if not any("qf_bogus_modelpatcher.py" in x and "NO recognized" in x for x in _defs3):
            print("[FAIL] selftest: family-module arm did NOT flag a family module with no QF*Model seam "
                  "(defs=%r)" % _defs3); bad += 1
        if any("qf_ok_modelpatcher.py" in x and "NO recognized" in x for x in _defs3):
            print("[FAIL] selftest: family-module arm FALSE-POSITIVED a module that has a seam"); bad += 1
    finally:
        _PLUGIN_DIR = _real_plugin_dir
        shutil.rmtree(_d3, ignore_errors=True)
    print("REJECT_LIST_SELFTEST:",
          ("PASS — comfy-absent run is loud ([SKIP]+exit %d); audit model-agnostic (LTXV+Krea2); derived-roster "
           "gate proven both ways (clean real dir → no defect; unaudited seam + vacuous scan + seamless family "
           "module → defect)" % _SKIP_EXIT)
          if bad == 0 else "FAIL (%d wrong)" % bad)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(_selftest())
    comfy_root = _resolve_comfy_root(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COMFY_ROOT", ""))
    if not comfy_root:
        sys.exit(_emit_skip("comfy root not found (pass it as argv[1] or set COMFY_ROOT)"))
    sys.exit(check(comfy_root))
