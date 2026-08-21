#!/usr/bin/env python3
"""Death-rule for the seq-219 / #B3 cfg_context_key contract (src/gemm/lighting/CLAUDE.md #B3).

The engine keys THREE per-step caches off the ABI cfg_context_key: the L2 memoization caches (ctx_cache_,
cross_kv_cache_) AND the STATEFUL L3 First-Block/TeaCache trajectory cache (fbcache_). The key must be
(a) DISTINCT per distinct conditioning (else L2 emits one branch's text for another — the seq-219 defect,
a 0/1 role index colliding two DIFFERENT-content conds a stock ConditioningCombine puts in one bucket);
(b) DISTINCT per branch even for IDENTICAL content (else the STATEFUL fbcache_ slot is shared → #B3, why a
CONTENT HASH is forbidden); and (c) STABLE for a conditioning across steps INCLUDING under a within-generation
COMPOSITION CHANGE (ConditioningSetTimestepRange drops a cond mid-run → a POSITION-derived key renumbers the
survivors → a later step HITs an earlier step's DIFFERENT-branch cross-KV = stale wrong reuse, since the L2
caches don't re-verify content on a hit).

The production key is `_CtxKeyAssigner`: it maps comfy's per-conditioning UUID (`transformer_options["uuids"]
[i]`, a fresh uuid4 per entry, stable across steps, position-independent) to a small NON-ZERO int; and when
comfy exposes NO uuid it returns 0 = the engine's kNoCtxKey sentinel, which DISABLES all three caches and
recomputes bit-exact (WanTransformerLighting.cpp `have_ctx_key`) — the sanctioned safe degradation, never a
poisonable position-derived reuse.

Self-contained + mutation-sensitive on the PRODUCTION `_CtxKeyAssigner` (AST-extracted — pure Python, no
comfy import, runs under a box torch ABI break). Inline mutants make each arm discriminate:
  • `_PosOnlyMutant` (the REJECTED positional scheme) SHIFTS a survivor's key on a drop → proving both why
    the uuid is needed (arm 3) and why the fallback must be 0, not positional (arm 5);
  • `_ConstMutant` (one key for all) FAILS seq-219 (arm 1).
The stock ConditioningCombine reachability arm runs with comfy and SKIPs (not fails) otherwise.

Run: python3 cfg_context_key_collision_test.py   (exit 0 = pass/skip; exit 1 = the contract is broken)
"""
import ast
import os
import sys
import uuid as _uuidmod

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.dirname(_HERE)
_SRC_PATH = os.path.join(_PLUGIN, "qf_modelpatcher.py")


def _read_src():
    with open(_SRC_PATH, encoding="utf-8") as fh:
        return fh.read()


def _load_assigner(src):
    """AST-extract `_CtxKeyAssigner` + the `_KNO_CTX_KEY` constant it references, exec standalone (pure
    Python — no comfy/torch). Runs the REAL on-disk code → a mutation to `.key()`/`.reset()`/the sentinel is
    caught."""
    tree = ast.parse(src)
    ns = {}
    for node in tree.body:                    # pull the module-level _KNO_CTX_KEY assignment first
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_KNO_CTX_KEY" for t in node.targets):
            exec(compile(ast.Module([node], []), "<_KNO_CTX_KEY>", "exec"), ns)   # noqa: S102
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_CtxKeyAssigner":
            seg = ast.get_source_segment(src, node)
            exec(compile(seg, "<_CtxKeyAssigner>", "exec"), ns)   # noqa: S102 — pure-Python class, isolated ns
            return ns["_CtxKeyAssigner"], ns.get("_KNO_CTX_KEY")
    raise RuntimeError("_CtxKeyAssigner not found in qf_modelpatcher.py — the fix was removed/renamed")


def _extract_callsite_key_derivation(src):
    """AST-extract the ACTUAL cfg_context_key derivation from the CALL SITE — the `cuid` and `ctx_key`
    assignments inside `_apply_model`. That site (not the _CtxKeyAssigner class body the other arms cover) is
    where this design has been rewritten five times, each rewrite fixing the prior iteration's real defect,
    so it is where a future content-derived key would be reintroduced (CR#4 / revA). Returns the two
    statements' source joined, ready to exec with a controlled namespace. Raises LOUDLY if the call-site
    derivation is gone (a structural change the test must fail on, not silently skip)."""
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_apply_model"), None)
    if fn is None:
        raise RuntimeError("_apply_model not found — the call site vanished")
    cuid_src = ctx_src = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "cuid":
                cuid_src = ast.get_source_segment(src, node)
            elif node.targets[0].id == "ctx_key":
                ctx_src = ast.get_source_segment(src, node)
    if not cuid_src or not ctx_src:
        raise RuntimeError("call-site cuid/ctx_key derivation not found in _apply_model — the site changed shape")
    return cuid_src + "\n" + ctx_src


def _mutate_callsite(callsite, override):
    """Insert an override of `cuid` (a MUTATION) right BEFORE the ctx_key assignment, so a negative control
    drives the REAL on-disk derivation (`_extract_callsite_key_derivation` output) with a KNOWN-BAD key input
    — wiring the control to the PRODUCTION rule, not a hard-coded fixture (F2). The mutated source recomputes
    ctx_key = key(<bad cuid>) exactly as production would, so the arm's stability check flips iff the arm is live."""
    lines = callsite.split("\n")
    j = next(k for k, ln in enumerate(lines) if ln.strip().startswith("ctx_key"))
    return "\n".join(lines[:j] + [override] + lines[j:])


# ---- INLINE mutants (the rejected schemes) — used to prove each arm discriminates ---------------------
class _PosOnlyMutant:
    """The REJECTED positional fallback (per-step ordinal) — SHIFTS a survivor's key on a drop = stale reuse."""
    def __init__(self): self._s = None; self._r = 0
    def key(self, cond_uuid, step_index):
        if step_index != self._s:
            self._s = step_index; self._r = 0
        self._r += 1
        return self._r


class _ConstMutant:
    """One key for everything (role-index-style collision)."""
    def key(self, cond_uuid): return 1


def _drive(assigner, steps):
    """Drive the PRODUCTION assigner (key(cond_uuid)) across sampler steps. steps = per-step list of uuids
    present that step (a DROPPED cond is absent). Returns [{uuid: key} per step]."""
    return [{u: assigner.key(u) for u in step} for step in steps]


def _drive_pos(mutant, steps):
    return [{u: mutant.key(u, si) for u in step} for si, step in enumerate(steps)]


def _find_comfy_root():
    cands = [os.environ.get("COMFY_ROOT", ""),
             os.path.dirname(os.path.dirname(os.path.dirname(_HERE))),
             "/root/qf-comfy-nat/ComfyUI", "/media/jonathan/Data/ComfyUI"]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "nodes.py")):
            return c
    p = os.getcwd()
    for _ in range(6):
        if os.path.isfile(os.path.join(p, "nodes.py")):
            return p
        p = os.path.dirname(p)
    return None


def _conditioning_combine_reachability():
    comfy_root = _find_comfy_root()
    if not comfy_root:
        return None
    sys.path.insert(0, comfy_root)
    try:
        import torch
        from nodes import ConditioningCombine
    except Exception:      # noqa: BLE001 — missing/ABI-broken comfy → SKIP
        return None
    torch.manual_seed(0)
    a = torch.randn(1, 77, 4096)
    b = torch.randn(1, 77, 4096)
    combined, = ConditioningCombine().combine([[a, {}]], [[b, {}]])
    return len(combined)


def _cou_ge2_reachability():
    """★ M6 REACHABILITY: cou>=2 is produced by the REAL comfy path, NOT the synthetic cou=[0,2,2] the M6
    key-derivation arms drive. A full live drive of calc_cond_batch needs a diffusion model (out of a unit
    test's scope, and comfy itself SKIPs here — see arm 9), so this pins the TWO structural facts the cou>=2
    reachability actually rests on to comfy's OWN source (an API rename breaks the arm instead of silently
    un-covering it — the discipline arm 9 uses for the cou<2 ConditioningCombine case):
      (a) BOTH real 3-cond guiders feed a 3-element cond list to comfy.samplers.calc_cond_batch — nested
          DualCFGGuider passes [negative_cond, middle_cond, positive_cond] and perp-neg's guider carries
          {positive, empty_negative_prompt, negative} — so a 3rd cond makes the cond_or_uncond ROLE index reach 2.
      (b) perp-neg's guider builds a 3-element cond list [positive_cond, negative_cond, empty_cond] fed to
          calc_cond_batch (nodes_perpneg.py) — same 3rd-cond → role index 2 reachability.
      (c) the aligned-append pair (cond_or_uncond.append(o[1]) AND uuids.append(p.uuid)) runs in the SAME
          loop iteration, so cond_or_uncond[i] and uuids[i] stay ALIGNED at cou>=2 (the invariant the per-uuid
          key derivation relies on — the M6 battery asserts the derivation is correct GIVEN that alignment;
          this arm shows the alignment is real). ★ comfy.samplers.calc_cond_batch is a thin context-handler
          DISPATCH wrapper (samplers.py:207); the real per-cond loop lives in the PRIVATE _calc_cond_batch
          (reached via _calc_cond_batch_outer's WrapperExecutor), so we inspect THAT symbol — and use the
          wrapper as a NEGATIVE CONTROL (the pair must be ABSENT there) so the substring check is proven
          SPECIFIC to the loop body, not boilerplate (this file's mutant convention). A comfy refactor that
          renames/moves the loop trips one of these → loud FAIL/SKIP, never a silent green.
    Returns (dualcfg_3cond, perpneg_3cond, cou_uuid_aligned, cou_check_discriminates) or None (SKIP)."""
    comfy_root = _find_comfy_root()
    if not comfy_root:
        return None
    sys.path.insert(0, comfy_root)
    try:
        import inspect
        import comfy.samplers as _cs
        import comfy_extras.nodes_custom_sampler as _ncs
        import comfy_extras.nodes_perpneg as _npn
        dualcfg_3cond = "[negative_cond, middle_cond, positive_cond]" in inspect.getsource(_ncs)
        perpneg_3cond = "[positive_cond, negative_cond, empty_cond]" in inspect.getsource(_npn)

        def _pair(s):   # the aligned cond_or_uncond/uuid append pair
            return "cond_or_uncond.append(o[1])" in s and "uuids.append(p.uuid)" in s
        loop_src = inspect.getsource(_cs._calc_cond_batch)   # PRIVATE fn: the real per-cond loop
        wrap_src = inspect.getsource(_cs.calc_cond_batch)    # PUBLIC dispatch wrapper: no loop
        cou_uuid_aligned = _pair(loop_src)
        cou_check_discriminates = not _pair(wrap_src)        # negative control (mutant convention)
    except Exception:      # noqa: BLE001 — missing/ABI-broken/renamed comfy → SKIP (mirrors arm 9)
        return None
    return (dualcfg_3cond, perpneg_3cond, cou_uuid_aligned, cou_check_discriminates)


def main():
    bad = 0
    src = _read_src()
    Assigner, kno = _load_assigner(src)
    if kno != 0:
        print(f"  [FAIL] _KNO_CTX_KEY is {kno!r}, must be 0 (the engine's kNoCtxKey 'caches off' sentinel)")
        bad += 1

    # ★ F1 PREMISE GUARD (behavioral, NOT a source-scan): the M5 non-affine OUT-OF-SCOPE judgment relies on the
    #   per-uuid key derivation being an UNBOUNDED ORDINAL — distinct uuids -> distinct, strictly-ordinal keys — so
    #   the engine's CtxCacheKey-keyed step cache never COLLIDES two distinct conds. Premise home (pinned):
    #   PLUGIN assigner qf_modelpatcher.py:133 · ENGINE cache src/gemm/lighting/lighting_step_cache.h:86-107 ·
    #   consumers WanTransformerLighting.h:227/450/469. If a future fixed-width/modulo key broke it, M5's scope
    #   would SILENTLY EXPIRE — so drive the PRODUCTION assigner with K=512 distinct uuids + assert distinct+ordinal.
    #   (The ENGINE-side cache-collision half is guarded, ctest-registered, by QuantFunc
    #   tests/cpp/test_step_cache_key.cpp — but at K=64 there (see its line 126, `for (i = 1; i <= 64; ...)`), NOT
    #   512: that is 8x the engine's own count-8 bound, yet a modulo/fixed-width period in [64, 512) ESCAPES the
    #   ENGINE ctest and is caught only by THIS plugin K=512 guard. Bumping that engine K 64->512 is the OPEN
    #   ledgered residual `engine-step-cache-ctest-k64-below-modulo-periods`, deferred to the next test-enabled
    #   main-repo build. Do NOT read this as closed.)
    _K = 512
    _pa = Assigner()
    _pk = [_pa.key(_uuidmod.uuid4()) for _ in range(_K)]
    if len(set(_pk)) == _K and _pk == list(range(1, _K + 1)):
        print(f"  [OK ] premise (M5 out-of-scope): {_K} distinct uuids -> {_K} PAIRWISE-DISTINCT, STRICTLY-ORDINAL "
              f"keys (1..{_K}) -> unbounded-ordinal per-uuid key; the non-affine M5 scope holds")
    else:
        print(f"  [FAIL] premise (M5 out-of-scope EXPIRED): {_K} distinct uuids did NOT yield {_K} distinct strictly-"
              f"ordinal keys (distinct={len(set(_pk))} head={_pk[:4]} tail={_pk[-3:]}) -> a bounded/modulo key COLLIDES "
              f"distinct conds and the cfg_context_key M5 non-affine out-of-scope judgment NO LONGER HOLDS")
        bad += 1

    uA, uB, uN = _uuidmod.uuid4(), _uuidmod.uuid4(), _uuidmod.uuid4()

    # (1) seq-219 FIX: distinct conditionings in one bucket get DISTINCT non-zero keys.
    prod = _drive(Assigner(), [[uA, uB, uN]])
    if len(set(prod[0].values())) == 3 and all(v != 0 for v in prod[0].values()):
        print(f"  [OK ] seq-219: distinct conditionings -> DISTINCT non-zero keys ({prod[0]})")
    else:
        print(f"  [FAIL] seq-219: a collision/zero ({prod[0]}) -> fix broken"); bad += 1
    m = [_ConstMutant().key(u) for u in (uA, uB, uN)]
    if len(set(m)) == 1:
        print("  [OK ] mutation: a constant/role-index key COLLIDES them (the defect the uuid fixes)")
    else:
        print("  [FAIL] the constant mutant did not collide"); bad += 1

    # (2) #B3: content-INDEPENDENT — distinct entries (even identical CONTENT) get distinct keys.
    if prod[0][uA] != prod[0][uB]:
        print("  [OK ] #B3: distinct-uuid entries get distinct keys regardless of content -> no shared fbcache_ slot")
    else:
        print("  [FAIL] #B3: two distinct entries collided"); bad += 1

    # (3) COMPOSITION-CHANGE STABILITY (uuid path): a cond is DROPPED mid-generation; survivors keep keys.
    prod2 = _drive(Assigner(), [[uA, uB, uN], [uB, uN]])
    if prod2[0][uB] == prod2[1][uB] and prod2[0][uN] == prod2[1][uN]:
        print(f"  [OK ] composition-change: survivors keep stable keys across a drop "
              f"(uB {prod2[0][uB]}=={prod2[1][uB]}, uN {prod2[0][uN]}=={prod2[1][uN]})")
    else:
        print(f"  [FAIL] composition-change: a survivor's key SHIFTED ({prod2}) -> stale cross-KV reuse"); bad += 1
    mp = _drive_pos(_PosOnlyMutant(), [[uA, uB, uN], [uB, uN]])
    if mp[0][uB] != mp[1][uB]:
        print(f"  [OK ] mutation: a position-only key SHIFTS a survivor on a drop (uB {mp[0][uB]}->{mp[1][uB]})"
              f" -> the reordering defect the uuid fixes")
    else:
        print("  [FAIL] the position-only mutant did not shift"); bad += 1

    # (4) FALLBACK = kNoCtxKey: no uuid -> 0 (engine disables its step caches, recompute bit-exact).
    if Assigner().key(None) == 0:
        print("  [OK ] fallback: uuid=None -> 0 (kNoCtxKey) -> engine disables all step caches (safe recompute)")
    else:
        print("  [FAIL] fallback did not return 0/kNoCtxKey"); bad += 1

    # (5) FALLBACK + COMPOSITION CHANGE (the required arm): with NO uuid the production must emit 0 for EVERY
    #     group so caching is OFF regardless of how the composition changes — never a position-derived
    #     non-zero key that reuses a dropped branch's slot. The rejected positional fallback does the latter.
    a = Assigner()
    fb_keys = [a.key(None) for _ in range(3)] + [a.key(None) for _ in range(2)]   # 3 groups then 2 (a drop)
    if all(k == 0 for k in fb_keys):
        print("  [OK ] fallback+composition-change: no-uuid -> ALL keys 0 (kNoCtxKey) -> caches OFF -> no reuse")
    else:
        print(f"  [FAIL] fallback emitted a NON-zero key ({fb_keys}) -> position-derived stale-reuse risk"); bad += 1
    if mp[0][uB] != mp[1][uB]:   # same _PosOnlyMutant run — it is the rejected positional fallback
        print("  [OK ] mutation: a position-derived fallback SHIFTS on a drop (the c0eb7f4 defect 0/kNoCtxKey avoids)")

    # (6) PER-GENERATION RESET: reset() restarts first-seen numbering + prevents cross-gen leak.
    a = Assigner(); a.key(uA); a.key(uB); a.reset()
    if a.key(uN) == 1:
        print("  [OK ] reset: per-generation reset restarts numbering at 1 (no cross-gen key leak)")
    else:
        print("  [FAIL] reset did not restart numbering"); bad += 1

    # (7) CALL-SITE BEHAVIOURAL arm — ALL rejected key schemes this churny site has historically risked (each
    #     reintroduced-and-reverted here) are covered by BEHAVIOUR through the ACTUAL on-disk _apply_model
    #     key-derivation SOURCE (rewritten 5x), NOT the _CtxKeyAssigner class body the arms above cover, and NOT
    #     the magic-string check in arm (8). The call-site locals a bad key would key on — `cou`
    #     (cond_or_uncond ROLE), `ci` (content), `i` (position) — are ALL provisioned in the exec namespace, so a
    #     retrofit onto any of them EXECS rather than silently NameError-passing. Provisioning is NECESSARY but NOT
    #     SUFFICIENT to CATCH it: the value must ALSO be EXERCISED to a NON-inert setting. (The a250211 generality
    #     NO-GO was exactly this gap — a role-conditional `cuuids[i] if cou[i]==0 else i` EXECS fine, but every arm
    #     ran cou ALL-ZERO so the `else i` branch never fired and it FALSE-GREENED.) So each family is driven to a
    #     setting that separates the mutant from the per-uuid key:
    #       A) SAME-ROLE COLLAPSE: two DISTINCT conds a ConditioningCombine puts in ONE role bucket (cou=[0,0])
    #          with IDENTICAL content MUST get DISTINCT keys. Catches a key derived from CONTENT (hash a228fee-
    #          era / arithmetic, any spelling), from the cond_or_uncond ROLE INDEX (the ORIGINAL a228fee
    #          `cou[i]+1` defect), OR from tensor IDENTITY (data_ptr/id, #329) — all THREE collapse two
    #          same-role identical-content conds; only comfy's per-ENTRY uuid keeps them distinct.
    #       B) COMPOSITION-CHANGE STABILITY: a survivor's key is STABLE when the cond list changes shape.
    #          B1/B2/B3 catch ANY affine i/len(cuuids) POSITION key (c0eb7f4 ordinal, both FORWARD `.key(i)` AND
    #          REVERSED `.key(len-1-i)`) via the invariance property proven at that block; A cannot (in one pass
    #          distinct positions are distinct keys, so A never sees the renumbering). The ROLE-CONDITIONAL family
    #          (`cuuids[i] if cou[i]==0 else K(i,len)`) has the SAME >=2 affine DOFs, so it needs the FULL battery
    #          driven at cou!=0: A-uncond (collapse, cou=[0,1,1]) + B4 (forward) + B2-uncond (reversed) close ALL
    #          affine i/len keys for the UNCOND role too — exactly as A + B1 + B2 do for cond. That closes the
    #          a250211 NO-GO and its whole AFFINE family (K=a*i+b*len+c per role; non-affine is OUT OF SCOPE
    #          by the M5 reachability argument at the ★ SCOPE note below), not just the literal `else i` spelling.
    callsite = _extract_callsite_key_derivation(src)   # loud-fails if the call-site derivation vanished
    try:
        import torch
    except ImportError:
        print("  [SKIP] call-site behavioural arm (torch not importable here — run in a torch env)")
    else:
        import hashlib as _hashlib
        uX, uY = _uuidmod.uuid4(), _uuidmod.uuid4()               # DISTINCT conditioning identities
        uZ = _uuidmod.uuid4()                                     # a 2nd cond for the REAL cou=[0,1] uncond-role arm (B4)
        uW = _uuidmod.uuid4()                                     # a cou=2 role (perp-neg / 3-cond DualCFGGuider) for M6 (cou>=2)
        uW2 = _uuidmod.uuid4()                                    # a 2nd cou=2 role for the M6 same-role-collapse arm
        uP, uQ, uR = _uuidmod.uuid4(), _uuidmod.uuid4(), _uuidmod.uuid4()   # 3-cond list for the middle-drop scenario
        torch.manual_seed(0)
        # `ci`/`cou`/`os`/`step_index` are forward-provisioned: HEAD's extracted derivation reads only `cuid`, so they are
        # inert on HEAD, but a retrofit onto content (`ci`), role (`cou`) or the os module must EXEC and hit an
        # assertion — not a NameError (an accidental catch that a future namespace widening would silently void).
        # RESIDUAL (generality-CR, f963c1f): step_index is now provisioned + closed by M7; but a retrofit onto
        # ANY OTHER non-provisioned name (a fresh `self.`-attribute, a new global) fails CLOSED via
        # NameError/AttributeError -> nonzero exit (CI catches it) but WITHOUT a clean [FAIL]/summary line -- it
        # is loud-but-undiagnosed, NOT a silent green. The provisioned set + M6/M7 cover the key-forming inputs
        # reachable in this seam identified so far; a genuinely new one must be added here (provision + arm).
        shared_ci = torch.randn(1, 77, 4096)                     # IDENTICAL content fed to every group

        def _run_callsite(stub, cuuids, cou, idx, step=0, _src=None):
            ns = {"self": stub, "cuuids": cuuids, "cou": cou, "ci": shared_ci, "i": idx, "step_index": step,
                  "os": os, "torch": torch, "hashlib": _hashlib}
            exec(_src if _src is not None else callsite, ns)   # noqa: S102 — the REAL on-disk derivation (or, for
            return ns["ctx_key"]                               #   the F2 negative controls, a _mutate_callsite of it)

        class _CallSiteStub:
            def __init__(self): self._ctx_key_assigner = Assigner()

        # A) SAME-ROLE COLLAPSE: two distinct conds in ONE role bucket (cou=[0,0]), identical content -> distinct
        #    keys. A content-, role-index- (cou[i]), or identity-derived key collides them; only the uuid doesn't.
        sA = _CallSiteStub()
        a0 = _run_callsite(sA, [uX, uY], [0, 0], 0)
        a1 = _run_callsite(sA, [uX, uY], [0, 0], 1)
        okA = (a0 != a1)
        # B) COMPOSITION-CHANGE STABILITY — a survivor's key MUST be invariant when the conditioning list changes
        #    shape mid-generation. A position/ordinal key renumbers survivors on a drop; a per-uuid key does not.
        #    ★ INVARIANCE PROPERTY this probe set enforces (EXTEND it so the PROPERTY still holds — do NOT just
        #    bolt on one more spelling): NO single affine transform K = a*i + b*len(cuuids) + c of the call-site
        #    position may stay invariant across the whole probe set; only a true per-uuid key survives.
        #    Why these scenarios close it — K passes through the injective per-gen assigner, so a survivor's OUTPUT
        #    keys are equal iff its K INPUTS are, and for a survivor seen at P1=(i1,len1) and P2=(i2,len2) the
        #    mutant is CAUGHT iff K(P1)!=K(P2) i.e. a*(i1-i2) + b*(len1-len2) != 0:
        #      * B1 drop-FIRST (uY: (1,2)->(0,1), delta (-1,-1)) catches every affine with a+b != 0
        #        -- incl. the FORWARD ordinal .key(i)        [a=1,b=0 -> a+b=1].
        #      * B2 drop-LAST  (uX: (0,2)->(0,1), delta ( 0,-1)) catches every affine with b   != 0
        #        -- incl. the REVERSED ordinal .key(len-1-i) [a=-1,b=1 -> a+b=0 ESCAPES B1, but b=1 here].
        #        (B1 ALONE silently false-greened the reversed ordinal: len-1-i == 0 at BOTH (1,2) and (0,1).)
        #      * the ONLY affine escaping both is a=b=0 -- a CONSTANT key -> caught by sub-check A (same-role collapse).
        #    B1 + B2 + A therefore close ALL AFFINE i/len position keys in BOTH directions. B3 (3-cond middle-drop,
        #    len 3->2) adds a different length regime (length-conditional keys).
        #    ★ SCOPE (M5, generality-CR): the closure is of the AFFINE family K=a*i+b*len(cuuids)+c per role; it
        #    does NOT close a NON-AFFINE key (e.g. `else i*i`). DELIBERATE scope, not a gap: the real derivation is
        #    per-uuid, and the REACHABLE bug class is affine substitution of position/role/len (a renumber, a
        #    role-index, a constant). No realistic cache-key derivation computes a NONLINEAR function of the batch
        #    position, so a non-affine role-conditional key is adversarially CONSTRUCTIBLE but NOT REACHABLE; every
        #    'closes' claim here is scoped to 'affine'. TWO axes ORTHOGONAL to the cond/uncond (i,len) closure are
        #    ALSO reachable and closed by their own arms: the cou>=2 role-conditional family (perp-neg /
        #    DualCFGGuider — this reachability is DEMONSTRATED against comfy's own source by arm (10), not merely
        #    asserted here; the battery below then proves the DERIVATION is correct at that reached cou>=2) ->
        #    the M6 COLLAPSE+FORWARD+REVERSED battery below (a single M6 forward probe
        #    false-greened a reversed mutant -- self-CR Reviewer B -- so cou>=2 needs the SAME 3-arm battery as
        #    cond/uncond); the denoise step_index (in-scope at the derivation site, so a `key((cuid,step))` is
        #    reachable) -> M7 below. So the FULL enumeration of key-forming inputs the arms close is
        #    uuid / position / role / length / cou>=2 / step_index.)
        # B1) drop-FIRST: uY at pos 1 (len 2) then pos 0 (len 1). FORWARD ordinal .key(i) shifts it 2->1.
        sB = _CallSiteStub()
        _run_callsite(sB, [uX, uY], [0, 0], 0)                    # uX at pos 0 (primes the assigner)
        b_uY0 = _run_callsite(sB, [uX, uY], [0, 0], 1)           # uY at pos 1 (len 2)
        b_uY1 = _run_callsite(sB, [uY], [0], 0)                   # uY at pos 0 (len 1) -- uX dropped
        okB_first = (b_uY0 == b_uY1)
        # B2) drop-LAST: uX at pos 0 in BOTH, len 2->1. REVERSED ordinal .key(len-1-i) shifts it 1->0.
        sC = _CallSiteStub()
        c_uX0 = _run_callsite(sC, [uX, uY], [0, 0], 0)           # uX at pos 0 (len 2)
        c_uX1 = _run_callsite(sC, [uX], [0], 0)                   # uX at pos 0 (len 1) -- uY dropped
        okB_last = (c_uX0 == c_uX1)
        # B3) middle-drop (len 3->2): uR survives, uQ removed. Hardens the length regime (len-conditional ONLY; NOT
        #     a general non-affine closure — see the ★ SCOPE (M5) note above).
        sD = _CallSiteStub()
        d_uR0 = _run_callsite(sD, [uP, uQ, uR], [0, 0, 0], 2)     # uR at pos 2 (len 3)
        d_uR1 = _run_callsite(sD, [uP, uR], [0, 0], 1)           # uR at pos 1 (len 2) -- uQ dropped
        okB_mid = (d_uR0 == d_uR1)
        # B4) UNCOND-ROLE stability at the REAL cou=[0,1] shape comfy produces by default (cond=0, uncond=1).
        #     B1/B2/B3 (and A) all run cou ALL-ZERO, so a ROLE-CONDITIONAL position key
        #     `cuuids[i] if cou[i]==0 else i` (uuid for cond, POSITION for uncond) is INERT there — cou[i]==0
        #     always picks the uuid branch — and FALSE-GREENS (the a250211 generality NO-GO). Here uY is the
        #     UNCOND (cou=1) and a ConditioningCombine adds a 2nd cond (uZ), moving uY from pos 1 (cou=[0,1]) to
        #     pos 2 (cou=[0,0,1]); its ROLE never changes. A per-uuid key is STABLE across the two; the hybrid
        #     keys uY on its POSITION (the cou==1 branch) -> 1 vs 2 -> renumbered -> CAUGHT. B4 is the uncond
        #     B1-ANALOG (forward ordinal, delta (i,len)=(+1,+1)); ON ITS OWN it closes only that sub-case (the
        #     a250211 literal). The COLLAPSE and REVERSED-ordinal sub-cases of the role-conditional family are
        #     closed by A-uncond + B2-uncond below — the family needs the full A+B1+B2 battery PER ROLE (the same
        #     ≥2 affine degrees of freedom that made B2 necessary in addition to B1 for the cond role).
        sE = _CallSiteStub()
        _run_callsite(sE, [uX, uY], [0, 1], 0)                    # uX cond pos 0 (primes)
        e_uY0 = _run_callsite(sE, [uX, uY], [0, 1], 1)           # uY UNCOND at pos 1, cou=1 (len 2)
        _run_callsite(sE, [uX, uZ, uY], [0, 0, 1], 0)           # uX cond pos 0
        _run_callsite(sE, [uX, uZ, uY], [0, 0, 1], 1)           # uZ cond pos 1 (the added cond)
        e_uY1 = _run_callsite(sE, [uX, uZ, uY], [0, 0, 1], 2)   # uY UNCOND at pos 2, cou=1 (len 3)
        okB_uncond = (e_uY0 == e_uY1)
        # A-uncond) SAME-ROLE COLLAPSE on the UNCOND side (the cou==1 mirror of arm A). Arm A only ever drives
        #     cou=[0,0], so a key that COLLAPSES two DISTINCT uncond entries — a role-index `cou[i]` (constant 1)
        #     or a constant-for-uncond `... else <const>` — is invisible to A/B1/B2/B3/B4. A ConditioningCombine on
        #     the NEGATIVE prompt (mirror of seq-219's positive-side example) yields cou=[0,1,1]; the two DISTINCT
        #     unconds MUST get DISTINCT keys. Real per-uuid: distinct; an uncond-collapse key: SAME -> CAUGHT.
        sAu = _CallSiteStub()
        _run_callsite(sAu, [uP, uX, uY], [0, 1, 1], 0)          # uP cond pos 0 (primes)
        au0 = _run_callsite(sAu, [uP, uX, uY], [0, 1, 1], 1)   # uX UNCOND pos 1
        au1 = _run_callsite(sAu, [uP, uX, uY], [0, 1, 1], 2)   # uY UNCOND pos 2 (DISTINCT uuid, SAME uncond role)
        okA_uncond = (au0 != au1)
        # B2-uncond) REVERSED ordinal on the UNCOND side (the cou==1 mirror of B2). B4 moves uY with delta
        #     (i,len)=(+1,+1) (catches a+b!=0) but MISSES the reversed ordinal len-1-i (a=-1,b=1 -> a+b=0), which
        #     stays constant while uY is last. Here uY is UNCOND at a FIXED index 0 while a cond is APPENDED after
        #     it (len 2->3): delta (0,+1) -> len-1-i shifts 1->2 -> CAUGHT. A-uncond + B4 + B2-uncond thus close
        #     ALL affine i/len keys for the uncond role, exactly as A + B1 + B2 do for the cond role.
        sF = _CallSiteStub()
        _run_callsite(sF, [uY, uX], [1, 0], 1)                    # uX cond pos 1 (primes)
        f_uY0 = _run_callsite(sF, [uY, uX], [1, 0], 0)           # uY UNCOND at pos 0, cou=1 (len 2)
        _run_callsite(sF, [uY, uX, uZ], [1, 0, 0], 1)           # uX cond pos 1
        _run_callsite(sF, [uY, uX, uZ], [1, 0, 0], 2)           # uZ cond pos 2 (appended AFTER uY)
        f_uY1 = _run_callsite(sF, [uY, uX, uZ], [1, 0, 0], 0)   # uY UNCOND at pos 0, cou=1 (len 3)
        okB_uncond_rev = (f_uY0 == f_uY1)
        # M6) COU>=2 FULL BATTERY -- perp-neg (nodes_perpneg.py calc_cond_batch(pos, neg, empty)) + the 3-cond
        #     DualCFGGuider (nodes_custom_sampler.py) produce cond_or_uncond values BEYOND {0,1} (samplers.py:300
        #     cond_or_uncond.append(o[1]) in lockstep with uuids.append(p.uuid)). Every (i,len,cou in {0,1}) arm
        #     above is INERT for a key that BRANCHES on cou>=2. The cou>=2 role-conditional family has the SAME
        #     >=2 affine DOFs as cond/uncond, so it needs the FULL battery -- self-CR Reviewer B mechanically
        #     FALSE-GREENED a single forward probe with a `cuid if cou[i]<2 else len-1-i` REVERSED mutant (the
        #     exact one-probe-per-class trap this file's cond/uncond A+B1+B2 math already closes):
        #       M6-COLLAPSE -- two DISTINCT cou=2 conds MUST get DISTINCT keys (a cou-role / `else <const>` key
        #                      COLLAPSES them; only per-uuid keeps them apart).
        #       M6-FORWARD  -- uW cou=2 renumbered pos 2->3 (delta +1): catches every affine with a+b != 0.
        #       M6-REVERSED -- uW cou=2 held at idx 0 while len 2->3 (delta (0,+1)): catches len-1-i (a+b==0,b!=0).
        #     The ONLY affine escaping FORWARD+REVERSED is a=b=0 (a CONSTANT cou>=2 key) -> caught by COLLAPSE.
        sM6f = _CallSiteStub()          # M6-COLLAPSE
        _run_callsite(sM6f, [uP, uW, uW2], [0, 2, 2], 0)           # uP cond pos 0 (primes)
        m6c_0 = _run_callsite(sM6f, [uP, uW, uW2], [0, 2, 2], 1)   # uW  cou=2 pos 1
        m6c_1 = _run_callsite(sM6f, [uP, uW, uW2], [0, 2, 2], 2)   # uW2 cou=2 pos 2 (DISTINCT uuid, SAME cou=2 role)
        okM6_collapse = (m6c_0 != m6c_1)
        sM6 = _CallSiteStub()           # M6-FORWARD (a prepended cond renumbers uW pos 2->3)
        _run_callsite(sM6, [uX, uY, uW], [0, 1, 2], 0)              # uX cond pos 0 (primes)
        _run_callsite(sM6, [uX, uY, uW], [0, 1, 2], 1)              # uY pos 1
        m6_uW0 = _run_callsite(sM6, [uX, uY, uW], [0, 1, 2], 2)     # uW cou=2 at pos 2 (len 3)
        _run_callsite(sM6, [uZ, uX, uY, uW], [0, 0, 1, 2], 0)      # uZ cond PREPENDED at pos 0
        _run_callsite(sM6, [uZ, uX, uY, uW], [0, 0, 1, 2], 1)      # uX pos 1
        _run_callsite(sM6, [uZ, uX, uY, uW], [0, 0, 1, 2], 2)      # uY pos 2
        m6_uW1 = _run_callsite(sM6, [uZ, uX, uY, uW], [0, 0, 1, 2], 3)  # uW cou=2 at pos 3 (len 4) -- renumbered
        okM6_fwd = (m6_uW0 == m6_uW1)
        sM6r = _CallSiteStub()          # M6-REVERSED (uW cou=2 held at idx 0, a cond APPENDED after it, len 2->3)
        _run_callsite(sM6r, [uW, uX], [2, 0], 1)                   # uX cond pos 1 (primes)
        m6r_0 = _run_callsite(sM6r, [uW, uX], [2, 0], 0)           # uW cou=2 at pos 0 (len 2)
        _run_callsite(sM6r, [uW, uX, uZ], [2, 0, 0], 1)           # uX cond pos 1
        _run_callsite(sM6r, [uW, uX, uZ], [2, 0, 0], 2)           # uZ cond APPENDED at pos 2
        m6r_1 = _run_callsite(sM6r, [uW, uX, uZ], [2, 0, 0], 0)   # uW cou=2 at pos 0 (len 3) -- len-1-i shifts 1->2
        okM6_reversed = (m6r_0 == m6r_1)
        okM6 = okM6_collapse and okM6_fwd and okM6_reversed
        # M7) STEP-INDEX INDEPENDENCE (Finding #2): step_index (DenoiseStepParams.step_index, sigma-derived) is
        #     IN SCOPE at the derivation site (the QF_NATIVE_DEBUG_CTXKEY debug print reads it) but is NOT a
        #     cfg_context_key input: the key is per-uuid + step-STABLE (the engine composes (ctx_key, step) itself;
        #     a step-VARYING ctx_key would defeat that step cache). A `key((cuid, step_index))` retrofit is REACHABLE
        #     (step_index sits right there in the same loop) and would make the SAME cond's key vary per step; the
        #     (i,len,cou) arms above never varied step so they would MISS it, and the M5 enumeration omitted it. Here
        #     uX (same cond, same pos/role/len) is queried at step 0 and step 7; the per-uuid key MUST be identical.
        #     A step-keyed mutant: different -> CAUGHT. (step_index is now forward-provisioned, so the retrofit EXECs
        #     and is caught by THIS arm instead of NameError-ing.)
        sM7 = _CallSiteStub()
        m7_s0 = _run_callsite(sM7, [uX, uY], [0, 0], 0, step=0)
        m7_s1 = _run_callsite(sM7, [uX, uY], [0, 0], 0, step=7)     # SAME cond (uX pos 0, cou 0, len 2), different STEP
        okM7 = (m7_s0 == m7_s1)
        # ★ NEGATIVE CONTROLS (F2, generality-CR + the arm-10 lesson): every arm above proves the REAL per-uuid
        #   derivation PASSES; these prove the M6-REVERSED / B4 arms DISCRIMINATE — by MUTATING the REAL on-disk
        #   derivation (a _mutate_callsite override of _extract_callsite_key_derivation's output, driven through the
        #   SAME assigner + scenario, NOT a hard-coded fixture) and asserting the SAME arm CATCHES it. An earlier
        #   fixture form of these controls called a stand-in with literals and never touched the production rule -> it
        #   PROVED NOTHING while looking like it did (F2; the next stage of arm-10's wrong-symbol miss). Wired now.
        m6_mut = _mutate_callsite(callsite, 'if cou[i] >= 2: cuid = "m6revcou:%d" % (len(cuuids) - 1 - i)')
        m5_mut = _mutate_callsite(callsite, 'if cou[i] == 1: cuid = "m5nonaff:%d" % (i * i)')
        # M6-NEG: the reversed-cou MUTATION of the real derivation SHIFTS uW across the M6-REVERSED scenario
        #   (uW cou=2 held at idx 0, len 2->3) -> the M6-REVERSED STABILITY arm CATCHES it (m6n_0 != m6n_1). A live
        #   arm flips; an inert arm would not -> okM6_neg would go False -> the suite FAILs.
        sM6n = _CallSiteStub()
        _run_callsite(sM6n, [uW, uX], [2, 0], 1, _src=m6_mut)                 # uX cond pos 1 (primes)
        m6n_0 = _run_callsite(sM6n, [uW, uX], [2, 0], 0, _src=m6_mut)         # uW cou=2 idx 0 (len 2)
        _run_callsite(sM6n, [uW, uX, uZ], [2, 0, 0], 1, _src=m6_mut)
        _run_callsite(sM6n, [uW, uX, uZ], [2, 0, 0], 2, _src=m6_mut)
        m6n_1 = _run_callsite(sM6n, [uW, uX, uZ], [2, 0, 0], 0, _src=m6_mut)  # uW cou=2 idx 0 (len 3) -- MUTANT shifts
        okM6_neg = (m6n_0 != m6n_1)   # True = the REAL M6-REVERSED arm CATCHES the real-derivation mutation
        # M5-NEG: the NON-AFFINE (i*i) MUTATION of the real derivation SHIFTS uY across B4 (uY uncond pos 1->2:
        #   i*i 1->4) -> B4 CATCHES it. EVIDENCE for the ★ M5 SCOPE note: the arms catch REACHABLE non-affine role
        #   keys (any whose value moves with a reachable position change); only an adversarial key i-invariant across
        #   exactly these deltas yet non-affine elsewhere escapes = OUT OF SCOPE (no realistic cache-key derivation is
        #   nonlinear in batch position) -> the surviving M5 mutant is a scoped choice, not a blind spot.
        sM5n = _CallSiteStub()
        _run_callsite(sM5n, [uX, uY], [0, 1], 0, _src=m5_mut)                 # uX cond pos 0 (primes)
        m5n_0 = _run_callsite(sM5n, [uX, uY], [0, 1], 1, _src=m5_mut)         # uY uncond pos 1 (i*i=1)
        _run_callsite(sM5n, [uX, uZ, uY], [0, 0, 1], 0, _src=m5_mut)
        _run_callsite(sM5n, [uX, uZ, uY], [0, 0, 1], 1, _src=m5_mut)
        m5n_1 = _run_callsite(sM5n, [uX, uZ, uY], [0, 0, 1], 2, _src=m5_mut)  # uY uncond pos 2 (i*i=4) -- MUTANT shifts
        okM5_neg = (m5n_0 != m5n_1)   # True = the REAL B4 arm CATCHES the real-derivation non-affine mutation
        okB = okB_first and okB_last and okB_mid and okB_uncond and okB_uncond_rev and okM6 and okM7
        if okA and okA_uncond and okB and okM6_neg and okM5_neg:
            print(f"  [OK ] call-site: same-role distinct conds->DISTINCT keys (cond {a0}!={a1}, uncond {au0}!={au1}) "
                  f"AND survivors STABLE across drop-first ({b_uY0}=={b_uY1}), drop-last ({c_uX0}=={c_uX1}), "
                  f"mid-drop ({d_uR0}=={d_uR1}), uncond-fwd@cou=1 ({e_uY0}=={e_uY1}), uncond-rev@cou=1 ({f_uY0}=={f_uY1}), "
                  f"cou>=2@M6[collapse {m6c_0}!={m6c_1}, fwd {m6_uW0}=={m6_uW1}, rev {m6r_0}=={m6r_1}], step-indep@M7 ({m7_s0}=={m7_s1}) "
                  f"-> per-uuid key only: neither content-, role-index-, identity-, nor ANY affine i/len position "
                  f"key in EITHER role, stable at cou in {{0,1,>=2}} AND across step_index (A+B1+B2 affine closure "
                  f"BOTH roles + M6 cou>=2 + M7 step-independence); NEG-CONTROLS: a MUTATION of the REAL derivation "
                  f"is CAUGHT by M6-REVERSED ({m6n_0}!={m6n_1}) + B4 ({m5n_0}!={m5n_1}) -- wired to production, not a fixture")
        else:
            print(f"  [FAIL] call-site: same-role okA={okA} (cond {a0},{a1}) okA_uncond={okA_uncond} (uncond {au0},{au1}); "
                  f"composition okB={okB} [drop-first uY {b_uY0}->{b_uY1}, drop-last uX {c_uX0}->{c_uX1}, "
                  f"mid-drop uR {d_uR0}->{d_uR1}, uncond-fwd uY {e_uY0}->{e_uY1}, uncond-rev uY {f_uY0}->{f_uY1}, "
                  f"cou>=2@M6[collapse {m6c_0},{m6c_1}, fwd {m6_uW0}->{m6_uW1}, rev {m6r_0}->{m6r_1}], step-indep@M7 uX {m7_s0}->{m7_s1}] "
                  f"-> a content-/role-index-/identity- OR affine i/len position key, OR a cou>=2-branch key, OR a "
                  f"step_index-keyed key, in some role; NEG-CONTROLS okM6_neg={okM6_neg} okM5_neg={okM5_neg} (False = an arm went INERT)")
            bad += 1

    # (8) WIRING (a cheap SYNTACTIC tripwire — NOT the content-hash coverage; arm (7) covers that by behaviour).
    wired = ('transformer_options.get("uuids")' in src and "self._ctx_key_assigner.key(" in src)
    no_hash = ("content_ctx_key" not in src) and ("hashlib" not in src)
    reset_wired = "self._ctx_key_assigner.reset()" in src
    kno_wired = "_KNO_CTX_KEY" in src
    if wired and no_hash and reset_wired and kno_wired:
        print("  [OK ] wiring: _apply_model reads uuids + keys via the assigner; per-gen reset + kNoCtxKey present; no content hash")
    else:
        print(f"  [FAIL] wiring: uuid={wired} no_hash={no_hash} reset={reset_wired} kno={kno_wired}"); bad += 1

    # (9) REACHABILITY: stock ConditioningCombine really puts 2 conds in ONE bucket (SKIP without comfy).
    reach = _conditioning_combine_reachability()
    if reach is None:
        print("  [SKIP] ConditioningCombine reachability (comfy/torch not importable here — run in a ComfyUI env)")
    elif reach == 2:
        print("  [OK ] reachability: stock ConditioningCombine yields 2 entries in ONE bucket (seq-219 scenario is real)")
    else:
        print(f"  [FAIL] ConditioningCombine yielded {reach} entries, expected 2 (comfy API changed?)"); bad += 1

    # (10) ★ M6 cou>=2 REACHABILITY (companion to the M6 key-derivation battery above, which drives a SYNTHETIC
    #      cou=[0,2,2]): show cou>=2 is a REAL comfy path, not an adversarial construction — DualCFGGuider +
    #      perp-neg both feed a 3-cond list to calc_cond_batch (3rd cond -> cou role index 2), and calc_cond_batch
    #      keeps cond_or_uncond[i]/uuids[i] aligned in one loop. Pinned to comfy's source (arm 9's discipline for
    #      cou<2), SKIP without comfy. This closes the M6 gap: the battery proved the DERIVATION is correct at
    #      cou>=2; this proves that cou>=2 is REACHED (so the battery guards a real scenario, not a synthetic one).
    r_cou2 = _cou_ge2_reachability()
    if r_cou2 is None:
        print("  [SKIP] cou>=2 reachability (comfy not importable here — run in a ComfyUI env)")
    elif r_cou2 == (True, True, True, True):
        print("  [OK ] reachability: DualCFGGuider + perp-neg both feed a 3-cond list to calc_cond_batch (cond_or_uncond "
              "role reaches >=2); cond_or_uncond[i]/uuids[i] append in ONE loop of _calc_cond_batch (aligned), and the "
              "check discriminates (pair absent from the calc_cond_batch dispatch wrapper) -> M6's cou>=2 is REACHABLE, not synthetic")
    else:
        print(f"  [FAIL] cou>=2 reachability: dualcfg_3cond={r_cou2[0]} perpneg_3cond={r_cou2[1]} "
              f"cou_uuid_aligned={r_cou2[2]} cou_check_discriminates={r_cou2[3]} (comfy API changed?)"); bad += 1

    print("CFG_CONTEXT_KEY_COLLISION:", "PASS" if bad == 0 else f"FAIL ({bad} wrong)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
