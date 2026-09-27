"""qf_native.qf_ltx_modelpatcher — the LTX-2 seam: a comfy ModelPatcher whose `.model` is an
LTXV-shim (disable_unet) that a STOCK KSampler drives, forwarding through the QuantFunc engine's
LTX-2 external denoise SESSION (quantfunc_denoise_begin/step/finalize/end, the SAME generic ABI the
wan seam uses — reused from qf_engine.py).

WHY this is DIFFERENT from the wan seam (design findings, dossier seq-229..234):
- cfg_context_key: seq-229 established the LTX-2 lighting transformer engages NONE of the
  key-trusting step caches (ctx_cache_/cross_kv_cache_/the engine's block-cache state slot ) — historically every step passed
  cfg_context_key=0 (kNoCtxKey). SINCE the step-cache session gate (merge 516a6d78) the key IS
  consumed (one EcEntry per cond branch; key=0 force-computes), so BOTH step loops (t2v + AV) now
  derive uuid-symbolic keys via the shared _CtxKeyAssigner (wan #B3 pattern) — see [step-cache-key]
  comments at the loops.
- PACKED latent. The engine session denoises a packed token latent [1,N,128] (N=F_lat*H_lat*W_lat),
  NOT comfy's 5D [1,128,F,H,W]. seq-230: the mapping is a plain REVERSIBLE reshape+transpose (row-major
  (F,H,W)), the exact inverse of the engine's pack_video_tokens — nothing inferred.
- CONNECTOR BRIDGE. seq-231/234: for LTX-2.3-22B comfy's TE uses `dual_linear` and returns the
  PRE-connector `unprocessed_ltxav_embeds` [*,S,6144] (video 4096 | audio 2048); the connector runs in
  comfy's diffusion MODEL (av_model.preprocess_text_embeds), which this seam BYPASSES. The engine
  session expects POST-connector video_embeds [1,S,4096]. So this seam RUNS the connector itself:
  instantiate comfy's Embeddings1DConnector (av_model config: 32 heads x 128 = inner_dim 4096, 2 layers,
  split_rope, double_precision_rope, caption_proj_before_connector=False) + load
  model.diffusion_model.video_embeddings_connector.* from the comfy LTX-2.3 checkpoint, and run
  `video_embeds = connector(c_crossattn[:, :, :4096])[0]` before packing. DETECTOR B (dossier seq-234)
  is the decisive gate: engine generate_video (its OWN connector) vs plugin->engine (comfy's connector),
  same + SHORT prompt, image compare.
- VIDEO-ONLY. The engine REFUSES a joint-AV checkpoint external session (has_audio_) — the engine side
  must load a VIDEO-ONLY svdq checkpoint; the comfy TE still runs dual/AV for the conditioning.

Landing HELD (build+verify only). Only THIS loader node is swapped into an otherwise-STOCK LTX-2.3 T2V
workflow.
"""
import ctypes
import time
import json
import math
import os

import torch

import comfy.model_base
import comfy.conds
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import comfy.nested_tensor

from . import qf_engine as qfe
# Reuse the shared model-layer helpers. NOTE: the engine cache (_get_engine + the liveness
# tracker) lives in __init__.py (which imports THIS file), so those are threaded into register()
# via deps to avoid a circular import — same seam as every family module.
from . import qf_modelpatcher as qfmp
from .qf_modelpatcher import (_qf_dtype, _QFStub,
                              _interrupt_poll_end_session_on_raise,
                              QFSessionModelMixin)


# ── LTX-2.3-22B connector config (av_model.py video connector; dossier seq-234) ─────────────────────
# inner_dim = 32*128 = 4096 (NOT the Embeddings1DConnector default 30*128=3840); 2 layers; the video
# slice of the dual TE output is [:, :, :_LTX_VIDEO_DIM].
_LTX_VIDEO_DIM = 4096          # cross_attention_dim (video) - the connector inner_dim + the video slice width
# The connector ARCH (num_layers / num_heads / head_dim / num_registers / inner_dim) is DERIVED from the
# checkpoint by _derive_connector_arch (Finding #1) -- nothing below is a hardcoded arch dim.
# ★ SECURITY BOUNDS (vuln-CR + self-CR Reviewer A): EVERY value derived from the untrusted `connector_ckpt`
# string node input that feeds Embeddings1DConnector's EAGER __init__ (blocks/Linears/registers built BEFORE
# load_state_dict, so the 0/0 load guard CANNOT intercept it) is magnitude-bounded AT the derivation site. A
# stray huge transformer_1d_blocks.<idx> / learnable_registers width / to_gate_logits.bias length in a
# corrupt/hostile checkpoint would otherwise drive a host-RAM-overshoot SIGKILL (#543 class, unguardable after
# the alloc; block Linears are O(inner_dim^2)). The old hardcoded dims were accidentally ALSO bounds; deriving
# from untrusted input replaces a constant with a control surface, so the bound lives here. REJECT (not clamp)
# so a genuinely-larger future connector fails visibly. Real LTX: depth-8, inner 4096/2048, 32 heads, 128 regs.
_MAX_CONNECTOR_LAYERS = 16          # 2x the real depth-8
_MAX_CONNECTOR_INNER_DIM = 8192     # 2x the real video inner 4096 (block+FFN Linears are O(inner_dim^2))
_MAX_CONNECTOR_HEADS = 64           # 2x the real 32 (to_gate_logits weight is [num_heads, inner_dim])
_MAX_CONNECTOR_REGISTERS = 256      # 2x the real 128 (learnable_registers is [num_registers, inner_dim])
# ★ PER-LOAD FOOTPRINT CEILING (native-connector-combined-footprint-ceiling): the 4 per-axis bounds above are each
#   satisfiable while their PRODUCT is not, AND load() holds MORE THAN ONE connector at once. Two levels of "bounded
#   != bounded":
#     (1) per-axis bounded != PRODUCT bounded -- at simultaneous max (16 layers x 8192^2 x 64 heads x 256 regs) a
#         SINGLE connector's EAGER bf16 alloc is ~24 GiB (connector_arch_derivation_test.py:145 "all-axes-max").
#     (2) per-CONNECTOR bounded != per-LOAD bounded -- load() builds the video connector AND (join_audio_prompt=True,
#         a shipped switch) the audio connector CONCURRENTLY, so two <=8 GiB connectors = ~16 GiB COMBINED.
#   On the GPU path either surfaces as a CATCHABLE torch.cuda.OutOfMemoryError (comfy's node except -> workflow
#   fails); on the CPU/no-GPU dev+test path it is a HOST-RAM SIGKILL (#543 class: NO exception, no catch reaches it,
#   the process silently vanishes) -- a DISTINCT failure class. So bound BOTH, BEFORE the eager __init__,
#   REJECT-not-clamp (same semantics + same estimate as the axis bounds; no second semantics). The estimate
#   (_connector_est_bytes) is the DOMINANT block-Linear term + gate/register terms, x2 (bf16); it is a FAITHFUL
#   LOWER BOUND vs the real Embeddings1DConnector (real 8L/4096 ratio est/actual ~0.9998, self-CR-measured vs the real module on device=meta; est<=real across the axis space)
#   so it can NEVER over-estimate -> structurally cannot false-reject. _MAX_CONNECTOR_TOTAL_BYTES is the PER-LOAD
#   total: enforced per-connector in _derive_connector_arch (a single connector <= ceiling) AND as the video+audio
#   running total in _accumulate_connector_footprint (their SUM <= ceiling). CEILING = 8 GiB, GROUNDED (NOT
#   arbitrary, and deliberately NOT tied to runtime free-VRAM, which would be non-deterministic + card-dependent +
#   false-reject on a small card):
#   (a) ~2.1x the largest LEGITIMATE per-load pair -- the real 8-layer/4096 VIDEO connector is ~3.0 GiB bf16 by this
#       estimate (connector_arch_derivation_test.py:95 "real video 8-block") + the 8-layer/2048 AUDIO ~0.75 GiB
#       (test:101 "real audio") = ~3.75 GiB combined -- and it MUST still admit a single deep connector: the
#       depth-ceiling 16L/4096 = ~6.0 GiB (test:136 "at depth ceiling n_layers=16") passes, which a 4 GiB ceiling
#       would have WRONGLY rejected (that test pins the floor under this value); and
#   (b) a survivable HOST allocation on the smallest box that can run this (comfy+torch+models need >=16 GiB RAM),
#       well under the ~24 GiB single-max / ~16 GiB two-connector products that SIGKILL.
#   DOCUMENTED FALSE-REJECT BAND (currently UNREACHABLE, per B's CR of 39135a3 + this estimate at n_layers=8): the
#   8 GiB single-connector ceiling at the real depth (8 layers) binds at inner_dim ~6687 (solve _connector_est_bytes
#   for 8 GiB, n_layers=8), BELOW the 8192 axis bound -- so a hypothetical 8-layer connector with 6687 < inner <=
#   8192 (~8-12 GiB) passes the axis bounds but is REJECTED by this ceiling. No real LTX connector reaches that band
#   (real inner=4096, ~1.6x below the 6687 bind point); the per-LOAD COMBINED arm likewise leaves the real
#   video+audio pair (~3.75 GiB) ~4.25 GiB of headroom, so it false-rejects no real join_audio_prompt pair. If a
#   future connector legitimately widens past ~6687 at depth 8 (or a real pair exceeds ~8 GiB combined), RAISE
#   _MAX_CONNECTOR_TOTAL_BYTES deliberately (REJECT-not-clamp makes that a visible, chosen change).
_CONNECTOR_BLOCK_LINEAR_MULT = 12   # per transformer_1d block: ~4*inner^2 attention (q/k/v/out) + ~8*inner^2 FFN
_MAX_CONNECTOR_TOTAL_BYTES = 8_589_934_592   # 8 GiB (= 8 * 1024**3) -- see the grounding above


def _connector_est_bytes(n_layers, num_heads, inner_dim, n_registers):
    """Estimate a connector's eager bf16 footprint (bytes) from its derived arch. Per BLOCK (x n_layers):
    ~4*inner^2 attn q/k/v/out + ~8*inner^2 FFN (= _CONNECTOR_BLOCK_LINEAR_MULT*inner^2) + the gate [num_heads,inner].
    Once (GLOBAL): learnable_registers [n_registers, inner]. x2 (bf16). NOTE the gate term is x n_layers (per block) --
    it was missing that factor (~0.07% under-count; A's CR of 39135a3): the estimate must be a TRUE lower bound (real
    8L/4096 ratio est/actual ~0.9998, self-CR-measured against the real Embeddings1DConnector -- NOT the ~0.9992 in
    39135a3, which was the PRE-fix formula's ratio) so it can never OVER-estimate -> structurally cannot false-reject;
    the ×n_layers factor keeps that lower bound honest (it TIGHTENS the bound 0.9992->0.9998). Pure; the SINGLE
    source for BOTH the per-connector bound (_derive_connector_arch) and the per-load COMBINED running total
    (_accumulate_connector_footprint)."""
    return 2 * (n_layers * (_CONNECTOR_BLOCK_LINEAR_MULT * inner_dim + num_heads) * inner_dim
                + n_registers * inner_dim)


def _accumulate_connector_footprint(budget, n_layers, num_heads, inner_dim, n_registers, desc):
    """★ PER-LOAD COMBINED-FOOTPRINT running total (native-connector-combined-footprint-ceiling, agg-NO-GO fix):
    load() holds the video connector (:unconditional) AND the audio connector (gated by the SHIPPED join_audio_prompt
    switch) CONCURRENTLY. Each is per-connector-bounded in _derive_connector_arch, but per-connector bounded !=
    COMBINED bounded -- two each-<=8 GiB connectors = ~16 GiB, the same #543 host-RAM SIGKILL one level up (this
    file's header already notes 'two connectors ~48 GiB all-max'). So add each connector's est to the caller's
    PER-LOAD running total -- `budget` is a 1-element list local to ONE load() call; there is NO cross-call/module
    state (a module-level accumulator that did not reset on unload would DRIFT upward across use and start
    FALSE-REJECTING legitimate configs after N runs = an intermittent defect, worse than a one-time false reject) --
    and REJECT if the RUNNING TOTAL exceeds _MAX_CONNECTOR_TOTAL_BYTES, BEFORE the eager __init__. budget=None (a
    standalone connector load with no per-load context) skips the combined check; the per-connector bound in
    _derive_connector_arch still applies. REJECT-not-clamp (identical semantics to the axis + per-connector bounds)."""
    est = _connector_est_bytes(n_layers, num_heads, inner_dim, n_registers)
    if budget is None:
        return est
    budget[0] += est
    if budget[0] > _MAX_CONNECTOR_TOTAL_BYTES:
        raise RuntimeError(
            f"QuantFuncNativeLoader: the COMBINED per-load connector footprint reached ~{budget[0] / 1024 ** 3:.1f} "
            f"GiB at {desc} (each connector within its OWN bound, but load() holds video + audio CONCURRENTLY) -- "
            f"exceeds the {_MAX_CONNECTOR_TOTAL_BYTES // 1024 ** 3} GiB per-load ceiling. Refusing before the eager "
            f"__init__ (an all-max PAIR is a #543 host-RAM SIGKILL on the CPU/no-GPU path, reached via the shipped "
            f"join_audio_prompt=True switch). RAISE _MAX_CONNECTOR_TOTAL_BYTES deliberately if a larger pair is real.")
    return est


# LTX VAE scale factors (engine kS=32 spatial, kT=8 temporal, kC=128 channels — LTX2VideoPipeline).
_LTX_SPATIAL = 32
_LTX_TEMPORAL = 8
_LTX_DEFAULT_FPS = 25.0   # informational default; the sampler/graph owns real timing
_LTX_CHANNELS = 128


def _connector_config_heads(model_dir):
    """The authoritative video-connector head count from the model dir's diffusers LTX2TextConnectors config
    (connectors/config.json, video_connector_num_attention_heads), or None when no config declares one: gated checkpoints
    need none, and a non-gated one then fail-louds in _derive_connector_arch. The staged video-only dir may symlink or
    omit connectors/, so the transformer's real parent is probed too. A config that is there but unreadable, or that
    declares an unusable count, RAISES: swallowing it hid a Windows cp936 decode failure (#738) behind "no
    authoritative head count", and a dropped count also skips the gate-weight cross-check."""
    for cand in (os.path.join(model_dir, "connectors", "config.json"),
                 os.path.join(os.path.dirname(os.path.realpath(os.path.join(model_dir, "transformer"))),
                              "connectors", "config.json")):
        if not os.path.isfile(cand):
            continue
        try:
            with open(cand, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"QuantFuncNativeLoader: {cand} is unreadable: {exc}. It is the model's diffusers "
                               f"LTX2TextConnectors config: restore it from the model's download.") from exc
        if not isinstance(cfg, dict):
            raise RuntimeError(f"QuantFuncNativeLoader: {cand} must hold a JSON object. It is the model's diffusers "
                               f"LTX2TextConnectors config: restore it from the model's download.")
        heads = cfg.get("video_connector_num_attention_heads")
        if heads is None:
            continue
        if type(heads) is not int or not 1 <= heads <= _MAX_CONNECTOR_HEADS:
            raise RuntimeError(f"QuantFuncNativeLoader: {cand} declares video_connector_num_attention_heads={heads!r}; "
                               f"expected an integer in 1..{_MAX_CONNECTOR_HEADS}. Restore the file from the model's download.")
        return heads
    return None


def _derive_connector_arch(sd, desc, authoritative_heads=None):
    """Derive the FULL Embeddings1DConnector arch from the checkpoint state_dict -- EVERY dim is a checkpoint
    property, NOT a constant (dossier seq-250/252 + Finding #1). Returns (n_layers, num_heads, head_dim,
    n_registers, has_gate) -- 5 values. has_gate IS returned and IS load-bearing: the caller passes it as
    Embeddings1DConnector(apply_gated_attention=...). A non-gated checkpoint is refused ONLY when no
    authoritative head count is available (see below); with one it loads with has_gate False.

      * n_layers = 1 + max transformer_1d_blocks index (^-anchored). A wrong value trips the 0/0 load guard
        (unexpected/missing weights) -> that is how the quarter-depth 2-vs-8 bug surfaced (unexpected=100).
      * inner_dim = learnable_registers[-1]; n_registers = learnable_registers[0].
      * num_heads = attn1.to_gate_logits.bias.shape[0] -- one gate logit PER attention head. ★ Finding #1: the
        attention weights (to_q/k/v = inner_dim x inner_dim; q_norm = [inner_dim] full-width) do NOT encode the
        head split, so a wrong num_heads loads 0/0 CLEAN yet mis-groups attention -> a SILENT mangled out_vid.
        The gate is the ONLY weight encoding num_heads, so a NON-gated connector is REFUSED (fail loud) rather
        than guessed (self-CR Reviewer B: guessing silently mis-groups). All known LTX connectors are gated.
      * head_dim = inner_dim // num_heads, cross-checked num_heads*head_dim == inner_dim (fail LOUD).
      * ★ SECURITY: n_layers / inner_dim / num_heads / n_registers each feed the EAGER __init__ (built BEFORE
        load_state_dict), so EACH is magnitude-bounded here, before construction (vuln-CR + self-CR Reviewer A --
        block Linears amplify O(inner_dim^2); an unbounded value from a hostile checkpoint = #543 SIGKILL)."""
    import re as _re
    n_layers = 1 + max((int(m.group(1)) for k in sd
                        for m in [_re.search(r'^transformer_1d_blocks\.(\d+)\.', k)] if m), default=-1)
    # ★ BOUND-BEFORE-CONSTRUCT (vuln-CR): reject an out-of-range depth BEFORE the connector's eager __init__
    # builds n_layers real ~768 MB blocks (pre-load, unguardable by the 0/0 check). Catches BOTH a corrupt/
    # hostile huge index (-> host-RAM SIGKILL, #543 class) AND zero block keys (default=-1 -> n_layers=0 -> a
    # degenerate 0-block connector that would otherwise load "cleanly"). REJECT loudly, do NOT clamp. The
    # ^-anchored regex prevents a `not_transformer_1d_blocks.N.` decoy key from registering a phantom block.
    if not (1 <= n_layers <= _MAX_CONNECTOR_LAYERS):
        raise RuntimeError(
            f"QuantFuncNativeLoader: {desc} derived n_layers={n_layers} outside [1, {_MAX_CONNECTOR_LAYERS}] "
            f"-- refusing to eagerly build the connector. A real LTX connector is depth-8; this is a corrupt/"
            f"hostile checkpoint (stray transformer_1d_blocks.<idx> key) or has zero block keys. If a genuine "
            f"future connector is deeper, RAISE _MAX_CONNECTOR_LAYERS deliberately.")
    gate_bias = next((sd[k] for k in sd if k.endswith('attn1.to_gate_logits.bias')), None)
    has_gate = gate_bias is not None
    if not has_gate and authoritative_heads is None:
        # ★ self-CR Reviewer B: a NON-gated connector's head split is NOT derivable from its weights (to_q/k/v
        # are inner_dim x inner_dim; q_norm is full-width) -> guessing a fallback would SILENTLY mis-group
        # attention (a 30-head/3840 connector guessed at 32 gives (32,120); 3840%32==0 so divisibility passes,
        # bypassing the 0/0 load guard). The gate is the ONLY WEIGHT encoding num_heads -> without an
        # AUTHORITATIVE head count (the model dir's connectors/config.json
        # video_connector_num_attention_heads -- the LTX-2 19B family ships NON-gated connectors with the
        # head split declared THERE), REFUSE rather than guess.
        raise RuntimeError(
            f"QuantFuncNativeLoader: {desc} is NON-GATED (no attn1.to_gate_logits) and no authoritative "
            f"head count is available -- its attention head split is not derivable from the weights, and "
            f"guessing would silently mis-group attention. Provide a model_dir whose connectors/config.json "
            f"declares video_connector_num_attention_heads (the diffusers LTX2TextConnectors component), or "
            f"use a gated (LTX-2.3) connector checkpoint.")
    reg = sd.get('learnable_registers')
    if reg is None:
        raise RuntimeError(f"QuantFuncNativeLoader: {desc} has no learnable_registers -- cannot derive "
                           "inner_dim / num_registers.")
    inner_dim = reg.shape[-1]
    n_registers = reg.shape[0]
    num_heads = int(gate_bias.shape[0]) if has_gate else int(authoritative_heads)
    if has_gate and authoritative_heads is not None and int(authoritative_heads) != num_heads:
        # two authorities disagreeing is a wiring error, never silently pick one
        raise RuntimeError(
            f"QuantFuncNativeLoader: {desc} head-count authorities disagree -- gate weight says "
            f"{num_heads}, connectors/config.json says {authoritative_heads}. Fix the mismatched "
            f"checkpoint/config pairing.")
    # ★ MAGNITUDE BOUNDS (vuln-CR + self-CR Reviewer A): inner_dim / num_heads / n_registers ALL feed the eager
    # __init__ (blocks O(inner_dim^2), gate [num_heads, inner_dim], registers [n_registers, inner_dim]) BEFORE
    # load, so EACH is bounded HERE. n_layers was bounded above; this closes the WIDTH axis Reviewer A proved
    # open (a hostile in-bound-depth checkpoint with inner_dim=1M was accepted). REJECT, don't clamp.
    if not (1 <= num_heads <= _MAX_CONNECTOR_HEADS):
        raise RuntimeError(f"QuantFuncNativeLoader: {desc} derived num_heads={num_heads} outside "
                           f"[1, {_MAX_CONNECTOR_HEADS}] -- refusing (hostile to_gate_logits.bias width).")
    if not (1 <= inner_dim <= _MAX_CONNECTOR_INNER_DIM):
        raise RuntimeError(f"QuantFuncNativeLoader: {desc} derived inner_dim={inner_dim} outside "
                           f"[1, {_MAX_CONNECTOR_INNER_DIM}] -- refusing (block Linears are O(inner_dim^2)).")
    if not (1 <= n_registers <= _MAX_CONNECTOR_REGISTERS):
        raise RuntimeError(f"QuantFuncNativeLoader: {desc} derived num_registers={n_registers} outside "
                           f"[1, {_MAX_CONNECTOR_REGISTERS}] -- refusing (hostile learnable_registers height).")
    if inner_dim % num_heads != 0:
        raise RuntimeError(f"QuantFuncNativeLoader: {desc} head split inconsistent -- inner_dim {inner_dim} "
                           f"not divisible by num_heads {num_heads}.")
    # ★ SINGLE-CONNECTOR footprint bound (native-connector-combined-footprint-ceiling): per-AXIS bounded != PRODUCT
    #   bounded, so also bound the product-estimated eager bf16 alloc, BEFORE the __init__, REJECT-not-clamp. This is
    #   the ONE-connector arm; the video+audio COMBINED running total (two connectors held CONCURRENTLY in load()) is
    #   enforced separately by _accumulate_connector_footprint -- a per-connector pass ALONE misses that SIGKILL.
    est_bytes = _connector_est_bytes(n_layers, num_heads, inner_dim, n_registers)
    if est_bytes > _MAX_CONNECTOR_TOTAL_BYTES:
        raise RuntimeError(
            f"QuantFuncNativeLoader: {desc} arch (n_layers={n_layers}, inner_dim={inner_dim}, "
            f"num_heads={num_heads}, n_registers={n_registers}) is within EVERY per-axis bound but its SINGLE-connector "
            f"estimated eager footprint ~{est_bytes / 1024 ** 3:.1f} GiB exceeds the {_MAX_CONNECTOR_TOTAL_BYTES // 1024 ** 3} "
            f"GiB ceiling -- refusing (per-axis bounded != product bounded; an all-max product is a #543 host-RAM "
            f"SIGKILL on the CPU/no-GPU path). RAISE _MAX_CONNECTOR_TOTAL_BYTES deliberately if a larger connector is real.")
    return n_layers, num_heads, inner_dim // num_heads, n_registers, has_gate


def _load_ltx_video_connector(connector_ckpt, device, dtype=torch.bfloat16, _budget=None,
                              authoritative_heads=None):
    """Instantiate comfy's Embeddings1DConnector with the av_model LTX-2.3-22B VIDEO config and load
    `model.diffusion_model.video_embeddings_connector.*` from the comfy LTX-2.3 checkpoint. This is the
    exact connector comfy's diffusion model would run in preprocess_text_embeds — we run it here because
    the A' seam bypasses that model. Returns the ready connector Module on `device`.

    ARCH (num_layers / num_heads / head_dim / n_registers) is DERIVED from the checkpoint by
    _derive_connector_arch (which magnitude-bounds each + REFUSES a non-gated connector, so apply_gated_attention
    is always True here -- NOT "from the model config"). split_rope=True / double_precision_rope=True are the
    av_model rope_type="split" defaults (verified against av_model.py, dossier seq-252) with ZERO state_dict
    footprint (not checkpoint-derivable). A weight mismatch fails LOUD at load_state_dict (0/0 mandatory)."""
    from comfy.ldm.lightricks.embeddings_connector import Embeddings1DConnector
    import comfy.utils
    sd_full = comfy.utils.load_torch_file(connector_ckpt, safe_load=True)
    prefix = "model.diffusion_model.video_embeddings_connector."
    sd = {k[len(prefix):]: v for k, v in sd_full.items() if k.startswith(prefix)}
    if not sd:
        raise RuntimeError(
            f"QuantFuncNativeLoader: no '{prefix}*' weights in {connector_ckpt} - this must be the "
            "comfy LTX-2.3 checkpoint that carries the video_embeddings_connector (the engine svdq "
            "model_dir does NOT; see dossier seq-234).")
    # DERIVE + BOUND the connector ARCH from the checkpoint (num_layers / num_heads / head_dim / n_registers,
    # each magnitude-bounded at the derivation site; a non-gated connector is refused). See
    # _derive_connector_arch + dossier seq-250/252.
    n_layers, num_heads, head_dim, n_registers, has_gate = _derive_connector_arch(
        sd, "video_embeddings_connector", authoritative_heads=authoritative_heads)
    # ★ per-load COMBINED running total (video + audio held concurrently in load()); REJECT before this eager build
    _accumulate_connector_footprint(_budget, n_layers, num_heads, num_heads * head_dim, n_registers,
                                    "video_embeddings_connector")
    conn = Embeddings1DConnector(
        in_channels=_LTX_CHANNELS,
        cross_attention_dim=2048,               # av_model default; self-attn connector (no cross-context) -> unused
        attention_head_dim=head_dim,            # DERIVED inner_dim // num_heads
        num_attention_heads=num_heads,          # DERIVED attn1.to_gate_logits.bias -> [num_heads]
        num_layers=n_layers,                    # DERIVED (22B video connector = 8 blocks)
        num_learnable_registers=n_registers,    # DERIVED learnable_registers.shape[0] (+ magnitude-bounded)
        split_rope=True,                        # av_model rope_type="split" default (zero state_dict footprint)
        double_precision_rope=True,             # av_model default (zero footprint; verified seq-252)
        apply_gated_attention=has_gate,         # gated: heads from the gate weight; non-gated (LTX-2 19B):
                                                # heads from the model dir connectors/config.json (authoritative)
        dtype=dtype,
        device=device,
        operations=torch.nn,
    )
    missing, unexpected = conn.load_state_dict(sd, strict=False)
    # CLEAN load is MANDATORY (0/0). ANY missing OR unexpected == connector arch != checkpoint -> mangled
    # out_vid. The OLD guard only caught ALL-missing and let unexpected=100 through silently (the bug).
    if missing or unexpected:
        raise RuntimeError(
            f"QuantFuncNativeLoader: video connector load NOT CLEAN (missing={len(missing)} "
            f"unexpected={len(unexpected)}; derived n_layers={n_layers} heads={num_heads}) -- arch mismatch. "
            f"unexpected(sample)={sorted(unexpected)[:4]} missing(sample)={sorted(missing)[:4]}")
    conn = conn.to(device=device, dtype=dtype).eval()
    qfe.info("[qf_native] LTX video connector loaded CLEAN (n_layers=%d heads=%d head_dim=%d regs=%d "
                 "gated=%s; 0/0)", n_layers, num_heads, head_dim, n_registers, has_gate)
    return conn


# ── LTX-2.3-22B AUDIO connector (av_model.py audio_embeddings_connector) ─────────────────────────────
# The audio connector arch is DERIVED by _derive_connector_arch (shared with video). The audio slice of the
# dual TE output is [:, :, _LTX_VIDEO_DIM:_LTX_VIDEO_DIM+_LTX_AUDIO_DIM]. Written for the (a)-style / non-full-
# skip path where the 22B a2v branch needs the 2048 audio prompt; the engine external step splits the 6144
# into video_prompt|audio_prompt (dossier seq-241). The DERIVATION is unit-tested (connector_arch_derivation_
# test.py covers the audio inner=2048/head_dim=64 shape); the E2E audio PATH (join_audio_prompt=True) is
# DORMANT pending engine audio support -- the fox verification used the (b) video-only skip (join=False).
_LTX_AUDIO_DIM = 2048   # audio cross_attention_dim (av_model audio_cross_attention_dim); the audio TE slice width


def _load_ltx_audio_connector(connector_ckpt, device, dtype=torch.bfloat16, _budget=None):
    """Mirror of _load_ltx_video_connector for the AUDIO branch: comfy's audio_embeddings_connector
    (32 heads x 64 = inner_dim 2048) loading `model.diffusion_model.audio_embeddings_connector.*` from the
    comfy LTX-2.3 checkpoint. Lets this seam supply the POST-connector audio prompt (2048) alongside the
    video (4096) = the full 6144 the engine external step splits into video_prompt|audio_prompt for the 22B
    a2v branch. Fails LOUD on a weight-shape mismatch (never a silent-wrong connector)."""
    from comfy.ldm.lightricks.embeddings_connector import Embeddings1DConnector
    import comfy.utils
    sd_full = comfy.utils.load_torch_file(connector_ckpt, safe_load=True)
    prefix = "model.diffusion_model.audio_embeddings_connector."
    sd = {k[len(prefix):]: v for k, v in sd_full.items() if k.startswith(prefix)}
    if not sd:
        raise RuntimeError(
            f"QuantFuncNativeLoader: no '{prefix}*' weights in {connector_ckpt} - the comfy LTX-2.3 "
            "checkpoint must carry the audio_embeddings_connector (dossier seq-241).")
    # DERIVE + BOUND the connector ARCH from the checkpoint (shared _derive_connector_arch: num_layers /
    # num_heads / head_dim / n_registers, each magnitude-bounded at the derivation site; non-gated refused).
    n_layers, num_heads, head_dim, n_registers, _has_gate = _derive_connector_arch(sd, "audio_embeddings_connector")
    # ★ per-load COMBINED running total (this ADDS to whatever the video connector already committed this load());
    #   REJECT before this eager build if video+audio together exceed the ceiling
    _accumulate_connector_footprint(_budget, n_layers, num_heads, num_heads * head_dim, n_registers,
                                    "audio_embeddings_connector")
    conn = Embeddings1DConnector(
        in_channels=_LTX_CHANNELS,
        cross_attention_dim=_LTX_AUDIO_DIM,      # av_model audio_cross_attention_dim; self-attn connector -> unused
        attention_head_dim=head_dim,             # DERIVED inner_dim // num_heads (audio = 64)
        num_attention_heads=num_heads,           # DERIVED attn1.to_gate_logits.bias -> [num_heads] (audio = 32)
        num_layers=n_layers,                     # DERIVED (audio = 8 blocks)
        num_learnable_registers=n_registers,     # DERIVED learnable_registers.shape[0] (+ magnitude-bounded)
        split_rope=True,                         # av_model rope_type="split" default (zero state_dict footprint)
        double_precision_rope=True,              # av_model default (zero footprint; verified seq-252)
        apply_gated_attention=True,              # _derive_connector_arch REFUSES non-gated -> always gated here
        dtype=dtype,
        device=device,
        operations=torch.nn,
    )
    missing, unexpected = conn.load_state_dict(sd, strict=False)
    # CLEAN load MANDATORY (0/0) — the old ALL-missing-only guard let unexpected through silently.
    if missing or unexpected:
        raise RuntimeError(
            f"QuantFuncNativeLoader: AUDIO connector load NOT CLEAN (missing={len(missing)} "
            f"unexpected={len(unexpected)}; derived n_layers={n_layers} heads={num_heads}) -- arch mismatch. "
            f"unexpected(sample)={sorted(unexpected)[:4]} missing(sample)={sorted(missing)[:4]}")
    conn = conn.to(device=device, dtype=dtype).eval()
    qfe.info("[qf_native] LTX audio connector loaded CLEAN (n_layers=%d heads=%d head_dim=%d regs=%d; 0/0)",
                 n_layers, num_heads, head_dim, n_registers)
    return conn


@qfe.console_safe_methods   # an exception leaving it is console-safe (#738)
class QFLTXModel(QFSessionModelMixin, comfy.model_base.LTXV):
    """LTX-2 svdq pipeline exposed as a native comfy MODEL (native-KSampler seam), t2v + i2v.

    i2v is WAN-ALIGNED (user 2026-08-22 "只关注latent"): the model consumes ONLY latents.
    The workflow's official LTXVImgToVideoInplace node VAE-encodes the image, writes it into
    the latent's leading frame(s) and sets noise_mask = 1-strength — all comfy-side, zero
    cond keys. comfy's KSamplerX0Inpaint then blends x against the clean latent per step
    OUTSIDE this model (via the inherited BaseModel.scale_latent_inpaint — deliberately NOT
    overridden), which is exact for this seam because the engine step is STATELESS in x
    (latents in, velocity out; the sampler owns x). No loader image socket, no plugin-side
    VAE, no engine begin_edit."""

    def __init__(self, model_config, engine, connector, device=None, audio_connector=None):
        super().__init__(model_config, device=device)   # disable_unet honored in BaseModel.__init__
        self.diffusion_model = _QFStub()
        self._qf = engine                    # QFEngineHandle (lib + pipeline + open session)
        self._connector = connector          # comfy Embeddings1DConnector (video), weights loaded
        self._audio_connector = audio_connector  # comfy audio_embeddings_connector (2048); None -> video-only 4096
        self._num_steps = 0                  # DERIVED per run from sample_sigmas (len-1) at _begin
        self._num_frames = 0                 # DERIVED per run from the latent: (Tlat-1)*8 + 1
        self._fps = _LTX_DEFAULT_FPS         # informational (rides options_json); LTX conditions on its own
        self._step_i = 0
        self._sess_denoise = 0
        self._out = None                     # reused packed velocity_out buffer [1,N,128]
        # [step-cache-key] symbolic per-conditioning cfg_context_key (_CtxKeyAssigner; engine lighting CLAUDE.md #B3).
        # Historically LTX passed 0 (kNoCtxKey) because it engaged NO key-trusting cache
        # (dossier seq-229) — the merged the step cache session gate NOW trusts the key (its §6.5
        # no-key guard force-computes at key=0, silently disabling EC). uuid-derived keys give
        # each cond branch its own EcEntry (no cross-branch collapse) at zero cost when EC off.
        self._ctx_key_assigner = qfmp._CtxKeyAssigner()
        # Max POST-CONNECTOR seq len across a run's cond groups (pos+neg), accumulated in extra_conds and used
        # to size the engine begin context MAXIMA (see _begin). pos/neg can have
        # DIFFERENT prompt lengths and reach the engine as SEPARATE B==1 step calls against ONE session, while
        # _begin runs on the first group only — so it must be sized to the LARGEST group, not just the first.
        # ★ POST-connector, NOT raw (§6.5 correctness NO-GO): comfy's Embeddings1DConnector does NOT preserve
        # the seq dim — its forward tail-pads with tiled learnable_registers to
        # n_reg*ceil(max(1024, S)/n_reg) (MEASURED: S=1200→1280, 2000→2048, ≤1024→1024 at n_reg=128), so a
        # raw accumulator UNDER-sizes the ceiling whenever a >1024-raw group is not the group that _begins.
        self._max_ctx_seq = 0
        self._post_seq_cache = {}            # raw S -> measured post-connector S (see _post_connector_seq)

    # ── engine-conditioning safety layer (added for the CR conformance NO-GO) ──
    # LTXV.extra_conds (+ the LTXV.concat_cond / BaseModel.extra_conds callees) can consume mask / keyframe /
    # guide / c_concat / controlnet / i2v-concat channels when a corresponding node is wired. This external-
    # session seam runs the engine's OWN t2v schedule from the loader widgets + the connector video_embeds and
    # consumes NONE of them, so a bare _apply_model(**kwargs) would SILENTLY drop them → a plausible-but-wrong
    # video with no warning. FAIL LOUD instead. This is a DEFENSIVE SUPERSET — every
    # consumable key that is neither emitted (cross_attn) nor documented-accepted (attention_mask/frame_rate);
    # its completeness against THIS comfy is machine-checked by tests/reject_list_completeness.py (LTXV entry).
    # denoise_mask is deliberately NOT in this tuple: the Inplace i2v latent route rides it
    # (comfy's sampler consumes it via KSamplerX0Inpaint + the inherited scale_latent_inpaint;
    # it never needs the engine) — see the class docstring + reject_list_completeness.py.
    _ENGINE_IGNORED_COND_KEYS = ("concat_mask", "keyframe_idxs", "guide_attention_entries",
                                 "noise_concat", "cross_attn_controlnet", "concat_latent_image",
                                 # LTX-AV-era cond keys this comfy's LTXAV.extra_conds exposes —
                                 # from LTXVReferenceAudio / the keyframe generators. The native
                                 # session consumes neither; a wired producer must fail loud, not
                                 # be silently dropped.
                                 # ★ NEVER put a closing bracket in a comment INSIDE this tuple:
                                 # the reject-list audit parses it with a  [^ closing-bracket ]*
                                 # character class, so the first one TRUNCATES the parsed set and
                                 # every key after it is reported UNCOVERED. Measured 2026-08-21:
                                 # an aside in brackets hid ref_audio + generated_keyframes.
                                 "ref_audio", "generated_keyframes")

    def extra_conds(self, **kwargs):
        # RUN-START clean slate FIRST (#2 session lifecycle) — BEFORE the loud-fails below, so a rejected
        # bad-wiring requeue still closes a session STRANDED by a prior Interrupt (else a permanently-erroring
        # graph leaks a GPU-resident stranded session across requeues). extra_conds is comfy's ONLY hook that
        # fires at the start of EVERY run even for an EMPTY (denoise=1.0) latent, and comfy CACHES+REUSES this
        # model instance across a requeue — so closing here forces _apply_model to _begin fresh. Idempotent
        # across the pos+neg extra_conds calls of a run (the 2nd finds nothing open → no-op).
        was_open, ok = self._qf.end_session_if_open()
        # Retention (2026-08-24): even a REFUSED close (previous run's step still draining
        # engine-side) must NOT let this run silently REUSE that session — force the first
        # _apply_model through _begin (whose materialize-first close + busy-retry recovers
        # correctly). Cleared at the gate; re-armed every run start.
        self._qf_needs_begin = True
        if was_open:
            qfe.info("[qf_native] LTX: closed a pre-existing session at run start "
                  f"(prior run interrupted/uncleaned); end ok={ok}", flush=True)
        # masked-latent i2v (comfy LTXVImgToVideoInplace): capture the run's denoise_mask
        # as PER-LATENT-FRAME sigma scales for the engine session. This is the comfy
        # LTXV.process_timestep mechanism (conditioned tokens run at mask*t) — without it
        # the engine reads the outer-blended near-clean conditioned frames at the GLOBAL
        # sigma and emits corrupt velocities (measured: red-speckle degeneration, run
        # b16cbc7f). Cleared per run here; pos+neg extra_conds set identical values.
        self._pending_frame_t_scale = None
        _dm = kwargs.get("denoise_mask")
        if _dm is not None:
            _scales = self._frame_scales_from_mask(_dm)
            # all-1.0 = no conditioning anywhere -> omit (keeps the engine's t2v scalar
            # path byte-identical instead of a mathematically-equal per-token pass).
            self._pending_frame_t_scale = None if all(v >= 0.9999 for v in _scales) else _scales
        # ★ CLOSE THE SILENT-DISCARD CLASS: a wired keyframe / guide-attention / concat node reaches here
        # as one of these keys and would be dropped — fail loud. (denoise_mask is NOT here: the Inplace
        # latent route rides it and comfy's sampler consumes it entirely outside this model.)
        for _k in self._ENGINE_IGNORED_COND_KEYS:
            if kwargs.get(_k) is not None:
                raise RuntimeError(
                    f"qf_native LTX: '{_k}' conditioning is wired, but the QuantFunc LTX native session "
                    f"consumes only latents + the connector video_embeds - it cannot consume comfy's "
                    f"'{_k}', which would be silently ignored. Remove the node feeding it. For "
                    f"image-to-video use LTXVImgToVideoInplace on the LATENT path (its noise_mask is "
                    f"applied by comfy's sampler and IS supported); LTXVAddGuide / keyframe nodes are not.")
        # MIRROR ONLY c_crossattn (the engine's connector consumes it), building `out` BY HAND —
        # we deliberately do NOT call super().extra_conds. Reason: emit EXACTLY the one channel the engine takes,
        # and don't re-enter comfy's cond-building (LTXV.extra_conds + its concat_cond/encode_adm callees), which
        # would re-populate the very keys we reject. (LTXV does not override concat_cond — the effective
        # BaseModel.concat_cond is concat_keys-gated and never touches diffusion_model — so calling super() would
        # not crash here; building by hand keeps the emitted set minimal and explicit.) frame_rate is unused by this seam; attention_mask IS consumed (the connector mask — 19B mask fix).
        out = {}
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
            # [19B mask fix] forward the clip's attention_mask: gemma LEFT-pads, so the connector MUST
            # mask the pad rows (the engine's own internal path builds a content-tail mask before its
            # connector forward — "text mask active=N/S"). The old all-ones simplification aggregated
            # PAD rows into the registers → garbage conditioning → structured-mush output (MEASURED:
            # stock comfy with the SAME TE chain coherent, this seam mush, fixed by this mask).
            am = kwargs.get("attention_mask", None)
            if am is not None:
                out["attention_mask"] = comfy.conds.CONDRegular(am)
            # Accumulate the max POST-CONNECTOR seq len across this run's cond groups (pos+neg BOTH call
            # extra_conds before sampling). ★ POST, not raw (§6.5 correctness NO-GO — an earlier revision
            # claimed "the connector PRESERVES the seq dim"; FALSE: comfy's Embeddings1DConnector tail-pads
            # to n_reg*ceil(max(1024, S)/n_reg), so raw 1200 becomes vemb 1280). _begin sizes the session
            # context maxima from this accumulator vs the FIRST group's actual vemb; a raw accumulator
            # under-sizes it whenever a >1024-raw group (a long negative) is NOT the group that _begins —
            # that group's step then exceeds the begin maxima and the engine LOUD-REJECTS
            # (quantfunc_api.cpp "positive and per-dim <= the begin maxima"; correctness, not silent).
            # _post_connector_seq MEASURES the length from the real loaded connector (cached).
            self._max_ctx_seq = max(self._max_ctx_seq, self._post_connector_seq(int(cross_attn.shape[1])))
        return out

    def _frame_scales_from_mask(self, dm):
        """denoise_mask -> per-LATENT-frame sigma scales (comfy process_timestep parity).

        Two shapes reach this seam (MEASURED):
        - video-only 5D channel-repeated [B,C,F,H,W] (plain latent path; the mask is
          channel-uniform so channel 0 is representative);
        - joint-AV FLAT [B,1,total] — comfy packs the nested (video, audio) latent and
          flattens the mask alike (total = video_elems + audio_elems; measured
          1,818,368 = 128*16*22*40 + 8*126*16). Split via self.latent_shapes.
        Inplace masks are FRAME-uniform; a mask varying within a frame (spatial inpaint)
        or masking the AUDIO lane is refused loud — the engine scales per FRAME only."""
        import math as _m
        m = dm
        if m.ndim == 3:
            ls = getattr(self, "latent_shapes", None)
            if not ls or len(ls) < 1:
                raise RuntimeError(
                    "qf_native LTX: flat denoise_mask but the model carries no "
                    "latent_shapes to split it - unsupported mask source.")
            n_video = int(_m.prod(ls[0][1:]))
            flat = m.reshape(-1)
            if int(flat.shape[0]) < n_video:
                raise RuntimeError(
                    f"qf_native LTX: flat denoise_mask has {int(flat.shape[0])} elements "
                    f"but the video latent needs {n_video} - geometry mismatch.")
            audio_part = flat[n_video:]
            if audio_part.numel() and float(audio_part.min()) < 0.9999:
                raise RuntimeError(
                    "qf_native LTX: the denoise_mask masks the AUDIO lane - audio "
                    "conditioning is not supported on this seam (video Inplace only).")
            m = flat[:n_video].reshape(list(ls[0]))[0]      # [C,F,H,W]
            m = m.movedim(1, 0)                             # [F,C,H,W]
        elif m.ndim == 5:
            m = m[0, 0]                                     # [F,H,W]
        else:
            raise RuntimeError(
                f"qf_native LTX: denoise_mask with shape {tuple(dm.shape)} is not a "
                f"[B,C,F,H,W] latent mask nor the packed AV flat form - unsupported "
                f"mask source.")
        per_frame = []
        for f in range(int(m.shape[0])):
            mf = m[f]
            lo, hi = float(mf.min()), float(mf.max())
            if hi - lo > 1e-4:
                raise RuntimeError(
                    "qf_native LTX: denoise_mask varies WITHIN a latent frame - the "
                    "engine session scales the timestep per FRAME (Inplace-style masks "
                    "only); spatial inpaint masks are not supported on this seam.")
            per_frame.append(max(0.0, min(1.0, hi)))
        return per_frame

    # scale_latent_inpaint is deliberately INHERITED (comfy BaseModel), NOT loud-failed (2026-08-22
    # wan-align pivot): comfy's KSamplerX0Inpaint calls it only when a denoise/noise mask is wired
    # (LTXVImgToVideoInplace's frame-0 conditioning, or SetLatentNoiseMask) and blends x against the
    # clean latent per step OUTSIDE _apply_model. That blend is EXACT for this seam because the
    # engine step is stateless in x (latents in, velocity out — the sampler owns x), so masked
    # conditioning needs zero engine involvement. This is the i2v mechanism now.

    def _derive_geometry(self, xin, transformer_options):
        """DERIVE the session geometry from the graph (official-loader shape — the loader has no
        geometry widgets). xin: [B,128,F_lat,H_lat,W_lat]; LTX VAE scale temporal 8 / spatial 32.
        Step count from the sampler's own sigma schedule. PARTIAL / TRIMMED ranges are
        ACCEPTED (two-stage official workflows: stage-A 1.0->0.975 low-res, stage-B
        0.85->0 refine after the x2 latent upsample): the external session is PURELY
        sigma-driven — the engine's session step consumes the driver's sigma each call
        and its session path has ZERO num_steps / step-index consumption (verified by
        source sweep + c5.8 legC bit-identical under a full external Euler drive). The
        old refusal's rationale ("engine runs its OWN internal schedule") described the
        INTERNAL generate_video loop, not this seam. Only pathological schedules
        (fewer than 2 sigmas / non-decreasing) are refused."""
        Tlat = int(xin.shape[2])
        self._num_frames = (Tlat - 1) * _LTX_TEMPORAL + 1
        sigmas = transformer_options.get("sample_sigmas") if isinstance(transformer_options, dict) else None
        if sigmas is None or len(sigmas) < 2:
            raise RuntimeError(
                "qf_native LTX: the sampler did not publish a sigma schedule "
                "(transformer_options['sample_sigmas']) - the engine session needs the step count. "
                "Use a stock KSampler / SamplerCustom on this model.")
        self._num_steps = len(sigmas) - 1
        try:
            s_first, s_last = float(sigmas[0]), float(sigmas[-1])
        except Exception:  # noqa: BLE001 - non-tensor sigmas: keep the count, skip the range check
            return
        if s_first <= s_last:
            raise RuntimeError(
                f"qf_native LTX: sigma schedule must be strictly DECREASING (got "
                f"{s_first:.4f}->{s_last:.4f}) - a non-decreasing schedule would drive the "
                f"session backwards.")

    # ── connector bridge: comfy pre-connector dual TE [B,S,6144] -> POST-connector [B,S,6144 | 4096] ──
    def _post_connector_seq(self, raw_s):
        """MEASURED post-connector seq length for a raw cross_attn length `raw_s` — by running the REAL
        loaded video connector on a zeros shape-probe and reading the output seq (cached per raw_s).
        WHY a probe, not arithmetic: comfy's Embeddings1DConnector forward tail-pads the sequence with
        TILED learnable_registers to n_reg*ceil(max(1024, S)/n_reg) (embeddings_connector.py forward;
        MEASURED S=1200→1280 / 2000→2048 / ≤1024→1024 at n_reg=128, and 1200→1216 at n_reg=64 — the
        rounding follows the checkpoint-derived n_reg, and the 1024 floor is a literal in COMFY's code) —
        transcribing that law here would silently drift on a ComfyUI upgrade, so we ask comfy's own module.
        Registers-less connector (num_learnable_registers falsy) keeps S unchanged → identity, no probe.
        Probe input dim comes from the module itself (learnable_registers.shape[1] == inner_dim, the
        dimension comfy's own forward cat requires hidden_states to carry when the registers are appended;
        the ctor's in_channels kwarg is stored as out_channels and never read in forward) — NOT the
        _LTX_VIDEO_DIM constant, so the probe can never disagree with the module it measures. Cost: one tiny no-grad zeros forward per DISTINCT raw
        length per model lifetime (a run has ≤2: pos+neg). The audio connector needs no probe: _run_connector
        cat(dim=-1) already REQUIRES audio post-S == video post-S, so the video length IS the vemb length."""
        cached = self._post_seq_cache.get(raw_s)
        if cached is not None:
            return cached
        regs = getattr(self._connector, "learnable_registers", None)
        if regs is None:
            post = int(raw_s)                      # registers-off connector: comfy keeps S unchanged
        else:
            with torch.no_grad():
                out = self._connector(
                    torch.zeros(1, int(raw_s), int(regs.shape[1]), device=regs.device, dtype=regs.dtype))[0]
            post = int(out.shape[1])
        self._post_seq_cache[raw_s] = post
        return post

    def _run_connector(self, c_crossattn, attention_mask=None):
        """Run comfy's video AND (when present) audio connectors on the pre-connector dual TE output
        [B,S,6144] (video 4096 | audio 2048) -> POST-connector video_embeds 4096 | audio_embeds 2048.
        The engine external step splits the 6144 into video_prompt|audio_prompt for the 22B a2v branch.
        If the engine's video-only skip needs ONLY the video prompt, build the loader with no audio
        connector (self._audio_connector=None) -> this returns just the 4096 video_embeds."""
        if not hasattr(self, "_connector_gated"):
            self._connector_gated = any("to_gate_logits" in n for n, _ in self._connector.named_parameters())
        ctx = c_crossattn
        # video slice width = the CONNECTOR\'s own inner_dim (learnable_registers.shape[1] -- the width its
        # forward cat requires), NOT the 22B constant: the 19B family is 3840-wide. Registers-less fallback
        # keeps the 22B constant (all known connectors carry registers).
        _regs = getattr(self._connector, "learnable_registers", None)
        vid_dim = int(_regs.shape[1]) if _regs is not None else _LTX_VIDEO_DIM
        # [19B convention] the OFFICIAL 19B forward does NOT run the embeddings connector on the clip
        # output: comfy\'s av_model.preprocess_text_embeds RETURNS AS-IS when the context width equals
        # caption_channels*2 (= vid 3840 | aud 3840 -- the 19B ltxv clip\'s aggregate-projected output).
        # MEASURED: running the connector here anyway made the engine forward conditioning-INSENSITIVE
        # (pos/neg velocities near-identical) -> structured mush, while stock comfy (connector skipped
        # by that width test) is coherent. The 19B family is detected by its NON-GATED connector
        # (self._connector_gated False); the 2.3/2.5 gated family keeps the connector path unchanged.
        if not getattr(self, "_connector_gated", True) and ctx.shape[-1] == 2 * vid_dim:
            return ctx[:, :, :vid_dim].contiguous()
        need = vid_dim + (_LTX_AUDIO_DIM if self._audio_connector is not None else 0)
        if ctx.shape[-1] < need:
            raise RuntimeError(
                f"qf_native LTX: c_crossattn last dim {ctx.shape[-1]} < {need} - expected the LTX-2.3 dual "
                "TE output [*,S,6144] (video 4096 | audio 2048). Wire a stock LTX-2.3 dual_linear CLIP "
                "(DualCLIPLoader gemma_3_12B_it + ltx-2.3_text_projection_bf16, type=ltxv).")
        dev = next(self._connector.parameters()).device
        context_vid = ctx[:, :, :vid_dim].contiguous()
        mask = None
        if attention_mask is not None:
            mask = attention_mask.to(dev)
            if mask.ndim > 2:                      # some clips emit [B,1,S] / [B,S,S]-style masks
                mask = mask.reshape(mask.shape[0], -1)[:, -ctx.shape[1]:]
        with torch.no_grad():
            video_embeds = self._connector(context_vid.to(dev, dtype=torch.bfloat16),
                                           attention_mask=mask)[0]                                # [B,S,4096]
            if self._audio_connector is None:
                return video_embeds                                                               # [B,S,4096]
            context_aud = ctx[:, :, vid_dim:vid_dim + _LTX_AUDIO_DIM].contiguous()
            audio_embeds = self._audio_connector(context_aud.to(dev, dtype=torch.bfloat16),
                                                  attention_mask=mask)[0]                          # [B,S,2048]
        return torch.cat([video_embeds, audio_embeds], dim=-1)                                     # [B,S,6144]

    def _begin_extra_opts(self):
        """Extra begin options_json keys — the QFLTXAVModel hook (audio_dims +
        av_unprocessed_ctx). Video-only base: none."""
        return {}

    def _begin(self, x_group, vemb_group):
        """Open the t2v external denoise session. x_group = [1,128,F,H,W] latent; vemb_group =
        [1,S,4096] POST-connector video_embeds. Geometry: engine derives F_lat/H_lat/W_lat from
        num_frames + width/height (spatial 32, temporal 8)."""
        lib = self._qf.lib   # MATERIALIZE FIRST (see qf_h3_modelpatcher._begin: a deferred wrapper
        # no-ops the close while the cached engine still holds an interrupted run's open session)
        self._qf.end_session_if_open()
        self._ctx_key_assigner.reset()   # per-generation uuid->key numbering (no cross-gen leak)
        bpx = qfe.DenoiseBeginParams()
        ctypes.memset(ctypes.byref(bpx), 0, ctypes.sizeof(bpx))
        bpx.struct_size = ctypes.sizeof(bpx)
        # latent is [1, 128, F_lat, H_lat, W_lat]; target pixels = latent * VAE scale (spatial 32).
        bpx.width = int(x_group.shape[-1]) * _LTX_SPATIAL
        bpx.height = int(x_group.shape[-2]) * _LTX_SPATIAL
        bpx.num_steps = self._num_steps
        # Size the context MAXIMA to the LARGEST POST-CONNECTOR vemb seq across this run's cond groups
        # (accumulated in extra_conds for pos+neg via _post_connector_seq — same quantity as this group's
        # vemb.shape[1], so the max compares like with like), NOT just this first group's — a longer
        # different-length negative reaches the engine as a SEPARATE B==1 step against this SAME session and
        # must fit the begin maxima. max(...) with this group is the safe fallback if the accumulator was
        # never populated (cross_attn-less flow). (POST-length fix per §6.5.)
        _max_seq = max(self._max_ctx_seq, int(vemb_group.shape[1]))
        bpx.max_context_dims = (ctypes.c_int * 3)(int(vemb_group.shape[0]), _max_seq,
                                                  int(vemb_group.shape[2]))
        self._max_ctx_seq = 0             # reset for the next run's extra_conds accumulation
        # LTX takes NO pooled cond.
        bpx.max_pooled_dims = (ctypes.c_int * 2)(0, 0)
        bpx.cond_dtype = _qf_dtype(vemb_group.dtype)
        _mask_opt = ({"video_frame_t_scale": self._pending_frame_t_scale}
                     if getattr(self, "_pending_frame_t_scale", None) else {})
        bpx._opts = json.dumps({**{"num_frames": self._num_frames, "fps": float(self._fps)},
                                **self.residency_opts(),
                                **_mask_opt,
                                **self._begin_extra_opts()}).encode()
        bpx.options_json = bpx._opts
        session = ctypes.c_void_p()
        # ONE begin path (wan-align 2026-08-22): the session is ALWAYS a plain t2v-shape begin.
        # i2v conditioning is entirely comfy-side (LTXVImgToVideoInplace latent + noise_mask via
        # KSamplerX0Inpaint) — the engine never sees an image or a cond latent.
        import os as _os
        _prof = _os.environ.get("QF_NATIVE_PROF") == "1"
        if _prof:
            import time as _time
            _t0 = _time.perf_counter()
        st = lib.quantfunc_denoise_begin(self._qf.pipeline, ctypes.byref(bpx), ctypes.byref(session))
        if st != qfe.QUANTFUNC_OK and "busy" in (qfe.last_err(lib) or ""):
            # ONE bounded recovery (2026-08-24 busy incident): a just-interrupted run's final
            # step may still be draining engine-side — its refused end RETAINED our pointer,
            # so an end+retry can win once the step lands. Non-busy refusals fall through.
            time.sleep(2.0)
            self._qf.end_session_if_open()
            st = lib.quantfunc_denoise_begin(self._qf.pipeline, ctypes.byref(bpx), ctypes.byref(session))
        if _prof:
            qfe.say(f"[qf_prof] begin_call {(_time.perf_counter()-_t0)*1000:.0f} ms", flush=True)
        self._begin_keep = bpx
        if st != qfe.QUANTFUNC_OK:
            raise RuntimeError(f"denoise_begin (LTX) failed: {qfe.last_err(lib)}")
        self._qf.current_session = session
        self._step_i = 0
        self._sess_denoise = 0
        qfe.info(f"[qf_native] LTX SESSION OPEN handle={session.value:#x} steps={self._num_steps} "
              f"cond={tuple(vemb_group.shape)} frames={self._num_frames}", flush=True)

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        sigma = t
        if c_crossattn is None:
            raise RuntimeError("qf_native LTX: no c_crossattn cond - wire a stock LTX-2.3 CLIPTextEncode")
        if control is not None:
            raise RuntimeError(
                "qf_native LTX: a ControlNet is wired, but this seam does not consume comfy control "
                "hints - remove it (native LTX ControlNet is not supported through this loader).")
        dev = x.device
        xin = x.to(torch.bfloat16).contiguous()          # [B, 128, F, H, W]
        B = int(xin.shape[0])
        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        # [step-cache-key] comfy's per-conditioning uuids (aligned with cond_or_uncond) — the
        # symbolic-key source (see _CtxKeyAssigner for why NOT the 0/1 role index).
        cuuids = transformer_options.get("uuids") if isinstance(transformer_options, dict) else None
        if B > 1 and (cou is None or len(cou) != B):
            raise RuntimeError(f"qf_native LTX: engine forward is B==1 per cond group but got batch={B} "
                               f"with cond_or_uncond={cou} - batch_size>1 latents are not supported")
        # ★ loader-widget vs graph-latent geometry cross-check (CR conformance) — before any engine work.
        self._derive_geometry(xin, transformer_options)
        # CONNECTOR BRIDGE: comfy pre-connector [B,S,6144] -> POST-connector video_embeds [B,S,4096].
        vemb = self._run_connector(c_crossattn, attention_mask=kwargs.get("attention_mask")
                                   ).to(dev, dtype=torch.bfloat16).contiguous()
        if self._qf.current_session is None or getattr(self, "_qf_needs_begin", False):
            self._qf_needs_begin = False
            # shared black-video guard (see qf_modelpatcher.refuse_all_zero_initial_latent).
            qfmp.refuse_all_zero_initial_latent(xin, "LTX")
            self._begin(xin[0:1].contiguous(), vemb[0:1].contiguous())
        # packed velocity_out buffer [1, N, 128]
        _, C, F, H, W = xin.shape
        N = F * H * W
        if self._out is None or self._out.shape != (1, N, C):
            self._out = torch.empty((1, N, C), dtype=xin.dtype, device=dev)
        sig_all = sigma.reshape(-1) if torch.is_tensor(sigma) else None
        step_index = self._sigma_step_index(sigma, sig_all, transformer_options)  # sigma-schedule-derived (QFSessionModelMixin)
        out5d = torch.empty_like(xin)
        for i in range(B):
            # Interrupt poll INSIDE a session-clearing guard (CR #2): comfy raises InterruptProcessingException
            # here; UNCAUGHT it would strand current_session (GPU-resident) across a cached-node requeue. The
            # try/except-BaseException mechanics live in the SHARED _interrupt_poll_end_session_on_raise
            # (one copy for WAN + LTX — §6.5 simplicity: the hardening had drifted between the two classes).
            _interrupt_poll_end_session_on_raise(self._qf)
            xi = xin[i:i + 1].contiguous()               # [1,128,F,H,W]
            # PACK [1,128,F,H,W] -> tokens [1,N,128] (row-major (F,H,W); exact inverse of unpack_video_tokens)
            tokens = xi.reshape(1, C, N).transpose(1, 2).contiguous()     # [1,N,128]
            vi = vemb[i:i + 1].contiguous()
            sig_i = float(sig_all[i].item()) if (sig_all is not None and sig_all.numel() >= B) else \
                (float(sig_all[0].item()) if sig_all is not None else float(sigma))
            p = qfe.DenoiseStepParams()
            ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
            p.struct_size = ctypes.sizeof(p)
            p.latent_in = tokens.data_ptr()
            p.velocity_out = self._out.data_ptr()
            p.velocity_out_capacity = self._out.numel() * self._out.element_size()
            p.dims = (ctypes.c_int * 5)(1, N, C, 0, 0)   # packed 3D [1,N,128]
            p.dtype = _qf_dtype(tokens.dtype)
            p.sigma = sig_i
            p.step_index = step_index
            p.total_steps = self._num_steps
            p.context = vi.data_ptr()
            p.context_dims = (ctypes.c_int * 3)(*vi.shape)
            p.context_dtype = _qf_dtype(vi.dtype)
            # [step-cache-key] symbolic key from comfy's per-conditioning uuid (was constant 0 =
            # kNoCtxKey when LTX engaged no key-trusting cache, dossier seq-229; the step-cache
            # session gate now keys one EcEntry per cond branch and force-computes at key=0).
            cuid = cuuids[i] if (cuuids is not None and i < len(cuuids)) else None
            p.cfg_context_key = self._ctx_key_assigner.key(cuid)
            if os.environ.get("QF_NATIVE_DEBUG_CTXKEY"):   # off by default; probes the uuid path
                qfe.say(f"[qf_native] LTX CTXKEY step={step_index} grp={i} cuuids_none={cuuids is None} "
                        f"len={0 if cuuids is None else len(cuuids)} cuid={str(cuid)[:8]} key={p.cfg_context_key}",
                        flush=True)
            self._call_denoise_step(
                p, f"LTX denoise_step[step={step_index},group={i},key={p.cfg_context_key}]")  # QFSessionModelMixin
            # UNPACK velocity [1,N,128] -> [1,128,F,H,W] (exact inverse of the pack)
            out5d[i:i + 1] = self._out.transpose(1, 2).reshape(1, C, F, H, W).to(xin.dtype)
            self._qf.step_count += 1
            self._sess_denoise += 1
        self._step_i += 1
        self._qf.sampler_step_count += 1
        return self.model_sampling.calculate_denoised(sigma, out5d.float(), x)

    def process_latent_out(self, latent):
        # finalize + end, then hand back to comfy's LTXV latent post-processing.
        # R1 (design seq-517): the old body zeroed the struct (latent=NULL) and IGNORED the
        # status — quantfunc_denoise_finalize REJECTS a null latent, so the call was a silent
        # no-op. Harmless-looking for t2v (engine finalize early-returns anyway) but a hard
        # BLOCKER for i2v: the step fn zeroes frame-0 velocity, so without the finalize pin
        # frame-0 would decode as pure noise. Mirror the WAN implementation: pass the REAL
        # latent in the session's STEP-pin shape and CHECK the status (fail-loud).
        if self._qf.current_session is not None:
            try:
                lib = self._qf.lib
                # The LTX session's step pin is the PACKED 3D token shape (1, N, C) — see
                # _apply_model's p.dims — NOT the 5D latent. Pack exactly like the step does:
                # [1,C,F,H,W] -> [1,N,C], BF16 contiguous (the pin dtype; comfy's latent is
                # fp32 — a raw pass would mismatch the pin and be REFUSED).
                lat5 = (latent[0:1] if latent.shape[0] > 1 else latent)
                Bc, C, F, H, W = lat5.shape
                N = F * H * W
                tokens = (lat5.reshape(1, C, N).transpose(1, 2)
                          .to(torch.bfloat16).contiguous())          # [1,N,C]
                fp = qfe.DenoiseFinalizeParams()
                ctypes.memset(ctypes.byref(fp), 0, ctypes.sizeof(fp))
                fp.struct_size = ctypes.sizeof(fp)
                fp.latent = tokens.data_ptr()
                fp.latent_capacity = tokens.numel() * tokens.element_size()
                fp.dims = (ctypes.c_int * 5)(1, N, C, 0, 0)
                fp.dtype = _qf_dtype(tokens.dtype)
                fst = lib.quantfunc_denoise_finalize(self._qf.current_session, ctypes.byref(fp))
                if fst != qfe.QUANTFUNC_OK:
                    raise RuntimeError(f"qf_native: LTX denoise_finalize failed: {qfe.last_err(lib)}")
                # The session is always t2v-shape: engine finalize early-returns (buffer
                # read-not-written) — keep the original fp32 latent untouched (bit-exact).
                # i2v frame-0 imposition lives comfy-side (X0Inpaint), never in finalize.
                qfe.info(f"[qf_native] LTX SESSION CLOSED after {self._step_i} sampler steps, "
                      f"{self._sess_denoise} denoise_step calls, finalize=OK", flush=True)
            finally:
                self._qf.end_session_if_open()
        return super().process_latent_out(latent)


# ── LTX-2.5 joint audio+video (c5.8b) ────────────────────────────────────────────────
# Packed audio latent row width: comfy audio latent [B,8,L,16] ↔ engine rows [1,L,128]
# with d = c*16 + f (engine diffusers _unpack_audio_latents: packed[b,l,d] →
# unpacked[b, d/16, l, d%16]) — i.e. permute(0,2,1,3).reshape.
_LTXAV_AUDIO_CH = 8
_LTXAV_AUDIO_MEL = 16
_LTXAV_AUDIO_PACK = _LTXAV_AUDIO_CH * _LTXAV_AUDIO_MEL   # 128


@qfe.console_safe_methods   # an exception leaving it is console-safe (#738)
class QFLTXAVModel(QFLTXModel):
    """LTX-2.5 JOINT AUDIO+VIDEO through the native session (t2av).

    Differences vs the video-only QFLTXModel base:
    - The comfy latent is the AV pair (video [B,128,F,H,W], audio [B,8,L,16]) — nested or
      sampler-packed flat (split via self.latent_shapes, the H3-measured dual form).
    - NO plugin-side connector: the LTX-2.5 'with-proj' TE emits the UNPROCESSED dual-proj
      concat (extra unprocessed_ltxav_embeds=True) and the ENGINE runs its own checkpoint-
      loaded connector tail (begin av_unprocessed_ctx=true →
      LTX2TextConnectors::forwardProjected) — one connector implementation, the
      7-tier-verified internal one.
    - Every step drives BOTH lanes via quantfunc_denoise_step_multi at the SAME sigma (the
      LTX co-denoise contract — no H3-style shift/carry; audio_scale is inert at 1.0).
    BASE-CLASS note: deliberately QFLTXModel-only (NOT + comfy.model_base.LTXAV) —
    model_base.{LTXV,LTXAV} are BaseModel SIBLINGS whose __init__ super()-chains pass
    different unet_model kwargs, so a diamond MRO crashes ("unexpected keyword argument
    'unet_model'", MEASURED first run). Nothing needs the LTXAV model class here: the
    AV latent_format + memory factor ride the supported_models.LTXAV CONFIG the loader
    passes; comfy's nested-latent packing is model-agnostic (samplers.py pack_latents +
    inner_model.latent_shapes); the only comfy isinstance on model_base.LTXAV
    (lora.py:371) tests (LTXV, LTXAV) — satisfied via the LTXV base."""

    def __init__(self, model_config, engine, device=None):
        # i2v-AV (wan-align 2026-08-22 "只关注latent"): NO image/vae plumbing here — the
        # workflow's LTXVImgToVideoInplace conditions the VIDEO half of the joint latent
        # (encode + frame-0 write + noise_mask), and comfy's sampler applies the mask
        # outside the model. The session begin is identical to t2av.
        QFLTXModel.__init__(self, model_config, engine, connector=None,
                            device=device, audio_connector=None)
        self._out_audio = None            # reused packed audio velocity buffer [1,L,128] fp32
        self._ctx_unprocessed = False     # set per run by extra_conds from the TE's marker
        self._audio_rows = 0              # L, set from the actual audio latent at _apply_model

    def _post_connector_seq(self, raw_s):
        # The ENGINE connector tail preserves S (registers REPLACE pad rows — no comfy-style
        # tail-pad), and this seam sizes the begin maxima on the RAW ctx it actually sends.
        return int(raw_s)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        # The LTXAV 'with-proj' TE marks its output UNPROCESSED (dual-proj concat, connector
        # NOT run). Required by this seam — checked fail-loud at _begin_extra_opts.
        self._ctx_unprocessed = bool(kwargs.get("unprocessed_ltxav_embeds", False))
        return out

    def _run_connector(self, c_crossattn, attention_mask=None):
        # AV seam: pass the RAW dual-proj concat through — the engine connector tail runs
        # per distinct cond (see the class docstring). attention_mask is unused: the comfy
        # LTXAV TE already TRIMS the left-pad to the active tail before projection
        # (lt.py encode_token_weights `out[:, :, -sum(mask):]`), so every row is valid.
        return c_crossattn

    def _begin_extra_opts(self):
        if not self._ctx_unprocessed:
            raise RuntimeError(
                "qf_native LTX-AV: the wired text encoder did not mark its output "
                "unprocessed_ltxav_embeds - this loader needs the LTX-2.5 'with-proj' TE "
                "(gemma4-12b-with-proj-*.safetensors through the comfy LTXAV clip), whose "
                "connector runs ENGINE-side. A 19B/2.3 connector-file TE chain belongs on "
                "the video-only QuantFuncNativeLoader path.")
        if self._audio_rows <= 0:
            raise RuntimeError("qf_native LTX-AV: audio latent rows unset at begin - wiring error")
        return {"audio_dims": [1, int(self._audio_rows), _LTXAV_AUDIO_PACK],
                "av_unprocessed_ctx": True}

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        # [qf_prof] per-step wall probe (diagnostic, QF_NATIVE_PROF=1 gated print only —
        # not a production-path switch): total wall per sampler step incl. all comfy-side
        # conversion; the engine-internal share rides the engine's own logs.
        import os as _os
        if _os.environ.get("QF_NATIVE_PROF") != "1":
            return self._apply_model_timed(x, t, c_concat=c_concat, c_crossattn=c_crossattn,
                                           control=control, transformer_options=transformer_options,
                                           **kwargs)
        import time as _time
        _t0 = _time.perf_counter()
        try:
            return self._apply_model_timed(x, t, c_concat=c_concat, c_crossattn=c_crossattn,
                                           control=control, transformer_options=transformer_options,
                                           **kwargs)
        finally:
            qfe.say(f"[qf_prof] step wall {(_time.perf_counter()-_t0)*1000:.0f} ms", flush=True)

    def _apply_model_timed(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):
        sigma = t
        if c_crossattn is None:
            raise RuntimeError("qf_native LTX-AV: no c_crossattn cond - wire the LTX-2.5 "
                               "gemma4 'with-proj' TE (CLIPTextEncode -> LTXVConditioning)")
        if control is not None:
            raise RuntimeError("qf_native LTX-AV: a ControlNet is wired, but this seam does "
                               "not consume comfy control hints - remove it.")
        # AV latent arrives nested OR sampler-packed flat (the H3-measured dual form).
        _nested = getattr(x, "is_nested", False)
        if _nested:
            streams = x.unbind()
            x_video, x_audio = streams[0], streams[1]          # [B,128,F,H,W], [B,8,L,16]
        else:
            ls = getattr(self, "latent_shapes", None)
            if not ls or len(ls) < 2:
                raise RuntimeError(f"qf_native LTX-AV: packed latent (type={type(x).__name__} "
                                   f"shape={tuple(getattr(x, 'shape', ()))}) but model.latent_shapes "
                                   "is unset - cannot unpack the AV pair.")
            n = int(math.prod(ls[0][1:]))
            xf = x.reshape(int(x.shape[0]), -1)
            x_video = xf[:, :n].reshape(list(ls[0]))
            x_audio = xf[:, n:].reshape(list(ls[1]))
        if x_audio.ndim != 4 or int(x_audio.shape[1]) != _LTXAV_AUDIO_CH \
                or int(x_audio.shape[3]) != _LTXAV_AUDIO_MEL:
            raise RuntimeError(f"qf_native LTX-AV: audio latent shape {tuple(x_audio.shape)} - "
                               f"expected [B,{_LTXAV_AUDIO_CH},L,{_LTXAV_AUDIO_MEL}] "
                               "(LTXVEmptyLatentAudio / LTXVConcatAVLatent)")
        dev = x_video.device
        xin = x_video.to(torch.bfloat16).contiguous()          # [B,128,F,H,W]
        B = int(xin.shape[0])
        cou = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        if B > 1 and (cou is None or len(cou) != B):
            raise RuntimeError(f"qf_native LTX-AV: engine forward is B==1 per cond group but got "
                               f"batch={B} with cond_or_uncond={cou}")
        self._derive_geometry(xin, transformer_options)
        La = int(x_audio.shape[2])
        self._audio_rows = La
        # RAW dual-proj ctx (no plugin connector — see _run_connector).
        vemb = self._run_connector(c_crossattn, attention_mask=kwargs.get("attention_mask")
                                   ).to(dev, dtype=torch.bfloat16).contiguous()
        if self._qf.current_session is None or getattr(self, "_qf_needs_begin", False):
            self._qf_needs_begin = False
            # shared black-video guard (see qf_modelpatcher.refuse_all_zero_initial_latent).
            qfmp.refuse_all_zero_initial_latent(xin, "LTX-AV")
            self._begin(xin[0:1].contiguous(), vemb[0:1].contiguous())
        _, C, F, H, W = xin.shape
        N = F * H * W
        if self._out is None or self._out.shape != (1, N, C):
            self._out = torch.empty((1, N, C), dtype=xin.dtype, device=dev)
        if self._out_audio is None or self._out_audio.shape != (1, La, _LTXAV_AUDIO_PACK):
            self._out_audio = torch.empty((1, La, _LTXAV_AUDIO_PACK), dtype=torch.float32, device=dev)
        sig_all = sigma.reshape(-1) if torch.is_tensor(sigma) else None
        step_index = self._sigma_step_index(sigma, sig_all, transformer_options)
        # [step-cache-key] the AV class has its OWN step loop (this one), so the base t2v
        # loop's uuid-derived key never runs here — extract uuids for THIS loop too (the
        # e1_w2-measured miss: AV sessions kept passing key=0 → EC silently disabled).
        cuuids = transformer_options.get("uuids") if isinstance(transformer_options, dict) else None
        out5d = torch.empty_like(xin)
        out_audio = torch.empty_like(x_audio)
        for i in range(B):
            _interrupt_poll_end_session_on_raise(self._qf)
            xi = xin[i:i + 1].contiguous()
            tokens = xi.reshape(1, C, N).transpose(1, 2).contiguous()          # [1,N,128]
            # pack audio [1,8,L,16] → rows [1,L,128] (d = c*16 + f — engine unpack order)
            arows = (x_audio[i:i + 1].float().permute(0, 2, 1, 3)
                     .reshape(1, La, _LTXAV_AUDIO_PACK).contiguous())
            vi = vemb[i:i + 1].contiguous()
            sig_i = float(sig_all[i].item()) if (sig_all is not None and sig_all.numel() >= B) else \
                (float(sig_all[0].item()) if sig_all is not None else float(sigma))
            p = qfe.DenoiseStepParams()
            ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
            p.struct_size = ctypes.sizeof(p)
            p.latent_in = tokens.data_ptr()
            p.velocity_out = self._out.data_ptr()
            p.velocity_out_capacity = self._out.numel() * self._out.element_size()
            p.dims = (ctypes.c_int * 5)(1, N, C, 0, 0)
            p.dtype = _qf_dtype(tokens.dtype)
            p.sigma = sig_i
            p.step_index = step_index
            p.total_steps = self._num_steps
            p.context = vi.data_ptr()
            p.context_dims = (ctypes.c_int * 3)(*vi.shape)
            p.context_dtype = _qf_dtype(vi.dtype)
            # [step-cache-key] uuid-symbolic key (was constant 0 = kNoCtxKey; the EC session
            # gate force-computes at 0 — the AV loop was the measured miss, see loop head).
            cuid = cuuids[i] if (cuuids is not None and i < len(cuuids)) else None
            p.cfg_context_key = self._ctx_key_assigner.key(cuid)
            if os.environ.get("QF_NATIVE_DEBUG_CTXKEY"):
                qfe.say(f"[qf_native] LTX-AV CTXKEY step={step_index} grp={i} "
                        f"cuuids_none={cuuids is None} cuid={str(cuid)[:8]} key={p.cfg_context_key}",
                        flush=True)
            mp = qfe.DenoiseStepMultiParams()
            ctypes.memset(ctypes.byref(mp), 0, ctypes.sizeof(mp))
            mp.struct_size = ctypes.sizeof(mp)
            mp.base = p
            mp.audio_latent_in = arows.data_ptr()
            mp.audio_velocity_out = self._out_audio.data_ptr()
            mp.audio_velocity_out_capacity = self._out_audio.numel() * self._out_audio.element_size()
            mp.audio_dims = (ctypes.c_int * 4)(1, La, _LTXAV_AUDIO_PACK, 0)
            mp.audio_dtype = _qf_dtype(arows.dtype)
            mp.audio_scale = 1.0            # LTX co-denoise: same-sigma, no carried variable
            self._call_denoise_step_multi(mp, f"LTX-AV denoise_step_multi[step={step_index},group={i}]")
            out5d[i:i + 1] = self._out.transpose(1, 2).reshape(1, C, F, H, W).to(xin.dtype)
            # unpack audio velocity rows [1,L,128] → [1,8,L,16] (exact inverse of the pack)
            out_audio[i:i + 1] = (self._out_audio.reshape(1, La, _LTXAV_AUDIO_CH, _LTXAV_AUDIO_MEL)
                                  .permute(0, 2, 1, 3).to(out_audio.dtype))
            self._qf.step_count += 1
            self._sess_denoise += 1
        self._step_i += 1
        self._qf.sampler_step_count += 1
        # calculate_denoised per lane on the SAME sigma (LTX joint schedule), returned in the
        # SAME form the sampler handed us (the H3-measured contract).
        if _nested:
            den_v = self.model_sampling.calculate_denoised(sigma, out5d.float(), x_video)
            den_a = self.model_sampling.calculate_denoised(sigma, out_audio.float(), x_audio)
            return comfy.nested_tensor.NestedTensor((den_v, den_a))
        Bx = int(x.shape[0])
        vel_flat = torch.cat([out5d.reshape(Bx, -1), out_audio.reshape(Bx, -1)], dim=1).reshape(x.shape)
        return self.model_sampling.calculate_denoised(sigma, vel_flat.float(), x)

    def process_latent_out(self, latent):
        # Finalize on the VIDEO half (the session's step pin is the packed video tokens;
        # t2av: engine finalize early-returns — buffer read-not-written), close the session,
        # then hand the ORIGINAL joint latent to comfy's LTXAV post-processing untouched.
        if self._qf.current_session is not None:
            try:
                # The final latent arrives NESTED or sampler-PACKED flat (the same dual form
                # _apply_model handles — MEASURED: process_latent_out runs BEFORE comfy's
                # un-pack, so the packed [B,1,N_total] shape reaches here). Extract the video
                # half either way.
                if getattr(latent, "is_nested", False):
                    lv = latent.unbind()[0]
                elif latent.ndim == 5:
                    lv = latent
                else:
                    ls = getattr(self, "latent_shapes", None)
                    if not ls or len(ls) < 2:
                        raise RuntimeError(f"qf_native LTX-AV: finalize got a non-nested ndim="
                                           f"{latent.ndim} latent and model.latent_shapes is "
                                           "unset - cannot extract the video half")
                    n = int(math.prod(ls[0][1:]))
                    lv = latent.reshape(int(latent.shape[0]), -1)[:, :n].reshape(list(ls[0]))
                if lv.ndim != 5:
                    raise RuntimeError(f"qf_native LTX-AV: finalize expected the 5D video latent, "
                                       f"got ndim={lv.ndim} (nested={getattr(latent, 'is_nested', False)})")
                lat5 = (lv[0:1] if lv.shape[0] > 1 else lv)
                Bc, C, F, H, W = lat5.shape
                N = F * H * W
                tokens = (lat5.reshape(1, C, N).transpose(1, 2)
                          .to(torch.bfloat16).contiguous())
                lib = self._qf.lib
                fp = qfe.DenoiseFinalizeParams()
                ctypes.memset(ctypes.byref(fp), 0, ctypes.sizeof(fp))
                fp.struct_size = ctypes.sizeof(fp)
                fp.latent = tokens.data_ptr()
                fp.latent_capacity = tokens.numel() * tokens.element_size()
                fp.dims = (ctypes.c_int * 5)(1, N, C, 0, 0)
                fp.dtype = _qf_dtype(tokens.dtype)
                fst = lib.quantfunc_denoise_finalize(self._qf.current_session, ctypes.byref(fp))
                if fst != qfe.QUANTFUNC_OK:
                    raise RuntimeError(f"qf_native: LTX-AV denoise_finalize failed: {qfe.last_err(lib)}")
                qfe.info(f"[qf_native] LTX-AV SESSION CLOSED after {self._step_i} sampler steps, "
                      f"{self._sess_denoise} denoise_step_multi calls, finalize=OK", flush=True)
            finally:
                self._qf.end_session_if_open()
        # No comfy-side frame-0 pin here (wan-align 2026-08-22): i2v conditioning is the
        # Inplace latent + noise_mask, and KSamplerX0Inpaint's per-step blend already leaves
        # the final x's conditioned frames at the clean latent — the decode input is right
        # by construction.
        return super(QFLTXModel, self).process_latent_out(latent)


FAMILY = "ltx2"

def _file_has_prefix(path, prefix):
    """Cheap safetensors HEADER probe: does any key start with `prefix`? Unreadable/not a
    safetensors => False. Module-level: used by the AV discriminant (vocoder.*) AND the
    [ltx25-allin] te_file relaxation (text_embedding_projection.*)."""
    import struct as _st
    try:
        with open(path, "rb") as fh:
            n = _st.unpack("<Q", fh.read(8))[0]
            if n > (1 << 31):
                return False
            hdr = json.loads(fh.read(n))
        return any(k.startswith(prefix) for k in hdr)
    except Exception:  # noqa: BLE001 - unreadable = no keys
        return False




def matches(pipeline_class, transformer_class=""):
    """The ENGINE's own detector (src/LTX2VideoPipeline.cpp ltx2_pipeline_detect):
        pipeline_class == "LTX2Pipeline"
    — an EXACT match with no transformer_class half, so this seam has none either."""
    return str(pipeline_class) == "LTX2Pipeline"


def register(deps):
    """Return the ltx2 family BUILDER. `deps` gives the package-level helpers (engine cache,
    liveness registry) without importing __init__."""


    def build(transformer1_path, bundle_dir=None, lora_entries=(), pinned_memory=False):
        """File-based (ComfyUI single-file) loading for LTX-2.5 — the shared staging pattern.
        CONNECTORS-SOURCE contract (user 2026-08-31 — fully support the transformer-only
        export; supersedes the 2026-08-22 one-file-only ruling):
        - transformer/  <- the transformer export (ALL-IN or transformer-only);
        - connectors/   <- resolved in priority order (a QUALIFYING source packs BOTH
          modality blocks — model.diffusion_model.{video,audio}_embeddings_connector.* —
          the joint-AV discriminant; a video-only source is rejected, see below):
            (1) the transformer file itself, when it qualifies (ALL-IN — the
                official-comfy-single-file mirror; byte-identical to the old path);
            (2) a same-dir sibling `*connector*.safetensors` that qualifies —
                the shared, recipe-independent completion file (extracted VERBATIM from
                the official comfy model file's connector keys; one copy serves every
                quantized transformer variant).
          NEITHER found -> fail loud naming the dir probed, the BOTH-modality
          requirement, and both remedies (a video-only-connector source additionally
          gets the joint-AV explanation — it must not fall through to the retired 2.3
          video-only path whose connector_ckpt widget no longer exists). The engine
          #565 comfy25 branch consumes either source identically (prefix detect ->
          rename+strip).
        text_embedding_projection is deliberately NOT required from the model file: the
        workflow's with-proj clip applies it TE-side (comfy lt.py dual_linear), the
        session cond arrives post-projection, and the engine loads it opportunistically
        (single-file first, te-dir fallback, else skip — LTX2VideoPipeline.cpp:376-397).
        AV capability = the packed audio_embeddings_connector in the RESOLVED connectors
        source (the engine's denoise_only has_audio_ probe + the _is_av discriminant
        below both read that source; every resolved source qualifies -> file-mode
        always arms AV); audio decode is the workflow's own audio
        VAELoader, never engine-side.
        Engine create runs denoise_only=True (VAE decode weights skipped; TE lazy)."""
        # CONNECTORS-SOURCE resolution (user 2026-08-31). The engine applies the
        # embeddings-connector tail itself on the unprocessed dual-proj cond — the
        # official comfy contract (lt.py marks TE output `unprocessed_ltxav_embeds`;
        # the model side owns the connector) — so a weight source MUST exist; no
        # workflow module can substitute it (the TE-side compat branch is legacy-shape
        # only and upstream-marked TODO:remove). The probe is CONTENT-based (header
        # prefix via _file_has_prefix), never name-based trust; the name filter merely
        # bounds which siblings get their header read.
        # A QUALIFYING source packs BOTH modality blocks (video AND audio) — the same
        # joint-AV discriminant `_is_av` reads downstream. LTX-2.5 file-mode is
        # joint-AV; a video-only-connector source cannot arm it, and accepting one
        # would fall through to the retired 2.3 video-only path whose connector_ckpt
        # widget no longer exists (an unactionable dead-letter refusal) — so it is
        # rejected HERE with an accurate message instead.
        _CONN_VID = "model.diffusion_model.video_embeddings_connector."
        _CONN_AUD = "model.diffusion_model.audio_embeddings_connector."
        def _qualifies(path):
            return (_file_has_prefix(path, _CONN_VID)
                    and _file_has_prefix(path, _CONN_AUD))
        conn_src = None
        if _qualifies(transformer1_path):
            conn_src = transformer1_path   # ALL-IN: self-link, byte-identical to the old path
        else:
            _dir = os.path.dirname(transformer1_path)
            try:
                _sibs = sorted(f for f in os.listdir(_dir)
                               if f.lower().endswith(".safetensors")
                               and "connector" in f.lower())
            except OSError:  # unreadable dir = no siblings; the raise below names it
                _sibs = []
            _qualified = [os.path.join(_dir, _f) for _f in _sibs
                          if os.path.join(_dir, _f) != transformer1_path
                          and _qualifies(os.path.join(_dir, _f))]
            if _qualified:
                conn_src = _qualified[0]   # deterministic: sorted-first
                if len(_qualified) > 1:
                    qfe.info(f"[qf_native] ltx2: {len(_qualified)} qualifying connectors "
                          f"completion files in {_dir!r}; using the alphabetically-first "
                          f"{os.path.basename(conn_src)} (others: "
                          f"{', '.join(os.path.basename(p) for p in _qualified[1:])})",
                          flush=True)
            if conn_src is None:
                _why = ("packs the video_embeddings_connector but NOT the audio one - "
                        "LTX-2.5 file-mode is joint-AV and needs BOTH - "
                        if _file_has_prefix(transformer1_path, _CONN_VID)
                        else "is a transformer-only export ")
                raise RuntimeError(
                    f"qf_native ltx2: {os.path.basename(transformer1_path)} {_why}"
                    f"and no connectors completion file was found "
                    f"next to it (probed {_dir!r} for *connector*.safetensors packing "
                    f"BOTH {_CONN_VID}* and {_CONN_AUD}*). The engine applies the "
                    f"embeddings-connector tail "
                    f"itself (official comfy contract: TE output is unprocessed_ltxav_embeds), "
                    f"so a source is required. Fix: place the shared ltx-2.5-connectors file "
                    f"in the same folder, or use an ALL-IN export (*-allin-*.safetensors).")
            qfe.info(f"[qf_native] ltx2: transformer-only export - connectors completion file: "
                  f"{os.path.basename(conn_src)}", flush=True)
        extra = {"connectors": conn_src}
        model_dir = qfmp.stage_denoise_only_package(bundle_dir, transformer1_path, extra_links=extra)
        return _build_from_package(model_dir, os.path.basename(transformer1_path),
                                   lora_entries=lora_entries,
                                   # pinned memory is the loaders' pinned_memory switch (user 2026-09-26, default
                                   # OFF), as for every family; LTX-2.5 no longer forces it on (it did since
                                   # 2026-08-23, for the two-stage flow's offload round trips)
                                   create_extra={"denoise_only": True}, pinned_memory=pinned_memory)

    def _build_from_package(model_dir, model_name,
              connector_ckpt="(none)", lora_entries=(), create_extra=None, pinned_memory=False):
        if connector_ckpt and connector_ckpt != "(none)":
            import folder_paths as _fp
            connector_ckpt = _fp.get_full_path_or_raise("checkpoints", connector_ckpt)
        else:
            connector_ckpt = ""
        # NOTE (CR simplicity): the joint audio+video (a2v) path needs an engine external-step
        # split (makeExternalStepFn) this seam does not implement, and since the widget was
        # removed there is no way to request it — so the old join_audio_prompt flag and its two
        # refusal branches are GONE (structurally unreachable code is not a guard). The AV path
        # below is LTX-2.5's own engine-side joint AV, which is a different mechanism.
        # ── LTX-2.5 JOINT-AV auto-detect (c5.8b): the SAME discriminant the engine's own
        # has_audio_ uses — the engine model_dir ships audio_vae/ weights (the video-only
        # 19B staging deliberately omits it) — so plugin and engine agree by construction.
        # AV → QFLTXAVModel (t2av; engine-side connector via av_unprocessed_ctx; NO
        # connector_ckpt needed). Video-only → the existing path, byte-unchanged.
        # CR conf-1: mirror the engine's FULL has_audio_ discriminant (LTX2VideoPipeline
        # :440): the audio DECODE chain staged (audio_vae weights + a vocoder — 2.3
        # separate dir or 2.5 bundled vocoder.* keys), OR — the denoise_only AV-lane
        # probe (user 2026-08-22 "官方工作流都是独立组件"): the staged connectors source
        # packs the audio_embeddings_connector blocks (the all-in single file, exactly
        # like the official comfy single-file; the split-layout connectors file carries
        # them too). The native seam never decodes audio engine-side (comfy's own
        # LTXVAudioVAEDecode does), so audio-VAE staging is NOT required for AV.
        # A mismatch either way would hit the engine's begin-time audio-lane refusal
        # (loud, but confusing); agreeing by construction avoids it. Decided once per staged package: a LoRA rebuild
        # of the same package gets the same answer.
        _avae_dir = os.path.join(model_dir, "audio_vae")
        _avae_files = ([os.path.join(_avae_dir, f) for f in os.listdir(_avae_dir)
                        if f.endswith(".safetensors")]
                       if os.path.isdir(_avae_dir) else [])
        _voc_dir = os.path.join(model_dir, "vocoder")
        _voc_ok = os.path.isdir(_voc_dir) and any(
            f.endswith(".safetensors") for f in os.listdir(_voc_dir))
        _voc_bundled = (not _voc_ok) and any(
            _file_has_prefix(f, "vocoder.") for f in _avae_files)
        _conn_staged = os.path.join(model_dir, "connectors", "model.safetensors")
        _is_av = (bool(_avae_files) and (_voc_ok or _voc_bundled)) or (
            _file_has_prefix(_conn_staged,
                             "model.diffusion_model.audio_embeddings_connector.")
            or _file_has_prefix(_conn_staged, "audio_embeddings_connector."))
        if _is_av:
            return qfmp.family_build(
                deps, model_dir, create_extra, comfy.supported_models.LTXAV,
                {"image_model": "ltxav", "disable_unet_model_creation": True}, QFLTXAVModel,
                f"[qf_native] loaded QuantFuncNativeLoader (LTX-2.5 JOINT-AV svdq) package={model_name} "
                f"capacity=native Prepared query (create deferred)", pinned_memory)(list(lora_entries))
        if not connector_ckpt:
            raise RuntimeError("QuantFuncNativeLoader: connector_ckpt (comfy LTX-2.3 ckpt with the "
                               "video_embeddings_connector) is required")
        # NOTE: the plugin-side a2v (joint audio+video) connector split is NOT implemented — it
        # needs an engine external-step split (makeExternalStepFn). There is no widget for it, so
        # it cannot be requested; _load_ltx_audio_connector + the combined-footprint accounting
        # helpers (_derive_connector_arch / _accumulate_connector_footprint, exercised by
        # connector_arch_derivation_test arm-9) are kept as the BOUND for that future work.
        # LTX svdq is PRE-quantized: pass the MINIMUM to create (minimal=True → empty config_json).
        # The svdquant metadata carries the layout/precision (embedded config: cross_attn_mod=True /
        # audio_cross_attn_mod=True / cross_attention_dim 4096 / num_heads 32 → 9-mod). ANYTHING
        # supplied on top (auto_optimize / height / width / a precision map) competes with it and the
        # engine resolves to the 19b default (connector 3840 / transformer 6-mod — the E2E run-1..8
        # create mis-size). The harness's whole svdq entry is {backend:svdq, model_dir}; mirror that.
        # RESIDUAL (tracked, §6.5 wan text_precision round): an EMPTY config also means the engine's
        # resolve-at-entry text-precision falls to its SM-DEFAULT tier (est::smDefaultTextPrecision:
        # fp4 on SM120+, int4 below). That default is what BROKE the wan loader (UMT5 rejects 4-bit);
        # it currently WORKS here because LTX's TE tiers accept the default — but if an LTX TE arch
        # without a wired 4-bit tier ever routes through this minimal create, it hits the same class.
        # No fix now (adding keys back defeats minimal=True's purpose); this note is the tripwire.
        # [19B non-gated connector] authoritative head count from the ORIGINAL model dir's diffusers
        # LTX2TextConnectors config (the 19B family ships NON-gated connector weights; the head split
        # lives ONLY here). Absent -> None; unreadable -> raises (_connector_config_heads).
        auth_heads = _connector_config_heads(model_dir)

        def _video_model(model_config, engine, device):
            # ★ PER-BUILD combined-footprint running total: the video connector AND (when the future a2v split lands)
            # the audio connector are held CONCURRENTLY, so their host allocs ADD. `conn_budget` is a fresh 1-element
            # list scoped to THIS build (NO cross-call/module state — a module accumulator would drift upward across
            # loads and false-reject legitimate configs after N runs). Each connector load adds its est to it and
            # REJECTS if the running total exceeds _MAX_CONNECTOR_TOTAL_BYTES before its eager build. audio_connector
            # is None: _run_connector returns the 4096 video-only width (ltx's RULED path).
            conn_budget = [0]
            connector = _load_ltx_video_connector(connector_ckpt, device, _budget=conn_budget,
                                                   authoritative_heads=auth_heads)
            return QFLTXModel(model_config, engine, connector, device=device, audio_connector=None)
        return qfmp.family_build(
            deps, model_dir, create_extra, comfy.supported_models.LTXV,
            {"image_model": "ltxv", "disable_unet_model_creation": True}, _video_model,
            f"[qf_native] loaded QuantFuncNativeLoader (LTX-2 svdq) package={model_name} "
            f"capacity=native Prepared query (create deferred)", pinned_memory)(list(lora_entries))

    return build
