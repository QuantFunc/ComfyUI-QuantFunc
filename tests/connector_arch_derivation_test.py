#!/usr/bin/env python3
"""Death-rule test for `_derive_connector_arch` (qf_ltx_modelpatcher.py) -- the checkpoint-driven LTX
connector arch derivation + its vuln-CR SECURITY BOUNDS.

WHY THIS EXISTS: the fix replaced hardcoded connector dims (num_layers / num_heads / head_dim / num_registers)
with values DERIVED from an untrusted `connector_ckpt` string node input. Those values feed comfy's
`Embeddings1DConnector`, which EAGERLY builds real transformer blocks/Linears/registers in __init__ BEFORE
load_state_dict -- so the 0/0 load guard cannot intercept an oversized value, and a hostile checkpoint drives
a host-RAM-overshoot SIGKILL (#543 class; block Linears are O(inner_dim^2)). Deriving from untrusted input
replaces a constant with a control surface, so EVERY such value must be magnitude-bounded AT the derivation
site, before construction. This test pins:
  * the DEPTH bound (n_layers) + its floor (n_layers=0) + the ^-anchor (decoy key),
  * the WIDTH bounds (inner_dim / num_heads / n_registers -- self-CR Reviewer A),
  * the NON-GATED refusal (head split not derivable without the gate -- self-CR Reviewer B),
  * the head-split-mismatch reject (Finding #1), and the real video + AUDIO shapes.

It extracts the REAL on-disk function via AST (the plugin uses relative imports, so a plain import fails) and
drives it with synthetic state_dicts (mock tensors carrying only `.shape` -- no real allocation). Run directly:
    python3 tests/connector_arch_derivation_test.py
"""
import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# QF_CONNECTOR_TEST_SRC lets a reviewer point this at a MUTATED copy of the source (e.g. the combined running-total
# check stripped) to prove the arms are PROVEN-ABLE-TO-FAIL (RED on mutation), not guaranteed-fail fixtures. Default
# = the real on-disk source.
_SRC = os.environ.get("QF_CONNECTOR_TEST_SRC") or os.path.join(_HERE, "..", "qf_ltx_modelpatcher.py")
_CEILINGS = ("_MAX_CONNECTOR_LAYERS", "_MAX_CONNECTOR_INNER_DIM", "_MAX_CONNECTOR_HEADS",
             "_MAX_CONNECTOR_REGISTERS", "_CONNECTOR_BLOCK_LINEAR_MULT", "_MAX_CONNECTOR_TOTAL_BYTES")


def _extract_fn(src_text, name):
    for node in ast.parse(src_text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src_text, node)
    raise RuntimeError(f"{name} not found in qf_ltx_modelpatcher.py -- the fix was removed/renamed")


def _extract_const(src_text, name):
    for node in ast.parse(src_text).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"{name} not found -- a security ceiling was removed")


class _T:  # a mock tensor: the helper only ever reads `.shape` (no real allocation)
    def __init__(self, *shape):
        self.shape = shape


def _blocks(n, heads=32, inner=4096, gated=True, regs=128):
    """A synthetic connector state_dict: n real transformer_1d_blocks, gated by default, learnable_registers
    of shape [regs, inner]. Real video = (8, 32, 4096, gated, 128); real audio = (8, 32, 2048, gated, 128)."""
    sd = {"learnable_registers": _T(regs, inner)}
    for i in range(n):
        sd[f"transformer_1d_blocks.{i}.attn1.to_q.weight"] = _T(inner, inner)
        if gated:
            sd[f"transformer_1d_blocks.{i}.attn1.to_gate_logits.bias"] = _T(heads)
    return sd


def main():
    src = open(_SRC).read()
    consts = {c: _extract_const(src, c) for c in _CEILINGS}
    max_layers = consts["_MAX_CONNECTOR_LAYERS"]
    ns = dict(consts)  # the functions reference these module globals
    # exec the 3 REAL functions into ONE namespace so they call each other (_derive_connector_arch ->
    # _connector_est_bytes; _accumulate_connector_footprint -> _connector_est_bytes). `re` for the derive regex.
    exec("import re", ns)  # noqa: S102 -- trusted repo source
    for _fn in ("_connector_est_bytes", "_accumulate_connector_footprint", "_derive_connector_arch"):
        exec(_extract_fn(src, _fn), ns)  # noqa: S102 -- trusted repo source
    derive = ns["_derive_connector_arch"]
    accum = ns["_accumulate_connector_footprint"]
    bad = [0]

    def check(label, cond, detail=""):
        print(f"  [{'OK ' if cond else 'FAIL'}] {label} {detail}".rstrip())
        bad[0] += 0 if cond else 1

    def rejects(label, sd, needle):
        try:
            derive(sd, label)
            print(f"  [FAIL] {label}: NOT rejected (should REJECT before eager construction)"); bad[0] += 1
        except RuntimeError as e:
            ok = needle in str(e)
            print(f"  [{'OK ' if ok else 'FAIL'}] {label}: REJECTED ({str(e)[:52]}...)"); bad[0] += 0 if ok else 1
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {label}: raised {type(e).__name__}, expected RuntimeError"); bad[0] += 1

    # 1) real 8-block gated VIDEO connector -> (n_layers, num_heads, head_dim, n_registers)
    try:
        check("real video 8-block", derive(_blocks(8), "video") == (8, 32, 128, 128),
              f"-> {derive(_blocks(8), 'video')} (expect (8, 32, 128, 128))")
    except Exception as e:  # noqa: BLE001
        check("real video 8-block", False, f"raised {e!r}")
    # 1b) real AUDIO shape (inner=2048 -> head_dim=64) -- Reviewer A: audio shape was untested
    try:
        check("real audio (inner=2048)", derive(_blocks(8, inner=2048), "audio") == (8, 32, 64, 128),
              f"-> {derive(_blocks(8, inner=2048), 'audio')} (expect (8, 32, 64, 128))")
    except Exception as e:  # noqa: BLE001
        check("real audio (inner=2048)", False, f"raised {e!r}")

    # 2) DEPTH vuln: a hostile huge block index -> REJECT before construct (eager ~400 GB alloc)
    sd = _blocks(8); sd["transformer_1d_blocks.499.attn1.to_q.weight"] = _T(4096, 4096)
    rejects("hostile depth n_layers=500", sd, "500")

    # 2b) WIDTH vuln (self-CR Reviewer A -- n_layers in-bound but hostile width feeds the SAME eager __init__):
    rejects("hostile inner_dim=1M", _blocks(8, inner=1 << 20), "inner_dim")           # block Linears O(inner^2)
    rejects("hostile num_heads=4096", _blocks(8, heads=4096, inner=4096), "num_heads")  # divides 4096, still huge
    rejects("hostile n_registers", _blocks(8, regs=99999), "num_registers")           # learnable_registers height

    # 3) FLOOR: connector keys but ZERO transformer_1d_blocks -> n_layers=0 -> REJECT
    rejects("zero-block n_layers=0", {"learnable_registers": _T(128, 4096),
                                      "x.attn1.to_gate_logits.bias": _T(32)}, "n_layers=0")

    # 4) NON-GATED (self-CR Reviewer B): no to_gate_logits -> head split NOT derivable -> REFUSE, don't guess
    #    (a fallback would silently mis-group attention, e.g. a 30-head/3840 connector guessed at 32 -> (32,120))
    rejects("non-gated connector", _blocks(8, gated=False), "NON-GATED")

    # 5) HEAD-SPLIT MISMATCH (Finding #1 reject branch): inner_dim not divisible by num_heads -> REJECT
    rejects("head-split mismatch (4097/32)", _blocks(8, heads=32, inner=4097), "not divisible")

    # 6) ANCHOR: a `not_transformer_1d_blocks.499.` decoy must NOT register a phantom block (^-anchored regex)
    sd = _blocks(8); sd["not_transformer_1d_blocks.499.foo"] = _T(1)
    try:
        nl = derive(sd, "decoy")[0]
        check("decoy anchor", nl == 8, f"-> n_layers={nl} (expect 8)")
    except Exception as e:  # noqa: BLE001
        check("decoy anchor", False, f"raised {e!r} (anchor let it through as 500)")

    # 7) DEPTH ceiling BOUNDARY: exactly at the ceiling passes; one above rejects (REJECT-not-clamp)
    try:
        derive(_blocks(max_layers), "ceil"); check(f"at depth ceiling n_layers={max_layers}", True)
    except RuntimeError:
        check(f"at depth ceiling n_layers={max_layers}", False, "wrongly rejected")
    rejects(f"over depth ceiling n_layers={max_layers + 1}", _blocks(max_layers + 1), "outside")

    # 8) PER-CONNECTOR PRODUCT CEILING (native-connector-combined-footprint-ceiling): per-axis bounded != PRODUCT
    #    bounded -- drive the REAL derive() BOTH ways, each axis individually LEGAL in both directions:
    #    OVER -- all 4 axes AT their max bound (16 / 8192 / 64 / 256), each legal, SINGLE-connector product ~24 GiB
    #    -> REJECT before the eager __init__ (a #543 host-RAM SIGKILL on the CPU/no-GPU path).
    rejects("all-axes-max product ~24GiB", _blocks(max_layers, heads=consts["_MAX_CONNECTOR_HEADS"],
            inner=consts["_MAX_CONNECTOR_INNER_DIM"], regs=consts["_MAX_CONNECTOR_REGISTERS"]), "SINGLE-connector")
    #    WITHIN -- a wide 4-layer/8192 connector (~6 GiB < the 8 GiB ceiling): all axes legal AND product under ->
    #    ACCEPT (proves the ceiling does NOT false-reject a per-axis-large-but-product-ok single connector).
    try:
        r = derive(_blocks(4, inner=8192), "wide-within")
        check("within: 4L/8192 (~6 GiB) accepted", r == (4, 32, 256, 128), f"-> {r}")
    except Exception as e:  # noqa: BLE001
        check("within: 4L/8192 (~6 GiB) accepted", False, f"raised {e!r}")

    # 9) PER-LOAD COMBINED CEILING (the agg-NO-GO fix): load() holds video + audio CONCURRENTLY (join_audio_prompt=
    #    True, a shipped switch), so their host allocs ADD -- per-CONNECTOR bounded != per-LOAD bounded. Drive the
    #    REAL _accumulate_connector_footprint through a per-load budget[0], BOTH ways. This is a negative control by
    #    execution: it drives the PRODUCTION function, so if the running-total check were removed the OVER arm's 2nd
    #    accumulate would NOT raise and this arm reds -- it is PROVEN able to fail, not a guaranteed-fail fixture.
    #    OVER -- two each-LEGAL connectors (12L/4096 ~4.5 GiB each, single < 8 GiB) -> combined ~9 GiB > 8 -> REJECT
    #    on the SECOND (exactly the join_audio_prompt=True pair the load() path builds).
    budget = [0]
    try:
        accum(budget, 12, 32, 4096, 128, "video_embeddings_connector")   # 1st ~4.5 GiB: within, no raise
        accum(budget, 12, 32, 4096, 128, "audio_embeddings_connector")   # 2nd: pushes ~9 GiB -> REJECT
        check("combined join_audio 2x4.5GiB -> REJECT", False, f"NOT rejected (budget={budget[0]})")
    except RuntimeError as e:
        check("combined join_audio 2x4.5GiB -> REJECT", "COMBINED" in str(e), f"({str(e)[:46]}...)")
    #    WITHIN -- real video (~3.0 GiB) + real audio (~0.75 GiB) = ~3.75 GiB < 8 -> BOTH accumulate, no raise
    #    (proves the per-load guard does NOT false-reject a legitimate join_audio_prompt pair).
    budget = [0]
    try:
        accum(budget, 8, 32, 4096, 128, "video_embeddings_connector")    # real video ~3.0 GiB
        accum(budget, 8, 32, 2048, 128, "audio_embeddings_connector")    # real audio ~0.75 GiB
        check("combined real video+audio (~3.75 GiB) -> ACCEPT",
              0 < budget[0] < consts["_MAX_CONNECTOR_TOTAL_BYTES"], f"-> {budget[0] / 1024 ** 3:.2f} GiB")
    except Exception as e:  # noqa: BLE001
        check("combined real video+audio (~3.75 GiB) -> ACCEPT", False, f"raised {e!r}")

    print("CONNECTOR_ARCH_DERIVATION: " + ("PASS" if bad[0] == 0 else f"FAIL ({bad[0]} wrong)"))
    return 0 if bad[0] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
