# Host memory integration checkpoint, 2026-09-20

This is an undeployed integration branch, not an ALL PASS or H3 acceptance claim.
Base for this checkpoint: a626fa8c6f4de4c0fde011845d9c4fd1aed83b33.

The Python bridge now requires `quantfunc_unload_sync_ex(pipeline, uint64_t*)`
for counted full reclamation. It returns confirmed native operation bytes, not
file footprint, and does not treat a previous unload flag as proof of zero
current resources. A matching native build is required; older libraries fail
explicitly at this operation. Partial errors also propagate; partial shortfall
is left to ComfyUI LoadedModel's existing decision rather than an adapter-side
full-release fallback.

Explicit detach completes native reclamation synchronously before calling the
official base detach. Measured residual residency refuses successful removal.
Clone detach(False) preserves residency and official callbacks. The private
120-second timer and its per-family cancellation hooks are removed.

## Evidence

Use COMFY_ROOT=/media/jonathan/Data/ComfyUI and Python
/home/jonathan/anaconda3/envs/comfy/bin/python for the host test/suite.
The host test uses actual ComfyUI ModelPatcher, LoadedModel, free_memory and
PromptExecutor in a CPU-only process. Native calls/free-device query are doubled;
this is not CUDA or HTTP server acceptance.

- `tests/host_reclaim_contract_test.py`: 13 tests; RED 8 failures, GREEN 13 pass.
  Logs `/media/jonathan/Data/temp/qf-release-bytes-{red,green}-20260920.log`.
- `tests/host_scheduler_contract_test.py`: partial decision, synchronous full
  eviction, clone callback, object-patch restore, failure registration and next
  prompt recovery. Timer-stage RED 3 failures; residual-stage RED 1 failure;
  final GREEN 10 tests. Logs `qf-host-detach-red-20260920.log` and
  `qf-detach-residual-{red,green}-20260920.log` in the same directory.
- Session-retention test now requires an explicit exception, not a successful
  zero, for an unendable session; still verifies no native reclaim and retained
  pointer. All its arms pass.
- Full suite: 12 files, 8 pass / 4 fail / 0 skip. The same four failing files as
  the preservation baseline remain: enhance_switch, loader_dispatch, qfltx_safety,
  reject_list_completeness. Log `qf-host-lifecycle-final-20260920.log`. No claim of
  a clean full suite; these failures have not been deleted to make it green.

## Remaining deployment gates

Native currently exposes device-scope counters/reclaim. Multiple engines and
the legacy primary/shadow split still need one consistent ownership contract;
device totals must not be charged to each engine. Synchronous primary eviction
is not proof that shared-view protection is solved. The residual check can
refuse detach when another same-device allocation remains: safe failure is
intentional until the ownership contract is complete, not a final user workflow.

The actual ComfyUI clone pop-before-handoff exception window remains; no core
changes or private-registry monkey patch are included. Full-request future peak
estimation and host load-budget enforcement remain unfinished. Exact native ABI
build/test, final-SHA six-dimensional independent reviews and remote H3 0.3→0.9
same-process acceptance are required before deployment.

Source and user changes remain preserved in both independent worktrees and the
previous verified local/remote backup archives. No rsync or remote restart has
been performed by this checkpoint.
