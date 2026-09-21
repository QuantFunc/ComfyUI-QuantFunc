# Host memory integration checkpoint

## 2026-09-21: retained native resource -> actual Comfy scheduler adapter

`QFNativeResourcePatcher` now exposes an already-existing `NativeResource` to
the actual host scheduling protocol. It derives CUDA device identity from
native metadata, forwards native aggregate occupancy and confirmed eligible
release counts, rejects non-Ready results, and saturates the host's oversized
release sentinel without uint64 wrapping. It contains no byte ledger, model
family formula or native model creation. The caller must strongly retain one
canonical adapter per resource; clone returns that same adapter. Default
QFModelPatcher and lazy/native creation paths remain unchanged and undeployed.

This is a non-loading occupancy dependency, NOT model capacity/demand/grants.
Full detach deliberately refuses even at Ready zero: a sampled zero does not
fence subsequent native growth. Automatic identity/lifetime binding and the
native eviction/restore/admission handshake remain required before rollout.

Remote `remote-native-resource-host-final-20260921.log`: 23 actual-Comfy CPU
contract tests pass (12 existing + 11 resource-adapter tests), plus the existing
13 NativeResource tests. Only the native C calls are doubled, not LoadedModel,
ModelPatcher or the host load/free loop. No GPU model inference is claimed.
Plugin syntax also passes Python3.9 grammar parsing. Source hashes:
qf_modelpatcher `6b16502a352580999436b06e35820e2b8762b734bd9d5290b2ee3331df19f8fa`;
host test `3aab54f7ab971837b66babffdf8c40f1cdaa4b4c3ffa9d1c1a40bf1ffc8106b3`.
Remote Comfy is not a Git checkout; tested model_management hash
`ddf5fe6399398d9101850d42604dce20eec0a6b6cba18135ea4ad5d3da3de99e`,
model_patcher `ac214200ea7991b9903c24f28457e2924357b025dbb9e353f1d3d2d397a9bd17`.

**Separate acceptance failure, not hidden in the passing count:** run the
same host test with `--probe-clone-handoff`. A native Busy result after official
detach(False) causes the host to lose its prior resource record. The explicit
probe fails with `[] != [original]`, recorded in
`remote-native-resource-host-green-handoff-20260921.log` (controller exit1).
Setting is_clone=False is not a workaround: actual repeat-load testing produced
four records instead of two because the host inserts reused records again.
The adapter therefore keeps the standard clone predicate. A separate request
for authority to fix an independent Comfy core worktree is pending. No core
edit, monkeypatch, private-list production mutation or deployment was made.
The final constructor revision also reproduces the same acceptance failure in
`remote-native-resource-host-handoff-probe-final-20260921.log`; its controller
checks the expected failing probe status, not a successful integration.
Independent scoped implementation CR is GO with no residual scoped blocker.
That verdict does not cover automatic integration or the final six reviews.

## 2026-09-21: native accounted-residency aggregate (not host integration)

`NativeResource.residency()` forwards the additive native query without summing
categories in Python. Ready zero is valid; Busy/Unknown/Closed have None bytes;
native/FFI errors propagate. Old V1 category/release operations remain usable
when the additional export is absent. The aggregate covers accounted CCA live,
cache, deferred and arena backing only, excludes overlapping pins, and is not
a capacity/demand/full-unload or complete-backend-coverage certificate.

Evidence: 13 Python tests pass on Python3.9 and default local Python; existing
16 host-reclaim tests pass. Remote final source/test hashes match and 13 tests
plus actual-library residency query pass. Actual engine host SHA256
`f00ae0ad901fbcc87d4a41dc35c1e68bfcb027bf71a292572d056f30039f2096`;
kernel `66a3fd49241dabe17d86a632d062ad8763a5d0b6d6f92aebfa3e59ff0346166c`.
Native validation: 64/64 tests, zero-error CUDA memcheck, fresh actual-DSO
C11/C++20 consumers, shared allocator/TLS identity and real split-runtime test.
Expanded tests cover real pinned/deferred/cache categories and checked overflow
through the actual C result-publication boundary. Scoped independent CR: GO.

No ModelPatcher/Shared registration path has been switched or deployed by this
slice. Complete unload/restore, full-request demand/grants, all-model execution,
the exact-ratio 768x448 -> 1920x1120 H3 double sample and final six-dimensional
review remain open. Prior full-suite failures below are not erased or resolved
by the narrow test results above.

## 2026-09-21: standalone retained-resource bridge (not host integration)

`qf_engine.NativeResource` wraps native ABI version1: independent owned/Shared
resource views, physical-category queries and eligible-only releases. Native
identity and categories are forwarded, with non-Ready byte fields represented
as None. This is neither model capacity nor demand, and it adds no byte sums,
ModelPatcher wiring, eviction policy or full-unload capability. Legacy methods
are unchanged. Missing ABI fails explicitly when the new factory is requested.

View close releases only the native CPU handle. Calls and close share a lock;
finalization retains the library, and failed/interrupted Python adoption returns
the native handle at most once. Acquisition still requires caller synchronization
with raw model destruction; an acquired native view survives model destruction.

Evidence under /home/jonathan/Documents/Codex/2026-09-19/new-chat/work/:

- `plugin-native-resource-py39-red-20260921.log`: actual Python3.9.22 import
  rejects initial union annotations. Replaced them with Optional; the matching
  `py39-green` log has11 passing tests, as does `supported-green`.
- Existing host-reclaim contract:16 tests also pass on Python3.9. No native
  compilation ran locally.
- `remote-resource-python-abi-interruption-20260921.log`:11 tests and actual
  ctypes/engine Shared query/zero-release/lifetime/error probe pass. Host SHA
  4e6bc4f31797a7344e4c2de1b896447459fb016ea3f1e49b2b95ea92d9d4571a;
  kernel SHA66a3fd49241dabe17d86a632d062ad8763a5d0b6d6f92aebfa3e59ff0346166c.
  The final annotation-only revision is rerun separately in the
  `remote-resource-python-abi-py39-compatible-20260921.log` record.
- Fresh unchanged plugin HEAD4685db1 archive baseline:13files7pass5fail1skip,
  three reported skipped arms. Modified suite:14files8pass5fail1skip, same arms
  and same five failing files (enhance_switch, host_loader, host_scheduler,
  qfltx_safety, reject_list_completeness). See `plugin-native-resource-{fresh-
  baseline,full-suite}-20260921.log`. Local Comfy cannot import
  comfy_aimdo.storage; other pre-existing test failures also remain. The
  loader_dispatch file prints SKIP but exits0: it is not exercised coverage.

These are scoped bridge checks, not successful owned-model Python acquisition,
nonzero Python-native reclaim, Comfy scheduler acceptance, or a clean full suite.
Full native eviction/restoration, exact demand/grants, canonical host hooks,
768x448→1920x1120 actual video and same-process acceptance remain open. No active
ComfyUI deployment/restart or attention changes accompany this checkpoint.

## Historical: 2026-09-20 host lifecycle work

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
