# Security Policy

We take the security of QuantFunc and its users seriously. Because QuantFunc ships a
closed-source native engine that runs on your machine, we hold ourselves to a clear,
public security process. This document explains how to report a vulnerability and what
to expect in return.

> **Companion documents:** [`doc/NETWORK.md`](doc/NETWORK.md) enumerates exactly what the
> engine sends over the network (and how to verify it yourself); [`doc/PRIVACY.md`](doc/PRIVACY.md)
> states what the license server stores.

## Reporting a vulnerability

Please report security issues **privately** — do **not** open a public GitHub issue for a
suspected vulnerability.

- **Email:** `security@quantfunc.com`
  *(⚠ PENDING MAINTAINER CONFIRMATION — confirm this is the real, monitored security
  address before this policy is published/shipped.)*

When reporting, please include:

- A description of the issue and its security impact.
- Steps to reproduce (proof-of-concept, affected version, OS/GPU, plugin + engine version).
- Any logs, crash traces, or packet captures relevant to the finding.
- Whether you intend to disclose publicly, and on what timeline.

If you wish to encrypt your report, request our PGP key in an initial (unencrypted) email
with no sensitive details.

## Our commitment (response SLA)

- **Acknowledgement:** within **3 business days** of your report.
- **Triage & initial assessment:** within **7 business days** (severity, scope, reproduction).
- **Status updates:** at least every **14 days** while the issue is open.
- **Remediation target:** a fix or mitigation for confirmed High/Critical issues within
  **90 days**; lower-severity issues on a best-effort schedule.
- **Coordinated disclosure:** we will agree a disclosure date with you and credit you (if you
  wish) in the release notes / a published advisory. We aim to publish an advisory once a fix
  is available.

## Scope

**In scope** (we want to hear about these):

- The plugin Python/JS wrapper in this repository (node layer, format adapters, the
  **auto-updater** in `auto_update.py`, model loaders, web assets).
- The engine's **network / authentication / key-map** path and the **auto-update download +
  verify-before-replace** flow (authenticity, integrity, signature handling).
- The model-obfuscation / key-map DRM handling.
- Memory-safety or injection issues at the worker ↔ engine IPC boundary (`worker.py`) and at
  any path/file handling (e.g. zip-slip, path traversal).
- Any behavior that contradicts the network/privacy guarantees in `doc/NETWORK.md` /
  `doc/PRIVACY.md`.

**Out of scope / please note:**

- The closed CUDA/quantization kernels are proprietary; you may report crashes, memory-safety,
  or correctness issues you observe, but we are not obligated to disclose engine internals.
- Vulnerabilities in third-party dependencies should also be reported upstream; we will track
  and patch our usage.
- Findings requiring a already-compromised host, physical access, or social engineering of the
  user are generally out of scope.

## Safe harbor

We consider security research conducted in **good faith** and in accordance with this policy to
be authorized. We will not pursue or support legal action against researchers who:

- make a good-faith effort to avoid privacy violations, data destruction, and service disruption;
- only interact with accounts/devices they own or have explicit permission to test;
- give us a reasonable time to remediate before any public disclosure;
- do not exploit a finding beyond what is necessary to demonstrate it.

If in doubt about whether your testing is in scope, contact us at the address above before
proceeding and we will work with you.

---

*This policy and the security contact will be finalized under the commercializing legal entity
prior to publication. (⚠ PENDING MAINTAINER CONFIRMATION of the legal entity name + the
`security@` address.)*
