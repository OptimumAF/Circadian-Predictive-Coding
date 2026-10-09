# ADR-0200 — Own stable actor and shadow forks

## Context

Native learner ports and arrival/permission contracts are accepted. Native training
and restore may modify several arrays or fields before returning; sharing the same
model between serving and learning could expose a partial mutation. Consolidation
has the same risk. Actor outputs also need a stable version identifier before
promotion or live resource sharing is introduced.

## Decision

Extend the native port through a separate owned-fork protocol. Supported NumPy
adapters reuse their current complete owned constructors and native policy. From a
quiescent trusted source, keep a private stable actor fork and a private candidate
fork. Give actor reads their own serialized gate and detached versioned results.
Give all candidate operations a separate exclusive nonblocking gate and reuse the
existing arrived-event inbox and shared wake budget.

Pass detached candidate snapshots to trusted consolidation transforms; restore
only returned complete state and retain committed receipts. Bound lifetime IDs
and attempted transforms, including failures. Use existing soft wall/RSS boundaries
and refuse uncomposed limits. Expose inbox stopped status, and poison uncertain
native cancellation/partial failure while propagating the original exception.

## Alternatives

- Holding one lock across actor reads and candidate training would prevent partial
  reads but make serving wait for all learning and consolidation.
- Sharing native arrays without ownership would make stable version tags misleading.
- Handing a mutable candidate learner to callbacks would escape the write boundary.
- Parameter-only copies would omit chemistry, aliases, replay, topology or RNG.
- Automatic promotion or rollback would bypass still-unimplemented acceptance policy.

## Consequences

Concurrent actor reads are testable during partial native mutation and detached
native sleep, independently of an unproven circadian advantage. Source and snapshot
ownership are proven for the supported native adapters, not malicious custom Python
objects. Construction requires a quiescent source and an exclusively owned budget.
Actor copies cost memory; record bounds do not bound payload bytes. Nonblocking
candidate refusals need caller deferral. Native calls can overrun soft boundaries;
hard preemption, live latency fairness, promotion/rollback, privacy and scientific
source/final-release authority remain separate gates.
