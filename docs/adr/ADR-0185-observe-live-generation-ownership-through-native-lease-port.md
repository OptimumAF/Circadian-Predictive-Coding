# ADR-0185: Observe live generation ownership through a native lease port

## Context

H binds complete V2 physical request metadata but owner strings cannot prove a
live exclusive holder. Existing V14 uses native local OS locks and belongs to the
unchanged original150-source closure. Ownership lifetime is distinct from actual
loaded-code/import closure, sequential arrivals, source independence and admission.

## Decision

Add immutable full ownership scope/point-in-time observations and an inner live
claim/observe protocol. App composes the unchanged complete H/g preflight before
claim, rereads after acquisition and checks all physical files/owner around yield
and every success/failure exit. Outer adapter uses nonblocking Windows byte lock/
POSIX flock in a single configured canonical local registry, retaining a permanent
single-link regular one-byte file. Key by whole inner generation request so changes
to outer owner/file identities cannot evade contention. Never unlink/reclaim the
lock path. Bind actual device/inode/scope/nonce/monotone UTC and sequence while
held; reject released/detached/aliased/drifting/foreign/type-coerced observations.
Nested finally checks preserve owner verification on physical failure and native
release on all exceptions; OS release is verified with actual process death.

Why this: a focused native lifecycle gate is necessary before execution and can
be verified without changing scientific settings or existing source interfaces.
Keep immutable observations explicitly historical outside the live context. The
native registry is cooperative/local, not a global/distributed source fence.

## Alternatives

Treat owner declaration/file existence as live ownership: no held handle observed.
Use outer request hash/owner as the lock key: differing outer copies bypass the
same complete generation request. Delete stale lock files on release/reclaim: a
new inode can admit another holder while the old handle remains locked. Modify
V14/exhausted original closure or fold owner into runtime proof: breaks preserved
evidence or claims code execution from a lock. None adopted.

## Consequences

Real full V2/native/same-process/subprocess/death/error/corruption/time/late-drift
fixtures run under raising science guards; native standard modules and UUID/time
are non-scientific metadata operations. All49 older files/original150 closure and
parent criteria remain. No dependencies, baseline/metric/source/RNG/algorithm/
count/cap changes or experiments. Runtime closure, actual arrivals/assignment/
chronology, prior effects/unrecycled independence/resource/repeat/b3 gates stay
required; fresh/execution/precision false and independence unknown. i alone can
complete after full behavioral/static/whole-preservation/terminal acceptance.
