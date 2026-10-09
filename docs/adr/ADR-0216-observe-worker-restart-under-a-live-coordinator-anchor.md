# ADR-0216: Observe worker restart under a live coordinator anchor

## Context

Typed recovery metadata can reject inconsistent relationships but supplies no
authentic OS time or process liveness. A PID lookup after a worker dies can lose
the original object or inspect PID reuse. Same-process checkpoints still retain
original Python budget/clock/RSS handles, and no durable native restore exists.

## Decision

Use documented Windows query/synchronize handles registered while processes are
alive, retain their process objects across worker exit and check creation identity.
Use an original live coordinator handle as the epoch anchor for the first worker-
restart policy. Observe precise biased interrupt time in nanoseconds and current/
peak absolute RSS, checking the anchor before and after observations. Unsupported
APIs, missing/ended anchors, invalid/backwards time, invalid RSS and unknown wait
states refuse and permanently disable the observer. Close each owned registration
on failure; report native close failures without claiming cleanup succeeded.

Why this: actual process objects support liveness after exit; time must keep the
original OS origin and include downtime. The inner observation port returns facts
without manufacturing an owner fence. Independent transactional authority must
bind those facts and preserve original policy/resources through publication.

## Alternatives

Caller epoch strings, newly reset clocks, PID-only lookup and timeout-based death
claims cannot preserve original authority. Undocumented boot GUID structures are
not made a supported API here. Coordinator restart cannot silently adopt a new
anchor and preserve the old epoch. Unbiased time would exclude suspend downtime.

## Consequences

Initial support is worker restart while the original coordinator survives. This
does not complete full original R3.5b2 or R3.7, and coordinator loss/reboot/native
durable recovery remain explicit unfinished work. All original acceptance is
preserved. Deterministic API faults precede one bounded real child exit observation;
no model work, dependencies or scientific baseline/seed/metric changes occur.

Structure, official primary sources, example, exact commands and limits are in
`docs/windows-recovery-observation.md`. Local source/evidence binding is in
`artifacts/runs/r35b2b-observation-20261007/`.
