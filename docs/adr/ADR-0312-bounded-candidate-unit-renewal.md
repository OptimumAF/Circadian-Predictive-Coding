# ADR-0312: Bound candidate unit renewal outside model state

## Context

R2.1 supplies native gradient history, while R2.2 requires mature-unit refresh
with protected units, stable identities, safe reset and an equal-capacity control.
Accepted tracked histories require original IDs and uniform ages. Renewal must
not reinterpret those snapshots or let native state restores renew permission
to spend replacements.

## Decision

Add explicit renewable native classes, heterogeneous history format2 and complete
snapshot format3. Freeze maturity, gradient threshold, event cap and replacement
seed in a validated policy. Select only eligible units and reset all affected
parameter/history/CPC fields atomically with deterministic per-new-ID RNG.

Reuse the existing owned learner, stream runner and actor/shadow consolidation.
A single-use application helper owns a finite permanent replacement allowance,
charged before private preparation. Snapshot restoration cannot refund it. The
helper reports publication independently from charges and stops after failure.

Why this: renewal changes unit identity and age semantics; explicit formats keep
accepted state meaningful. An external allowance separates continuation state
from authority without introducing another runner or persistence framework.

## Alternatives

- Reinterpret accepted tracked format2: breaks its original-ID/uniform-age contract.
- Put remaining allowance in snapshots: older restores could refund failed work.
- Reset serving actors directly: bypasses the existing candidate isolation.
- Select with heldout loss or fresh-probe gains: contaminates evaluation and policy.
- Introduce growing retired-unit histories: unnecessary for bounded continuation.

## Consequences

Default and tracked modes remain available. Renewal is fixed-width and bounded;
low gradient is explicitly a proxy, not demonstrated causal utility. In-memory
owned continuation is supported. Disk/capture, tracked observer composition,
combined rehearsal and comparative learning benefit are separate requirements.
Engineering evidence and criterion mapping: r22-bounded-renewal-20261016/RESULTS.md.

## Follow-up: isolated adaptation scenario, 2026-10-16

The existing attribution CLI selects a small composition adapter for three new
renewable arms. The old four no-renewal controls remain postrun analysis data;
training never opens their final/probe scores. An explicit renewable diagnostic
reader validates format3/history/policy/geometry/clock before FF/SVD and preserves
unknown parents. Existing disposable probes restore full continuation state but
run no renewal helper. The first frozen policy selected zero units; raw curves
equal controls and efficacy after reset remains untested. Evidence and costs:
r22-renewal-adaptation-20261016/RESULTS.md. No threshold change or retry follows.
