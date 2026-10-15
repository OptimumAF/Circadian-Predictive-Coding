# ADR-0283 — Native failure accounting boundaries

Date: 2026-10-15

## Context

Existing runtime tests cover scalar mutation and late work accounting. Original
R3.7 also requires native failed-update behavior with stable actor availability.
The inbox distinguishes started-but-unacknowledged mutation from failure after
receipt insertion, and both RuntimeError paths stop further candidate operations.

## Decision

Add one fixed native PC/neutral/active integration for each boundary. A thin
owned wrapper executes the genuine native update before raising in uncertain
mode. In committed mode an actual completed observer raises after the original
budget and receipt have committed. Terminal notifications preserve the primary
error and serve the actor cache under the actual candidate operation gate.

Retain full stopped native/CPC/history, inbox, cursor, original budget and serving
evidence. Verify blocked retry, consolidation, public snapshot and label admission
without copying poison, invoking callbacks or changing revision/identity/quota.
Compare complete native after-state across fault positions and matched PC/neutral
raw parameters. Existing core/app semantics remain unchanged.

## Alternatives

Scalar-only fixtures omit complete native state. Suppressing a completion receipt
or patching runtime accounting would invalidate the boundary being tested.
Changing clocks to mimic a resource stop would exercise different documented
semantics. Automatic retry after uncertain mutation could duplicate work.

## Consequences

The runnable control makes physical work, completed accounting and applied
identity distinguishable. Private native reads are diagnostic only. The control
does not authorize uncertain training recovery, durable owner replacement or a
scientific comparison. Broader recovery and long-run requirements remain open.
