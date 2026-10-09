# ADR-0211: Bound owned payload copies with monotonic byte reservations

Date: 2026-10-07. Status: implemented optional byte policy; full R3.6 acceptance open.

## Context

Authorized ingress bytes and holder quotas miss retained checkpoint/model/inbox/
promotion/rollback/cache copies. Exact current-byte reclamation is unsafe while
retired, prepared, failed or external owners might still retain an array. Native
builders and arbitrary object hooks cannot be measured before opaque invocation.

## Decision

Add an original-authority lifetime allowance for attempted owned payload array
copies. Reserve before ingress, replay growth, checkpoint snapshots/preparation/
handoff, promotion metadata/model and cache copies. Never refund on failures,
discard, cleanup, GC or handoff. Observe retained raw replay/inbox/checkpoint
arrays plus supported auxiliary arrays under all-holder leases and verify they
fit the original charged allowance. Preserve original grants, ages, clocks,
budgets, sharing/consent hooks and identifiers. Reject alternate checkpoint wrappers.

Why this: monotonic reservations bound retained catalog bytes conservatively
without depending on uncertain object-lifetime/GC evidence. NumPy composition
supplies strict bounded no-copy graph ports and an inspectable source builder;
opaque builders/graphs are refused before copying/invocation. Legacy cleanup
without this explicit optional policy retains its behavior.

## Alternatives

- Treat ingress as the full retained-byte quota: fails when snapshots multiply data.
- Refund on discard/delete: another retired/prepared/failed owner can retain a copy.
- Call arbitrary serialization/deepcopy to measure: invokes hooks after admission
  should already have been decided and can allocate unknown graphs.
- Claim RSS/all-object memory bounds: wrong metric and unsupported allocator proof.

## Consequences

The allowance may exhaust while observed bytes are lower; callers cannot renew it
by recreating a controller, deleting records or collecting owners. Count supported
payload arrays;parameters/RNG/counters,temporary callbacks/native buffers,caller
copies,Python/string overhead and RSS remain outside the metric. Explicit ports
are trusted contracts,not source/graph security attestation. Automatic elapsed
purge remains unfinished;full R3.6b2b and its original parents stay unchecked.
