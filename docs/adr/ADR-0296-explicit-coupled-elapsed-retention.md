# ADR-0296: Explicit coupled elapsed retention

## Context

Coupled learner payload measurements and retained-copy accounting are available.
The basic and copy lifecycle factories explicitly reject elapsed policies.
The application already owns deadline checks, cleanup and a bounded expiry worker.

## Decision

Add a separate adapter factory requiring an explicit elapsed policy. Supply the
existing coupled measurements to ManagedDataLifecycle. The caller explicitly
constructs and starts the existing RetentionExpiryDriver; the factory creates no
thread and does not renew any clock or allowance. Preserve the old factory refusals.

Why this: composition keeps state ownership and concurrency in the established
application layer and makes worker lifetime visible to the caller.

The runnable synthetic control uses one original injected elapsed clock through
handoff, refuses overdue access before native work, then starts a real worker at
the deadline. It waits within the declared bound and always stops/joins in finally.
Stop closes public access permanently even after successful cleanup.

The caller explicitly wakes the worker after starting it: its first poll can
encounter the startup operation lease. Poll/shutdown interleaving can produce
additional empty cleanup passes. Validate the original driver's four-poll and
six-cleanup-attempt limits rather than an incidental exact number of purges.

## Alternatives

Extending the copy factory would silently expand its contract. Starting a thread
inside the lifecycle factory would hide shutdown responsibility. A new scheduler
or cleanup algorithm would duplicate existing application behavior.

## Consequences

Cleanup drops owned raw references conservatively across retired/current inboxes
and pending checkpoints. Learned state, receipts and spent byte/update allowances
remain. Caller copies, physical RAM overwrite, unlearning and process recovery
are outside this guarantee. Injected elapsed time validates boundary semantics;
the real local worker validates scheduling and shutdown, not physical deadline
latency or continuous service. Broader research/runtime requirements remain open.
