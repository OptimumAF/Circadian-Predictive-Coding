# ADR-0236: Check original resources after payload copying

## Context

Complete managed capture called `before_final()` before acquiring the sampler
gate. That call could block on the original sampler lock. Its observations also
predated allocations made by the payload copy.

## Decision

Acquire the original sampler gate without waiting, before invoking its reader.
Keep that gate and every original owner gate held through capture. An internal,
expiring observation capability binds the reader, interval, gate, stop signal,
worker and calling thread. Reject terminal, unavailable, corrupt, reentrant or
changed sources. Preserve the original baseline and cumulative peak/count; reader
failure records its original exception and stops that sampler.

Use the original budget and progress before and after payload copying. Recheck
original retention and elapsed time as well. A refusal returns no capture and
keeps consumed copy charges and observations. Refresh budget/progress/sampler,
clock and lifecycle metadata using the same deepcopy memo: one detached payload
graph, followed by fresh observation records without a second array allocation.

Why this: admission and reported state must describe the same original resource
authority. Releasing gates or starting a replacement sampler would lose that
relationship. Shared inner records and the same memo preserve aliases across
the final budget/progress observations.

## Alternatives

Calling public sample/snapshot while holding its nonreentrant gate deadlocks.
Sampling before the gate can wait and does not protect a common interval.
Independent post-copy snapshots lose graph aliases and can reset authority.

## Consequences and limits

Public sampling and budget admission remain available with their prior behavior.
No new sampler or budget instance fields, dependencies or resource allowances
are introduced. The capture port changes to accept the app's shared copy memo.

Tests use bounded injected RSS/clock readings, real lock contention and joined
threads. They prove this admission path, not a hard physical process RSS ceiling
or arbitrary callback termination. Public background sampling remains a separate
telemetry path. Full pending/retired/native-variant and original source/consent
qualification, canonical composite bytes and recovery authorization remain open.

The post-payload check precedes the final bounded metadata refresh. This is not a
claim that memory cannot grow after the check or after capture returns.
