# ADR-0203: Admit candidate work at complete native update boundaries

## Context

R3.4 owns separate complete serving/candidate state. ExperienceInbox drains every
ready pair in a single poll. Checking contention once before that drain cannot
respond to a serving arrival or pause between subsequent native updates. Complete
runtime checkpoints and actual live serving latency measurements are still absent.

## Decision

Add an opt-in wrapper and short-lock gate around actual actor serving calls and
individual candidate native updates. Bound active/queued serving admission and
updates per poll, preserve a lifetime admitted-work quota, and admit new work only
without serving contention/manual pause/active training or resource refusal. Call
the external exact-bool resource probe outside the gate lock, then recheck state.

The inbox accepts an optional per-update exact-bool context and poll limit, keeping
its default behavior. Denied work remains pending before payload/native access.
Charge admitted attempts before native work; release leases on all failures and
keep consumed quota/committed identities. Active native work completes; pause is
cooperative. Serving never takes the candidate gate. Snapshot/poll records are
observations, not durable complete checkpoints or budget renewal authority.

## Alternatives

One pre-drain check misses later contention. Locking serving behind candidate work
would defeat actor isolation. Cancelling arbitrary native operations mid-mutation
would poison state and require algorithm-specific safe cursors. A hidden scheduler
or OS priority layer adds dependencies and unproven performance behavior.

## Consequences

Explicit admission supplies deterministic priority/defer behavior and bounded work
without changing native equations or role/arrival rules. Outer callers must use the
wrapper consistently; the original APIs remain independently available. Resource
callbacks are trusted monitors and existing budget checks remain soft complete-
boundary limits. Native GIL/CPU/memory contention may still delay serving. R3.5b must
add complete supported checkpoint/restore before R3.5c measures actual matched idle/
training actor p50/p95. The original R3.5 parent stays open until both gates pass;
isolated timing or a cooperative gate cannot establish live latency performance.
