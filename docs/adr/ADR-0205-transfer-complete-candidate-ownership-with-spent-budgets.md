# ADR-0205: Transfer complete candidate ownership with spent budgets

## Context

Restoring weights alone loses pending labels, consumed IDs and consolidation
attempts. Constructing a fresh runtime budget renews work and wall-time quotas.
Live clocks/resource samplers and pending promotion tickets also belong to an
actual owner; they are not safely reconstructed from an inbox DTO.

## Decision

Support complete same-process checkpoint and replacement through one retained
shared-runtime handle. Use bounded identity tokens and independent native
preparation under candidate/paused-sharing leases. Retain the original budget,
event/wall clocks, progress/RSS sampler, gate/resource probe and actor. Validate
native state/policy, complete inbox/consolidation history and stale revisions;
publish one new owner while retiring the old candidate under the actor read gate.
Current pending promotion authority is never transferred. Stopped state stays
stopped; preparation attempts are lifetime bounded and never refunded.

## Alternatives

An in-place weights restore leaves old owner references capable of duplicate
work. A fresh budget/clock loses cumulative accounting. Public serialized Python
graphs provide neither native source authority nor portable monotonic/RSS state.
Cross-process recovery requires a separate explicit resource/time policy.

## Consequences

Supported handoff resumes exactly without native replay and preserves serving.
Failed preparation leaves old candidate native/history usable, though resource
telemetry can advance normally. Exclusive trusted ownership is required; private
field mutation is unsupported. Full latency measurements remain R3.5c and durable
restart recovery remains separate unfinished work.
