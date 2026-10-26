# ADR-0599 — Explicit causal shadow TRAIN diagnostics

## Context

The fitted chooser82 exactly reproduced drift's stalled repair. Its deployed
actor feedback was clean even though the trained shadow had regressed on old
TRAIN. Existing snapshots and the observation port can expose this gap without
another runtime or access to evaluation outcomes.

## Decision

Add an opt-in fixed-drift variant that scores only the oldest pending packet with
arrived TRAIN labels on a detached candidate. Record original label times and
current post-label diagnostic time separately, bind candidate version/update
counters and validate unchanged state around the callback. Charge every attempted
forward, reconstruction, copy and repeated read after deferral. Leave defaults,
fixed controls and fitted policies unchanged.

Why this: the observed failure concerns hidden candidate errors, so changing the
observation is justified before fitting another chooser or broadening streams.

## Alternatives

Using live task outcomes or final/probe labels would violate isolation. Quietly
changing baseline drift would erase the negative control. Transplanting82's fit
would change its observation semantics. Additional capture/provenance machinery
would not resolve the missing signal. None is part of this change.

## Consequences

More informative TRAIN feedback adds computation and memory costs. Candidate
accuracy remains distinct from guarded publication and closed-loop success. The
new development variant is not a generic controller or an efficiency/CPC claim.
No source/budget authority is restored or renewed. Distinct stream validation and
long deployment remain separate unmet requirements.
