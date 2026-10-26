# ADR-0300: Retry failed coupled payload cleanup without reopening training

## Context

Busy cleanup leaves ownership and tokens intact until quiescence. Erasure failure
has different ordering: the lifecycle revokes declarations and invalidates tokens
before invoking trusted erasure, then stops candidates if that callback raises.

## Decision

Add a fixed manual-poll failure control. Bind one exact-learner validating,
one-shot failing erasure callback when the original lifecycle is constructed.
After RuntimeError, observe remaining owned raw references and public refusals,
then invoke the original bounded finish_cleanup once after removing the fault.

Why this: construction-time composition exercises real failure bookkeeping without
replacing live authority or weakening the normal supported erasure port. Keep the
fault callback in a dedicated adapter and share typed lifecycle composition.
Public default factories and previous modes retain their behavior.

## Alternatives

Replacing a callback after birth would obscure which authority was trusted.
Restarting a driver would renew limits. Reporting a completed purge as recovery
would hide that candidates remain stopped and the driver still records failure.

## Consequences

The failed attempt invalidates a 144-byte pending checkpoint, so owned raw bytes
fall512->368 before successful erasure; charged656 stays spent. Remaining inboxes,
metadata and caches clear on terminal retry. Caller aliases are outside the owned
ledger and no physical overwrite is promised.

Serving initially refuses lifecycle failure; after cleanup it refuses terminal
driver state. Candidate and stale-token guards retain their own precedence.
Successful cleanup clears pending data but preserves failed-driver RuntimeError
and stopped candidates. No training replay, allowance renewal, unlearning,
physical-latency, continuous-service or durable process-recovery claim follows.
