# ADR-0299: Finish committed-serving cleanup after a busy stop

## Context

Committed expiry clears both serving generations under quiescent ownership.
A real holder may still be busy when the caller requests a bounded stop. Existing
RetentionExpiryDriver exposes unfinished cleanup separately from a joined worker.

## Decision

Add an explicit busy committed-serving integration using the original driver's
stop and finish_cleanup ports. Hold the candidate lease, obtain a real busy poll,
start/wake/stop the worker, and require joined-but-pending cleanup. Release the
lease before measuring references or retrying cleanup. Public serving, checkpoint
restoration and rollback remain closed while pending. Successful finish invalidates
the original receipt and clears both owned generations without allowance renewal.

Why this: existing ports already express the required bounded retry semantics.
A fixed phase exercises the actual ownership boundary without another scheduler
or cleanup mechanism. Pending and committed automatic-expiry controls remain.

## Alternatives

Ignoring stop's Boolean result could leave retained payloads unreported. Starting
a replacement driver would renew authority illegally. Repeated unbounded retries
would hide contention rather than demonstrate a resumable bounded terminal state.

## Consequences

Stopped is not equivalent to cleanup complete. The caller must retain original
ownership and retry finish_cleanup only after join and holder release. Pending
rollback refuses terminal access; it becomes stale only after cleanup invalidates
its receipt. Native state and historical receipts remain; raw cleanup does not
unlearn parameters, overwrite RAM or promise physical latency/process recovery.
