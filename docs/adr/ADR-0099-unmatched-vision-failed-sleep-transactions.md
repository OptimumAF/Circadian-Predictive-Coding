# ADR-0099: Preserve failed unmatched-vision sleep attempts at a retryable cursor

## Context

The v1/v2/v3 unmatched-vision runner saved a `before_sleep` checkpoint, but
an exception in its pre-guard, core, or post-guard left no attempt record.
The head and process random streams could also be advanced by a failed
attempt. An explicit restart therefore needed a truthful error history and
the same pre-attempt model state before retrying the completed wake epoch.

## Decision

Snapshot the head and Python, NumPy, Torch CPU, and selected Torch CUDA
process random streams around each sleep attempt. On an exception, restore
those states, describe the failed stage, retain any returned core proposal
with zero applied work, append a typed error, save the restored state at
`before_sleep`, and re-raise. The same process-RNG helper is shared with the
fixed-feature Torch runner because both use the same transaction boundary.

Count examples when each guard batch completes. Failed pre/post passes retain
only that completed prefix; a nonfinite aggregate retains all batches that
were scored. Known pre/post scores are recorded according to the failure
stage. Checkpoint preflight binds error order to the unresolved epoch,
selected guard-batch prefix, role/hash/metric, reason, and attempt counter
before model or process-RNG restoration. A later explicit resume may append
another same-epoch error or one resolved decision without repeating wake.

## Alternatives

- Drop failed attempts and retry from the earlier checkpoint. This hides
  attempts and gives reports a misleading denominator.
- Count a partially entered batch as scored. The evaluator has not produced
  a complete score for that batch, so the count would overstate exposure.
- Persist a partially changed head. It would make restart behavior depend on
  where the exception occurred and could change the baseline comparison.

## Consequences

Bounded CPU tests cover v1/v2/v3, both rollback metrics, pre/core/post and
unguarded failures, partial and nonfinite passes, repeated same-epoch errors,
strict JSON, sealed final test, model and baseline parity, random-stream
continuation, and malformed-history rejection before restore. Checkpoint
format 2 already requires the event sequence, so this adds no new file
format. Actual-device CUDA telemetry and continuation remain P3.10c4c; CPU
tests do not establish that gate.
