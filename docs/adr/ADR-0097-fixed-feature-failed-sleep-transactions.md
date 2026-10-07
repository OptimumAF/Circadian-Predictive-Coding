# ADR-0097: Keep failed fixed-feature sleep attempts at a retryable cursor

## Context

A fixed-feature Torch guard or core sleep call can fail after wake training,
and a post-guard failure can occur after the core has returned a proposal.
The prior runner restored the head and raised, but a trusted checkpoint did
not retain that attempt. The format-2 history validator assumed one resolved
decision per epoch and could not accept a failed attempt followed by a retry.

## Decision

Treat each sleep call as a transaction over the head and the Python, NumPy,
Torch CPU, and selected Torch CUDA process random streams. On pre-guard,
core, post-guard, or rollback-delta failure, restore those states, append a
typed `error` event, save the restored `before_sleep` checkpoint when a store
is present, and re-raise. A new explicit resume retries sleep for that epoch
without repeating its wake batches. The ordered history may contain multiple
errors for an epoch before its resolved decision.

Count examples after each completed guard batch. A nonfinite completed pass
counts its entire exposure; an exception partway through a pass counts only
completed batches. Error records keep only known scores and, when the core
returned before the failure, its full proposed facts. Applied changes remain
zero. Checkpoint preflight binds exposure to prefixes of the selected guard
batch sizes, validates stage, reason, role, and legacy counters, and rejects
impossible histories before restoring the model. The historical report guard
counter still counts completed two-pass decisions only; partial failed
exposure belongs to typed events. Format 2 remains compatible because the
payload shape is unchanged.

## Alternatives

- Record an error only in memory. A process restart would lose the failed
  attempt and could not explain a repeated sleep epoch.
- Count each exception as a full guard pass. That would invent scored examples
  when a later batch raised before evaluation completed.
- Add partial failed exposure to the historical guard counter. That would
  change its completed-decision meaning and prior benchmark comparisons.

## Consequences

CPU file resume retains pre/core/post errors, partial exposure, completed
proposals, and same-epoch retry order. Wall-time, checkpoint-memory, and
fixed-width CPU paths preserve model and baseline metrics in bounded tests.
The original exception still stops the current invocation. Device-specific
CUDA telemetry and restart acceptance remain P3.10c3c.
