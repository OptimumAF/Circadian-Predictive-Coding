# ADR-0390: Use the existing FIFO learner for a recurring capacity control

## Context

The recurring raw control acquires all three known rooms and retains A. The
invariant refusal route is source accepted, but its sole native case exhausted
180 seconds and discarded its report before publication. That failed allowance
and unknown learner outcome remain unchanged. A future-only publication repair
has separate model-free verification; it does not recover missing evidence.

Existing FIFO/core/wrapper ports already implement bounded insertion order,
atomic count updates, retirement aggregates and complete typed snapshots. The
recurring runner lacks their explicit composition. Its shared counter also pins
both old count families to 64 entries, so new admission must be narrow.

## Decision

Add mutually exclusive `invariant_fifo_memory` / `--invariant-fifo-memory` to the
existing runner. Use the existing bound FIFO learner, fixed 36-entry/87,264-byte
table and separate 16-byte retirement metadata. Two original 18-key tasks define
the capacity prospectively; no seed or successful outcome selects it.

Add a distinct typed TRAIN evaluator entry point. An explicit private training
option selects FIFO36; default counter admission remains 64 entries. Preserve
raw/invariant defaults, exposures, seeds, metrics, original source/budget/receipt
guards and final exclusion. Capture five complete typed observations, not a new
snapshot chain. Keep native commits and returned paid receipts distinct on loss.

Why this: finite-capacity interference and task return are learner questions that
the existing stream and backend can answer without another runner or mechanism.
Canonical support of the original TRAIN traces is 54 disjoint keys, so this is
actual capacity pressure, not a context label that disappears under projection.

## Alternatives

- Broaden old counters to any capacity: rejected; old callers retain their fixed
  policy and cannot silently admit a different saved learner.
- Reimplement eviction or orchestration: rejected; working components exist.
- Retry the failed refusal case: rejected; its allowance remains spent.

## Consequences

Focused tests use real counts/stream and artificial physics. Native evidence
requires a separate frozen, reviewed, single bounded case. Deadlines, negative
outcomes and censored work remain binding; failed reports must survive publication.
Known TRAIN success or a memory tradeoff would not establish held-out skill,
reward learning, equal-compute superiority, neural recovery or circadian benefit.
