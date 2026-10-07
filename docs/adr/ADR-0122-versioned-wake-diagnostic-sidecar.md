# ADR-0122: Capture wake update returns in a separate measured sidecar

## Context

The fixed v14 train-only and scored JSON contain sleep, replay,
structure, guard, work, and final metrics, but the shared named-update
helper discards each core method's `train_epoch` diagnostic. P5.2
requires genuine per-epoch metrics without changing the frozen v14
science protocol or reading final labels during training.

## Decision

Return the existing typed core result from the shared helper. The v14
runner opts in to copying one metric after each successful train-only
update; other callers ignore the return as before. Preflight its full
six-cell order, method definitions, finiteness, matched work, and role
seal. Serialize diagnostics separately, then publish them only after
the same study passes the global final gate and its complete P5.1
bundle verifies. Give the sidecar and additive measured JSONL/CSV
projection new IDs and source-bound manifests. Keep all v14 raw bytes,
algorithm settings, baselines, seeds, and P5.2a streams unchanged.

## Alternatives

- Re-evaluate the train batch after update: that adds work and changes
  the metric's timing and potentially model state.
- Put metrics into frozen v14 JSON: this changes known hashes and
  conflates observation schema with the fixed experiment protocol.
- Derive wake values from guard/final scores: those use different
  roles and measurement times.

## Consequences

The sidecar records Backprop BCE and two differently defined PC
energies, each with explicit pre-parameter-update timing. A completed
run can be audited and repeated locally while historical v14 results
remain byte identical. The hashes detect accidental changes; they
are not signatures or independent proof of numeric values. P5.3
still owns atomic publication and recovery from partial writes.
