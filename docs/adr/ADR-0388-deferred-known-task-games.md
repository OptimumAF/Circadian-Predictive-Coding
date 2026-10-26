# ADR-0388: Optional post-training known-task game readouts

## Context

The fixed invariant exact-memory A128/F16/A16 assay already measures isolated
learning, old-task retention and return fit. Inline endpoint games repeatedly
validate all training cohorts at every native read; the preserved once-only
native case expired at its original20-second endpoint after4steps, with no game.
Per-read optimizations passed scoped checks but did not establish native games.

## Decision

Add an explicit optional `deferred_probe_rollouts` recipe to the existing
MiniGrid learnability adapter/CLI, mutually exclusive with inline games and
restricted to the complete exact-memory recipe. The existing app endpoint
observer captures four genuine paid typed states before subsequent updates.
Private immutable in-memory payloads retain at most four states (256KiB each,
1MiB serialized total and620544 logical table bytes); no scalar reconstruction
or retained original learner. Original completed probe/return budgets remain
bound and checked without observing new clocks or renewing allowances.

At phase entry keep original benchmark budget references, binding and deep
limits before the full proof. That trusted proof observes its original clock/RSS;
construct the readonly witness afterward against the pre-entry authority, and
check it before the closing full proof. This preserves observed progress without
accepting changed controls or in-game unobserved updates.

After all fixed learning, full original cohort/source/report/receipt proofs
bracket the readout phase, including failures. Two existing memory readers
restore the four captured states, validate complete state equality and use the
existing zero-update/replay game evaluator with its original20-second endpoint
allowance. Original180-second global time/RSS caps include training, captures,
restores and games. Return paid reports before closing proof refusal; missing
admitted reports leave unknown costs and validated lower bounds.

Why this: games cannot influence later training or selection when all training
has already finished. The new explicit phase contract moves original training
cohort mutation detection to entry/exit; it does not claim inline refusal timing.
Strict inline/default behavior stays available. Final evaluation, targets,
teachers, action masks and repair remain unavailable to games.

## Alternatives

Continuing small comparator changes leaves the demonstrated repeated cohort
traversal cost. Silently skipping inline checks would violate its contract.
An additional orchestration framework or checkpoint export is unnecessary:
the existing stream, isolated probe runner, endpoint observer and evaluator
already supply the required ports.

## Consequences

Capture/copy/decode/storage/factory/restore costs are reported separately from
learning and live games. Tests and source review establish implementation scope;
a separately frozen native case is required for game execution evidence.
Known finite A/F games cannot establish heldout transfer, reinforcement learning,
or a circadian advantage. Preserve all prior failures and unfinished roadmap.
