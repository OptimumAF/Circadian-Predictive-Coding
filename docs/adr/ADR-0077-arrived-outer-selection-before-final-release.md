# ADR-0077: Select arrived outer candidates before global final release

## Context

The v6 continual runner separates phase-local train, inner guard, outer
selection, and final-test roles, but its public run scores final tests under
one fixed setting. Calling it once per candidate would expose final labels
before the other settings train and let test results influence selection.
Format 6 binds only one setting, so it cannot itself freeze a candidate set.

## Decision

Add a separate ordinary `continual_arrived_outer_selection_v7` app route.
The caller predeclares two to four candidate IDs/configurations and an
ordered seed list with at most eight candidate-seed trials per method.
Candidates must share data generation, roles, phase epochs, model order,
inference work, sleep/replay budgets, and guard policy; only the three
method learning rates may differ, and each method must receive distinct
rates. Every candidate trains all three methods on every seed before any
outer scoring. This uses the existing v6 phase-arrival and inner-guard
training path without changing v1–v6 report/checkpoint identity.

For each method/candidate/seed, score the saved Phase A and final models
on only the arrived A/B outer roles. The predeclared objective is the mean
across seeds of `0.5 * (A accuracy after B + B accuracy after B)`.
The first declared candidate wins an exact tie. Keep every trial's full
development identity, role-use/guard/task ledger, training and guard/outer
example counts, sleep and replay records, and outer scores. A freeze record
digests the ordered candidate manifest, all trial rows, and the three
independent per-method choices. Only after that freeze does the runner
release final input/labels and score the selected model for each method.

The selection route holds all trained candidate states in memory for this
small correctness gate. It does not yet persist or resume a candidate
manifest; that is P1.3c3b2c3b. V6 format 6 remains a single-setting file.

Follow-up: ADR-0078 adds a distinct format-7 candidate-manifest checkpoint
without changing this ordinary selection rule or format 6.

## Alternatives

- Call the public v6 benchmark for each candidate. That scores final tests
  before the candidate set is complete and breaks the global seal.
- Choose one shared candidate from the three methods' combined scores.
  That would give each method a setting affected by other methods' outer
  performance, obscuring equal per-method selection effort.
- Search candidate counts or seeds after seeing final results. That would
  invalidate the predeclared comparison and is excluded.

## Consequences

The ordinary two-candidate/two-seed canary proves all four training trials
finish and all three choices freeze before any final source field opens in
both model orders. Final-label perturbation changes the released final
hash while leaving every trained state, outer trial, choice, and freeze
unchanged. Changing only outer labels changes outer hashes and scores
without changing trained state. Mixed per-method choices use exactly their
selected model states; ties use the first candidate. The fixed one-epoch
smoke reports every low-scoring and tied trial rather than adjusting the
metric or seeds. This is a tiny synthetic correctness check, not a model
ranking or the full checkpointed strict-online confirmation.
