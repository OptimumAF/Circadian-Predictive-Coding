# ADR-0072: Persist unscored continual seeds in format 5

## Context

V5 ordinary runs train every configured seed before opening a final test.
The earlier continual checkpoint schema stores scored results and test
digests at each `seed_complete` transaction. Reusing that transaction
would reveal seed 17's test labels before seed 19 trains. A resumable
run needs completed model states and development-role identity instead.

## Decision

Use checkpoint format 5 with `unscored_seeds` records. Each record stores
the seed, arrived development-role digest and hashes, detached baseline
and circadian states, and no final-test role, hash, score, or digest.
Active Phase A/B checkpoints retain the existing transaction cursor and
phase-specific development digest. At `seed_complete`, save the newly
trained unscored state before starting the next seed. After all seeds
finish, regenerate deterministic roles, validate the saved development
identities and model/replay provenance, then bind test hashes and score.
The terminal checkpoint remains unscored; resuming it repeats reporting
without retraining.

Format-5 preflight validates every earlier unscored seed before any new
wake update. It checks role identity, baseline step counts, phase-A and
final circadian wake progress, replay budget, and replay membership in
arrived training roles. V1–v4 checkpoints keep their existing report and
test-digest fields. The trusted local file store retains its checksum
and pickle trust boundary.

## Alternatives

- Store scored earlier seeds in the existing format. This breaks the
  all-seed final-label seal.
- Store held-out arrays in checkpoints for later scoring. This expands
  the training-side payload with final labels.
- Restart and retrain earlier seeds after interruption. That adds work
  and weakens exact continuation evidence.

## Consequences

Fresh and interrupted/resumed format-5 runs match the ordinary v5
report for the fixed two-seed fixture. Raising source sentinels cover
both model orders and A/B, seed-boundary, and terminal interruption
points. Saved payloads contain development hashes and unscored states,
with empty scored-result and test-digest fields. Altered prior-seed
development roles, model step counts, and future-seed replay examples
are rejected before another update. Format-5 storage grows with the
number of completed seeds. Its checksum catches accidental file damage;
it is not authentication for untrusted pickle files. Disjoint inner
guard/outer selection, setting selection, and an arrival ledger remain
open under P1.3c3b2c.
