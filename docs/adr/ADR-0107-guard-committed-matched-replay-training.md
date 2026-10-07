# ADR-0107: Apply shared replay only after guarded sleep commits

## Context

ADR-0106 fixes one prediction-independent A→B replay schedule, but its
PC/backprop rows and work counts were only planned. In the existing NumPy
runner, a circadian sleep can be rejected by the inner guard or fail in the
core. Updating either baseline first would leave an unequal applied replay
budget even if circadian restores its pre-sleep state.

## Decision

Add a separate `continual_matched_replay_training_v9` train-only runner above
the schedule session. It constructs all three models with the same arrived
train rows and retention policy. Before each periodic sleep, it checks the
circadian retained IDs, copied-array byte count, retained order, selected
IDs, and detached row contents against the bound schedule. The schedule
checks its complete manifest on every advance. The runner then calls the
existing guarded sleep path on the arrived inner-guard role.

Only an accepted event with exactly the selected number of circadian
replay examples and optimizer updates permits baseline replay. PC and
backprop each receive separate copies of the selected rows, the same
declared replay learning rate, and the same number of optimizer calls. PC
uses its declared replay inference count; the circadian core uses its own
declared count. The trace reports successful calls times those fixed inner
loop counts, without treating them as equal computational work. A rejected
or skipped sleep applies zero baseline replay. A core failure raises after
the existing circadian rollback, before either baseline call. The runner
also checks that sleep never advances wake clocks or refills retained
memory. It returns unscored models and a role/sleep audit; it does not
release final tests.

## Alternatives

- Replay to PC/backprop before circadian sleep and roll them back if its
  guard rejects. This needs two additional snapshots and a cross-model
  transaction for no improvement to the fixed protocol.
- Replay to baselines regardless of guard outcome. That would hide an
  unequal *applied* replay budget after a rejected circadian sleep.
- Let each model select its own rows. The selection would no longer be a
  matched exposure control.

## Consequences

Both fixed seeds, FIFO and seeded bottom-k, and both model orders run the
same copied rows. Tests observe actual per-row calls and inference-step
arguments, detached arrays, A→B arrival, preflight failures before sleep,
guard rejection, core failure, and a two-epoch interval. The local four-row
trace is byte reproducible and has no model score or winner. It establishes
replay application parity only. A later matched outcome artifact must
release final tests once for every fixed trial and audit metrics, state,
and irreducible inner-work differences before P4.4 closes.
