# ADR-0362: Read original recurrent TRAIN gradients without an update

## Context

Accepted MiniGrid diagnostics report failed joint cue acquisition. Existing
isolated probes and activation health measure fit and activity, but do not
compare the gradients induced by the two cue episodes. The matched momentum
control is less harmful than the reward continuation on the recorded CE scores;
it still fails acquisition. These negative results do not justify tuning a new
learning rule or rerunning matched baselines.

## Decision

Add a pure bounded signed geometry reducer and a small native adapter using the
existing recurrent mean-CE features and zero-reset forward. Read both gradients
on detached parameter clones with `autograd.grad`; preserve original optimizer,
gradients, counters, inputs, RNG and caller grad mode. Validate before copying
nested snapshot state. Refuse unknown changes after the last native callback.
Return only scalar geometry and paid attempted/completed work, retaining failed
prefixes with no geometry. Keep data provenance and stream composition with the
caller and existing runner.

Why this: it exposes an absent learnability diagnostic without changing the
deployed learner, the objective, or the evaluation boundary. Float64 accumulation
and explicit undefined zero-norm cosines preserve numerical interpretation.

## Alternatives

- New weighted loss or adaptive schedule: no supporting causal evidence yet.
- Read gradients left by a previous update: they belong to a different state
  and may already include accumulation or different episode exposure.
- Backward on the source learner: would risk its stored gradient state.
- Another orchestration framework: existing snapshot and stream ports suffice.

## Consequences

The component pays forward/backward work and temporary gradient/copy storage
even with zero optimizer steps. Native validation constructs one temporary SGD
configuration checker. Whole-episode alignment can hide cue-specific conflict;
this is neither causal evidence nor proof of useful adaptation. A separately
qualified TRAIN checkpoint composition remains necessary. Existing negatives,
scientific/release gates, final isolation and matched baselines remain intact.

## Checkpoint composition

The existing MiniGrid checkpoint scorer optionally reads three original TRAIN
blue/red pairs only for recurrent BC, after preserving six paid raw scores.
A small helper keeps the original packet binding, scalar results and separate
work ledger; it does not own a new runner or data release authority. Returned
payloads and failed work are retained before any subsequent guard can refuse.
Private expected counters remain independent from public report counters.

The ordinary CLI option is explicit. Default outputs and all feature controls/
probes are preserved. New-mode forward totals include gradient and optional
activation reads; no-gradient legacy totals keep their original definition.
Distinct protocol IDs preserve the original/history/health configuration.
Qualifying a small native checkpoint does not validate the whole five-learner
run, useful acquisition, retention/recovery or circadian advantage.

## Accepted A128 measurement

A thin RNN-only use case composes existing prepare_warm_start output with that
same scorer. A saved JSON reference cannot supply native weights or momentum;
reconstructing128updates is paid anew. Keep the original five raw parity checks
and endpoint native laws, returned source/guard/packets and original warm
deadline. A separately bounded zero-update diagnostic phase does not renew warm
training. This obtains a scientifically relevant state without regenerating
all five learners or probes, and makes no change to the original objective.

## Isolated finite-update follow-up

Why this: near-opposed A128 episode gradients do not predict an actual finite
native update with carried momentum. The explicit follow-up forks that same
full state twice, uses one original blue/red mean-CE episode respectively, and
scores all six TRAIN packets on both outputs. Shared momentum stays intact;
neither arm is selected for deployment or benchmark continuation. Existing
raw/gradient ports and typed update dispatch suffice, so no new runner is needed.
Keep both arms, paid failure prefixes and separate observed budget/native charge
facts. The nested two-update budget does not renew original warm/readout clocks.
