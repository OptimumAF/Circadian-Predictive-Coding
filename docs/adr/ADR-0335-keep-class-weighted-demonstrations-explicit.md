# ADR-0335: Keep class-weighted demonstrations explicit

## Sequential train-only array composition, 2026-10-16

The explicit weighted train-live recipe binds its64 array releases to actual
owned complete TRAIN episodes through TorchTrainWeightedLearner. Native update
count indexes the next original episode; exact arrays admit its existing typed
weighted method. Weight derivation uses only that current episode. Why: reuse
the existing delayed stream without inventing transitions or silently changing
ordinary mean-CE behavior. Native optimizer/state/snapshot formats stay unchanged.

Schedule/objective guards and detached forks are local adapter metadata; original
stream receipts and caller guards retain release/checkpoint authority. Native
snapshots alone do not attest the historical objective. Fixed raw unweighted
scoring and legal train-only readout stay separate from updates. Reuse the
accepted mean64 comparison only under unchanged inputs/initialization/evaluation
assumptions; retain the earlier weighted8 negative and undefined prior refusals.
No new mathematics, final-data access, runner or state-record chain is introduced.

## Context

The existing mean-CE recurrent learner fits a majority action on reserved traces.
Training labels are imbalanced. This motivates an experiment without establishing
a cause or improvement. Baseline behavior and final-data isolation must survive.

## Decision

Add a separate weighted episode method and typed update. Derive inverse-frequency
weights from the released train episode only, preserve complete temporal order,
and normalize native CE by observed target-weight mass. Revalidate the canonical
derived tuple before dispatch. Retain ordinary mean-CE and reward objectives.

Reuse the existing native owner, BPTT, optimizer, atomic capture and runtime ports.
Weights are explicit update inputs; snapshot/policy versions and fields stay the
same. There is no new schedule, runner, evaluation release or replay format.

## Alternatives

Silently weighting baseline arrays changes the established objective. Resampling
individual transitions loses temporal packet meaning. Persistent objective modes
add policy/snapshot state unnecessarily when each update declares its objective.

## Consequences

The method requires neutral-weight and independent NumPy BPTT/SGD checks, atomic
failure/role tests and complete owned continuation. All rows and failed native work
retain their original costs. A separate frozen comparison is required for any
learning claim; negative results and original mean-CE/RL baselines remain valid.
