# ADR-0357: Reuse the accepted MiniGrid control for a distinct weighted objective

## Context

The original fixed-A128 recurrent control is accepted but fails both-cue
acquisition:17/18raw labels, one blue error, red goal/blue negative terminal.
Existing weighted CE/BPTT and its array-port wrapper support148/K7, but no
MiniGrid composition exists. Sparse reward collection/run works while its eight
rollouts supplied no reward. Useful acquisition remains a prerequisite.

## Decision

Load the fixed accepted A128 artifact once through existing bounded JSON and
saved-report validation, reconstruct its TRAIN source DTOs, and retain its exact
unweighted control/receipts/readouts as historical evidence. Train only the new
weighted recurrent arm on the identical128 A packets with unchanged seed/dose,
checkpoint set, raw unweighted metrics and native all7first-argmax readouts.
Keep its control name and native objective distinct. Derive weights only from
the current released TRAIN episode using existing native math.

Bind original source DTOs to recorded physical transitions and retained receipt
identities/clocks. New initial raw/native law must match the retained control
within fixed1e-7absolute/0relative probability tolerance. Incomplete wholecase
comparisons remain unknown. Historical TRAIN readouts already exist; keep them
out of the training factory, and finish all new training before new readouts.
No final/probe data, tuning, promotion or replay is introduced.

Read-only measurement found full-schedule metadata snapshot cost.141s. Replace
recursive schedule equality with whole pickle-byte equality for both exact owned
records and their seal, retaining record-field/owner/policy/native state checks.
The measured corrected snapshot is.018863s; no loss or budget semantics change.

## Alternatives

Rerunning the accepted baseline and teacher traces duplicates completed work.
Relabeling weighted loss as the old baseline conceals a changed objective.
Extending sparse reward dose or choosing a rewarding seed does not qualify this
explicit control. A new trainer duplicates the existing guarded stream.

## Consequences

Historical and new exposures/costs are separate; neither equal computation nor
shared opaque state is claimed. The fixed exploratory control may still fail.
It establishes no CPC benefit, retention/recovery or final generalization.
Native comparison requires final correctness gates and a frozen reviewed budget.
The original fresh-pair draft and all failed checks/measurements remain preserved.

## Observed consequence

The sole new weighted case completed within its original budget. Both controls
finish17/18, with opposite failed cues and the same one-goal/negative-terminal
live aggregate. Weighting did not satisfy joint acquisition or improve live
return. This does not justify selecting an objective, dose or seed after the
fact. Exact commands, costs and negatives remain in the stage RESULTS.md.
