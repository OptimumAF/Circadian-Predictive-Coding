# ADR-0426: Select cue-relative inputs in the existing constructive TRAIN route

## Context

The pure cue-relative codec and separate native family passed their scoped gates.
The existing constructive BP fit already supplies TRAIN-only admission, delayed
updates, detached fixed checkpoint scoring and permanent accounting. Its bounds,
factory and checkpoint guards previously assumed the original invariant inputs.
The original128 fit reaches45/54; the later512 fit reaches52/54 with both A
cue-turn errors. Neither result establishes the cause or a circadian advantage.

## Decision

Add an explicit exact-bool `cue_relative=False` option to the existing
constructive route and its bounds/catalogue helpers, exposed as
`--constructive-bp --cue-relative`. Preserve the original default report and
native BP recipe. The selected protocol, bounds definition and encoding identity
name the representation. Validate full original current and next views before
constructing models. Keep catalogue and stream arrays raw296 so the bound
learner applies the non-idempotent mapping once for training and prediction.

Use the separate exact snapshot family, fixed native policy and paid clocks for
checkpoint admission. Retain the original runtime, budget, source/actor guards,
partial work records, fixed readout points and no heldout argument. Refuse the
old cue-turn gradient option with this recipe because its reader explicitly
binds the original projection. Do not silently reuse that reader.

## Alternatives

A separate runner would duplicate accepted isolation and accounting behavior.
Pre-projecting the catalogue while using the bound learner would apply the
red-cue permutation twice. Updating the gradient reader is independent work
that does not block fitting or probability readouts. Replacing defaults would
obscure the existing negative control. The BP-specific constructive route does
not imply full public-pilot registration or qualification of all four methods.

## Consequences

The same original six TRAIN episodes can exercise the new representation using
the ordinary CLI and stream. Bounds use its projected input keys and report
full admission/projection costs; role admission remains outside the pure codec.
No native objective, optimizer, dependencies, environment settings or benchmark
learner are changed. A separately reviewed fixed128 run can test the TRAIN fit
hypothesis against the preserved original control; its result cannot establish
retention, recovery, generalization or CPC benefit. Full-suite/release and new
family diagnostic readers remain separate requirements.

See [usage](../constructive-train-fit.md) and
[scope/evidence](../../artifacts/runs/cue-relative-fit-20261018/SCOPE.md).
