# ADR-0444: Separate continuous-action demonstrations

## Context

R4.4 plans a ManiSkill state-observation reach-task BC baseline. The existing
demonstration type encodes categorical action IDs, toy bounds and one terminal
flag. `NativeLearner` is already generic over input/target/prediction types.

## Decision

Add a pure core continuous record boundary with explicit per-axis action units
and inclusive bounds, finite native vectors, complete ordered/seamed episodes,
separate terminated/truncated flags and matching training-role extraction.
Re-copy records with validation; do not clip, infer/convert units, fabricate
missing steps or merge the end flags. Reject inexact integer-to-float conversion
before bound checks so rounding cannot admit an out-of-bounds action.

## Alternatives

Extending categorical IDs into vectors would change old replay/encoding contracts.
Implicit unit inference or clipping would hide a mismatched controller. A new
runner or release framework is unnecessary: reuse `NativeLearner` and existing
orchestration when the actual continuous BC adapter is ready.

## Consequences

The old categorical path remains intact. New simulator adapters must declare and
validate the actual controller units/bounds and convert native arrays explicitly.
Static limits are1024 observation coordinates,16actionaxes and256steps. Supporting
larger records needs an explicit bounded change. Both end flags may be true;
task success/reward are not inferred from either. Role validation alone grants no
runtime data release or training authority. This component does not implement a
simulator, learner, robot task or scientific result; those R4.4 requirements remain.
