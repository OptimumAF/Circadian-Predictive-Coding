# ADR-0020: Verify continual phase states under reversed model order

## Context

The corrected continual runner trains backprop, predictive coding, and
circadian models in a fixed sequence in both phases. Phase A states are
deep-copied before Phase B data are built, and both final-test roles are
scored after all Phase B training. That separation prevents future labels
from entering Phase A, but the fixed sequence did not verify whether model
order affects replay, noisy structural changes, or the retained Phase A
state used for forgetting measurements.

## Decision

Add `ContinualShiftConfig.model_order` as a validated permutation, checked
before data loading and recorded in each seed result and formatted report.
Use one per-epoch training helper in both phases; keep circadian scheduled
sleep after all three wake updates in an epoch. Preserve the default order,
existing role allocation, legacy route, and protocol IDs.

A tiny corrected-protocol CPU fixture reverses the order for seed 13 with
80 examples per source phase, three epochs in each phase, prioritized
replay, and nonzero split noise. It captures the replayed snapshots'
priorities and input hashes, structural decisions, the deep-copied Phase A
model hashes, and final Phase B model hashes including the NumPy generator
state. It compares all six phase-role hashes and the retention/adaptation
metrics after unrelated global NumPy draws. A sentinel separately checks
that both execution orders keep final-test roles sealed until Phase B ends.

## Alternatives considered

- Seeding only once before the three trainers would not prove order
  independence if a training path later used a global random stream.
- Recreating models at the Phase B boundary would erase the state whose
  retention is being measured.
- Scoring Phase A test while deciding Phase B sleep would contaminate the
  retention comparison; the final-test boundary remains after training.

## Consequences and open work

The fixture is a local CPU reproducibility gate, not evidence that replay or
structural growth improves retention. The corrected runner still plans its
full A+B sleep horizon before Phase A and therefore is not strict-online.
Actual CIFAR loader execution and GPU numeric behavior remain unverified.
The legacy route remains available with its original default order and
historical data allocation. Full P1.7 remains open for environment and
stream coverage beyond this local fixture.
