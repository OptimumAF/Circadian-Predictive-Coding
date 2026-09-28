# ADR-0019: Exercise toy replay and structural streams in reversed order

## Context

The guarded NumPy toy runner had deterministic dataset and split seeds but
trained its three models in one fixed order. The NumPy circadian model owns
a structural-noise generator seeded separately from initialization; replay
selection sorts stored wake snapshots and does not draw random numbers. The
Torch circadian head likewise owns a separate split-noise generator. Code
inspection alone did not prove that unrelated model or process RNG use leaves
actual replay, split decisions, and trained states unchanged.

## Decision

Add an optional `ExperimentConfig.model_order` permutation to the toy runner
and record it in `ExperimentResult`. Validate the permutation before dataset
loading. Keep the default order and existing toy protocol IDs unchanged;
order is an execution control, not a different data-access rule. A tiny
guarded CPU fixture reverses model training order while circadian training
performs prioritized replay and a noisy split. It compares actual replay
priorities selected at sleep, split indices, serialized trained-model/RNG
hashes, role hashes, losses, and validation/final metrics. It also perturbs
the global NumPy RNG between runs. A separate Torch CPU fixture reverses
two independently seeded heads amid global Torch draws, comparing split
indices and the existing trained-head state/RNG hash.

## Alternatives considered

- Inferring stream independence from `default_rng` and `torch.Generator`
  construction would miss accidental global draws in training or sleep.
- Reusing a single fixed model order cannot exercise the claimed order
  invariant.
- Changing the default order or assigning a new protocol ID would disrupt
  existing toy reproduction without changing the role or metric contract.

## Consequences and open work

The gate covers a small local CPU toy schedule and a direct Torch head split.
The NumPy replay buffer receives the same training role on each epoch; its
priorities change with the model, and the fixture confirms the selected
priority sequence is stable. The test is not evidence that replay improves
accuracy. Actual CIFAR loader execution, real GPU numerical behavior,
strict-online replay, and broader random-stream separation remain open in
P1.7. Full resumable snapshots including RNG and replay state remain a
separate P3.3 requirement.
