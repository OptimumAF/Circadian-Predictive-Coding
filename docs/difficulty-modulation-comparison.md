# P4.6b fixed matched difficulty comparison (v11)

This protocol was fixed before reading any v11 training or final outcomes.
It compares the existing supervised-error modulation switch with the
unmodulated control. It does not select a new scaling heuristic.

## Data and roles

- Seeds: 17 and 19. Backends: NumPy shallow circadian network and CPU Torch
  shallow circadian head, analyzed separately because their output heads
  and numeric precision differ. Arms: historical modulation on/off.
- Each seed has independent A and B phases with forty balanced development
  rows and forty balanced final rows per phase. A centers are (-1.2, 0)
  and (1.2, 0); B centers are (0, -1.2) and (0, 1.2), with independent
  Gaussian feature noise of standard deviation 0.35. Development and
  final RNG streams are independent. The final source is opened only by
  the global release function.
- The existing class-stratified splitter reserves 20% inner guard and 20%
  outer selection from development rows. Neither role changes a setting
  in this fixed comparison. On B only, the first class-zero training row
  is left clean, has its label flipped to one, or has its second feature
  moved to 4.0. A and both held-out final roles stay clean and identical
  across conditions. The effective train-row digest records the one-row
  edit separately from the clean partition identity.

## Matched training and diagnostics

Each arm starts from the same backend-specific seed and 8-wide shallow
head. It receives two full-batch updates on A then two on B, with learning
rate 0.02, two latent-inference iterations per update, and inference rate
0.1. There is no sleep, replay, structural change, minibatch sampling,
early stopping, or setting selection. Chemical dynamics are the existing
defaults in both arms; only `use_reward_modulated_learning` differs.
Within each backend, each condition/seed/arm has equal row exposure,
optimizer updates, and inference work.

At each update, record the actual core scale, pre-update feedforward
train-only mean absolute error clipped per row at 0.5, its ratio to the
previous clipped-error EMA (0.95 decay, first ratio 1), and pre/post
train-only BCE. `prior_step_loss_improvement` is the *previous* update's
pre/post BCE difference, initially absent. The diagnostics never feed
the current optimizer. They are not the core's relaxed-state error and
do not claim to replace it.

## Global release and outcome rules

All 24 trials (3 conditions × 2 seeds × 2 arms × 2 backends) train
before final access. Preflight the complete Cartesian trial set,
development-role and effective-train hashes, four-update clock, examples,
inference counts, absence of sleep/replay, finite diagnostic trace, and
unit actual scale in controls. Then release each seed's A/B final roles
once. Score A-after-A, A-after-B, and B-after-B accuracy from saved
states. Forgetting is A-after-A minus A-after-B, including a negative
value when retention improves. Preserve every seed and condition; report
modulated minus control deltas without selecting a winner or seed.

Outputs are deterministic local JSON with manifest and role digests. This
small synthetic comparison can reject or support a local hypothesis but
does not establish behavior on real feature distributions or repeated
corruption. A new heuristic needs its own protocol gate.

## Observed fixed run (after protocol freeze)

The local JSON at `data/difficulty-modulation-v11-result.json` contains
all 24 trial rows, every update diagnostic, role/effective-train hashes,
and work. A second independent invocation produced the same SHA-256
`caf939687d54ba4b480e60b7d9593005093a479968989c42789de230b978b9b5`.
The manifest digest is
`8a274c3b7dd6c747284489f83e918c55cca4ebcfc5f7fa80b8abca2cbe5520d1`.
Within each backend, seed, and train condition, the control and modulated
arms have identical final accuracies and forgetting at 40-row resolution.
The table prints that common value, preserving all twelve matched pairs.

| Backend | B train | Seed | A after A | A after B | B after B | Forgetting |
|---|---|---:|---:|---:|---:|---:|
| NumPy | clean | 17 | .975 | 1.000 | .000 | -.025 |
| NumPy | clean | 19 | .000 | .000 | .000 | .000 |
| NumPy | label flip | 17 | .975 | 1.000 | .000 | -.025 |
| NumPy | label flip | 19 | .000 | .000 | .000 | .000 |
| NumPy | feature outlier | 17 | .975 | 1.000 | .000 | -.025 |
| NumPy | feature outlier | 19 | .000 | .000 | .000 | .000 |
| Torch CPU | clean | 17 | 1.000 | 1.000 | .875 | .000 |
| Torch CPU | clean | 19 | .775 | .775 | .000 | .000 |
| Torch CPU | label flip | 17 | 1.000 | 1.000 | .700 | .000 |
| Torch CPU | label flip | 19 | .775 | .650 | .000 | .125 |
| Torch CPU | feature outlier | 17 | 1.000 | 1.000 | .875 | .000 |
| Torch CPU | feature outlier | 19 | .775 | .775 | .000 | .000 |

Each trial completed four optimizer updates, 96 example presentations,
eight full-batch inference loops, and 192 example-inference iterations,
with zero replay and sleep events. Controls recorded actual scale 1.0.
Modulated NumPy B-phase scales ranged from 1.0358 to 1.3204 across the
fixed rows; Torch CPU ranged from 1.0029 to 1.0068. The modulation
switch therefore changed update magnitudes, but produced no detectable
held-out accuracy/forgetting difference on this budget. In Torch seed
17, one flipped training label reduced B accuracy from .875 to .700 in
both arms; the outlier did not change B accuracy. NumPy did not learn B
under this two-update schedule, and seed 19 failed A as well. These are
limitations of the predeclared budget, not grounds for retuning this
artifact. The train-only clipped-error diagnostic was already near its
0.5 ceiling at B arrival, limiting its discrimination here. Prior-step
loss improvement remained a diagnostic and never drove an update.

Decision: no superiority claim and no new heuristic. Later work can
predeclare a separate, better powered protocol without overwriting v11.
