# P4.6a supervised-error signal audit

The existing `use_reward_modulated_learning` switch scales supervised wake
updates from the mean absolute output error of the labeled batch relative
to an EMA of earlier batch errors. It is a difficulty proxy, not an
environmental reward or a reinforcement-learning return. The ratio is
clipped to `reward_scale_min`/`reward_scale_max` (defaults 0.75/1.5), and
the EMA is updated after the ratio is computed. With modulation disabled,
the scale is one and the reward-error EMA is not updated. NumPy and the
Torch head use the same signal rule; the NumPy wake-only replay policy
adds a read-only baseline path for replay, not a new reward definition.

## Fixed causal probe

`tests/test_difficulty_signal_audit.py` uses four synthetic train rows, a
fixed logistic reference with weight 2 on the first feature, and no
validation or final-test role. Clean logits give probabilities 0.2/0.8
for the correct binary labels. One condition flips only the first label.
Another changes only the first feature from `-log(4)/2` to `3`, making
its otherwise unchanged label a confident error (probability about
0.9975). Both backends receive the same precomputed float64 output-error
arrays at their existing scale methods. No parameter update or benchmark
selection occurs in this probe.

For diagnostics only, per-row absolute error is clipped at 0.5 before
averaging and divided by the clean clipped-error baseline. The 0.5 limit
is the binary uncertainty boundary and was fixed before reading the
condition results. A separate loss-improvement measure applies the same
fixed 20% movement from each probability toward its observed label, then
computes pre/post mean binary
cross-entropy. This constructed correction is not the NumPy or Torch
optimizer, and its post-update loss is unavailable when choosing the
current update scale. Its 20% amount is a probe constant, not a tuned
learning rate or a reported training budget.

| Train condition | Mean absolute error | Historical scale | Error clipped at 0.5 | Clipped-error ratio | Constructed BCE improvement | Unmodulated scale |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 0.200000 | 1.0 | 0.200000 | 1.000 | 0.048790 | 1.0 |
| First label flipped | 0.350000 | 1.5 | 0.275000 | 1.375 | 0.183539 | 1.0 |
| First feature outlier | 0.399382 | 1.5 | 0.275000 | 1.375 | 1.137313 | 1.0 |

The historical factor saturates on either corrupted batch and does not
distinguish its cause. Clipping reduces their diagnostic ratios, but the
small probe does not establish a better learning outcome. Constructed
loss improvement is largest for the feature outlier because the
label-directed correction fixes its very confident error; that value
does not identify whether the row is useful for generalization. Using
same-step improvement as a training factor would additionally require
lookahead. The unmodulated control stays at one in both backends. These
are signal-level findings, not accuracy or forgetting results, and no
candidate is selected.

## Next gate

P4.6b must compare actual learning under historical modulation and an
unmodulated control on predeclared clean, label-noisy, and feature-outlier
train streams with matched model/work/role budgets. The clipped-error
and loss-improvement diagnostics can be logged on the same train-only
boundaries; a prior-step definition is required before any improvement
signal could drive a future update. Keep existing reward-named config and
checkpoint identities, seal final labels until all training choices are
frozen, and retain negative or null outcomes. No new heuristic is
authorized by this probe.
