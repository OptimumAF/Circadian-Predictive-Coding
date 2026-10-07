# ADR-0025: Bound NumPy and Torch parity with a common output gauge

## Context

The production NumPy PC/circadian networks train a binary sigmoid output;
the Torch heads train multiclass softmax outputs. Their default dtypes,
initialization scales, diagnostics, replay, and structural behavior also
differ. Matching hidden width or a final accuracy does not establish
cross-backend numerical parity. P2.7 needs a common fixture that can test
actual production paths without silently replacing either training rule.

## Decision

Use a deterministic one-hidden CPU fixture in float64. Copy identical
input weights and hidden biases into the NumPy network and Torch head. For
a NumPy output margin `z=hV+c`, set the Torch two-class logits to
`[-z/2,z/2]` by assigning output columns `[-V/2,V/2]` and biases
`[-c/2,c/2]`; map a binary target to its class index. Then
`softmax([-z/2,z/2])[1]=sigmoid(z)` and two-class cross-entropy equals
binary cross-entropy in the unsaturated fixture. Both paths have the same
tanh hidden prior and latent drive `e+(q-y)Vᵀ` before parameter updates.

At the **same scalar learning rate**, the output parameter steps do not
match. If `g=hᵀ(q-y)/B`, NumPy updates its one output column by `-ηg`.
Torch updates the two columns by `[+ηg,-ηg]`, so its output margin moves by
`-2ηg`. The same factor applies to the output bias. The actual one-step
fixture verifies equal initial probabilities and supervised loss, hidden
traffic and hidden-parameter updates, and the factor-two margin difference.
It runs both ordinary and neutral circadian heads; circadian chemistry,
plasticity, and importance agree under matched settings. It does not call
the production trainers or later trajectories fully numerically equivalent.
The NumPy backprop MLP and Torch matched MLP head use the same gauge in a
separate zero-momentum SGD fixture. Their initial probability/loss and
first hidden-weight step agree; the Torch output margin again moves twice
as far at the equal scalar rate.

For structural decisions, use separate freshly initialized split-only and
immediate-prune-only fixtures with identical float64 states, non-tied
chemical/importance/weight scores, static thresholds, budgets, and zero
split noise. The two-logit output row norm is uniformly `1/√2` times the
binary row norm, so min-max-normalized ranking scores agree. Compare
selected indices, width, mapped parameters, chemistry, and feedforward
probabilities after each event; a split must preserve those probabilities.

## Alternatives considered

- Choosing half the Torch learning rate aligns the output margin step but
  halves the hidden-parameter step. It does not fix full-update parity.
- Comparing the default NumPy binary and Torch multiclass tasks directly
  would mix output objectives and encodings.
- Reimplementing one trainer as a test-only mirror would check equations
  but not the executed production paths.

## Consequences and open work

These are local CPU boundaries, not a benchmark ranking. The production
Torch heads normally initialize float32; this fixture deliberately replaces
their parameters with float64 tensors. NumPy selects both split and prune
indices before mutation, whereas Torch selects pruning after applying a
split, so a joint split/prune event has no parity claim here. NumPy and
Torch use different RNGs and Torch draws float32 split noise; stochastic
split trajectories are not compared. NumPy can consolidate a labeled replay
buffer while this Torch head cannot, so replay is disabled. The P1.7/P1.8
GPU, data, and fairness gates remain open, as does P2.6a's deeper matched
formulation.
