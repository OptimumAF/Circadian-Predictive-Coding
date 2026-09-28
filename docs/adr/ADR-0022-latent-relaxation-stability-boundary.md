# ADR-0022: Bound local relaxation claims and detect nonfinite states

## Context

P2.4 requires evidence for stable latent relaxation without treating the
reported training diagnostics as a common optimized energy. One-hidden
NumPy PC/circadian and Torch PC/circadian have the local objective derived
in `docs/learning-mathematics.md`. Ordinary multilayer NumPy PC has a
different fixed-prior top-down rule whose scalar objective remains open
under P2.6. The prior implementation accepted nonfinite controls and data;
an overflowing linear prior could even become a finite `tanh` output.

## Decision

For a fixed one-hidden prior and output weights, test the local objective
`J = mean cross-entropy + sum((h-p)^2)/(2B)`. The executed latent step is
gradient descent with effective step `alpha * B` on `J`. Its per-example
Hessian is bounded by `I + V H_out V^T`, where the binary sigmoid or
multiclass softmax cross-entropy Hessian `H_out` has spectral norm at most
one. Thus `alpha <= 1/(1 + ||V||_2^2)` is a conservative sufficient
small-step bound for objective descent with fixed weights and prior. The
float64 fixtures use `alpha=0.2`, which satisfies that bound; they check
every objective step, an explicit 60-step residual-norm reduction, and the
actual NumPy/Torch final training state against test-only autograd. A
separate multilayer NumPy fixture checks the documented simultaneous
fixed-prior recurrence and its local residual reduction, without an energy
monotonicity claim.

Training now rejects nonpositive or nonfinite learning rates, nonpositive
or noninteger inference-step counts, and nonfinite NumPy inputs/targets or
Torch features. It raises `FloatingPointError` for nonfinite linear priors,
relaxed states, or final logits before the weight update. NumPy checks each
latent step. Torch checks its initial feature batch and makes one combined
finite check after relaxation, avoiding a host synchronization per step.
No clipping, adaptive step size, early stopping, or maximum step cap was
introduced; valid finite runs retain their original numeric update.

## Alternatives considered

- Asserting that the historical `energy` decreases would confuse a
  width-normalized diagnostic with the local update objective.
- Clipping a divergent latent or reducing its step size silently would
  change the algorithm and make failures harder to diagnose.
- Synchronizing Torch after every latent step would add device stalls to
  the benchmark timing scope.

## Consequences and open work

Invalid or divergent training calls now fail before the parameter-gradient
assignment. A circadian call may already have advanced cooldown or prune
state before an intermediate failure; this change is a detection boundary,
not an atomic training transaction. The Torch finite checks add a bounded
host/device synchronization cost, so earlier timing artifacts remain
historical and P1.8 requires fresh target-hardware confirmation. P2.6
still owns the multilayer objective decision. P2.8 retains broader shape,
label-range, parameter, and post-update numerical validation. No CIFAR or
CUDA claim follows from the local CPU fixtures.
