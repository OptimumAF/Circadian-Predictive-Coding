# ADR-0264: Match raw-coordinate BP initialization and update factors

## Context

The local nonlinear PC/CPC mapping specifies raw/effective weight and bias
coordinates plus raw SGD factors. Existing BP uses the same tanh/logit-BCE
representation but a uniform effective learning rate. It cannot alone isolate
scaling or PC inference effects in a new comparison.

## Decision

Reuse exact CoupledScalingConfig/State and shared raw initialization/effective
map/update helpers. Compute the existing full feedforward BP derivatives, then
apply raw chain factors. Keep raw diagnostics separate from stable direct-gradient
updates and validate complete candidate feedforward loss/gradients before return.
Accept PC-only states; retain beta as matching metadata, with no BP precision term.

Provide owned fixed-rate learner ports and a small matching/continuation CLI.
Snapshots describe raw coordinates; the explicit adapter owns update algorithm
and fixed settings. Existing native BP and all prior results/defaults stay intact.

## Alternatives

Matching only initial weights leaves per-layer optimizer factors unmatched.
Replacing the BP objective with held-latent PC derivatives would remove the
intended algorithmic control. Duplicating scaling rules would risk drift. Adding
CPC gates to BP would introduce another mechanism before the baseline is fixed.

## Consequences

BP/PC can share initialization, representation, factors and exposure while keeping
their native objective/procedure/work differences visible. Independent full-loss
derivatives, strict state/ownership, extreme products and real runtime continuation
must pass before measurements. This engineering prerequisite does not establish
comparative learning quality, width stability or equal compute.
