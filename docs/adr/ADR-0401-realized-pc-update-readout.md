# ADR-0401: Read realized ungated PC directions without publishing updates

## Context

The frozen joint TRAIN study did not acquire exact traces. Existing prior-CE
gradient geometry does not measure the direction PC takes after finite latent
relaxation. ADR0021 and local-autograd tests already explain the distinction
between the local CE-plus-hidden objective and the legacy squared diagnostic.
Changing the diagnostic identifier or repeating that math would not answer the
missing realized-direction question.

## Decision

Add one TRAIN-only adapter accepting bounded complete encoding-bound PC/neutral
snapshots and original TRAIN episodes. Reuse existing validation/copy/encoding/
geometry helpers. Read finite relaxation, latent residual/stationarity and the
native candidate's exact float32 parameter delta against the prior-CE gradient.
Evaluate counterfactual prior CE on the same TRAIN rows. Return scalar reports
and paid work only; no learner/optimizer/update publication or new runner.

Preserve the original equations, methods, seeds, budgets and legacy diagnostic.
Validate the pre-tanh linear prior so overflow cannot be hidden by saturation;
validate the reported objective sum as well as its components. Failures retain
paid partial work. Preserve caller complete state, grad metadata, RNG and mode;
refuse observed changes rather than restoring external side effects.

## Alternatives

- Prior-CE gradients alone omit finite-inference direction and rounding.
- Stepping a deployed/benchmark learner contaminates the measurement.
- Treating active gates as neutral would report the wrong realized law.
- Replacing the legacy diagnostic changes historical curves and sleep behavior.

## Consequences

PC and matched neutral CPC have a tested read-only direction API; active gates
and BP are explicitly unsupported by this ungated contract. Correctness does
not establish useful acquisition or its failure's cause. Copies/backwards/
counterfactual forwards are charged, not free. Opt-in composition into the existing
joint checkpoint route is next; scientific use needs its own frozen reviewed
budget. Prior failures, negative results and missing-probability requirements
remain unchanged and full R2.1/release stays open.
