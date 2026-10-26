# ADR-0462: Describe TRAIN-only continuous learner conditioning

## Context

Existing continuous BC residuals describe action errors. The fresh native guard
case rejected the larger nearest candidate; it does not justify further tuning
on those guards. A neural learnability study needs activation/actual-gradient
measurements without changing its learner or exposing held-out data.

## Decision

Extend the existing pure diagnostics module and continuous BC adapter with a
bounded TRAIN-only conditioning method. Reuse the actual objective and forward
kernels, with two forwards and one backward per successful call. Return only
immutable scalar/tuple summaries and explicit successful work counts. Use stable
norm reductions and refuse nonrepresentable results. Preserve all defaults,
residual reports, optimizer state, RNG and role boundaries.

## Alternatives

Changing the existing gradient kernel return schema would reduce duplicate
forward work but broaden a correctness-sensitive change. A new runner or general
inspection framework is unnecessary. Guard/probe/final data cannot substitute
for a TRAIN diagnostic or become training/probe-selection inputs.

## Consequences

The extra forward and copies are real costs and are reported separately from
numeric payload/process memory. A failed call may consume work without producing
a report; callers retain attempted/returned counts and failures. Saturation and
gradient magnitude describe the actual objective, including binary CE's logit
derivative; they do not prove useful learning. A separately frozen isolated
TRAIN learning protocol is the next research step, with all negative results
and unfinished parent acceptance criteria preserved.
