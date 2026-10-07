# ADR-0015: Select matched heads from an equal-trial validation ledger

## Context

The fixed-feature three-head runner scores final test on each invocation.
Calling it once per hyperparameter candidate would expose test labels before
selection and make accidental cherry-picking easy. Existing image-level
tuning scripts use validation-only candidate reports, but they do not cover
the shared-representation heads or establish equal candidate budgets.

## Decision

Add a separate `vision_matched_head_equal_trial_tuning_v1` route. The caller
declares a full base config, exactly the same candidate count for each head,
and one fixed tuple of distinct seeds. Restrict candidate changes to the
respective head's optimization settings. Reject target-accuracy stopping,
unfrozen backbones, mismatched initial widths, duplicate settings, and more
than eight candidate-by-seed trials per head. Keep guard, data, backbone,
width, epoch cap, and selection metric fixed.

Within each seed, materialize train, guard, and outer-validation features
once, then run every candidate from equal initial tensors with a reset RNG.
Record every attempt's status, config, and seed. Record each successful trial's
hashes, validation accuracy, guard and validation example counts, parameter count, elapsed time,
and wake/relaxation/sleep work. Select one candidate per head by mean outer-
validation accuracy over all declared seeds, breaking ties by declaration
order. Do not choose seeds. The trial type contains no test score. Only after
the selection tuple is fixed, materialize final-test features once per seed
and score only selected heads. Keep confirmations in a separate result field.
A failed candidate raises with all attempts and completed trial rows; selection
and final test do not proceed.

## Alternatives considered

- Reusing the standard fixed-feature runner for each candidate would score
  test on every attempt and violate the selection boundary.
- Tuning different fields or counts for different heads would obscure the
  search budget and let one family receive more selection opportunities.
- Retraining selected heads after validation selection would add training
  work and another random run; the bounded route retains trained candidates
  until confirmation instead.
- A larger automated sweep is premature while capacity and process-isolated
  memory gates remain open.

## Consequences and open work

The route provides an inspectable, machine-readable validation ledger and
separate final confirmation, with a small local CPU smoke and sealed-loader,
test-label perturbation, failure, and candidate-order tests. It equalizes trial count
and validation access, not compute per trial; reported work counts expose
that difference. Failed-attempt evidence remains in memory through the
exception; durable on-disk journaling would be needed before long sweeps.
Retaining all candidate heads uses memory and limits the route to small
gates. ADR-0016 subsequently added a fixed-width capacity control.
ADR-0017 subsequently added process-isolated memory observations. Repeated
seeds and larger confirmation remain open under P1.8. No head-family winner is inferred from
the tiny synthetic random-feature smoke.
