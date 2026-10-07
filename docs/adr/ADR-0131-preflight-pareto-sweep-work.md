# ADR-0131: Preflight the multi-seed Pareto sweep's training work

## Context

P5.5b covers the remaining legacy sweep after ADR-0130 gated the
circadian-policy script. `run_pareto_hard_tuning.py` contained 10 backprop,
12 predictive, and 12 circadian search cells inside functions that also
trained models. Its three fixed seeds, 20 epochs, 2,500 synthetic rows,
and batch size 64 imply 81,600 possible wake optimizer calls before
early stopping. The old main initialized Torch before enumerating the
search space. Selection already uses outer validation and keeps final
testing pending.

## Decision

Extract the three ordered candidate lists into side-effect-free builders.
Pin their original canonical JSON SHA-256 values in tests, then use the
exact returned lists for both `estimate_vision_candidate_work` and the
existing three runner calls. Reject a candidate that changes fields
outside its model family before Torch starts. Print candidate/seed counts,
maximum training updates, and row exposures before resource construction;
`--estimate-only` exits there. Keep the same 1,000-planned-update automatic
ceiling as the policy script. An explicit sufficient
`--max-planned-training-updates` permits launch. New output records the
estimate and chosen ceiling.

The ceiling is a launch gate, not an optimizer-interrupt rule. It does
not select a smaller candidate subset or alter any seed, score, baseline
rate, model setting, or historical file.

## Alternatives

- Copy candidate counts into a separate constant: rejected because it
  could drift from the lists actually trained.
- Build datasets and count DataLoader batches afterward: rejected
  because Torch/data costs would occur before preflight.
- Truncate the cross-product to fit the ceiling: rejected because it
  would change the comparison and could bias the reported best model.

## Consequences

The unqualified Pareto command reports 34 candidates, 102 seed-candidate
trials, 81,600 possible updates, and 5,100,000 row exposures, then refuses
launch. The estimate excludes guard, validation, inference, weight setup,
time, and memory. Runtime limits and explicit stop reasons remain P5.5c/d.
No training result or scientific inference was produced for this task.
