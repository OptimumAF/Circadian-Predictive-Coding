# Frozen TRAIN standardization for the MiniGrid BP control

This explicit adapter tests input conditioning while preserving the original
BP/PC/CPC defaults and native mathematics. It is not the recommended game learner:
the first fixed observation improved row accuracy but worsened complete games.

## Contract and use

```python
from pathlib import Path
from scripts.run_minigrid_linear_fit_control import read_training_sources
from src.adapters.standardized_minigrid_learner import fit_policy, StandardizedMiniGridLearner

episodes = read_training_sources(Path("artifacts/runs/joint-catalogue-study-20261018/bp.txt"))
policy = fit_policy(episodes)
learner = StandardizedMiniGridLearner(policy)
```

The source decoder already returns raw initial/current296 histories; do not
encode history again. `fit_policy` validates complete distinct TRAIN sources,
canonicalizes only current observations and reuses population-statistics code.
It reads no targets/next views to choose statistics. Means/scales are frozen;
constant channels use scale1 and retain new-input deviations. No floors/clipping,
class weighting, automatic refit or task-specific cue rule is introduced.

This is an **offline** fit using all supplied TRAIN frames, not a claim that
future stream observations are available to an online controller. Non-TRAIN
sources are refused. Raw-array update calls retain the usual NativeLearner
contract; the existing stream owns role, causal arrival, budget and permissions.

The policy fixes native BP296/H32/K7/seed17/lr.03/momentum.9. Both training and
prediction apply the same raw canonicalization then `(x-mean)/scale`. Snapshots,
forks and restores own complete native optimizer/gradient/localRNG state and the
frozen transform. Foreign transforms/policies/malformed raw rows are refused
before publication of updates. The encoding ID describes a contract, not a
security boundary or authentication of arbitrary external weights.

Use the existing `AttributionStream`/`ToyBudgetSession` for updates. The training
runtime rejects any configured replay limit, including0; omit that optional
field and use the original training-only/no-replay execution path. Readonly game
budgets may explicitly limit updates/replay to0. Capture the candidate before
`stream.finish()` seals its public getter.

`evaluate_standardized_neural_checkpoint` in the existing MiniGrid evaluator
admits only fixed54-row source0/0 or endpoint128/6912 and reads both known TRAIN
A cue games. It delegates to the original frozen evaluator/physics with raw
seven-action first argmax, no mask/teacher/repair/live updates/final/probe data.
Source/state/transform/budget ownership remains checked during readout.

## Fixed observed negative

Original54x296 TRAIN features have210constant channels contributing193.72 of the
mean262.1852squared row norm. This motivated a conditioning hypothesis, not a
confirmed saturation bug. One standardized128update/6912row case reused the exact
original source/settings and saved unstandardized control; no baseline rerun.

| Metric | Original H32 BP | Standardized H32 BP |
| --- | ---: | ---: |
| TRAIN correct actions | 51/54 | 52/54 |
| Exact TRAIN traces | 3/6 | 4/6 |
| TRAIN mean CE | 0.1862326696 | 0.0818148483 |
| A-blue success / return | 0 / -1 | 0 / -1 |
| A-red success / return | 1 / 0.49375 | 0 / -1 |

Both standardized games use the wrong cue-dependent turn at step2 and terminate
after7steps with the wrong-key penalty. Better mean fit did not produce better
behavior. Standardization remains opt-in; no acquired retention/recovery,
generalization, capacity/optimization impossibility, efficiency or CPC benefit.
The accepted learned FIFO36 count-memory control already solves both cues and
its acquired-task/return evidence is reused rather than retrained.

The first native invocation failed at admission before any update because of the
unsupported training replay-limit field. It remains failed. A prospectively
amended same-case admission continuation fixed that field, kept source/seed/data/
metrics unchanged, and charged all work cumulatively within original limits.
This is not two independent experiments or an outcome-based retry.

26scoped checks/platform qualified via24initial plus2pure admission regressions;
four-file Ruff/format/fresh default+Linux mypy passed. The same agent reviewed
source/procedure/results; no independent acceptance or repository-wide claim.
Exact commands, failures, limits, full native observations and cumulative costs:
`artifacts/runs/standardized-minigrid-bp-20261023/RESULTS.md` and issue61.

## Structure and extension

```text
src/adapters/standardized_minigrid_learner.py  frozen statistics/native BP port
src/adapters/minigrid_policy_evaluation.py   narrow known-A facade/shared evaluator
tests/test_standardized_minigrid_learner.py  ownership/equivalence/refusal/admission
```

Run focused checks with `python -m pytest -q tests/test_standardized_minigrid_learner.py`.
Lint/format/type commands and paid fixture costs are recorded in RESULTS.md.
New normalization variants require explicit separate policy identities, valid
TRAIN-only sources and prospectively frozen experiments. Do not tune this one
against the failed games or use final/probe content to refit statistics.

Next reconcile the existing successful FIFO36 control's held-out-layout evaluation
support/evidence, then select a bounded held-out game outcome without retraining
or selecting a favorable intermediate checkpoint. Full R4.2 heldout/recurrentRL,
useful neural adaptation, scientific and release requirements remain open.
