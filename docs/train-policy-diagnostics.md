# Train-only closed-loop acquisition

## Existing weighted objective option

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-weighted-train-live --describe
```

Omit `--describe` for the explicit weighted recipe. It keeps the same64 updates,
seed17, source packets, five raw unweighted train-score points and legal live
readout at0/64. Only the training objective changes to the existing per-episode
weighted mean CE. Each present action class gets weight
`episode_length / (present_classes * class_count)`; absent classes get neutral
weight1 without inventing labels. Native CE divides by observed target-weight
mass. Weights derive only from the current complete TRAIN episode.

`src/adapters/torch_train_weighted_learner.py` binds the original sequential array
releases to owned complete training episodes. It validates matching arrays,
schedule prefix and counters before delegating the existing native weighted
method. Prediction, live sessions, optimizer and native snapshot format remain
shared; forks own independent native state and schedule copies. The adapter
does not grant release authority or prove physical provenance: original stream
receipts and caller state guards retain that responsibility. Explicit restore
validates native state against the retained schedule prefix; snapshots themselves
do not authenticate which objective originally produced arbitrary weights.

Why this: supported but poorly fitted rare required actions motivate testing an
existing objective at the same dose. Imbalance is not established as the cause
of failure. The earlier weighted BC8 negative remains valid. This option has no
final/probe data, promotion, sweep, automatic dose extension or selected seed.
The ordinary mean-CE commands and report schema remain unchanged.

The new component tests are `tests/test_torch_train_weighted_learner.py`; caller
tests are `tests/test_weighted_train_live_pilot.py`. Exact checks, observed costs
and comparative results belong to
[the weighted stage](../artifacts/runs/r21-weighted-train-live-20261016/RESULTS.md).
Correct execution alone cannot establish useful adaptation or CPC advantage.

The combined263 focused cases and bounded native CLI pass. The measured weighted
endpoint has raw CE.9859493 versus mean.9456660, tied accuracy.625 and zero complete
sequences/live successes. At16/32 updates weighted accuracy is higher, but its raw
CE is worse at every nonzero fixed point; all points remain reported. Both priors
still refuse and their comparison is undefined.64SGD/512rows/640unrolls/58native
predictions/8environments/70physicssteps/8priorcallbacks cost19.1850599parent
seconds, including adapter validation. These are not concurrent speed measurements.
The simpler mean endpoint remains preferable in this single train-only case.

The previously missing [A128 equal-A-exposure mean control](train-acquisition.md)
now reuses accepted A64/B64/A64 evidence (128A+64B), with128vs192 total updates
explicit. It reaches partial train acquisition only. Dose128 follows that
existing exposure, not a larger dose selected from failed weighted outcomes.

## Original mean-CE diagnostic

From the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-train-live --describe
```

Omit `--describe` to execute the fixed local recipe. It retains the existing
64-update A cue± mean-CE acquisition stream, seed17, H8, rate .03 and momentum .9.
After all training finishes, owned checkpoints0/64 each play both training cues
with the existing legal-conditioned greedy evaluator, at most12 steps each.
Final/probe data are absent; diagnostic results never update training or serving
weights. The original `torch-key-door-train-fit` route remains available.

The pooled training-label prior plays both cues once, with identical legal
information. It is a descriptive reference with unequal computation. A valid
prior can assign zero mass to the actual legal support. That episode explicitly
refuses: its applied prefix and pending unexecuted decision remain in the report,
but success and aggregate comparison are unavailable. There is no smoothing,
fallback action or imputed failed baseline score. Any other fault stops work.

Why this: teacher-forced fit alone does not establish operational control. The
accepted case improves train CE1.6098073→.9456660 and accuracy.125→.625, but all
four learner episodes reach the12-step limit with zero return. Each prior cue
refuses after three applied steps; the learner/prior comparison is undefined.
This is a negative train-resubstitution result, not generalization or CPC benefit.

## Structure and boundaries

```text
src/adapters/train_acquisition_pilot.py     existing stream/receipts/state authority
src/adapters/train_policy_diagnostics.py   six bounded slots; owned live readout
scripts/run_coupled_attribution.py          explicit CLI dispatch
tests/test_train_policy_diagnostics.py     support/refusal/isolation/cost behavior
tests/test_train_live_acquisition_pilot.py  caller authority/default compatibility
```

The helper accepts the completed training report, owned states0/64, unchanged
training prior and original budget/guard. It returns episodes, gains, refusals
and attempted/completed work. It does not train, select probes, promote models,
read final data or perform IO. Existing evaluator and native mathematics remain
the execution boundaries. Six fixed slots and frozen preceding records bound
storage. Actual visible support, source/state/reference and original receipts,
clock and charges are checked before admitting more work.

No dependencies, environment variables or settings are added. Existing CPU/thread
controls apply. The single native case used64SGD/512train rows/640unrolls,
58native predictions,8environments/70physics steps and8nonnative prior callbacks;
parent8.6589473seconds. Entry and5ms recipe RSS samples passed their declared
ceilings; these are not a continuous whole-worker memory certification.

## Verification and extension

Prefix `.venv/Scripts/python.exe -X utf8 -B`:

```text
artifacts/runs/r21-train-live-20261016/run_checks.py statics final
artifacts/runs/r21-train-live-20261016/run_checks.py focused final
```

These already executed successfully and save exclusive receipts; do not overwrite
them. The controller records exact Ruff format/check and Windows/Linux mypy argv,
and the eight selected pytest files.197 latest cases pass after six preserved
test-expectation failures. [Commands, costs and limitations](../artifacts/runs/r21-train-live-20261016/RESULTS.md)
and [independent review](../artifacts/runs/r21-train-live-20261016/REVIEW.md).

Extend with an explicit objective option through this stream, keeping original
mean-CE reports and seeds/dose/metrics fixed. Reuse existing weighted-episode
mathematics rather than adding another learner or orchestration framework.
Review/freeze any new comparison before native work; preserve failure and null
comparisons. Useful acquisition, retention, recovery and release remain open.
