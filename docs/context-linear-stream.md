# Arrived affine TRAIN updates

The separate `ContextLinearLearner` begins untrained. By default each admitted
homogeneous public-cue batch replaces one head using fixed one-hot least squares.
The other head remains unchanged; unfitted heads predict uniform K7.

The adapter also exposes explicit `train_batch_minimum_change(features, targets)`
for use under the same exclusive admitted TRAIN lease. The first fit is identical
to the default; later fits reuse existing core math to preserve coefficients
invisible to the current rows. This does not guarantee preserving all old
predictions. It is nonsticky: a later `train_batch` still cold-refits. No old rows,
optimizer, role authority or budget is stored in the model.

The stream accepts `train_update="minimum_change"`; the choice follows every
owned candidate fork and does not change a restored snapshot's default operation.
The CLI exposes `--train-update minimum_change` with `--context-linear-stream`,
`--example-affine-play` and `--example-affine-recurring`. Omission keeps cold fits.
Private, A4, reference and unrelated CLI profiles refuse that new option before
data/model work. For example, this describes the recipe without running it:

```powershell
python -m scripts.run_minigrid_linear_fit_control --example-affine-recurring --train-update minimum_change --describe
```

Original data, schedules, role admission, budgets, clocks, receipts and output
limits remain. Changed update intent refuses further work. The
[adapter evidence](reports/minigrid-minimum-change-20261031.md) and
[stream/CLI evidence](reports/minigrid-minimum-change-stream-20261031.md) are
correctness checks. Integration fits, predictions and game results are scripted;
they do not qualify learned navigation, held-out layouts, comparisons or release.
Any genuine execution needs a separately budgeted protocol.

The [single incremental recurring-game qualification](reports/minigrid-minimum-change-native-20261031.md)
completed 10 solves/252 row exposures and all four final G/A games, each nine
steps. Old tasks stayed correct in the TRAIN curves, with the original A/C
confidence regressions. The earlier cold recipe already has the same
correctness curve and game successes; this result establishes no additional
navigation benefit, native recovery after forgetting or generalization.

Why this: the successful offline joint model already knows A/B/C and cannot
establish new-task learning. This owner exposes chronological acquisition and
forgetting through the existing delayed stream, without another runner.

## Use

From the repository root, with the accepted local TRAIN source available:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_minigrid_linear_fit_control --context-linear-stream --describe
.\.venv\Scripts\python.exe -m scripts.run_minigrid_linear_fit_control --context-linear-stream --source artifacts/runs/joint-catalogue-study-20261018/bp.txt --output new-stream-result.json
```

Fixed schedule: A blue/red → B blue/red → A blue/red, one nine-row episode per
update, label delay one, six fits / 54 row exposures. The opt-in command refuses
fit overrides and existing output files. Score four TRAIN episodes only after
all training, at fixed states0/2/4/6. Budget each invocation prospectively.
Existing frozen-game and joint-fit commands remain unchanged.

## Ownership and accounting

- The app owns TRAIN roles, arrival, receipts and original lifetime budgets.
  The array port cannot establish a role; use it through application admission.
- An update counts one committed analytic solve and every exposed row, including
  repeated A. It is not an SGD step. Diagnostic is fitted one-hot **squared
  error**, distinct from reported CE.
- Routing uses only raw initial/current public296 and the initially visible ball
  color. Task names and current targets cannot select the head.
- Full snapshots contain two optional validated fits, real committed counters
  and one bounded failed-work tuple. No row bank or success chain. Wrappers are
  detached; immutable tuple leaves may be shared. At most4,158 coefficients /
  33,264 logical float64 bytes per fitted owner, excluding object overhead.
- Fork/restore never renew app budgets. Failed owners refuse further fits or
  restoration that would erase paid work. Preceding models, attempted/returned
  counts and unknown work survive through `TrainingUpdateFailed`.
- Later failures retain prior original receipts and bounded states. Missing
  receipt evidence prevents completion. Copying, validation and diagnostics count
  toward costs. No data access, RNG, teacher, physics or promotion in the port.

## Executed observation — issue107

| State | A actions correct | B actions correct |
| --- | ---: | ---: |
| Untrained | 2/18 | 2/18 |
| After A | 18/18 | 13/18 |
| After B | 13/18 | 18/18 |
| After A returns | 18/18 | 13/18 |

Both current-phase nine-action traces fit; neither complete other-task trace is
retained. Returned A coefficients exactly equal the first A fit. This is TRAIN
acquisition, forgetting and recovery, with no joint retention, native task
completion, generalization, statistical reliability or CPC advantage established.
Final/probe/controller-validation labels and updates remain unused.

Sole run:6fits54rows,12fitted-core forwards108rows plus4uniform reads36rows,
16metric calls144rows,107snapshot copies165model validations. Body6.261s,
parent6.950s, sampled peak62,722,048B. Logical coefficients are not FLOPs.
31distinct focused cases qualify per platform, scoped statics and self-review;
not independent acceptance or release readiness. Two local-artifact integration
cases executed here and explicitly skip when the source is absent elsewhere.

[Protocol](../artifacts/runs/context-linear-stream-20261024/PROTOCOL.md),
[commands/results](../artifacts/runs/context-linear-stream-20261024/RESULTS.md),
[issue107](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/107).
Next: qualify native A/B/A gameplay from these causal states through the existing
evaluator, without refitting or another capture/provenance framework.
