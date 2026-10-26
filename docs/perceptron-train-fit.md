# Bounded multiclass perceptron TRAIN control

The existing saved-TRAIN fit CLI now has a separate `--perceptron` option. It
uses the ordinary delayed TRAIN-only attribution stream and owned learner ports.
The original least-squares, BP, PC, neutral CPC and active CPC recipes stay frozen.
This is a simple trainable classification control: the earlier52/54 least-squares
fit does not establish linear classification impossibility.

## Use

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_linear_fit_control --perceptron --describe --fit-updates 128 --fit-seconds 30
```

To execute after freezing and reviewing a small real-source protocol, add
`--source PATH_TO_TRAIN_REPORT --output NEW_RESULT.json` and remove `--describe`.
The CLI refuses existing outputs. It decodes only complete original148-frame
TRAIN records, constructs raw initial/current296 histories, and admits all six
causal source episodes before resources or learners. No final or reserved probe
records enter training. `--bounds-only`, `--constructive-bp`, cue-relative encoding
and gradient diagnostics cannot combine with `--perceptron`. The original CLI
routes retain their defaults and dispatch. No dependency or product environment
variable is added.

The API is `minigrid_constructive_fit.diagnose_train_fit(episodes,
perceptron=True, updates=128, max_wall_seconds=30)`. The existing default remains
the original H8 BP512 fit. Explicit perceptron callers share the existing1..512
pass and(0,90]second limits; fixed checkpoints are0/min(128,updates)/updates,
deduplicated. All requested passes execute before scoring: no best-checkpoint
selection, early-success stop, shuffling, averaging or tuning.

## Rule and complete state

Core policy admits D1..296/K2..16, batch B1..128, finite exact float64 features
in[-1,1] and hard labels. Coefficients have shape(D+1,K), initially all zero;
an intercept1 is appended. Each `train_batch` makes one supplied-order pass.
For each row, choose the first maximum linear-score class. On a mistake, add the
augmented row to the target column and subtract it from the predicted column,
using fixed unit step. Correct rows do not change coefficients. One pass always
advances the pass counter and charges every row, even with zero mistakes.

`TrainingDiagnostic.definition` is `categorical_perceptron_mistakes_v1`; its value
is the number of row mistakes in that pass. This is not BP CE, PC energy, an
optimizer step or a common loss. Prediction uses stable softmax and the existing
CE/first-argmax/exact-trace scoring; CE improvement is not the update objective.

Immutable snapshots contain the exact policy, coefficient tuples and pass/row/
mistake counters. Counters are exact bounded native integers; coefficient
magnitudes are bounded by lifetime mistakes, keeping admitted scores finite.
Copies, restore and fork own complete records. Invalid batches, policies or
snapshots fail before committing a native pass. No RNG or optimizer state exists.
The MiniGrid wrapper binds a distinct snapshot family to the original invariant
encoding and296/K7; raw/generic/foreign-encoding snapshots refuse. It projects
every supplied raw history before native updates. Data-role authority stays in
the caller's existing TRAIN admission and stream.

Source and actor stay unchanged. The original budget owner, clock, limits,
sampler, charges and completed candidate remain guarded through scoring. Existing
failure output retains stream receipts and observed native counters separately:
a completed native update with a lost return can leave native1pass/54rows while
the stream has0receipts/charges. That prefix and unknown-work flag are preserved.

## Work and storage

`native_completed_updates` means passes; `native_training_rows` counts row
exposures. `work.perceptron` reports completed passes, rows, online decision rows
and mistake updates. Each row evaluates seven linear scores; every mistake
updates two297-coordinate columns. Prediction calls/rows are separate existing
work counters. Optimizer/backward counts are0. Projection/admission/validation,
copies, Python objects, wall time and process memory are additional costs.

The coefficient table has2079numbers/16632float64-array-equivalent bytes, not an
actual resident-array or process-memory measurement: stored tuples have Python
overhead. The existing stream owns54-row input copies per pass, retaining up to
three checkpoints. This has different capacity, dtype, update objective and work
from H8 neural methods; equal row exposure does not make computation matched.

## Executed evidence

Independent scalar-list arithmetic checks every ordered update, ties and
intercept. A fixed separable three-class example learns after eight prescribed
passes. Full snapshot/fork/restore, atomic invalid input/state refusal, RNG,
role/encoding gates and existing default fit regressions pass. **179distinct
scoped cases accepted**,258test executions all pass; nine-path Ruff/format,
five-source Windows/Linux mypy and independent SOURCE/PROTOCOL/ACTUAL GO.

The public synthetic case completes3passes/162rows with12scorecalls/108rows.
It has161mistake updates, full-fit witness **false**, exact-trace accuracy0 and
row accuracy2/9; CE1.9459101491→3.6434873611. Independent saved arithmetic checks
all108probability rows. This negative fixture result remains preserved and does
not establish real-source acquisition, linear impossibility, retention, recovery,
generalization or CPC benefit. No real-source fit ran in this increment.

Both focused and final commands retain distinct reports. Across them:126pure
update attempts/1168input rows,60returnedpasses/826rows;88prediction attempts/
842rows,42returns/536rows. Attempts include refused invalid inputs; validation
and wrapper-only refusals add work outside those native entry counters. Original
Torch observed work stays0; there are0new teachers, physics or final episodes.
Initial review found possible report overwrite and it was repaired before tests;
two literal patch-context failures were simplified without any model execution.
Existing JUnit record_property/xunit2 warnings remain recorded. No failed tests.

[Frozen scope/commands/receipts](../artifacts/runs/perceptron-train-fit-20261018/RESULTS.md),
[independent review](../artifacts/runs/perceptron-train-fit-20261018/REVIEW.md).

## Structure and checks

```text
src/core/categorical_perceptron.py          pure rule, policy/state, probabilities
src/adapters/perceptron_learner.py          NativeLearner ownership/atomic commit
src/adapters/invariant_perceptron_learner.py raw296 projection and family binding
src/adapters/minigrid_constructive_fit.py   existing TRAIN stream/budget/report
scripts/run_minigrid_linear_fit_control.py  existing saved-source CLI option
tests/test_categorical_perceptron.py        independent arithmetic/separable task
tests/test_perceptron_learner.py            native ownership/atomicity
tests/test_minigrid_perceptron_fit.py       encoding/CLI/stream/failure costs
```

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_categorical_perceptron.py tests/test_perceptron_learner.py tests/test_minigrid_perceptron_fit.py
.venv/Scripts/python.exe -m ruff check src/core/categorical_perceptron.py src/adapters/perceptron_learner.py src/adapters/invariant_perceptron_learner.py src/adapters/minigrid_constructive_fit.py scripts/run_minigrid_linear_fit_control.py
.venv/Scripts/python.exe -m mypy --python-version 3.11 --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/core/categorical_perceptron.py src/adapters/perceptron_learner.py src/adapters/invariant_perceptron_learner.py src/adapters/minigrid_constructive_fit.py scripts/run_minigrid_linear_fit_control.py
```

Use `ruff format --check` with the same source paths and `--platform linux` for
the second type check. Exact executed test selection, exclusions and budgets are
in RESULTS. Historical next action, completed below: freeze/review one real-source original54-row/128-pass fit using
saved TRAIN sources, then execute and retain all fixed readouts/full-fit outcome
and costs. No solver/dose/seed/metric selection or baseline replacement. The
control's finite failure is inconclusive about general separability. Extend the
existing learner ports/stream, not a new orchestration layer.

## Real-source128-pass case

The frozen original54-row TRAIN case now completed and qualified:52/54correct,
4/6exact,CE0.5659225130,1743mistakeupdates. The full-fit witness remains false;
A-blue step2/B-blue step1 miss, and A-blue CE worsens to3.0701134573. Both fixed
readouts/128receipts/all108probability rows were independently verified. No
retention/recovery/linear impossibility/CPC claim, no retry/dose extension.
[All results, costs and next nonlinear control](reports/perceptron-original-train-fit-20261018.md).
