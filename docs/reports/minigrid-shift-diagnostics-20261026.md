# Arrived-only native MiniGrid shift diagnostics

[R2.7-MG.D / issue230](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/230).
Previous/current goal turns PROGRESS. Scoped software implemented and qualified
through the existing reference stream, with source/procedure/result **self-review**,
not independent acceptance. Broader R2.7, goal and parents21/23/33 remain open.
No genuine learned/native experiment was run.227remains failed/spent, with
unknown joint task retention; earlier negative results are unchanged.

## Behavior and design

`run_context_linear_stream(..., reference_policy=ReferencePolicy("periodic"),
retain_arrived=True, shift_diagnostics=True)` observes all six newly labelled
TRAIN packets exactly once before that tick's reference choice and fit. The
flag defaults to False, requires an exact bool and the existing reference lane.
Drift's original explicit feedback requirement is unchanged. No new CLI flag,
fresh-workflow option, controller, budget, runner, promotion or new dependency.

Inspection exposed a concrete prerequisite: existing ArrivedShiftObserver
requires consecutive fit counts and <=16features. A checkpoint-only bridge
would miss labels under no_learning/waiting and count joint replay as arrivals.
Instead, the existing reference loop now accepts an optional observe_arrived
callback after original label admission. It receives detached TRAIN packet and
current candidate under the original lease/budget. No callback is added to
probe/final, contextual controller, default run_until or pause/restore paths.
Default reports omit the new field. Pending packets remain pending; no tail
drain or new fit is introduced.

The native wrapper verifies the original joint fit template, then passes only
the newly arrived nine rows, not older replay rows, to one readonly reader.
Predictions retain full296public history coordinates and the current candidate's
actual version/update count. Those counts may stay zero throughout the window;
arrival index is separate. Source/candidate/actor inputs and Python/NumPy RNG
remain unchanged in scoped fixture checks.

Eight fixed public feature values feed the existing categorical shift metrics:
initial/current frame object/color/state channel means and direction. No context,
target or outcome chooses features. H(p)/log(7), mean(1-p_true), default3-batch
reference/2-batch window/2-persistence thresholds, and onset/continued/clear
incident math remain unchanged. Core binary/K7/ArrivedShiftObserver16-column
contracts are not widened. Measurements occur after labels arrive; pre-label
alarm usefulness is not claimed.

Source/actor/candidate/runtime/inbox/clock/budget/input changes and reentrant
updates are refused before later work. Guarded callback attempts/completions,
full-template input bytes and unknown failed-call work are separate from native
prediction/metric attempts/returns on nine current rows. Paid failures stop the
window. The diagnostic stores one reader and six scalar slots, never a model
history or unlimited success chain. Reports are detached. Native reader numeric
peak and explicit input/guard/summary/returned-prediction arrays are subtotals;
state/Python/wrapper copies, mean temporaries and native projection allocations
are covered by process RSS, not claimed exhaustively counted allocations.

## Combined source and executed checks

```text
src/app/coupled_attribution.py                 optional reference hook/guard/counters
src/adapters/minigrid_shift_diagnostics.py     bounded native reader/public summary
scripts/run_minigrid_linear_fit_control.py    Python stream option/composition only
tests/test_minigrid_shift_diagnostics.py      causal/ownership/failure/metric checks
artifacts/runs/minigrid-shift-diagnostics-20261026/
  PROTOCOL.md, qualify.py, supervise.py, *-parent.json, *.xml, *.txt
  combined.patch, software.json, statics.json, review.json
```

AST review confirms only AttributionStream.run_reference_window,
_run_policy_window and the new _observe_reference_arrival helper change in the
existing app module; only run_context_linear_stream changes in the command
module. Parser/main and all existing modes remain unchanged. The native learner,
solver, shift metrics/incidents and original observer have unchanged source.
Inherited dirty snapshots are retained; no unrelated or Git changes.

32distinct cases/platform:27initial new cases plus one legacy ordinary stream
regression, then three timing cases and one invalid-prediction case. Windows
44instances=43pass/1intentional regression failure; Linux36pass. No skipped or
errored cases. Final affected8/platform pass. Four reference controls compare
enabled/disabled actual choices, receipts and unscored native states exactly.
No-learning receives six diagnostics/54rows with zero fits and six pending
packets. Joint templates expose90rows to the generic callback, while signals
use54current rows. Real pure summary/entropy/residual/detector arithmetic runs
on literal heads; no genuine learned forward, fit, solver or simulator runs.

Self-review reproduced a real timing bug: a backward end clock marked the row
failed but left completed_arrivals=1. Windows3 failed that assertion after one
literal prediction/nine rows/one metric, with0fits, .9125721001s body/
2.365179s parent/RSS81,670,144B. Completion now advances only after valid timing;
invalid start/end permanently stops and retains known read/metric costs. Returned
array bytes are charged before subsequent guard failure, and metric attempts/
returns are distinct. Original failed attempt and all costs remain preserved.

The first Windows receipt counted only15explicit fixture callbacks, omitting
native adapter delegates and metric calls. It remains frozen. The budget ledger
conservatively adds an upper64callback calls and64metric calls for that cohort,
based on its four six-arrival studies and three two-arrival failed studies.
Subsequent runs directly meter all core callback/K7 metric calls. No missing
work is treated as free or retrospectively claimed metered. Initial three
argument-mapping mypy errors were corrected with explicit annotations; no ignores.

| All software attempts | Windows | Linux |
|---|---:|---:|
| Body seconds | 9.1112525999 | 17.9265511780 |
| Parent seconds | 14.9897262999 | 32.9942867002 |
| Sampled peak RSS bytes, after imports | 81,936,384 | 74,825,728 |
| Literal owners / snapshots | 223/2,484 | 158/1,709 |
| Scripted fits / rows | 60/702 | 42/486 |
| Scripted prediction attempts / rows | 140/1,476 | 97/1,017 |
| Charged callbacks / metrics | upper134/119 | metered70/54 |

All original180body/220parent/512MiB/count/storage caps hold. Genuine learned
work, LAPACK, gradients/latent updates, teacher/physics, model random draws,
probe/final data, optional simulator/Torch imports and scientific raw reads0.
The literal model itself performs public-input projection and table arithmetic;
that is not a learned forward. No live/native or comparison advantage inferred.

Final six-file Ruffformat/check/default+Linux-target mypy all pass. Total
observed static/format calls27.0404657s, below120s. Exact commands/receipts:

```text
.venv/Scripts/python.exe -B artifacts/runs/minigrid-shift-diagnostics-20261026/supervise.py windows-1 .venv/Scripts/python.exe -B artifacts/runs/minigrid-shift-diagnostics-20261026/qualify.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/minigrid-shift-diagnostics-20261026/supervise.py windows-4 .venv/Scripts/python.exe -B artifacts/runs/minigrid-shift-diagnostics-20261026/qualify.py windows-4 tests/test_minigrid_shift_diagnostics.py::test_should_keep_choices_receipts_states_and_rng_while_diagnosing_all_arrivals tests/test_minigrid_shift_diagnostics.py::test_should_not_count_an_arrival_complete_after_failed_timing tests/test_minigrid_shift_diagnostics.py::test_should_charge_a_returned_invalid_prediction_and_failed_metric
```

Linux uses the same qualifier through cached WSL Ubuntu Python, CPU1/OMP1,
timeout200s; full expanded environment/path/commands in linux parent records.
Static commands target the six Python paths above with `ruff format --check`,
`ruff check`, and `mypy --explicit-package-bases --follow-imports=silent`, also
`--platform linux`. No production dependencies or environments changed.
One literal Windows wildcard and one absent guessed path read were simplified
to exact source paths; no tooling workaround framework or experiment retry.

No whole-repository, new wheel/install, UI, CI/GPU, independent review or native
scientific experiment was performed; those were outside this scoped bridge.
The accepted229wheel intentionally predates these new source changes. It is not
relabelled as containing230. Earlier BP/PC/neutral/active and negative/native
evidence remains unchanged. Historical plans/log stay frozen; live issue handoff
records status under AGENTS. Final metadata records cache and storage footprints.

## Next useful scientific task

230's software acceptance is complete after self-review; useful/deployable
shift detection remains unverified. Do not add another CLI, package, capture or
diagnostic framework as an automatic follow-up.

Next precise action: freeze a small four-case native protocol using the existing
AttributionStream/reference callback and this adapter: fixed no_learning and
periodic controls on stationary repeated-A and original A/B/A TRAIN arrivals,
with the same six-arrival window, default detector, source/seed/metrics and all
costs fixed before execution. Preserve default reference contamination/late
alarms/misses honestly; do not move the reference window or tune thresholds to
force a useful trigger. Use existing packaged TRAIN examples, no teacher/physics,
probe/final data or private F comparison. Reuse software gates, review the small
protocol before any fits, and budget discarded work explicitly. This asks whether
the new signals distinguish these fixed streams and what the readouts cost;
it grants no automatic run or renewed227allowance and cannot establish general
deployability/CPC benefit. Full goal remains ACTIVE and incomplete.
