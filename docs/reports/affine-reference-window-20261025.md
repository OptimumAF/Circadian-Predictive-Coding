# R7.2-MG — finite TRAIN reference windows, partial native qualification

[Issue177](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/177)
owns acceptance. The fixed-window API is implemented and software-qualified.
Three native TRAIN windows executed successfully; their game comparison failed
and remains incomplete. The implementing agent reviewed source, procedure and
results; this is not independent acceptance or repository-wide readiness.

## Delivered behavior

`src/app/coupled_attribution.py` adds terminal `run_reference_window` and detached
`reference_report` to the existing stream. Fixed no-learning/periodic/budget-aware
policies share the same finite arrivals, FIFO/native port, causal labels, original
budget, source/actor isolation and receipts. Window-complete and training-complete
are separate; pending packets remain visible. Unsupported scopes fail before
ticks, no implicit drain/acknowledgement/evaluation/authority renewal.

`src/adapters/minigrid_policy_evaluation.py` adds a distinct strict analytical
reference-prefix readout, allowing0..6 actual fits and explicitly unfitted heads.
Original recipes retain their fit/row requirements. A post-invocation repair
extends the existing A3/A4 branch-specific episode naming to this new role.
The small internal ID helper preserves original names for all old roles.

New tests: `tests/test_attribution_reference_window.py` and
`tests/test_affine_reference_readout.py`. [Usage](../attribution-reference-window.md).
No new learner/objective/controller/scheduler/dependency or state format.

## Software evidence and failures

Stage `artifacts/runs/affine-reference-window-20261025/` retains protocols,
self-review, exact argument arrays, logs, XML, counters and parent receipts.

| Corrected scoped cohort | Windows body / parent seconds | Linux body / parent seconds |
| --- | ---: | ---: |
| Window39PASS | .951012 / 1.7854084 | 5.802672063 / 11.6370114 |
| Prefix21PASS | .7502832 / 1.5723514 | 5.412430871 / 10.6373483 |
| Post-failure ID3PASS | .737512 / 1.6028360 | 7.269682884 / 13.7662948 |

Window RSS85,684,224B/87,199,744B; prefix91,713,536B/88,543,232B. Software sampling
covers bodies after imports, not import/process-tree peaks. Each final window
cohort observed59scripted train calls/59rows and6scripted predictions/6rows;
52literal train calls cross-check directly against fixture counters. Prefix
checks use22empty analytical metadata owners/platform and no solve/prediction/
physics. ID repair RSS86,716,416B/88,580,096B, with no model work.
63distinct checks qualify/platform;
affected checks were executed on final relevant source, unchanged checks reused.

The first Windows window cohort had38PASS/1FAIL: the test incorrectly expected
ToyBudget's committed counter to charge an unreturned call. Original behavior is
committed0/receipt0, attempted1/unknowntrue with a stopped owner. The assertion
was corrected without changing that contract. Its counter instrumentation also
missed literal classes due pytest's duplicate module name:7Spy calls were
observed,52additional literal calls/rows are source-bounded, not retroactively
observed. Both defects and the original receipt remain; one prospectively allowed
39case repair used importlib module identity. Cumulative Windows<=118scripted
train calls/rows,12predict calls/rows; Linux59/6. No genuine model work in those
scripted cohorts. Additional prefix and ID checks are metadata-only.

Initial statics caught a missing test annotation, pytest/mypy module namespace
collision, mixed launcher-result typing and unused case import. Repaired with
an explicit annotation, explicit-package-bases, a typed elapsed comparison and
removal of unused import; no ignores. One no-write patch miss and optional
orientation-path misses are retained. Final scoped Ruff/format and default+
Linux-target mypy pass; exact commands/durations in static JSON receipts. Final
metadata records whitespace/cache/storage. No full-suite/clean-clone/current-
wheel/browser/independent/neural/robot/release qualification was run.

Exact entry commands (native entry is SPENT, not a rerun instruction):

```console
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/affine-reference-window-20261025/qualify.py windows-final
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/affine-reference-window-20261025/qualify.py windows-readout readout
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/affine-reference-window-20261025/qualify.py windows-identities identities
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/affine-reference-window-20261025/case.py
```

Linux equivalents use cached WSL Python, CPU1 environment and finite timeouts;
their complete arrays are in `linux-*-parent.json`. Final static commands include
the two production files, two tests and two local finite launchers.

## Sole native invocation — FAILED/SPENT

Frozen before execution: same fresh A3/A4/A3 six TRAIN packets/labeldelay1,
seed17/K7/firstargmax; no_learning,periodic1,periodic2. All training before any
games. Fixed causal ticks2/4/6, four A3/A4 blue/red TRAIN games per phase. Original
training and readout owners/time/RSS/counter/storage caps; no old research model/
JSON reader, controller fit, tuning, final/probe/replay/sleep/promotion.

All three native TRAIN windows completed:

| Fixed reference | Fits / rows | Pending packets | Actual applied ticks |
| --- | ---: | ---: | --- |
| No learning | 0 / 0 | 6 | none |
| Periodic1 | 6 / 54 | 0 | 1,2,3,4,5,6 |
| Periodic2 | 3 / 27 | 3 | 2,4,6 |

Each had the same seven opportunities and54arriving TRAIN rows. Source/actor
and original training-budget binding checks passed after training. The sparse
reference trained FIFO packets0/1/2, leaving packets3/4/5 pending; it did not
silently reach later returning-context packets. Equal opportunities are not
equal optimization, training exposure, memory or computation.

The first no-learning game phase returned four closed episodes/64prediction and
physics steps, then `_require_phase` correctly refused duplicate A3/A4 episode
IDs. The new role had not entered the existing branch-row naming condition.
That prevents qualified phase acceptance; no complete game comparison or
old/new/recovery score was produced. All three `outcomes` remain null.

Exit3, body5.915575200s/parent6.338354900s, sampled RSS83,275,776B including optional
imports. Observed10empty owners/1,142snapshots,9native fits+9LAPACK solves/81rows,
64live predictions/64rows,6environment starts/closes,82total steps including
18teacher steps for two fresh A4 sources. All observed caps held. Full final RNG
and broad post-readout preservation checks were not reached and remain unknown;
later software tests cannot backfill them. No retry, native continuation,
endpoint reconstruction or extra raw result reader. Original output/protocol/
start marker/receipts remain unchanged. Scalar terminal result was consumed in
the original invocation. Weights were not saved.

The ID source repair and three pure regression checks qualify naming only.
Repaired native readout/rollback/full comparison remain unexecuted. Do not turn
this failed engineering qualification into a negative scientific ranking or a
successful learning demonstration.

## Remaining scope and next action

Issue177's native comparison acceptance remains unchecked; full R7.1/R7.5/R7.7,
R2/R4/R5 and science/release goals remain open. #89's earlier incomplete/spent
comparison, #103's negative validation, #175/#154 UI gate and all previous
negative or unknown evidence remain unchanged. Historical plans/log are frozen
under the current issue workflow.

Next implementation: expose the qualified fixed-window TRAIN API through an
explicit option in the existing affine CLI, with literal routing/exclusive-output/
pending-work tests and no automatic game evaluation or native retry. This makes
the new behavior accessible without a research-only script. Scope that product
entry separately; do not manufacture completion of this spent game qualification.
