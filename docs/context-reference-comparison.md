# Fixed references on a learnable contextual policy

```powershell
$env:PYTHONPATH = (Get-Location).Path
.venv\Scripts\python.exe -m scripts.run_bp_ticket_serving --source artifacts/runs/context-head-control-20261023/actual.json --context-controller-comparison --periodic-report artifacts/runs/context-head-live-20261023/actual.json --output local-context-comparison.json
```

Output must not exist. This finite development comparison reuses the exact
accepted periodic record and runs only no-learning, drift and budget-aware, once
each, through the existing contextual live worker. It never deploys probe weights
or reads final evaluation. Source weights and periodic bytes are pinned; a failed,
corrupted or replaced preceding case prevents admission of another control.

For one control, use `--context-head-serving-live --context-live-reference drift`.
The live default remains periodic. Explicit nonperiodic modes can complete the
fourteen opportunities with unapplied TRAIN; `training_complete` distinguishes
that from completing both planned updates. No credits or clocks are renewed.

## Observed comparison

All modes use the same six known tasks at ticks7–12, including the same stale
scenario. The no-learning window supplies both baseline counts. Old and new
refer to four original tasks and two legacy tasks; v2legacy was already known.

| Mode | Old tasks | Legacy tasks | New passes / rows / mistakes | Pending TRAIN | Published |
| --- | --- | --- | --- | --- | --- |
| No learning | 4/4 | 1/2 | 0 / 0 / 0 | 2 | No |
| Periodic, reused | 4/4 | 2/2 | 2 / 9 / 2, historical | 0 | Yes |
| Drift | 4/4 | 1/2 | 1 / 3 / 1 | 1 | No |
| Budget aware | 4/4 | 1/2 | 0 / 0 / 0 | 2 | No |

Drift updates at tick1. The later old-task packet has zero errors on the deployed
actor, so the original drift rule waits. Pending repair prevents drained
publication; these serving outcomes do **not** show that its shadow learner
cannot fit the task. Budget-aware's unchanged reserve4 exceeds cap2, so it waits;
this is a disclosed unsuitable configuration, not general controller inferiority.

The unchanged objective reports utilities −0.111109, +0.388406, −0.118613 and
−0.120366 in table order. Only the task outcomes support this local adaptation
comparison. Full invocation time/RSS include conditional guard/rollback work;
periodic ran earlier and alone, while the new controls retained preceding reports.
These are unequal-work, unpaired cost proxies, not an efficiency or statistical
ranking. There is no CPC advantage, heldout generalization or concurrent inference
claim. All wrong workflows, denials and prior negative results remain saved.

## Boundaries and extension

`scripts/context_reference_comparison.py` composes the existing runner, existing
paid-report guard and pure `ControllerObjective`. It validates the frozen task
window before scoring, never trains a chooser, and never creates a new runtime.
Teacher, prediction, initialization, copying, worker and diagnostic costs count.
Source history12/72/16 and reused periodic2/9/2 are carried separately from the
new controls' total1pass/3rows/1mistake.

Extend a new scientific protocol prospectively through existing stream/controller
ports. Do not transfer the older native fitted chooser to this different source
version, refit from these outcomes, or retune the frozen controls. Broader unseen
stream adaptation, long/noisy deployment and repository release remain open.

Evidence: `artifacts/runs/context-reference-comparison-20261023/` contains protocol,
raw results, exact commands, readback, costs, independent review and preserved
failed software attempts. See [ADR0597](adr/ADR-0597-context-reference-comparison.md)
and [live task81](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/81).
