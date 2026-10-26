# Frozen controller validation on complementary context orders

Use the existing comparison runner with an explicit opt-in:

```powershell
$env:PYTHONPATH = (Get-Location).Path
.venv\Scripts\python.exe -m scripts.run_bp_ticket_serving --source artifacts/runs/context-head-control-20261023/actual.json --context-stream-protocol --context-withheld-orders
.venv\Scripts\python.exe -m scripts.run_bp_ticket_serving --source artifacts/runs/context-head-control-20261023/actual.json --context-stream-comparison --context-withheld-orders --fitted-report artifacts/runs/context-controller-learning-20261023/actual.json --output local-withheld-orders.json
```

The output must not exist. The second command performs model work; freeze a
budget before running it. Do not repeat the paid qualification just to recreate
a handoff. Original protocols, recipes and default comparison remain unchanged.

Two separate six-packet recipes use old/new/new/old/old/new with delay1 and its
complement with delay2. Each has27 clean ordinary TRAIN rows. All14 applications
are fresh: no learning, periodic, drift, budget aware, frozen classifier chooser,
candidate diagnostic drift and verified-acknowledgement drift for each recipe.
The qualified chooser is the original classifier fit82; failed task-fit87 is
unavailable. Accepted fixed task-return heuristic88 is outside this scoped
classifier comparison; this is not a comparison against every heuristic.

Admission checks original receipt identity, FIFO prefix, observation/arrival/
consumption clocks, native packet indices and cumulative updates/rows/mistakes
from source72rows/16mistakes. Acknowledgements require a complete zero-mistake
candidate TRAIN diagnostic and are counted separately from optimizer work.
Changed, replaced or incomplete preceding paid evidence stops further cases;
all14 must complete before utilities are computed. No final/probe/replay access.

## Interpretation

The sole qualification invocation was incomplete: four cases completed, and
the fifth (A/fitted) published at10 then refused a stale chooser decision at11.
Five paid cases and20updates/90rows/7mistakes remain saved; nine cases never
started and no utility scores were computed. No cohort retry is authorized by
this record. It establishes no comparative outcome.

The subsequent software repair retires a contextual chooser after verified
drain/publication. The owner emits only wait after checking captured generation/
version, all original labels/consumption, no pending predictions, no ready work
and cleared feedback. The chooser's stale-version refusal remains intact.
Thirteen scalar checks per platform and source review pass; the native serving
tail and rollback after this repair remain unexecuted/unverified. The fifth
failed case's original rollback is also unverified; its saved actor remains c0.

The common live window17..22 has four old and two new opportunities, including
intentional denial22. These are known TRAIN-template development tasks, not a
sealed external final. Order and delay vary jointly. Serial retained heap and
asymmetric diagnostics make elapsed/RSS utility descriptive cost proxies, not
paired efficiency. Historical source training and discarded fit costs are
carried separately; no amortized benefit, unseen-task or CPC claim follows.

Extend the existing recipe tuple/runner only under a separately frozen question
and budget. Do not refit, select favorable orders or reinterpret a failed cohort.

[Issue89](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/89)
owns status. Local protocol and evidence:
`artifacts/runs/withheld-context-orders-20261023/`.
