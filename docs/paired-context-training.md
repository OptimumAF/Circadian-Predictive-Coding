# Paired-head ordinary TRAIN adaptation

The existing contextual stream and live worker can learn v1 parcel and v2
password tasks, rehearse four old tasks, and publish through the original inner
guards. Training comes directly from ordinary TRAIN teachers. Disposable probe
weights and final evaluation never enter this path.

Two explicit recipes share six packets of sizes9,9,9,9,6,6, delay1, and the
original six-update/60-second/512MiB session:

| Recipe | First four packets, alternating v1/v2 | Last two packets |
|---|---|---|
| `context_paired_current9_return` | New3 repeated three times | Old6 for v1, then v2 |
| `context_paired_joint9_return` | New3 plus old6 | Old6 for v1, then v2 |

Each arm pays48rows,24per head, in six native passes. Current-only pays36new and
12old exposures; joint pays12new and36old. Old-row reuse is paid TRAIN rehearsal;
`replay_examples=0` only describes the absence of the separate replay helper.
Equal rows and native passes do not establish equal total computation.

The paired family is old v1invoice/password,v2invoice/parcel plus new
v1parcel,v2password. It uses the existing24tick worker, contention/stale/revoked
permission scenarios, promotion guards and rollback. The source-free preview
enumerates every worker opportunity. Original recipes and comparison cohorts
remain unchanged. Paired selection supports periodic and no-learning references;
unsupported feedback/controller combinations refuse before source IO.

From the physical repository on D:, preview without source or model work:

```powershell
Set-Location D:\Dev\Repositories\Circadian
$env:TEMP = 'D:\Dev\_Temp'
$env:TMP = 'D:\Dev\_Temp'
.venv/Scripts/python.exe -B -m scripts.run_bp_ticket_serving --source unused --context-stream-protocol --context-stream-recipe context_paired_joint9_return --output artifacts/paired-preview.json
```

Output must be new. A separately budgeted new execution replaces
`--context-stream-protocol` with `--context-head-serving-live` and supplies a
qualified source. The API is also available:

```python
from scripts.context_head_live import run_case
from scripts.context_training_streams import CONTEXT_PAIRED_REHEARSAL_RECIPES

result = run_case(source_path, stream_recipe=CONTEXT_PAIRED_REHEARSAL_RECIPES[1])
```

The executed comparison used this API and the existing4MiB scientific writer.
The ordinary CLI parser/dispatch/preview were checked; a full native report
through that CLI's distinct output encoder was not exercised. Do not describe
this increment as installed-wheel qualification: the previously qualified
wheel predates these recipes.

One frozen local comparison found joint retained4/4old tasks after learning,
versus2/4 for current-only. Both learned2/2new tasks, reached4/4old after return,
passed the original promotion guards at tick14, and served7/8 of the fixed late
ordinary slots. Both published the same final learner. This demonstrates no
serving superiority, held-out generalization or circadian advantage.

See [executed checks, costs, failures and limits](reports/paired-context-training-20261029.md).
The sole comparison is spent; these usage examples do not authorize its replay.
Extend the existing data/stream/guard ports with explicit ownership and causal
label release. Keep any future private probes detached from serving.
