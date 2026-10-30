# Publish learning before the return phase

The paired ordinary TRAIN API can assess publication at the end of learning,
then continue the predeclared old-task return on the same runtime and budget.

```python
from scripts.context_head_live import run_case
from scripts.context_training_streams import CONTEXT_PAIRED_REHEARSAL_RECIPES

result = run_case(
    source_path,
    stream_recipe=CONTEXT_PAIRED_REHEARSAL_RECIPES[1],
    publish_after_learning=True,
)
```

The existing command exposes the same option. Preview without source reads,
teachers or learning:

```powershell
Set-Location D:\Dev\Repositories\Circadian
$env:TEMP = 'D:\Dev\_Temp'
$env:TMP = 'D:\Dev\_Temp'
.venv/Scripts/python.exe -B -m scripts.run_bp_ticket_serving --source unused --context-stream-protocol --context-stream-recipe context_paired_joint9_return --context-publish-after-learning --output artifacts/boundary-preview.json
```

For a separately budgeted execution, replace `--context-stream-protocol` with
`--context-head-serving-live` and supply the qualified source path. Choose
`context_paired_current9_return` for the fixed current-only control. Both output
paths must be new. `--context-live-reference periodic` is optional for live
execution; the preview declares the default periodic reference. Other references
and additional feedback/withheld/controller options are refused before work.

The new flag's output limit is3MiB. Other command modes keep their existing
limits, including the old contextual live route's1MiB. If encoding or the byte
limit fails, the command returns1 and saves a failed scalar summary of reported
work, original budget/clocks and memory, at most32KiB. Missing/invalid values are
null. Detailed outcomes and publication receipts are unavailable; this summary
cannot authorize continuation or establish that publication did or did not occur.
It does not repair or recover a failed capture.

Run from the physical repository on D:, with process-local `TEMP` and `TMP`
set to `D:/Dev/_Temp`. This explicit option performs learning. It supports only
the two exact paired recipes with the periodic reference and no extra feedback
or catalog/controller mode. Invalid combinations refuse before source reads.
Defaults and the existing full-stream publication route remain unchanged.

The first four packets alternate v1/v2 learning; old-v1/v2 return observations
occur at ticks12/13, with labels13/14. All six packets still total48rows and
use the original six-update/60-second/512MiB session. The ordinary24worker
opportunities, fault scenarios and promotion thresholds are unchanged.

A qualified first candidate publishes when its four receipts drain. A measured
rejection preserves the actor and admits only the declared return window.
Once return training drains, the original guard reassesses the repaired candidate.
An early accepted publication skips this second attempt. There are at most two
guard records and one accepted publication; rollback restores the original source
while retaining generation history and all charges.

The rejection path requires the actual typed guard refusal, original decision,
state digests, revision and actor/candidate values. Missing or lost callback
returns, altered/replaced evidence, changed model/configuration/metadata, or
changed original budget/receipt/sharing counters stop further work. Ordinary
serving may populate its cache; that does not authorize model or counter changes.
Future labels are recorded against the actor actually observed, never rebound.

Reports expose `learning_boundary_plan`, `learning_boundary_tick`,
`learning_boundary_guard`, `drain_publication`, optional `return_publication`,
`publication_tick` and `return_publication_attempted`. Candidate and deployed
state differ after early publication: further private candidate training does
not update the actor implicitly. Published readouts and rollback readouts each
cover all six declared tasks. A failed later native return keeps the known
publication and permanent attempted-work charge; it does not claim rollback.

One frozen local case published joint rehearsal at tick10 and current-only's
repair at16. Joint passed8/8 fixed late ordinary slots, current-only7/8. Both
ended with old4/4 and new2/2 candidate tasks and passed6/6 published readouts.
This is conventional known-template usefulness, not generalization, statistical
reliability, a circadian advantage or equal total computation.

[Executed commands, costs, failures and limitations](reports/paired-learning-boundary-20261029.md)
include the spent one-shot protocol. Do not rerun that case as a usage example.
The API's native behavior is reused from that case. The command's parser,
dispatch, preview, module entry and bounded encoder now have separate zero-model
[software qualification](reports/context-boundary-cli-20261029.md). No new native
CLI comparison was run; installed delivery remains unqualified, and the previously
installed wheel predates this increment. No new environment variables or dependencies.
