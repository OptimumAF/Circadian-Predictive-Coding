# Contextual recurrence and TRAIN label noise

```powershell
Set-Location D:\Dev\Repositories\Circadian
$env:TEMP = 'D:\Dev\_Temp'
$env:TMP = 'D:\Dev\_Temp'
$env:PYTHONPATH = (Get-Location).Path
.venv\Scripts\python.exe -m scripts.run_bp_ticket_serving --source artifacts/runs/context-head-control-20261023/actual.json --context-recipe-validation --output local-context-recipes.json
```

Output must not exist. This finite engineering validation runs the clean then
noisy contextual recipe once each. It does not fit or compare controllers.
For one case, use `--context-head-serving-live --context-stream-recipe
context_recurring_clean` (or `context_recurring_noisy`); fixed reference selection
remains available with `--context-live-reference`.

## Data and execution

`ContextTrainingRecipe` explicitly separates D9 contextual packets from the
existing D7 native recipes. It uses ordinary TRAIN prototypes: three v1legacy
rows and six v1invoice/password rows. The new sequence repeats new/old three
times, giving six packets and27rows with label delay2.

The adapter validates exact original TRAIN trace row IDs, public context and
workflow features, and clean target actions. The encoder omits the concept at
sealed/staged, so those rows use their original trace identity and validate that
masking. The ready row also validates its public concept. Synthetic corruption
selects `v1:invoice:ready` by name even after row reordering, changing route to
defer in each old packet. Each corruption records source and owned row IDs and
both actions. It is experimental noise, not a claim of teacher truth.

Output packets own their IDs and arrays. Clean prototypes and inner guards stay
unchanged; probe/final/guard/live outcomes cannot become TRAIN. Native shapes,
native noise offsets and all original recipes stay unchanged.

Recipe cases use24worker opportunities with six common task reads at17–22,
one declared six-update/60-second/512MiB budget per new owner, and original
publication/rollback ports. No existing budget is renewed. Original no-recipe
behavior remains14opportunities/two updates/45seconds. Nonperiodic policies may
leave TRAIN pending. New recipe candidate feedback has an explicit144-row cap;
the original diagnostic cap stays84. Fitted-controller source/version constraints
remain intact. All attempted and discarded work must be charged.

## Interpretation and extension

Both frozen engineering cases completed6updates/27rows. Clean publication was
accepted at tick14 (new6/6, old12/12); the noisy candidate was rejected for old
retention11/12. Source weights stayed unchanged, and clean rollback restored them.

The pre-run scenario description was incomplete: the existing worker explicitly
lists stale ticks4/12/20 and revoked ticks6/14/22. The24-step cases therefore include
three events omitted from that description. The common17–22 window includes a
deliberate denied v2legacy episode: clean passed5/6, noisy4/6. Do not count that
denial as a learning failure, replace it with an ordinary episode, or claim a
fully prospectively enumerated comparison. Results are retained without rerunning.

These recipes reuse known teacher templates. Different order/delay, synthetic
noise and disjoint IDs do not establish unseen-task or heldout generalization.
Guard rejection is a valid completed engineering outcome. Task success, classifier
fit, promotion and finite execution completion remain distinct.

A [bounded distinct-stream engineering comparison](context-stream-comparison.md)
now reuses these two cases and the frozen scalar fit, completing ten missing
applications without retraining them. It demonstrated no new controller task
benefit; full R7.5 unseen-task/rate/noise generalization remains open. Extend via
existing ports with separately frozen protocols; retain all negatives and costs.
The incomplete prospective scenario-declaration requirement stays open in issue84;
all24task/scenario pairs must be explicit before any new comparison. Existing
cases are engineering evidence with that limitation, not confirmation evidence.
Evidence: `artifacts/runs/context-training-recipes-20261023/` and
[issue84](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/84).

## Equal-row joint rehearsal and return

`CONTEXT_REHEARSAL_RECIPES` adds two explicit Python recipes to the same runner:

| Recipe | Learning packets | Common return | Total |
|---|---|---|---|
| `context_current9_return` | twice legacy3 repeated3times | invoice/password6 |3updates24rows|
| `context_joint9_return` | twice legacy3 then invoice/password6 | invoice/password6 |3updates24rows|

Both use delay1, the existing24-opportunity worker and original promotion/rollback
guards. No learner algorithm changes. `current9-v1` and `joint-v1` packets own
arrays and distinct exposure IDs; roles, semantic trace IDs, features and clean
labels are revalidated from the original TRAIN prototypes. At most8packets72rows.
Noise cannot be combined with these modes. Old named recipes/IDs/defaults remain.
Explicit old-row reuse is training exposure, not free data or probe conversion.

For a separately budgeted new local run, use the existing Python API from physical
`D:\Dev\Repositories\Circadian`, with shell TEMP/TMP set to `D:\Dev\_Temp`:

```python
from scripts.context_training_streams import CONTEXT_REHEARSAL_RECIPES
from scripts.context_head_live import run_case

result = run_case(
    "artifacts/runs/context-head-control-20261023/actual.json",
    stream_recipe=CONTEXT_REHEARSAL_RECIPES[1],
)
```

The ordinary CLI now accepts both names through `--context-stream-recipe`.
Preview a selected recipe before source or model access:

```powershell
.venv\Scripts\python.exe -B -m scripts.run_bp_ticket_serving --source unused --context-stream-protocol --context-stream-recipe context_joint9_return --output joint-preview.json
```

`--source` remains a required CLI argument but is not opened in preview mode.
The fresh output lists all24version/concept/scenario/contention rows, including
stale4/12/20 and revoke6/14/22, plus each planned TRAIN packet's label arrival,
rows and repeated old/new exposure. It describes the default periodic reference
and original finite budget; it does not validate a source, run models or grant
permission/budget/runtime authority. Selected-recipe preview cannot be combined
with withheld comparison or live reference/feedback options. Existing unselected
comparison/withheld previews retain their old behavior.

For separately budgeted new local work, the existing live command dispatches
the exact selected recipe to `run_case`:

```powershell
.venv\Scripts\python.exe -B -m scripts.run_bp_ticket_serving --source artifacts/runs/context-head-control-20261023/actual.json --context-head-serving-live --context-stream-recipe context_joint9_return --output joint-live.json
```

Both outputs must be new files. Existing reference/feedback options keep their
original requirements; nonperiodic choices can leave training pending. A failed
case keeps its known work and returns nonzero; a preview is not a successful
learning result. Frozen scientific wrappers and spent comparisons must not be
rerun as usage examples. [CLI checks and no-work execution](reports/context-rehearsal-cli-20261028.md)
qualify this command path, not another learning experiment or repair of264.

One completed bounded observation found joint new2/2 and old4/4 after learning
and return. Equal-row current-only learned new2/2 but lost v1password (old3/4),
then recovered it on the shared old return (new2/2,old4/4). Known TRAIN task
endpoints, added readouts/costs and qualifications are recorded in
[the result report](reports/context-joint-rehearsal-20261028.md). These facts
do not establish heldout/reliable/CPC benefit or equal total computation. The
worker scenario declaration and post-body RSS qualifications remain explicit;
full prospective comparative qualification is unfinished in issue264.
