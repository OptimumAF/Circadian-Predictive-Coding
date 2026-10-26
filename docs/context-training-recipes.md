# Contextual recurrence and TRAIN label noise

```powershell
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
