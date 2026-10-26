# Bounded reuse of arrived TRAIN episodes

`--context-linear-catalogue` adds one earlier same-cue TRAIN episode to each
arriving packet, then fits old and current rows together. It uses the existing
catalogue composer, affine learner and delayed stream. The original replacement
mode remains unchanged.

Why this: fitting the current episode and then replaying an old episode in a
separate fit would replace the head again. A joint fit can retain both tasks;
its extra exposure, storage and computation must be counted.

## Run

From the repository root, with the accepted local source available:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_minigrid_linear_fit_control --context-linear-catalogue --describe
.\.venv\Scripts\python.exe -m scripts.run_minigrid_linear_fit_control --context-linear-catalogue --source artifacts/runs/joint-catalogue-study-20261018/bp.txt --output new-catalogue-result.json
```

Each invocation needs a prospective local budget. The mode refuses fit overrides
and existing output files. It runs a fixed A blue/red → B blue/red → A blue/red
TRAIN recipe, six fits, label delay one, checkpoints0/2/4/6. All fits finish
before TRAIN scores are computed. No games, final/probe/controller-validation
labels, promotion, random initialization or baseline retraining.

## Causality and ownership

This is a finite declared recipe. The application holds private planned buffers,
as the existing stream does; the learner receives only delayed admitted arrays.
It is not a live adaptive replay service or deployment/consent-policy claim.

The bank starts empty and holds one earlier episode per public initial cue.
That episode's labels must have arrived by the current observation tick. Task
names and targets cannot choose the cue. Each native fit sees9 or18 independent
rows, with `episode=None`; these are not fabricated eighteen-step trajectories.
Unique exposure IDs and original packet spans preserve repeated-row provenance.
Arrays are detached from sources and other wakes. Original roles, receipts,
clocks, budgets, source and actor remain authoritative.

## Actual costs and result

One frozen TRAIN case completed:

| State | A correct | B correct | Replacement baseline A/B |
| --- | ---: | ---: | --- |
| Untrained | 2/18 | 2/18 | 2/18, 2/18 |
| After A | 18/18 | 13/18 | 18/18, 13/18 |
| After B | 18/18 | 18/18 | 13/18, 18/18 |
| A returns | 18/18 | 18/18 | 18/18, 13/18 |

Both complete nine-action traces for each task classify correctly after B and
after A returns. The saved issue107baseline was reused without refitting or
rescoring; the after-A state is exactly equal to its saved state.

The new run used **90 fitted row exposures versus54**:54arriving exposures
(including18repeated A rows) plus36extra catalogue replay rows, from36distinct
source rows. There are still six analytic fits, not SGD steps. All90 count in
native `training_examples`. The explicit replay ledger is36; the old separate
consolidation-event replay counter stays0 and must not be read as zero replay.

Classification is not perfect one-hot interpolation: each later joint head has
squared error about0.5. Current-task CE worsens from about1.165422 under
replacement to1.189879 with reuse, despite retained class accuracy. This is an
exposure/confidence/compute tradeoff, not an overall or matched-compute win.

The logical bank peaks at18rows/85,392float64-equivalent bytes for current+next
observations and targets, sharing source schedule objects. This excludes
reward/metadata/Python overhead and is not total process storage. Materialized
joint inputs use213,840array bytes, peak temporary joint template42,768bytes.
Full source, stream copies, validations, snapshots and diagnostics count in RSS.

The body took12.111seconds, parent13.550seconds, sampled peak63,143,936bytes.
Six fits/90rows, six LAPACK calls,12fitted-core forwards/108rows plus4uniform
reads/36rows,16metric calls/144rows,840nested public projections. Source, actor,
source bytes, caller NumPy RNG, original budgets/clocks and receipts stayed intact.

## Verification and limits

25distinct focused checks qualify on Windows and Linux; scoped lint/format/type
checks and source/procedure/result self-review pass. This is not independent
acceptance or release qualification. One local-data check executed here and
skips explicitly when the source is absent elsewhere. Failed fixture attempts
and all paid work are retained.

The result demonstrates bounded TRAIN acquisition and retention through return.
The separate [retained after-B native follow-up](retained-context-linear-games.md)
now solves four known old/new games without refitting. Generalization,
long-term/reliable retention, CPC benefit and online deployment remain unproved.
Issue108's different failed/spent native case remains open; no reconstruction,
rescoring or retry was performed.

[Protocol](../artifacts/runs/arrived-cue-catalogue-20261024/PROTOCOL.md),
[results and commands](../artifacts/runs/arrived-cue-catalogue-20261024/RESULTS.md),
[issue109](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/109).
The next useful outcome after that follow-up is bounded C_train acquisition
with retained A/B and explicit replay/storage costs through the existing stream.
