# Capacity mechanisms through the existing stream

The existing rehearsal CLI exposes the complete frozen R2.6 recipe:

```powershell
python -m scripts.run_stream_rehearsal_pilot --describe --scenario capacity-comparison
python -m scripts.run_stream_rehearsal_pilot --run --scenario capacity-comparison
```

Declare a fresh local envelope before another experiment. This recipe has no
tuning/seed-selection flags: one seed, 23 small cohorts, 414 native wakes, 3,312
training row exposures and 184 final score forwards. CPU/BLAS1, GPU0, 60 seconds,
256 MiB process RSS, 2 MiB report and no downloads/replay/fresh-task probes.
Existing scenarios and the original learner objectives remain available.

## Cohorts and comparisons

| Representation | Cohort | Width and operation | Methods |
|---|---|---|---|
| Raw structural | Fixed pruned width | 3 from a data-blind prune of the untrained split source | BP, PC, neutral CPC, active CPC |
| Raw structural | Fixed wide | 4 from the untrained width 2 isometric split | All four |
| Raw structural | Scheduled growth | 2→4 after wake 6 | All four |
| Raw structural | Split/prune | 2→4 after wake 6, attempt guarded 4→3 after wake 12 | All four |
| Ordinary effective | Fixed control | 4, native uniform update rate | All four |
| Ordinary effective | Bounded renewal | 4, original mature/low-gradient policy every 6 wakes, maximum 3 permanent units | PC, neutral CPC, active CPC |

Raw methods retain mean-field gamma1.5 and the native scaling/precision/update
factors. Ordinary methods own detached effective parameters and precision from
the same untrained wide source. Raw and ordinary updates have different factor
semantics; comparisons are kept within each representation. No BP renewal is
invented. Both raw fixed references are always shown, regardless of gate outcome.
The width 3 reference has a different initial function: removing a unit is not
isometric. It is constructed without data/scores, and initial/absolute levels
remain visible beside gains. Equal final stored width does not imply equal
function, objective, work or effective capacity.

The stream is the existing recurring A6/B6/A6 recipe, training RNG7301 and final
RNG9101, eight rows per wake and 64 final rows per context, delay two, checkpoints
0/6/12/18. Methods share initialization within each mode. Learning rate .01,
PC inference two steps at .05, seed23 and all native baseline mathematics remain
fixed. Final data is unavailable to training, pruning, selection and tuning.

## Guarded pruning and evidence

`StreamCapacityPrune` composes the existing structural prune consolidator and
observed-damage gate. One coordinate is removed per hidden layer: H1 removal
`(2,)` means width4→3. The complete original root/transition history survives.
Two reserved validation batches (RNG307, four rows per A/B context, available
at tick zero) are independent of training/final data. Limits for prediction
change, BCE increase and inferred-energy increase are each .05, fixed before
outcomes. Gate PC inference/energy remains a diagnostic even on the BP arm.

Accepted gate evidence is distinct from publication. Only the original runtime
receipt proves candidate publication. An exact rejected decision, no publication
and unchanged full candidate allow training to continue with the gate spent.
Failed/unknown evidence stops. A late publication/budget failure preserves its
receipt and original error; available usage is read without new admission and
secondary measurement failure is separate. No budget, clock, threshold or
consent is reset. The optional `retain_decision=False` seam defaults to no held
decision; opting in retains one owned validated decision and clears it before
the next call. It never repeats a gate to reconstruct diagnostics.

## Costs and outcomes

The capacity helper weights actual wake calls by the model used before each
wake. Initial/post-transform measurements affect peaks/future wakes only.
Reports separate parameters, parameter bytes, chemistry/history bytes, actual
wakes/rows, measurement time, initialization work/time, pruning diagnostics/time
and charged/attempted/returned/committed renewal units/time. All values and failed
or unreached rows are retained. Complete probe pairs establish known validation
work; failed partial work is not fabricated as zero.

There are 15 initial and eight scheduled splits, four initial pure prunes and
four guarded prune attempts. Full gate work is 16 score forwards/64 row forwards
and 48 fixed-step energy evaluations, separate from training and final scoring.
Reserved validation arrays total 192 float64 bytes; private copies, serialized
binding metadata, temporary activations/gradients and source/actor/scorer payloads
are excluded from committed candidate byte counts. Full wall time and one joined
RSS sampler include these other allocations/work without attributing causal cost
or claiming matched FLOPs.

Primary BCE clips at 1e-12 (lower is better); accuracy uses inclusive .5 (higher).
New-task gain is B0−B12, retention loss A12−A6, return gain A12−A18. Every context
score, endpoint metric and declared paired difference is kept. Negative gains,
rejected pruning and zero-eligibility renewal are valid results. No significance,
equivalence, circadian advantage or release-readiness claim follows from a single
fixed synthetic case.

## Modules and checks

Executed evidence: [all outcomes and curves](reports/full-capacity-comparison-20261016.md),
[commands and preserved failures](../artifacts/runs/r26-full-capacity-20261016/RESULTS.md),
and [independent acceptance](../artifacts/runs/r26-full-capacity-20261016/REVIEW.md).

```text
src/app/stream_capacity_prune.py           bounded prune policy/evidence driver
src/app/stream_capacity.py                 wake-use measurements and transforms
src/adapters/capacity_comparison_pilot.py  fixed cohort composition and reports
tests/test_stream_capacity_prune.py        state/ownership/failure behavior
tests/test_capacity_comparison_pilot.py    recipe/CLI/isolation integration
```

Run focused tests with `python -m pytest -q tests/test_stream_capacity_prune.py
tests/test_capacity_comparison_pilot.py`. Affected Ruff, format and Windows/Linux
mypy commands and exact bounded qualification receipts are recorded in
`artifacts/runs/r26-full-capacity-20261016/`. No new dependency or environment
variable is introduced. Use existing native ports for additional cohorts and
freeze/review any new protocol before comparative execution.
