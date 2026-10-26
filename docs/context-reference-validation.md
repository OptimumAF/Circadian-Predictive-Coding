# Fixed references under role-separated stress validation

The existing live runner now has a small outer composition that binds the full
execution configuration before reading data. It reuses accepted development
results as known controller-training metadata and admits a new validation stream
through `ControllerChronology`. Validation cannot enter its training reader.
No fitting callback, sealed source or final callback is supplied.

Why this: `ReferencePolicy` alone does not describe candidate feedback and
correct-example acknowledgement. The outer binding records those flags together
with the policy and fixed recipe, without changing the core chronology or runner.
Execution settings are fixed before validation; the core freeze receipt is issued
afterward and remains unused. These are different events.

## Executed result

One fixed stream used A2/B3/A3, eight packets/39 available TRAIN rows, label delay
one and five named old-invoice route→defer corruptions. It shares known semantic
templates with development. It tests this joint stress condition, not unseen task
semantics or the separate causal effect of each factor.

| Fixed control | Live success | Updates / TRAIN rows | Correct acknowledgements | Publication attempt | Utility |
|---|---:|---:|---:|---|---:|
| No learning |15/24|0 / 0|0|None|−0.2497|
| Periodic |15/24|8 / 39|0|Rejected at18|−0.2547|
| Candidate feedback + acknowledgement |15/24|3 / 15|5|Rejected at8|−0.2817|

Every control retained4/4 old successes and obtained0/2 new successes in the
original common window. Both trained candidates scored5/6 on the clean new guard,
unchanged from the source, and11/12 on the old guard versus12/12 for the source.
The original new-utility, gain and retention criteria rejected both. There was no
actor publication or rollback. Final actor state and qualified source remained
exact. All predeclared denied commits remain failures; applied permission/revision
violations were zero in the observed episodes.

Correct acknowledgement matches an arrived target; it does not detect false
labels. All eight candidate reads cost39 rows. The two candidates required three
mistake corrections each. The smaller update count did not yield useful deployed
adaptation here. No learning had the highest original utility in this case.
Serial time and process RSS remain unpaired cost proxies, not efficiency proof.

## Isolation, evidence and use

`scripts.context_reference_validation.run_study(source, development)` accepts the
qualified source and exact accepted102 development JSON. It declares all three
bindings and role groups before access, completes known training metadata first,
then executes validation through existing ports. Original inner clocks and
lifetime budgets remain in each report; outer0/1/2 describe aggregate trial
events. The controller stays fixed while inner learners train on arrived TRAIN.
Paid failed prefixes stop subsequent work and suppress aggregate scores.

The spent local cohort is under
`artifacts/runs/context-reference-validation-20261024/`. Inspect saved results
without running models:

```powershell
python -B artifacts/runs/context-reference-validation-20261024/readback.py
```

PROTOCOL.md, RESULTS.md, native/test/parent receipts and the original engineering
storage-limit failure are retained. The one-shot launcher refuses a second launch.
29 distinct focused cases passed per platform after four Windows fixture repairs;
scoped Ruff/format/mypy and source/procedure/result self-review passed. This was
self-review, not independent acceptance. No final, probe, replay, fitting, PC/CPC
or simulator work ran. No general robustness, generalization, scientific advantage
or repository/release readiness follows. [Issue103](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/103)
contains acceptance and current handoff; broader parents remain open.
