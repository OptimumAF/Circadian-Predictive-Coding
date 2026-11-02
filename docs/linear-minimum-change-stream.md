# Explicit single-head minimum-change TRAIN in the ordinary stream

The existing chronological ticket workflow accepts an explicit incremental
least-squares operation without separate context heads:

```text
python -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --train-update minimum_change --output NEW.json
```

Use a new output path and a prospectively budgeted local run. This command
performs training; viewing its saved report does not. Omitting `--train-update`
preserves the original cold current-packet fit. Existing separate-head options
keep their behavior. Paired serving and reserved probes still require their
explicit separate-head modes; this single-head option does not enable them.
Run from the updated checkout. The retained #349 development wheels predate
this single-head integration; a new installed-package check has not been run.

The operation calls the already supported `LinearFitLearner` minimum-change
update on the existing owned stream candidate and forks. An unfitted first
update uses the cold solve. Warm updates fit current-row residuals against prior
coefficients; no observation bank, retained-label replay or extra head is added.
Only coefficient components invisible to the current design are preserved.
That is **not a guarantee that old predictions or tasks are retained**.

Ordinary TRAIN arrival/consent/receipts, the original12-update/72-row planned
stream and30s/512MiB source envelope remain. Source and deployed actor stay
separate from the stream candidate. Final and probe labels remain unavailable.
The report records `linear_train_update="minimum_change"` and the single-head
minimum-change residual definition. Source/candidate/actor counters and original
clocks, attempted/returned/uncertain work and refused prior-evidence mutations
remain guarded. Failure cannot authorize another update or erase paid work.

[Software evidence](reports/linear-minimum-change-stream-20261030.md) comprises
tiny actual adapter updates plus scripted full captures using the actual stream
and guards. The [sole genuine ordinary observation](reports/linear-minimum-change-stream-native-20261030.md)
learns both contexts and recovers a lost A invoice, but loses B invoice on return;
final ordinary3/4 is incomplete joint retention. Its original observer coverage
gate failed for typed-affine/teacher hooks; guarded owner/source-derived costs
are separately qualified. The spent case cannot be rerun to fill those counters.
Earlier cold/shared-head and minimum-change/separate-head results remain
different development protocols. No matched-compute, generalization or CPC
benefit follows. A separate frozen protocol is required for any new experiment.
