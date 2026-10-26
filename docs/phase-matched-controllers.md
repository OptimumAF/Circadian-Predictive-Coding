# Controllers compared with same-phase no-learning controls

Use the existing native CLI and fresh output path:

```console
python -m scripts.run_bp_ticket_serving --source artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --fitted-report artifacts/runs/learned-contextual-controller-20261023/actual.json --policy-change-controller-comparison --output policy-comparison.json
```

Both sources must already exist locally. Their original hashes are checked; there
is no download, source warmup/refit or new chooser training. The original
--controller-comparison and final evaluation stay unchanged. This new mode opens
development tasks only, after all five candidates finish and pass validation.

## Frozen procedure

Order is no_learning, periodic, drift, budget_aware, fitted. Each receives the
same [verified policy corpus](versioned-policy-stream.md):12six-row TRAIN batches,
delay1,13opportunities, original12-update/120s/512MiB budget. Existing references
and exact fitted chooser retain their parameters. Candidates use original full
qualified BP96/576row state/optimizer. There is no publication, replay, probe or
final-data access. New outcomes do not tune/select policies or extend the horizon.

Preceding complete native work/source/actor/RNG/envelope/phase evidence must remain
valid before another trial. Lost work, changed source/fit/protocol or replaced prior
results stop further work. All five deferred trials freeze before any development
readout. Existing owned native readouts then execute all15 sandbox episodes/case
at opportunities0/4/8/12, with explicit v1/v1/v2/v2 ticket policies. Partial or
duplicate/wrong-role/wrong-policy/unknown outcomes have no score. Reported success
must agree with actual final sandbox state; permission/revision effects stay external.

### Same-phase scoring

Changing a ticket rule can make an unchanged model appear to improve. To avoid
that confound, compare invoice+parcel at fixed phase ends4/8/12 to **no_learning
at those same phases/policies**: six current-task outcomes. Compare unchanged
password retention over those three phase ends: three outcomes. Retain every
opportunity0/4/8/12 result; no favorable checkpoint is selected.

The original unit-weight objective is reused: current-task gain minus clamped
password degradation minus wall/120s, sampled RSS/512MiB and applied violations.
Higher utility is the declared direction. Own native-trial callback plus readout
callback time and maximum own sampled RSS are proxies; shared driver overhead is
separate. This grouping differs from the old study and does not rewrite its metrics.
Historical fit16updates96rows are carried; exact fit overhead is unavailable and
all-in utility remains null. Same access/opportunities do not mean matched compute.

Why this: a fixed source already knows v2. Cross-policy success changes cannot
establish new learning. The existing comparison driver, deferred trial and readout
ports provide the needed control without a new runner or learning algorithm.

## Observed result

One prospectively bounded Windows cohort passed first launch, with22SGD132rows:

| Controller | Updates/rows | Current tasks /6 | Password /3 | Raw utility |
|---|---|---|---|---|
| no_learning | 0/0 | 4 | 3 | -0.665374 |
| periodic | 6/36 | 4 | 3 | -0.649234 |
| drift | 4/24 | 4 | 3 | -0.649586 |
| budget_aware | 8/48 | 4 | 3 | -0.650405 |
| fitted | 4/24 | 4 | 3 | -0.647739 |

All task outcomes are identical. V1 invoice still fails at opportunity4, including
after four updates for drift/fitted; password remains correct. Later v2 tasks were
already solved by the original source. Fitted and drift have identical actions
and full native phase states. Periodic full states exactly match saved issue68.

Every raw utility is negative. Fitted is numerically highest in this observation,
but the difference is wholly in wall/RSS proxies, with zero task gain or retention
change. The first no_learning callback took2.325s; later callbacks took.290–.327s.
Fixed order, initialization/cache effects and sampled process memory prevent an
efficiency ranking. The values are preserved; there was no timing retry or new
warmup. This study supports no task/controller/CPC/generalization/efficiency or
statistical-equivalence advantage. No-learning avoids new SGD for the same
observed task outcomes; that is not a proof of lowest total deployment cost.

75development episodes complete/50successes, including stale recoveries and desired
revocation refusals;0applied permission/revision violations in this fixture. All
failure traces and four-to-twelve unapplied batches per case remain recorded.

Physical work:184head constructor entries7176constructor matrix slots,327head
forwards,1129snapshots159216logical tensor elements,245typed predictions and
425sandbox dispatches.0PC/probe/final/replay/newfit. Internal8.8146s, sampled peak
345194496B; partial instrumentation is not a complete FLOP/allocation measure.

27focused checks/platform and three-file Ruff/format/fresh default+Linux mypy pass.
Source/procedure/result review is one-agent self-review, not independent acceptance.
No failed checks/native attempt or budget revision; exact commands, source pins,
costs, skips and honest interpretation are in issue69 and
`artifacts/runs/phase-matched-controllers-20261023/RESULTS.md`; ADR-0585.

```console
python -m pytest -q tests/test_phase_matched_controllers.py -k "not sole_frozen"
python -m ruff check scripts/ticket_controller_comparison.py scripts/run_bp_ticket_serving.py tests/test_phase_matched_controllers.py
python -m ruff format --check scripts/ticket_controller_comparison.py scripts/run_bp_ticket_serving.py tests/test_phase_matched_controllers.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent scripts/ticket_controller_comparison.py scripts/run_bp_ticket_serving.py tests/test_phase_matched_controllers.py
```

Next, diagnose v1 TRAIN fit/margins from saved states with the existing readonly
ticket diagnostic before adding algorithms or tuning dose. Reuse the existing
isolated R2.1 probe path if the evidence requires a fresh-task learning comparison.
