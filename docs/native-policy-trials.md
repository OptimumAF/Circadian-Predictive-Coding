# Native reference trials on the policy-change stream

The [same-phase comparison](phase-matched-controllers.md) now runs five frozen
controllers through these deferred trials. It found no task gain beyond
no_learning and preserves the original single-case failures and utility limits.

The existing native runner has an explicit development mode:

```console
python -m scripts.run_bp_ticket_serving --source artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --policy-change-reference periodic --output policy-trial.json
```

Use a fresh output path and the existing local qualified BP source. Changed source
bytes refuse; no download, source refit, model-family conversion or promotion is
performed. Other fixed references are no_learning, drift and budget_aware. Their
new-policy outcomes have not yet been run in a matched comparison.

## Finite execution and isolated readouts

This uses the [verified policy stream](versioned-policy-stream.md):12six-row TRAIN
batches, delay1, three fixed phases, original7features/6classes. One original
runtime/full optimizer/source/actor, clock, budget, sharing gate and action envelope
cover13opportunities with at most12updates. The controller cannot renew limits.
Old eight-batch/reference/live/followup modes and their metrics remain unchanged.
This mode excludes live serving, publication, other recipes, counterfactual base
overrides and replay; permission/version/role/source safeguards remain external.

Four bounded phase records at opportunities0/4/8/12 retain the actual committed
update count and complete native state. An opportunity is not an update. No queue
is force-drained. After TRAIN closes, owned frozen copies execute development
sandbox tasks under each record's explicit ticket policy. All prediction/copy/task
work is counted; source, stable actor, private final state and caller RNG remain.
Readout outcomes never choose updates or promotion. Password retention, current
invoice correctness and obsolete still-permitted route errors remain separate.

Python trials can pass `policy_change=True, deferred_evaluation=True` with a fixed
reference or an original-bound contextual/fitted chooser. Deferred execution
returns phase states with **zero task readouts**. Stale source-state/version-bound
fitted policies still refuse. A later host owns its frozen development-scoring
chronology. Neither this function nor the new CLI opens final/probe data.

Legacy final scoring and fixed-group utility refuse policy reports before clock,
source or model access, including malformed/missing mode flags when other policy
markers remain. Reusing old v2 groups would silently change the meaning of new
policy outcomes. A new comparison must freeze its metrics/access/cost procedure
explicitly; old final results and weights are not rewritten.

Why this: the truthful corpus already exists, and the original native controller
trial ports already own learning. A finite opt-in seam connects them without a
new runner, source-loader, model schema, checkpoint framework or learning rule.

## First engineering observation

One fixed periodic2 Windows case passed first launch. It committed6updates36rows
from12arrived batches; six batches remain unapplied. Original source96/576rows
became private102/612rows; stable actor stayed96. No return-window batch was
applied. The four native readouts were:

| Opportunity | New updates | Ticket policy | Invoice | Parcel | Password |
|---|---|---|---|---|---|
| 0 | 0 | v1 | Fail | Fail | Pass |
| 4 | 2 | v1 | Fail | Fail | Pass |
| 8 | 4 | v2 | Pass | Pass | Pass |
| 12 | 6 | v2 | Pass | Pass | Pass |

Invoice v1 acquisition was not demonstrated at opportunity4. Password stays
correct. Later v2 successes are on rules the source already knew. The preserved
raw helper field `new_parcel_success_gain=1` compares different policies/masks;
it is a descriptive change, **not isolated learning gain**. A same-phase
no-learning control is needed. This run demonstrates no return learning,
recovery from forgetting, controller/CPC/generalization or efficiency advantage.
It completed15episodes/10successes, including two stale recoveries and desired
commit-revocation refusal, with0applied permission/revision violations in this
fixture. It is a private development readout, not a deployed serving claim.

Physical new work:6SGD/backward36rows,40head constructor entries1560constructor
matrix slots,67native head forwards,229snapshots32304logical tensor elements,
49typed predictions85sandbox dispatches.0PC/probe/final/replay/controllerfit.
Internal5.0712s/meter7.1846s, sampled peak337272832B. Instrumentation is partial;
these are not full FLOPs/allocations or comparative performance measurements.

101distinct scoped checks/platform, five-file Ruff/format/fresh default+Linux
mypy and saved-result review pass. One missing gate-report test failure and one
patch-context failure are preserved. Root self-review is not independent review.
Original native family math/baselines and #54negative controller results are reused.
No matched new-policy experiment, full-repository/clean-clone/release test or sweep
was run. Protocol/commands/failures/costs/source pins: issue68 and
`artifacts/runs/native-policy-trial-20261023/RESULTS.md`; ADR-0584.

```console
python -m pytest -q tests/test_native_policy_trial.py -k "not sole_native"
python -m ruff check scripts/run_bp_ticket_serving.py scripts/ticket_workflow_native.py scripts/ticket_controller_streams.py scripts/recurring_ticket_workflow_example.py tests/test_native_policy_trial.py
python -m ruff format --check scripts/run_bp_ticket_serving.py scripts/ticket_workflow_native.py scripts/ticket_controller_streams.py scripts/recurring_ticket_workflow_example.py tests/test_native_policy_trial.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent scripts/run_bp_ticket_serving.py scripts/ticket_workflow_native.py scripts/ticket_controller_streams.py scripts/recurring_ticket_workflow_example.py tests/test_native_policy_trial.py
```

Extension: use the existing comparison driver with deferred trials; finish and
validate every candidate before development readouts. Freeze same-phase controls,
metrics/directions and limits prospectively. Keep old final evaluators closed and
original full fitted-state binding. Never extend the horizon or select a favorable
checkpoint after inspecting task outcomes.
