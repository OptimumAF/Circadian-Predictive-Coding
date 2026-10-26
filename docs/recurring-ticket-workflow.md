# Chronological ticket learning, retention and return

## Native family option

The same recipe accepts `--method bp|pc|neutral_cpc|gated_cpc`, with the existing
CPU Torch environment. Default `perceptron` and its replay options are unchanged.

```powershell
python -B -m scripts.recurring_ticket_workflow_example --method bp --output fresh-native-tickets.json
```

This performs native TRAIN and task evaluation. Use the finite
[protocol](../artifacts/runs/native-recurring-tickets-20261023/PROTOCOL.md) for
recorded observation scope; do not launch a sweep or select a favorable fit.
Native replay combinations are explicitly refused for this fixed observation.
The existing stream and episode loop remain the only runners.

All native methods use existing seed17/D7/H3/K6 and default lr.03, inference2×.04,
BPmomentum.9. The original12wake36-row A→B→A teacher/delay/checkpoints and11task
episodes are shared. Frozen typed readouts copy complete native checkpoints;
source/seed/global RNG and original wake budget stay separate. JSON observations
include native optimizer/gradients/adaptive history/RNG tensor values and clocks;
these are scalar evidence, not an authority-bearing persistence/restore format.
Source mistakes are unavailable for native CE/PC energy and explicitly None.

Work records count physical native updates across forks and each frozen typed
attempt/return, including failed prefixes. Costs replace perceptron-only affine
and zero-randomness fields: full-class D×H+H×K nominal attempted terms, PC training
latent iterations/BP returned optimizer steps, full checkpoint tensor bytes and
sampled RSS. Initialization/random slots and snapshot copy work are externally
metered in the recorded observation; unknown backend RNG draw count stays None.
Copies/native validation/gradient work and other live owners add real overhead.
Equal updates/rows/geometry do not imply equal computation or native objectives.

One fixed WSL four-family observation completed48updates144rows. **All families
failed both ordinary tasks at0/4/8/12**, and stale cases failed; revoked-commit
events were never reached. All four change metrics are zero because no complete
task succeeded, not because retention was demonstrated. BP afterA executes open
then correct inspect but fails commit; later choices get stuck at wrong phases.
PC and neutral CPC parameters and full task traces match exactly. Existing
perceptron learns/relearns tasks in saved evidence but forgets; it remains the
working simpler option on this recipe, with different algorithm/capacity/work.
No CPC benefit, unseen-task generalization, reliable native adaptation, efficiency
ranking or release readiness follows. [All results/costs/failures](../artifacts/runs/native-recurring-tickets-20261023/RESULTS.md).

Implementation files: existing `scripts/recurring_ticket_workflow_example.py`,
small native support `scripts/ticket_workflow_native.py`, focused
`tests/test_native_recurring_tickets.py`. The helper owns native construction,
observations and costs, without data selection/scheduling or task effects.
Scoped checks:

```powershell
python -m pytest -q tests/test_native_recurring_tickets.py tests/test_recurring_ticket_workflow.py tests/test_ticket_workflow_rehearsal.py
python -m ruff check scripts/recurring_ticket_workflow_example.py scripts/ticket_workflow_native.py tests/test_native_recurring_tickets.py
python -m ruff format --check scripts/recurring_ticket_workflow_example.py scripts/ticket_workflow_native.py tests/test_native_recurring_tickets.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/recurring_ticket_workflow_example.py scripts/ticket_workflow_native.py tests/test_native_recurring_tickets.py
```

Next extend through TRAIN-only fit/action-margin diagnosis of the failing native
BP phase decisions before another adaptation comparison or new mechanism.

## Original conventional recipe

Run a fixed small conventional learner on invoice→parcel→invoice TRAIN phases,
with complete old/new ticket episodes between phases:

```powershell
python -B -m scripts.recurring_ticket_workflow_example --output fresh-recurring-tickets.json
```

Requires the existing NumPy environment, no weights/downloads/Torch/native model.
The output must be fresh. Existing frozen [ticket driver](local-ticket-workflow.md)
and all native drivers retain their defaults. The experiment has a frozen finite
protocol; do not run a sweep or extend its fit based on the result.

## What runs

Use the same public vocabulary, catalog, zero-initialized multiclass perceptron
and authentic v2 TRAIN demonstrations as the existing workflow. Four3-row passes
teach invoice, four teach parcel, four teach invoice again:12updates/36exposures,
no replay. Repeated templates overlap (open/commit steps are shared); this is not
unseen-task transfer. Labels arrive one logical tick after observations through
the existing `AttributionStream`/inbox, with one original12-update budget.

At fixed candidate checkpoints0/4/8/12, freeze a detached typed predictor and run
ordinary invoice and parcel episodes. Also run stale parcel at8, stale invoice
and revoked invoice at12:11episodes with at most6decisions each. All five active
tools remain candidates at every phase, and the existing dispatcher checks
current revision/capabilities at effect time. A wrong permitted workflow may
commit and remains a task failure; permission refusal differs from task success.

Why this: complete task behavior between training phases answers more than the
existing all-concepts-fit-before-readout driver. A new-task gain can coexist with
old-task degradation, and return training can damage the new task. Report these
separately and preserve all checkpoint outcomes, not just the best endpoint.

## Isolation and interpretation

The stream owns a shadow candidate fork. Checkpoint snapshots come from that
candidate; the original seed and stable actor stay separate. Readouts cannot
train, change the candidate or consume update budget. This is checkpoint task
evaluation, not actor promotion or a continuous concurrent deployment claim.
The fixed data/dose/sequence cannot use episode outcomes for tuning. Final/probe
data is absent, and existing isolated R2.1 probes are not replaced by this case.

The report retains full requests/probabilities/effects, delayed receipts, source
versions, paid failure prefixes and four bounded snapshots. Costs include
attempted/returned updates/rows, mistake counts, candidate comparisons, effects,
elapsed time and sampled RSS. Nominal attempted affine terms do not certify work
completed after a throwing call. Coefficient byte counts describe retained report
snapshots; Python/actor/stream/copy/softmax/validation overhead remains additional.
Neither these counts nor the small case establish total FLOPs, memory efficiency,
generalization, calibration, a CPC advantage or repository-wide readiness.

## Structure and verification

```text
scripts/recurring_ticket_workflow_example.py  fixed existing-component recipe/CLI
tests/test_recurring_ticket_workflow.py       chronology/isolation/failure/CLI
```

No new learning algorithm, configuration variable or dependency. Existing core,
app and adapter boundaries are unchanged. [ADR-0555](adr/ADR-0555-chronological-ticket-readouts.md).

```powershell
python -m pytest -q tests/test_recurring_ticket_workflow.py
python -m ruff check scripts/recurring_ticket_workflow_example.py tests/test_recurring_ticket_workflow.py
python -m ruff format --check scripts/recurring_ticket_workflow_example.py tests/test_recurring_ticket_workflow.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/recurring_ticket_workflow_example.py tests/test_recurring_ticket_workflow.py
```

Tests fit tiny NumPy learners; run within a declared engineering allowance.
Frozen protocol, commands, all failures and results are in
`artifacts/runs/recurring-ticket-workflow-20261022/` and
[issue #38](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/38).
Extend using existing stream/typed learner ports and freeze any comparative study
separately. Preserve all earlier models' negative evidence and resource costs.

## Optional bounded trajectory replay

```powershell
python -B -m scripts.recurring_ticket_workflow_example --replay recent --output fresh-recent.json
python -B -m scripts.recurring_ticket_workflow_example --replay coverage --output fresh-coverage.json
```

These explicit options reconstruct authentic three-transition synthetic TRAIN
episodes and use the existing `StreamRehearsal`. No new learning rule/controller.
The default remains `none`. Only completed TRAIN wakes enter memory. The compiler
checks every input/action against the original local teacher, links observations,
and uses an evidence-only terminal zero vector and final reward. Reward is not a
training target. Final/probe data and observed episode outcomes never enter replay.
This fixed example authorizes separate synthetic packet reuse; it does not renew
inbox permissions or grant consent for arbitrary/private managed data.

Both policies retain at most two complete episodes/six transitions,2416encoded
array bytes. At every second wake (midpoint/end of each four-pass phase), replay
one three-row episode: six additional native updates/eighteen rows. One original
12-update wake budget and a separate18-row replay budget remain distinct. Native
update counts include published replay:0/6/12/18 at wake checkpoints0/4/8/12.
`source_updates` denotes wakes; `state.completed_updates`/`source_rows` include
replay, and training_work reports all attempts, including unreturned work.

The additive public `AttributionStream.applied_consolidations` returns detached
quiescent receipt values. Native deltas require exact observed publication and
event doses; historical prefix changes/counter/budget mismatch stop work. Getter
copies are expected; this does not attest private object identity against hostile
Python code. Replay/source/work stay unchanged during frozen task readouts.

Recent FIFO and declared-context coverage have equal caps, replay exposure and
opportunities. Coverage has stronger released TRAIN context information and
different retention/order/occupancy, so the comparison concerns that policy
bundle. Saved no-replay is a descriptive **lower-work** reference, not matched
compute. Report retained encoded bytes separately from complete event records,
Python/copy/teacher generation and sampled process RSS;2416B is not total memory.

The one frozen comparison found: recent replay still forgets the other task.
Coverage preserves invoice during B but fails parcel at B's checkpoint and stale
case; after return, both ordinary tasks succeed. Both deny revoked commit. This
tradeoff is not an unconditional improvement, statistical evidence, CPC advantage
or unseen transfer. [Full results/costs/failures](../artifacts/runs/recurring-ticket-replay-20261023/RESULTS.md),
[ADR-0556](adr/ADR-0556-ticket-trajectory-replay.md),
[issue #39](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/39).

Additional structure: `scripts/ticket_workflow_rehearsal.py` compiles the fixed
teacher episodes and composes existing policies/progress checks;
`tests/test_ticket_workflow_rehearsal.py` tests real replay, codec, causal roles,
publication/source/budget guards, paid failures and CLI. Existing core codec,
retention and learner math are unchanged. Scope checks to these files and the
modified stream/example/default test; use a frozen allowance before model tests.

## Explicit policy-change option

[Versioned policy stream](versioned-policy-stream.md) adds --policy-change to this
same runner. It uses sandbox-verified v1/v2 labels and returns under current v2;
unchanged-task retention is separated from obsolete-rule use. The original
fixed-v2/default/replay recipes and their negative results remain preserved.
