# Isolated warm and fresh ticket-policy probe

```powershell
python -B -m scripts.probe_ticket_policy_learning --input artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --output fresh-policy-probe.json
```

This command executes native TRAIN/probe work. Use the existing CPU Torch/NumPy
environment and the [fixed local protocol](../artifacts/runs/policy-learning-probe-20261023/PROTOCOL.md);
it is not a zero-compute report viewer. Output must be fresh; the source must
match its original SHA and bounded full BP state. No install, model download,
source re-fit, controller, final data or external action is required.

## What runs

The existing `run_fresh_task_probe` owns each disposable learner. Warm starts at
the saved qualified96updates/576rows; fresh starts at its original saved0 state,
same D7/H3/K6, seed17, learning rate.03 and momentum.9. Source history is separate
from the new budget; equal added updates do not mean equal total computation.
Full optimizer/gradient state is preserved, so this is not a weights-only reset.

Sixteen `probe_train` updates alternate three-row v1invoice/password templates.
Two detached `probe_test` copies measure all fixed0/4/8/16 checkpoints. They are
TRAIN-derived equal-feature diagnostic templates with distinct IDs/storage,
**not held-out generalization data**. Separate old TRAIN invoice/parcel readouts
perform no updates. The same fork/optimizer then receives16alternating old-v2
TRAIN return updates under its distinct original session. No replay occurs.

At both existing endpoint-observer seams, a frozen native copy executes four
ordinary sandbox tasks: new v1invoice/password and old v2invoice/parcel. All
episodes have a six-step bound, typed allowed-action normalization, actual effect
validation and actual workflow-success truth. Native/global6class row metrics
and permitted-mask task outcomes are reported separately. Labels, seeds, dose,
masks, checkpoints and order are fixed; outcomes never select more updates.

The recipe checks source bytes/full states, original source owners, input arrays,
global NumPy/Torch RNG and previous paid results around native calls. Endpoint
copies must leave the disposable learner unchanged, including after clock reads.
Incomplete probe/return/task work stops subsequent work and retains paid prefixes;
corrupted preceding evidence is refused and its saved original result restored.
This synchronous recipe is not hostile-Python containment or durable authority.

## Observed result

The single two-arm case completed64updates/192row exposures,16episodes and72typed
decisions. All original owners/source/RNG were unchanged. Initial warm old TRAIN
accuracy was6/6; original source task ability is reused, not retrained here.

| Arm / endpoint | New v1 invoice | New password | Old v2 invoice | Old parcel |
|---|---|---|---|---|
|Warm after16probe updates|fail|pass|pass|pass|
|Warm after16return updates|fail|pass|pass|pass|
|Fresh after16probe updates|fail|fail|fail|fail|
|Fresh after16return updates|fail|fail|fail|fail|

Warm new-invoice CE1.44306→1.00762 during probe, accuracy2/3 throughout. Password
was already3/3 and remains so. Both old TRAIN contexts remain3/3. Fresh probe
new-task accuracy0→1/3; return improves it to2/3 but still fails whole tasks.
Every intermediate curve, confusion report and task trace is retained. No new
invoice acquisition, loss of old task ability, or recovery from such a loss was
demonstrated. This case does not establish plasticity loss or CPC/controller benefit.

### Conflicting policy truth is a representation limit

Both versions encode ready-invoice identically under original D7 features, while
requiring route in v1 and inspect in v2. Both actions are permitted in both
versions. The head cannot simultaneously choose both on the same input. Old
invoice is therefore a previous-policy counterfactual; unchanged parcel retention
and password persistence must remain separate. This qualification was recorded
**before execution** in [PROTOCOL_NOTE.md](../artifacts/runs/policy-learning-probe-20261023/PROTOCOL_NOTE.md).
It does not prove why the fixed probe failed to switch rules.

The existing D9 context-observation path, joint TRAIN conflict checker and
[context-head probe](context-head-probe.md) now provide that separate functionality.
That older head probe uses a different legacy cohort; it is not a matched D7/D9
comparison on this invoice/password cohort. Frozen D7 outcomes remain unchanged.

`run_arm(..., input_diagnostics=True)` optionally forwards the generic
[TRAIN input constraints](categorical-fresh-task-probes.md#optional-train-input-constraints)
report. Its default remainsFalse. The current explicit live catalog probe enables
it after the original probe/return/endpoints; unsupported or failed diagnostics
do not gate arm admission. It compares the48 supplied F records and6 old TRAIN
readout records, excluding duplicate return templates and probe-test labels.
Equal record weighting does not represent the old phase's actual update exposure.
The report-only same-task D7/D9 data qualification is recorded separately in
[R2.1-ID.C](reports/probe-input-diagnostics-20261027.md); it does not fit a model,
alter this recipe's encoding, explain a particular failure or establish task benefit.

## Costs, structure and checks

```text
scripts/probe_ticket_policy_learning.py    Fixed composition, guards and CLI
tests/test_ticket_policy_probe.py          Scripted probe/sandbox integration
src/app/fresh_task_probe.py                Existing owned probe/return machinery
```

Actual64SGD/backward calls/192rows,164native head initializations/6396matrix slots,
200forwards,2362snapshot constructions/281856logical tensor elements,2teacher/
2packet builds and96pure sandbox dispatches. Input arrays4224bytes; source prior
96updates/576rows reported separately. Native copy initializations are additional
work, not useful learning. Meter26.7171s, sampled peak333,926,400bytes; these costs
do not establish efficiency. Full commands, failed statics, software work, source
pins and budget receipts: [RESULTS.md](../artifacts/runs/policy-learning-probe-20261023/RESULTS.md).

```powershell
python -m pytest -q tests/test_ticket_policy_probe.py
python -m ruff check scripts/probe_ticket_policy_learning.py tests/test_ticket_policy_probe.py
python -m ruff format --check scripts/probe_ticket_policy_learning.py tests/test_ticket_policy_probe.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent scripts/probe_ticket_policy_learning.py tests/test_ticket_policy_probe.py
```

13checks/platform and scoped default+Linux type checks pass. Tests run the real
generic probe and sandbox with scripted learners; native work occurs only in the
separate actual case. Source/procedure/results are self-reviewed, not independently
accepted or repository-wide readiness. Extend through the existing probe seam;
freeze new data/roles/budgets before any later experiment and preserve all negatives.
