# Learned joint-context ticket control

```powershell
python -B -m scripts.run_ticket_workflow --joint-policy-context --output fresh-joint-control.json
```

The existing local example now has an explicit joint-policy mode. It learns the
same fixed public-context TRAIN tasks as the contextual BP qualification:
v1invoice/password and v2invoice/parcel. It uses the original perceptron rule and
eight passes, not a lookup table, new optimizer, replay mechanism or longer BP fit.
The default D7 example and single-v2 `--policy-context` mode remain available;
the two context flags are mutually exclusive.

## What it does

Shared `build_joint_context_batch` in the existing teacher module supplies the
same12rows, D9 features, class indices, targets and queries used by BP. It checks
exact TRAIN/context groups, schema, public-query alignment and no contradictory
targets before fitting. The native qualifier delegates to this data helper;
there is no circular script import or duplicated task selection.

`AttributionStream` performs eight12row passes,96row exposures, with delay1 and
zero-initialized60float64 coefficients. It preserves the original seed/actor and
separate candidate. All TRAIN finishes before fixed snapshot0/8 readouts; no
metric selects updates or a model. Each state has12separately charged typed TRAIN
reads for permitted accuracy/margins, then all four tasks under ordinary,
stale_once and revoke_commit scenarios. Task success comes from actual sandbox
effects, distinct from row accuracy.

Source snapshots/data, frozen predictor, original budget origin/clock/owners,
limits/update count, global NumPy RNG and paid task prefixes are checked. Altered
prior task evidence is restored and refused; missing teacher/task returns retain
attempted and unknown work and stop later reads. This is a synchronous local
recipe with trusted ports, not hostile-Python or process containment. There is
no final/probe data, model promotion, external tool effect or random draw.

## Observed fixed result

| State | TRAIN permitted fit | Ordinary tasks | Stale tasks | Revoked commits |
|---|---:|---:|---:|---:|
|Unchanged0|4/12 (ties)|0/4|0/4|0denials reached|
|Learned8|12/12|4/4|4/4|4correct denials|

All four learned tasks finish in3steps; stale cases recover in5steps. Ready-action
target margins are1/4/3/3 for v1invoice/password and v2invoice/parcel. Revocations
are correct refusals, not completed workflows.19mistake corrections occur across
the fixed96TRAIN row exposures. All24episode records and all eight delayed
receipts/snapshots/teacher rows remain in the report.

This supplies a working conventional learned control for these known templates.
Both contexts were available throughout TRAIN: it does not establish chronological
acquisition, retention of unseen contexts, recovery of forgotten skills, or
generalization. Four stale recoveries are recovery from ticket invalidation,
not a forgetting result.

The preserved BP case fits10/12 and solves2/4ordinary/stale tasks. The task/data
set is the same, but perceptron has60float64 coefficients versus BP54float32
parameters, another model/initialization/update rule and eight passes versus96SGD.
Runtime environments/startup/validation work also differ. This is not an
equal-capacity/compute comparison, universal efficiency finding or CPC advantage.
The simpler control is the supported choice for the next bounded adaptation demo.

## Costs and checks

Actual8passes96rows19mistakes;3learner/zero initializations,174snapshot copy/
validation entries10440logical coefficient slots,140typed reads (24teacher+
116task),2teacher/1packet builds164encodes140sandbox dispatches. Nominal TRAIN
affine work5760terms, task reads5800, teacher reads1200, coefficient additions380.
Retained coefficient payload960bytes for two snapshots excludes Python tuple/
object/report overhead. No Torch/SGD/PC/probe/final/replay work occurred.

Recipe1.2676s/meter1.8425s; sampled RSS44,802,048bytes/boundary44,535,808bytes.
Memory is sampled at existing boundaries; it is not a hard OS peak guarantee.
[Protocol, commands, failures, all costs and artifacts](../artifacts/runs/context-perceptron-control-20261023/RESULTS.md).

```text
scripts/run_ticket_workflow.py              Shared teacher and existing local modes
scripts/qualify_ticket_bp_fit.py            Existing native mode delegates data prep
tests/test_joint_context_perceptron.py       Small learning/failure/source checks
```

```powershell
python -m pytest -q tests/test_joint_context_perceptron.py tests/test_ticket_workflow.py
python -m ruff check scripts/run_ticket_workflow.py scripts/qualify_ticket_bp_fit.py tests/test_context_bp_qualification.py tests/test_joint_context_perceptron.py
python -m ruff format --check scripts/run_ticket_workflow.py scripts/qualify_ticket_bp_fit.py tests/test_context_bp_qualification.py tests/test_joint_context_perceptron.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent scripts/run_ticket_workflow.py scripts/qualify_ticket_bp_fit.py tests/test_context_bp_qualification.py tests/test_joint_context_perceptron.py
```

83distinct checks qualified on Windows (81+3affected) and83pass on WSL, with
scoped default+Linux type checks. Tests execute separately budgeted small real
perceptron fixtures; native BP is not rerun. Source/procedure/results are
self-reviewed, not independent acceptance or repository/release readiness.

Next use the existing recurring stream with a fresh zero source and an explicit
D9 true A→B→A recipe, frozen before numerical work. Reuse this joint fit as an
ability control, never as a secretly pretrained source for acquisition. Keep
same-policy task readouts at every phase, delayed labels, fixed no-learning
comparison, costs and all failures; do not extend BP dose or tune CPC.
