# Reserved-task CLM checkpoint probes

## Workflow32 source and original TRAIN readout

The existing adapter now exposes `run_workflow_reserved_probe`. It accepts a
spent32-update workflow catalog source and the original canonical twelve-row
TRAIN tuple, with its five candidates. Its `old_task` packet must borrow those
exact tensors with role `train`; this is a known-row diagnostic. The other two
groups keep the fixed reserved markers and suffix templates, three candidates
and their original `probe_train`/`probe_test` roles. Marker action features must
match the corresponding original workflow action rows.

Supply the original independent `ToyBudgetSession` with
`ToyExecutionBudget(max_training_updates=6)` and training-only consent. The
adapter wraps the existing runner/native probe with full source/caller Torch RNG and
typed work protection. The source stays at32updates; only the separate probe
learns. Six delayed updates consume12marker query/18candidate rows/36logits.
Before/after group shapes are2x3,2x3,12x5: six reads,32query/22candidate rows.
There is no TRAIN-to-probe conversion, replay, installation or final-data access.

```python
from src.adapters.clm_checkpoint_probes import run_workflow_reserved_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.experience import ExperiencePermissions

# source: successfully completed original32-update workflow owner.
# training: fixed reserved probe_train batch/targets, three action candidates.
# readouts: probe_train/probe_test/old_task, the last aliases original_train.
result = run_workflow_reserved_probe(
    source, training, readouts, original_train, {},
    permissions=ExperiencePermissions(training=True),
    budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=6), lambda: 0.0),
    parameter_scope="state_only",  # Omit to keep the all-parameter default.
)
```

The explicit `state_only` mode reuses the existing catalog probe: only its state
head enters the optimizer; its action head and logit scale retain their original
values and no gradients. The report adds `parameter_update_scope` with optimized,
fixed and total parameter-element counts and fixed-value preservation. This
protects the disposable probe's candidate geometry as well as the unchanged
source. Changed scope evidence or fixed parameters stop further work, retaining
the paid prefix. It does not establish that restricted updates improve learning.

The app-level opt-in is `workflow_training_readout=original_train` on
`run_reserved_probe(..., phase="after_return", catalog_training=True)`; the
phase names the current policy, while `source_update=32` and explicit workflow
metadata distinguish it from the original temporal checkpoint. Use the guarded
outer adapter for native sources. Rehearsal/marker-control/extra arrived-TRAIN
combinations are refused. Original temporal defaults and CLI stay
unchanged. The old whole-temporal scorer is not a workflow32 scorer; the new
entry returns raw facts, with no selection or scoring during learning.

[37 final checks per platform and all failed/repeated costs](reports/clm-workflow-probe-20261025.md)
qualify the software. The separately frozen [native source32→probe6 case](reports/clm-workflow-fresh-native-20261025.md)
completes all38updates: F TRAIN0/2→1/2, suffix0/2→1/2, old TRAIN12/12→10/12
inside the disposable probe. Legacy/parcel ready choices lose defer→route;
sealed/staged stay correct. The source stays unchanged, with no return-A update
or deployed adaptation. Original native and saved-review allowances are spent;
do not rerun them. Any new pretrained question requires its own prospective
protocol and finite budget. No new dependencies or environment variables.

[The workflow state-only integration](reports/clm-workflow-probe-state-only-20261025.md)
passes59 final focused cases per Windows/Linux, including fixed action/scale,
full source/caller preservation, wrong scope/consent refusals, paid failures and
exact preceding all-parameter reports. It has software qualification only; no
native state-only workflow acquisition/retention observation has run. Existing
underlying state-only mathematics and temporal results remain separately scoped.

The [prospective native state-only control](reports/clm-workflow-state-only-native-20261025.md)
passed three composition checks/platform but stopped at its frozen RAM preflight
before loading a model. Its attempt is spent; native acquisition, old-task damage
and comparison with the all-parameter result remain unobserved. No gate was
lowered or native retry run.

## Optional full-catalog objective

Combine `--catalog-training --reserved-task-probes` to use the explicit catalog
objective for both the benchmark learner and each disposable probe. Default
paired behavior stays available. The same markers, labels, six-update dose,
checkpoints and metrics are retained; no new data or runner is introduced.

Catalog training is phase-bound: baseline/A uses IDs inspect/retire/route and
target indices0/2; B/return uses defer/inspect/route and indices1/2. Every probe
update scores both queries against all three active candidates. Four independent
phase records carry explicit probe_train roles; none is converted into TRAIN.
The app validates canonical IDs/indices/features, actual objective-tagged receipts,
consent/clocks/budgets and all prior paid evidence before further work. The outer
hook preserves the complete benchmark state/work and four child record owners.

Main work is12updates36query rows36candidate rows108logits; probes add24updates
48query rows72candidate rows144logits. Query-target `training_pairs` retains its
legacy count; candidate/logit costs are separate permanent fields. All training
finishes before any cached group scoring. Cached admission requires the complete
main/child objective, indexed labels, exact ledger and actual causal receipts.
Missing objective tags with catalog ledger markers are refused before decoding.

```powershell
python -m scripts.run_clm_temporal_sequence --model-dir LOCAL_ENCODER --head LOCAL_HEAD --output NEW_REPORT.json --catalog-training --reserved-task-probes
python -m pytest -q tests/test_clm_catalog_reserved_probe.py
python -m ruff check src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py tests/test_clm_catalog_reserved_probe.py
python -m ruff format --check src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py tests/test_clm_catalog_reserved_probe.py
python -m mypy --explicit-package-bases --follow-imports=silent src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py tests/test_clm_catalog_reserved_probe.py
```

Use the existing qualified assets/environment and a frozen reviewed budget before
a native run. No dependencies or environment variables were added. Support/compare
cross-products remain refused before loading. A16-template native case must bind
its own tokens/control/baseline; do not assume12-template batch parity. Historical
paired comparisons change main and probe objectives, so they do not isolate a
probe effect or establish matched computation, circadian benefit or generalization.
[Protocol and evidence](../artifacts/runs/clm-catalog-reserved-probe-20261022/).

## Catalog-objective observations

200 focused checks per platform and scoped statics pass. One fixed native case
completes36updates84query108candidate252logits,1encoder/16sequences,134headforwards.
Wrapper264.101s; sampledRSS16.398GB, not true peak. No retry or changed dose.

| source checkpoint | probe TRAIN /2 | suffix /2 | old task /4 |
|---|---|---|---|
| baseline |0 ->2|0 ->1|1 ->2|
| A |1 ->1|1 ->2|2 ->2|
| B |0 ->1|0 ->1|3 ->0|
| return |0 ->1|0 ->1|3 ->0|

New-task learning is partial and old-task retention fails in these disposable
owners. A TRAIN tie hides one gain and one loss; baseline/A old-task aggregates
also hide correct-to-wrong flips. All probe TRAIN/suffix NLL/Brier improve while
A TRAIN/suffix ECE worsen. Old NLL worsens in every phase. The source learner
remains protected; main accuracy1/4->2/4->3/4->3/4, invoice still wrong under v2,
no prespecified recovery. Neither successful isolation nor a better endpoint
establishes benefit. Historical paired comparison changes both main/probe
objectives and is descriptive only; preserve the worse retention outcomes.

[Every metric/case/loss endpoint and costs](../artifacts/runs/clm-catalog-reserved-probe-20261022/RESULTS.md).

## Behavior and boundaries

The optional `--reserved-task-probes` argument extends the existing CLM temporal
stream. Default behavior is unchanged. It admits only the original `none` recipe;
combining it with selected catalog support or comparison fails before loading.
Existing standalone `repeat`/`route` studies remain available separately.

In default paired mode, at baseline, after A, after B and return, a disposable warm-head owner learns
`tavrix -> inspect` and `vemlot -> route` for six two-row paired updates. Each
uses fresh native AdamW/OneCycle and explicit training-only consent, a separate
budget/clock/inbox and actual delayed-label receipts. The main learner stays at
its original12updates36rows; the probes collectively pay24updates48rows.

Each probe reads its two TRAIN markers, two ` please` suffix templates and all
four phase-current old-task choices before and after adaptation. Targets enter
cached scoring only after every source/probe TRAIN update finishes. Accuracy,
raw NLL, Brier, five-bin ECE and obsolete errors reuse existing metric arithmetic.
All phases/endpoints are retained. No final evaluation, tuning, promotion,
replay, probe installation or export. Private trusted Python is not a sandbox.

Why this: new-task fit, suffix transfer and within-phase old-task damage answer
different questions. Across phases old policy truth may change. The marker task
is reserved from this CLM main stream, not known novel to pretraining; tavrix was
used earlier with another model. A correct initial answer is a ceiling. This
fixed tiny split does not establish broad generalization or circadian benefit.

## Modules

```text
src/core/temporal_clm_probe.py          fixed task/role/target declarations
src/app/clm_reserved_probe.py          six updates and unscored receipt/readout facts
src/app/clm_temporal_sequence.py       existing main stream; explicit checkpoints
src/adapters/clm_checkpoint_probes.py  feature binding, source/prefix proofs, cached metrics
scripts/run_clm_temporal_sequence.py   existing local asset composition and CLI
tests/test_clm_reserved_probe.py       paired integration/refusal/scoring checks
src/core/clm_catalog_probe.py          exact catalog probe_train roles and targets
src/adapters/clm_catalog_probe.py      independent catalog native owner
tests/test_clm_catalog_reserved_probe.py  catalog integration/refusal/scoring checks
```

App/core depend inward and never load adapters or model assets. The outer adapter
uses existing ports; no new orchestration framework, dependency or checkpoint chain.
Complete child identities and values stay protected before any later paid source
or probe work. Failed child work remains charged and cannot resume through a fresh
report. Reports retain bounded scalars, not weights, optimizer state or features.

## Commands

Use the prepared pinned optional CPU environment and local assets described by
[the encoder guide](clm-frozen-encoder.md#run-using-prepared-local-assets). The CLI
requires `--model-dir`, `--head`, `--output`; output must be new. Add
`--reserved-task-probes`. No download is necessary. A new experiment requires a
prospective finite protocol; the completed stage's allowance is not renewable.

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_clm_temporal_sequence.py tests/test_clm_reserved_probe.py
.venv/Scripts/python.exe -m ruff check src/core/temporal_clm_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_reserved_probe.py tests/test_clm_temporal_sequence.py
.venv/Scripts/python.exe -m ruff format --check src/core/temporal_clm_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_reserved_probe.py tests/test_clm_temporal_sequence.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent src/core/temporal_clm_probe.py src/app/clm_reserved_probe.py src/adapters/clm_checkpoint_probes.py src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_reserved_probe.py tests/test_clm_temporal_sequence.py
```

[Fixed protocol and resource limits](../artifacts/runs/clm-reserved-probe-20261022/PROTOCOL.md) ·
[Independent gates](../artifacts/runs/clm-reserved-probe-20261022/REVIEW.md) ·
[Decision](adr/ADR-0541-fixed-clm-checkpoint-probes.md).


## Historical paired-objective observations

93 focused tests per platform pass; scoped Ruff/format/mypy Windows+Linux-target
pass after two preserved typing failures and a reviewed identity/annotation repair.
No numerical fixture rerun. The sole native case completed36updates84rows with
one frozen16-sequence encoder,134head forwards and72lifetime head pairs. Wall
280.452s including fresh admission; sampledRSS16.400GB, not true peak. Probes pay
24updates48rows separately from the main12updates36rows. No downloads or retry.

| source checkpoint | probe TRAIN before ->after /2 | suffix before ->after /2 | old-task before ->after /4 |
|---|---|---|---|
| baseline |0 ->0|0 ->0|1 ->1|
| A |1 ->2|1 ->1|3 ->3|
| B |0 ->1|1 ->1|3 ->3|
| return |0 ->1|0 ->1|3 ->3|

A's initially correct marker is a ceiling; baseline fails even on exact TRAIN.
B's tied suffix accuracy hides one gain and one loss. No old-task choice flips
occur in these probes; old probability metrics still worsen at baseline/A.
All C NLL/Brier endpoints improve, while some ECE endpoints worsen. This is
partial fixed-task learning, incomplete transfer and no demonstrated recovery
or CPC benefit. Active catalogs also differ across phases. Keep all endpoints;
no broader capacity/generalization or release claim follows.

[All case/metric/loss endpoints and costs](../artifacts/runs/clm-reserved-probe-20261022/RESULTS.md).
That historical next dependency has since been delivered: explicit opt-in TRAIN
catalog learning, isolated catalog probe ownership, and the integration documented
above. Its fixed native observation has a separate protocol and acceptance gate.
Existing LoRA consolidation still does not supply a dense CLM contract.
