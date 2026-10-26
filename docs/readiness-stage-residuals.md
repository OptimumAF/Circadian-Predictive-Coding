# Readiness phase residuals

The existing continuous learner can report fit separately for recorded ALIGN,
DESCEND, GRASP, LIFT and SETTLE rows. It uses its existing diagnostic prediction;
no extra fit, optimizer update, gradient, latent step or model forward is added
by selecting phase labels. Scalar grouping, summaries and validation do cost work.

Why this: the zero-success robot cases still have substantial translation/gripper
errors, but their old checkpoints used `all` for every row. Aggregate loss cannot
locate the phase failure. Old reports remain unchanged and their phase-specific
fit remains unavailable; no old models are reconstructed to fill this gap.

## Existing API

```python
from src.core.readiness_stage_diagnostics import ReadinessStageBinding
from src.adapters.continuous_learnability_demo import run_continuous_learnability

# Episodes and stages come from the existing qualified TRAIN teacher reader.
bindings = (
    ReadinessStageBinding(warm, tuple(warm_metadata["stages"])),
    ReadinessStageBinding(new, tuple(new_metadata["stages"])),
)
result = run_continuous_learnability(
    warm, new, diagnostic_stage_bindings=bindings,
)
```

Bindings own complete episodes, not just IDs. Admission checks full matching
observations/actions/specs/roles/flags, warm17/new19 order, exact stage counts,
known names and command semantics. They require known readiness demonstration
identities and TRAIN roles; provisional query/correction identities and probe/
final/guard roles are refused. A binding is not a teacher qualifier or arbitrary
saved-file authenticator. Existing reader qualification remains required.

Omitted bindings keep the old kwargs, output and `all` group. Selected reports add
bounded `diagnostic_stage_sources`; each existing checkpoint residual contains
stage labels, original row indices and per-axis signed bias/MAE/RMSE/half-MSE/max
error. Aggregate residual math and comparison directions are unchanged. Labels do
not enter model features, gradients, loss weights, selection, probes or promotion.
BP, PC, neutral CPC and active CPC share these readouts, regardless of their native
update objective. The metadata/row partitions are checked before further updates
and after the final elapsed-time callback; paid receipts survive a refused result.
Each checkpoint must match private expected native update and cumulative row counts,
including scheduled replay; a complete report requires all four endpoints. Missing,
duplicated or reordered checkpoints cannot release a selected result.

## Existing robot CLI

Add `--readiness-stage-residuals --readiness-teachers` to the existing
`python -m scripts.run_continuous_robot_learning` command with its normal output
and warm/new teacher paths. The qualified reader supplies the stages. Python
`run_learning_rollout` accepts the same optional pair. Raw inputs bind first;
canonical quaternion projection derives matching model bindings explicitly.
Returned source/partition metadata must match before SDK work. Unrequested stage
metadata is refused. This is the existing robot path, not a new runner.

Running that CLI spends training/simulator work and requires its own prospective
local budget. This implementation session ran only tiny correctness streams and
fake native environments; no new physical simulation or native comparative case.

## Files and extension

```text
src/core/readiness_stage_diagnostics.py       owned records and bounded metadata
src/adapters/continuous_learnability_demo.py  selected existing stage readouts
scripts/run_continuous_robot_learning.py      raw/model binding and qualified CLI
tests/test_readiness_stage_diagnostics.py    independent pure boundary tests
tests/test_readiness_stage_integration.py    actual stream/fake IO/error oracles
```

Core contains no IO/model/qualification/phase inference. Source bindings retain
at most100 rows in two episodes; reports retain metadata and existing four bounded
readouts, not another success chain. Work records stage summary/index visits;
metadata copies/validation/scalar reductions are additional to existing dense
MAC counts. No new dependency, environment variable or orchestration framework.

Future representations should bind their original raw episodes first, then derive
matching model-view bindings, preserving actual stage labels and command receipts.
Never infer readiness stages from elapsed step numbers or relabel query episodes
as qualified teachers. The next learner hypothesis is causal executed-action
history with an equal-width zero-history control; it does not establish that
missing history caused current failures. Existing elapsed-clock negatives remain.

## Executed checks

106 selected cases pass on Windows and pinned Linux, including independent manual
residual arithmetic using already paid predictions, all four learner families,
actual qualified-reader CLI admission, frozen fake IO, corruption before an update
and refusal after paid final work. Five-file lint/format and mypy win32/linux pass.
[Commands, costs, failures and review qualification](../artifacts/runs/readiness-stage-residuals-20261020/RESULTS.md).

```powershell
python -m pytest -q tests/test_readiness_stage_diagnostics.py tests/test_readiness_stage_integration.py
```

Exact affected regression selectors and five-file static command arguments are
recorded in ENDPOINT_TEST_COMMANDS.json and receipts at the linked stage. These fixtures
do not demonstrate useful acquisition, adaptation, reliability or CPC advantage.
