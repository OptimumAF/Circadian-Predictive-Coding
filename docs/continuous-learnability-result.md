# Retain the detached continuous learner

Python callers can use `run_continuous_learnability(warm, new,
expected_standardization=None)` to receive `ContinuousLearnabilityResult`:

The optional [explicit binary-gripper objective](continuous-learnability-objective.md)
uses the existing kernel; default MSE and the ownership contract below are unchanged.

- `report`: the existing owned JSON-compatible learning report, including partial
  receipts, errors and costs.
- `candidate`: the existing detached final `NumpyContinuousBCLearner` after all
  checks pass; `None` for an incomplete execution. Admission errors still raise.

```python
from src.adapters.continuous_learnability_demo import run_continuous_learnability

result = run_continuous_learnability(warm_train_episode, new_train_episode,
                                   expected_standardization=frozen_a_statistics)
if not result.report["complete"]:
    raise RuntimeError(result.report["error"])
candidate = result.candidate
assert candidate is not None
# Candidate is detached, mutable local state. Inference/update authority and
# any new native experiment budget still belong to the caller.
state = candidate.snapshot_state()
```

`run_continuous_learnability_demo` still returns `.report` only; the existing
stream CLI schema/defaults remain unchanged. The new entry point runs the same
fixed A8/B8/Areturn8 recipe once. It returns the already allocated readout owner;
there is no additional initialization, SGD call, fork, persistent checkpoint,
callback or orchestration framework. No model is published to serving.

The result container is frozen. Its report and candidate are independently owned
mutable objects; the container does not promise immutable model arrays. Report
edits cannot change the learner, and snapshots/forks own their native arrays.
Any caught execution error explicitly sets `complete=False` and releases no
candidate, including a stale success flag. Paid receipts remain in the report.

## Structure and boundaries

```text
src/adapters/continuous_learnability_demo.py   Result API and unchanged fixed recipe
tests/test_continuous_learnability_result.py  Ownership/failure/native-gate checks
scripts/maniskill_continuous_smoke.py         Existing frozen MLP robot entry point
```

The existing robot entry point already admits this exact MLP class and checks the
complete original learner and private inference fork. Returning a detached owner
closes the missing connection between the learning recipe and this native caller.
It grants no promotion, data release, evaluation selection or final-test access.

## Verification and use

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_continuous_learnability_result.py tests/test_continuous_learnability_demo.py tests/test_continuous_stream_demo.py tests/test_visual_continuous_stream.py tests/test_maniskill_bc_smoke.py -k 'not direct_reference'
.venv/Scripts/python.exe -m ruff check src/adapters/continuous_learnability_demo.py tests/test_continuous_learnability_result.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/continuous_learnability_demo.py tests/test_continuous_learnability_result.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux src/adapters/continuous_learnability_demo.py
```

The accepted once-global direct-reference test is reused; repeated checks exclude
it rather than spending its original budget again. New tests share one tiny run
and count constructors/forks/updates, verify full state and alias isolation, and
exercise explicit incomplete behavior. Native-gate tests use only synthetic
reports and fake IO; real simulator work requires a separately frozen protocol.

The current native case reproduces the unchanged fixed recipe solely to obtain
its in-memory owner, because the earlier report deliberately stored no model.
That repeated training is separately charged. Before either environment is
created, all24 losses/eight endpoint half-MSEs/six signed metrics must match within
predeclared absolute1e-10, and policy/IDs/clocks/counts must match exactly. The
native case uses TRAIN17 then19 once each; neither task success nor failure can
select seeds, metrics or retries. Other physical safety constraints remain
unmeasured beyond action bounds and declared numerical checks.

[Recipe and metric definitions](continuous-mlp-learnability.md) ·
[Frozen native protocol, exact commands, observed outcomes and limits](../artifacts/runs/visual-mlp-native-20261019/RESULTS.md).
Full R4.4/R4.5, generalization and release claims remain separate and unfinished.

The accepted native case completed both TRAIN17/19 rollouts with unchanged state,
50 controls each and zero task return/success. Actions stayed in bounds; other
physical constraints remain unmeasured. Offline fit did not establish robot skill.
The negative result is preserved; no retry is admitted by its protocol.

## Fixed phase-count option

The existing runner now supports [bounded explicit phase updates](continuous-phase-updates.md), default8 unchanged. Historical24-update arms/protocols stay fixed; the option does not renew budgets or establish that more training improves native task skill.

## Optional continuous PC/CPC family continuation

The existing runner now accepts learner_family="pc", "neutral-cpc", or
"active-cpc"; the returned detached candidate is NumpyContinuousPCLearner and
contains complete adaptive state. BP/default behavior and native robot gates
remain unchanged; this opt-in does not make existing BC-only robot gates accept
PC or grant native/probe authority. Reports retain actual native update work
separately from common task-head diagnostics. See the current
[family contract, commands and limits](continuous-mlp-learnability.md).

## Frozen robot PC/CPC followup

The BC-only robot admission dependency above is now extended through the SAME
frozen-rollout entry point to exact continuous PC owners, with complete native
policy/wake preservation and owned basepolicy geometry. Existing corrective
updates and serving actor backends stay BC/nearest-only. Native experiments and
useful domain recurrence/return remain separately budgeted, unfinished claims.
[Admission, tests and limits](frozen-bc-robot-rollout.md).
