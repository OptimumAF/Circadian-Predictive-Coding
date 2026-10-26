# Arrived corrections for bounded nearest control

The existing corrective updater now accepts `NumpyNearestContinuousBCLearner`.
It appends one complete batch of arrived provisional TRAIN labels to a private
candidate. The original source and serving fork remain unchanged. The MLP path
continues to use eight SGD batches and its existing result/report schema.

```text
src/adapters/continuous_correction_demo.py   admission, private update, paid evidence
scripts/maniskill_continuous_smoke.py        existing collection/update/rollout pipeline
tests/test_nearest_corrective_adaptation.py  nearest behavior and callback regressions
```

## Python API

```python
from src.adapters.continuous_correction_demo import run_continuous_correction_update

# source is an existing TRAIN nearest learner with released prototypes.
# batch is the original PickCubeCorrectionBatch from closed TRAIN collection.
result = run_continuous_correction_update(
    source, batch, now=batch.labels[-1].labels_arrived_at
)
if result.report["complete"]:
    candidate = result.candidate  # detached; the source was not updated
```

Nearest calls return `NearestContinuousCorrectionResult(report, candidate)`.
Incomplete work returns no candidate. The existing MLP return type remains
`ContinuousLearnabilityResult`. Both expose the same two fields.

All labels must have arrived before any fork or private runtime is created.
The original executed actions remain in the trajectory; appended prototype
actions are the separately paired provisional targets. Frozen statistics are
reused without fitting. Capacity overflow, empty prototype sources, other roles
and invalid pairing refuse before private work. No eviction or promotion occurs.

**Conflicting exact inputs:** the original earliest-row nearest rule is preserved.
An appended label cannot override an earlier prototype for an identical normalized
input. Both records remain stored and charged. Appending corrections therefore
does not guarantee correct predictions at every corrected input or task success.

The runtime charges one update for the append. Reports distinguish that operation
from zero optimizer updates, gradients and forward passes. They count unique rows,
prototype growth, normalization and selected numeric payload. Validation copies,
Python/JSON containers and process memory are additional. The existing frozen
rollout summaries now retain nearest prediction/distance/prototype work.

Why this: a working nearest controller should learn from encountered states
through the existing arrival-gated path. Repeating the MLP's eight-batch recipe
would duplicate prototypes and spend capacity without adding information.

## End-to-end pipeline

`scripts.maniskill_continuous_smoke.run_corrective_adaptation` accepts either
supported learner. It collects the unchanged source on TRAIN19 with provisional
queries, performs its private update, then tests the candidate on TRAIN17 and
TRAIN19. Original source, full candidate state, receipts, charge and statistics
are checked before further work and release. Nearest guards check native owners
after public getters, including late receipt/clock reads. Corruption remains a
visible failure; there is no retry or rollback that erases a permanent charge.

This API creates real simulator episodes. Its use requires a separate bounded
protocol. The single pilot in the linked stage is one-use evidence; do not rerun
it simply to confirm documentation. No final, guard or probe input is admitted.
The correction generator's privileged geometry is used only to produce explicitly
provisional TRAIN labels; it is not a learner observation or an expert qualifier.

## Validation

Both platforms passed the same 27 cases: 23 new nearest cases and four existing
MLP compatibility cases. These include actual fake-port pipelines, original
prototype ties, wrong-role/unarrived/capacity refusal, permanent failed charges,
and source/readout mutations after earlier public observations. Scoped Ruff,
format and mypy checks and independent engineering review passed.

```powershell
python -m pytest -vv tests/test_nearest_corrective_adaptation.py
python -m ruff check src/adapters/continuous_correction_demo.py scripts/maniskill_continuous_smoke.py tests/test_nearest_corrective_adaptation.py
python -m ruff format --check src/adapters/continuous_correction_demo.py scripts/maniskill_continuous_smoke.py tests/test_nearest_corrective_adaptation.py
python -m mypy --follow-imports=silent src/adapters/continuous_correction_demo.py scripts/maniskill_continuous_smoke.py tests/test_nearest_corrective_adaptation.py
```

See [frozen protocols, exact commands, outcomes and costs](../artifacts/runs/nearest-corrective-adaptation-20261020/RESULTS.md)
and [independent review](../artifacts/runs/nearest-corrective-adaptation-20261020/REVIEW.md).
Scoped correctness does not establish generalization, CPC benefit, physical joint
constraint coverage, committed serving or repository-wide readiness.

## Fixed native pilot: no B improvement

The separately frozen one-use pilot completed all three scenes. Source B19 used
50 controls with return 0; after the 50-label private append, A17 retained return 1
in 29 controls and B19 still returned 0 in 50 controls. All six predeclared
acquisition/retention deltas were zero. The source stayed at 29 prototypes and
the candidate had 79. This verifies execution and retention, with no corrective
benefit established.

The pilot used 129 controls, 50 provisional geometry labels and zero fits, model
RNG draws, SGD or gradient calls. Its allowance is spent. Original qualified-B
nearest successes and all neural negatives remain accepted evidence. See the
linked results for costs, resource limits, warning, failures and next metric
control; changing the distance transform requires a separate prospective case.
