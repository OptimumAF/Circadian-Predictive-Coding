# Fixed public21 MLP optimization-depth study

## Accepted observed outcome, 2026-10-19

The single fixed arm completed: final A half-MSE.0271289794 and B.0456705521,
versus p8 .0482477274/.1944966594. All four phase endpoints and all signed axis/
metric differences remain. Three of six metric differences worsen; final Bdy
half-MSE and worst gripper errors also worsen.96newupdates/2816rows cost4xthe
oldtraining exposure and~2.67xtotal dense math. No native rollout was performed.

72checks pass perplatform, scoped lint/format/types and independent source/gate/
actual readback pass. Complete final parameters/momentum96/2816 are saved.
This is mixed fixed-TRAIN optimization evidence, not native/generalization/CPC
or release benefit. The next separately frozen check must test native utility.

This one-use composition runs the existing continuous learnability runner with
32updates per A/B/A phase. It compares all phase endpoints with the accepted
p8public21 report without retraining the baseline. The same default seed17,
hidden64, lr.01, momentum.9, MSE, public21 input and A-only normalization remain.

Why this: default24-update neural arms still fail the native task while a
simpler nearest control works locally. A fixed count tests optimization depth
and its cost. The count32 was selected prospectively as four times the training
exposure; it cannot be changed in response to a desired result. This study
does not establish that more training solves native control or task retention.

## Existing paths and structure

```text
artifacts/runs/public-vector-mlp-depth-20261019/
  run_case.py       fixed composition, corresponding-phase comparison, snapshot
  run_check.py      existing serial bounded command recorder
  PROTOCOL.md      sources, settings, endpoints, work and prospective limits
  REVIEW.md        independent source/gate/actual readback
  RESULTS.md       exact executed commands, outcomes, costs and next action
tests/test_public_vector_mlp_depth.py  manual reports/receipts and archive cases
```

The composition reuses accepted qualified teacher readers, fullraw789→public21
projection, old24-arm admission helpers, typed learner snapshots and the existing
runner's private candidate/source/actor ownership checks. No production learner,
optimizer, orchestration framework or dependency changes. Baseline policy,
statistics, progress, source projection, work and finite measured metrics validate
before the new fit. The single new A-only fit must equal accepted21statistics
before model construction or SGD. No baseline state archive is used for training.

The existing runner returns the full paid sequence before comparison or snapshot
checks. Initial measured checkpoint0 must exactly equal the accepted p8 readout;
this is checked after the run returns, with no extra diagnostic/gradient work or
pre-SGD callback. A mismatch invalidates the comparison and retains paid updates.
Incomplete executions release no model, retain their progress and are not retried.
Existing outputs refuse before reads/model work; evidence is never overwritten.

## Comparison and cost

The frozen endpoints correspond to phases: initial0vs0, warm32vs8, new64vs16,
return96vs24. For both A and B keep half-MSE and all five numeric summaries of
each of four action axes. The primary comparison is final p32-minus-p8 half-MSE,
negative better. All six metric differences remain: acquisition/recovery gains
positive better, retention losses/return deficit negative better. No axis or
checkpoint selection is used to decide the headline.

Training uses96updates/2816rows/59unique/2757repeat/0replay; p8 used24/704.
This is fourfold training exposure and higher computation, **not matched compute**.
The existing cost report includes normalization, fixed endpoint conditioning/
residual/actor readouts, dense math, gradients, input payloads and model statistics.
Planned full-run totals include11680512denseMACs versus4381440p8,1387776gradient
bytes and563200inbox bytes. These are numeric estimates, not allocation/peak
process-memory bounds. Projection/rendering/source exposure are separate.

Complete final96/2816 parameters and momentum save as the existing eight float64
NPZ arrays, with policy/counters in case.json. It is not a new serialization codec
or checkpoint chain. A later native test would need its own prospective protocol;
this study performs no simulator controls, teacher collection or new labels.

## Developer checks and limits

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_public_vector_mlp_depth.py
.venv/Scripts/python.exe -X utf8 -B -m ruff check artifacts/runs/public-vector-mlp-depth-20261019/run_case.py tests/test_public_vector_mlp_depth.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check artifacts/runs/public-vector-mlp-depth-20261019/run_case.py tests/test_public_vector_mlp_depth.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --python-version 3.12 --platform linux artifacts/runs/public-vector-mlp-depth-20261019/run_case.py
```

Tests are synthetic0fit/0SGD; initialized/restored counters are metadata, not
training. The real one-use command is separately gated by source review and
focused final checks, with30s wall time/512KiB report/128KiB archive/8MiB log.
Do not rerun it or overwrite evidence. No sweep, selected seeds, new baseline
settings or final/guard/probe access. Same-PickCube reset contexts are not
distinct tasks; offline imitation does not establish native skill, generalization
or a circadian advantage. Earlier negatives remain scientifically valid.

[Exact results, commands and handoff](../artifacts/runs/public-vector-mlp-depth-20261019/RESULTS.md) ·
[Protocol](../artifacts/runs/public-vector-mlp-depth-20261019/PROTOCOL.md) ·
[Independent review](../artifacts/runs/public-vector-mlp-depth-20261019/REVIEW.md).

## Subsequent native evidence

[Zero-retraining native validation](public-vector-mlp-depth-native.md) matches all42saved scalars exactly but both TRAIN contexts still return0/no success after50steps. All6p8differences0. The mixed offline improvement did not establish native skill.
