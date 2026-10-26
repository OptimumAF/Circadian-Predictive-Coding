# Frozen-hidden TRAIN readout diagnostic

## What it does

`diagnose_readout_fit` accepts an existing detached BP, PC, neutral CPC or active
CPC stream readout and complete released TRAIN episodes. It freezes the hidden
representation and fits a disposable output head once. It reports physical MSE
on the original targets, before and after, plus target clipping, numerical rank,
singular values, conditioning, coefficient norm and work. It does not install the
head, change the learner or select a deployed policy.

Why this: the accepted BP learner improves imitation loss yet fails closed-loop
control, including after explicit gripper decoding. This diagnostic separates a
constructive fit of fixed features from the existing optimization outcome before
more simulator work. Successful TRAIN fitting does not establish native success,
generalization, reliable adaptation or a circadian advantage.

## Existing stream usage

```python
from src.adapters.continuous_readout_fit import diagnose_readout_fit

# result is an already completed run_continuous_learnability result;
# warm and new are its already released complete TRAIN episodes.
diagnostic = diagnose_readout_fit(result.candidate, (warm, new))
print(diagnostic.fit.physical_mse_before, diagnostic.fit.physical_mse_after)
print(diagnostic.fit.rank, diagnostic.fit.work["column_nullity"])
```

No new runner, CLI flag, configuration, environment variable or dependency is
required. The adapter accepts exact owned learner types, at most16 distinct
complete TRAIN episodes and128total rows, matching source units and bounds.
Probe-training, probe-test, final-test and guard episodes are refused. The caller
retains release authority; the adapter never converts roles or fits statistics.

## Mathematical convention and costs

For frozen hidden activations `H=tanh(X W1+b1)`, append an intercept and solve
`[H,1] C = atanh(clip(normalized_targets, -1+1e-6, 1-1e-6))` with
`numpy.linalg.lstsq`, fixed `rcond=1e-12`, one call. Constant action axes use zero
logit targets and exact constant physical predictions. Predictions retain the
existing bounded tanh/physical convex blend. Returned arrays are detached.

This minimizes a transformed logit-space objective at the declared numerical
rank. It is not a minimizer of the original nonlinear physical or normalized
MSE. The reported physical MSE averages unmodified targets over ALL rows/axes;
it differs from the benchmark half active-axis normalized loss. Rank-deficient
results are valid. Full row rank in an underdetermined design does not imply a
unique head: report column nullity alongside the nonzero singular ratio.

Input/hidden widths are bounded to64, action width16. Logical payload sizes and
known forward dense MAC are reported. LAPACK arithmetic/workspace, elementwise
operations, validation copies, owner snapshots, Python objects and RSS are extra.
No SGD, RNG, replay, empirical-statistics fit or simulator work occurs in this
API. A failed solve is not automatically retried; experiment callers must retain
its attempted work. Do not use this API to bypass the normal learner/promotion
path. Extend via a separately specified diagnostic objective and isolation tests.

## Modules and checks

```text
src/core/continuous_readout_fit.py       bounded pure mathematics
src/adapters/continuous_readout_fit.py   TRAIN admission and source isolation
tests/test_continuous_readout_fit.py    math and boundary checks
```

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_continuous_readout_fit.py
.venv/Scripts/python.exe -m ruff check src/core/continuous_readout_fit.py src/adapters/continuous_readout_fit.py tests/test_continuous_readout_fit.py
.venv/Scripts/python.exe -m ruff format --check src/core/continuous_readout_fit.py src/adapters/continuous_readout_fit.py tests/test_continuous_readout_fit.py
.venv/Scripts/python.exe -m mypy --platform linux --follow-imports silent src/core/continuous_readout_fit.py src/adapters/continuous_readout_fit.py tests/test_continuous_readout_fit.py
```

For this increment the exact platform commands and prospective fixture/study
budgets are in [the evidence stage](../artifacts/runs/frozen-hidden-readout-fit-20261020/PROTOCOL.md).
Scoped correctness is separate from the frozen TRAIN study and release readiness.


## Accepted result (2026-10-20)

29 focused cases per platform and scoped statics passed independent acceptance.
The fixed TRAIN37 study reduced physical MSE0.1256569783→2.9054054e-13, with
rank37/65columns, nullity28, condition93960.5 and coefficient norm9354.94.
The head was not installed or served; prior native failures remain. See the
[complete result and costs](../artifacts/runs/frozen-hidden-readout-fit-20261020/RESULTS.md).
