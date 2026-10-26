# Isolated categorical linear TRAIN fit

This optional control asks whether one fixed one-hot least-squares head reproduces
every label in a saved public TRAIN catalogue. It owns a disposable fit, without
touching a deployed, benchmark or fresh-task learner. It is a different objective
from native cross-entropy and PC energy; it is not a matched algorithm comparison.

## Run

From the repository root, using the existing NumPy dependency:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_linear_fit_control --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_linear_fit_control --source <saved-report.json> --output <new-result.json>
```

The source must contain `training_sources`: 1..16 exact complete TRAIN episodes,
raw148 observations, seven actions, at most16 steps per episode and128 total rows.
Input files are bounded at16MiB. Existing outputs are refused. The saved scientific
case in the evidence below is SPENT; the example is a usage interface, not approval
to repeat that experiment. No environment variables or new dependencies are added.

## Responsibilities and boundaries

```text
scripts/run_minigrid_linear_fit_control.py    bounded JSON/CLI boundary
src/adapters/minigrid_linear_fit_control.py   TRAIN admission, costs, owned report
src/core/categorical_linear_fit.py           pure fixed solve and prediction
tests/test_categorical_linear_fit.py         analytic math and ownership
tests/test_minigrid_linear_fit_control.py    role, source and failure behavior
tests/test_minigrid_linear_fit_cli.py        saved-source and CLI integration
```

The existing causal initial/current encoder produces296 coordinates; the existing
canonicalizer validates current and next views separately. Only current views and
action labels enter the fit. The core appends an intercept, solves float64 equal
row-weight one-hot least squares exactly once with `numpy.linalg.lstsq(rcond=1e-12)`,
and predicts with stable softmax. It has no role, filesystem, RNG or runtime policy.
Why this: a fixed simple objective provides a separate capacity/fit question
without changing an accepted online learner or selecting a favorable optimizer.

The adapter returns evidence from an immutable model, all raw probabilities/labels,
per-episode and aggregate CE/argmax/exact-trace metrics, all-class margins and
attempted/completed operation/row/array/time costs. SSE is computed explicitly
even when LAPACK returns an empty residual array. Costs for copied arrays are
logical array sizes, not total process RSS. The process supervisor measures RSS
separately. `native_forwards=0` and `optimizer_steps=0` do not mean no model work:
one supervised NumPy fit and one prediction are real work.

Changed source, detached input or model evidence is refused without repairing
foreign state. Failed post-fit work retains the fit, paid counters, raw predictions
and completed score records where available. Pre-fit canonical refusal raises
`LinearFitAdmissionFailed` carrying its partial cost report. No final or probe
section is supplied to the fit; no promotion, tuning, replay or stream runner is
introduced. JSON decoding reads the saved report, but only admitted TRAIN records
are used as learning inputs.

## Executed evidence and interpretation

The sole frozen54-row case completed with52/54 actions correct, four of six traces
exact, rank25 and SSE4.212268570088096. Both A traces miss zero-based step2; B and
C_train traces fit their action labels. This is a **negative all-label fit result**.
It neither proves linear classification impossible nor demonstrates adaptation,
generalization, retention, recovery or circadian benefit. Original online results,
failed comparisons and their budgets remain unchanged.

[Protocol, exact commands, failures, costs and independent review](../artifacts/runs/linear-train-fit-20261018/RESULTS.md).

## Verification and extension

Focused pytest, six-path Ruff/format and Windows/Linux mypy commands are recorded
verbatim in the linked results. The final combined gate passes145 cases.

Extend by composing existing native learner and attribution-stream ports under a
separate reviewed protocol. Keep any full-catalogue acquisition control explicitly
distinct from sequential online baselines, account for every repeated row and
retained array, and reuse existing isolated fresh-task/old-A/return probes. Do not
change this fixed solver or rerun its spent scientific case to manufacture a fit.
