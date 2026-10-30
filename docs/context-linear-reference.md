# Disposable context ticket linear reference

The existing ticket qualification API accepts
`run_qualification(policy_context=True, linear_reference=True)`; CLI flag
`--linear-reference` requires `--policy-context`. The default BP qualification
and its fixed dose are unchanged. Unsupported combinations and existing output
paths refuse before fitting. This is source functionality; the279wheel predates
this addition and is not qualification of the new option.

The reference uses the existing bounded fixed-intercept one-hot least-squares
fit, equal row weights, float64 and `rcond=1e-12`. It fits original12joint TRAIN
rows/D9/K6 once, then reads original TRAIN CE/correctness/phase margins and runs
four known tasks with ordinary/stale/revoked scenarios in the existing sandbox.
The source budget is one fit/30seconds/512MiB; output remains bounded512KiB.
Both contexts are available from the outset. This is not chronological learning,
forgetting/return, an online controller, a matched BP/PC/CPC comparison or a
disposable reserved-probe score. No final/probe/replay labels or native/deployed/
benchmark owner are used; it must not become a promotion/tuning input.

The owned readout returns full affine logits for diagnostics and stable allowed-
candidate probabilities for typed decisions. It guards original model/spec/work
and retains attempted versus validated-return reads. Joint LS minimizes squared
one-hot residual, not CE; mean CE is a separate descriptive measure. Failed LS
classification would not prove representation nonlearnability.60float64
coefficients480bytes differ from H3 native54float32 parameters216bytes plus
gradients/optimizer; equal row/update counts would not establish matched compute.

Task attempts are recorded before delegation. Returned task/work is copied and
saved before post-task guards, with `accepted=False` until validated. A failed
late guard cannot discard costs or make a partial result look accepted. Lost
returns retain a paid pending attempt and unknown work. Input/schema/RNG/budget
origin/limits and previous paid records are checked before further work.

The [frozen study](reports/context-linear-reference-20261029.md) is **FAILED/SPENT
before fitting** due to Windows path separators in its Linux source preflight.
No genuine fit, solver or task ran; scientific outcomes are UNKNOWN. The
27focused software checks/platform do not replace that missing study. Its
producer/protocol/pins remain unchanged; no retry, renewal or renamed capture.

Developer checks execute actual tiny numerical fits/task fixtures and require
their own engineering budget:

```text
.venv/Scripts/python.exe -B -m pytest -q tests/test_ticket_linear_reference.py
.venv/Scripts/python.exe -B -m ruff check src/core/categorical_linear_fit.py src/adapters/linear_fit_decision.py scripts/ticket_linear_reference.py scripts/qualify_ticket_bp_fit.py tests/test_ticket_linear_reference.py
.venv/Scripts/python.exe -B -m ruff format --check src/core/categorical_linear_fit.py src/adapters/linear_fit_decision.py scripts/ticket_linear_reference.py scripts/qualify_ticket_bp_fit.py tests/test_ticket_linear_reference.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent src/core/categorical_linear_fit.py src/adapters/linear_fit_decision.py scripts/ticket_linear_reference.py scripts/qualify_ticket_bp_fit.py tests/test_ticket_linear_reference.py
```
