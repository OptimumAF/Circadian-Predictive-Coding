# Owned current-packet linear learner

`src.adapters.linear_fit_learner.LinearFitLearner` exposes the existing fixed
one-hot least-squares rule through `train_batch`, `predict`, `predict_logits`,
`snapshot_state`, `restore_state` and `fork`. Use a `LinearFitPolicy(feature_dim,
num_classes)` from `src.core.categorical_linear_state`. The existing recurring
ticket API now accepts `run_example(context_return=True, method="linear_fit")`;
CLI selection is `--context-return --method linear_fit --replay none`.
This is source functionality; the279wheel predates this method.

Ordinary context return and explicit `--separate-context-heads` are supported.
Private/paired modes, replay, `--record-train-fit` and class-balanced BP effects refuse before fitting
or reserving output. Existing perceptron and native-family defaults are unchanged.
The same original12updates72rows/+1label delay/A4-B4-A4/source30s512MiB envelope,
0/4/8/12 checkpoints and24task slots apply. The command retains failed-work
receipts and reports LS versus affine/unfitted read work separately.

Each successful update fits **only the current packet**, replacing prior
coefficients. It does not accumulate training rows, replay examples, optimizer
momentum or an unlimited success chain. The core keeps its fixed intercept,
equal row weights, `rcond=1e-12`, float64 and existing1..128row/1..296feature/
2..16class bounds. Arrays require finite exact float64 features in[-1,1] and
hard class indices. No random initialization or model draws occur.

The immutable snapshot contains policy, last fitted model or `None`, cumulative
completed updates and training examples. Initial state has no fabricated fit
record: logits are zero and probabilities uniform without a solve. Binary
`predict` returns B×1 class-one probabilities; categorical returns B×K.
`predict_logits` always returns full B×K scores. Caller-visible outputs and
snapshots are owned copies. Restoring a snapshot grants no budget, consent,
replay permission or whole-stream recovery authority.

The update validates input/count limits before solver delegation, stages copied
arrays and complete model/counters, then verifies original policy/state owners
and values, caller/staged arrays and NumPy/Python RNG before a single commit.
Foreign changes are refused without erasing them. Use under the existing
runtime's exclusive lease; the adapter is not a concurrent-operation manager.
Squared residual has a distinct native diagnostic definition, not CE or a
perceptron mistake count.

Failures after delegation raise the existing `TrainingUpdateFailed` with an
owned bounded scalar work receipt: fit attempt/row count, fit returns/returned
rows, candidate commits0, unknown-return and unpublished flags. Committed model
state stays unchanged. The original exception remains its cause. Pre-fit
validation errors pay no solver call. The caller must preserve the receipt;
the existing stream propagates it and permanently stops uncertain failed work.
Committed budget updates0 must not be reported as zero physical work.

The existing AttributionStream supplies ordinary TRAIN roles, consent, delayed
label release and immutable budgets. The adapter itself accepts arrays, not role
authority. It must not be called with final/probe labels or used as permission to
train a deployed/benchmark owner. Two-update fixtures verify actual arrival
before each solve, unchanged source/actor, original charges and0evaluation
forwards, including the existing D9/K6 ticket packet geometry.

[Evidence, exact commands, costs, review and limitations](reports/current-linear-learner-20261029.md).
The281component qualification ran no scientific study. A later distinct
[chronological observation](reports/linear-context-return-20261029.md) learned A
and B tasks, retained one A task during B, recovered A invoice on return and lost
both B tasks. It observes frozen candidate policies,
not a published deployed actor. The280joint one-fit study remains
FAILED/SPENT/UNKNOWN; the chronological result does not replace it.

For public-context isolation, use `ContextLinearFitLearner` with
`ContextLinearFitPolicy(feature_dim, num_classes, context_columns)` or add
`separate_context_heads=True` to the existing context-return API. Each homogeneous
packet projects out the two public columns and fits only its active head, leaving
the other unchanged. Missing, ambiguous or mixed TRAIN contexts refuse before
solving; typed queries require exactly one known policy_v1/v2 tag. No row bank or
context discovery is added. The complete owned snapshot has two bounded head
snapshots and their summed counters, with None for unfitted heads. Failure work
distinguishes a private staged commit from the zero published candidate commits.

```powershell
python -m scripts.recurring_ticket_workflow_example --context-return --method linear_fit --separate-context-heads --output context-linear-heads.json
```

Use a fresh output path from the physical D: checkout with D: temporary storage.
The original source limits and schedule remain. Maximum coefficient storage is
96float64/768B versus single-head60/480B. Routing projection rows/input/projected
values, full-query token visits and selected-head affine/unfitted work are API
quantities; validation/copies/SVD are not all-in quantified compute. One
[frozen case](reports/context-linear-heads-20261029.md) retains all four known tasks
after B and on A return. No acquired skill is lost, so recovery is null. Public
cues/increased capacity preclude matched-capacity or method-superiority claims.
Deployment, generalization, broader reliable adaptation and release stay open.

The existing [isolated ticket probe API](context-linear-probe.md) now accepts the
owned contextual LS learner and its in-memory snapshot. It runs private fresh-task
and old-TRAIN return phases, with initial and both endpoint task reads, preserving
source/benchmark isolation and paid failure costs. This is software qualification;
no new further-learning scientific case or CPC claim follows from its tests.

Focused developer checks perform actual small numerical fits:

```text
.venv/Scripts/python.exe -B -m pytest -q tests/test_linear_fit_learner.py
.venv/Scripts/python.exe -B -m ruff check src/core/categorical_linear_state.py src/adapters/linear_fit_learner.py tests/test_linear_fit_learner.py
.venv/Scripts/python.exe -B -m ruff format --check src/core/categorical_linear_state.py src/adapters/linear_fit_learner.py tests/test_linear_fit_learner.py
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent src/core/categorical_linear_state.py src/adapters/linear_fit_learner.py tests/test_linear_fit_learner.py
```
