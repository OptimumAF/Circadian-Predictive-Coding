# Typed decision retrieval reference

```powershell
python -m scripts.run_decision_reference --output routing.json
```

The fixed six-case example selects only from each immutable allowed candidate set.
It is a lexical retrieval control for R5.2, not a trained modern model or a tool
executor. Candidate descriptions and query words determine Jaccard similarity;
softmax temperature is fixed at1. Ties use candidate ID, independent of input
order. No fitted vocabulary, labels, parameters, RNG or external service.

The example deliberately includes a lexical miss and two missing-target cases.
Every case remains in accuracy, candidate recall, Brier and calibration summaries.
A missing target has probability0: unconditional NLL is infinite, represented by
JSON null plus explicit status/count rather than clipping or dropping the case.
Brier sums squared errors over allowed IDs union the target. ECE uses five fixed
bins across every case; a small descriptive ECE does not establish calibration.
One reversed order per case is an explicit order control, not generalization.

## API and boundaries

```python
from src.core.typed_decision import DecisionCandidate, DecisionRequest
from src.core.decision_retrieval import retrieve_decision

request = DecisionRequest("query-1", "reset password", (
    DecisionCandidate("password", "password reset login account"),
    DecisionCandidate("shipping", "shipping parcel delivery tracking"),
))
result = retrieve_decision(request)
```

Requests/results are exact frozen records and tuples with bounded native fields.
These are domain records; the existing generic `Experience` envelope still owns
stream roles, clocks and permissions. This helper does not create another stream.
Validation rechecks bypassed frozen constructors. Every result scores all and only
allowed IDs, has finite normalized probabilities and chooses their maximum with
an ID tie rule. This is a categorical reference contract, not a verified native
Open-Jev/CLM interface. Preserve released scalar-head/objective semantics when
that separate adapter is specified; do not silently normalize native outputs.

`DecisionCase` keeps target, source group and role outside `DecisionRequest`.
`evaluate_decision_reference(cases, predict=...)` passes only detached requests to
the prediction port; final/probe/guard roles refuse, and train/evaluation groups
must be disjoint. It evaluates a private admitted snapshot, not mutable caller
records. Supplied-copy mutation is refused. Callbacks are not sandboxed: arbitrary
injected callback learning/actions are unknown, separately from zero evaluator-owned
training/actions. A deterministic custom predictor must preserve canonical outputs
under the recorded reverse-order control.

`DecisionEvaluationFailed.report` retains completed cases and at most one pending
case, with attempted/returned/validated counts. Returned invalid payloads have a
bounded scalar snapshot before validation; nonfinite/unsupported/missing fields
are explicit unavailable markers. Unreturned and unvalidated work is unknown,
not zero. Predictor token/candidate work is reported separately from evaluator
copies/metrics/validation and is unverified for injected callbacks. Interrupts and
process crashes are outside this helper's recovery scope.

The CLI writes once to a fresh path or stdout, saves a partial report on ordinary
evaluation failure with exit1, and refuses an existing output before prediction.
It executes no external action. No task-file loader, training runtime or new
orchestration framework was introduced. Existing stream/runtime components remain
the path for future learned/adaptive controls.

## Structure and checks

```text
src/core/typed_decision.py                 immutable reference schema/validation
src/core/decision_retrieval.py             pure lexical scoring
src/app/decision_reference_evaluation.py   isolated labelled scoring and costs
scripts/run_decision_reference.py          frozen example and exclusive JSON IO
```

```powershell
python -m pytest -q tests/test_typed_decision_reference.py
python -m ruff check src/core/typed_decision.py src/core/decision_retrieval.py src/app/decision_reference_evaluation.py scripts/run_decision_reference.py tests/test_typed_decision_reference.py
python -m ruff format --check src/core/typed_decision.py src/core/decision_retrieval.py src/app/decision_reference_evaluation.py scripts/run_decision_reference.py tests/test_typed_decision_reference.py
python -m mypy --follow-imports=silent src/core/typed_decision.py src/core/decision_retrieval.py src/app/decision_reference_evaluation.py scripts/run_decision_reference.py tests/test_typed_decision_reference.py
```

Why this: establish an honest, reproducible reference before external model or
adaptation claims. R5.1 model pins, a learned linear reference, broader calibration,
modern native input, streams and multi-step agent outcomes remain separate gates.
[Decision](adr/ADR-0515-type-allowed-candidate-reference-decisions.md).
