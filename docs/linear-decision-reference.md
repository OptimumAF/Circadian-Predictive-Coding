# Learned linear routing reference

```powershell
python -m scripts.run_linear_decision_reference --output linear-routing.json
```

This fixed local example learns three support intents with the existing
`PerceptronLearner` and training-only `AttributionStream`. It performs two old-task
passes, two new-vocabulary passes and two return passes: six updates,18 exposed
rows. No RNG, replay, final data, promotion, probe or external model is involved.
Every checkpoint is evaluated only after all training finishes. The example
preserves negative or unchanged results and does not select the best checkpoint.

The explicit six-word vocabulary comes from declared TRAIN words. Binary query
token presence supplies features; labels, source groups and scheduling context
stay outside the readout. Development queries intentionally overlap those words
and add an ignored token. This tests finite learnability and interference, not
unseen-word generalization. Results cannot establish calibration or CPC benefit.

## Structure and API

```text
src/core/decision_features.py             immutable schema and pure projection
src/adapters/perceptron_decision.py        TRAIN packet bridge and snapshot readout
scripts/run_linear_decision_reference.py  fixed composition using existing stream
```

`DecisionFeatureSpec(vocabulary, catalog)` fixes sorted canonical tokens and sorted
candidate IDs/descriptions. Catalog3..8 avoids existing two-class binary stream
semantics. Prediction accepts any nonempty subset of that catalog; changed or
unknown candidates refuse. `encode_decision(request,spec)` returns an owned
float64 row and parsed query-token count. No vocabulary fitting is performed.

`build_decision_training_batch(cases,spec,context)` accepts only TRAIN cases with
the full catalog and allowed targets. It maps IDs to fixed class indices and
returns an ordinary `AttributionBatch`. Context does not enter features. The
caller still owns cross-partition source declarations and execution budgets;
these records do not attest physical data provenance.

`PerceptronDecisionPredictor(spec,state)` owns a detached validated native snapshot.
It computes its affine coefficients only for allowed IDs and applies stable
temperature-one softmax over those logits. Renormalizing global probabilities
could underflow all allowed candidates when the global winner is excluded.
The perceptron update rule is unchanged; its probabilities are a readout, not its
training objective. Canonical candidate-ID ties and arithmetic ignore input order.

The unchanged evaluator keeps injected-callback behavior flags unknown. This
specific adapter is read-only by source; the evaluator is not a sandbox. Logical
query-token/candidate-column counts exclude validation/copying/metrics overhead.
The report retains all0/2/4/6 states, signed accuracy contrasts, metrics by task,
receipts, coefficient payload sizes and observed process RSS. Four coefficient
payloads nominally occupy672bytes; Python/runtime overhead is separate.

CLI output is fresh/exclusive; existing paths refuse before model construction.
Ordinary failures save the observed checkpoint/evaluation prefix and committed
budget counts, not a success claim or automatic retry. Uncommitted attempted work
is unknown in that report; the bounded execution evidence uses a separate entry
meter. Interrupt/process-crash recovery and portable deployment are outside scope.

## Focused checks

```powershell
python -m pytest -q tests/test_linear_decision_reference.py tests/test_typed_decision_reference.py
python -m ruff check src/core/decision_features.py src/adapters/perceptron_decision.py scripts/run_linear_decision_reference.py tests/test_linear_decision_reference.py
python -m ruff format --check src/core/decision_features.py src/adapters/perceptron_decision.py scripts/run_linear_decision_reference.py tests/test_linear_decision_reference.py
python -m mypy --follow-imports=silent src/core/decision_features.py src/adapters/perceptron_decision.py scripts/run_linear_decision_reference.py tests/test_linear_decision_reference.py
```

Extension: preserve the fixed class/vocabulary schema and use existing stream
roles, independent probes and explicit budgets. Freeze additional task sequences
before fitting. Released Open-Jev/CLM/ContextLM native contracts remain separate
R5.1/R5.2 gates; this conventional head is not a substitute for those models.

## Observed bounded example

The accepted run learns new-vocabulary development accuracy1/3→3/3 while retaining
old3/3. The return phase changes no coefficients, so recovery after forgetting
is untested. ECE increases0→0.3347590442251782 despite improved accuracy; no
calibration claim or post-outcome temperature adjustment. Six updates/18 rows,
48 readouts and every checkpoint/error/cost are retained.
[Executed evidence and independent review](../artifacts/runs/typed-linear-reference-20261022/RESULTS.md).
