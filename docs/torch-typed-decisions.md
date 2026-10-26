# Frozen typed decisions from native categorical learners

The existing BP, PC, neutral CPC and gated CPC `TorchFeatureLearner` can now
serve the same typed decision callbacks as the ticket dispatcher. It uses a
complete detached checkpoint, not reconstructed parameter-only history.

```python
from src.adapters.torch_decision import TorchDecisionPredictor

# spec: fixed DecisionFeatureSpec, learner: existing TRAIN candidate.
predict = TorchDecisionPredictor(spec, learner.snapshot_state())
result = predict(request)  # Existing DecisionRequest -> DecisionResult.
predict.assert_unchanged()
print(predict.work)  # Detached immutable counters, including failed attempts.
```

Requires the optional existing CPU Torch environment; no new dependency, model
download or training method. Vocabulary width must match the learner and remain
within its existing policy bounds; catalog has3..8 sorted IDs/descriptions under
the existing feature schema. No vocabulary fitting or implicit class remapping.
The caller owns inference admission, time/memory/call limits and data roles.

## Behavior and isolation

`TorchFeatureLearner.predict_logits(features)` returns owned float64 NumPy BxK
values from native float32 logits. It validates the existing complete snapshot,
input geometry/conversion and finite CPU float32 native output. Existing
`predict` retains its float32 softmax, binary Bx1 class-one output, categorical
BxK output and previous calculation order.

The typed predictor restores the complete checkpoint: parameters, native PC/CPC
traffic/chemistry/history/ages/lineage/generator, or BP momentum and gradients,
plus original policy and update/example counters. Later source updates or input
snapshot mutations cannot alter the copy. Before and after each native readout,
the complete local state must equal its frozen pin. A fault latches refusal;
subsequent calls cannot restart training, restore state or renew a budget.

Every call projects **all** classes through the existing head. Only permitted
candidate logits enter a stable scalar softmax; forbidden high logits cannot
erase permitted probability mass. Scores use sorted IDs with deterministic ID
tie breaking, matching the existing typed port. Request descriptions must match
the original catalog. This callback selects an action; the existing dispatcher
still owns revision and permission checks at effect time.

## Costs and limits

`work` counts decision attempts, raw-logit call attempts/returns and successful
typed returns. A raw-logit attempt can fail its own validation before reaching
the native forward; a native return followed by failed state validation remains
charged and is not a successful decision. Token visits and permitted comparisons
are returned through `DecisionResult`. These counters are not exhaustive FLOPs.

The existing constructor+restore path initializes two heads before replacing
their tensors; seed-local random draws and copies are paid overhead. For one
row, native affine multiplies nominally D×H+H×K, plus tanh/bias, validation,
snapshot tensor/RNG copies, scalar decoding and Python work. Inference does not
run latent training iterations. Storage is one native copy and one pin with
fixed counters, independent of request count; ephemeral snapshot copies add peak
memory. This is synchronous trusted Python, not hostile-code containment or
concurrent training/promotion. No final/probe selection authority is provided.

## Files and checks

```text
src/adapters/torch_feature_learner.py  Existing native learner + owned raw logits
src/adapters/torch_decision.py         Frozen typed callback and paid call counters
tests/test_torch_decision.py          Native parity/isolation/dispatcher behavior
```

```powershell
python -m pytest -q tests/test_torch_decision.py
python -m ruff check src/adapters/torch_feature_learner.py src/adapters/torch_decision.py tests/test_torch_decision.py
python -m ruff format --check src/adapters/torch_feature_learner.py src/adapters/torch_decision.py tests/test_torch_decision.py
python -m mypy --explicit-package-bases --follow-imports=silent src/adapters/torch_feature_learner.py src/adapters/torch_decision.py tests/test_torch_decision.py
```

These commands perform native work. Use the finite recorded
[correctness protocol](../artifacts/runs/torch-typed-decision-20261023/PROTOCOL.md)
when reproducing its scope; retain failed attempts. No automatic sweep or
experiment is implied. The ordinary/stale/revoked dispatcher fixtures use
literal analytic phase-routing weights: integration correctness, **not learned
task success**. Existing negative adaptation/replay results remain unchanged.

To extend safely, compose this callback with the existing stream's candidate
checkpoints and fixed full-task episode protocol. Freeze dose, seed, endpoints,
controls and all inference/diagnostic/copy costs before a separate small family
observation. Do not replace the deployed learner or select a favorable endpoint.
