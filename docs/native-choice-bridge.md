# Open-Jev Choice wire bridge

## Native TRAIN update

`OpenJevTrainingLearner` takes the original loaded model, native compile_request
and instructions. It accepts DecisionRequest plus a hard allowed candidate ID
through NativeLearner, including the existing ExperienceInbox delayed-label
path. Torch imports are optional and lazy. The fixed recipe is native training
temperature1 NLL +.1 sum-squared Brier; head lr1e-4, LoRA lr5e-5, AdamW
decay.01/betas(.9,.999)/eps1e-8 and global clip1. Only head/LoRA A/B update.
CPU float32 is a declared local subset, not released BF16 parity.

```bash
HF_HUB_CACHE=/path/to/pinned/hub timeout 600 python -B -m scripts.run_open_jev_native_update \
  --source /path/to/pinned/source --checkpoint /path/to/pinned/checkpoint \
  --output /path/to/fresh-result.json --save-checkpoint /path/to/fresh-adapter
```

Use the [frozen local protocol](../artifacts/runs/native-open-jev-update-20261022/PROTOCOL.md)
and its offline/eight-thread settings. One load, one TRAIN label, one attempted
update, four readouts, thirteen DecisionModel forwards maximum; checkpointed
internal backward recomputation is additional work. No warmup/retry/fallback,
replay, probe/final data, tuning or deployed selection. The old/new development
targets never enter inference callbacks. The source TRAIN query differs from
the new readout. One exposure and four observations do not demonstrate useful
adaptation or generalization. Failures and unknown random-draw costs are kept.

Successful local snapshots detach trainable weights, AdamW, gradients and private
CPU RNG. They bind to this learner and its unchanged frozen base using normal
Tensor identity/version checks; hostile Python or `.data` writes are outside the
trust boundary. Restoring never renews permanent attempted work. On uncertain
mutation the adapter raises TrainingUpdateFailed and stops further updates.
No full-base copy, actor fork, portable optimizer resume or checkpoint publication
transaction is promised. Fresh native inference output has an explicitly inherited
unrecalibrated temperature after weight changes; it must not be presented as a
newly calibrated model.

The native Predictor's `checkpoint_sha256` stays its original-load fingerprint
after an in-memory update. It is checkpoint-origin metadata, not an updated-weight
digest. The driver records separate trainable pre/post hashes and a fresh saved
package; reload and portable training resume remain separate unproven claims.

```powershell
python -m pytest -q tests/test_open_jev_training.py tests/test_open_jev_native_smoke.py
python -m ruff check src/adapters/open_jev_training.py scripts/run_open_jev_native_update.py scripts/run_open_jev_native_smoke.py tests/test_open_jev_training.py
python -m ruff format --check src/adapters/open_jev_training.py scripts/run_open_jev_native_update.py scripts/run_open_jev_native_smoke.py tests/test_open_jev_training.py
python -m mypy --follow-imports=silent src/adapters/open_jev_training.py scripts/run_open_jev_native_update.py scripts/run_open_jev_native_smoke.py tests/test_open_jev_training.py
```

Run kernel fixtures in the pinned Torch CPU environment. An unavailable optional
Torch backend produces explicit skips, not evidence of kernel correctness. Extend
using existing stream/learner ports and a separately frozen small task protocol;
do not copy this smoke into repeated nearly identical update/capture increments.

This pure local bridge prepares a native request and validates a returned byte
body. It loads no model, opens no network connection and executes no candidate.
Use it with the separately [pinned Open-Jev contract](upstream/open-jev.md).

```python
import json
from src.core.typed_decision import DecisionCandidate, DecisionRequest
from src.adapters.open_jev_choice import build_open_jev_choice, decode_open_jev_choice

request = DecisionRequest("request-1", "reset my password", (
    DecisionCandidate("password", "Password and login help"),
    DecisionCandidate("billing", "Charges and refunds"),
))
payload = build_open_jev_choice(request, "Choose the support intent.")
body_to_send = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf8")
# A separately budgeted transport must obtain actual response_body bytes.
# observed = decode_open_jev_choice(request, response_body)
```

The existing immutable request schema supports1..8 candidates and bounded text:
this is a declared subset of native Choice1..255, not the entire Noul/Score or
JSON-state API. All query/description strings, IDs and order are preserved.
Labels, roles and source groups are not serialized. Instructions are explicit
caller data; this helper cannot certify that they contain no leaked answer.

Decode only exact bytes,1..65536B, strict UTF8/JSON with duplicate and nonfinite/
overflow numeric refusal. A pre-parse scan limits container nesting to64 levels,
ignoring brackets inside quoted/escaped strings; this bound is independent of
Python/test-runner recursion limits. One answer must match the requested question ID and
candidate set, native type and selected maximum. Exact ties use REQUEST order,
which can differ from the lexical control's ID rule. JSON object key order in
the reply does not replace the original request's tie rule.

Native probability mass and confidence consistency use the inspected1e-6
tolerance. Probabilities are preserved, including admitted rounding differences;
there is no local renormalization or softmax. Native confidence is a normalized
margin over chance, not max probability or calibrated correctness. It is checked
against the normalized mass for consistency without replacing observed values.
No raw score, tokenizer count or model work is fabricated. The reply is separate
from `DecisionResult`, whose raw-score/tie contract is different.

`OpenJevChoiceReply` is a frozen decoded observation with immutable probability
pairs and the exact original body. Extra root metadata remains uninterpreted in
that body. The decoder does not authenticate provider/checkpoint identity, verify
server-reported work, or bind the response to state beyond the declared question/
candidate checks. The API echoes no request nonce; transport correlation and
actual model provenance require separate evidence. Any refusal raises ValueError
without a fallback prediction, retry, budget renewal or evidence deletion.

## Structure and checks

```text
src/adapters/open_jev_choice.py    pure native wire encoding/validation
tests/test_open_jev_choice.py     analytic wire fixtures; no real model
docs/upstream/                    distinct primary model contracts and limits
```

```powershell
python -m pytest -q tests/test_open_jev_choice.py tests/test_typed_decision_reference.py
python -m ruff check src/adapters/open_jev_choice.py tests/test_open_jev_choice.py
python -m ruff format --check src/adapters/open_jev_choice.py tests/test_open_jev_choice.py
python -m mypy --follow-imports=silent src/adapters/open_jev_choice.py tests/test_open_jev_choice.py
```

Extend by adding an explicit transport with bounded local inference protocol and
actual checkpoint/usage receipts. Keep native probabilities, confidence and
candidate-order controls distinct from existing lexical/linear references.
Passing fixture plumbing does not demonstrate trained-model execution, adapter
learning, calibration, generalization, workflow success or CPC advantage.

## Bounded native checkpoint smoke

`scripts/run_open_jev_native_smoke.py` uses the original Open-Jev loader and
predictor on one frozen support-intent query, then its reversed candidate order.
`tests/test_open_jev_native_smoke.py` uses recording fakes only. The driver
retains native JSON before checking metadata/Choice semantics, owns the frozen
request and refuses mutation. Valid wrong or order-sensitive answers remain
completed observations. A failed call stops the pair without retry.

For the pinned source/checkpoint locations, exact revisions, separate dependency
overlay and download allowances, use
[`PROTOCOL.md`](../artifacts/runs/open-jev-native-smoke-20261022/PROTOCOL.md).
The local runtime needs the source's pinned Transformers5.10.2, PEFT0.19.1 and
Accelerate1.13.0 with compatible Torch and safetensors. These optional inference
dependencies are outside the repository's lightweight NumPy installation.

```bash
HF_HUB_CACHE=/path/to/pinned/hub timeout 600 python -B -m scripts.run_open_jev_native_smoke \
  --source /path/to/pinned/source --checkpoint /path/to/pinned/checkpoint \
  --output /path/to/new-result.json
```

The output must be fresh. Unset inherited `JEV_*` overrides. The driver selects
CPU float32, offline loading, seed17, eight threads, batch1/max256, no cache or
generation. It records file origins, versions, native scorer/forward/token work,
observed placeholder initialization, prediction RNG hashes and OS peak RSS.
The external timeout bounds the whole invocation; RSS checks occur at phase
boundaries. This is a direct Python API call, not an HTTP server/transport test.
CPU float32 changes released BF16 arithmetic, so released quality/parity remains
unproven. No training, tuning, probe, final evaluation or action is performed.
Extend via a separately frozen task protocol using existing stream/evaluation
ports; preserve this smoke's observations and the original native math.

## Probability-only scoring

`decode_open_jev_probabilities(request, body)` projects a validated native reply
to `ProbabilityDecisionResult`. It preserves probabilities, reported margin
confidence and optional reported token counts without inventing raw logits.
The caller retains the source body. Missing token counts stay `None`, distinct
from reported zero; these remain provider assertions, not measured compute.

`src.app.decision_reference_evaluation.evaluate_probability_reference` accepts
that result through the same label-free `DecisionRequest` callback boundary.
It shares existing case admission, NLL/Brier and five-bin ECE calculations.
ECE uses maximum probability; `reported_confidence` remains the native margin.
The probability schema admits1e-6 mass rounding without normalization. A generic
provider may choose any exact maximum; the native decoder still enforces its
original request-order tie rule. Original/reverse probability differences and
selection disagreements are retained as adverse observations, without retry.
The older raw-score evaluator and its exact-order-control semantics are unchanged.
Final/probe roles refuse before prediction. Callbacks are unsandboxed; evaluator
isolation does not certify a callback's external side effects or training state.

Reprocess the already recorded native smoke, making **zero new model calls**:

```powershell
python -m scripts.score_open_jev_smoke --smoke-report artifacts/runs/open-jev-native-smoke-20261022/actual.json --output probability-report.json
python -m pytest -q tests/test_probability_decision_reference.py tests/test_typed_decision_reference.py
python -m ruff check src/core/typed_decision.py src/app/decision_reference_evaluation.py src/adapters/open_jev_choice.py scripts/score_open_jev_smoke.py tests/test_probability_decision_reference.py
python -m ruff format --check src/core/typed_decision.py src/app/decision_reference_evaluation.py src/adapters/open_jev_choice.py scripts/score_open_jev_smoke.py tests/test_probability_decision_reference.py
python -m mypy --follow-imports=silent src/core/typed_decision.py src/app/decision_reference_evaluation.py src/adapters/open_jev_choice.py scripts/score_open_jev_smoke.py tests/test_probability_decision_reference.py
```

The CLI binds the saved query/candidates/instructions/order and raw/owned replies
before invoking the evaluator, records the source hash and requires fresh output.
It scores one labeled query; the reverse is an order control, not a second sample.
Its reported232tokens are inherited from the original paid run, not new forwards.
One descriptive case cannot establish calibration or task quality. Extend using a
separately frozen matched development protocol; keep final data unavailable and
probe updates separate. Do not turn native probabilities into invented logits to
fit the older result schema or create a second evaluation/stream framework.

## Fixed six-case comparison

`scripts.run_native_decision_comparison` reuses the saved six-case lexical report
and the unique final update6 linear snapshot. It validates the frozen vocabulary,
candidate descriptions and source cases, then pays for12 linear readouts without
training. The original/reverse native requests use the same six public development
cases, including two targets absent from allowed candidates. Their errors and
infinite NLL stay in the report. It saves the completed linear report before
loading the native model, then retains native raw replies and partial costs.

```bash
HF_HUB_CACHE=/path/to/pinned/hub timeout 600 python -B -m scripts.run_native_decision_comparison \
  --lexical-report artifacts/runs/typed-decision-reference-20261022/frozen-example.json \
  --linear-report artifacts/runs/typed-linear-reference-20261022/linear-example.json \
  --source /path/to/pinned/source --checkpoint /path/to/pinned/checkpoint \
  --output /path/to/fresh-comparison.json
```

Use the [prospective protocol](../artifacts/runs/native-decision-comparison-20261022/PROTOCOL.md)
and its explicit CPU thread/offline settings before a model invocation. This
distinct workload declares one load/12requests/32candidate forwards/8192tokens;
the original smoke loader defaults stay2/4/1024. All native limits are enforced
through the existing loader. No warmup, retry, training or fallback is performed.
Report differences are native minus control: accuracy/recall higher favorable,
Brier/NLL/ECE lower favorable. Non-finite NLL differences remain unavailable.
Task inputs are matched; capacity, pretraining, exposure, computation and memory
are different. This descriptive six-case study cannot establish calibration,
generalization, adaptation, circadian benefit or matched-compute superiority.

Focused check:
`python -m pytest -q tests/test_native_decision_comparison.py tests/test_open_jev_native_smoke.py`.
Extend with separately frozen task protocols and existing evaluators/learners,
preserving all missing-target cases, original metrics and negative results.
