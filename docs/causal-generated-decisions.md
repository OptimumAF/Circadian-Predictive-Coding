# Bounded generated-ID decisions

`src/core/causal_decision.py` validates one causal generation against an existing
`DecisionRequest` and its allowed candidates. It returns an exact selected ID
and detached raw evidence. It performs no model/tokenizer/IO, sampling, training,
effects, permissions or calibration. Existing probability-based decision APIs
are unchanged; generated text is never converted into invented logits, one-hot
probabilities, confidence or token work.

## Contract

- `CausalModelReference`: asserted model ID plus40lowercasehex revision field.
  This is a revision-shaped binding, not authentication of loaded/released weights.
- `CausalGeneration`: request/model identity, raw exact UTF-8 string<=4096bytes,
  provider finish reason and optional reported input/output token totals. Raw
  failure/prose/empty/truncated output and unknown counts remain representable.
- `decode_causal_decision`: revalidates exact types/constructor-bypassed fields,
  request/model identity, exact allowed ID, `finish_reason="stop"`, knownpositive
  reported counts within explicit positive safe-integer callercaps.
- `GeneratedDecisionResult`: request/selectedID/detached generation only. No
  scores/probabilities/confidence/logits fields. `validate_generated_decision_result`
  rechecks the result against request/model/caps at a receiving boundary.

There is no trimming, quote stripping, prose extraction, case folding or unknown
ID fallback. An apparently valid ID returned with `length`, missing counts or a
different source is not qualified. Missing usage stays`None`, not0. Explicit raw
0usage is preserved but cannot support a nonempty qualified selection. Counts
are provider assertions, not measured FLOPs, cache visits or authenticated work.

Why this: the existing `DecisionResult`/`ProbabilityDecisionResult` require real
probability evidence. A causal model may instead expose only text/tokens; forcing
that into a probability result would fabricate information and undermine fair
baseline/calibration comparisons. The core layer depends only on stdlib and
existing core request/text/count validation; no adapter or runtime dependency.

## Metadata-only example

This constructs a fixture. It does not load/invoke Qwen or establish its native
token counts. The separately [pinned causal reference](upstream/context-and-causal.md)
is distinct from ContextLM RL weights, whose release identity remains unresolved.

```python
from src.core.typed_decision import DecisionCandidate, DecisionRequest
from src.core.causal_decision import (
    CausalModelReference, CausalGeneration, decode_causal_decision,
    validate_generated_decision_result,
)

model = CausalModelReference(
    "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"
)
request = DecisionRequest(
    "r1", "Route this request",
    (DecisionCandidate("approve", "Approve"), DecisionCandidate("reject", "Reject")),
)
raw = CausalGeneration("r1", model, "approve", "stop", 8, 2)  # fixture usage
result = decode_causal_decision(
    request, raw, expected_model=model, max_input_tokens=8, max_output_tokens=2
)
assert result.selected_id == "approve"
validate_generated_decision_result(
    request, result, expected_model=model, max_input_tokens=8, max_output_tokens=2
)
```

Only exact immutable scalar leaves are shared; result/generation/model record
shells are detached from caller input. Revalidate when crossing another boundary:
Python's `object.__setattr__` can bypass frozen records. The result does not
authorize execution or establish semantic correctness of an allowed but wrong ID.
Revalidation checks current consistency and bindings; it cannot authenticate
what a provider previously emitted. An outer runtime must retain its own
original-observation and paid-prefix proof when temporal identity matters.

## Tests and extension boundary

`src/adapters/causal_choice.py` supplies the local text bridge:
`causal_messages(request)` serializes the original candidate order, and
`observe_causal_choice(...)` accepts a caller-owned eval model/tokenizer and
explicit token/EOS/vocabulary limits. It returns raw `CausalGeneration` before
strict selection. A caller must retain that raw value before passing it to
`decode_causal_decision`; rejection does not permit resampling. The caller owns
inference mode, a pristine generation configuration, asset verification and
resource admission. The module imports no numerical/model library itself.

The bridge checks one unpadded input sequence, unchanged generated prefix and
bounded integer IDs. It counts generated EOS as an output token, removes only
one final explicit EOS from decoded content, and preserves every other special
token and all whitespace. A full output cap without EOS reports `length`; an
early return without EOS reports `unobserved`, never assumed completion.

[Issue191's bounded qualification](../artifacts/runs/causal-native-selection-20261026/RESULTS.md)
has39literal checks passing on Windows and Linux, with scoped statics and
self-review. Its sole native attempt **failed before model loading**: nine
cached asset checks passed, then the protocol's tokenizer-EOS assertion failed.
The pinned tokenizer config maps `<|im_end|>` to248046 and `<|endoftext|>` to248044;
its EOS is248046 and pad is248044, while the model text config lists EOS248044.
These are distinct metadata fields. That mismatch must be resolved explicitly
in any future generation policy; it is not evidence of corrupt assets. No
model was loaded, no token was generated and no genuine selection is qualified.
The original failed attempt is preserved and will not be retried under that
allowance. This is separate from the accepted metadata-only core fixtures.

```powershell
python -m pytest -q tests/test_causal_decision.py
python -m ruff check src/core/causal_decision.py tests/test_causal_decision.py
python -m ruff format --check src/core/causal_decision.py tests/test_causal_decision.py
python -m mypy --explicit-package-bases --follow-imports=silent src/core/causal_decision.py tests/test_causal_decision.py
```

[Executed evidence](../artifacts/runs/causal-generated-decisions-20261026/RESULTS.md)
includes64cases/platform, two unchanged scalar-probability regressions and Linux
targettypechecks. Model libraries were not imported. No native generation,
tokenizer/template correctness, physical-work provenance, model quality or
release readiness is qualified by these tests.

An eventual qualified provider must verify genuine immutable assets and compatible pinned
runtime/tokenizer/template/hardware, observe actual stop/usage semantics and keep
TRAIN/probe/final authority separate. Capture failed returns and physical charges
before decoding; a `ValueError` grants no free resampling or budget renewal.
Freeze the small native protocol before any load/generation. Reuse existing
loader/budget/reference components; do not add another orchestration framework
or claim this causal baseline substitutes for trained ContextLM RL behavior.
