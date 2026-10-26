# Released CLM frozen features and typed choices

The optional CPU path connects the pinned Qwen3-8B embedding backbone and released
CLM projection heads to `DecisionRequest` / `ProbabilityDecisionResult`. It can
also produce paired TRAIN features for the existing CLM head learner. It does not
train the encoder or create a separate stream runner.

## Modules

```text
src/adapters/clm_frozen_encoder.py   bounded token recipe and frozen pooling
src/adapters/clm_decision.py         native head factory and typed readout
src/infra/clm_local_loader.py        verified local assets and lazy native loading
scripts/run_clm_frozen_choice.py     one fixed CPU qualification request
tests/test_clm_frozen_encoder.py     fake-backend and asset-refusal tests
```

Core identity/schema types stay independent of Torch and Transformers. The outer
script supplies the head factory to the local loader, then composes the returned
frozen model/tokenizer and head pair with adapters. The loader downloads nothing,
uses no remote Python, and refuses missing, wrong-size or replaced pinned assets.
The separate preparation helper follows a prospective transfer allowance; model
assets are held outside the repository and not bundled in the demo package.

## Recipe and limits

This is an explicit **HF CPU plain-text TRAIN recipe**, separate from native
vLLM serving parity. State text appends instructions after a blank line and keeps
the token tail. Candidate descriptions keep their token heads. Tokenization adds
no special tokens, falls back to a blank for empty text, and retains at most
`max_tokens-1` tokens (2047 under the pinned2048 recipe). Manual right padding
and attention masks select the last nonpadding hidden state. Float32 features
are divided by their norm plus1e-12, then the native frozen heads project and
normalize them. Capped exponential log scale and insertion-order softmax/margin
confidence reproduce the selected CLM schema.

Encoder model ownership is transferred from the loader; a16GiB model is not
deep-copied. Native parameters remain frozen/eval. Tokenizer and model CPU RNG
effects are isolated. Bounded one-slot readouts retain paid token/feature facts;
they do not form a history or projection cache. Attempts cannot be refunded;
an uncertain native failure stops that owner. Supplied Python/tokenizer/model
code is trusted, and access is synchronous. These checks do not sandbox arbitrary
Python or physically attest a declared encoder identity.

`FrozenCLMEncoder.training_batch(states, actions, role='train')` admits2..8 paired
plain strings and returns the existing `FrozenEmbeddingBatch`. NonTRAIN roles
are refused before tokenization. `CLMDecisionPredictor` accepts the existing typed
request, owns small frozen head copies and returns provider probabilities without
inventing calibration or unreported native token-usage fields. Physical work is
reported separately. Head changes require a new frozen predictor; projections
are recomputed on every readout.

## Run using prepared local assets

Use the existing optional WSL environment; no dependency installation is needed
in the existing local environment. Windows' default NumPy environment does not
include Transformers. CPU BF16 is explicit; this path does not offload or quantize.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
PYTHONPATH=. /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python \
  -B scripts/run_clm_frozen_choice.py \
  --model-dir /mnt/c/Users/Avery/.cache/circadian/clm-frozen-encoder-20261022/encoder \
  --head /mnt/c/Users/Avery/.cache/circadian/clm-frozen-encoder-20261022/CLM_v0.1-8B.pt \
  --output clm-frozen-choice.json
```

Existing result files are refused. The fixed request is checked against the
pinned token recipe, raw paid hidden states, normalization and native head/schema
decode. A wrong workflow is retained; it does not trigger another request or
tuning. The prospective native allowance is one encoder forward, four sequences,
96 tokens/sequence and no optimizer updates. Reproduction commands do not grant
another experiment allowance. Model load attempts and failures remain observable
in partial output; wrapper/process termination evidence is recorded separately.

## Observed local qualification

The released CPU request completed and passed source/token/reference/resource
checks, but selected **refund** for the password query (probability0.804). That
negative task result is preserved. It does not establish useful adaptation or
vLLM serving parity. The numerical path is qualified only at this local scope.
See [results and costs](../artifacts/runs/clm-frozen-encoder-20261022/RESULTS.md).

## Focused verification

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_clm_frozen_encoder.py
.venv/Scripts/python.exe -m ruff check src/adapters/clm_frozen_encoder.py src/adapters/clm_decision.py src/infra/clm_local_loader.py scripts/run_clm_frozen_choice.py tests/test_clm_frozen_encoder.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/clm_frozen_encoder.py src/adapters/clm_decision.py src/infra/clm_local_loader.py scripts/run_clm_frozen_choice.py tests/test_clm_frozen_encoder.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent src/adapters/clm_frozen_encoder.py src/adapters/clm_decision.py src/infra/clm_local_loader.py scripts/run_clm_frozen_choice.py tests/test_clm_frozen_encoder.py
```

Fake checks do not establish released-model execution. See the separate
[prospective protocol](../artifacts/runs/clm-frozen-encoder-20261022/PROTOCOL.md)
and recorded review for actual acceptance. The preceding head-training stage's
132/128 resource failure remains preserved. Full R5.1/R5.3, vLLM serving, useful
adaptation, calibration, circadian advantage and release claims remain open.

**Why this:** useful native CLM learning requires features from the exact released
encoder/head rather than a tiny synthetic substitute. After this boundary is
qualified, use the existing learner/inbox for a small frozen adaptation protocol.
