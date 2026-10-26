# Rectangular frozen CLM choice readout

`CLMHeadLearner.predict_choices` ranks M candidate embeddings for N query
embeddings, with N and M independent. It enables the existing temporal-rule
corpus's four development queries against three active candidates. It uses the
same owned native heads, normalized projections, capped exponential scale and
row softmax. No encoder call, training, promotion or projection cache is added.

## Interface

```python
from src.core.clm_head import FrozenChoiceBatch

batch = FrozenChoiceBatch(
    identity=learner.identity,
    states=query_features,       # CPU float32 N x encoder dimension
    actions=candidate_features,  # CPU float32 M x encoder dimension
    role="outer_selection",
)
probabilities = learner.predict_choices(batch)  # detached CPU float32 N x M
```

Queries may have1..policy.max_rows rows, candidates2..policy.max_rows. Both
matrices must be finite and match the fixed encoder width and declared identity.
Outputs preserve query and candidate order; each row sums to one over supplied
candidates. Reordering candidates reorders columns. Callers bind columns to the
matching candidate IDs and decide how to report ties. Output mutation cannot
change learner state. Changing heads recomputes projections on every call.

Choice records admit only TRAIN diagnostics or `outer_selection` development.
They contain no labels and cannot enter `train_batch` or old paired `predict`,
even with square shapes or `role="train"`. `FrozenEmbeddingBatch`, paired native
training, objective, final projection width and old prediction defaults remain
unchanged. Probe/final roles are not accepted. Raw arrays and declared identities
remain a trusted caller boundary; role names do not attest data provenance.

Readout validates before charging work, then uses the existing staged state,
private CPU RNG, eval/no-grad and original-state checks. Native failures retain
attempts and poison that owner. Aggregate prediction attempts/returns/queryrows
include both APIs. Additional `choice_prediction_attempts`,
`choice_prediction_returns`, `choice_candidate_rows_attempted` distinguish choice
work and remain permanent across restore. A native return can be counted even
when a later source-state check refuses publication; inspect the unknown-work
flag and exception as well as counts.

## Existing corpus integration

Use `temporal_examples(phase, "outer_selection")` from
`src.core.temporal_decision_sequence`. Its ordered requests supply four query
rows, while `examples[0].request.allowed_candidates` supplies three active
candidate columns. Build vectors from the exact text/instructions/encoder recipe
for those records; an ID alone is not an embedding-cache identity. Keep
observation-bound TRAIN target mappings separate from development readout.

The paired update still supplies equal state/action counts with matching native
group codes. This readout does not convert categorical labels into a new training
objective. A later temporal experiment must freeze the corpus, dose, comparison,
metrics and costs before running; keep development scoring after all training.

## Focused verification

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_clm_choice_readout.py tests/test_clm_head_learner.py::test_should_recompute_fixed_dimension_probabilities_without_owning_the_callers_output
.venv/Scripts/python.exe -m ruff check src/core/clm_head.py src/adapters/clm_head_learner.py tests/test_clm_choice_readout.py
.venv/Scripts/python.exe -m ruff format --check src/core/clm_head.py src/adapters/clm_head_learner.py tests/test_clm_choice_readout.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent src/core/clm_head.py src/adapters/clm_head_learner.py tests/test_clm_choice_readout.py
```

The source-owned fixture integrates actual temporal records using tiny literal
features. This is interface evidence, not native semantic adaptation. Existing
8B known-TRAIN acquisition/retention remains bound to its original executed
sources; no encoder reload is authorized by this component's engineering
allowance. Prior132/128resourceFAIL, no newrefund acquisition/no recovery and
all unfinished R5.1/R5.3/G5/scientific/release requirements remain open.

**Why this:** rectangular readout is the concrete missing learner interface for
the existing changing-tool/rule corpus. It keeps label-bearing paired learning
distinct from candidate ranking, without building another runner.

## Verified scope

Both Windows and WSL passed 25 checks: 24 new cases and one selected old square
regression. Scoped lint/format/type checks passed without a source repair or
fixture repeat. The fixed engineering allowance includes all reference work and
parameter-free oracles. No pretrained encoder was loaded. These checks qualify
the numerical/readout boundary, not released temporal task performance.
[Results and costs](../artifacts/runs/clm-rectangular-readout-20261022/RESULTS.md).

```text
src/core/clm_head.py                 paired and choice declarations
src/adapters/clm_head_learner.py      owned updates and both readout APIs
tests/test_clm_choice_readout.py      rectangular behavior and isolation
```
