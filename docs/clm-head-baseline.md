# Frozen-feature CLM head learner

`src.adapters.clm_head_learner.CLMHeadLearner` implements the pinned conventional
Contrastive-LM paired objective over frozen state/action embeddings. It fits the
existing `NativeLearner` port and `ExperienceInbox`; no separate training runner
or queue is required. This optional adapter requires an existing Torch runtime.

Recorded numerical checks pass, but cumulative correctness use was 132 logical
head pairs versus 128 allowed. The resource gate failed; the R5.3-B qualification
remains unfinished. Original failure/repair/warnings are preserved in
[results](../artifacts/runs/clm-head-baseline-20261022/RESULTS.md). No further run
or retrospective budget renewal is implied by the commands below.

## Inputs and outputs

Declare `FrozenEncoderIdentity(model_id, revision, tokenizer_revision, pooling,
recipe, embedding_dim)`, then supply a `FrozenEmbeddingBatch(identity, states,
actions)` of finite CPU float32 paired NxD tensors and `CLMTargets((group_codes...))`.
Only TRAIN is admitted. Identity is a trusted declaration, not proof that a
particular encoder produced the tensors. Each batch has 2..64 paired rows, further
bounded by policy. Same-group off-diagonal negatives are excluded.

Inject separate deterministic native state/action heads and a scalar log scale.
`CLMHeadPolicy` fixes projection width, update horizon, learning rate, weight decay,
parameter/row limits and restore allowance. Training owns copies, preserves caller
features/modules/RNG, and returns the existing `TrainingDiagnostic`. Updates use
AdamW, clipping at 1 and cosine OneCycle with the native .1 warmup fraction.
The upstream horizon10 singularity remains a loud refusal; no alternate schedule
is silently substituted. Stochastic modules and buffers are unsupported.

`predict(batch)` recomputes normalized projections and returns detached CPU NxN
candidate probabilities. It does not reuse projections across updates. A serving
integration should bind caches to both the encoder identity and `generation`.
This paired-batch interface does not implement the native server's arbitrary
request schema or distinct state/candidate row counts.

`snapshot_state()` captures weights, scale, gradients, AdamW, scheduler, module
modes, private CPU RNG and logical counters. `restore_state(snapshot)` validates
the fixed recipe and publishes staged state. Physical attempts and restore caps
are permanent; restoring old state does not renew the update horizon. An uncertain
native failure stops that owner and retains paid work. Snapshots are trusted-local
objects, not durable files or a secure loader for untrusted Python modules.

## Fixed engineering example

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OMP_NUM_THREADS = '2'
.venv/Scripts/python.exe -X utf8 -B scripts/run_clm_head_baseline.py --output clm-fixture.json
```

The example compiles only the two inspected pinned function definitions from the
local source artifacts, creates one seed23 tiny head pair, and sends four delayed
TRAIN packets through the existing inbox. It compares all heads, scale, gradients,
optimizer and scheduler state with the native update. Existing output is refused;
ordinary failures retain an incomplete report and paid prefix.

This example is **not** a trained encoder/head checkpoint, useful task learning,
consolidation, native serving qualification or a scientific comparison. Production
metadata names a 4096-dimensional frozen encoder and 512-dimensional projection;
the fixture uses 2-dimensional inputs/outputs solely to check implementation.

## Focused checks

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_clm_head_learner.py
.venv/Scripts/python.exe -m ruff check src/core/clm_head.py src/adapters/clm_head_learner.py scripts/run_clm_head_baseline.py tests/test_clm_head_learner.py
.venv/Scripts/python.exe -m ruff format --check src/core/clm_head.py src/adapters/clm_head_learner.py scripts/run_clm_head_baseline.py tests/test_clm_head_learner.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent src/core/clm_head.py src/adapters/clm_head_learner.py scripts/run_clm_head_baseline.py tests/test_clm_head_learner.py
```

Use the prospective resource protocol for numerical qualification; these commands
do not grant repeated experiment allowance. Recorded gates and costs live in
[the stage](../artifacts/runs/clm-head-baseline-20261022/PROTOCOL.md).

**Why this:** the conventional head update was missing while retrieval and local
linear feedback already worked. Qualify that baseline before adding consolidation
mechanisms. Next connect the exact released encoder/head and bound embedding
recipe, then test useful acquisition/retention/return under a frozen small protocol.
