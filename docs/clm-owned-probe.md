# Disposable CLM head probe

[Reserved temporal integration](clm-reserved-probes.md) now composes this accepted
owned boundary through the existing stream with separate consent and budgets.
The software evidence below retains its original scope.

## Scope

An explicit reserved-task boundary for the existing frozen CLM head learner.
It copies warm weights into a separate owner with a fresh AdamW/OneCycle schedule.
It does not alter the source/deployed/benchmark learner, encode text, choose a
fresh task, score labels, promote a model or renew the source's spent work.
The owned boundary is implemented and97 focused checks pass on each platform;
scoped statics and independent software review pass. No native probe study is
authorized or claimed by this software increment.

```text
src/core/clm_probe.py             exact reserved-role frozen-feature declarations
src/adapters/clm_head_probe.py    disposable native probe owner and finite readouts
src/adapters/clm_head_learner.py  existing paired engine/strict normal admission
tests/test_clm_head_probe.py     tiny literal ownership/role/failure behavior
```

## API and roles

```python
from dataclasses import replace
from src.adapters.clm_head_probe import CLMHeadProbe
from src.core.clm_probe import FrozenProbeEmbeddingBatch, FrozenProbeChoiceBatch
from src.core.clm_head import CLMTargets

# source is an exact healthy CLMHeadLearner; features are already-bound CPUfloat32.
policy = replace(source.policy, max_updates=2, max_restores=0)
probe = CLMHeadProbe(source, policy=policy, max_readouts=3)
training = FrozenProbeEmbeddingBatch(source.identity, train_states, train_actions)
readout = FrozenProbeChoiceBatch(source.identity, test_states, candidate_actions)
probabilities_before = probe.predict_choices(readout)
probe.train_batch(training, CLMTargets(train_group_codes))
probe.train_batch(training, CLMTargets(train_group_codes))
probabilities_after = probe.predict_choices(readout)
# Keep probe_test labels in application scoring after the fixed training dose.
# Discard probe; its weights have no public snapshot/export/install API.
```

Paired update records admit only probe_train; readout records admit probe_train,
probe_test or outer_selection (old-task development), never final roles. Normal
CLMHeadLearner rejects probe records and still accepts its original TRAIN kinds.
A square readout is still not a paired update. The generic float64 probe is not
coerced into this float32paired interface. Feature provenance remains the caller's
responsibility; records do not attest encoder identity or empirical split independence.

## Native policy and costs

Declare1..16probeupdates except10 and1..64readouts. Other CLMHeadPolicy fields
must exactly preserve source settings; max_restores must be0. Horizon10 is
rejected before copies/nativeconstruction because unchanged OneCyclepct_start.1
has a zero-length initial phase. Native paired symmetric group-masked loss,
clipping1, AdamW, schedule and cosine choice readout are reused unchanged.

Warm source weights, scale, modes and sourceRNG are copied. Probe begins with
no gradients/moments and zero progress; source optimizer/gradients/schedule/RNG/
physical work stay unchanged. This is fresh-optimizer adaptation, not continuation.
Count head copies, source comparison captures, fixed new optimizer initialization,
probe row exposures, forwards/backwards/readouts, memory/time and any failures
separately from source work. Failed paid work poisons the probe; no restore/fork
method can reset its allowances. No unlimited state or success-record history.

Read-only identity/policy/work copies and source progress scalars are available;
returned probability tensors are detached copies. Private internals remain trusted
Python and are not a security sandbox. A future temporal integration must use
separate consent/data roles/budgets and keep probe results out of training,
promotion, replay, tuning and final evaluation.

[Decision rationale](adr/ADR-0540-owned-clm-probe-roles.md) ·
[Protocol and gates](../artifacts/runs/clm-owned-probe-20261022/PROTOCOL.md).

## Executed checks and next extension

The literal fixture starts from a spent source (2updates/4rows plus one paid setup
restore/generation3). Its disposable probe makes2updates/4rows and3readouts while
source full native state/work stays unchanged. Injected lost returns keep costs
and poison only the probe. Exact normal admission/defaults, source/RNG/gradient/
optimizer separation and early horizon10 refusal are checked alongside existing
rectangular readout and twelve-update temporal regressions.

97PASS/platform; Windows19.295381s/WSL32.660448s. Static gates pass after one
negative-test local union annotation; original failed mypy and both original
passing numerical receipts retained. Independent byte reversal proves the
annotation adds no runtime change. BOTH130headpairs/106paidrows/38backward/
144forwards/408projectedrows; zero pretrained encoder, random parameter draws,
Transformers, downloads or simulator interactions. Source comparison captures
are bounded tensor copies included in memory/time, not free model initialization.
No native task or comparative benefit follows from these tiny checks.

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_clm_temporal_sequence.py tests/test_clm_choice_readout.py tests/test_clm_head_probe.py
.venv/Scripts/python.exe -m ruff check src/adapters/clm_head_learner.py src/core/clm_probe.py src/adapters/clm_head_probe.py tests/test_clm_head_probe.py tests/test_clm_choice_readout.py tests/test_clm_temporal_sequence.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/clm_head_learner.py src/core/clm_probe.py src/adapters/clm_head_probe.py tests/test_clm_head_probe.py tests/test_clm_choice_readout.py tests/test_clm_temporal_sequence.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent src/adapters/clm_head_learner.py src/core/clm_probe.py src/adapters/clm_head_probe.py tests/test_clm_head_probe.py tests/test_clm_choice_readout.py tests/test_clm_temporal_sequence.py
```

Next integrate a default-off reserved-task plan into the existing CLM temporal
runner with separate probe consent/budgets/inbox receipts and fixed new-task/
old-task readouts. Begin with literal integration checks; freeze/review native
texts, checkpoints, dose, comparisons and costs before any model case. Keep
probe_test labels and all probe results out of deployed/benchmark updates,
promotion, replay, tuning and final evaluation. No new orchestration framework.
[Full receipts and failures](../artifacts/runs/clm-owned-probe-20261022/RESULTS.md).
