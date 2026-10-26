# Encoding-bound feature BP with explicit class-weighted CE

## Bounded schedule metadata checks

The bound BP array adapter uses a complete in-process schedule witness. Exact
mutable fields/types/dictionaries and immutable tuple/string ownership remain
checked at every operation. Full episode semantics and copied values/float bits/
string aliases are validated when owners are created; forks bind fresh witnesses.
The cached immutable payload remains bound, without recurring pickle. RNN behavior
is preserved. Private data replacement can refuse even if equal-valued: use public
fork/restore interfaces. No TTL, history chain or objective/weight change.

Final205 tests and fixed model-free timing pass:89%lower metadata-check median,
with2.2MB extra witness estimate and measured binding/setup cost on128episodes.
This is not a full runtime or scientific result.
[Commands/costs/failures](../artifacts/runs/weighted-schedule-witness-20261018/RESULTS.md),
[design](adr/ADR-0395-immutable-schedule-witness.md).

This adapter adds a distinct TRAIN objective to the existing bound BP owner. It
supports complete1..16-row raw296/K7 episodes and the existing native CPU policy.
It does not automatically change `train_batch`, schedule work, select weights or
run evaluation. Existing unweighted BP/PC/neutral/active behavior remains intact.

```python
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner, InvariantTorchFeaturePolicy,
)
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.adapters.torch_feature_weighted import train_weighted_episode
from src.core.demonstration_class_weights import episode_class_weights

learner = InvariantTorchFeatureLearner(
    InvariantTorchFeaturePolicy(
        TorchFeaturePolicy("bp", 296, 8, 17, num_classes=7)
    )
)
# episode is an already released complete TRAIN296 demonstration.
weights = episode_class_weights(episode)
diagnostic = train_weighted_episode(learner, episode, weights)
```

The existing formula is n/(present_classes*target_count), absent classes1. The
function also accepts explicit K-tuples validated by the existing bounded weight
validator. Keep weight construction TRAIN-only; do not choose it from probe/final
scores. Final/probe-role episodes are refused. The caller owns label release,
budgets and attempted-work accounting before invoking this function.

## Objective and publication

For projected public-history inputs, `logits=tanh(x W1+b1) W2+b2` and
`L=sum_i(w[target_i]*CE_i)/sum_i(w[target_i])`. Dividing by observed target-weight
mass makes absent-class weights irrelevant; all-one weights match ordinary BP
within the qualified float32 tolerance. Native SGD learning rate/momentum and
complete optimizer/gradient continuation are retained. The diagnostic identity is
`torch_invariant_feature_episode_class_weighted_cross_entropy_v1`.

The adapter restores a detached head/SGD candidate, runs forward/backward/step,
validates all candidate tensors and optimizer policy, then checks the original
owner/payload/policy, gradient metadata, complete numerical snapshot, episode,
Torch/NumPy/Python RNG and caller modes before one payload publication. Grad mode
must already be enabled; inference/no_grad callers are refused. Malformed terminal
next views are refused as complete evidence; next views/rewards never enter CE.

Normal failures publish nothing. A detected external donor/RNG change is refused
without erasing that foreign change. Failed candidate work still costs computation
and must be charged by the caller; live update/example clocks advance only on
successful publication. No snapshot/policy format changes are introduced. A
snapshot alone does not authenticate the historical objectives used to reach it;
retain the separately named diagnostic/receipts with the calling protocol.

## Verification and extension

```powershell
python -m pytest -q tests/test_torch_feature_weighted.py
python -m ruff check src/adapters/torch_feature_weighted.py tests/test_torch_feature_weighted.py
python -m ruff format --check src/adapters/torch_feature_weighted.py tests/test_torch_feature_weighted.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/torch_feature_weighted.py tests/test_torch_feature_weighted.py
```

The implementation's acceptance evidence is119 distinct scoped cases from an
expanded118-pass/1-fixture-failure run plus its corrected one-case pass; the
failed invocation remains recorded. Both mypy platforms and scoped Ruff pass.
See [commands, costs and failures](../artifacts/runs/weighted-feature-bp-20261017/RESULTS.md)
and [ADR-0392](adr/ADR-0392-explicit-weighted-feature-bp.md).

## Existing sequential array and delayed stream port

```python
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner
from src.core.demonstration_episode import episode_arrays

# released_episodes is an exact tuple of complete TRAIN raw296/K7 records.
weighted = TorchTrainWeightedLearner(learner.policy, released_episodes)
for episode in released_episodes:
    diagnostic = weighted.train_batch(*episode_arrays(episode))
```

The adapter preflights the entire owned schedule before native construction and
binds every array update to its next episode. It derives the existing weights;
no external weight selector is added. Fork/restore/deepcopy preserve the complete
native state and schedule prefix. Prediction works; BP refuses recurrent sessions.
Detached candidate work never publishes over an observed donor/input/RNG change.
The caller still owns label release, attempts, computation costs and budgets.

The existing `run_attribution_stream` consumes this adapter without a new runner.
52 new cases plus40 RNN and37 native weighted cases pass in a final129-test gate,
including three real native delayed updates/nine rows with unchanged source/actor
and zero score forwards. These use synthetic legal public frames, not game physics
or a comparative learning result. Initial failures remain recorded.

```powershell
python -m pytest -q tests/test_weighted_feature_stream_adapter.py tests/test_torch_train_weighted_learner.py tests/test_torch_feature_weighted.py
python -m ruff check src/adapters/torch_train_weighted_learner.py tests/test_weighted_feature_stream_adapter.py src/adapters/minigrid_weighted_bc_pilot.py
python -m ruff format --check src/adapters/torch_train_weighted_learner.py tests/test_weighted_feature_stream_adapter.py src/adapters/minigrid_weighted_bc_pilot.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/torch_train_weighted_learner.py tests/test_weighted_feature_stream_adapter.py src/adapters/minigrid_weighted_bc_pilot.py
```

Both mypy target platforms pass; actual Linux execution remains untested. See
[stream evidence](../artifacts/runs/weighted-feature-stream-20261018/RESULTS.md) and
[ADR-0393](adr/ADR-0393-weighted-feature-stream.md). The explicit opt-in
route is described below; next freeze a small matched protocol. Preserve
unweighted fit metrics and isolated probes. No comparative result, new dependency,
environment variable or durable format is introduced by this integration.

## Opt-in MiniGrid learnability route

The existing CLI accepts the distinct weighted benchmark objective explicitly:

```powershell
python -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --weighted-bp --distinct-probe --old-task-readouts --return-to-a --describe
```

The API equivalent is `minigrid_learnability.describe_protocol` or `run_pilot`
with `public_history=True, invariant_history=True, qualification_method="bp",
weighted_bp=True`. Keep `--describe` while freezing a prospective comparison;
omitting it runs the existing bounded native case. Invalid routes fail before
resource construction. The flag is off by default; no PC/CPC objective changes.

The same128 encoded TRAIN packets use fixed current-episode weights and named
weighted receipts. Raw fit remains ordinary unweighted CE/accuracy/exact sequence.
Each isolated fresh-task probe starts through a separate ordinary BP template,
proves equality to the original zero-update source, and privately restores the
selected full checkpoint/optimizer. Probe SGD and optional A return remain
unweighted. Original source/actor guards, permanent costs and failed evidence stay
active. Previously observed metrics and optimizer history can confound plasticity
interpretation; there is no learning-benefit or circadian-advantage guarantee.

See [ADR-0394](adr/ADR-0394-weighted-learnability-route.md) for the boundaries.


Route verification commands:

```powershell
python -m pytest -q tests/test_minigrid_weighted_feature_route.py tests/test_minigrid_bound_qualification.py tests/test_minigrid_bound_qualification_cli.py tests/test_minigrid_prior_gradient_route.py tests/test_minigrid_invariant_learnability_cli.py
python -m ruff check src/adapters/minigrid_weighted_feature_route.py src/adapters/minigrid_learnability.py scripts/run_coupled_attribution.py tests/test_minigrid_weighted_feature_route.py
python -m ruff format --check src/adapters/minigrid_weighted_feature_route.py src/adapters/minigrid_learnability.py scripts/run_coupled_attribution.py tests/test_minigrid_weighted_feature_route.py
python -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/minigrid_weighted_feature_route.py src/adapters/minigrid_learnability.py scripts/run_coupled_attribution.py tests/test_minigrid_weighted_feature_route.py
```

[Route evidence and honest composite qualification](../artifacts/runs/weighted-learnability-route-20261018/RESULTS.md):
153 distinct cases through152 unchanged passes plus one corrected native case,
not one all-green153 invocation. Both mypy platforms pass; actual Linux and full
optional native F/A-return composition remain unexecuted in this increment.


## First native recurring-task attempt

The [frozen seed17 recurring comparison](../artifacts/runs/matched-weighted-bp-20261018/RESULTS.md)
remains incomplete/SPENT. Ordinary BP completes192 updates; weighted completes128
TRAIN, initial F16 and two A-return updates, then stops at the original body cap.
All raw exact TRAIN traces remain0. Inputs/initial predictions match and saved
arithmetic validates, but missing later F/A work prevents a qualified full pair.
No retry, tuning, useful-skill or circadian-advantage claim. Full source/actor and
reached budget/state guards remain in the records.

A separate model-free profile measures the existing two full-schedule pickle
checks at29.23ms median on a6.2MB schedule; measured combined metadata31.93ms.
That is not a full runtime causal attribution or a speedup. Next reduce this
specific cost with complete immutable-field/tuple ownership proof and refusal
regressions, before any future independently reviewed native protocol.
