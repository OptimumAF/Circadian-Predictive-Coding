# Public-cue native neural baseline

`ContextTorchFeatureLearner` routes causal raw296 MiniGrid history to two owned
instances of the existing native neural learner. The initial visible blue/red
cue selects a head; canonical296 values enter that head. Task names, targets,
future views and private environment state do not select it.

```python
from src.adapters.minigrid_context_torch_feature_learner import (
    ContextTorchFeatureLearner, ContextTorchFeaturePolicy,
)
from src.adapters.torch_feature_learner import TorchFeaturePolicy

# Construction and all training/readout work require a prospective allowance.
learner = ContextTorchFeatureLearner(
    ContextTorchFeaturePolicy(TorchFeaturePolicy("bp", 296, 32, 17, num_classes=7))
)
# learner.train_batch(raw_history, labels)  # Homogeneous initial cue only.
# learner.predict(raw_history)             # Mixed cues preserve row order.
# saved = learner.snapshot_state()         # Complete detached continuation.
```

Methods `bp`, `pc`, `neutral_cpc`, and `gated_cpc` use the same two-head geometry
and native seeded parameter initialization. Their existing objectives, optimizer,
relaxation, chemistry, unit histories and split RNG remain native. Equal capacity
or exposure does not imply equal computation or a scientific advantage.

Why this: the older neural representation pooled both cues. An explicit existing
public-cue route tests a concrete learning/control gap without inventing another
objective, widening the native feature policy or changing old defaults. The
additional head is additional capacity; old pooled results are not causal controls.

Every snapshot contains both full native snapshots, including weights, momentum,
gradients, adaptive state and RNG. Global update/row counts sum the head counts;
an inactive head is unchanged. Fork and restore copy data without constructing
models. Actual native operations stage the original constructor/restore/train
ports, and those repeated initializations and draws are paid work. A failed staged
attempt preserves the preceding complete state plus one bounded scalar record
of attempted/returned work and possible unobserved work. That owner refuses
continuation and restoration that would erase the failure. No success chain,
cached labels, role/budget authority or unbounded state history is added.

## Fixed conventional BP workflow

The existing control module and evaluator expose one fixed known-A gate:

```python
from scripts.run_minigrid_linear_fit_control import read_training_sources
from src.infra.minigrid_affine_example import example_affine_training
from src.adapters.minigrid_context_control import (
    describe_neural_a_protocol, run_neural_a_control,
)

protocol = describe_neural_a_protocol()  # Metadata only.
# Execute only under the separately declared local allowance:
# with example_affine_training() as path:
#     report = run_neural_a_control(read_training_sources(path)[:2])
```

Original packaged Ablue/red TRAIN sources enter the existing delay-one
AttributionStream: fixed128 alternating nine-row updates,1152 exposures,
64updates/576rows per head, BP hidden32/seed17/LR.03/momentum.9. No fitting follows
the detached TRAIN monitors at0/32/64/96/128. These ten calls score ninety rows
after learning has ended; they cannot choose training, dose or checkpoints.
Only the complete original128-receipt endpoint enters the original native loop
for Ablue/Ared, branch3/seed17/max16steps/all7actions/firstargmax. Both games use
known TRAIN layouts. Native success and imitation scores are separate outcomes.

The unchanged native0 source and actor remain separate from the candidate.
Whole300s/workstop290/768MiB, TRAIN180s/128updates/no-replay and separate
native150s/0updates/0replay clocks cannot renew each other. Preceding reports,
protocol, full states and budget bindings are checked before admitting more
work. Failed and incomplete reports retain paid prefixes and unknown work.
The optional `endpoint_observer` receives one detached full data snapshot after
validating all128 causal receipts, native/head counters, policy and failed-work
state, BEFORE fallible stream finalization. This is a paid data prefix, not an
eligible game endpoint. `paid_prefix.status` stays `unqualified`; the separate
`finalization` field reports `not_started`, `running`, `complete` or `failed`.
Only a complete unchanged finalization admits `native_endpoint`, monitoring and
games. Callback failure, replaced runtime, changed receipts or failed finish
stop the workflow while the observer can retain its owned data and the report
preserves the validated receipt/prefix metadata. No clock, consent, promotion or
deployment authority is exported. Ordinary case-specific saving needs no new
persistence framework; one prefix replaces an unlimited success-record chain.

The first actual qualification failed during stream admission, before any update
or game: ActorShadow rejects any replay ceiling, including zero. The new TRAIN
composition now omits that unsupported budget field while retaining the fixed
no-consolidation path and a guard requiring the permanent replay charge to remain
zero. A real-stream regression exercises the exact H32 workflow and first two
delayed updates. The original actual failure and protocol remain frozen; the
original failed invocation is retained. A prospectively amended single engineering
re-entry kept all work/scientific caps cumulative, including the first invocation's
time and initialization. It completed128updates/1152rows, then failed because the
composition requested a snapshot from the sealed stream. No scores or games ran,
and the full endpoint was not saved; it cannot be reconstructed or refitted within
the spent allowance. Receipt/checkpoint summaries and both failures remain frozen.

The lifecycle repair binds the original runtime identity and uses its existing
read-only candidate snapshot after transport finalization. Full state and counter
comparison continues; the stream stays sealed and replaced runtimes are refused.
The regression now uses actual stream finalization with a scripted counter learner,
which proves lifecycle composition but provides no learned/native-game evidence.
No third training/native entry is permitted by this increment.

The subsequent paid-prefix ordering repair is checked without native models:
actual stream finish failures before and after sealing, observer failure and
detachment, exact receipt/clock/policy/failure refusals, and mutation at the
callback/finalization boundary. It cannot restore the already-lost endpoint or
convert the historical failed run into scored/native evidence. Its separate
software protocol/evidence is in `artifacts/runs/paid-neural-prefix-20261024/`.

The two input templates contain18rows/42,768float64 array bytes. Repeated training
allocates128batches/2,737,152input array bytes, plus the existing stream/copy/native
and report overhead. At most five local full checkpoints support the fixed
monitors; the returned report contains JSON summaries, not tensor history.
Optional native dependencies use the existing pinned MiniGrid and Torch setup.
No new dependency, environment variable, CLI flag or configuration layer is added.

## Verification and evidence

Focused files are `tests/test_context_torch_feature_learner.py` (32numerical and
isolation cases) and `tests/test_context_neural_control.py` (20orchestration and
native-state admission cases). Final targeted coverage is52distinct cases, including
affected repair checks after the actual failure. Native fixtures count60updates/
76rows per platform, including references, deliberately failed calls, disposable
probe work and the real workflow's two-update regression. Scripted/literal cases
perform no optimizer updates; all software checks perform no real physics.
Source/procedure/results receive self-review, not independent
acceptance. See the frozen protocols and result record in
`artifacts/runs/context-neural-baseline-20261024/` and
[issue125](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/125).

Use scoped Ruff/format and fresh mypy on the three affected adapters and these
two tests. The finite stage launcher preserves both platform costs; rerunning
unchanged numerical fixtures is unnecessary. A future method-specific workflow
must declare its own scientific protocol and endpoint geometry, use these native
ports, retain all negatives and keep private/final evaluation unavailable to
training, stopping, tuning and probe selection.

Known-task usefulness, broader acquisition/retention/return, heldout layouts,
matched comparative science and release readiness remain separate gates. This
route does not establish a circadian advantage. Earlier pooled neural,
Perceptron, private and timed-out results remain unchanged.
