# Explicit public nuisance-invariant MiniGrid control

This opt-in representation removes wall-object color and absolute compass
direction for the declared egocentric cue/key/door task. It preserves the full
pixel order and every other decoded object/color/state coordinate, including
cue and key colors, door state, and goal. It reads no context, layout, teacher,
label, future input or private state. This is a task-specific hypothesis, not a
general guarantee for other games where wall color or direction may matter.

The raw public-history encoder and original exact/FIFO learners remain available
with their original defaults. The new count wrapper uses a distinct policy and
encoding identity; it never silently restores raw native snapshots. No saved
baseline state is converted or retrained by this module.

## Modules and boundaries

```text
src/core/minigrid_invariant_encoding.py
  public frame/history projection and causal owned episode conversion
src/adapters/invariant_action_memory_learner.py
  explicit encoding-bound policy/state; composition of existing native counts
tests/test_minigrid_invariant_encoding.py
  pure invariance, information preservation, ownership and causal reset
tests/test_invariant_action_memory_learner.py
  state admission, atomic updates, isolated probe and delayed TRAIN stream
```

Core validates148 normalized coordinates against discrete public integer codes
(object0..10/color0..5/state0..2 and direction0..3), allowing at most1e-8 decoded
roundoff. Wall object2 gets grey color5; direction becomes0. Other decoded values
are reencoded deterministically, so repeated projection is idempotent. Invalid
episode reset clears the preceding initial frame; invalid current input does not
consume a different episode's history. Complete original148-frame episodes are
validated before projection can hide a discontinuity. Roles/IDs/actions/rewards/
terminal values survive owned episode conversion; conversion grants no role release.

The adapter accepts raw296 first/current history float64 arrays,1..128 rows. It
projects every row before an existing native train or predict call. Exact memory
retains atomic capacity refusal; FIFO retains insertion-order retirement and its
native lifetime/retired exposure accounting. Counts remain storage updates, not
SGD or a shared loss. Source learners, checkpoints and budgets remain the existing
app helper/stream's responsibility. No environment, persistence codec, file IO,
serving authority or new runner is added.

## Usage and configuration

```python
from src.adapters.invariant_action_memory_learner import (
    InvariantActionMemoryLearner,
    InvariantMemoryPolicy,
)
from src.core.fifo_action_memory import FifoActionMemoryPolicy

exact = InvariantActionMemoryLearner()
fifo = InvariantActionMemoryLearner(InvariantMemoryPolicy(
    FifoActionMemoryPolicy(feature_dim=296, num_classes=7,
                          max_entries=64, max_payload_bytes=155136)
))
# x is raw public first/current history, shape(B,296), float64.
# y is hard action indices0..6, shape(B,1), float64.
# exact.train_batch(x, y); probabilities = exact.predict(x)
# owned = exact.snapshot_state(); child = exact.fork()
```

Use raw `PublicInitialFrameHistory` to produce wrapper input; the wrapper owns
projection. The separate pure `InvariantInitialFrameHistory` and
`encode_invariant_history_episode` functions expose canonical features for other
explicitly identified future controls. Existing neural/raw snapshot protocols do
not acquire a new encoding identity merely because their width is296.

`InvariantMemoryPolicy` binds the native control/cap and fixed
`minigrid_public_wall_direction_invariant_history_296_v1` identity. Snapshots bind
that complete policy and native state; restore validates exact types/fields,
geometry, counters, retirement and canonical keys. Raw, wrong-cap/control/encoding
and noncanonical states refuse atomically. Forks own their complete state.
No environment variables or dependencies are introduced.

The retained table costs2424 logical bytes/key for296 features/seven counts.
FIFO's16 retirement metadata bytes, encoding identity, outer policy/state and
Python/copy overhead are extra; this is not equal whole-model memory to raw
controls. Projection allocates a new2368-byte numeric feature row; decoded
temporary/Python allocations are additional. Explicit history keeps one1184-byte
initial numeric frame. No measured game latency or memory advantage is claimed.

## Verification and remaining work

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest tests/test_minigrid_invariant_encoding.py tests/test_invariant_action_memory_learner.py tests/test_minigrid_public_history.py tests/test_action_memory.py tests/test_fifo_action_memory.py -q
.venv/Scripts/python.exe -X utf8 -B -m ruff check src/core/minigrid_invariant_encoding.py src/adapters/invariant_action_memory_learner.py tests/test_minigrid_invariant_encoding.py tests/test_invariant_action_memory_learner.py
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check src/core/minigrid_invariant_encoding.py src/adapters/invariant_action_memory_learner.py tests/test_minigrid_invariant_encoding.py tests/test_invariant_action_memory_learner.py
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/core/minigrid_invariant_encoding.py src/adapters/invariant_action_memory_learner.py tests/test_minigrid_invariant_encoding.py tests/test_invariant_action_memory_learner.py
```

Repeat the type command with `--platform linux`; this is not a Linux runtime gate.
Final226 focused cases and scoped statics pass:58 new plus168 existing raw history/
count checks. Both native controls run through the existing isolated probe and
delayed TRAIN stream on array fixtures, preserving source state/budgets/labels.
Fixture nuisance-invariance is implemented behavior, not actual game transfer,
independent confirmation, useful neural adaptation or circadian benefit.
[Evidence, failures and exact costs](../artifacts/runs/minigrid-invariant-encoding-20261017/RESULTS.md).

## Saved A32 recipe and fixed A/E readout

The opt-in recipe now composes the existing TRAIN arm with this bound learner.
It validates the original saved raw0/A32 states and A-blue/A-red demonstrations,
binds the reused raw counts to the exact32-packet A dose, then trains one fresh
invariant exact control for32 updates/288 rows with the original delayed stream.
It creates no new teacher traces and restores no old budget or serving authority.
After training completes, the existing frozen evaluator runs raw0/raw32/invariant0/
invariant32 in that order, each on the fixed A-blue/red and inspected E-blue/red
slots. Predictions contain the raw296 public input; the bound learner projects it.
A separate zero-update readout budget remains subject to the original body clock
and completed-training guard. Final D is unavailable. No E labels or updates occur.

```powershell
# Preview: no source reads, models or environments.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_invariant_pilot --describe
# New invocation only under its own prospectively frozen local allowance.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_invariant_pilot --run --source artifacts/runs/minigrid-action-memory-live-20261017/native.json --output NEW_RESULT.json
```

Existing default recipes and evaluator entry points retain their admission.
The shared TRAIN arm also admits the explicitly bound FIFO control for engineering
checks; the frozen native pilot runs only the exact control. Prior captures and
future inputs are checked after external callbacks. A corrupted/replaced prior
capture stops work; private paid reports survive refusal, and missing delegated
receipts retain unknown costs. Four slots bound retained report storage.

```text
src/adapters/minigrid_learning_pilot.py       existing TRAIN count-family admission
src/adapters/minigrid_policy_evaluation.py    separate representation admission/shared game runner
src/adapters/minigrid_invariant_pilot.py      saved A32 recipe using that TRAIN arm
scripts/run_minigrid_invariant_pilot.py       bounded source IO/preview/exclusive output
tests/test_minigrid_invariant_{training,evaluation,pilot,cli}.py
```

The same row dose and native table cap do not equal compute or total memory.
Returned projected training arrays and prediction arrays are reported separately;
validation, Python objects, copies and failed-attempt allocations add overhead.
Historical raw training time belongs to its original run. This is an exploratory
known-task/previously inspected nuisance test, not independent generalization,
plasticity, reinforcement learning or evidence of a circadian advantage.
[Protocol, checks, failures and executed outcome](../artifacts/runs/minigrid-invariant-pilot-20261017/RESULTS.md).

To extend safely, use the existing stream and evaluator ports, retain explicit
encoding identity and source guards, and freeze any new comparative exposure
before running it. Do not change raw snapshots or select favorable seeds/actions.

## Encoding-bound neural heads

`src/adapters/invariant_torch_feature_learner.py` applies the same public projection
to all four existing feedforward methods. It composes each native head and binds
the encoding in a distinct policy/snapshot. Native objectives, diagnostics,
optimizer/gradient/adaptive/RNG continuation and raw defaults are preserved.

```python
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner, InvariantTorchFeaturePolicy,
)
from src.adapters.torch_feature_learner import TorchFeaturePolicy

model = InvariantTorchFeatureLearner(InvariantTorchFeaturePolicy(
    TorchFeaturePolicy("pc", 296, 8, 17, num_classes=7),
))
# raw_history_features is float64 Bx296 public initial/current history.
# targets is float64 Bx1 with hard seven-action class indices.
diagnostic = model.train_batch(raw_history_features, targets)
owned_state = model.snapshot_state()
independent_model = model.fork()
```

Select bp, pc, neutral_cpc or gated_cpc in the native policy; the public method is
public_invariant_ followed by that name. All rows validate/project before native
training or prediction. Raw native snapshots cannot be restored implicitly;
typed encoding metadata is a contract, not provenance authentication.

The existing MiniGrid TRAIN arm admits these explicit methods through a supplied
factory at its original32/128 schedules, with fixed296/H8/K7/seed17/native objective
settings. The separate `evaluate_neural_representation_checkpoint` uses the existing
frozen runner, only declared checkpoints and fixed target-free A/E slots, zero
training allowance and the caller's preceding-success guard. It records raw input
history, bound encoding/native method and returned projection array counts. It
provides no final D or arbitrary-layout release. Existing raw CLI modes remain unchanged;
the explicit comparison below selects the bound path.

[Exact checks, costs, preserved failure and next action](../artifacts/runs/minigrid-invariant-neural-20261017/RESULTS.md).
Native/manual-canonical equivalence, full continuation, PC-neutral parity and
existing isolated stream/probe composition pass. Four32-update TRAIN integrations
and frozen fake-game boundaries pass; fake games do not demonstrate navigation.
Generic probe projection/copy overhead is not fully measured; no useful neural
adaptation, retention, generalization or circadian benefit is claimed.


## Matched neural A32 comparison

The existing history-smoke CLI has an explicit eight-arm comparison mode. It
loads the original saved TRAIN A-blue/red pair once, trains raw BP/PC/neutral CPC/
active CPC and the four corresponding bound heads for32 updates/288rows each,
and checks each paired full native initial state. All eight streams finish
before any games. Separate zero-update readout then evaluates each initial/end
state on A-blue/red and previously inspected E-blue/red. No final D, E training,
raw-score labels, promotion or probe selection is available.

```powershell
# Resource-free preview; no source artifact required.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_history_evaluation_smoke --canonical-neural-comparison --describe
# Each new training invocation needs its own prospectively frozen local budget.
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_history_evaluation_smoke --canonical-neural-comparison --source-artifact artifacts/runs/minigrid-action-memory-live-20261017/native.json
```

The result reports all signed end-minus-initial, canonical-minus-raw endpoint/
gain and active-minus-PC/neutral gain differences, including zero and negative
results. Sixteen fixed private paid captures and corresponding planned slots
bound storage; preceding source/actor/charges/receipts remain checked across
callbacks. Malformed complete metrics stop further work and retain an explicit
failed receipt with unavailable contrasts. Missing delegated reports preserve
unknown costs. Body300s/parent330s/sharedcleanup340s/768MiB/whole32MiB/CPU1/GPU0
are maximums for this frozen recipe, not permission to rerun a spent experiment.

Changed structure uses existing recipe/evaluator files plus
`tests/test_minigrid_representation_comparison.py` and
`tests/test_minigrid_raw_neural_representation.py`; no new runner or dependencies.
The raw matched evaluator is `evaluate_raw_neural_representation_checkpoint`.

Checks: `python -m pytest tests/test_minigrid_representation_comparison.py tests/test_minigrid_raw_neural_representation.py tests/test_minigrid_history_smoke.py tests/test_minigrid_fixed_a_acquisition.py tests/test_minigrid_fixed_a_evaluation.py -q`.
Run scoped Ruff/checkformat and mypy for win32/linux using the exact commands in
[comparison evidence](../artifacts/runs/minigrid-neural-representation-comparison-20261017/RESULTS.md).
The protocol is exploratory known-task acquisition/nuisance transfer; shared row
counts and shapes do not equal total computation or memory. Neural forward
probabilities cannot be independently reconstructed from this JSON because
weights are not exported. Retention, return learning, independent generalization
and useful CPC benefit require separate evidence. Extend through the existing
stream/probe ports with separate probe budgets; preserve native objectives and
freeze new exposure before execution.


The sole reviewed native comparison is incomplete/SPENT: original300s guard
exhausted after all256updates, eight raw readouts and a partial ninth readout.
All contrasts remain unavailable; no comparative benefit is established. The
stdlib checker explicitly returned incomplete without full trajectory acceptance.
See RESULTS.md above for exact charges, unknown work and preserved failed attempts.


## Existing learnability diagnostics and isolated probes

The existing history-learnability scenario now accepts --invariant-history,
composing bound feedforward methods with original native objectives, TRAIN-fit
readouts and independently budgeted disposable E probes. Genuine owned in-memory
states are required; no JSON neural-state restore or continuation of the spent
game experiment is available. Scoped synthetic native/source-isolation checks
pass; the full800-update native scientific protocol remains unexecuted.
See [usage, checks and limits](minigrid-learnability.md#encoding-bound-public-history-route).
