# Recurrent behavior-cloning component

`TorchRecurrentLearner` adds learned episode history through the existing native
array/owned-state interface. It is a CPU Elman tanh recurrent classifier trained
by mean per-transition cross entropy and full backpropagation through time with
native SGD. This is a separate `recurrent_bc` control, not a change to BP, PC,
neutral CPC or active CPC, and not recurrent reinforcement learning.

Why this: actual key/door cues disappear before later choices. Identical current
observations can therefore require different actions. The accepted feedforward
case has320 illegal-action failures/zero success; it remains preserved in the
[game results](../artifacts/runs/r42-key-door-20261016/RESULTS.md). A history-capable
architecture is a prerequisite, not evidence that training solves the game.

## Interface and boundaries

Each `train_batch(features, targets)` is **one ordered complete episode**, with
1..16 rows (or the lower declared policy limit). Features are exact finite float64
TxD arrays; targets exact float64 Tx1 integral class indices. No padding, shuffling,
mixed episodes, teacher forcing of prior actions, hidden targets or automatic
truncation. Hidden activation starts at zero for every train/predict array call.
`train_episode` additionally validates/copies the existing complete
`DemonstrationEpisode` and refuses final-test roles or a different class count.
The array API has no role metadata; the existing application's role boundary
must still prevent final data entering it.

For feature width D, hidden width H and classes K:

```text
h0 = 0
ht = tanh(xt Wi + h(t-1) Wr + bh)
logits_t = ht Wo + bo
loss = mean_t cross_entropy(logits_t, action_t)
```

Shapes are Wi(D,H), Wr(H,H), bh(1,H), Wo(H,K), bo(1,K). Counts are D1..16,
H1..32, K2..16. CPU float32 parameters initialize from a private seed generator;
no RNG is consumed during updates/predictions. Initialization, fork and restore
leave global NumPy/Torch RNG unchanged. Default rate.03/momentum.9 are component
settings, not selected from held-out outcomes. Inputs/output arrays are owned.

Parameter payload is4(DH+H²+H+HK+K) bytes; active gradients and nonzero-momentum
SGD each add that payload after an update. One live activation adds4H bytes,
excluding its owned learner/birth snapshots, metadata, temporary BPTT graph and
runtime overhead. These array sizes are not a measured process peak or an
equal-memory comparison with the feedforward controls.

Whole-episode `predict` returns float64 TxK probabilities **including K2**. This
new interface is explicit; old binary feature learners still return their scalar
class-one probabilities. The existing binary attribution scorer expects the old
scalar contract; this component's tested stream integration uses categorical K5.
No claim of drop-in K2 binary stream compatibility is made.

## Frozen learned state and live activation

`snapshot_state`, `restore_state` and `fork` own all parameters, gradient buffers,
native SGD configuration/momentum and completed-update/example counters. There
is no stochastic native state after initialization. Policy, exact fields, finite
CPU float32 geometry and counter consistency are checked before publication.
Each update builds an independent candidate and publishes only after validating
the completed result. Failed native work leaves the source learned state intact.

Use a new `learner.start_episode()` for each rollout. Its predictor owns a frozen
learner copy and zero hidden activation. Only `session.predict(one_row)` advances
that separate activation/counter, at most the declared episode limit. The session
does not train or renew any application budget. Its `snapshot_state` reads actual
learned/optimizer state and refuses any departure from its birth state. Input
activation is staged as a clone; a failed forward cannot partly publish it.
Existing full learned-state checks can surround the callback without ignoring
arbitrary mutation. A new episode requires a fresh session; there is no public
reset that silently extends an existing rollout.

The existing `run_key_door_episode(session.predict, ...)` accepts explicit
`EvaluationIdentity(method="recurrent_bc", ...)`. Final-role history may evolve
inside that isolated episode, but never updates learned parameters or chooses
training/probes/checkpoints. No action masking, repair, retry or teacher fallback.
State is owner-local component continuation, not durable runtime recovery.

## Structure and verification

```text
src/adapters/torch_recurrent_learner.py      native BPTT/SGD, owned state, live session
src/app/line_policy_evaluation.py           common identity accepts recurrent_bc
tests/test_torch_recurrent_learner.py       independent mathematical/state tests
tests/test_recurrent_stream_integration.py  existing replay/stream/live composition
```

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_torch_recurrent_learner.py tests/test_recurrent_stream_integration.py
```

Use the saved bounded verification instead of repeating native work when sources
are unchanged. [Results and exact commands](../artifacts/runs/r42-recurrent-port-20261016/RESULTS.md)
record executed checks, failures, skipped suites and costs;
[independent review](../artifacts/runs/r42-recurrent-port-20261016/REVIEW.md) states
the accepted scope. No dependency, environment variable, runner or CLI was added.
Existing optional Torch setup and project lint/type commands apply.

## Observed verification

Independent review accepts199 distinct cases/597latest passing phases:197 first
passes and two repaired integration fixtures. The initial invocation remains
failed; fresh final A± contexts were added alongside D± to satisfy the unchanged
stream gate. Aggregate32nativeSGD steps/313unrolled timesteps fit the original
caps. Independent two-update BPTT/SGD tensors/gradients/momentum match exactly;
ordinary replay and live callbacks execute. No standalone comparison or trained
key/door-success result follows from these component checks.

## Recurrent key/door recipe

The existing key/door recipe now has an explicit recurrent scenario:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --describe --scenario torch-key-door-recurrent
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-recurrent
```

It runs only recurrent replay off/on, reusing the accepted original four-method
evidence as a separate reference. The original `torch-key-door` route keeps its
eight matched feedforward arms. No new orchestration or external dependency.
The descriptor freezes the same seed17, D8/H8/K5, rate.03/momentum.9, eight actual
complete A→B→C→A training episodes, final A/B/C/D±, checkpoints0/2/4/6/8 and
original replay selection/charges. It is an exposure-matched small control, not
an equal-architecture, initialization, objective, memory or computation claim.

Every complete-episode classifier call resets history; all80 live slots start a
fresh zero-step owned frozen session. The report includes parameters, actual
SGD/gradient tensor bytes,32B activation, complete training/classification
unrolled steps and callback attempts/returns/applied steps. Failed rows keep
partial/unreached slots and mark potentially unobserved native work explicitly.
Copy/graph/Python/runtime memory is excluded from logical tensor payload counts;
sampled process RSS covers the full invocation separately.

Prospective native caps:20SGD calls/160training rows,80classifier calls/640rows,
80live episodes/max960live predictions,1760max recurrent timesteps,90s body/
120s parent/768MiB sampled RSS/8MiB report, CPU/BLAS/Torch/interop1/GPU0. No
additional epochs, final feedback, probe, rate/seed tuning or desired-success gate.
Reuse [saved recipe evidence](../artifacts/runs/r42-recurrent-key-door-20261016/RESULTS.md)
when unchanged. Component gradient correctness and history sensitivity do not
prove learned navigation, old-task retention, context recovery or held-out-layout
generalization. Full R4.2 and separately evaluated recurrent RL remain unfinished.

The sole accepted local case completes20updates/970unrolled timesteps in12.6665067s,
with sampled process RSS313,589,760B. All80 recurrent live episodes fail on illegal
actions, zero success/return. Both C CE gains are negative. All original320
feedforward failures remain preserved; neither result establishes useful task
acquisition or retention. [Every arm, curve and cost](reports/recurrent-key-door-20261016.md).

The [categorical complete-episode probe](categorical-fresh-task-probes.md) now passes
its native isolation checks. [Fixed checkpoint measurements](key-door-learnability.md)
at0/8 reduce reserved E CE without changing any original benchmark trajectory.
Their50% endpoint accuracy equals a majority reference, and all80original live
failures remain. This is limited trace fitting, not useful task learning. Full
R4.2 still needs a separately evaluated recurrent reward-learning control.

## Separate explicit reward update

The same learner now also offers [reward-trajectory updates](recurrent-reward-learning.md)
through a separate exact record/API, preserving this supervised CE interface and
snapshot format. Actual policy collection uses owned frozen sessions and private
caller-owned RNG. Independent objective/gradient/continuation checks pass, but the
small engineering episodes still fail. Existing BC results remain unchanged;
a separately evaluated RL recipe is next.
