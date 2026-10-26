# Complete demonstration rehearsal on the Torch route

```powershell
.venv/Scripts/python.exe -m scripts.run_coupled_attribution --describe --scenario torch-trajectory-rehearsal
.venv/Scripts/python.exe -m scripts.run_coupled_attribution --scenario torch-trajectory-rehearsal
```

The second command executes eight fixed CPU arms once. Declare a fresh local
envelope before another execution. It uses existing native BP, PC, neutral CPC
and active CPC heads, each with replay disabled/enabled. No dependency, download,
GPU or environment configuration is added. Default CLI scenarios are unchanged.

## What is learned

`LineNavigation` actually steps left/right on integer positions−2..2 and stops at
the goal. The scripted demonstrator collects complete two-transition episodes:
train start0/goals+2(A) or−2(B), in A,A,B,B,A,A order. Final-test episodes use
start−1/goal+1 and start+1/goal−1, with unseen goal cues. Observations are
`(position/2, goal/2)`; the two action labels train the unchanged native hard
binary objectives. Rewards0/1, next observations and terminal flags remain
trajectory evidence. They are not TD targets, policy gradients or reward loss.

This is feedforward behavior cloning from complete demonstrations. It does not
implement recurrent RL, autonomous policy evaluation, MiniGrid or general task
acquisition. Demonstrator reward1 is not a measured learner return. The accepted
case starts at action accuracy1 for both contexts and ends at B accuracy0 in all
arms; small BCE gains do not establish useful adaptation.

## Boundaries and data contract

```text
core/demonstration_episode.py   immutable transition/episode validation + codec
core/line_navigation.py        deterministic legal transitions, reward, terminal
app/coupled_attribution.py     exact episode ingress, arrival, isolation, resume
app/stream_rehearsal.py        optional complete-episode retention/charges/replay
adapters/torch_feature_rehearsal_pilot.py   fixed recipe with existing native heads
```

Each `DemonstrationEpisode` has an ID, context, explicit train/final_test role
and ordered `DemonstrationTransition` records. Steps are0..T−1; neighboring
states agree exactly, only the final transition terminates, states/rewards are
finite in[−1,1], and actions are native integers0/1. Episode IDs/context fit
64/32 UTF8 bytes; dimensions and fixed episode length are1..16. A copied frozen
record is revalidated, including after tampering. No arrays are retained there.

`AttributionBatch(..., episode=packet)` must have matching context/role,
`episode_id:step:k` row IDs and exact ordered observation/action arrays. Every
batch in an episode stream has metadata, and episode IDs are globally unique
across training/final. Complete preflight precedes helper binding or model work.
Runtime experience and delayed label records use the real episode ID. Original
metadata-free batches retain their previous behavior. Observer copies cannot
alter the training packets; final episodes never enter training or replay.

Configure the existing helper with
`episode_schema=EpisodeReplaySchema(observation_dim, steps_per_episode)`.
A schema helper requires episode ingress; an ordinary row helper refuses it.
Teacher distillation with episode mode is explicitly unsupported. The learner
still receives one ordinary `train_batch(features, action_targets)` per whole
selected episode, preserving chronological row order. Reward/terminal/next-state
metadata remain attached in retained packets and bounded event reports.

## Whole episodes and costs

Why this design: the existing `SharedReplayBuffer` already provides bounded
selection, deduplication and rotation. One canonical float64 row encodes one
whole episode, including UTF8 metadata, states/actions/rewards/next states/
terminal flags; there is no second retention framework. The row width is
`98 + T*(2*D+3)`, plus one dummy target scalar. For D2/T2 the full array payload
is904 bytes per episode, including header padding. Identical numerical paths
with different episode identities remain distinct.

Native-transition count caps are floored to complete episodes. Byte caps include
the full encoded row/dummy array; required partitions must each fit one packet.
The fixed recipe allows four retained transitions/two packets/1808 bytes total,
with two-transition/904-byte recent and reservoir quotas. Overlap may retain
fewer unique packets; it is not evidence of deduplication. Each event's two-
transition selection cap selects one whole packet, never a fragment.

Before each native replay call, both transitions are permanently charged to the
original replay budget. Insufficient budget for the whole event refuses before
native work. Attempted/returned/committed native calls remain separate from
transition exposures and wake receipts; failure preserves charges. Replays also
advance original native optimizer/update/chemistry/age clocks. No allowance,
receipt or clock is renewed by continuation.

The existing owner-local pause/resume reconstructs native parameters, optimizer,
gradients/local RNG, delayed inbox, encoded buffer, rotation cursor and events
under the original runtime/actor/budgets. Schema identity and content remain
bound, and malformed/replaced continuation evidence is refused. This is not
durable restart, managed sharing or a transferable authority token.

The eight-arm case has48 wakes/12 replay calls,96 wake/24 replay transition
exposures,64 final scoring forwards and16 resumes. Same-mode BP controls share
data, initializer, selections, replay memory and exposure. Off/on exposure differs;
native objectives and optimizer/adaptive-state storage are explicit, not equal
compute. Encoded replay bytes exclude Python objects, staging/copies, checkpoints
and native model state; native tensor payloads and process RSS are reported
separately. Total runtime includes codec/validation/continuation costs without
claiming a causal overhead estimate.

## Verification and extension

Focused behavior tests:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_demonstration_episode.py tests/test_trajectory_rehearsal.py
```

The exact budgeted affected-scope Ruff/format/Windows+Linux mypy commands and
focused legacy checks are in [results](../artifacts/runs/r24-trajectory-rehearsal-20261016/RESULTS.md).
See [all outcomes/costs](reports/torch-trajectory-rehearsal-20261016.md) and
[independent criterion map](../artifacts/runs/r24-trajectory-rehearsal-20261016/REVIEW.md).

Extend the existing ports for a demonstrated input/objective need. Do not silently
flatten incomplete trajectories, grant final-role replay permission, change native
objectives, or infer policy success from action BCE.
[Live policy evaluation](line-policy-evaluation.md) now executes bounded actual
episodes from every frozen checkpoint after training, with legal actions,
termination, return and work reported separately. Its accepted case retains every
final B failure; broader MiniGrid/action-schema and useful adaptation remain open.

## Optional categorical packets

[Categorical demonstration replay](categorical-trajectory-rehearsal.md) now adds
explicit2..16 action counts through the same ports, with bound encoded metadata,
isolated CE/accuracy and whole-packet charges. This binary recipe and its original
encoded bytes/defaults stay unchanged. Synthetic classifier evidence does not
establish real key/door behavior or useful adaptation.
