# Explicit key/door action remapping

## Fill a shared interaction quota

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-legal-interaction-quota --describe
```

This optional recipe fills twelve physical steps for each original context/cue
phase across fresh episodes. An eight-step goal is followed by a fresh episode
capped at four remaining steps. Each episode trains once with the existing reward
objective and its own receipt; one original budget allows at most sixteen updates.
There are exactly 96 TRAIN steps on success, with variable optimizer updates and
reset costs. Failed physical calls remain charged and stop the arm.

Phase checkpoints record both physical steps (0/24/48/72/96) and actual native
update counts. All final readouts wait until all quotas and preceding evidence
validate. No final metric selects training duration or a probe. The original
action-remap and other recipes retain their protocols. This enables equal
interaction allowances; it does not by itself establish a matched comparison.

The collector API accepts `interaction_quota=PhysicalInteractionQuota(12)` from
`src.core.physical_interaction_quota`; reuse the SAME object across episodes.
Do not create a replacement after a failed attempt. Remaining, attempted,
completed and failed counts are distinct. The default `None` preserves old output.
No additional package or environment configuration is required.

Tests: `-m pytest -q tests/test_physical_interaction_quota.py tests/test_quota_reward_collection.py tests/test_interaction_quota_pilot.py`.
[Executed evidence and limitations](../artifacts/runs/r43-interaction-quota-20261017/RESULTS.md)
and [decision](adr/ADR-0376-shared-game-interaction-quota.md).

## Original action-mapping recipe

Use the existing reward CLI:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-legal-action-remap --describe
```

Removing `--describe` runs one bounded engineering recipe: eight sampled reward
updates, seed17/private sampling29, A+/A-/B+/B-/C+/C-/A+/A-. A/B/C geometry and
pixels agree. Only B changes controls to `(1,2,3,0)`. C returns to the identity
law and is not a genuinely novel third task. D is a final-only branch-row1 layout.
This route uses one recurrent policy, no replay/probes/teacher warmup or promotion.
Existing `torch-key-door-legal-rl` and other default routes remain available.

## Mapping semantics

```python
from src.core.key_door_navigation import KeyDoorLayout, KeyDoorNavigation

layout = KeyDoorLayout("B", movement_action_map=(1, 2, 3, 0))
game = KeyDoorNavigation(layout, cue=1)
# Control3 now means spatial north. Observation channel0 still means north.
game.step(3)
```

The exact tuple must permute native integers0..3. Interaction4 keeps its meaning.
Mapping happens before inverse geometry rotation. Legal actions name controls;
the first four observation channels name spatial directions. Owned episodes
revalidate and copy configuration. Scripted collection uses inverse controls,
separately from learner prediction. Illegal/failed work never gets repaired into
a successful action, and private sampling draws remain charged on failure.

## Layers, evidence and extension

The existing core owns layout/dynamics/teacher inversion; the app collector and
evaluator own frozen sessions, roles, recorded decisions and costs. Existing
key_door_reward_pilot owns the explicit recipe; run_coupled_attribution exposes
it. No new dependencies or environment variables. The stage-only wrapper clamps
CPU threads and GPU access for recorded qualification, not global settings.

Use key_door_layout_record for saved legacy layout metadata. Identity yields the
original five fields; nonidentity adds movement_action_map and
observation_basis=spatial_directions_v1. Direct dataclasses.asdict now exposes the
new Python field. The mapped protocol is not admitted by the current inspector;
inspect the saved JSON instead. Full source/actor/checkpoint/RNG/budget guards
and finish-only final scoring remain. No final observations select controls,
seeds, metrics, training duration or probe conditions.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_key_door_action_mapping.py tests/test_mapped_reward_collection.py tests/test_action_remap_pilot.py
```

See [all executed commands and limitations](../artifacts/runs/r43-action-remapping-20261017/RESULTS.md)
and [design decision](adr/ADR-0375-explicit-game-action-mapping.md).
The integrated focused suite covers all24permutations×4rotations×2cues,
own sampled episodes, fresh sessions, permanent failure costs, delayed final
access and original reward/default routes. Run scoped Ruff/check-format and
mypy with win32/linux as recorded there; this is not whole-repository readiness.

## Explicit environment instances

The existing quota route accepts a declared pair of environment seeds:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-legal-interaction-quota --environment-seed 0 --heldout-environment-seed 23 --describe
```

Both must be distinct exact integers0..23. These seeds enumerate a finite
catalogue: `rotation=s%4`, `branch_row=1+(s//4)%3`, `swap_keys=bool(s//12)`,
appearance0. They are instance indices, not independent random draws. A/B/C use
the training index; only B remaps movement. D uses the distinct heldout index.
The protocol and run row record both roles. Heldout observations, teacher packets
and live episodes are unavailable until all96training steps complete. The factory
only constructs declared layout configuration; it never observes outcomes.

Learner initialization17 and private sampling29 retain their separate meanings.
Omit both flags to preserve the original protocol and layout defaults. Other
routes, partial pairs and equal indices fail before model/resource construction.
Do not use final scores to choose instance indices or train/probe conditions.
This input support is not an executed repeated-instance comparison.

## Frozen recurrent nonlearning control

The existing quota runner also exposes a frozen initialized recurrent policy:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-frozen-interaction-quota --environment-seed 0 --heldout-environment-seed 23 --describe
```

Python callers pass `frozen_recurrent_control=True` with the legal, mapped,
quota options. Omitting or setting this flag false preserves the existing reward
route. Both environment inputs remain optional as a pair with their original
admission rules. `method=recurrent_rl` identifies the architecture family;
`learning_mode=frozen_recurrent_control_v1` and protocol classification explicitly
identify a nonlearning control. It performs no SGD, BPTT or inbox submission.

Each policy collects its own96physical steps across fresh episodes under a
zero-update budget. Parameters, optimizer, native updates/training rows and logical
clock stay unchanged. Collection records have their own validation flag, no
manufactured native receipt. Physical checkpoints0/24/48/72/96 remain distinct
despite actual native index0 at all five. Finish-only scoring/live evaluation
uses the existing helpers one physical stage at a time:40scorecalls320rows and
40fresh live sessions. Paid partial scores and physical failures stay visible.

This control answers whether reward updates outperform unchanged initialization
under equal physical allowances in a future declared comparison. Computation,
SGD and resets can differ and must be reported. It does not replace trained
BP/PC/neutral CPC/active CPC baselines or satisfy full R4.3 by itself.

Executed evidence:428combined checks and61closing checks pass with independent
SOURCE review. The sole native report completes0SGD96collectionsteps and40live
episodes (zero successes), but its PowerShell5 parent launcher-exit check fails.
That qualification remains FAILED/SPENT with the original supervisor and raw
report preserved. Harmless subprocesses reproduce nullable exit readback and
verify early handle retention preserves both0 and2. The separate future repair
was reviewed; it does not retroactively pass or authorize retry of the qualifier.
[Full commands, limits, failures and readback scope](../artifacts/runs/r43-frozen-recurrent-control-20261017/RESULTS.md).

For future comparison, reuse the implemented quota and explicit instance inputs.
Freeze a small matched own-policy protocol and declare its catalogue indices
prospectively; record variable native updates, resets and computation. Each
policy must collect its own trajectories under the same interaction allowance;
do not substitute common teacher trajectories for different on-policy data.
Freeze/review that small protocol first. Legal support can force corridor actions,
so this scenario alone cannot prove learned remapping or circadian advantage.
