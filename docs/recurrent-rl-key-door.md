# Separate recurrent RL key/door route

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --describe --scenario torch-key-door-rl
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-rl
```

The existing CLI now runs one recurrent reward-learning arm separately from BC
and the four original matched models. It uses existing key/door layouts, native
reward objective, actor/inbox runtime, original budget and final evaluator. The
original torch-key-door, torch-key-door-recurrent and learnability routes keep
their behavior. There is no new runner, dependency, environment variable or codec.

## Runtime composition and isolation

`RewardUpdate` owns a complete policy-generated episode and its full behavior
checkpoint. `TorchRewardUpdateLearner` implements the existing opaque NativeLearner
port; its train_batch accepts that exact packet plus a matching episode-ID string.
The string is a release identifier, not a teacher class label or new authority.
The wrapper delegates the unchanged explicit native reward API at gamma1.
Supervised arrays are refused, nontrain roles/stale fullstate refuse, and forks
own native policy/optimizer/gradient state. Snapshots retain the native schema.

For each of eight A±B±C±A± tasks, restore the current owned candidate checkpoint
on a distinct collection fork. A private PCG64seed29 samples a fresh zero-hidden
session in actual dynamics. Preserve its actual decisions, spent RNG and partial
costs. Only its train episode and behavior checkpoint enter an ordinary train-role
Experience and matching LabelArrival. The existing runtime supplies exactly-once
admission/receipts, candidate isolation and permanent original-budget charges.
A synchronous native observer retains the commit before any late resource stop.

Before more collection, validate the preceding full checkpoint, budget identity/
limits/clock/progress, monotonic observed time, exact sampler progression and the
bounded training records. Changed or replaced evidence refuses. Failures preserve
primary/secondary errors, spent work and all unreached slots. Storage is bounded:
eight planned train/inbox records, five checkpoints and one overwritten comparison
copy, not an unlimited chain. Training rows vary with policy trajectories.

After all eight commits, existing checkpoint scoring accepts explicit training
context strings without fabricated teacher training batches. A private prediction-
only type protocol expresses that scoring requires no train_batch class signature.
The common evaluation identity explicitly admits recurrent_rl, and the existing
live helper starts fresh frozen sessions for that method. No final labels, rollouts
or metrics enter updates, replay, tuning, promotion or probe selection.

## Fixed local protocol and evidence

Seed17/D8/H8/K5/rate.03/momentum.9/gamma1/maxnative16/maxsample12; no replay,
BC warmup, reward shaping, masking, repair, entropy, baseline or normalization.
Only eight final teacher packets are collected. Evaluate all0/2/4/6/8 checkpoints,
known A/B/C and final-only D±, with original CE directions and actual live success/
return/steps. Reuse accepted baselines without rerunning or tuning them. Objectives,
initialization histories and variable exposure differ; no equal-work/CPC claim.

125distinct scoped cases qualify: first125/375PASS plus one changed descriptor
case rechecked after inherited architecture metadata was corrected. Final ten-path
Ruff/format/Windows+Linux types and independent review gate the sole native case.
It completes8SGD/11trainrows/382unroll within all limits. All eight train episodes
return0; parameters stay unchanged. All40final episodes fail at illegal actions,
zero success/return. This verifies runtime functionality, not useful learning.
[All actions, costs and critical path](reports/recurrent-rl-key-door-20261016.md).

## Structure and checks

```text
src/adapters/torch_reward_update_learner.py   reward-only opaque native port
src/adapters/key_door_reward_pilot.py         existing-runtime recipe/report
src/adapters/key_door_pilot.py                reused layouts/live/helpers
src/app/coupled_attribution.py               reused scoring, explicit contexts
src/app/line_policy_evaluation.py            common RL evaluation identity
scripts/run_coupled_attribution.py           explicit route
```

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_torch_reward_update_learner.py tests/test_key_door_reward_pilot.py tests/test_reward_score_contexts.py tests/test_reward_pilot_integrity.py tests/test_key_door_pilot.py tests/test_recurrent_key_door_pilot.py tests/test_key_door_learnability_pilot.py tests/test_line_policy_evaluation.py
```

Exact affected lint/format/type commands and resource/failure receipts are in
[RESULTS](../artifacts/runs/r42-recurrent-rl-pilot-20261016/RESULTS.md).
Why this: the generic runtime already supports opaque training; forging class
labels or adding another orchestration framework would lose the reward boundary.
[ADR-0331](adr/ADR-0331-route-on-policy-rewards-through-existing-runtime.md).

The [typed demonstration→reward phase mode](demonstration-reward-phases.md) now
uses this runtime and preserves the pure-RL route. Its native phase boundary is
engineering-verified; a [fixed matched comparison](warm-start-control.md) now
measures its native sixteen-update arm. All live episodes still fail. Next add
per-decision/class diagnostics to the existing reserved fresh-task probe.
Do not tune previously scored finals or treat zero-reward retention as useful skill.
