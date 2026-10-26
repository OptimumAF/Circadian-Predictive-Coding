# Actual hidden-cue key/door navigation

The local feature-observation game collects actual demonstrations, adapts the
existing native learners through the ordinary stream, then executes bounded live
policies. It is a MiniGrid-like local implementation. Official MiniGrid package
integration, learned recurrent policies and recurrent RL remain unfinished.

## Run or inspect

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --describe --scenario torch-key-door
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door
```

Describe is Torch-lazy and performs no environment steps or sampling. Execution
needs the existing optional CPU Torch environment. Reuse the
[saved output](../artifacts/runs/r42-key-door-20261016/pilot.json) to inspect the
recorded case without another allowance. No dependency or environment variable
was added. Exact commands, outcomes and limitations are in
[results](../artifacts/runs/r42-key-door-20261016/RESULTS.md).

## Observed local result

The sole fixed eight-arm native case completed80 training calls/640 rows and320
live episodes in21.795272s, with sampled RSS316,186,624B. All320 learned episodes
ended on illegal actions with zero success/return; all16 scripted demonstrations
reached the exit. Action-score changes do not establish useful adaptation.
[All curves and costs](reports/key-door-navigation-20261016.md) retain every arm.
The [recurrent BC scenario](recurrent-behavior-cloning.md#recurrent-keydoor-recipe)
now uses this same recipe with fresh episode-state resets and frozen learned
state. Its two new arms also fail all80live episodes; old baselines are reused,
not retrained. Next extend the isolated categorical fresh-task learnability probe;
no teacher cue injection or final feedback. Recurrent RL remains separately open.

## Game and information boundary

A layout reference (`#` wall, `R/B` keys, `S` start, `D` door/exit):

```text
##D##
##.##
#R.B#
##.##
##S##
```

Actions are up/right/down/left/interact. Pick up one key at its cell, return to
the corridor, interact next to the door, then enter the open exit. Reaching the
exit terminates with reward1; a wrong key at the door terminates with reward−1;
other legal actions yield0. Illegal steps refuse without mutation. A correct
scripted teacher takes eight actions, including pickup and unlock interactions.

The required color appears only in the initial cue. After any legal action it
vanishes permanently, even after returning to the start. Observations contain
only four physical neighbor tile codes, the current tile, held key color, transient
cue and appearance. No global coordinates, map or remembered target enter learner
features. Legal actions depend on physical state and held-key presence, not the
required color. Two cue cases therefore reach identical observations before
requiring different branch decisions. The focused tests explicitly prove that
aliasing and all48 rotation/branch/swap/cue teacher paths.

Layouts A/B/C change rotation, key placement and appearance; training returns to
A after C. Both cues are demonstrated per phase, giving eight wakes. Layout D
changes branch position and appearance and is final-only. Fresh A/B/C episodes
measure known-room recall, not unseen-layout generalization. D outcomes are
reported separately. Scripted collection uses declared layout/cue information;
native policy evaluation receives only observations, with no teacher fallback.

## Learning, evaluation and costs

Matched seed17 D8/H8/K5 native BP, PC, neutral CPC and active CPC use the original
objectives and rates. Replay is independently off/on: eight wakes per arm, four
additional native calls per on arm. Whole eight-transition packets are charged
before replay; the two-packet4032B cap includes headers and dummy targets. Same-mode
BP shares initialization, data, selection, memory and exposure. Off/on computation
differs, and diagnostic units remain native. No equal-compute or causal replay claim.

All training for an arm completes before final CE/accuracy or any live episode.
Five fixed complete checkpoint copies act in all eight layout/cue configurations:
320 live episodes, at most12 decisions each. Greedy argmax uses unmodified full
probabilities with lowest-index ties. No action mask, clipping, repair or retry.
Success, wrong-key, illegal-action and step-limit outcomes remain explicit.
Unexpected faults preserve owned partial reports, callback/native costs and all
unreached slots. Complete native state is checked before/after predictions; a
changed copy stops. Source learner and deployed actor stay unchanged.

Every episode carries the common method/seed/local actor/task/checkpoint/replay
identity, layout partition and cue, evaluation/learning=false state and actual
observations/probabilities/legal actions/transitions/return/termination. Context
matches the owned layout ID; known/held-out partition agrees with the declared
training sequence. Local evaluation actor versions are not historical publication.
No final result selects training, replay, promotion, probes, tuning or checkpoints.

Prospective caps:90s body/120s parent, CPU/BLAS/Torch/interop1/GPU0,768MiB sampled
process RSS/8MiB JSON,80 native training calls/640training rows,320 classification
forwards plus at most3840 live forwards. Sixteen collected demonstrations take128
actual steps; live episodes take at most3840. One RSS sampler is joined. The larger
bounded time/report envelope covers full episode records; prior spent limits and
negative results are unchanged. No sweeps, probes, sleeps, renewal or backbone.

## Structure and checks

```text
src/core/key_door_navigation.py             owned geometry, dynamics, scripted teacher
src/core/categorical_probability_scores.py  shared target-free probability admission
src/app/key_door_policy_evaluation.py       common identity and actual bounded episodes
src/adapters/key_door_pilot.py              fixed recipe through existing stream
scripts/run_coupled_attribution.py          existing CLI scenario
tests/test_key_door_navigation.py           game/evaluator behavior
tests/test_key_door_pilot.py                recipe, isolation and failure behavior
```

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_key_door_navigation.py tests/test_key_door_pilot.py
```

The recorded worker also includes affected categorical metrics and binary line
regressions. Exact scoped Ruff/format/Windows+Linux mypy commands are in
[results](../artifacts/runs/r42-key-door-20261016/RESULTS.md), with
[independent review](../artifacts/runs/r42-key-door-20261016/REVIEW.md).
See [ADR-0325](adr/ADR-0325-evaluate-actual-hidden-cue-key-door-episodes.md).

This feature is not full R4.2 completion or evidence of useful adaptation or a
circadian advantage. Memoryless feedforward controls cannot universally remember
the erased cue. A future memory/recurrent comparison must reset episode state,
use only observed information, retain the frozen known/held-out boundary and
account for its extra memory, training and interaction work. Recurrent RL remains
a separate substantive requirement.
