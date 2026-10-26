# Actual MiniGrid cue/key/door game

Reserved E-blue/E-red contexts now use this same episode port with matching cues
and probe_train/probe_test roles. The original four benchmark layouts are unchanged.
See [full-width isolated probes](categorical-fresh-task-probes.md#full-minigrid-observations)
for separate episode, input, role and budget requirements.

This optional R4.2 game uses MiniGrid's native grid, objects, seven actions,
visibility and step clock. An initial colored ball indicates which of two keys
will unlock a neutral door. The cue disappears after the first action. The agent
must collect the matching key, unlock the door and enter the goal.

## Run

From the repository root, after installing `requirements-minigrid.txt`:

```powershell
.venv/Scripts/python.exe -m scripts.run_minigrid_key_door_demo --describe
.venv/Scripts/python.exe -m scripts.run_minigrid_key_door_demo
```

The fixed demo collects both blue/red cues in TRAIN sequence A/B/C_train/A,
then two separate final_test demonstrations in held-out layout D. It prints all
actual public observations, actions, returns, ending flags and costs as JSON.
The teacher is scripted; no model is trained, selected or promoted.
`--describe` constructs no environment and loads no optional native package.
No window, rendering, dataset, server or environment variable is required.

## Public observations and the teacher

Every learner input contains the full public 7×7×3 image and direction, normalized
as in the [existing MiniGrid boundary](minigrid-memory.md): 148 coordinates, with
object/color/state divided by 10/5/2 before mapping to [-1,1]. A fixed mission
contains no key color. Raw pixels are retained in evidence.

The teacher receives only that normalized observation. A fresh controller reads
the visible ball, remembers its color and derives its navigation phase from the
public history. It finds an adjacent matching key, turns, picks it up, verifies
the publicly visible held key, turns back and approaches the door. It never
receives the layout, cue argument, seed, environment, report or private target.

**Why a neutral door:** a door colored to match the required key would disclose
the answer after the cue disappeared. This door remains grey in both tasks. Its
private lock checks the held key, then uses native door opening. A wrong-key
toggle ends with reward -1; no-key toggle is a valid blocked action. Correct goal
arrival uses MiniGrid's native shaped reward. The first action removes the ball
before the single native step; removal adds no hidden action or clock increment.
After identical first-action histories, opposite cues give identical current
inputs until an agent's choices change the visible world.

## Declared layouts and roles

| Context | Clockwise rotation | Key row | Keys swapped | Wall color |
|---|---:|---:|---|---|
| A | 0 | 3 | No | Grey |
| B | 1 | 4 | Yes | Blue |
| C_train | 2 | 3 | Yes | Red |
| D | 3 | 4 | No | Yellow |

The base map is 7×7; rotations transform the full scene and starting orientation.
Layouts are explicit, with no seed search. D requires final_test at this adapter;
the fixed demo collects all TRAIN episodes before constructing D episodes.
Role-preserving complete records use existing `DemonstrationEpisode`,
`episode_arrays` and `AttributionBatch` validation. Training records roundtrip the
existing replay codec; final/probe records remain refused by training replay.

One call executes one native action, including blocked moves and no-ops.
The limit is 16 native steps. Terminated and truncated flags remain separate;
timeouts never become complete demonstration records. The shared lifecycle keeps
unknown effects of throwing steps, permanent charges, detached records and
post-fault refusal from the accepted Memory adapter. Default Memory behavior and
base dependency requirements remain available unchanged.

## Files and safe extension

```text
src/core/minigrid_key_door.py          layouts and public-history teacher
src/infra/minigrid_key_door.py         actual environment and owned episode port
src/infra/minigrid_memory.py           shared private lifecycle, original Memory API
scripts/run_minigrid_key_door_demo.py  fixed bounded demonstration collection
tests/test_minigrid_key_door.py        independent contracts and actual episodes
```

Core has no environment import. Actual MiniGrid imports remain lazy in infra.
The shared lifecycle was extracted because two real tasks need the same ending,
role and failure handling; no new trainer or orchestration framework was added.
Next learner work should feed these complete TRAIN packets into the existing
stream/native ports and reserve final layout access until training is finished.
Use a fresh recurrent policy session per live episode and account for every
prediction, exposure, diagnostic and replay byte.

## Checks and limitations

```powershell
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/r42-minigrid-cue-key-door-20261016/run_checks.py statics
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/r42-minigrid-cue-key-door-20261016/run_checks.py focused
```

Actual tests require the optional pinned packages; pure teacher tests do not.
[Session evidence](../artifacts/runs/r42-minigrid-cue-key-door-20261016/RESULTS.md)
records observed outcomes, errors and costs. Scripted demonstrations establish
environment behavior and source compatibility, not learned acquisition, old-task
retention, recovery savings or held-out generalization. Those R4.2 requirements,
separately evaluated recurrent RL, and the full repository goal remain open.
Existing negative results establish no circadian advantage and remain unchanged.

This session: 370 focused cases passed with no skips, plus five-path lint/format/
Windows and Linux type checks. The sole fixed CLI case completed all ten episodes
in nine steps each (90 native steps), return0.49375 each. No model was updated.
