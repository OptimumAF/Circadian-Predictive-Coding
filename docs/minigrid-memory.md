# Optional actual MiniGrid boundary

This adapter executes **MiniGrid-MemoryS7-v0**, a memory task with an initial
object cue and a later matching-object choice. It is a prerequisite for the full
R4.2 recurring key/door game, which remains unfinished. The earlier custom
key/door environment and its failed retention results remain available.

## Install and run

From the repository root, using the existing virtual environment:

```powershell
.venv/Scripts/python.exe -m pip install --only-binary=:all: -r requirements-minigrid.txt
.venv/Scripts/python.exe -m scripts.run_minigrid_memory_smoke --describe
.venv/Scripts/python.exe -m scripts.run_minigrid_memory_smoke
```

The optional file pins MiniGrid3.1.0, Gymnasium1.2.3, pygame-ce2.5.7 and the two
small supporting packages. The base requirements/default commands are unchanged.
MiniGrid3.1 supports this checkout's Python3.14 and uses pygame-ce; see the
[pinned package configuration](https://github.com/Farama-Foundation/Minigrid/blob/v3.1.0/pyproject.toml).
No rendering, window, dataset download, service or environment variable is needed.
`--describe` loads no optional environment package and performs no steps.

The fixed smoke uses seed17: upper-branch script, sixteen left turns, then the
same upper-branch script again. It checks native ending flags and determinism,
preserves success or failure, and prints all evidence as JSON. No learner is
constructed or updated. This script is not an expert policy or learning result.

## Contract and costs

- Preserve the entire public uint8 image, shape7x7x3, C order. Normalize object,
  color and state channels as `2*x/[10,5,2]-1`, append `2*direction/3-1`:
  **148 native float coordinates**, each in[-1,1]. Raw images remain in evidence.
- Validate and retain the fixed mission string; it is constant and is omitted
  from numeric input. Changed missions, shapes, dtypes and ranges fail explicitly.
- Actions0..6 are the native left/right/forward/pickup/drop/toggle/done actions.
  All are valid inputs, including ineffective actions. No private-grid mask,
  target read, action translation, crop or remembered-cue reconstruction occurs.
- Configure native max_steps in1..16. One requested action calls native step
  exactly once. Attempts and normal returns are separate permanent counts.
  A throwing call may have changed the environment: effect stays unknown and
  further steps are refused. Malformed normal returns remain charged.
- Preserve `terminated` and `truncated` independently, including bothtrue.
  `terminal` on the adapter means either ending, as an interaction stop signal.
  `to_demonstration()` requires genuine termination **without** truncation;
  timeout cannot be relabeled as a complete demonstration.
- Typed records retain explicit train/final_test/probe_train/probe_test roles.
  Existing replay accepts train only. Conversion preserves actual scripted
  actions/rewards, including unsuccessful terminal traces; it claims no expertise.

Existing episode and Torch policy ports admit their old widths1..16 plus the
one fixed148 geometry. Steps, hidden widths, action bounds, losses, snapshots and
old wire bytes are unchanged. The full16-step/K7 replay packet costs39,072 array
bytes, including its dummy target. Old small replay budgets must not be reused
without accounting for this increase. No replay retention cap is changed here.

## Files and extension

```text
src/infra/minigrid_memory.py              optional environment lifecycle/encoding
scripts/run_minigrid_memory_smoke.py      fixed environment-only runnable example
tests/test_minigrid_memory_adapter.py    dependency-independent failure contracts
tests/test_minigrid_memory_live.py       optional actual environment parity
tests/test_minigrid_feature_dimensions.py existing DTO/codec/policy geometry
```

The infra adapter depends on core records; the app/stream does not import Gym.
Feed complete train-role episodes through `episode_arrays`, `AttributionBatch`
and the existing `AttributionStream`; keep final/probe roles separate. Recurrent
policies still require a fresh owned episode session. Adding a new environment
requires an explicit observable encoding/action/end contract, not reuse of this
fixed mission check or a change to its semantics.

## Checks and evidence

```powershell
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/r42-minigrid-boundary-20261016/run_checks.py statics
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/r42-minigrid-boundary-20261016/run_checks.py focused
```

Actual parity tests skip when optional pinned packages are absent. This session's
observed outcomes and exact command receipts are recorded in
[RESULTS](../artifacts/runs/r42-minigrid-boundary-20261016/RESULTS.md).
314focused cases passed, including both actual parity cases. The real CLI completed
30steps/three episodes. Five existing native methods each completed one update
through the existing TRAIN-only stream on the same saved seven-step action trace;
the initial unsupported-budget harness failure remains recorded separately.
No broad comparison, hidden-cue learning, held-out generalization or circadian
advantage follows from boundary correctness.
