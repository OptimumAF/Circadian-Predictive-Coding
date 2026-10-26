# Categorical demonstrations in the existing stream

Complete packets can declare2..16 actions across transition, episode, replay
schema and attribution batch. The existing stream handles wake updates, whole-
episode rehearsal, owner-local continuation and isolated final scores. The native
feature port already supports these class counts; no new trainer is introduced.

## Run the fixed local case

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --describe --scenario torch-categorical-rehearsal
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-categorical-rehearsal
```

`--describe` performs no native training or memory sampling. Execution needs the
existing optional CPU Torch environment. Reuse the
[saved case](../artifacts/runs/r42-categorical-stream-20261016/pilot.json) to inspect
the accepted observation without spending a new allowance. It uses seed17,
D2/H3/K3, original native BP/PC/neutral CPC/active CPC, replay independently off/on,
and four three-transition synthetic packets in A/B/C/A order. These are explicit
synthetic records, not measured environment transitions, rewards or policy return.
No new dependency or environment variable is added.

All eight arms have four wakes. Replay every two wakes adds two native packet
calls per on arm, permanently charging three transition exposures before each.
The six-transition1936B buffer uses recent/reservoir quotas3/968B each. Shared
selection is prediction independent. Same-mode BP shares initializer, data,
exposure and replay limits; off/on work differs and native objectives keep their
original units. Pause after wake2 reconstructs the candidate/inbox/buffer under
the same runtime, actor, budgets, clocks and receipts. Copying renews no allowance.

The case is bounded: CPU/OMP/MKL/OpenBLAS/Torch/interop1/GPU0,60s,768MiB process
RSS,2MiB JSON;40 native training calls/120 training rows,120 final forwards and
eight resumes. No probe, sleep, renewal, backbone, promotion or metric tuning.
Every planned arm/checkpoint/context and failed/unreached record stays visible.

## Contract and isolation

```python
transition = DemonstrationTransition(
    step=0, observation=(0.0, 0.0), action=2, reward=1.0,
    next_observation=(0.5, 0.0), terminal=True, num_classes=3,
)
episode = DemonstrationEpisode("train-1", "A", "train", (transition,), num_classes=3)
schema = EpisodeReplaySchema(2, 1, num_classes=3)
```

These types live in `src.core.demonstration_episode`. Existing positional calls
retain default2. Each packet, its transitions, batch and replay schema must agree
on the declared count; every batch in a stream shares that count before external
learner work. For K≥3, labels are finite exact float64 B×1 integral class indices
0..K−1. Binary generic batches preserve their existing soft[0,1] contract.
Categorical replay uses complete packet metadata; raw binary row replay remains
unchanged. All observations, next observations, rewards, step order and terminal
boundaries survive encoding. Reward/next observation do not change the native loss.

Default binary replay bytes and two-field helper schema reports stay unchanged.
Categorical rows add one float64 class-count header at offset98 before transitions;
a different count cannot decode the packet even if all its observed actions fit
both counts. For D2/T3/K3, the retained arrays occupy968B including the dummy
target. Byte quotas include that header; transition limits floor to whole packets.
Encoded array payloads exclude Python objects, staging, snapshots and native model
state, which have separate tensor/process accounting.

Final-role packets cannot enter replay. Final targets reach only the application
metric after all training, on detached checkpoint copies; no final target selects
updates, probes, promotion, replay or tuning. Categorical predictions must be
float64 B×K finite probabilities in[0,1], row sums within absolute1e-6/relative0.
Scores are CE=mean negative log of true-class probability clipped1e-12..1, and
argmax accuracy with lowest-index ties. No normalization or action repair. Binary
BCE/threshold semantics remain unchanged. The code does not infer a class count
from labels or predictions.

Unexpected faults retain permanent charges and attempted/returned work; a
secondary readback error has its own field and preserves the primary failure.
Owner-local snapshots remain bounded; no durable restart, historical serialized
state migration or transferred authority is claimed.

## Files, checks and extension

```text
src/core/demonstration_episode.py              count-bound packets and codec
src/core/categorical_probability_scores.py     hard-label and probability metrics
src/app/coupled_attribution.py                 count preflight and final-only scoring
src/app/stream_rehearsal.py                    original buffer/charges/continuation
src/adapters/categorical_trajectory_pilot.py    fixed CLI recipe over existing stream
tests/test_categorical_trajectory.py           packet/metric/stream behavior
tests/test_categorical_trajectory_pilot.py     recipe/CLI/failure behavior
```

Focused invocation (the saved worker also includes affected binary regressions):

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_categorical_trajectory.py tests/test_categorical_trajectory_pilot.py
```

Exact scoped Ruff/format/Windows+Linux type checks, commands, failures, budgets
and actual native outcomes are in
[results](../artifacts/runs/r42-categorical-stream-20261016/RESULTS.md) and
[independent review](../artifacts/runs/r42-categorical-stream-20261016/REVIEW.md).
See [ADR-0324](adr/ADR-0324-carry-declared-action-count-through-existing-rehearsal.md)
and [native categorical port](categorical-feature-actions.md).

This scope is a categorical stream prerequisite. It does not establish actual
key/door performance, hidden-cue memory, held-out-layout generalization or a
circadian advantage. [Actual key/door demonstrations and live episodes](key-door-navigation.md)
now use these ports; all320 learned episodes fail illegal. Next add trainable
recurrent behavior cloning with owned episode resets. Preserve original R4.2
A→B→C→A, changed layouts/visuals, old-room recall, held-out layouts and separately
evaluated recurrent RL requirements; do not rename synthetic packets as a game.
