# Categorical complete-episode fresh-task probes

The existing `run_fresh_task_probe` now accepts bound categorical data and complete
reserved episodes. It still trains only a disposable owned fork with a separate
fresh probe budget. No probe update changes the source/serving actor, promotes a
candidate or selects a benchmark checkpoint. No new runner or learner is added.

## Full MiniGrid observations

The same probe API admits widths1..16 or exactly148 within the original65536
aggregate feature/target entry cap. Rows, batches, classes, disjoint arrays/IDs,
roles, checkpoints and fresh independent budget limits remain unchanged.
The larger width reduces how many packets fit that cap. Four nine-step K7
packets occupy42912 array bytes, excluding records, Python objects and native state.

Use `predefined_probe_layouts()` from `src.core.minigrid_key_door` for reserved
E-blue/E-red contexts, each rotation1/branch_row3/unswapped/yellow walls. Collect
each with its matching explicit cue and probe_train/probe_test role through
`MiniGridKeyDoor`. Ordinary training/final roles and mismatched cues refuse
before native environment creation. Original A/B/C_train/D layouts stay unchanged;
D remains final-only. Context/cue audit metadata adds no policy input.

Keep the two complete nine-step episodes separate: the existing evaluator keys
scores by unique context and the recurrent learner resets activation per packet.
Pass training/evaluation tuples, a source-owned snapshot and a separate fresh
budget to `run_fresh_task_probe`; use its existing boundary callback when composing
with the stream. Evaluation labels stay in metrics; probe updates stay on the
disposable fork. These deterministic E traces do not establish novel-task
generalization or old-task retention. The full R4.2 readout milestone remains open.

[R2.1 MiniGrid checks and native component smoke](../artifacts/runs/r21-minigrid-probe-boundary-20261016/RESULTS.md)
record this extension separately from the original accepted probe implementation.

Why this: the recurrent key/door control returns full five-class probabilities,
but the original probe scorer expected scalar binary probabilities. The accepted
[400 failed live episodes](reports/recurrent-key-door-20261016.md) do not establish
that a learner has lost the ability to learn a reserved fresh task.

## Roles, classes and complete calls

`DemonstrationEpisode` explicitly admits `probe_train` and `probe_test` alongside
the original `train` and `final_test`. These are separate declared roles; no
automatic mapping/relabeling of benchmark data occurs. Construct reserved data
with probe roles at collection. Existing `AttributionBatch` role/context/row ID/
array/class checks require every transition to match its episode record.

All batches in one probe must have one validated class count and feature width.
Use complete episode metadata on every batch or on none. Episode probes retain
one complete ordered packet per `train_batch` or `predict` call, never concatenate
episodes, shuffle timesteps or inject remembered cues. The recurrent native port
therefore starts each packet with zero activation. Rewards and next observations
remain evidence, not supervised targets. Training/test arrays have disjoint
storage and global row identities; complete record copies are owned too.

Only `probe_train` labels reach the disposable learner. Probe-test labels stay in
application metrics. Benchmark train/final-test batches refuse probe admission;
reserved probe records refuse ordinary benchmark ingress and replay encoding.
The native recurrent `train_episode` convenience method still accepts only train
records; the validated probe app deliberately delegates its reserved packet arrays
to the disposable fork's native array API. Declared roles do not establish physical
data provenance; callers must reserve the dataset independently of final data.

For K>=3, metrics use the existing categorical mean CE (`p_true` clipped at1e-12)
and lowest-index argmax accuracy. K2 keeps original scalar class-one BCE, inclusive
half threshold and soft-target behavior. The recurrent port's full K2 output is
not silently reinterpreted as a scalar binary prediction; this integration uses K5.

## Accounting and failures

The original update/checkpoint limits, permanent charges, attempts/completions,
diagnostic IDs, partial score checkpoints and timing semantics are unchanged.
Data and opaque state detach before external fork/restore/timer/native callbacks.
Invalid public data/metadata refuses before those callbacks. Operational failures
retain measured partial work without scoring later contexts or retrying updates.

Categorical reports add `num_classes`; complete-packet reports add `episode_scope`
with detached training/evaluation IDs and aggregate transition counts. Existing
`reserved_data_bytes` counts owned feature/target arrays only. Record/Python/native
state/copy payloads are explicitly excluded, not silently claimed to cost zero.
If data detachment is incomplete, bytes/scope remain unavailable (`None`). If only
opaque state detachment fails after data copies, known data/scope remain recorded.
Old binary reports without episode metadata keep their original shape.

## Structure and checks

```text
src/core/demonstration_episode.py            explicit reserved complete record roles
src/app/fresh_task_probe.py                 existing isolated probe/metric dispatch
tests/test_categorical_fresh_task_probe.py  fake arithmetic/roles/ownership/failure checks
tests/test_recurrent_fresh_task_probe.py    actual recurrent source/actor isolation
```

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_categorical_fresh_task_probe.py tests/test_recurrent_fresh_task_probe.py tests/test_fresh_task_probe.py
```

The saved focused run also covers existing codec/categorical stream regressions.
[Exact commands, budgets, outcomes and review](../artifacts/runs/r21-categorical-probe-20261016/RESULTS.md)
and [ADR-0328](adr/ADR-0328-preserve-reserved-roles-in-complete-episode-probes.md).
No dependency, environment variable, CLI or runtime authority was added.

## What is verified

197 focused cases/591 passing phases FIRSTPASS, four-path Ruff/format/Windows+
Linux type checks and independent review qualify the scoped component. Native
integration performs one prior source update and two isolated probe updates:
three SGD steps/72unrolled timesteps/40actual teacher steps. Source/serving actor,
captured state, NumPy/Torch RNG and original data stay unchanged. The separate
probe charges two updates/16training transitions and completes six full-episode
score calls/48rows; reserved array payload2304B. There is no desired-score gate.

These checks establish functioning categorical probes and isolation, not a new
comparative learnability result, learned cue use, navigation success or CPC
advantage. All prior negative scientific results remain intact.

The fixed checkpoint measurement now composes this probe at wake0/8 through the
existing key/door runner. [Usage and isolation](key-door-learnability.md) and
[all measured curves/costs](reports/key-door-learnability-20261016.md). Four E fits
reduce CE, ending at50% accuracy, equal to a simple majority rule. The complete
original benchmark trajectory and its80failures stay unchanged. This is limited
same-trace fit, not learned cue memory or navigation; recurrent reward learning
is the next substantive R4.2 dependency. No larger probe/capture/audit increment.
