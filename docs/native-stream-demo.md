# Native adapter stream example (R3.8)

Run from the repository root using the normal NumPy development environment:

```powershell
python -m scripts.run_native_stream_demo --seed 23
```

The command prints one JSON report and needs no remote service, dataset download,
GPU, environment variables or optional benchmark packages. Redirect stdout to a
new file to retain a run. Seeds are explicit nonnegative 32-bit integers; the
default is 23. There is no parameter search or automatic winner selection.

## What runs

Both `BackpropLearner` (a native gradient binary head) and `CircadianLearner`
consume the same fixed four synthetic rows, binary labels and logical schedule.
Each uses input width 2, hidden width 4, seed 23 by default, learning rate 0.03,
and a separate four-update `ToyBudgetSession`. CPC retains its native two latent
inference steps at inference learning rate 0.2. These are existing adapter
settings, not new learning equations or a compute-matched scientific comparison.

| Tick | Observation | Label arrives / candidate update |
|---|---|---|
| 0 | s0 | none |
| 1 | s1 | none |
| 2 | s2 | s0 |
| 3 | s3 | s1 |
| 4 | none | s2 |
| 5 | none | s3 |

Actor predictions are recorded before each tick's arrival/update work. They use
the same fixed **unlabeled probe batch**, including rows not yet registered as
training observations. This is a training integration example, not strict online
evaluation. No final evaluation split or scoring data enters this path. The
actor remains `actor-0`; the detached `candidate-0` learns only after labels
arrive. The final candidate snapshot is restored into a separate adapter for
prediction through the public native state interface. Runtime internals are not
read or modified.

The report retains four diagnostics per method with their native definition IDs.
Backprop BCE and CPC's native energy are different quantities: do not rank them.
Equal training arrivals and update counts do not imply equal compute, replay
storage or optimization objectives. There is no sleep, replay training, topology
change, promotion, checkpoint persistence or latency measurement in this example.
CPC may retain its normal wake replay memory; that does not grant another update.

## Separate versions

Use `--scenario coupled-neutral` for the same delayed-label orchestration with
the new coupled PC/CPC objective. Its owned adapter preserves complete native
CPC chemistry, gates and wake counts. See [the coupled ports guide](coupled-learner-ports.md)
for the full state contract and additional native adaptation fields in its JSON.

Use `--scenario matched-baselines` for BP, PC and neutral CPC with exact shared
initial tensors and stable feedforward BCE at two hidden layers. Its protocol is
`matched_binary_adapter_stream_v1`; initial arrays and latent-step counts are
included. See [the matched baseline guide](matched-backprop.md). Full training
objectives and work differ; the report does not rank native diagnostics.

- `schema_version: 1` versions the presentation report.
- `protocol_id: native_adapter_stream_v1` identifies this fixed stream.
- Per-method `experiment_state.format_version: 1` describes the logical cursor
  (`next_tick: 6`) and completed updates (4).
- Per-method `model_state` identifies actor and candidate versions and the actual
  native snapshot format: backprop 1, CPC 2.

Full native snapshots stay in memory. The JSON contains version metadata and
observations, **not a resumable checkpoint**. Version identifiers provide local
traceability and do not authorize promotion or reusing spent runtime budgets.

## Boundaries and extension

```text
scripts/run_native_stream_demo.py       argument parsing and JSON presentation
src/adapters/native_stream_demo.py      fixed composition of native adapters
  -> src/app/actor_shadow.py            existing ownership and arrival runtime
  -> src/adapters/numpy_learners.py     existing native gradient/CPC state ports
tests/test_native_stream_demo.py        real native paths and CLI behavior
```

Why this: reusing the existing ports makes the example runnable without adding
another runtime, source loader or audit prerequisite. Dependencies continue inward;
no core or app module imports this example. See [learner ports](learner-ports.md)
for complete state ownership, native diagnostics and how to implement a new
adapter. A new model adapter should prove fork/snapshot/restore isolation and
native update parity before sharing this orchestration. New schedules or learning
comparisons need their own protocol and evaluation-isolation decisions.

## Focused checks

### Inspect a rejected candidate

```powershell
python -m scripts.run_native_stream_demo --scenario promotion-rejection --seed 23
```

This additional scenario performs one real backprop update, then calls the
existing serving promotion controller with two disjoint, arrived `inner_guard`
batches. A **zero serialized-state-byte allowance** is declared before the run
to exercise refusal. It is fault injection, not a tuned promotion policy or
learning comparison. Neither batch is training or final-evaluation data.

The JSON includes the actual guard policy, utilities, resource measurement,
decision reasons and sample identities. It verifies the complete serving snapshot
is unchanged and serves again from the original cache/version. Resource bytes
mean trusted local pickle encoding length, not RAM/RSS. Latency uses the real
local clock and may vary; no latency or learning advantage is asserted. Pickle
is used only to encode trusted local snapshots, never to decode external input.

Application callers can inspect the same refusal without reevaluating:

```python
from src.app.serving_promotion import PromotionRejected

try:
    ticket = controller.prepare(runtime, new_guard, old_guard,
                                now=now, training_ids=training_ids)
except PromotionRejected as error:
    report = error.report
    print(report.decision.reasons, report.evidence)
```

`PromotionRejected` remains a `ValueError` with the existing rejection message.
Each `.report` read is detached, and ordinary trusted local exception pickling
retains the report. A rejected report grants no serving ticket or permission.
Invalid roles, overlapping IDs and other invalid requests keep their existing
validation errors. Unexpected native errors still propagate.

### Commands

```powershell
python -m pytest -q tests/test_native_stream_demo.py tests/test_actor_shadow.py tests/test_experience_inbox.py tests/test_numpy_learner_ports.py tests/test_learner_step.py
python -m ruff check src/adapters/native_stream_demo.py scripts/run_native_stream_demo.py tests/test_native_stream_demo.py
python -m ruff format --check src/adapters/native_stream_demo.py scripts/run_native_stream_demo.py tests/test_native_stream_demo.py
python -m mypy --follow-imports=silent --no-incremental src/adapters/native_stream_demo.py scripts/run_native_stream_demo.py tests/test_native_stream_demo.py
```

These checks demonstrate software behavior and affected integration boundaries.
They do not establish repository-wide release readiness or a circadian advantage.
