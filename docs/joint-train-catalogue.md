# Joint TRAIN-catalogue batches

`compose_catalogue_training` makes owned joint batches for the existing native
learner and training-only attribution stream. Each wake contains every original
TRAIN row in order. Repeating a catalogue increases row exposure; transport IDs
do not turn it into new data. This changes the experience distribution relative
to sequential online learning and is not an equal-compute baseline comparison.

## Files and boundaries

```text
src/adapters/minigrid_train_catalogue.py       source admission and owned batches
src/adapters/minigrid_joint_catalogue_route.py public option/protocol composition
tests/test_minigrid_train_catalogue.py         independent composition checks
tests/test_minigrid_catalogue_stream.py        four native heads through the stream
tests/test_minigrid_joint_catalogue_route.py  independent public route checks
tests/test_minigrid_joint_catalogue_native.py complete native API integration
tests/test_minigrid_catalogue_support.py      source-set support/default regression
```

The composer accepts an exact tuple of1..16 complete causal raw296/K7 TRAIN
episodes, at most128 rows, and an exact integer wake count1..128. It reuses the
complete demonstration witness, validates both current and next public histories,
and requires a fixed causal initial frame matching the first current frame.
Only original current observations and action labels enter transport arrays.
Context, reward, next view and source IDs remain outside features.

Each `AttributionBatch` has role `train`, context `train-catalogue`, seven classes
and `episode=None`. Source spans preserve original episode ID/context and half-open
row boundaries. Why this: independent feedforward rows can be pooled, but joining
episodes must not manufacture a continuous trajectory. The composer is not a
recurrent or episode-replay adapter.

Features and targets are float64 owned arrays, disjoint between wakes. The existing
invariant native learner canonicalizes raw inputs at its normal boundary, once per
operation. Admission canonicalization is separate validation work and its logical
rows/bytes are reported. The composer does not train, select tasks, release labels,
evaluate, promote, create a stream runner or access files/RNG.

`CatalogueTrainingPlan` reports source spans/rows, repeated `row_exposures`, retained
`input_array_bytes`, temporary `source_template_array_bytes`, and validation
projection rows/logical bytes. Before transport allocation, aggregate retained
feature+target arrays must fit16MiB. These figures exclude Python records, temporary
validation tuples and copies owned by the stream/runtime/native backend; measure
RSS separately. Returned arrays are mutable owned inputs, not sealed authority.

## Use the existing runtime

```python
from time import monotonic
from src.adapters.minigrid_train_catalogue import compose_catalogue_training
from src.app.coupled_attribution import run_attribution_stream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget

# episodes: complete causal TRAIN296/K7 sources; learner: existing owned native port.
plan = compose_catalogue_training(episodes, updates=2)
budget = ToyBudgetSession(
    ToyExecutionBudget(max_training_updates=2, max_wall_seconds=30), monotonic
)
result = run_attribution_stream(
    learner, plan.training, (), checkpoint_updates=(0, 2),
    label_delay=1, budget=budget, training_only=True,
)
assert result["training_examples"] == plan.row_exposures
```

Use the existing source loader/causal encoder at an IO boundary when needed. Keep
experiments prospectively budgeted, with RSS supervision and original native
method/objective receipts. No dependency or environment variable is added.

## Executed composition scope

79 focused cases pass, including two real delayed54-row updates for each of BP,
PC, neutral CPC and active CPC:8 updates432rows,4 endpoint predictions216rows,
2BP optimizer steps. Source and actor stay unchanged; native CE and PC-energy
definitions stay distinct. No final data or game physics is used. Scoped Ruff,
format, Windows/Linux mypy and independent review pass. This demonstrates batch
composition and runtime operation, not successful acquisition or adaptation.

## Explicit checkpoint and probe row binding

The existing MiniGrid checkpoint/readout/probe helpers now accept the explicit
keyword `catalogue_rows=54` for the fixed six-source catalogue and bound neural
methods. This requires exact native counters `completed_updates == point` and
`training_examples == 54*point`, with native point0..128. Snapshot policy,
encoding, optimizer, source, input and original budget guards stay in force.
Other row counts, coerced primitive types, memory/unbound/paired modes refuse.
Omitting the keyword preserves nine-row clocks, original declared points and
default readout reports. Explicit reports add guarded `catalogue_rows_per_update`
and `expected_training_examples` fields.

`_run_probe` applies this law only to its initial benchmark checkpoint. Its owned
F and A-return updates retain their original additional nine-row exposures,
separate budgets and permanent charges. In one native BP engineering case,
two joint updates are followed by16F updates with old-A readouts and16A-return
updates. Source, actor and original checkpoint remain unchanged; later metadata
retyping is refused. Failed callbacks retain canonical original row evidence and
available paid results. Internal probe callers initialize the normal slot record,
including `report=None`, before invoking this helper.

163 distinct scoped cases qualify through retained, extended and repaired gates;
this is not one163-case green run. Final five-path Ruff/format and Windows/Linux
mypy pass; independent review qualifies the precise composite evidence. A failed
two-case fixture gate is retained. Native execution/isolation does not establish
skill acquisition, retention benefit or circadian advantage.

## Public acquisition and isolated adaptation route

The existing `minigrid_learnability.run_pilot` and CLI now accept explicit
`joint_catalogue=True` / `--joint-catalogue`. This requires invariant public
history, one explicit native method, distinct F probes, old-A readouts and A return.
It refuses paired/unbound methods, memory/weighted/acquisition-only modes,
optional health/gradient diagnostics and live rollout combinations. Omitting the
flag preserves the original sequential nine-row route.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --distinct-probe --old-task-readouts --return-to-a --joint-catalogue --describe
```

Removing `--describe` runs a native pilot and collects TRAIN/probe teacher traces;
budget and independently review an experiment before doing so. Other supported
methods are `pc`, `neutral_cpc` and `gated_cpc`. This is an explicit experience
variant, not a new learner or runner.

The route collects six original A/B/C_train blue/red sources and reuses all54 rows
on each of128 wakes:6912 benchmark exposures. It scores all six traces at every
original checkpoint0/32/64/96/128, then uses the existing isolated F16/old-A/A16
route from0/128. Complete work is192 native updates7488 rows and94 scores846 rows.
Joint acquisition means every observed training trace exact at128; sequential
forgetting interpretations are omitted. Source spans and clocks remain explicit.

Novelty support admits exactly six complete ordered raw TRAIN sources in an
explicit mode. Why this: the repeated joint batches have no episode and the
support set is unchanged by repeating its source rows. Default128-packet support
admission remains unchanged. The216 source/probe projections are charged separately
from all6912 benchmark exposures. Every F packet still needs9/9 unseen canonical
inputs. Final evaluation remains unavailable.

Retained benchmark arrays are16615584B, joint arrays16422912B, temporary template
128304B, validation projected equivalent255744B. These are logical array figures;
runtime copies, reports, native state and process RSS are additional. No dependency
or environment variable is added.

85 distinct passing cases qualify through composite gates (including one repaired
full native BP public API case and retained default checks); final eight-path Ruff,
format and Windows/Linux mypy pass. The first paid128-update support refusal and
initial seven type errors are retained. Independent review accepts the scoped
implementation. [Commands and costs](../artifacts/runs/joint-catalogue-route-20261018/RESULTS.md).

## Frozen four-method result

All four one-seed cases completed with original doses and source/actor isolation.
None acquired all six traces: BP45/54 actions, PC/neutral/active30/54; all exact
traces0/6. PC and neutral reported metrics match. Fresh-task CE improves, but old-A
was not acquired before F; retention/recovery claims remain open. No circadian
benefit follows. This is not matched computation or comparison with online9-row
training. All earlier negative results remain valid.

Protocol evidence deviation: TRAIN probabilities allow independent rescoring of
1080 rows; probe/return DTOs retain2304 decisions and losses, not their probability
vectors. All saved decisions/labels/class aggregates/curves were checked, but the
promised independent probability recomputation for those rows is unfinished.
Cases stay SPENT; the frozen protocol is unchanged and no rerun is implied.
[Full result, qualification and next action](../artifacts/runs/joint-catalogue-study-20261018/RESULTS.md).

[Checkpoint/probe commands, retained failures, costs and review](../artifacts/runs/catalogue-checkpoint-rows-20261018/RESULTS.md).

[Exact commands, failure, costs and review](../artifacts/runs/train-catalogue-composition-20261018/RESULTS.md).
