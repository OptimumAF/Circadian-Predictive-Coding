# Isolated fresh-task probe for fixed-input memory

For an evaluation-free demonstration-fit/conflict question, use the separate
[training-only diagnostic](nearest-training-diagnostic.md). It performs one private
delayed append through the same runtime and accepts no test packet. The original
18-row23diagnosis fits exactly with no exact-label conflicts; native success does
not follow from that result.

The existing `run_continuous_probe` now accepts an exact
`NumpyNearestContinuousBCLearner` with an explicit
`FixedInputNearestContinuousBCPolicy`. This fills the missing conventional memory
control alongside the existing BP, PC, neutral CPC and active CPC probes. It uses
the same delayed-label runtime and reserved-role plan; no new runner is required.

```python
from src.adapters.continuous_probe import ContinuousProbePlan, run_continuous_probe

# Supply an already qualified, nonempty fixed-input TRAIN memory checkpoint.
# These are separately qualified probe_train23 / score-only probe_test24 episodes,
# encoded with the same executed taskspace recipe as the source. No relabeling.
plan = ContinuousProbePlan(reserved_train23, reserved_test24, updates=1)
report = run_continuous_probe(
    memory_source, plan,
    training_readouts=(original_pick17, original_pick19, original_pick20),
)
if not report["complete"]:
    # Preserve attempted/returned work and unknown outcomes; do not retry blindly.
    print(report.get("error"), report["work"], report["progress"])
else:
    print(report["metrics"], report["work"]["nearest_retrieval"])
```

The snippet grants no data/model/native experiment authority. Reuse existing
[reserved input qualification](reserved-robot-probes.md) and
[fixed-input memory](taskspace-memory-control.md). It also composes with detached
checkpoint observers in the existing memory stream; each probe has its own declared
work allowance, separate from source TRAIN and serving. An optional endpoint
observer follows the existing synchronous borrowed-readout contract and owns its
separate readout/native costs. It must not retain or promote the private learner.

Exactly one append is allowed. Probe labels wait for their original delayed
arrival; only `probe_train` reaches the disposable candidate. `probe_test` and up
to three original TRAIN episodes are score-only. Final inputs are unavailable.
The original source and stable actor remain unchanged, and the report returns no
adapted learner. Full state, arrays, policy, owner identities, original runtime
budget/clock, receipts, and earlier scores are checked before more work.

The original 256-prototype ceiling applies: source rows plus the complete probe
packet must fit before any fork or score. There is no eviction, deduplication,
refit or repeated append. Multi-update plans, gradient diagnostics, zero-velocity
controls, fitted-input nearest sources and an empty source refuse before private
work. Four old readouts still refuse; observing three does not establish retention
of a fourth task. Existing neural defaults and diagnostic options are unchanged.

Why this: the working memory controller should be tested with the same separated
reserved inputs as the neural families, without importing score labels into
learning or treating prototype growth as free optimization.

`work` distinguishes append attempts, native returns, committed receipts and
unknown work. An acknowledgement lost inside the native delegate leaves completed
updates0 and `unreturned_append_possible=true`, with the attempted append slot
permanently spent. A failure after runtime commit retains completed1. Neither is
retried automatically. `optimizer_updates`, `gradient_evaluations` and
`dense_forward_passes` are0 for nearest; appends and distance calculations have
their own counts. Prediction receipts give returned distance candidates/feature
values, while failed attempts retain source-derived upper bounds.

Prototype payload fields are logical numeric lower bounds, excluding snapshots,
validation copies, temporary arrays, Python containers and process memory. Paid
counters live privately; exposed views cannot erase their costs. Each returned
score and both complete checkpoints retain nested identity/value guards. Tampered
or replaced evidence makes the probe incomplete, with known paid costs preserved.

An append need not improve a fresh task. Exact-input conflicts keep the original
earliest-row tie rule; the new record remains stored and paid. The literal tie
test correctly reports zero acquisition gain.

Verified scope: 36 literal cases, including actual private acquisition/old-score
retention, role/capacity/counter/owner/evidence refusal and four tiny neural
compatibility cases. This is software correctness, not a recorded robot outcome,
new physical task, reliability, recovery, generalization, matched computation or
circadian advantage. See [commands, costs, retained failure and review scope](reports/nearest-reserved-probe-software-20261101.md).

One actual [reserved23/24 observation](reports/nearest-reserved-probe-study-20261101.md)
has now completed: score-only24error.03654645568→.00895024504 after one private
18-row append; three stored old TRAIN scores and source remain unchanged. All
preparation/retrieval/copy costs are recorded. The [paired native follow-up](reports/nearest-reserved-native-study-20261101.md)
then completed: reset24fails before and after at50steps, while old17/19/20 still
succeed afterward at15/22/15steps. Imitation gain did not produce native24success.
Push retention remains unmeasured. The existing frozen native port accepts this
fixed-input nearest source for score-only reset24, with a detached inference fork.

```python
from pathlib import Path
from scripts.maniskill_continuous_smoke import run_frozen_rollout
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe

# Run only within a separately frozen native protocol and its work allowance.
outcome = run_frozen_rollout(
    Path("artifacts/runs/your-budgeted-case/before24.json"),
    learner=memory_source,
    probe_test_seed=24,
    action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
)
```

Use the same call with a synchronously borrowed probe readout at each endpoint
of the existing isolated probe. Do not retain the private readout or promote it.
Its qualified fixed-input TRAIN policy remains the original policy; probe updates
still enter only through the separate probe runtime. The output must be new.
Nearest guard31/37 and final29 remain refused. The reserved route admits no
corrections, collection, observation clocks, actors, projections or interventions.
Original BP/PC, nearest TRAIN and default routes remain available.

The reserved route pins complete source and private inference prototype values,
counters, policies, specifications and identities across observations, prediction,
history, native controls, work observation and cleanup. Returned controls stay
paid when later proofs fail. A failed work read or corrupted counter makes the
capture incomplete; independent `nearest_control_calls` survive.
`nearest_work_scope="validated_native_work"` marks checked native metadata;
`outer_call_bounds_native_metadata_untrusted` marks conservative outer-call
physics bounds, including unknown failed-call work, rather than measured physics.
Never treat an incomplete capture or a bound as a successful robot outcome.

[Scoped software evidence and failures](reports/nearest-reserved-native-software-20261101.md):
64 final literal fake-port cases pass. The separately budgeted native follow-up
above preserves the negative reserved24 result and successful old-task endpoints.

```powershell
.venv\Scripts\python.exe -I -B artifacts/runs/nearest-reserved-probe-software-20261101/check.py final
# The recorded phase is SPENT: do not rerun it or the compatibility models blindly.
# Future checks require remaining or separately justified prospective fixture limits.
```

## Isolated probes from bounded memory

`run_retained_nearest_probe` uses the existing delayed-label private runtime for
an exact retained nearest learner. It permits one append, four old TRAIN readouts,
and separate fresh-training and readonly probe-test scores. The source and stable
actor remain unchanged; the result contains reports and costs, never an adapted
learner for deployment. Native success and return recovery require separate proof.

```python
from src.adapters.continuous_probe import ContinuousProbePlan, run_retained_nearest_probe

# Use the original strict reader/consent binding. Its generic diagnostics=True
# profile is neural; the retained route explicitly has no gradient diagnostics.
plan = ContinuousProbePlan(loaded.training, loaded.evaluation, updates=1)
report = run_retained_nearest_probe(
    detached_retained_source, plan, training_readouts=old_train_episodes
)
```

The same fixed `run_sequence(..., fixed_input=True, retention=rule,
max_prototypes=capacity)` can provide the detached source at its final observer.
Supply owned, already released role-bound episodes; the adapter does not load or
qualify files. `probe_train` supplies the single private update, `probe_test` and
old TRAIN episodes supply readonly scores. None supplies replay, promotion,
tuning or final-evaluation decisions. The default public memory CLI still refuses
retention combined with `--training-probe` or `--development-probe`; this new
explicit API does not widen those routes or reserved native reset authority.

Reports distinguish lifetime training exposure from retained search rows, current
numeric/ID payload, discarded/duplicate content, per-role score rows and append/
retrieval work. A lost update return remains spent and leaves the resulting bank
unknown. Every declared append is checked against the existing pure retention
rule. Original state, rule owner, IDs, arrays, runtime clocks, budgets, receipts
and earlier paid results remain guarded; value-equal rule replacement refuses.

New focused checks: `python -m pytest -q tests/test_retained_memory_probe.py`.
Ruff and win32/Linux mypy cover that file plus `src/adapters/continuous_probe.py`
and `src/adapters/continuous_nearest_probe.py`. Existing unbounded probes and
BP/PC/CPC mathematics are reused. Extend private adaptation through these same
ports; keep bounded state accounting separate from new role or native authority.

The fixed FIFO128 observation completed: fresh23 training error reached0; readonly
24 error decreased. Old Pick17/20 fit improved, Pick19 stayed0, and Push17 fit
worsened. Source/actor unchanged. Original failed caller-profile attempt and all
costs remain preserved; a prospective correction preceded the completed case.
[Executed results, costs, review and limits](reports/retained-memory-probe-20261102.md).
These are imitation scores; bounded native task success and recovery remain open.

### Explicit old-task return

The retained API also accepts `return_training=old_train_episodes[0]`. The packet
must equal exactly one declared old TRAIN readout, including original role,
context, identity and values. One detached child learns that original packet
after the private fresh-task update. The borrowed checkpoint stays frozen.

```python
report = run_retained_nearest_probe(
    detached_retained_source,
    plan,
    training_readouts=old_train_episodes,
    return_training=old_train_episodes[0],
)
```

This explicit option returns `probe` and `return_phase` reports. Both must complete
and their shared zero-return checkpoint must agree before root `complete` is
true. The original private result and permanent charges remain recorded if return
fails. Separate return budget and logical-clock owners are admitted before any
private work; there is no renewal. The optional observer receives `before_probe`,
`after_probe` and `after_return`, using the same guarded readonly loan. Reports
include fresh/old readouts, original role-bound receipts, retained memory and
attempted work. Returning old TRAIN rows is additional data exposure even when
the runtime's distinct replay-receipt field is zero. The adapter exports no model.

Focused checks live in `tests/test_retained_memory_return.py`; qualification slots
and their cumulative limits in `artifacts/runs/retained-memory-return-20261102/`
are spent evidence, not an instruction to rerun them. Native recovery requires
its separately frozen observation; fitting recorded actions alone is insufficient.

The one fixed FIFO128 observation recovered old Pick17 in15controls and kept
new23/old19 successful18/22controls. Pick20 and Push17 still failed50. The first
attempt exceeded an engineering input-check byte limit before any native scene;
a prospective same-input amendment preceded the completed case. Both attempts,
costs and review limitations remain in the
[return report](reports/retained-memory-return-20261102.md). These are known-reset
results from a conventional nearest control, not generalization or CPC benefit.
