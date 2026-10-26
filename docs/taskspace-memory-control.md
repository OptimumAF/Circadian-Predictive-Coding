# Fixed-input taskspace memory control

The existing bounded nearest learner can explicitly keep already-scaled inputs.
`FixedInputNearestContinuousBCPolicy(spec)` selects identity copies and stores
`observation_standardization=None`. The original `NearestContinuousBCPolicy`
still requires frozen fitted statistics. Neither mode refits during an update.

Why this: executed taskspace12 already declares physical scaling. Inventing
fitted statistics to enter the memory adapter would change the information
geometry used by the conventional neural control.

## Composition

```python
from src.adapters.continuous_stream_demo import run_continuous_demo
from src.core.pick_cube_action_history import action_history_training_episode
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe

recipe = PickCubeTaskSpaceHistoryRecipe(mode="executed")
a = action_history_training_episode(qualified_train_a, recipe)
b = action_history_training_episode(qualified_train_b, recipe)
report = run_continuous_demo(
    a, b, fixed_input=True, checkpoint_observer=observer,
)
```

Use the existing qualification readers for saved TRAIN inputs. The stream fits
nothing, appends A, withholds B labels until tick2, then appends B once to its
private candidate. Source and stable actor retain A. Checkpoint observers receive
detached readouts; they may call the existing `run_frozen_rollout` with the exact
executed taskspace recipe. Native serving requires a pure fixed-input learner;
fixed-input actors, omitted/zero/47D histories, fitted+taskspace, corrective/query,
visual, clock and intervention combinations are refused before scene creation.

The existing full private prototype guards, actual float32 command history,
source/actor ownership, permanent update charges and bounded receipt checks
remain authoritative. Mutation after a paid control preserves that control and
refuses more work. Capacity is at most256 prototypes; there is no eviction.
Squared L2 retrieval retains the earliest row on an exact tie, including conflicting
duplicate inputs. Memory and prediction distance work are reported explicitly.

## Verification and limits

```powershell
python -m pytest -q tests/test_taskspace_memory_control.py tests/test_taskspace_memory_stream.py tests/test_nearest_continuous_bc.py::test_should_require_exact_valid_dimension_matching_frozen_statistics
python -m ruff check src/adapters/numpy_nearest_continuous_bc.py src/adapters/continuous_stream_demo.py src/adapters/continuous_stream_sequence.py scripts/maniskill_continuous_smoke.py tests/test_taskspace_memory_control.py tests/test_taskspace_memory_stream.py
python -m ruff format --check src/adapters/numpy_nearest_continuous_bc.py src/adapters/continuous_stream_demo.py src/adapters/continuous_stream_sequence.py scripts/maniskill_continuous_smoke.py tests/test_taskspace_memory_control.py tests/test_taskspace_memory_stream.py
python -m mypy --platform linux --follow-imports silent src/adapters/numpy_nearest_continuous_bc.py src/adapters/continuous_stream_demo.py src/adapters/continuous_stream_sequence.py scripts/maniskill_continuous_smoke.py tests/test_taskspace_memory_control.py tests/test_taskspace_memory_stream.py
```

Recorded checks ran as separate29+10 cohorts per platform under frozen local
budgets; consult [receipts and review](../artifacts/runs/taskspace-memory-control-20261020/REVIEW.md).
These scoped software checks do not establish robot success, generalization,
safety, matched computation or a circadian benefit. New serving modes need their
own declared input contract, role/ownership checks and finite fixture budget.

## Observed usefulness

The same A15+B22 memory completed a frozen TRAIN17 return in15 controls, selecting
A prototypes0..14. A separate paired TRAIN19 check starts from identical raw
observations: A-only memory fails50 controls; after one delayed22-row append,
B succeeds22 controls, selecting B prototypes15..36. Source/actor remainA-only;
native readouts are frozen. The A result is reused from the same deterministic
construction, not a new A/B/A rollout. Numeric prototype payload grows1920→4736 B;
distance candidates are750 in the failed warmB return and814 in the successful
postB return. Other memory, diagnostics, validation and native costs are additional.
[Full B evidence](../artifacts/runs/taskspace-memory-b-adaptation-20261020/RESULTS.md).

This is supervised memory adaptation on known training contexts. All neural
negatives remain. It demonstrates no CPC advantage, new physical task,
unseen-context reliability, safety or matched total computation.


## Same-information Push control

The existing `scripts.push_cube_continuation_study.roll_push` also accepts the
exact fixed-input TRAIN nearest learner on executed cubeD12. It pins the native
state, policy and prototype-array owners, retains each bounded retrieval receipt
before validation, reports candidates and feature visits, and closes on failure.
The BP route and its default report remain unchanged.

For six distinct qualified TRAIN episodes, the existing `run_sequence` accepts
`fixed_input=True, maximum_releases=5`. The default remains three releases;
five requires the explicit fixed-input opt-in. Capacity remains256 rows. Why:
six physical packet identities must survive admission, rather than being joined
into an invented trajectory. One new finite runtime budget pays five delayed
updates at ticks2/5/8/11/14; original collection authority and charges persist.

The fixed `scripts.push_cube_memory_control` composes the accepted198-row input
reader, this stream, and existing Push/Pick frozen serving. Warm A15; release
B50, TRAIN20 15, expert Push18 and the two50-row corrections. Detached readouts
at80 and198 prototypes each run Push17 and Pick17/19/20. It retains all negatives,
paid partial reports and prior evidence, binds the same resets, and charges
retrieval/storage/native costs. No fit, teacher query, promotion, final access,
probe update, new learner algorithm or orchestration framework is introduced.

Execute only under a separately frozen small protocol after its software gates:

```powershell
python -m scripts.push_cube_memory_control --output artifacts/runs/NEW-REVIEWED-STAGE/run.json
python -m pytest -q tests/test_push_cube_continuation_study.py tests/test_taskspace_memory_stream.py tests/test_push_memory_serving.py tests/test_push_memory_control.py
python -m ruff check scripts/push_cube_memory_control.py scripts/push_cube_continuation_study.py src/adapters/continuous_stream_sequence.py tests/test_push_memory_serving.py tests/test_push_memory_control.py
python -m ruff format --check scripts/push_cube_memory_control.py scripts/push_cube_continuation_study.py src/adapters/continuous_stream_sequence.py tests/test_push_memory_serving.py tests/test_push_memory_control.py
python -m mypy --platform linux --follow-imports silent scripts/push_cube_memory_control.py scripts/push_cube_continuation_study.py src/adapters/continuous_stream_sequence.py tests/test_push_memory_serving.py tests/test_push_memory_control.py
```

The output directory must exist and all main/six Pick paths must be fresh.
See the fixed [protocol and evidence](../artifacts/runs/push-memory-control-20261021/PROTOCOL.md).
The accepted BP128 result is descriptive and nonconcurrent: same information,
unequal objectives/exposure/order/storage/computation. Neither correctness nor
this comparison establishes a CPC advantage, unseen reliability or readiness.


The fixed198 control has now executed: Push17 changes failure50→success18;
Pick17/20 retain success15 at both80/198 checkpoints, while Pick19 remains
failure50. Successful Push selects the stored expert18 rows. Five delayed
charges/six total appends retain198rows, source/actor remainA15. Eight scenes
use228controls/1140known CONTROLphysics/29,804distancecandidates; numeric bank
10,240→25,344 B/owner, with additional copies/SDK/IO costs. This is known-reset
acquisition/retention, not unseen transfer or matched compute. All prior
negatives remain. [Full observed results and review](../artifacts/runs/push-memory-control-20261021/RESULTS.md).


## Explicit frozen Push reset

`roll_push(learner, label="memory-reset19", seed=19)` uses the existing TRAIN
Push port on seed17,19 or20. The default remains17. Both BP and fixed-input
nearest serving reject non-native integers and unsupported seeds before model
snapshot, clock or native work. The helper always uses TRAIN and exposes no role
selection; this
parameter supplies no evaluation or label authority. History, ownership,
retrieval receipts and the50-control horizon remain unchanged.

```python
from scripts.push_cube_continuation_study import roll_push

# learner must already own released TRAIN cubeD12 inputs under a fixed budget.
report = roll_push(learner, label="declared-memory-reset19", seed=19)
```

Focused seed coverage: `python -m pytest -q tests/test_push_cube_continuation_study.py tests/test_push_memory_serving.py tests/test_push_frozen_seeds.py`.
Scoped lint/format/types cover the existing study and new seed test module.
The separate [fixed fresh-reset protocol](../artifacts/runs/push-memory-transfer-20261021/PROTOCOL.md)
reuses accepted seed17/Pick results, rebuilds the same198 bank and observes
Push19/20 without new labels. It does not complete the different corrected-BP
guard31/37 question, whose negative/interrupted evidence remains preserved.


The fixed fresh-reset readout is independently accepted: unchanged198-row
memory succeeds on Push19 in19controls and Push20 in23, with no new labels or
updates. Together they use42controls/210known control-physics/8316distance
candidates. See [results and limitations](../artifacts/runs/push-memory-transfer-20261021/RESULTS.md).
This supports these two declared TRAIN transfers; population reliability,
held-out task performance and CPC benefit remain unproven.

## Complete B22 coverage and known-reset repair

`run_sequence(a, (b50, train20, push, old_corrections, current_corrections, b22),
fixed_input=True, maximum_releases=6)` retains all198 original rows before the
complete22-row demonstration. The3-release default, explicit5 behavior,
256-row capacity and earliest-row ties remain unchanged. No statistics are fit.

The independently accepted220-row candidate succeeds on Pick19 in22controls;
Pick17/20 and Push17 retain success in15/15/18. All are known TRAIN resets;
the stable source/actor remainA15. Extra storage/data/appends are paid, and this
is not a matched-compute comparison or evidence of neural/CPC generalization.
Fresh Push19/20 transfer of the220 bank has not been measured.

Focused checks: `python -m pytest -q tests/test_b22_memory_repair.py tests/test_push_memory_control.py`.
Scoped Ruff/format/mypy cover the sequence and these modules. Reuse the
[frozen protocol and executed commands](../artifacts/runs/b22-memory-repair-20261021/RESULTS.md)
instead of rerunning the spent native case. Add another release only for a
demonstrated coverage need, preserving complete identities/order/charges.
