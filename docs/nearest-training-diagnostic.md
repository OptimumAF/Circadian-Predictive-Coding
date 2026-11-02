# Training-only memory learnability diagnostic

`run_nearest_training_probe` performs one delayed private append through the
existing continuous probe/runtime, with **no evaluation packet**. It answers
whether a supplied demonstration is fit and whether exact input/label conflicts
or earliest-row ties prevent fit. It does not score transfer or run a simulator.

```python
from scripts.push_cube_reserved_probe_inputs import load_reserved_training_input
from src.adapters.continuous_probe import run_nearest_training_probe

# Only within a separately frozen input/model/diagnostic work allowance.
# source is the qualified, nonempty fixed-input TRAIN nearest checkpoint.
training, receipt = load_reserved_training_input()  # original23 only; no24open
report = run_nearest_training_probe(source, training)
if report["complete"]:
    print(report["metrics"]["training_fit_gain"])
    for point in report["checkpoints"]:
        print(point["scores"]["training"]["residual"]["overall"])
        print(point["scores"]["training"]["selected"])
else:
    print(report.get("error"), report["work"], report["progress"])
```

The explicit call owns one new private update allowance; do not call it repeatedly
without accounting for all previous work. Only arrived `probe_train` labels reach
the disposable candidate. Source/stable actor, complete policies/prototype values/
counters/identities, original receipts, budgets, clocks and earlier paid evidence
remain guarded. No adapted learner is returned or promoted. Lost append returns
retain permanent attempts and unknown work. An incomplete result is not accepted.

Inputs require an exact fixed-input nearest TRAIN owner and1..32same-spec D12
`probe_train` rows, within original prototype capacity. The existing consistency
kernel also requires four normalized axes and binary teacher gripper targets.
Ordinary TRAIN, probe_test and final packets refuse; fitted-input/empty sources
refuse. This separate entrypoint leaves the original pair API, neural diagnostics
and nearest two-packet defaults intact. The qualified reader preserves original
train23 byte/decoded-value/role/identity/history checks without opening test24.

Only `training_fit_gain=before−after` is reported, using normalized halfMSE
(lower better). Readouts retain original `probe_train` role, selected global
prototype index, source/probe origin and local row. Earliest prototype wins ties;
a private append cannot override an earlier conflicting exact input. Tests also
distinguish raw float64 label conflicts from equal native32 commands.

`training_consistency` contains the packet's internal exact pairs once and each
exact probe/source pair once. Existing kernels process source chunks plus the
packet, always at most100rows. They compute other nonexact/source-source pairs
too; those costs remain charged. Block `work` contains **cumulative** counters:
use final `work.training_consistency`, do not sum block counters. No distance
threshold, deduplication, eviction, policy/seed/metric selection or kernel change.
Snapshot/copy/container/RSS costs are additional to distance/append/pair counts.

The [original train23 diagnosis](reports/nearest-training-diagnostic-20261101.md)
fits all18rows exactly after one private append: halfMSE.02021162526→0, no exact
probe/probe or probe/source conflicts. This does not establish closed-loop23
success or explain the [earlier native24 negative](reports/nearest-reserved-native-study-20261101.md).
The [closed-loop qualification](reports/nearest-training-native-20261101.md) now
passes on original reset23: failure at50steps before the append becomes success
at18steps afterward, with all18TS12 model inputs/native actions exactly matching
the original demonstration. Fullraw42 equality/generalization/reliability is not
established; the earlier24negative remains unchanged.

An optional `endpoint_observer` now borrows each private before/after readout
synchronously through the existing guarded observer. Use
`run_frozen_rollout(..., probe_train_seed=23,
action_history_recipe=PickCubeTaskSpaceHistoryRecipe())` inside its supplied
`boundary(read_ports)` call. This narrow route requires fixed-input nearest and
preserves original probe_train/context, with new inference ID
`pick-cube-probe-train-smoke-23`; it never reuses the teacher ID or qualifies new
supervision. All ordinary/24/guard/final/neural rights stay unchanged.

The observer owns separate native work limits, exclusive outputs and complete
paid-prefix evidence. Pass an evidence guard as `check_boundary`; bind returned
native records before later work and refuse corrupted/replaced/incomplete prior
captures. A complete negative task return is valid. Keep only reports, never the
borrowed learner, and never promote it. Source/actor/fullnested state/identity/
budget/clock/receipt proofs enclose the callback. See the recorded protocol/source
for a complete one-shot example; its executed markers and raw-read grants are
SPENT. The ordinary first-party memory command now exposes this explicit mode
with `--training-probe ORIGINAL23_JSON` and optional `--native`. The CPU workflow
has executed; the new native CLI composition has scoped software qualification
and reuses the original native-port evidence.
[Inputs, commands and limits](taskspace-memory-demo.md#explicit-private-training23-probe).

Focused checks (recorded labels SPENT; future runs need remaining prospective caps):

```powershell
.venv\Scripts\python.exe -I -B artifacts/runs/nearest-training-diagnostic-20261101/check.py final
.venv\Scripts\python.exe -m ruff check --no-cache src/adapters/continuous_probe.py src/adapters/continuous_nearest_probe.py scripts/push_cube_reserved_probe_inputs.py tests/test_nearest_training_probe.py
.venv\Scripts\python.exe -m mypy --no-incremental --follow-imports=silent src/adapters/continuous_probe.py src/adapters/continuous_nearest_probe.py scripts/push_cube_reserved_probe_inputs.py tests/test_nearest_training_probe.py
```

Extend safely by using a new declared diagnostic scope through these existing
ports. Do not widen roles, prototype/100-row limits, automatically append during
readouts, import final/test labels, release candidates or invent a new runner.

## Bounded-memory original23 frozen readout

The existing `run_frozen_rollout` now supports an exact retained nearest learner
with `probe_train_seed=23` and the executed task-space12 recipe. This is readonly
native scoring on the original training reset. It creates a separate frozen fork;
it does not collect targets, update the source, grant a new reset, or promote a
private learner. The retained bank and lifetime training exposure stay separate.

Use it inside the existing retained private probe's synchronous endpoint loan:

```python
from scripts.maniskill_continuous_smoke import run_frozen_rollout
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe

reports = {}
def endpoint(readout, phase, guard):
    def read_ports():
        reports[phase] = run_frozen_rollout(
            fresh_paths[phase], learner=readout, probe_train_seed=23,
            action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
        ).report
    guard(read_ports)
```

The caller owns fresh captures, consent/input qualification, prospective native
budgets and report validation. Do not retain the borrowed learner. Ordinary TRAIN
native readouts remain available; retained test24, guard, corrective-label,
projection/decoding/roll-in/actor/visual variants remain refused. All four old-task
outcomes must be reported when testing retention, including existing failures.
Saved imitation scores do not establish native success or recovery.

The existing native-call witness now closes after the final retained public
getters. It binds the original role/reset/history/recipe/spec as well as work,
source/fork state, array ownership and paid call counts. Lost returns remain
possible work; broken metadata uses conservative original call bounds and
cannot authorize further steps. Both retained and nearest call summaries describe
the same native controls; do not add them together as extra physics.

Focused checks: `python -m pytest -q tests/test_retained_training_native.py`.
Ruff/format and win32/Linux mypy cover that module and
`scripts/maniskill_continuous_smoke.py`. No SDK is used by these literal checks.
Keep new readout modes explicit and extend these same ports rather than adding
another training or rollout framework.

One frozen FIFO128 native observation now changed original23 from failure50 to
success18 after one private18-row append. Pick19 retained success22. Pick17,
Pick20 and Push17 were already failures before and remained failures50; this
is not new native forgetting. Source/actor/charges stayed unchanged. See
[all outcomes, commands, costs and review limits](reports/retained-memory-probe-native-20261102.md).
This proves one known-reset acquisition/old-task retention case, not broad
retention, recovery, generalization, safety, circadian benefit or release readiness.


## Explicit retained TRAIN-only learnability

`run_retained_nearest_training_probe` now supports the bounded nearest learner
without an evaluation packet. It uses the existing delayed single-update runtime
and reports TRAIN fit/interference, consistency and current content coverage.

```python
from src.adapters.continuous_probe import run_retained_nearest_training_probe

# Within a separately declared input/model/diagnostic allowance:
# source: nonempty exact fixed-input TRAIN retained learner, already released.
# training: same-spec 1..32 TS12 probe_train rows, qualified by existing caller.
# old_training: up to four same-spec distinct original TRAIN readouts.
report = run_retained_nearest_training_probe(
    source, training, training_readouts=old_training,
)
if report["complete"]:
    print(report["metrics"]["training_fit_gain"])
    print(report["metrics"]["old_retention_loss"])
    point = report["checkpoints"][-1]["scores"]["training"]
    print(point["selected"], point["content_coverage"])
else:
    print(report.get("error"), report["work"], report["progress"])
```

The original single-role input reader `load_reserved_training_input()` opens only
qualified23TRAIN within its own input budget. This API performs no IO. It accepts
no evaluation object and reports evaluation_id=None/probe_test score rows0;
`train`, `probe_test` and `final` refuse as private-training roles. Old readouts
must be TRAIN; input qualification/release authority still belongs to the caller.
One private update is permanently charged, including lost acknowledgements.

Selections use current content IDs, so `matches_source` means membership in the
source bank at admission, not lifetime training history. `matches_incoming` means
membership in the supplied packet, including duplicates before append; both flags
can be true. Evicted historical content is not reconstructed. Coverage separates
query occurrences/distinct IDs, stored content/actual selected row, and equal-input
conflicting targets. Earliest ties and retention math remain unchanged.

The new `work.retained_content` records attempted/returned hashes/materialized
numeric upper bounds, mapping rows and semantic ID/feature visit bounds. It
reuses prediction receipts and adds no nearest forwards. Materialized numeric
bounds exclude Python concatenations/containers and other validation/snapshot
copies; total process memory/cost is not inferred from them. All earlier source,
actor, budget/clock, paid score/receipt and ownership guards remain.

Use the same call synchronously inside an existing stream checkpoint observer;
retain the returned report only, return before the observer releases its borrowed
source, and never retain/promote that learner. No new runner or loader is needed.
Oldappend-only and paired/return/native entries remain separate; the ordinary CLI
has no automatic new mode or native authority from this API extension.

[32-case software evidence and costs](reports/retained-training-learnability-20261102.md)
include actual fitting and forgetting plus failure/isolation checks. Independent
scoped source/result review accepted; reporting-limit failures are retained.
Original-input/native usefulness of this new diagnostic remains unmeasured.


The [fixed bounded TRAIN-only diagnosis](reports/retained-training-diagnosis-20261102.md)
now has original-input evidence: reservoir128 retained and selected14of18new
contents, with newTRAINloss.05217->.03155 and mixedoldinterference. No exact
input/target conflict involving the newpacket was observed. This supports a
coverage explanation, not native success. The firstcapture/extractionfailures,
prospectiveidenticalengineeringrequalification, allcosts and independentreview
limits remain recorded; neither readonly24 nor SDK was used.
