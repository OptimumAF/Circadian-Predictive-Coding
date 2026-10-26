# Simultaneous released TRAIN corrections

`run_joint_readiness_corrections` makes the fixed joint head-GN recipe available
without importing historical experiment files. It composes existing actor/shadow
and learner ports; legacy single-batch correction and separate replay APIs keep
their existing contracts.

```python
from src.adapters.joint_readiness_corrections import run_joint_readiness_corrections

# source: an owned native BP learner from the existing TS12 stream.
# reference: already-qualified raw42 observed-grasp TRAIN17 ReadinessStageBinding.
# corrections: arrived GoalConsistentObservedGraspCorrectionBatch for TRAIN19.
# Stage one: explicit None admits only released A+B; required argument.
first = run_joint_readiness_corrections(
    source, reference, corrections, None, now=99,
)
# Only a complete first stage provides the source for stage two.
assert first.report["complete"]
# curated: arrived CuratedReadinessBatch for TRAIN17.
result = run_joint_readiness_corrections(
    first.candidate, reference, corrections, curated, now=199,
)
if result.report["complete"]:
    adapted = result.candidate  # detached; source and serving actor are unchanged
else:
    assert result.candidate is None
    print(result.report["error"])  # retain the complete paid partial report
```

Inputs use existing qualification, collection and curation boundaries. The
adapter does not load files, grant expert qualification or query a teacher.
Raw reference bindings are trusted Python input, just as in the existing stream;
external reference files must first pass their existing qualified reader.

## Explicit old-information rehearsal control

The Python preparation and runtime APIs accept `old_information_control=False` by
default. Opting in repeats all15 rows of the original qualified A17 reference in
their original order, after the original A/B/curated-A rows. This enables the matched-row
comparison with the130-row retained20 recipe without adding TRAIN20 labels during
the new eight return updates. The source learner may already have acquired20.
This mode does not erase previous knowledge or qualify any external data.

```python
control = run_joint_readiness_corrections(
    source40, original_a17_reference, arrived_b19_corrections, arrived_curated_a17,
    now=1100, old_information_control=True,
)
# Only a complete result returns the detached control learner.
if not control.report["complete"]:
    print(control.report["error"])  # keep the paid partial receipts and work
```

Selected mode requires an exact bool, original15-row A reference, present arrived
curation, no `retained_reference` and no `sampling`. Unknown roles/authority,
recipes, unarrived labels and non-nativefloat32 reference commands refuse before
projection/private work. The original base128-row gate remains; with A15/B<=50/
curated-A<=50, selected base<=115 plus15repetitions<=130. No arbitrary count,
seed, stage or favorable-row selection. Existing omitted/False calls preserve
reports and delegated kwargs; the retained20 path is unchanged.

The additional `repeated_a` group preserves original episode/context/role/step,
features/raw observation, target, actual command, stage, teacher definition,
provenance and unknown source clocks. Each new matrix row has its own `row_index`,
`source_row_index`0..14 and `source_kind="old_reference_repetition"`; it is never
relabeled as TRAIN20. Its causal projection restarts history and is recomputed,
with15additional pure command validations counted separately. Conditional
`repeated_reference_*` work and per-group exposure distinguish repeated matrix
rows from new information. New derived packet availability does not rewrite
unknown original acquisition clocks. No mixed-context teacher episode is created.

The selected report contains a sealed `old_information_control` definition,
source identity and row-index mapping. Corruption or equal-valued replacement
refuses more work; existing source/actor isolation, original eight-update budget,
permanent charges and paid prefix are preserved. No learner is returned on
incomplete work. No probe/final/guard labels, native simulator or promotion occurs
as part of this software gate. Synthetic68focusedchecks/platform and independent acceptance passed;
The separately frozen native control succeeds17/19 but fails20 at50controls; saved
retained20 succeedsall3. All control-minus-retained20 metrics at20=-1,17/19zero;
control-minus-old115 allzero, independently accepted. Equal optimization shape does
not establish matched unique data, contextweighting, memory, diagnostic cost or
total computation, nor a pure label-causal/CPC/generalization claim.
[Native comparison, costs and review](../artifacts/runs/native-old-information-rehearsal-20261021/RESULTS.md).
[Executed software evidence, commands, costs and review](../artifacts/runs/rehearsal-old-information-control-20261021/RESULTS.md).

## Frozen acquisition validation

After the accepted direct-acquisition recipe below, exact40/2776 passed the fixed
scoring-only development guard31 in22 controls. The entire source/acquisition
reports reproduce before guard content is read; further training is closed. All
three scores improve by1 versus the cached complete source32 guard31 prefix.
This single reused reset is not untouched final evaluation or a reliability
estimate, and does not complete the original two-reset requirement (37 remains
unobserved after its preserved timeout). No promotion follows automatically.
[Execution, costs, provenance and independent review](../artifacts/runs/train20-acquisition-guard31-20261021/RESULTS.md).

## Direct acquisition using the existing retained-reference API

A qualified TRAIN20 binding can be supplied directly from the accepted32/1736
source, before the TRAIN20-only continuation. The existing eight-update130-row
call then produces40/2776; no new mode or API is needed. The frozen case succeeded
on17/19/20 in17/20/20 controls, with slight increases in all three old-group fit
losses. This is endpoint retention/acquisition in known TRAIN contexts, with920
old rehearsal exposures plus120 TRAIN20 exposures; it does not prove retention
through each intermediate update or generalization. Input qualification, unknown
source clocks, source/actor isolation and paid receipts retain the same contract.
[Executed case, commands, costs and independent review](../artifacts/runs/train20-acquisition-rehearsal-20261021/RESULTS.md).

## Optional retained TRAIN20 reference

To retain an already acquired TRAIN20 demonstration while old supervision returns,
pass `retained_reference=` to `prepare_joint_readiness_data` or
`run_joint_readiness_corrections`. This is an exact `ReadinessStageBinding` for the
separately qualified `pick-cube-observed-grasp-teacher-20` raw42 TRAIN episode;
external input still requires the existing observed-grasp teacher reader.

```python
# retained20: ReadinessStageBinding(qualified_raw20, original_recorded_stages)
result = run_joint_readiness_corrections(
    source, reference, corrections, curated, now=1100,
    retained_reference=retained20,
)
```

The original sources retain their128-row combined limit; with the optional source,
total rows are bounded at130, allowing the explicit115+15 case. Every source has
at most50 rows. The native GN kernel and recorded-work validator share this finite
130-row limit;131 is refused before forward/solve. Numerical equations, damping,
step bound and strict original-loss decrease are unchanged.

The new `retained_train20` group follows the original groups and starts its own
actual-command history. Original unknown clocks remain unknown; packet `now`
does not rewrite them. Its teacher commands/targets and qualified source kind stay
distinct from provisional correction authority. Original arrival gates precede
all projection/fork. Optional metadata and original caller input join integrity
guards; failure preserves paid prefixes and yields no candidate.

Omitted or explicit `None` preserves the original schema, rows, objective and
delegated keyword shape. Original warm17/new19 stage diagnostics remain ordered;
admitting the collected observed-grasp20 binding does not widen that pair.
Eight updates add8T retained row exposures and2T diagnostic prediction rows for
T retained examples. Preparation and command validation, input/array copies,
solver work, owner storage and RSS are additional. Actual validated exposures
appear in `group_training_exposures`; planned scope is not a paid success count.
The software checks alone do not establish physical retention. One separately
frozen native case now recovers known17/19 while preserving acquired20: all three
succeed in17/19/16 controls, versus old-only17/19 successes and20failure. Old-group
imitation errors worsen relative to old-only, and added120 retained exposures plus
diagnostics/solver work cost more. Original20-only forgetting remains. These known
same-task resets establish no unseen-task, reliability or CPC benefit.
[Executed comparison, commands, costs and independent acceptance](../artifacts/runs/native-train20-rehearsal-20261021/RESULTS.md).
Windows software RSS coverage remains unproven; its resource parent is open.

## Rules and behavior

- Exact BP TRAIN/common-MSE source, default identity TS12 preprocessing (no fitted
  standardization), H1..64; PC/CPC and binary-gripper overrides are refused.
- Exact observed-grasp TRAIN17 reference, goal-consistent observed TRAIN19 batch,
  optional explicit curated TRAIN17 batch (None selects A+B), same raw42 normalized four-axis spec. Each has1..50 rows,
  total<=128. Fixed executed TS12 Panda-table geometry only.
- Every supplied original label-arrival gate passes before any projection or fork. `now`
  is a release tick, never an invented original observation time. Early calls
  raise; delayed data does not become available by renewing a budget.
- Project each source's actual executed history separately. Targets do not become
  prior commands. Reference labels remain qualified; both correction families
  remain separately provisional. No mixed qualified episode is manufactured.
- Eight simultaneous equal-row head-GN updates use the existing original-MSE
  kernel, damping/step bound and strict decrease rule. No new optimizer or fit.
- Two (A+B) or three (A+B+C) separate before/after residual groups retain predictions and
  axis/stage summaries. Their scores do not select updates or accept a candidate.

All row metadata includes raw source observation42, model features12, original
target/actual command, episode/context/role/stage/definition/provenance and original
clocks. Unknown reference/curated ticks remain unknown; B's existing2i/2i+1 label
clocks and curated query/arrival/availability remain distinct. Joint runtime packet
`now/now` is explicitly a derived batch, not a qualified source observation.

Admission errors raise before forks. Once a private scope exists, failures return
`complete=False`, `candidate=None`, original validated/sealed prefixes, observed
native returns/receipts, attempted work, progress and native exception work.
Actual source mutations are reported, never restored away. A returned prediction
is retained before later integrity checks. Corrupted prior evidence is retained
and refused before more work; no implicit retry. Each scope retains at most eight
scalar update records and one latest native state pin, plus bounded input and
diagnostic records. Only full success returns a detached native candidate.

## Structure and costs

```text
src/adapters/joint_readiness_data.py         # typed admission, release, projection, row lineage
src/adapters/joint_readiness_corrections.py  # private runtime composition, work and detached result
tests/test_joint_readiness_corrections.py    # behavior, isolation and paid-failure regressions
```

Dependencies point from adapters to existing core/app/adapter ports. No core/app
imports an outer adapter. No new dependency, CLI, environment setting, background
worker, persistence, promotion, simulator, final evaluation or probe work.
`prepare_joint_readiness_data` returns owned inspection data; its container is not
standalone training authority. The runtime always admits the original typed inputs.

For N rows, an accepted call performs8N GN row exposures and2N residual prediction
rows, four axis solves per update and no initializer/SGD. Preparation reports2N
raw observation projections/84N raw coordinates/24N projected values/192N logical
projected bytes/N history advances. Inspectable raw-source payload adds42N values/
336N logical bytes. Each retained runtime batch owns12 features+4targets per row;
eight batches use1024N numeric bytes. Native work, known costs and unknown solver
arithmetic remain separate. Python objects, validation, copies, snapshots and
solver workspace add cost; these are not total FLOPs or RSS estimates.

## Focused development commands

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_joint_readiness_corrections.py
.venv/Scripts/python.exe -m ruff check src/adapters/joint_readiness_data.py src/adapters/joint_readiness_corrections.py tests/test_joint_readiness_corrections.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/joint_readiness_data.py src/adapters/joint_readiness_corrections.py tests/test_joint_readiness_corrections.py
.venv/Scripts/python.exe -m mypy --platform linux --follow-imports silent src/adapters/joint_readiness_data.py src/adapters/joint_readiness_corrections.py tests/test_joint_readiness_corrections.py
```

Tests spend small synthetic model work; use the declared local fixture budget.
Implementation/fixture evidence is separate from the historical native result:
[both-mode software acceptance](../artifacts/runs/joint-readiness-two-source-20261021/RESULTS.md),
[original known A/B recovery](../artifacts/runs/returning-a-curated-corrections-20261021/RESULTS.md),
[negative old-data control](../artifacts/runs/matched-old-information-control-20261021/RESULTS.md).
The original successful state32/1736 includes a preceding eight-update A15+B50
stage to24/816. The explicit None mode supplies that preceding stage; both-stage
supported stream integration and exact numerical/action equivalence remain open.
Native equivalence through this new API requires its own prospective integration;
synthetic correctness does not establish a new physical task or circadian benefit.

Extension rule: add a separately admitted typed data family and its release/role/
provenance tests before widening the source contract. Reuse the runtime and native
work reporting; do not pass unqualified raw arrays as new authority.

## Full stream composition

`src.adapters.joint_readiness_stream.run_joint_readiness_stream` composes the same
base A/B/A runner with the two correction stages at its detached B checkpoint:

```python
from src.adapters.joint_readiness_stream import run_joint_readiness_stream

result = run_joint_readiness_stream(
    warm_binding, new_binding, corrections, curated,
    corrections_now=99, curated_now=199,
)
if result.report["complete"]:
    adapted = result.candidate
```

The two bindings are original qualified raw42 TRAIN A17 observed-grasp and B19
readiness demonstrations, respectively. Saved corrective B and curated A remain
separate provisional typed inputs. The caller's existing qualified input reader
retains responsibility for qualification; this function cannot grant it. Both
release gates and ordered explicit ticks are admitted before initialization.
Fixed BP/H64/seed17/common-MSE/executed TS12 and8/8/8 base phases are preserved.
The B checkpoint runs8joint A+B updates then8joint A+B+C updates. The benchmark
continues its original A return independently. Full paid base/branch reports are
retained under stream/joint_ab/joint_abc; only successful full composition returns
the final adapted candidate. No native rollout, file IO, promotion or deployment.

A15/B22 plus B50/C50 yields source16/296→24/816→32/1736 for the adapted branch;
the untouched base ends24/416. Source and later branch/report/input mutations
are refused. Storage retains two bounded8-update branch reports and fixed phase
pins; it does not grow with an unbounded stream. The existing observer's scalar
result excludes the large branch reports.

Preflight admits/projects both stages, then runtime repeats preparation so its
original-input guards remain authoritative. `preparation_work` reports the two
preflight calls and extra A/B source projections separately from each branch's
preparation_work. For the original recipe these comprise397 episode-row encodings
(180preflight+180runtime+37source),794raw projections/33348raw coordinates/
9528projected values/76224logical projected bytes/397history advances. Validation,
ownership copies, snapshots and Python containers cost extra. No empirical
normalizer is fitted; the base preserves its existing identity construction.

```text
src/adapters/joint_readiness_stream.py   # fixed composition at existing observer
tests/test_joint_readiness_stream.py     # actual integration and controlled failures
```

Focused commands replace the previous test path with
`tests/test_joint_readiness_stream.py` and scope Ruff/mypy to the new module/tests.
Use the prospective local protocol in
`artifacts/runs/joint-readiness-stream-20261021/PROTOCOL.md` for fixture and
separate native-free equivalence budgets. No additional dependency/configuration.

## Scoring on reserved guard resets

The existing frozen rollout API also accepts explicit `guard_seed=31` or37:

```python
from pathlib import Path
from scripts.maniskill_continuous_smoke import run_frozen_rollout
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe

scored = run_frozen_rollout(
    Path("new-guard-31.json"), learner=adapted, guard_seed=31,
    action_history_recipe=PickCubeTaskSpaceHistoryRecipe(mode="executed"),
)
assert scored.corrections is None
```

Use a fresh output path and a prospectively declared native budget. Guard mode
keeps role inner_guard, its dedicated context and episode identity. It requires
an exact native BP/PC-family TRAIN source and executed TS12 history. Source and
frozen copy remain unchanged; no fitting, teacher queries, corrections, replay,
promotion or final evaluation. Native nearest/actor/clock/visual/context-drift/
decoding/monitoring modes are outside this guard API. The normal TRAIN17/19 API
and CLI keep their existing defaults. guard_seed cannot mix with train_seed19.
Reserved probe23/24 and final29 cannot enter through the guard option.

Reports carry guard_seed/scoring_only only for this mode. Output guard traces
retain role inner_guard; the function cannot qualify or convert them to TRAIN.
The causal history binds original role/reset/context/episode identity throughout
each return and final close, and commits actual float32 commands only after a
validated native return. Failed executions retain paid work and no usable labels.

Both31/37 scores must be reported under the fixed protocol; they cannot select
a favorable run or tune this learner. This evaluates unseen resets of the same
task, not a new physical task or a population reliability claim. Constructor and
reset physics remain unmeasured; completed control physics are counted separately.
Protocol/results: artifacts/runs/corrected-policy-guard-20261021/.


### Executed scope (2026-10-21)

48 focused cases pass on Windows and pinned Linux, with final scoped Ruff/format/
mypy and independent acceptance. The original fake-reward fixture failure is
preserved. A native rendering-device failure and its one environment-only repair
are both charged. Repair timed out: guard31 completed with zero reward/no success
in50 controls; guard37 has no returned result. No further native attempt is
included. Use explicit LP_NUM_THREADS=1 and the pinned Mesa Vulkan ICD when
reproducing the accepted CPU producer environment under a future declared budget.
The two-reset question and reliability claims remain open. See
[results and exact receipts](../artifacts/runs/corrected-policy-guard-20261021/RESULTS.md).

## TRAIN-only feature support

Request the diagnostic explicitly after the existing release/preparation boundary:

```python
from src.adapters.joint_readiness_data import prepare_joint_readiness_data
from src.adapters.joint_readiness_support import summarize_joint_readiness_support

data = prepare_joint_readiness_data(reference, corrections, curated, now=199)
support = summarize_joint_readiness_support(data)
print(support["by_source_stage"]["curated_a"]["lift"])
```

Use the original already-qualified typed reference and released corrective inputs.
The diagnostic does not grant qualification or label-release authority. It leaves
preparation, training and their existing reports unchanged. No new configuration
or dependency is needed; explicit curated=None still reports only A/B sources.

Every source and each of its five recorded stages has row counts,12 marginal
coordinate ranges, constant-coordinate indices, distinct feature/previous-action
counts, official grasp-flag counts and seven consecutive-closing-command bins.
An absent stage has zero counts and coordinate_ranges=None. Recorded stages are
metadata, not inferred teacher states; closing history is not contact evidence.
Envelopes do not prove joint coverage, reachable states, model fit or control.

Original TRAIN identities/roles/spec/groups are checked. Features are recomputed
from raw42 and actual nativefloat32 commands, with a separate history reset per
source; targets and clocks are never read. Altered geometry/history refuses
instead of silently normalizing evidence. Returned summaries own their containers.

The diagnostic reports additional projection/history/aggregation work separately
from data.work. Source-value counts denote distinct raw coordinates, not repeated
validation passes. Logical numeric bytes exclude Python objects/sets/RSS. Keep
diagnostics budgeted even though they perform no learner/RNG/query/native work.

```text
src/adapters/joint_readiness_support.py  # explicit read-only prepared-TRAIN summary
tests/test_joint_feature_support.py     # ranges, causal history, ownership, refusals
```

Focused commands: pytest -q tests/test_joint_feature_support.py; scope Ruff,
ruff format --check and mypy --platform linux --follow-imports silent to those
two files. The fixed115-row description has a separate prospective protocol in
artifacts/runs/train-feature-support-20261021/PROTOCOL.md. It reads no guard,
probe or final data and selects no learner, rows, seeds or settings.


The diagnostic is executed and independently accepted:34 focused cases pass per
Windows/pinnedLinux and one frozen115-row description completed. Curated A has
40lift/0settle rows;115 total rows yield114distinct feature vectors. These are
marginal TRAIN facts; no control/generalization benefit follows.
[Observed summaries, commands and accounting](../artifacts/runs/train-feature-support-20261021/RESULTS.md).

## Optional recorded-stage sampling control

Both existing APIs accept `sampling="stage_balanced"`; omit it for the original
equal-row recipe. For example:

```python
result = run_joint_readiness_stream(
    warm_binding, new_binding, corrections, curated,
    corrections_now=99, curated_now=199, sampling="stage_balanced",
)
```

The core planner requires all five recorded stages,5..128 rows and a row count
divisible by5. Each of the eight updates has equal stage quotas, in fixed
align/descend/grasp/lift/settle order; original indices cycle in order within
each stage across updates. Every source row must appear somewhere in the full
schedule. The original65/115-row stages therefore use quotas13/23. Missing stages
are refused, not invented. Both stream plans pass before model initialization.

The native per-batch MSE remains unchanged, while empirical weighting and source/
context exposure change. Original targets, raw observations, clocks and executed
history stay attached to their original rows; sampled neighbors do not create a
new causal episode. Residuals always evaluate all original rows. This is a
conventional TRAIN sampling control, not a circadian mechanism or compute match.

Selected reports add `sampling` with the original index schedule and planned
per-row/source counts. `row_training_exposures` and `group_training_exposures`
describe the validated native prefix; `attempted_row_training_exposures` describes
attempted payloads. `unvalidated_sampling_work_unknown` keeps failed/unvalidated
work unknown. Raw native work and original receipts remain charged separately.
Replacing a schedule refuses further work; failure capture uses its original
immutable mapping. The default keeps its original arrays/report schema and omits
the sampling keyword at delegation boundaries.

```text
src/core/joint_stage_sampling.py       # bounded deterministic original-row indices
tests/test_joint_stage_sampling.py    # plan, native receipts, failures, forwarding
```

Run focused integration regression with:

```powershell
python -m pytest -q tests/test_joint_readiness_corrections.py tests/test_joint_stage_sampling.py
```

Scope Ruff/format/mypy to the new core/tests and both existing correction/stream
adapters. Keep fixtures, training comparisons and any later native scoring under
separate prospective budgets. Protocol/results are in
artifacts/runs/stage-balanced-corrections-20261021/. The selected option is
experimental; the unbalanced default is unchanged.


The option has executed and independently accepted evidence:90 focused cases
pass per Windows/pinnedLinux and final scoped lint/format/types pass. The fixed
TRAIN comparison worsens all three final losses. The separately frozen native
case preserves A17 success17controls and loses B19 after50; saved unbalanced
controls succeeded17/18. Keep the original default. Additional empirical exposure,
planning/gather/diagnostic costs and all failures remain recorded; no reliability,
new-task or CPC claim follows.
[TRAIN comparison](../artifacts/runs/stage-balanced-corrections-20261021/RESULTS.md) ·
[Native comparison](../artifacts/runs/stage-balanced-known-context-20261021/RESULTS.md).
