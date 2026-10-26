# Paired own-policy recurrent quota control

The existing deterministic local MiniGrid-like feature-game CLI (not official
MiniGrid physics) pairs a frozen initialized recurrent policy with native
reward learning. Both use the same architecture and exact initial native state,
including optimizer/gradient state, but collect their own trajectories. Each pays
96 TRAIN interactions under eight original 12-step quotas. There is no replay,
probe, promotion or final feedback to training.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-paired-interaction-quota --environment-seed 0 --heldout-environment-seed 23 --describe
```

`--describe` creates no model. Without it the command executes one paired case
under the default 90-second/768-MiB process budget; numerical studies also need
a prospective local protocol. Omit both seed flags to use 0/23. A supplied pair
must be distinct exact catalogue indices in 0..23; these enumerate geometry and
are not independent random initialization seeds. Existing standalone, warm-start
and probe scenarios keep their semantics. No new dependency or configuration.

## Accepted bounded comparison

One explicitly frozen resource amendment completed original pairs0/23 and6/17,
with180s per pair and400s outer cap (actual192.381s). The prior90s attempt remains
incomplete/SPENT; this is the same cohort retried under a prospective engineering
amendment, not independent replication. All160live evaluations failed to succeed.
The first pair avoided2negative terminals at10moresteps while teacher-action
accuracy worsened; second pair all40signed cells unchanged. All80cells/raw costs,
negative results and original charges are retained. Independent review maps the
literal original R4.3 criteria to complete observed behavior; it does not establish
useful learning, matched compute, generalization or circadian benefit.
[Full results and costs](../artifacts/runs/paired-wall-allowance-20261020/RESULTS.md)
· [Criterion map](../artifacts/runs/paired-wall-allowance-20261020/REVIEW.md).

The existing inspector rejects this paired schema. Next extend that existing
projection/rendering path for both recorded arms, native/physical stages, signed
cells, costs, timing and source pointers; no additional numerical cohort.

## Results and interpretation

Both arms finish TRAIN before this pair collects any final packet. Five physical
boundaries retain all eight context/cue readouts, even when optimizer indices
differ or stay zero. The report includes original trajectories, receipts, sampled
probabilities, quotas, learned parameters, memory/work, and all 40 signed cells:

- CE reduction is frozen minus trained; higher is better.
- Accuracy, success and return gains are trained minus frozen; higher is better.
- Extra physical steps are trained minus frozen; lower is cheaper.

CE uses raw probabilities, while live decisions condition on public legal action
support. Singleton support can force a correct action. Neither task success nor
an imitation score alone proves learned remapping. Compare costs separately:
training updates, row exposure, episode resets and computation can differ.

The native implementation initializes during forks before restoring. Per pair,
two source initializers plus runtime, collection and evaluation forks total
38..54 initializers, each drawing 168 normal values. The descriptor separates
source initializers from total native work. Do not equate restored weights with
zero initialization cost. Tensor payloads and process RSS are separate measures.

## Files and boundaries

```text
src/adapters/key_door_quota_comparison.py   paired existing-arm composition and comparisons
src/adapters/key_door_reward_pilot.py       private initialization/deferral seam
scripts/run_coupled_attribution.py          explicit CLI scenario
tests/test_paired_quota_control.py          metadata/stub behavior and failure tests
```

The comparison adapter creates no new learner, objective or environment. Original
native guards validate the preceding success before more work. Bounded paid row
seals protect earlier costs/results across peer and resource observers; replaced
evidence is refused. Failed work is retained without renewing an original budget.
The historical frozen-control supervisor qualifier remains failed and spent.

## Checks and extension

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_paired_quota_control.py
.venv/Scripts/python.exe -m ruff check src/adapters/key_door_quota_comparison.py src/adapters/key_door_reward_pilot.py scripts/run_coupled_attribution.py tests/test_paired_quota_control.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/key_door_quota_comparison.py src/adapters/key_door_reward_pilot.py scripts/run_coupled_attribution.py tests/test_paired_quota_control.py
.venv/Scripts/python.exe -m mypy --platform linux --python-version 3.12 --follow-imports=silent src/adapters/key_door_quota_comparison.py src/adapters/key_door_reward_pilot.py scripts/run_coupled_attribution.py tests/test_paired_quota_control.py
```

36 focused cases pass per platform with six selected compatibility expansions;
the later initializer-descriptor check also passes per platform. Type analysis
above runs on Windows targeting Linux; it is not a Linux mypy execution. Ordinary
metadata tests stub numerical work. Mathematical/native behavior reuses existing
accepted evidence, while a real comparison needs its own recorded invocation.

Extend declared scenarios through the existing quota/learner contracts. Freeze
instance order, metrics and costs before a comparative run; never select them
using final results. [Protocol and checks](../artifacts/runs/paired-quota-control-20261020/PROTOCOL.md)
and [independent review](../artifacts/runs/paired-quota-control-20261020/REVIEW.md).

## Actual study remains incomplete

The one-use fixed0/23 then6/17 study completed both first-pair TRAIN allowances:
192collection steps total,9trained SGD updates/96rows, both full initial native
states equal and source/actor unchanged. All80rawscore calls completed. Frozen
40live episodes finished; trained31finished, onefailed before work and eightwere
unrun when the common90-second body deadline expired. Pair6/17 was never started.
The complete paired comparison remains unavailable; independent review withholds
actual qualification. Native process and recorder are terminal, allowance spent.

Known costs include1084predictioncalls/1740forward unrolled rows,1068completed game
steps incl64finalteacher steps, and40native initializations/6720normaldraws. Full
partial-work flags and negative outcomes remain; incomplete exposure cannot support
an advantage comparison. [Original records, exact commands and costs](../artifacts/runs/paired-quota-control-20261020/RESULTS.md).

A subsequent TRAIN-only scalar microprofile found only~0.77ms/call savings from an
existing report guard on that payload. It does not establish the whole-run cause,
and production guard comparisons were not changed. The later timing increment
adds per-arm spans and guard counters described below; it does not retrospectively
measure that failed run or qualify it. Do not retry the spent study,
reduce readout coverage or relax state/receipt checks to obtain a passing result.

## Phase and guard timing

Paired reports now include `timing` with definition
`paired_quota_inclusive_wall_v1`, using `time.perf_counter` independently of the
original monotonic budget clock:

```text
timing.arms.frozen.train / evaluation
timing.arms.trained.train / evaluation
timing.guards.pair / preceding / pending
```

Each of these seven fixed records has `attempts`, `returns`, `raised`,
`inclusive_seconds` and `clock_error`. Counts describe callback invocation:
a callback that returns a failed arm still increments `returns`; the arm's
status and original receipts remain authoritative. An unreached phase has zero
counts. A refused initial clock also leaves zero counts but records `clock_error`.

Times include nested work. Guard categories overlap each other and the phase
spans: **do not add them as disjoint costs**. Phase spans cover the wrapped arm or
evaluation callback, not every outer bookkeeping step. Clock reads and accounting
consume the existing wall budget; measurement is not free and may change the
instant when an unchanged deadline stops a run. Timings do not grant budget,
select data, or admit an update, probe or final result.

Only finite nondecreasing clock intervals enter the accumulated seconds. Invalid
or failed clock reads mark `clock_error`; the invalid interval is unmeasured,
not zero-time work. Ordinary measurement exceptions preserve an already raised
callback error. After a returned callback, an invalid end clock stops further
normal work while retaining its paid results. Timing lives outside the sealed
rows, learner/runtime state and receipts, so its counters cannot mutate the
protected evidence. Earlier saved reports have no retrofitted timing fields.

54 focused cases (24 timing + 30 prior paired regressions) pass on Windows/Linux;
scoped lint/format and Linux-target type analysis pass with independent acceptance.
The metadata checks create no actual models. A separate one-use initialized
component measurement then ran: 4 native initializers, no SGD, predictions,
environment or final/probe access. It completed with independently accepted
saved evidence. Mean snapshot 0.366370ms, same-reference comparison 0.036440ms,
fork 1.148100ms; cold initialization 1.986378s includes optimizer startup.
These do not explain full-run cost or represent post-SGD guard state. The original
study remains incomplete, with all failures and costs retained.
[Timing results, exact commands and limitations](../artifacts/runs/paired-quota-timing-20261020/RESULTS.md).

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_paired_quota_timing.py tests/test_paired_quota_control.py
.venv/Scripts/python.exe -m ruff check src/adapters/key_door_quota_comparison.py tests/test_paired_quota_timing.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/key_door_quota_comparison.py tests/test_paired_quota_timing.py
.venv/Scripts/python.exe -m mypy --platform linux --python-version 3.12 --follow-imports=silent src/adapters/key_door_quota_comparison.py tests/test_paired_quota_timing.py
```

## Prospective wall allowance

For this paired scenario only, `--paired-wall-seconds` accepts finite positive
seconds up to180; the omitted default remains90. API callers can supply the same
`wall_seconds` keyword to `describe_protocol` or `run_pilot`. Exact builtin ints
and floats are accepted; bools, numeric-like objects and invalid values fail
before runtime/model/clock/data work. CLI misuse on other scenarios is refused.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-paired-interaction-quota --paired-wall-seconds 180 --describe
```

The cap is captured once before both arms. They and their deferred readouts
share one original deadline; a later phase or mutable report cannot grant more
time. Final completion uses the same cap, and paid outcomes remain visible when
elapsed time makes the whole report incomplete. Existing native birth clocks,
receipt charges, physical quotas and evaluation isolation remain unchanged.
The descriptor reports the chosen cap. Changing this engineering cap requires a
prospective budgeted protocol; it cannot resume or renew a failed runtime.

105 focused metadata/stub cases (51 allowance +24 timing +30 paired controls)
pass on Windows and pinned Linux. Scoped lint/format/types and independent review
pass; real CLI describe differs only in chosen resource cap. No numerical study
is established by those tests. The original90s comparative attempt remains
incomplete/SPENT. [Current engineering and any separate amended result](../artifacts/runs/paired-wall-allowance-20261020/RESULTS.md).

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_paired_wall_allowance.py tests/test_paired_quota_timing.py tests/test_paired_quota_control.py
.venv/Scripts/python.exe -m ruff check src/adapters/key_door_quota_comparison.py scripts/run_coupled_attribution.py tests/test_paired_wall_allowance.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/key_door_quota_comparison.py scripts/run_coupled_attribution.py tests/test_paired_wall_allowance.py
.venv/Scripts/python.exe -m mypy --platform linux --python-version 3.12 --follow-imports=silent src/adapters/key_door_quota_comparison.py scripts/run_coupled_attribution.py tests/test_paired_wall_allowance.py
```

Extend this route through the existing runner. Freeze all resource caps and
scientific settings before any comparison; preserve original default behavior,
negative results and the full required context/cue readouts.

## Saved paired inspection

The existing standalone inspector accepts both complete and failed paired reports.
See [viewing, source links and scoped validation](paired-inspector.md). Physical
and native stages, signed cells, costs, nulls and inactive controls stay visible.
The original negative studies are reused unchanged; no new model work occurs.
