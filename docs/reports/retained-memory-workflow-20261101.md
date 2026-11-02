# Explicit retained-memory public workflow — #418 / R4.8-MEM.RETAIN

The existing CPU/native memory command now exposes bounded FIFO and deterministic
seeded-reservoir retention. Implementation, 38 final literal checks, two frozen
public-command qualifications and finite independent source/PRE/POST review are
accepted in scope. This is useful accessible functionality, not repository-wide
release acceptance or evidence of circadian advantage. Existing negative results,
including the original #417 failed capture attempt, remain unchanged.

Physical checkout `D:\Dev\Repositories\Circadian`, master HEAD
`d9119e80926a139a2f38460564d7eab1b6d2a2b0`, inherited dirty/untracked work preserved.
Goal/AGENTS/continuation/concise handoff and actual source reconciled. GitHub owns
live status; requested plan/log additions are archival evidence. Completed
repeated-update/isolation/matched-baseline/R2.1 diagnostic work was reused, not
reconstructed. No old raw-analysis or numerical baseline quota was renewed.

## Delivered behavior

`run_demo`, `run_native_demo` and `scripts.run_taskspace_memory_demo` accept an
explicit existing `ReplayRetentionPolicy` and capacity1..256. CLI requires an
explicit reservoir seed and refuses rule-less capacity/seed, FIFO seeds, invalid
capacity and retained/private/development conflicts before input/model work.
Default call shapes, strict exact replay, resource loader and old private-probe
route remain. No new runner, learner math, dependency or public data authority.

CPU `offline_retained_training_readout` compares returned predictions with the
four original recorded TRAIN command batches through the existing fidelity
helper. Mismatches are valid scored results only in this explicit mode. Retained
indices are incomparable to original full-bank indices. Saved success/rewards
remain explicitly original-recording observations; `native_outcomes=null`.
Native `retained_native_training_rollout` executes the existing four known TRAIN
scenes. Full retained-state/owner, original exposure/clock/receipt/permanent-charge,
preceding paid-return/capture, source/history and bounded storage guards remain.

The existing endpoint validator now takes an explicit stored-row count separately
from cumulative exposure. Default validation is unchanged. Native saved/returned
comparison canonicalizes JSON containers and preserves primitive types: tuples
can serialize as arrays, but changed `true` versus `1` refuses. Replaced/corrupted
prior proof stops further work and preserves trusted paid prefixes. No discarded
task/result is silently selected away or promoted.

Changed product files:

```text
src/adapters/taskspace_memory_demo.py       CPU options, retained pins/readouts
scripts/run_taskspace_memory_demo.py        explicit CLI admission/capture
scripts/taskspace_memory_native.py          same native stream/capture composition
scripts/push_cube_memory_control.py         exposure versus stored-row validation
tests/test_retained_memory_workflow.py      new literal boundary cohort
```

## Executed focused checks and failures

Frozen `SOFTWARE-PROTOCOL.md`, before-source copies and all receipts are under
`artifacts/runs/retained-memory-workflow-20261101/`. Exact invocations:

```powershell
& artifacts/runs/retained-memory-workflow-20261101/run-checks.ps1 -Slot first
& artifacts/runs/retained-memory-workflow-20261101/run-checks.ps1 -Slot second
& artifacts/runs/retained-memory-workflow-20261101/run-checks.ps1 -Slot third
& artifacts/runs/retained-memory-workflow-20261101/run-checks.ps1 -Slot fourth
```

Each wrapper runs `.venv/Scripts/python.exe -B -m pytest -q --tb=short
tests/test_retained_memory_workflow.py --basetemp D:/Dev/_Temp/circadian418-SLOT`.
Observed first31PASS; expanded second36PASS/1fixture-hookFAIL; repaired third37PASS;
final38PASS. The failed hook captured the identity projection instead of the full
report; its failure/cost remains SPENT. Fourth invocation was prospectively added
after self-review discovered the actual primitive-type capture gap. Original
90s aggregate and every numerical/copy/RSS ceiling stayed unchanged; no fifth.
143 executions/142passes/1failure, 38 distinct final cases. Host times
1.9914574/2.3201292/2.0634349/2.0894181s total8.4644396s, peak<=83,738,624B.

Cumulative literal work:55initializers385updates/rows825forks64predictions/queryrows,
163distance candidates1956feature visits2222public snapshots548352selected numeric
snapshot bytes51fidelity rows58fake scene attempts54fake controls. No real data,
SDK, physics, neural fitting, gradients, teacher or model RNG in this cohort.

Final scoped Ruff check/format PASS on the four product files/new tests; wrapper
Ruff also PASS. Win32 and Linux mypy PASS on those five files:

```powershell
.venv/Scripts/python.exe -B -m ruff check src/adapters/taskspace_memory_demo.py scripts/taskspace_memory_native.py scripts/run_taskspace_memory_demo.py scripts/push_cube_memory_control.py tests/test_retained_memory_workflow.py artifacts/runs/retained-memory-workflow-20261101/run-public.py
.venv/Scripts/python.exe -B -m ruff format --check src/adapters/taskspace_memory_demo.py scripts/taskspace_memory_native.py scripts/run_taskspace_memory_demo.py scripts/push_cube_memory_control.py tests/test_retained_memory_workflow.py artifacts/runs/retained-memory-workflow-20261101/run-public.py
.venv/Scripts/python.exe -B -m mypy --follow-imports=silent --cache-dir D:/Dev/_Temp/circadian418-mypy --platform win32 src/adapters/taskspace_memory_demo.py scripts/taskspace_memory_native.py scripts/run_taskspace_memory_demo.py scripts/push_cube_memory_control.py tests/test_retained_memory_workflow.py
.venv/Scripts/python.exe -B -m mypy --follow-imports=silent --cache-dir D:/Dev/_Temp/circadian418-mypy-linux --platform linux src/adapters/taskspace_memory_demo.py scripts/taskspace_memory_native.py scripts/run_taskspace_memory_demo.py scripts/push_cube_memory_control.py tests/test_retained_memory_workflow.py
```

Earlier mypy16annotation/test-typing errors were repaired, not suppressed. Wrapper
lint initially found an unused import; removed before frozen source pins/PRE.
Orientation tooling failures: guessed missing source path, invalid PowerShell
wildcard search and unmatched documentation patch context. Simplified to actual
paths/ordinary edits; none launched model work. No unresolved check failure.

## Frozen public qualification and observed results

After PRE, exactly one CPU command then one native command:

```powershell
& artifacts/runs/retained-memory-workflow-20261101/run-public.ps1 -Case cpu
& artifacts/runs/retained-memory-workflow-20261101/run-public.ps1 -Case native
```

`run-public.py` accounts external calls and invokes the actual first-party module
with `runpy.run_module("scripts.run_taskspace_memory_demo", run_name="__main__")`.
Arguments are `--retention recent_fifo --max-prototypes 128 --output STAGE/CASE.json`,
plus `--native` for the second case. CPU uses existing Windows `.venv`; native uses
the prepared WSL Ubuntu Python3.12/NumPy1.26.4 ManiSkill/SAPIEN CPU environment,
lavapipe and existing installed tool locations. Exact executable/environment/time/
timeout arguments are in `run-public.ps1`. TEMP/TMP/TMPDIR on physical D:, numerical
threads1, no bytecode. No installs, downloads, GPU or global setting changes.

`PUBLIC-PROTOCOL.md` fixed FIFO128, original bundle166439B SHA
`f8086cca0919d3bee24ccd166c10c0153b103c9b40c34594c117d8c76bbaeb6b`, original seeds,
metrics,50-control horizons and failure policy before input/model/SDK work. Native
run addresses the changed public interface/validator/capture integration; it is
not a renewed eight-arm comparison, baseline rerun or scientific replication.

Both CLI exits0/complete/budget_ok,7appends220exposures128stored90discarded2duplicates.
Source/stable actor remain warm1publication/15rows; six delayed runtime updates
consume205rows, no replay, permanent charges6/nexttick18. All30source pins match
before/after; original input identity/digest and canonical saved/returned binding
pass. No private/final/eval/promotion access or neural/optimizer/model RNG work.

| Scene | CPU recorded query mismatches | Fresh native outcome | Controls |
|---|---:|---|---:|
| Pick19 | 0/22 | success | 22 |
| Pick17 | 13/15 | failure | 50 |
| Pick20 | 13/15 | failure | 50 |
| Push17 | 10/18 | failure | 50 |

CPU:4prediction returns70queryrows8960candidates107520coordinates70fidelity rows,
1initializer7appends220rows15forks40snapshots365selected validators2,865,408selected
copy bytes378pair-hash calls22,811rows3,467,272selected hash bytes. Body0.5573251000s,
host0.6802397s, actual peak56,410,112B. Raw28118B SHA
`a623ca04de43a2caef1417ffc8350b6fe52f4d67aa777510ab47bfafc11bf19a`.

Native:172returned predictions/controls860known control physics steps, four
constructed/returned/closed scenes, no unknown failed work. 220exposures1initializer/
7appends18forks22,016candidates264,192coordinates2767snapshots8271selected validators/
132,397,312selected copy bytes8284pair-hash calls1,034,779rows157,286,408hash bytes,
1selected SDK RNG call, model RNG0. Body9.860220899s/GNU10.21s524220KiB/host13.0755075s,
actual peak536,526,848B. Raw350916B SHA
`a429f6a133b292f230c42b24133e24a9bb37b76defda2ff15d4d5e4eaa1f3ed7`.
GLVND installation warning retained; no environment repair attempted.

Each case reads original public input once166439B plus declared before/after byte
authentication; source-pin reads1,417,190B per case. All original scalar/copy/hash/
time/RSS/output caps held. Producer capture validation is distinct from one lead
and one peer raw analysis per new case; all analysis markers SPENT, use scalar
cache afterwards. No old #417 raw reopened. Numeric16,384B+encoded IDs8,192B per
bank=24,576selected payload;4,096digest payload describes same IDs, not additional
bank storage. Python/validation/SDK/RSS and other computation remain additional.

## Acceptance, limitations and next outcome

Lead reviewed combined source/procedure/results. One finite independent exception
used SOURCE/PRE/POST read batches0.1885/0.2237/0.1557367s total0.5679367s. Peer read
each new main report once, checked hashes/current30pins/modes/exposure/storage/
outcomes/costs/budgets/final38log and accepted the scoped public workflow. Caveat:
nested stream checkpoints/work were not separately extracted by peer; those rely
on unchanged qualified guards plus sourcepub1. Lead extracted exact six charges/
205rows/clock18 into `LEAD-SCALARS.json`. Self-review is not independent review.
No project imports/tests/model/SDK/data/oldraw/replication/writes/delegation by peer;
unused fourth batch CLOSED and exception ENDED. PRE/POST receipts retain limits.

Skipped: unchanged old model/native/BP-PC-CPC/probe cohorts, broad full suite,
clean clone/current wheel install, GPU, remote CI and release checks. No commit,
push, reset, deletion or unrelated dirty edit. #418 scoped criteria completed;
#18/#21/#23/R2.3/R4/R5/reliable adaptation/recovery/science/release remain open and
full goal ACTIVE. No overall memory-efficiency, matched-compute, generalization,
reliability, CPC advantage or recovery claim from these known-TRAIN results.

Critical path reassessed: do not tune capacity/seed, repeat comparisons or add
another checkpoint/capture/audit increment. Existing R2.1 diagnostic math and
unbounded fresh-task probes already work in scope. Actual source shows retained
private adaptation is missing: exact old-type admissions and use of cumulative
training_examples as retrieval/consistency bank length would misbind retained
state and costs. Next [#419](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/419)
answers bounded-memory fresh-task acquisition/old-task interference through the
existing isolated probe, with original consent and final isolation preserved.
Recovery remains separately unproved until an actual old-loss/return phase.

**One next implementation action:** inspect existing nearest probe state/owner/
score/consistency dispatch, define the explicit retained-source private adapter
contract and early refusals, then freeze a finite new literal cohort before model
work. No actual-data/native grant or automatic retry from this handoff.

Canonical scoped completion: https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/418#issuecomment-6566097719 . #418CLOSED,5scopedcriteria evidenced/readback0unchecked; tracker#18receipt6566101097. #419PLANNED, broaderparents/fullgoalOPEN-ACTIVE. Finalsource AST9unchanged private/default functions and current30pins PASS; scoped whitespace/format PASS (LF/CRLFwarnings retained). No further model/raw read after accepted qualification.
