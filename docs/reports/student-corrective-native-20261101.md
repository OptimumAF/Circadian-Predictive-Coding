# Fixed student-state corrective native case — 2026-11-01

[#415 / R2.1-CTRL.COVER.E](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/415)
completed one prospectively frozen, budgeted measurement. The result is negative:
50 new student-state labels and 16 private full-network updates greatly improve
their own fitting error, but neither fresh reset succeeds and all three previously
successful old resets fail in the private learner. The source/deployed learner
is unchanged. This establishes no acquisition, retention, recovery, generalization
or circadian benefit, and does not establish coverage as the cause of failure.

## Implemented and executed scope

The new stage composes existing frozen collection, explicit probe_train23 consent,
sealed corrective pairs and private probe runtime with the existing full-network
Gauss–Newton option. It changes no production learner or runner. Original student
commands remain in action history; goal-consistent provisional teacher targets
are separate and released only after collection. All rows of the first complete
trace are used; there is no filtering, forcing, retry or selection of a better seed.

The original qualified source remains 80 updates/17,600 row exposures. The private
child reaches 96/18,400 and is discarded. There is no original TRAIN replay,
source refit, model initialization/RNG, final access, reset24 learning/tuning,
promotion, exported private weights or budget/clock renewal. New label queries,
data and computation make this explicitly unmatched to the original18-row case.

Before actual work, [the protocol](../../artifacts/runs/student-corrective-native-20261101/PROTOCOL.md)
froze original23 collection, original24 readout and old17/19/20 roles/seeds;
16 private updates; uniform XYZ half-MSE plus original binary-gripper objective;
damping .001, rcond1e-12, norm cap1 and one strict-descent trial/update. All16
momentum resets are paid. Existing #414 correction checks and #412 math evidence
are reused under unchanged sources and assumptions.

The new collection itself observes source23 failure50. Accepted source24 failure50
and source old17/19/20 successes22/20/15 are reused only after exact checkpoint,
source, inference and reset binding. Collection23 is necessary new queried data;
there is no duplicate pure-before rollout. The original23 teacher export is not
read or converted by this case. Original24 remains readout-only, final unavailable.

## Outcomes

| Endpoint | Accepted/observed before | Private after | After controls |
| --- | --- | --- | --- |
| probe_train23 | collection failure | failure | 50 |
| probe_test24 | reused failure | failure | 50 |
| TRAIN17 | reused success22 | failure | 50 |
| TRAIN19 | reused success20 | failure | 50 |
| TRAIN20 | reused success15 | failure | 50 |

All six new scenes are complete and closed. Each has sparse return0, no success,
50 returned controls/250 known control-physics steps, no possible failed command
and no native optimizer update. Frozen state, original owner and executed history
proofs pass. Old success is lost only in the private copy; the deployed policy
does not regress. Return/relearning after loss is not measured.

| Score, unchanged definition | Before | After |
| --- | --- | --- |
| New50 corrective-label half-MSE | .24905687874391128 | .000008195138016434685 |
| Original24 half-MSE | .06436728616819498 | .0938737873669976 |
| Old17 half-MSE | .0015257813261950286 | .0803757239995486 |
| Old19 half-MSE | .002662961974564868 | .036237129099003215 |
| Old20 half-MSE | .0009462260756697378 | .07140601648305887 |

Mixed fitting objective .24905710607564732→.000008203093852997706. The new50
label score is a different label set from the original18 teacher fit. These
imitation scores do not replace primary native success/return outcomes. All
#399/#408/#411 partial failures and #413 completed negatives remain preserved.

## Commands, qualification and costs

Run from physical `D:\Dev\Repositories\Circadian`, with process-local
TEMP/TMP/TMPDIR=`D:\Dev\_Temp`, no bytecode and numerical threads1. Commands were
executed once unless explicitly identified as a repaired static diagnostic:

```powershell
.venv\Scripts\python.exe -I -B artifacts/runs/student-corrective-native-20261101/check-composition.py
.venv\Scripts\python.exe -I -B artifacts/runs/student-corrective-native-20261101/qualify.py
.venv\Scripts\python.exe -m ruff check artifacts/runs/student-corrective-native-20261101/study.py artifacts/runs/student-corrective-native-20261101/check-composition.py artifacts/runs/student-corrective-native-20261101/qualify.py
.venv\Scripts\python.exe -m ruff format --check artifacts/runs/student-corrective-native-20261101/study.py artifacts/runs/student-corrective-native-20261101/check-composition.py artifacts/runs/student-corrective-native-20261101/qualify.py
.venv\Scripts\python.exe -m mypy --follow-imports=silent artifacts/runs/student-corrective-native-20261101/study.py
.\artifacts\runs\student-corrective-native-20261101\run-case.ps1
```

The NEW tiny D12/H2 corrective/full-network combination passes: one update,
two rows, one ≤64-square solve, two gradients/four rows, five forwards/ten rows,
four literal owners, 67 snapshots/40,736 selected bytes; loss
.44902516648694446→.13244988237029892. It takes .340334000s/47,448,064B.
No constructor/RNG/ordinary SGD/data/teacher/SDK/native work occurs. This is
one new combination fixture, not renewal of an old numerical cohort.

Source-only qualifier passes11 checks in .016617200s/23,330,816B, zero project
imports/inputs/models/native. Seven shared helpers and Meter stop/ACK semantics
are AST-identical to #413; explicit new consent, dose, readout role, trace bound
and closure checks pass. Scoped Ruff/format3 and mypy1 plus PowerShell host syntax
pass. Unused import/three annotation errors were corrected before admission;
do not claim the other two helpers were type-checked. Final source pins cover30
direct inputs. The prospective pin count28→30 included all six newly used
teacher/correction sources; it changed no scientific or resource allowance.

Real command exited0, host PID11016, unified session32205. Producer26.590627488s,
host30.2925014s, GNU27.02s/user21.32/sys1.05, sampled peak543,907,840B and GNU
peak529,808KiB. Workstop110s/1792MiB, GNU125s TERM+5s KILL/130s2GiB acceptance,
host150s and stage8MiB limits held; RSS is sampled/observed rather than OS-enforced.
Inner16-update budget wall/RSS fields remain explicitly unset as frozen; outer
gates are separate. Original failed budgets elsewhere are unchanged.

Paid real work:

- 50 geometry queries and50 teacher attempts/returns; 50 labels released at tick99.
  Provenance `provisional_probe_train23_student_state_goal_consistent_v1`;
  source80/17,600, actual raw actions retained/targets separate.
- 16 updates/800 rows,16 proposals/solves,17,472 aggregate solver rows;
  32 gradients/1,600 rows,358 forwards/2,996 rows,310 predictions/596 rows.
- Ten residual readouts/296 rows: eight original24/old3 before/after plus two
  new50-label readouts. Both fitting endpoints complete; no replay exposure.
- Eight input reads/ACKs,1,032,304 bytes; nine forks/two readout restores;
  16,971 public snapshots,134,264 validator calls,2,642,377,920 selected copy bytes
  under the separately frozen4GiB logical cap. This is traffic, not peak RSS.
- Six creations/closures,300 controls/1,500 known control-physics steps.
  Selected kernel3,820,390,400 dense MAC/181,226,496 numeric bytes. Solver internals,
  workspace, other copies and SDK reset/initialization physics remain additional
  or unknown. No claim of matched total computation.

`complete`, `budget_ok`, `source_unchanged` are true; error/permanent_stop/hoststop
are null. All four30-pin checks (before inputs, collection, probe and after),
final checkpoint/bundle bytes, owner/data/paid-prefix and six closure proofs pass.
Each native trace is bounded1MiB; collection bound is checked before batch/fit.
Original consent, permanent charges, source clocks and retired evidence persist.

Raw result698,840B SHA256
`24a0e8a5d876b50b07ce4a89bdeabed7cef7d56ff249c66a7264c85b976534fb`.
Lead's one raw-analysis read is SPENT: use existing `LEAD-SCALARS.json`; no raw
reread or numerical replication. All frozen protocols, pins, markers, actual and
native traces, host/time, qualifier and compact scalar receipts remain under
`artifacts/runs/student-corrective-native-20261101/`. No private weights exported.

## Review and limits

Lead self-reviewed source, procedure and all executed results. Finite independent
source/procedure review used two batches (~.318s); the collection size gate was
fixed before PRE GO. The final #414 forced error-capture epilogue was inspected
for this admission, without retroactively changing #414's original review scope.

Reviewer POST batch3 performed its permitted one new raw-byte read/current-pins/
compact-host/time check, but the tool output exceeded available context and was
unavailable to the reviewer. Independent POST attestation is **incomplete**.
All three batches are spent; exception ENDED. There is no reread, replication,
new allowance or blanket independently accepted final-result claim. Lead-verified
execution and independently reviewed source/procedure are distinct evidence.

The real procedure did not fail. Failed optional source guesses/verbose displays
are tooling observations; initial static diagnostics were repaired prospectively.
The existing GLVND warning remains in stderr. Old unchanged math/runtime/native
cohorts, Linux fixture repeat, broad full suite, clean clone/current package,
GPU/remote CI and release checks were skipped. No install/download, commit/push,
reset/deletion/global settings or unrelated dirty edit; inherited masterd9119e8
tree remains. Scoped checks do not establish repository-wide readiness.

## Critical path and next action

#414 criterion5 is satisfied by this frozen/reviewed protocol before actual
inputs; #414 is closed. #415's scoped investigation can close with the completed
negative measurement and explicit POST limitation. Parents18/21/23/R2.1/R5,
reliable acquisition/retention/return, scientific/CPC and release remain OPEN;
full repository goal ACTIVE. No new correction round, optimizer dose or seed
selection follows automatically.

R2.1 diagnostics and isolated four-family probes already exist. Simpler memory
already acquires original23 (#405), retains known old tasks (#403) and has an
ordinary public workflow (#406). Existing nearest PromotableActor/native binding
and rejected guard protocol also exist; rebuilding them or publishing a rejected
or private probe candidate would not advance the goal. Reuse these scoped proofs.

The concrete missing behavior is bounded continued memory learning: current
nearest refuses additions above256, source220/private238 leaves36/18 rows, so
another50-row arrival cannot fit. Existing `ReplayRetentionPolicy` already has
recent_fifo/seeded_reservoir rules, but nearest has no eviction and its
`training_examples` equals stored rows. #416 / R2.3-MEM.CAP therefore addresses
storage-pressure retention through the existing learner/stream, with cumulative
paid exposures separate from retained rows and bounded metadata. This is a
different learner outcome, not another capture/probe-dose experiment.

**ONE next implementation action:** inspect snapshot/stream assumptions, write
the minimal explicit opt-in retention state/counter contract, then implement
pure row selection with one literal over-capacity acceptance/refusal test.
Preserve append-only defaults, role/release authority, source/deployed state,
all charges and negative evidence. Freeze finite checks before execution; no
actual data/native/teacher/probe allowance follows from planned #416.

## Recorded closeout

#415 is CLOSED as a completed negative investigation. The preceding conditional
closeout wording describes the report before the issue write. Canonical receipt:
https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/415#issuecomment-6564874784
Three new document link/whitespace checks, three requested archival entries,
host syntax and scoped git diff --check passed. Measured stage32files/1,921,745B,
logs1,687B, below8MiB/512KiB. LF/CRLF warnings remain. Two documentation patch
anchor failures left files unchanged; append-only closeout avoided further
anchor attempts. These were tooling failures, not scientific runs.