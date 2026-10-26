# R2.1-CLM.LW — context workflow software result, 2026-10-25

Owner [#166](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/166),
single `/root` implementation/tests/self-review. HEAD09ff811f plus inherited
dirty/untracked work; no commit, push, install or scientific run. Historical
plans/log remain frozen under AGENTS.md; live handoff is in GitHub issues.

## Delivered behavior and limits

The existing ticket command now accepts
`--linear-context-updates --public-observation-train`. It uses the original
public teachers, fixed full six-tool union, original delayed-label inbox, one
three-update budget, active five-tool typed providers and actual local dispatcher.
One cold plus two warm residual least-squares solves expose12+12+12 rows. Return
explicitly reuses12 original v1 TRAIN rows, with replay consent from birth.
Teacher preparation uses24 local transitions and18 frozen encoding sequences;
each live query adds one encode. No new scheduler, loader, objective, dependency,
native head optimization, final input or probe deployment was added.

The learner's explicit generic catalog mode is TRAIN-only with no choice
registry. Its original temporal defaults remain intact. `current_control` checks
the original owner/values and returns only the current immutable control.
Serving cannot obtain an unfitted control. Prior controls are not accumulated.
Features/target IDs/source heads/backbone/scale/RNG, budget/clock/consent, receipt
owners and completed report values/containers are guarded. Exact observed encoder
returns admit their counter changes. A lost solve/request, uncertain episode,
corrupt receipt/prefix or failed final save stops work and preserves its prefix.
Valid negative episodes complete without selecting or altering later training.

See [usage and boundaries](../clm-linear-context-workflow.md). This implementation
is a workflow composition through existing ports, not another reusable stream
framework. Existing R2 learnability/fresh-task probe work and matched BP/PC/
neutral/active-CPC controls remain separate. Previously accepted repeated-update
and baseline evidence was not reconstructed or rerun.

## Literal observed outcomes

The actual public CLI ran with a literal loader and a twelve-dimensional one-hot
encoder of **public state and policy context**, not private ticket IDs or target
labels. The action features are fixed metadata. No random parameter draw or
pretrained encoder is involved. This is a software oracle, not native usefulness.

| After exposure | v1 invoice | v1 password | v2 invoice | v2 parcel |
| --- | --- | --- | --- | --- |
| v1 TRAIN | success | success | max steps, failed | max steps, failed |
| v2 TRAIN | success | success | success | success |
| repeated v1 TRAIN | success | success | success | success |

All14 episodes completed. The fixed stale-password control succeeded in5 steps;
the parcel commit-revocation control stopped with external permission denial in3.
Ordinary successful tasks used3 steps; the two untrained-context negatives used6.
There were50 query predictions,52 total encoder forwards and68 encoded sequences
(18 preparation +50 query),3 actual solves/36 rows and receipts at observed0/2/4,
arrived/applied1/3/5. Every episode had0 training updates and0 external actions.
Old v1 tasks remained successful after v2 **before** return rehearsal. Therefore
learning recovery is **false**; stale-state episode recovery is true. No benefit
over a simpler learner, generalization, matched computation or circadian advantage
is established. The early v2 failures are retained.

## Checks and failures

Windows: initial22 checks produced21PASS/1FAIL; focused final follow-up4PASS.
The failure was an incorrect test expectation that all contexts must succeed
before both were trained. It was corrected to require retention of the observed
two complete max-step negatives. No model, data, seeds, metrics or training
schedule were changed to improve an outcome. Final-save corruption also now
preserves the original summary/prefix; its affected test passed in the follow-up.
Linux final combined tree:22/22PASS. There are22 distinct qualified checks per
platform (26 Windows executions,22 Linux), with0 skips.

These include19 new CLI/sequence/failure checks, one affected original temporal
label-ID/active-softmax boundary check, and two original single-fit workflow
checks sharing one12-row fit. No full12-update baseline or default contrastive
training rerun was needed; earlier accepted unchanged-port evidence is reused.
Checks cover actual command/output exclusivity, repeated-row/clock/receipt/work
accounting, old/new context outcomes, inactive tools, source/RNG preservation,
receipt/episode/work corruption, lost warm/prediction returns, final-save failure,
teacher-role refusal before encoding and invalid options before loader/output.

Initial scoped statics: Ruff/formatPASS, mypyFAIL with5 annotation errors.
Second: Ruff/formatPASS, mypyFAIL with1 remaining dictionary annotation error.
Direct annotation fixes used no suppressions. Final8-file Ruff, format check,
default-target mypy and Linux-target mypy allPASS. Static elapsed totals:
format0.1717002s +22.6584741s +2.1456200s +28.6412352s =53.6170295s, below180s.
All failed outputs remain in the stage.

Exact commands, from repository root:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-context-workflow-20261025/statics.py statics-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-context-workflow-20261025/statics.py statics-2
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-context-workflow-20261025/launch.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-context-workflow-20261025/launch.py windows-2 tests/test_clm_linear_context_workflow.py::test_should_execute_actual_cli_and_original_three_exposures tests/test_clm_linear_context_workflow.py::test_should_measure_retention_before_return_and_separate_recovery_claims tests/test_clm_linear_context_workflow.py::test_should_preserve_source_rng_and_account_cumulative_work tests/test_clm_linear_context_workflow.py::test_should_fail_final_save_and_preserve_completed_work
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-context-workflow-20261025/launch.py linux-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-context-workflow-20261025/statics.py statics-final
```

The separate initial formatting command/elapsed time is recorded in
`statics-format.json`. `statics.py` receipts contain the exact expanded8-file
commands. Parent receipts contain the exact WSL interpreter/environment/timeout.
The launchers disable third-party pytest plugin autoload, hold CPU threads to1
and use the existing cached WSL environment. No new environment was installed.

## Resource observations and failed copy reserve

| Execution | Body wall s | Parent wall s | Peak sampled RSS after imports B |
| --- | ---: | ---: | ---: |
| Windows1, failed assertion | 6.3194335 | 6.7369058 | 272166912 |
| Windows2, focused follow-up | 6.9028081 | 7.3785718 | 262094848 |
| Linux1 | 21.4482340 | 26.4307637 | 299323392 |

Windows cumulative:20 solves/231 rows,241 readout calls/255 rows,65 provider
bindings,258 fake encoder calls/412 sequences/412 visited IDs/412 padded slots,
36 literal module objects,67 complete/partial episodes and478 local dispatches
including teachers. Linux:14/159 solves/rows,141/155 readouts/rows,37 bindings,
154 encoder calls/276 sequences/IDs/slots,30 modules,39 episodes/330 dispatches.
Both:0 native head forwards, physical random parameter draws, optimizer/backward/
scheduler/simulator work. Model/readout/encoder/module/effect counts use direct
wrappers; absence of optimizer/simulator paths is also source-reviewed.

The original Windows16-solve/192-row/256-encode/384-sequence allowance did not
cover the focused follow-up. Before executing it,
[the recorded amendment](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/166#issuecomment-6403560834)
prospectively admitted20/231/258/412 for Windows only. Other operation, CPU,
two-cohort,90s body/120s parent, RSS and storage caps stayed fixed. Linux fit the
original limits. The initial failed assertion and all its charges remain.

**The original64MiB other-copy reserve did not pass.** The wrappers did not count
repeated report-guard JSON serialization. Source review of the actual new literal
capture found136029 bytes of episode JSON and at least24583000 bytes of episode
prefix serialization per full case. Four full Windows cases imply98332000 bytes,
already above67108864 before other report/source serialization. This is a
conservative lower bound on logical serialized bytes, not physical allocation,
peak RAM or FLOPs. It is recorded as a failed engineering reserve, without a
retrospective pass or another fixture run. The32M logical input/source/guard/RNG
element reserve is separately declared; no universal copy bound is claimed.
Functional checks, observed operation counts, timings and RSS remain valid scoped
evidence. They do not certify the original resource allowance or native readiness.

Source/cache inventory in `source-and-storage.json`: touched whole cache files
116414928B, separately from603452784B inherited/global cache files; the touched
160MiB ceiling held. Stage916589B before this review/report, below16MiB. The
inventory SHA256-binds the final5 product/test files and3 qualification scripts.
Earlier cohorts used the earlier test expectation/failure-report variant; the
Windows follow-up and complete Linux cohort cover the changed behavior. No
native artifact/bank/old scientific actual, ledger or detached review was read.

## Self-review, disposition and next action

Source review: default-off admission precedes loader/output; the old temporal
catalog and original workflow routes retain their semantics. Teacher roles,
public texts, catalog order and target IDs bind before encoding. Readouts consume
only current public requests with no teacher labels; final/probe inputs cannot
enter this composition. The original fitted control guards, label inbox and
external permissions/effect runner are reused. Only one current model/provider
scope is live; reports retain a finite three-exposure/fourteen-episode history.

Procedure/results review: inspected the final sources, explicit limits, direct
meters, JUnit/output/JSON receipts, failed assertion/static checks and the copy
reserve undercount. Functional acceptance is evidenced; the failed reserve is a
permanent limitation, not cost qualification. This is self-review, **not independent
acceptance**. Broader R2/R4/R5.1/R5.3/G5/R7/release parents and the full goal stay
open. The old wheel predates this feature. Clean-clone/package/full-suite/native/
scientific/GPU/CI/browser qualification was skipped. No frozen evidence was
rewritten and no historical live-plan/log checkboxes were mirrored.

**One next action:** freeze and self-review one fresh native context-workflow
protocol for the existing CLI, including loading/shared setup,18 preparation
sequences, fixed3 solves/36 rows and14 bounded development episodes, explicit
validation/serialization costs, original540s inner wall/RSS gates and stop-once
failure policy. Check host admission before any prepared-weight access. This
software result grants no such attempt. Do not reconstruct the transient #164
model, reuse its spent bank authority, tune failed contexts or start a sweep.
Only after an admitted observed case should the next isolated fresh-task probe
or matched-method experiment be selected.
