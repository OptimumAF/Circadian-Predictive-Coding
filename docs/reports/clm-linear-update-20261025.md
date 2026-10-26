# R2.1-CLM.LU — sequential wide categorical update

Issue [#162](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/162),
2026-10-25. Scoped software implementation and single-agent self-review complete.
Native usefulness, broad learning/retention/recovery, comparative benefit and
release readiness remain unestablished. No independent acceptance is claimed.

## Delivered behavior

The existing wide categorical control now has an explicit warm update:
`Wnext = W + lstsq([X,1], one_hot(Y)-[X,1]@W, rcond=1e-12)`.
It preserves prior coefficient components invisible to the retained design space;
it does not reset the model as a cold fit would. The existing thin solver,
intercept, class objective and stable softmax are reused. No tuning parameter,
new dependency or RNG. Last-batch rank/spectrum/residual/row metadata remains
distinct from lifetime work. The current immutable result retains no prior chain.

The thin adapter admits the exact original TRAIN or disposable probe_train role,
encoder identity and ordered catalog, using existing batch/target validation and
feature/source guards. Roles cannot cross in either direction; final/readout
batches cannot update. Existing readout accepts the new control; probe serving
remains prohibited by its unchanged boundary. Paid lost returns and changed
source/input evidence leave an incomplete/uncertain report and attempted/returned
prefix. A reused report key is refused. Lifetime budgets remain caller-owned.

Files and purpose:

```text
src/core/wide_categorical_linear_fit.py       appended pure residual update
src/adapters/clm_linear_update.py             role, catalog, source and work boundary
tests/test_wide_categorical_linear_update.py  nonzero-prior/projection/rank/overflow tests
tests/test_clm_linear_update.py               sequential learning, replay, role/failure tests
docs/clm-linear-updates.md                    usage, mathematics, costs and extension limits
artifacts/runs/clm-linear-update-20261025/    finite software qualification/receipts
```

Existing cold fit/readout functions and CLI/hook defaults were unchanged. The
existing temporal runner has not yet been extended to call this port. No new
stream framework, native model, deployment or scientific bank access was added.

## Executed checks and observed costs

```powershell
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-update-20261025/launch.py windows-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-update-20261025/launch.py linux-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-update-20261025/launch.py windows-2
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-update-20261025/launch.py linux-2
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-update-20261025/statics.py statics-1
.venv/Scripts/python.exe -B artifacts/runs/clm-linear-update-20261025/statics.py statics-final
```

The launcher records its exact child command/environment, terminal code, output
and parent time. Linux uses the existing cached WSL CPU runtime, no installation.
Pytest plugins disabled, numerical CPU threads1; sequential platform execution.

| Cohort | Outcome | Body seconds | Parent seconds | Sampled peak RSS, bytes |
| --- | --- | ---: | ---: | ---: |
| windows-1 | 25 PASS | 5.137363400 | 5.553783700 | 253,313,024 |
| linux-1 | 25 PASS | 12.051975082 | 17.113470600 | 286,208,000 |
| windows-2 | 15 PASS | 2.855951400 | 3.253032400 | 249,724,928 |
| linux-2 | 15 PASS | 8.511638157 | 11.870315800 | 282,611,712 |

Final27 distinct qualified cases per platform by reuse plus targeted checks,
not a single final27-case run. Cohort1 covers ten core cases, original13adapter
cases and two existing cold scalar/contradictory-row cases. Cohort2 executes the
final15adapter cases. Core product code remained unchanged; a test-only type
annotation changed. Accepted unchanged157/160 mathematics/role/readout evidence
was reused rather than rerunning whole suites or native experiments.

Core oracles include a nonzero prior with an invisible coefficient direction,
one analytic full4096-width rank-one projection, contradictory/rank-deficient
targets and invalid/nonfinite/overflow rejection. There is no solver oracle
comparison against the same implementation. Literal TRAIN and probe A→B→A
sequences learn the current labels, visibly interfere with the previous task,
then recover on return. Source coefficients and caller Torch RNG remain unchanged.
These are deliberately constructed correctness examples, not scientific results.

Self-review found the original replay example orthogonal and too easy. Before
cohort2, it was strengthened to old feature0 and new feature0.5 with opposing
labels: current-only update predicts the new class on both old rows, while
explicitly replaying the old constraint solves both distinct observations. Both
rows are counted, including replay. Reverse probe→TRAIN refusal and prior-value
corruption checks were added. No acceptance criterion or scientific metric was
weakened; initial passing outcomes keep their narrower interpretation.

Per platform: cohort1 paid27 real LSE/55fitting rows and14readouts/32rows;
cohort2 paid26LSE/52rows and11readouts/22rows. Total53solves/107row exposures,
25readouts/54rows, exactly one full4096-width solve; all others width<=4. Synthetic
nonfinite solver-return injection performs no real solve and is not included in
LAPACK counts. Limits64solves/256rows/64reads256rows/one fullwidth were not exceeded.
Bodies total7.993314800s Windows and20.563613239s Linux (<60s); parents8.806816100s
and28.983786400s (<90s), two cohorts each. Literal data only: zero pretrained
loads/encoder/head constructions/optimizer/backward/scheduler/physical random
parameter draws/simulator calls or scientific artifact reads. Reports count
logical array sizes, not all copies/LAPACK workspace/FLOPs. Sampled process RSS
is not model-attributed memory or an instantaneous allocation peak.

Initial statics: Ruff/format PASS, mypy1ERROR in the software meter's overloaded
function assignment;24.324436200s. Replaced that assignment with equivalent
`setattr`, no product mathematics or test workload change. Final seven-file Ruff,
format, default mypy and Linux-target mypy all PASS,22.250483300s. Total46.574919500s
plus bounded formatter commands remains below120s. Actual Linux statics were
unavailable/uninstalled. This meter typing-only fix was statically checked;
unchanged numerical/adapter behavior was not rerun a third time.

Conservative cache count from19:55UTC: touched109,690,896B (<160MiB), inherited
global596,512,213B measured separately. Stage27,466B/23files before final report
and self-review documents, below16MiB/platform. One atomic documentation patch
failed to match context and made no edits; exact-context repair succeeded. No
behavioral test failures, no resource-cap changes or renewed scientific budgets.
The downloaded historical execution prompt was absent; current AGENTS explicitly
supersedes its coordination policy, so this did not block authorized work.

## Self-review, skips and remaining claims

Reviewed the actual core update, adapter, both tests, qualification commands,
terminal XML/JSON/output, source/math/role contracts and results. The residual
objective includes the intercept; returned error is recomputed on the actual
updated coefficients and current one-hot labels. The solve protects no unseen
old constraint. Prior model/identity owners and values, features and targets
are checked around return. Output controls remain detached with the original
role. No older success-record chain or silent replay is retained. Core imports
only core logic; the adapter depends inward and reuses existing boundary helpers.

Skipped unchanged fullsuite, pretrained/native/CPC comparisons, real task
generalization, retained-bank analysis, CLI integration, package/cleanclone,
GPU/CI/browser and independent/release checks. No install/download/commit/push/
deletion. All prior scientific failures, negative results and spent allowances
are preserved. In particular, #161 remains a cold marker result and its paid
bank grants no future analysis; #153/#151/#159 budgets are unchanged.

Next implementation action: inspect `_run_head_stream` and existing catalog
stream ports, then add a bounded linear-update learner option to that existing
temporal runner, with arrived TRAIN packets, lifetime solve/exposure counters and
the existing score/probe boundaries. Test a complete literal sequence through
that runner before freezing a small current-only/replay study. Do not consume
the retained bank or reload8B as part of this software allowance. Broader
R2/R4/R5.1/R5.3/G5/R7 and release remain open; full goal ACTIVE. Historical plans
and development log stay frozen under the current GitHub issue workflow.
