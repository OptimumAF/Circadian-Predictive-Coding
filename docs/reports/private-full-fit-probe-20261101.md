# #412 — explicit disposable full-network fitting control

An exact BP probe can now select `optimizer="full-gauss-newton"` in the existing
`ContinuousProbePlan`/`run_continuous_probe`. The final21focused literal checks
passed, including two real private updates through the delayed-label runtime.
The original source and actor stayed unchanged. This qualifies software
composition, not native task success or acquisition/retention/recovery.

## Scope and design

`src/adapters/continuous_probe_fit.py` binds immutable tuples from an explicitly
released `probe_train` episode. Only identical float64 runtime arrays can enter
the shared owned full-network update. `probe_test`, final and old TRAIN readouts
are not training authority. The disposable owner retains its original TRAIN
policy, with separate reserved runtime authority; no label or role conversion.

`numpy_continuous_bc.py` extracts the existing update body into
`_apply_full_gauss_newton_batch`. The public entry still requires exact BP and
a TRAIN policy. AST comparison proves the inner owned commit body identical
apart from its docstring;17other methods are identical to this turn's snapshot.
The pure #409 objective/Jacobian/curvature/solver code is unchanged and its
accepted mathematical checks are reused, not rerun.

The existing recorded wrapper supplies the explicit binding only in the new
mode. Default SGD, PC/neutral-CPC/active-CPC and nearest behavior retain their
contracts; unsupported full-network choices and SGD-specific diagnostic/velocity
controls refuse before optimization. The new mode forces original source,
runtime-owner, budget and clock checks even without an endpoint observer.

Returned full updates retain original-loss diagnostics, after-loss, step norm,
all momentum-reset provenance and selected kernel/solver/normalization work.
Unknown flags aggregate as booleans; false solver-work knowledge is not a zero
cost claim. SGD-derived work is zero/inapplicable in this mode. Failed native
work and every returned paid row survive late corruption/refusal. The private
port permanently rejects further updates/forking after failure. Only bounded
scalar receipts return; no private learner, parameters, promotion or source
budget renewal. Old examples are readouts only, not implicitly authorized replay.

Why this: #411's fixed16SGD updates lowered imitation error but failed both
fresh native resets; all old imitation errors increased and old-native retention
was unfinished. A separate fitting control can test optimizer limitations
without a seed/dose search or another audit framework. No such actual comparative
experiment ran here. #411 remains FAILED/SPENT/PARTIAL, not resumed or repaired.

## Executed checks and failures

Physical working directory `D:\Dev\Repositories\Circadian`; TEMP/TMP on
`D:\Dev\_Temp`, bytecode disabled, numerical threads1. No constructors/model
RNG, actual inputs/datasets/checkpoints/probe artifacts, SDK/physics/native
episodes, installation/download or global environment changes.

Exactly two prospective software commands:

```powershell
.venv\Scripts\python.exe -I -B artifacts/runs/private-full-fit-probe-20261101/check.py initial
.venv\Scripts\python.exe -I -B artifacts/runs/private-full-fit-probe-20261101/check.py final
```

Initial18cases:10PASS/8FAIL, .933755400s/71,467,008B; only8literal sources,
zero numerical work. The literal binary-gripper unit was incorrectly named
`gripper`; correcting it to the required `normalized_gripper` preserved the
production contract. The original failed receipt/log remain. Distinct evaluation
targets were then used to make the binding refusal test substantive. Invalid
policy construction was included inside its refusal assertion. No optimizer
configuration, scientific dose, outcome metric or production criterion changed.

Final21cases:21PASS, exit0, .917775100s/69,054,464B, budget_ok true. Combined
39executions/31PASS/8fixtureFAIL,1.851530500s,43of48allowed owners. Actual work:
12full-update attempts/36rows,12solve attempts,23gradients/69rows,47forwards/
141rows,1defaultSGD/3rows,629snapshots/181,152selected payload bytes. Failed
solver calls are counted even when deliberately injected. Source count includes
native forks. Both command slots are SPENT; unused numerical quotas do not carry
over into experiments. The scoped60s/256MiB/24full-updates limits held.

The checks cover delayed labels/no premature update, source/actor/clock/budget
preservation, objective descent on the literal packet, momentum reset, unchanged
SGD schema, public TRAIN authority, option/family/packet refusals, failed solve
accounting/no retry/fork, post-proposal source corruption, paid-record erasure/
injection after two returns, and replacement of the runtime candidate after a
paid return. The final combined tests include the review's concrete repairs.

Final scoped static commands passed:

```powershell
.venv\Scripts\python.exe -m ruff check src/adapters/numpy_continuous_bc.py src/adapters/continuous_probe.py src/adapters/continuous_probe_fit.py tests/test_private_full_fit_probe.py artifacts/runs/private-full-fit-probe-20261101/check.py
.venv\Scripts\python.exe -m ruff format --check src/adapters/numpy_continuous_bc.py src/adapters/continuous_probe.py src/adapters/continuous_probe_fit.py tests/test_private_full_fit_probe.py artifacts/runs/private-full-fit-probe-20261101/check.py
.venv\Scripts\python.exe -m mypy --follow-imports=silent src/adapters/numpy_continuous_bc.py src/adapters/continuous_probe.py src/adapters/continuous_probe_fit.py tests/test_private_full_fit_probe.py
```

One initial plan-union typing error and three later annotation errors were
corrected. The final numerical cohort preceded the final annotation-only fixes;
statics were rerun afterward. Lead then restored the exact original public
first type-refusal guard, verified AST-identical to the old first guard. This
last defensive source-order restoration and module-docstring changes received
self-review/statics, not an additional numeric or independent rerun. The
unchanged exact-BP numerical path remains covered by the21checks. Early guessed
test/inbox paths and PowerShell wildcard searches failed during source discovery;
file enumeration resolved them without numerical retries.

## Independent review and remaining criteria

One explicit finite peer exception, three read-only batches<=180s, actual~.5s,
lead owns all writes and sole numerical budget. Reviewer found: injected paid
rows/partial restoration could distort costs, failed port fork could renew
authority, and a replaced runtime candidate could erase capture. All three were
fixed before the final cohort. Reviewer directly read tiny initial/final receipts
and final source, accepted scoped software/source/isolation/results, and ENDED.
No imports/tests/models/data/native/RNG or independent numerical replication.
The later restored original early type guard is explicitly self-reviewed only.

[Prospective limits](../../artifacts/runs/private-full-fit-probe-20261101/SOFTWARE-PROTOCOL.md)
and original/final receipts/logs/source snapshots/preservation metadata are kept
under `artifacts/runs/private-full-fit-probe-20261101/`.

Skipped unchanged #409 math/default probe/PC/CPC/native suites, extra numerical
runs, Linux cohort, broad suite/cleanclone/package/release/GPU/remote CI and all
final data. No commit/push/delete or unrelated dirty-work change; masterd9119e8.
No scientific advantage, simultaneous retention or reliable recovery claim.

#412software criteria1–3 were evidenced at the initial software handoff while
criterion4's separate protocol/review remained unchecked. Subsequently the
[fixed #413protocol and review](private-full-fit-native-20261101.md) fulfilled
criterion4 prospectively before inputs; #412scoped software is now CLOSED.
#413then completed a negative native measurement. No actual grant followed from
the original software milestone itself. Parents18/21/23/R2.1/R5/science/release
and full goal remain OPEN.

The software handoff directed one fixed full-network private case from the same
qualified80source and original23/24/old17/19/20 roles/seeds/objective, with extra
computation explicit and only future copy engineering limits revised. That
protocol was frozen/reviewed before inputs, then executed as #413 with all five
native outcomes negative. The spent SGD case and its missing native retention
were not rerun or retrospectively qualified. Current next implementation is
[#414](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/414), the
separate reserved-training corrective-query boundary described in the #413report.
