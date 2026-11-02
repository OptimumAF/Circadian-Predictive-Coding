# Single-head minimum-change TRAIN through the ordinary stream

R2.1-LIN.SHARED / [#351](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/351).
Implemented and executed scoped software checks. Native acquisition, retention
and recovery for this new ordinary single-head option remain **unmeasured**.
The full repository/scientific/release goal remains active.

## Working behavior and scope

The existing command now forwards explicit single-head TRAIN selection:

```text
python -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --train-update minimum_change --output NEW.json
```

`ticket_workflow_linear.counted_learner` calls the already accepted
`LinearFitLearner.train_batch_minimum_change` on its owned candidate/forks.
An unfitted owner uses its existing cold solve; warm updates use prior
coefficients. Omitted selection remains cold. Nothing was added to the kernel,
adapter, model state, feature map, task data, optimizer, replay bank or runner.
This can preserve only prior components invisible to current rows, not all old
predictions. Separate-context positives do not establish single-head retention.

The ordinary API and CLI admit the explicit operation without separate heads;
paired/probe/private/controller modes keep their separate restrictions. The CLI
now actually transports the selection rather than merely admitting the flag.
Report intent and cost objective distinguish the single-head minimum-change
residual from both cold and contextual residuals. Default scope wording is
unchanged; the explicit scope says minimum-change without claiming benefit.

Existing arrived TRAIN role/consent/label-delay checks, source/actor/RNG guards,
original budgets/clocks, paid prefix and failed-work receipts remain active.
Corrupted update intent or preceding checkpoint refuses more work after four
successful scripted commits. The original paid prefix survives. A lost real
warm fit leaves source, actor and stopped candidate unchanged, charges the
attempt and retains unknown return status. The failed public stream refuses
snapshot/retry access. No final/probe labels, promotion or allowance renewal.

Changed files: two existing scripts; two obsolete separate-head-only rejection
expectations in `tests/test_context_minimum_change_train.py`; new
`tests/test_linear_minimum_change_stream.py`; [usage](../linear-minimum-change-stream.md)
and a five-line README paragraph. Exact source/test preimages and readable scoped
diffs are in `artifacts/runs/linear-minimum-change-stream-20261030/`.
Inherited dirty changes and HEADd9119e8 remain. No commit or push.

## Executed checks, failures and costs

The prospective issue protocol is retained as `stage/PROTOCOL.md`; refinements
and failure handling were posted before continuation in the owning issue.
All commands use physical `D:/Dev/Repositories/Circadian`. Windows TEMP/TMP
are `D:/Dev/_Temp`; producer Linux TEMP/TMP/TMPDIR `/mnt/d/Dev/_Temp`.

| Invocation | Outcome | Seconds / sampled process peak RSS |
| --- | --- | --- |
| Windows original40 | 39 pass, 1 incorrect test expectation | 2.5669926 / 82,104,320B |
| Windows affected continuation | repaired1 pass | 1.0516066 / 68,775,936B |
| Linux first attempt | collection/capture failure, exit1 | body and peak unknown |
| Linux final40 | all40 pass, exit0 | 9.064542827 / 72,450,048B |
| Final five-file Ruff/format/Linux-target mypy | all pass | .1831109 / .1496639 / .611778 seconds |

Windows qualifies40 distinct cases by39 unchanged passes plus the single repaired
case; it did not rerun the full numerical cohort. Linux ran the final combined
tree. Source line-ending preservation changed no normalized code. The final
observer-only timing variable repair addresses type inference, not numerical
behavior, and does not require repeating passing fits. Each platform used its
two allowed invocations. The40 cases comprise21 new cases plus19 affected
existing admission/default/paired CLI cases. No test is a genuine full source
or native task experiment.

The tiny actual adapter fixtures test cold/explicit operation choice, fork
ownership and an orthogonal two-row retention example, plus a lost warm return
through the actual stopped stream. The complete capture fixtures use the REAL
runtime, roles, labels, receipts and guards with SCRIPTED fitting and task
outcomes. They script12updates/72rows and24 false task outcomes, never claim
learned success, and exercise single-head intent/cost/capture and refusal paths.

Observed across all instrumented successful/failed fixture invocations:
**14 actual tiny NumPy LS solves/14 fitted rows,28 scheduled kernel affine
products/30 rows;64 scripted callbacks/384 rows.** Attempted work is included.
These are logical schedules, not hardware FLOPs or matched computation.
Torch/physics imports are blocked; none reported in instrumented receipts.

The first Linux invocation failed during pytest collection when its FD capture
attempted `tmpfile.truncate()` on physical-D storage under WSL. It did not
produce a numerical ledger, so its actual body/RSS/work telemetry remains
unknown. The trace shows collection failure, not passing fixtures. The original
log and SPENT marker remain. Before the second/final allowed invocation, capture
was disabled with the repository's established `-s` pattern and exception
telemetry was made durable. No storage boundary was bypassed.

For conservative admission, `linux-interrupted-checks.json` permanently reserves
8solves/32fitrows,32affines/128rows and64scripted callbacks/512rows for that
missing ledger. These are **reservations, not measured actual counts**.
Including them, admission totals are22solves/46rows,60affines/158rows and128
scripted callbacks/896rows, below the original32/256,256/4096 and256/4096 caps.
The lost historical RSS/time receipt is not retroactively marked verified.
This does not renew a scientific or native allowance.

Initial statics retained two test annotation errors (4.0619296s combined tool
call) before fixtures. The first Windows failure tried to read a stopped stream
through its public API. The repaired test inspects preserved internal state and
asserts that public access refuses; production behavior did not change. Later
observer statics found five mixed-dictionary type errors after adding nullable
framework error metadata. A separate scalar timing variable fixed them; the
failed static JSON/log remains. A failed patch anchor changed no files, followed
by inspection and one exact-anchor repair.

An early saved diff incorrectly joined normalized lines without line endings;
the readable diffs were regenerated from the existing exact preimages, not
reconstructed source. Original unchanged script/test line endings were preserved.
An ancillary README byte-recovery read failed because the older package does
not include top-level README.md. It made no recovery mutation; a premature
milestone wording claim was explicitly corrected in issue comment6538807448.
No README byte proof is claimed. The actual narrow documentation paragraph
and inherited content were source-reviewed without searching more backups.

## Commands and evidence

```text
.venv/Scripts/python.exe -B artifacts/runs/linear-minimum-change-stream-20261030/check.py windows D:/Dev/_Temp/circadian351-pytest-windows
.venv/Scripts/python.exe -B artifacts/runs/linear-minimum-change-stream-20261030/check.py windows-failure D:/Dev/_Temp/circadian351-pytest-windows-failure tests/test_linear_minimum_change_stream.py::test_lost_single_head_fit_preserves_paid_attempt_and_original_runtime
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/linear-minimum-change-stream-20261030/check.py linux /mnt/d/Dev/_Temp/circadian351-pytest-linux
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/linear-minimum-change-stream-20261030/check.py linux-final /mnt/d/Dev/_Temp/circadian351-pytest-linux-final
```

Final static commands: `.venv/Scripts/python.exe -B -m ruff check`, `-m ruff
format --check`, and `-m mypy --platform linux --explicit-package-bases
--follow-imports=silent --cache-dir D:/Dev/_Temp/circadian351-mypy` on the exact
five paths in `statics-final.json`. Original/interrupted and final logs, JSON,
SPENT markers and static failures remain local in the stage. Process RSS is
sampled, not a hard instantaneous/process-tree allocation cap.

No full suite, clean committed clone, new installed package, real source12
fit, typed native task rollout, private probe, baseline/control study, browser,
GPU/simulator, final evaluation or release checks were run. Accepted kernel/math,
matched families and earlier cold/separate-head studies are reused within their
original scopes. #348/#344 and every older failed or negative result stay intact.
No seed/data/metric/direction/threshold/dose tuning or advantage claim.

## Acceptance boundary and next action

Lead source/procedure/result self-review is distinct from independent review.
The finite independent patch/receipt review in `stage/REVIEW.md` ACCEPTS covered
source integration and software evidence with no blocking defect: three
read-only commands totaling0.6529450s, no imports/tests/models/network/children.
It covered the actual patch/tests/observer/receipts/usage and the conservative
reservation, not a second execution. The report was written after its three-read
cap and remains lead-reviewed; no independent report acceptance is claimed.
The finite exception ended. Neither
scripted closed-loop outcomes nor a correct operation establishes native task
usefulness. R2.1 reliable learning, retention, recovery, matched science and all
broader repository/release requirements remain open.

Next precise action after scoped software acceptance: freeze and review ONE
small genuine single-head12-update/72-row A4/B4/A4 ordinary outcome case through
this existing command. Keep the original30s/512MiB clock, fixed checkpoints/
tasks/metrics, cost definitions and failure policy. Observe new-task acquisition,
old-task retention/loss and return recovery honestly; reuse accepted cold-fit
evidence only as a descriptive reference, not an independent matched-compute
comparison. No fresh baseline, new seeds, sweep, model or native experiment is
authorized by this software report. Do not add another capture/package sequel.


Canonical scoped acceptance: [#351](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/351#issuecomment-6538852216), closed after four software criteria evidenced.
Parent receipts:18/6538854900,21/6538855110,23/6538855303. Next #352 remains planned, no genuine study executed. Full goal active; installed wheels predate this new single-head option.
