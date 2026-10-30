# R2.1-CTX.LS.J.X — failed pre-fit comparison; admission repair verified

[Issue #287](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/287),
parents #18/#21/#23/#33 and full repository goal OPEN. Previous goal turn was
PROGRESS (#286 delivered dynamic joint replay). This turn implemented its small
comparison driver and executed the sole frozen case, which **FAILED/SPENT before
any fit or readout**. Comparative acquisition, retention and recovery remain
UNKNOWN. This is an engineering failure, not a negative learning outcome.

## Delivered code and boundaries

```text
scripts/compare_joint_context_retention.py       fixed two-arm existing-stream composition
tests/test_joint_context_retention_comparison.py pure/mocked and real zero-fit admission checks
artifacts/runs/joint-context-retention-comparison-20261029/
  PROTOCOL.md                                  pre-execution frozen question/limits
  actual.json                                  original FAILED/SPENT body receipt
  execution.json                               original FAILED/SPENT parent receipt
```

The script uses original contextual TRAIN teacher records and the strict TRAIN
batch builder. It never calls diagnostic cohort builders. The old four tasks,
two new legacy tasks, and four old returns use three-row complete episodes and
the accepted StreamRehearsal dynamic episode bank. Controls are joint/current_only
at fixed checkpoints0/4/6/10, initially identical unfitted LS heads. Both arms
must finish before any readout; all six task readouts are fixed prospectively.
No new learner, optimizer, runner, dependency, final/probe labels, tuning or
promotion. Existing static recipes and BP/PC/neutral/active CPC evidence are
preserved. This conventional LS diagnosis is not a new CPC comparison.

The script records permanent matrix exposures, physical fits/solves, typed and
sandbox work, retention/model coefficient bytes, elapsed/RSS scopes, and failures.
A whole-study cap exception bypasses helpers' model-Exception recovery because a
profiler exception disables observation; no unobserved continuation is admitted.
Exclusive output creation spends the case. Current source contains an admission
repair after the failed execution; **the current combined tree has not executed
the comparative model cohort**. Its only new runtime evidence is zero-fit admission.

## Actual outcome and costs — no retry

The one Linux command exited1:

```text
ValueError: wake runtime budget must not contain a replay limit
```

At execution, `_sessions` set the wake's `max_replay_examples=0` and the replay
session's `max_training_updates=0`. The runtime correctly distinguishes a present
zero limit from an absent limit. The wake/replay authority validator refused
before the first fit. The protocol's intended36fits192rows/48readouts did not
occur. The pure positive-flow tests had mocked this boundary; source review also
missed the zero/None distinction. No runtime guard or acceptance criterion was
weakened to make the command pass.

| Measured sole case, including failed producer work | Actual |
|---|---:|
| Shared original TRAIN teacher traces / sandbox dispatches |8 /24|
| Fit attempts / returned fits / physical solves / fit rows |0 /0 /0 /0|
| Route calls / rows; full-class probability calls |0 /0;0|
| Typed delegates / affine / unfitted reads |0 /0 /0|
| Readout episodes |0|
| Body seconds before JSON output |0.053608432|
| Body sampled peak RSS bytes |45,043,712|
| Pipeline seconds through serialization/write |0.055253896|
| Sampled peak RSS bytes through output |45,170,688|
| Parent launch/import/body/output/exit wall seconds |3.6041804|
| Parent/child exit and disposition |1 /1; FAILED/SPENT|

No Torch/neural/GPU/random model initialization, probe/final access or old
scientific raw reads. The8teachers24dispatches remain charged, not refunded.
The unused fitting/readout allowance is closed. No repaired rerun, retry, renamed
near-duplicate, seed/dose/metric search or renewed numerical allowance followed.
Both original receipts remain unchanged. No retained source model was loaded or
historical parameter matrix reconstructed. #280's failed static case and #285's
accepted private-probe negative result remain untouched.

## Repair, tests and review

The minimal post-failure source repair changes only the two irrelevant session
limits from0 to `None`: wake retains its original10update cap, replay retains
its original66row cap. The real constructor now admits this composition without
any fit. The new regression constructs the actual sessions, active RSS sampler,
unfitted ContextLinearFitLearner, StreamRehearsal and AttributionStream, then
checks original/source/candidate equality and zero budgets/coefficients. It
never advances the stream. An autouse refusal blocks actual teacher generation,
workflow episodes and `np.linalg.lstsq` throughout these engineering tests.

Final **15focused cases pass on Windows and Linux**, with no skips/failures.
Coverage includes protocol preview without data/model work, original producer
selection/order, TRAIN-only complete packets, owned arrays and hard actions,
both training arms before all readouts, failed/incomplete/changed training
refusal, old versus new metric scope, null recovery, permanent failed output,
exclusive output refusal, cap abort, update0 capture, and the real admission
boundary. This proves neither successful full comparative execution nor release.

Engineering invocations (all zero fits/teachers/rollouts/random draws):

| Platform | Invocation pass counts | Aggregate command wall seconds |
|---|---|---:|
| Windows |12,13,14,15|4.8395607|
| Linux |12,14,15|24.8996195|

Both stay within4invocations/180s; no additional fits were disguised as tests.
Whole-pytest RSS was not separately instrumented; active sampler/admission paths
were exercised. Do not infer a whole-suite peak-memory measurement from the study
receipt. Scoped static command allowance120s was not renewed.

Initial static checks found an unused import, wrong strict-validator argument
count, missing annotations, then Optional RSS narrowing. Repaired before final
checks. One attempted text replacement did not match and changed no behavior;
the exact replacement fixed it. Two patch-anchor failures changed no files;
switched to ordinary exact text edits instead of extending tooling. Self-review
caught `run_until(0)` refusal before model execution and changed it to initial
snapshot capture. All these costs/failures remain in issue history.

Independent protocol/source review,3readonly calls/90s allowance, accepted the
original TRAIN data path, fixed schedule/dose, evaluation separation and cap
abort. It required two fixes before GO: acquisition flags only for new tasks
and loss/recovery only for old tasks; and separate body/output/parent telemetry.
Both were repaired and re-reviewed. The reviewer missed the later observed
budget-admission bug. Source acceptance is not independent execution or result
replication. The finite exception ended; final result/repair review is separately
bounded to1readonly call60s, no models/tests/old raw/writes.

## Exact commands and qualifications

All commands use physical `D:/Dev/Repositories/Circadian`, Windows
TEMP/TMP=`D:/Dev/_Temp`, Linux TEMP/TMP/TMPDIR=`/mnt/d/Dev/_Temp`. Python-B.
Windows Python3.14.7; existing Linux Python3.12.3/NumPy1.26.4. HEAD remains
0aae1d6 with inherited dirty270–286 plus these changes; no commit/push/install/
download/delete/global settings or unrelated cleanup.

Final focused checks, exit0/15PASS each:

```text
.venv/Scripts/python.exe -B -m pytest -q --tb=short --maxfail=1 --basetemp=D:/Dev/_Temp/circadian-j287-win4 tests/test_joint_context_retention_comparison.py
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B -m pytest -q --capture=sys --tb=short --maxfail=1 --basetemp=/mnt/d/Dev/_Temp/circadian-j287-linux3 tests/test_joint_context_retention_comparison.py
```

Scoped statics: `.venv/Scripts/python.exe -B -m ruff check` and
`ruff format --check`, `mypy --explicit-package-bases --follow-imports=silent`
(default and `--platform linux`) over the new script and test. Final whitespace
command `git diff --check` passes with inherited CRLF advisories. Broad unchanged
math/native/baseline, whole repository, clean clone, current installed wheel,
remote CI, GPU/physics and release checks skipped outside this scope. #279's
wheel predates these changes.

Sole failed study command (historical reference; **do not rerun**):

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B -m scripts.compare_joint_context_retention --output artifacts/runs/joint-context-retention-comparison-20261029/actual.json
```

PowerShell owned an exclusive execution.json handle before launching WSL and
used Stopwatch across the command, captured child stdout/exit, checked parent
wall<=60s, and closed the receipt. Body-only completion cannot override terminal
or parent failure. No desired task outcome was asserted, selected or inferred.

## Unfinished outcome and precise next action

The software admission repair is delivered and tested. #287's intended
comparative task endpoints and dose qualification remain unchecked/UNKNOWN;
its numerical case is permanently FAILED/SPENT. Prior286six-workflow fixture
evidence remains accepted, not a substitute for this missing comparison.
Full goal/parents and broader acquisition/retention/recovery/generalization/
deployment/release remain OPEN. No advantage or cost-effectiveness conclusion.

Reassess beyond this failed comparison: current `context_head_live.run_owned_case`
delegates to `context_head_rehearsal.prepare_owned_source`, which calls the exact
perceptron `copy_context_snapshot` and hardcodes its12/72/16mistake checkpoint.
It therefore cannot admit the owned learned LS policy already available. The
next useful implementation is an explicit opt-in ContextLS source/predictor path
in that existing live serving composition, preserving strict default perceptron
admission and original budgets/receipts. First inspect its learner/predictor
construction and add zero-fit admission/refusal tests before implementing the
smallest typed LS branch. This targets serving learned behavior, not another
capture/count audit or a near-identical repaired comparison. No new model case
or scientific allowance is granted by this handoff.

GitHub remains the live tracker; requested plans/log entries are archival
evidence pointers only. Existing requirements, negative results and user changes
are preserved.


Final independent result/repair review accepted the failure receipts and minimal
None-limit repair, with the wording correction above; one readonly call, no
imports/tests/models/writes. It confirms UNKNOWN comparative endpoints and no
current-tree numerical execution, not independent replication. Exception ended.
