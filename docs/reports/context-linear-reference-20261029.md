# R2.1-CTX.LS — software qualified; scientific case failed before fitting

[Issue280](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/280)
remains OPEN at its scientific qualification criterion. Full goal and broader
parents18/21/23/33 OPEN. This turn is PROGRESS: implemented and executed a new
reference route, corrected a real paid-task accounting gap, and preserved a
failed scientific admission. No claim of LS task learning or native repair.
GitHub holds live scope/handoff; historical plans/log remain preserved per AGENTS.

## Working functionality and reviews

Existing `src/core/categorical_linear_fit.py` gains a full-logit read port using
the existing fixed design and coefficients. Fit/SVD/softmax definitions are
unchanged. New `src/adapters/linear_fit_decision.py` owns model/spec copies,
guards state/work identity and values, and normalizes candidate-local logits.
Pure underflow fixture shows allowed probabilities remain meaningful when an
unavailable class dominates global mass. Returned logits/work are owned copies.

Existing `scripts/qualify_ticket_bp_fit.py` gains default-off
`linear_reference=True`/`--linear-reference`, only with policy context. Its new
helper `scripts/ticket_linear_reference.py` reuses original TRAIN teacher, case-
packet alignment, fixed12row solve, original TRAIN metrics and actual sandbox.
One fit/30s/512MiB budget retains origin/limits. Input/model/schema/RNG and paid
prefix guards remain separate from fit/task outcomes. No fake native optimizer
receipt/state or deployed/benchmark owner;0Torch/neural/probe/final/replay work.

An independent source/math/data-isolation reviewer used2readonly calls and
found that post-task validation could discard an already returned task/work.
Repaired by recording a pending task before delegation and copying/pinning the
paid return before post-task validation; only validated complete returns become
accepted. Lost returns remain pending/unknown. Regression tests prove late fit-
record corruption retains the task and starts no further task, prior episode
corruption preserves the canonical prefix and stops, and lost return keeps its
attempt. One-read followup resolved the concrete finding, no further blocker;
finite exceptions ENDED/0imports/models/tests/raw/artifact reads/writes. Lead
combined-source/procedure/result self-review complete. Source review is not
independent numerical replication or scientific acceptance.

## Executed software checks, failures and costs

27distinct focused checks/platform PASS,0skips: API/CLI admission and exclusive
output/default forwarding, scalar math/owned logits/candidate normalization,
readout model/spec/work tampering, malformed arrays, TRAIN role/target alignment,
actual complete task fixture, RNG/input/solver/read failures and paid task prefix.
Success assertions check complete execution/isolation/accounting, not a scientific
task-success threshold or replacement study outcome.

Physical cwd `D:/Dev/Repositories/Circadian`; Windows TEMP/TMP=`D:/Dev/_Temp`.
Linux TEMP/TMP/TMPDIR=`/mnt/d/Dev/_Temp`, existing source interpreter
`/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python`.
Evidence stage `artifacts/runs/context-linear-reference-20261029/`.

Exact final Windows command:

```text
.venv/Scripts/python.exe -B artifacts/runs/context-linear-reference-20261029/check.py artifacts/runs/context-linear-reference-20261029/win-final-work.json tests/test_ticket_linear_reference.py --basetemp D:/Dev/_Temp/context-linear-win-3
```

Linux counterpart:

```text
wsl.exe -e bash -c 'cd /mnt/d/Dev/Repositories/Circadian && TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-linear-reference-20261029/check.py artifacts/runs/context-linear-reference-20261029/linux-final-work.json tests/test_ticket_linear_reference.py --capture=sys --basetemp /mnt/d/Dev/_Temp/context-linear-linux-2'
```

Initial26PASS/platform1.8321172s/6.913359733s. Intermediate Windows repair run
1PASS1FAIL retained in `win-paid-work.json`,1.8258704s/70,737,920B:2LSfits24rows,
49full-logit calls71rows47typed95dispatch. The test mutated the producer's old
return alias, which the ownership repair now detaches. Corrected fault injection
to the actual published record through the existing copy boundary; no production
test hook or acceptance weakening. Also fixed pending-dict/NumPy tuple typing
and unused import. Final27PASS1.9310759s/6.606579184s, each8LS96rows/65full-logit
calls120rows/59typed341dispatch. All failures/repetitions permanently counted.

| Software total | Windows | Linux |
|---|---:|---:|
| pytest invocations |3|2|
| body seconds |5.5890635|13.519938917|
| sampled peak RSS bytes |71,028,736|72,929,280|
| LS fit attempts / physical solves |17 /17|15 /15|
| TRAIN row exposure |204|180|
| full-logit API calls / rows |175 /296|126 /225|
| typed calls / local dispatches |162 /750|115 /655|

Two additional legacy global-softmax reference calls/2rows per platform occur
in the initial/final pure underflow fixture, outside the full-logit API counter.
Residual/validation matrix work is inside each fit, not claimed as forward API
calls/FLOPs. No model initialization/random draws, neural updates, physics or
scientific raw reads. RNG fault test changes/restores its own injected state,
not the captured qualification result. All original perplatform4call120s512MiB/
24fit288row/300read600row/500typed1000dispatch limits held, no budget revision.

Five-file Ruff/format/default+Linux-target mypy PASS; whitespace PASS with
inherited CRLF advisories. Exact scoped commands are in the usage document.
Broad core/native/simulator/release suites were not repeated: old fit/default
source is unchanged apart from the explicit new port/branch, existing accepted
evidence reused. No install/CI/clean clone/GPU/family cohort/release qualification.
The279wheel predates this option; do not claim current installed support.

## Sole scientific attempt: FAILED / SPENT; outcomes UNKNOWN

[Frozen protocol](../../artifacts/runs/context-linear-reference-20261029/PROTOCOL.md),
GO[6496840786](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/280#issuecomment-6496840786),
failure[6496869008](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/280#issuecomment-6496869008).
Exact parent command:

```text
wsl.exe -e bash -c 'cd /mnt/d/Dev/Repositories/Circadian && TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /usr/bin/python3.12 -B artifacts/runs/context-linear-reference-20261029/execute.py'
```

EXIT1/observed shell3.0370151s including WSL launch. `execute.py:16` fails in
`check_sources()` with FileNotFoundError for the protocol's key containing
`artifacts\runs\context-linear-reference-20261029\PROTOCOL.md`. The Windows
producer serialized its last3source-list paths with `str(Path)`; Linux treats
those backslashes literally. Failure occurs before the timer/attempt marker,
Popen/meter/qualifier/teacher/model/data access. Genuine fits/rows/solves/logit
reads/tasks/dispatches **0**. `attempt.json`, `execution.json` and raw output do
not exist; case directory is empty. All study fit/task metrics are **UNKNOWN**,
not zero success, not an LS negative, and not inferred from software fixtures.

The protocol makes any failure spend the case; it is not retried or renamed.
Original execute/meter/protocol/pins remain unchanged. Separate
`preflight-failure.json` records the observed failure. A pure string/path check
documents `Path.as_posix()` as an **unapplied proposal** in
`proposed-path-normalization.json`; no frozen producer/evidence was repaired,
no numerical work/new allowance/late projection occurred. Source pins may be
checked on Windows without interpreting them as a successful Linux preflight.
Older275/276/278negative/SPENT raw captures remain closed and unchanged.

## Open criteria and exact next implementation action

Software route is qualified, but280scientific criterion stays unchecked with
this FAILED/SPENT record. Do not reconstruct the archive, reopen the spent
experiment or start another packaging/capture/path-audit sequel. Native H3 fit,
joint LS usefulness, chronological retention/recovery and broad release remain
unproved. No circadian, capacity-cause, generalization or efficiency claim.

Next substantive implementation: bind the existing fixed LS core as an owned
ordinary array learner for `AttributionStream` (fork/snapshot/restore/train_batch/
predict), fitting only the current arrived TRAIN packet. Retain coefficients
and counters, not an unlimited success/row chain. Start two delayed-update
engineering tests for full source/actor isolation, immutable one-solve costs and
failed-call atomicity. This supplies a causal conventional learner control,
instead of repeating a joint static study. No new scientific allowance is
implied; preserve all fixed BP/PC/CPC controls and280's unanswered question.
