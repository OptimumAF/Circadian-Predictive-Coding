# R2.1-CTX.LS.A — owned current-packet learner software qualification

[Issue281](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/281).
Full goal and broad18/21/23/33learning/science/release parents OPEN.280scientific
criterion remains UNKNOWN/FAILED-SPENT, with no retry, capture, repair or renewed
allowance. This increment implements and tests a causal learner component, not
a joint-fit study or scientific comparison. Historical plans/log remain
preserved per AGENTS; GitHub holds live scope/handoff.

## Delivered behavior

New `src/core/categorical_linear_state.py` holds exact immutable policy/snapshot
records, bounded geometry/counters, explicit model=None initial state and pure
prediction. No fake fit lineage or retained row bank. New
`src/adapters/linear_fit_learner.py` supplies the existing learner ports with
owned fork/snapshot/restore, uniform initial probabilities, current-packet fixed
LS replacement, binary/categorical semantics and a distinct squared-residual
diagnostic. Existing core LS math, typed decision port, stream and other
learners are unchanged.

One staged solve validates arrays and counter limits, owns inputs, then verifies
original policy/state identity+values, caller/staged arrays and global RNG before
one complete commit. A failed candidate never replaces the model; foreign
mutations are refused rather than rolled back. No optimizer, new dependency,
scheduler/controller, unlimited success/row history or budget-restoration claim.
Exclusive execution belongs to the existing runtime lease.

Independent STATE/math review used3readonly calls and found that plain exceptions
left paid solver work observable only through fixture instrumentation. Reused
the existing `TrainingUpdateFailed` to carry owned scalar fit-attempt/row/return/
unknown/unpublished/candidate-commit0 receipts, with original cause. Pre-fit
refusals remain0solver; unchanged committed counts do not erase physical costs.
Ordinary stream failure exposes this receipt, stops and refuses another solve.
One-read followup resolved the finding, no further blocker. Finite exceptions
ENDED/0imports/tests/models/raw/artifact reads/writes. Lead reviewed combined
source, procedure and results. Source review is not independent numerical
replication or task-learning acceptance.

## Checks, commands and measured work

27distinct focused cases/platform PASS: explicit initial unfitted semantics,
current-packet replacement/argmax behavior, owned snapshots/forks/outputs,
binary B×1 versus categorical B×K, invalid arrays/classes/snapshots/policies/
counter overflow before solver, lost return and foreign source/policy/state/
input/staged/RNG changes, actual two-update delayed stream/source actor isolation,
role refusal, failed-work receipt ownership/stop and original D9/K6 TRAIN data.

Actual small3class stream verifies `observed+1==arrived<=applied` at delegation,
two committed updates/6rows, unchanged seeded source and actor,0score forwards.
D9/K6 uses two original TRAIN packets,2updates12rows/last-fit6, original source
and actor unchanged/0score forwards. No task-success readout is inferred from
these two data packets. The current-packet algorithm does not promise retention.

Physical cwd `D:/Dev/Repositories/Circadian`. Windows TEMP/TMP=`D:/Dev/_Temp`;
Linux TEMP/TMP/TMPDIR=`/mnt/d/Dev/_Temp`, `-B`. Evidence stage
`artifacts/runs/current-linear-learner-20261029/`, per-invocation JSON receipts.
Windows initial26PASS1.051714s/68,227,072B/13LS38rows9predict24rows; separate
D9casePASS1.1096281s/68,349,952B/2LS12rows. Linux initial27PASS5.486488219s/
71,024,640B/15LS50rows9predict24rows. Three-file statics PASS.

On the final tree,8affected failure/receipt cases passed1.1255057sWindows and
5.35078026sLinux, each8paid LS24rows. Then3positive current-packet/binary/
delayed-stream cases passed1.1772713sWindows and5.303326677sLinux, each5LS14rows/
7predict18rows. The unchanged role/validation/D9 cases reuse their earlier
accepted source/math evidence; changed exception/accounting paths and normal
commit paths were executed on the final combined tree.

Final Windows affected-command examples:

```text
.venv/Scripts/python.exe -B artifacts/runs/current-linear-learner-20261029/check.py artifacts/runs/current-linear-learner-20261029/win-failure-work.json tests/test_linear_fit_learner.py::test_should_refuse_lost_or_foreign_mutated_fit_without_erasing_changes tests/test_linear_fit_learner.py::test_should_stop_existing_stream_after_lost_fit_and_preserve_attempt_cost --basetemp D:/Dev/_Temp/current-linear-win-3
.venv/Scripts/python.exe -B artifacts/runs/current-linear-learner-20261029/check.py artifacts/runs/current-linear-learner-20261029/win-positive-work.json tests/test_linear_fit_learner.py::test_should_replace_only_current_packet_and_own_forks_snapshots_outputs tests/test_linear_fit_learner.py::test_should_run_two_delayed_train_updates_in_existing_stream_and_preserve_owners tests/test_linear_fit_learner.py::test_should_follow_existing_binary_probability_contract --basetemp D:/Dev/_Temp/current-linear-win-4
```

Linux uses the existing interpreter
`/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B`
with the same observer/test nodes, `linux-failure-work.json`/`linux-positive-work.json`,
`--capture=sys` and basetemps `/mnt/d/Dev/_Temp/current-linear-linux-2`/`-3`, from
physical repo via WSL. Initial/D9/full command argv retained in each work receipt.

| Software total, all attempts/repeats | Windows | Linux |
|---|---:|---:|
| pytest invocations |4|3|
| body seconds |4.4641191|16.140595156|
| sampled peak RSS bytes |68,349,952|71,024,640|
| fit attempts / physical LS calls |28 /28|28 /28|
| TRAIN row exposure |88|88|
| learner prediction calls / rows |16 /42|16 /42|

Each total includes16paid unpublished fits (4lost public returns) and12successful
commits. Residual/validation/copy arithmetic is inside these fits, not claimed as
prediction API calls, exhaustive allocations or FLOPs. No random model init/draws,
Torch imports, neural SGD, PC, probe/final/replay, physics/GPU or scientific raw
read. The literal RNG fault changes/restores its own test state, not captured
evidence or learned outcomes. No failed tests/skips or model/time/RSS budget
revision; all original4call120s512MiB/64fit256row/200predict400row limits held.

Cost correction: the D9fixture's existing teacher factory also generates8TRAIN
traces/24local sandbox dispatches per platform. This was omitted from the first
fit/prediction observer and is declared from the unchanged successful factory,
not separately profiled. It is not zero total sandbox interaction. No served
task-readout episode was executed. Issue6497167584 records the omission and a
prospective cumulative teacher limit16traces48dispatches/platform, including
the already consumed8/24; no further teacher generation occurred. Original
model caps stayed unchanged. No D9rerun just to repair accounting.

Scoped Ruff/format/default+Linux-target mypy PASS for the3new source/test files,
whitespace PASS with inherited CRLF advisories. No scientific case/protocol/GO,
no new seed/configuration search or method comparison. Broad tests/CI/clean clone,
installed wheel, task acquisition/retention/recovery and release qualification
skipped. Existing accepted core/280readout mathematics and all previous controls/
negatives reused, not recertified wholesale.279wheel predates this source API.
No install/deletion/global change/commit/push; inherited dirty270–280 work and
frozen failed captures remain intact.

## One precise next implementation action

Add explicit `method="linear_fit"` to the existing recurring context-return
recipe using this owned learner, initial unfitted typed readout and existing
snapshot/task guards. Keep BP/PC/neutral/active and perceptron defaults unchanged;
separate-head/private/paired modes remain their existing scopes. Preserve native
failure receipts in the runner, report actual fit/solver/row/readout costs with
the LS objective and coefficient storage, and begin role/delay/default/paid-
failure checks. Freeze any later chronological development case separately;
it must not retry280joint static study or imply allowance now. Full useful
learning/scientific/release parents remain OPEN.
