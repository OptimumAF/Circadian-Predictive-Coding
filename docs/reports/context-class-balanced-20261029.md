# R2.1-CTX.BAL — current-batch class-balanced BP, negative qualification

Owning [issue278](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/278).
Full goal and parents18/21/23 remain OPEN. AGENTS makes issues the live handoff;
historical DEVELOPMENT_PLAN, RESEARCH_ROADMAP and development-log are preserved.
No publication, install, commit, push or global setting changed. Inherited dirty
270–277 work and negative/frozen artifacts remain intact.

## Delivered and checked

`scripts/recurring_ticket_workflow_example.py` adds default-off
`class_balanced_bp=True` / `--class-balanced-bp`, restricted before output/data/
models to ordinary BP/context-return/replay none. `ticket_workflow_native.py`
routes that actual delayed TRAIN call through the qualified277array helper.
Weights are current rows/(present classes*count), absent1.0; in this recipe
common classes receive.75 and two rare ready classes1.5 per batch. One update
only; same seed17/H3/K6/lr.03/momentum.9/D9/A4-B4-A4/12updates72rows.

The bounded12-row journal records separate objective, counts/weights, attempts/
returns and prospective source/label witness. Exact TRAIN role/consent/key/
model version/+1arrival are verified before weighting; call arrays must match
the admitted packet. Newly produced full receipts match that witness plus
diagnostic; later receipts must retain their original object and all values.
Previous scalar records retain original list/row identities and values. Failure
stops further work; physical costs and the original paid prefix are preserved.
No unlimited success chain, new runner, state format, dependency or game-port
boundary change. Static diagnostic truth never chooses weights. All-one factory
continuation matches ordinary BP parameters/optimizer/gradients/counters exactly.

Final35focused tests/platform PASS:33new cases plus2existing ordinary BP cases
sharing one source run. Checks cover API/CLI admission and exclusive output,
actual ready source/label/targets, roles/delay, source/actor/RNG/budget preservation,
causal lineage, original/replaced paid records and receipts, lost optimizer
return, unsupported effects, all-one continuation and bounded JSON. This is
scoped passing behavior, not repository-wide readiness or task learning.

Independent source/math review used3readonly calls, found first-receipt witness
and same-valued receipt replacement gaps. Both were repaired with new tests.
One-call followup found the concrete fixes resolved, no further blocker; both
finite exceptions ENDED. No imports/models/tests/artifact reads/writes by reviewer.
Lead reviewed combined source, procedure and results. Source review is not
independent experimental replication or acceptance of a scientific benefit.

## Commands, failed attempt and software costs

Physical cwd `D:/Dev/Repositories/Circadian`; shell TEMP/TMP=`D:/Dev/_Temp`.
Linux TEMP/TMP/TMPDIR=`/mnt/d/Dev/_Temp`; existing Python3.12.3/Torch2.14.0+cpu.
Windows `.venv/Scripts/python.exe`3.14.7/Torch2.14.0+cpu. No installs/cache moves.
Stage `artifacts/runs/context-class-balanced-20261029/` contains finite profiler,
per-invocation JSON work receipts, frozen protocol/source pins and execution summary.

Final Windows command:

```text
.venv/Scripts/python.exe -B artifacts/runs/context-class-balanced-20261029/check.py tests artifacts/runs/context-class-balanced-20261029/win-repaired-work.json tests/test_context_class_balanced.py tests/test_native_context_return.py::test_should_execute_same_contexts_roles_receipts_and_full_native_snapshots[bp] tests/test_native_context_return.py::test_should_count_actual_d9_native_costs_and_bound_output[bp] --basetemp D:/Dev/_Temp/context-balanced-win-3
```

Linux uses the existing interpreter below with the same arguments, substituting
`linux-repaired-work.json`, `--capture=sys` and Linux basetemp
`/mnt/d/Dev/_Temp/context-balanced-linux-2`:

```text
wsl.exe -e bash -c 'cd /mnt/d/Dev/Repositories/Circadian && TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-class-balanced-20261029/check.py tests artifacts/runs/context-class-balanced-20261029/linux-repaired-work.json tests/test_context_class_balanced.py tests/test_native_context_return.py::test_should_execute_same_contexts_roles_receipts_and_full_native_snapshots[bp] tests/test_native_context_return.py::test_should_count_actual_d9_native_costs_and_bound_output[bp] --capture=sys --basetemp /mnt/d/Dev/_Temp/context-balanced-linux-2'
```

Initial27cases/platform passed14.4330957s/19.318411165s. The second Windows
invocation failed after15passing cases at the first update: the newly added
admission check used a Torch comparator that lacked NumPy array support.
Preserved `win-final-work.json`:5.1602839s/peak329,265,152B, one declared attempt
6rows, **zero weighted calls/SGD**. Reused the existing app array/dataclass
comparator; no acceptance/budget weakening. Initial annotation errors were also
fixed. Corrected final35cases pass24.1127096s/28.679462837s.

Totals, including repeats, references and failed attempt:

| Software budget | Windows | Linux |
|---|---:|---:|
| pytest invocations |3|2|
| observed seconds |43.7060892|47.997874002|
| sampled peak RSS bytes |337,035,264|351,850,496|
| physical SGD/backwards / TRAIN rows |52 /312|52 /312|
| weighted API calls / declared rows |36 /216|36 /216|
| component inits / nominal weight initializer slots |134 /6030|127 /5715|
| native forwards / all forward rows |493 /852|471 /819|
| snapshot constructions / logical tensor elements |9163 /884142|8702 /859248|
| typed calls / local dispatches |432 /696|411 /627|

All original4call/180s/1GiB/80step480TRAINrow/250init/etc limits held; no budget
revision.36weighted candidate steps/platform include2intentionally failed private
SGDs;16ordinary steps include references/ordinary BP regression. Software
observations are not a scientific rerun.0PC latent/NumPy fits/physics/GPU/skips.
Nominal slots/logical elements are not exhaustive allocations, RNG instructions,
FLOPs or OS-hard peak memory. Failed/refused native wrapper calls are also counted
in each receipt; they are not silently equated to actual optimizer steps.

Three-file Ruff/check-format and mypy with
`--explicit-package-bases --follow-imports=silent`, default and `--platform linux`,
PASS. `git diff --check` PASS with inherited CRLF advisories. No broad suite/CI,
clean clone, installed wheel, CUDA, physics or other-family retraining performed.
Accepted unchanged base/game/helper evidence reused, not recertified wholesale.

## Sole frozen development case: COMPLETE / SPENT

[Protocol](../../artifacts/runs/context-class-balanced-20261029/PROTOCOL.md),
GO[6496011001](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/278#issuecomment-6496011001).
Exact command `.venv/Scripts/python.exe -B artifacts/runs/context-class-balanced-20261029/execute.py`.
Invokes the actual existing CLI with `--context-return --record-train-fit
--class-balanced-bp --method bp`; all other settings and metric directions frozen.
The output512KiB and original source30s/512MiB/12update budget stayed unchanged.

Exit0/complete:12weighted SGD/backward72TRAINrows,4extra matrix diagnostic calls
48rows,25component inits1125nominal slots,151native forwards255forward rows,
2866snapshots282960logical elements,135typed calls183local dispatches.
No ordinary BP fits/latent/probe/final/replay/benchmark or deployed update.
Parent16.9196295s/body13.724131365s, sampled peak336,224,256B, output429,076B,
metadata.0034008s. Original source clock5.351783589→16.193957803,12committed;
source/actor/NumPy/Torch RNG unchanged. All24task readouts completed,12weight
records matched their original receipts. All13source/test/procedure pins held.

Only emitted scalar summaries from276were reused; its raw capture remains
closed. This run's sole raw read is consumed; `case/native-bp-actual.json` is also
**CLOSED**. No retries, raw rereads, rescoring, repair, late projections or renewed
allowance. Subsequent use is limited to `execution.json` emitted summaries/costs.

| Source updates |0|4|8|12|
|---|---:|---:|---:|---:|
| original unweighted TRAIN mean CE |1.788785106|1.771299674|1.735855024|1.693898731|
| allowed TRAIN correct /12 |6|8|8|8|
| ready correct /4 |0|0|0|0|
| sealed correct /4 |2|4|4|4|
| staged correct /4 |4|4|4|4|
| ordinary task success /4 |0|0|0|0|

Unweighted276has identical initial CE/correctness and final8/12fit; final CE is
1.646444208 versus weighted1.693898731. Some ready target margins are less
negative after weighting, but every ready action and ordinary task still fails.
No task was acquired, so no retention/recovery was established. This does not
prove the cause is imbalance, capacity, optimization or plasticity. Distinct
weighted training diagnostic values are not ranked against unweighted CE/PC
energy. No CPC, statistical, generalization, reliability or release benefit.

## Critical path and one next implementation action

The current TRAIN-objective increment is complete, with a negative result.
Do not start another weight/dose/seed/native-family cohort or checkpoint audit.
The simpler conventional serving control remains the working demonstrated path;
its accepted results are reused. Wider R2.1/useful native learning and release
requirements remain open. The latest qualified wheel274predates the native D9,
TRAIN diagnostic and weighted additions275–278, so source qualification is not
installed functionality.

Next: rebuild a fresh wheel in the existing D: installed-qualification namespace
and verify the new native/weighted imports and API/CLI admission from an isolated
installed environment with no checkout imports. Use bounded engineering fixtures
and preserved prior install evidence; do not repeat the spent scientific task
cohort. Leave the old wheel/environment intact and claim only installed scope.
