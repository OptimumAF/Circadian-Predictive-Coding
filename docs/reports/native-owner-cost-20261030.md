# Native owner/RNG validation cost and exact-scalar repair

R7.1-TW.NATIVE.COST / #346; parents #18/#21/#23/#31 remain open.
This follows #345's accepted negative scalar-receipt diagnosis. #344 remains
FAILED/SPENT on its original continuous source clock; no archived native model,
task, training, final evaluation or failed allowance was replayed or renewed.

The production change is four lines in `src/adapters/torch_decision.py`:
after refusing unequal exact types, compare exact builtin int/float/bool/str/
bytes/None values directly. Recursive traversal, custom/subclass fallback,
tensor type/dtype/device/shape/value checks and every caller/boundary stay
unchanged. Why: comparing hundreds of RNG integers repeatedly resolved Torch
even though those values have ordinary exact Python equality. No validity is
cached, and no check or source/receipt/counter/role/clock authority is removed.

## Observed engineering measurements

| Component, mean of all 30 samples | Before ms | After ms |
| --- | ---: | ---: |
| Complete native snapshot | 0.272321 | 0.249756 |
| RNG capture | 0.028501 | 0.026020 |
| Exact RNG comparison | 0.209717 | 0.057647 |
| Full unchanged-owner guard | 0.615996 | 0.406019 |

The observed guard mean fell 34.1%; RNG comparison fell 72.5%. These are one
fixed before/after engineering observation, not replicated confidence bounds,
whole-native speedup, proof of dominant native cost or completed return behavior.
Order is fixed before then after; no seed/trial selection or sweep. Unchanged
snapshot/RNG capture timings also vary, so do not attribute all observed elapsed
differences to the patch. Raw individual samples and all retained profile rows
are in `artifacts/runs/native-owner-cost-20261030/`.

Both successful timing invocations use the same fresh engineering fixture:
one BP D9/H3/K6 seed17 owner (its existing initializer creates a PC donor),
then explicit zero gradient/momentum buffers and synthetic payload counters12/72.
Those counters exercise populated schema; they are NOT performed SGD updates,
training exposure, a reconstruction of #344 or an accepted scientific source.
No task labels/data are loaded. Native `_state`, `_rng`, `_same_values` and
`_unchanged_owner` are the existing actual implementations. Thirty individual
timings/component plus one cProfile sample of five top-level guard calls, all
retained, no hidden warmup/best sample. Wrapper/timer/profiler overhead is present;
profile and unprofiled durations are different scopes.

Before profile:3670 require_torch/import_module calls, require_torch cumulative
0.004725654s versus guard0.012933004s; raw call/reccall counts are retained
(the guard profile reports7/2, five non-recursive calls). Recursive equality
had3750 total/3735 recursive calls. After equality still has3750/3735: the
entire traversal remains. The frozen guard>=0.25ms investigation trigger was
met; this justified investigating the scalar lookup cost, not omitting checks.
Do not sum nested cumulative profile times as disjoint costs.

Per successful timing:1owner/2heads/2normaldraws45values/69snapshots/70RNGcaptures/
141top-levelcomparisons/1topcopy/1384Tensor.clone-or-deepcopy calls18684logical
cells. Counts are identical before/after. Recursive leaves are not extra
top-level calls; captured profile includes them. Clone/deepcopy counts do not
cover all allocator traffic or full FLOPs. Genuine train/forward/backward/step/
task/final/physics/GPU/remote work ZERO. Refusal hooks are installed after the
ordinary constructor (which performs no train/forward work), before measurement.

## Commands, costs and preserved failures

Physical working directory D:/Dev/Repositories/Circadian; Windows TEMP/TMP
D:/Dev/_Temp, WSL TEMP/TMP/TMPDIR=/mnt/d/Dev/_Temp. Existing offline producer
Python3.12.3/NumPy1.26.4/Torch2.14CPU, threads1, no installs/global changes.

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp timeout --signal=KILL 20s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/native-owner-cost-20261030/measure.py before-fixed
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp timeout --signal=KILL 20s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/native-owner-cost-20261030/measure.py after
.venv/Scripts/python.exe -B artifacts/runs/native-owner-cost-20261030/check.py windows
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp timeout --signal=KILL 20s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/native-owner-cost-20261030/check.py linux-fixed
```

Successful BEFORE EXIT0/3.146760839s/322969600B sampled RSS/external4.9297921s;
AFTER EXIT0/3.071681105s/323104768B/external5.2510542s. Each below20s/768MiB,
external30s. Timing scope begins after SPENT/setup stdlib, includes Torch import,
fixture/model setup, all sampling and final sampler shutdown; excludes final
JSON serialization/write and interpreter startup. External shell time includes
WSL/startup and output. RSS sampled, not OS-enforced hard allocation cap.

Original BEFORE failed before any timing/snapshot/RNG capture: shared refusal
hooks installed before Torch's lazy optimizer import caused a duplicate-object
rule assertion. EXIT1/3.227721659s/319848448B/external6.5633914s, one BP attempt/
PC donor/2draw45values/4clones54cells. `before.json`, SPENT/log/shell and
`measure-failed.txt` preserved. A prospective engineering amendment moved hooks
after construction and allowed exactly one corrected BEFORE. Original failed
measurement was not relabeled or retried as scientific work. Three timing
attempts total including failure:9.446163603s internal/16.7442377s external,
3BP attempts/6heads/6draw135values, below amended60/90s ceilings. No fourth
attempt, no scientific retry/refund or #344clock extension.

Windows26/26 focused checks passed, no skips:3.280025000s/323096576B. Corrected
Linux26/26 passed, no skips:7.604448815s/334475264B, each below20s/512MiB.
Checks cover nested scalar tail corruption, exact container/scalar types,
subclass fallback equality, NaN, tensor value/dtype/shape/device mismatch, and
actual source parameter/gradient/momentum/payload/optimizer-binding and all
three global RNG mutation refusals. Test-only cleanup runs AFTER verifying
foreign changes remain observable; production never restores them.

First Linux check attempt (`check.py linux`) failed in pytest temporary-file
capture cleanup, EXIT1/external7.8112222s. Original log/shell/SPENT preserved;
no completed result JSON, fixture execution counts/RSS/internal cost UNKNOWN.
Conservatively carry its full allowance of2BP owners/4heads/4draw90values. A
prospective one-attempt engineering amendment disabled pytest capture with-s;
no test/acceptance weakening. Passing platform checks use2BP owners each,
4initializerheads/4draw90values each. Total timing plus correctness work bounded
by9BP attempts/18heads/18normaldraw405values including failed unknown allowance.
These are engineering fixture costs, not benchmark exposure or scientific credit.
The check driver records source-declared fixture maxima, not instrumented model
counters. No unobserved failure counts are replaced with zeros.

Initial unused-import/method-assignment diagnostics, four test annotation
diagnostics and three check-driver typing diagnostics are retained in static
history. Final four touched Python files pass Ruff format/check and Linux-target
mypy; driver-only final repair stores elapsed seconds in its existing local
variable, no production/test-source change after passing checks. No reason to
rerun native fixtures merely for that static annotation correction.

```text
.venv/Scripts/python.exe -B -m ruff format --check tests/test_torch_exact_values.py src/adapters/torch_decision.py artifacts/runs/native-owner-cost-20261030/measure.py artifacts/runs/native-owner-cost-20261030/check.py
.venv/Scripts/python.exe -B -m ruff check tests/test_torch_exact_values.py src/adapters/torch_decision.py artifacts/runs/native-owner-cost-20261030/measure.py artifacts/runs/native-owner-cost-20261030/check.py
.venv/Scripts/python.exe -B -m mypy --platform linux --explicit-package-bases --follow-imports=silent --cache-dir D:/Dev/_Temp/circadian346-mypy tests/test_torch_exact_values.py src/adapters/torch_decision.py artifacts/runs/native-owner-cost-20261030/measure.py artifacts/runs/native-owner-cost-20261030/check.py
```

Stage below2MiB/cache16302309B below64MiB. No dependency/install/delete/Git
mutation or unrelated edits. Baseline, mathematical, broad/clean-clone and
release suites skipped: no learner math, training, task, evaluation or stream
algorithm changed. Existing #341/#342/#343 accepted evidence reused at unchanged
source; #344 failure and all earlier negative results retained.

## Review and exact next action

Combined source/procedure/results self-review complete. Independent review
ACCEPT of actual patch/fixtures/timing/results and scoped interpretation:
four read-only commands, all EXIT0, total0.8370851s, no imports/tests/models/
numerical replication/network/children. Exact receipts/verdict retained in
stage/REVIEW.md; finite exception ended. The reviewer did not independently
fetch issue amendment chronology; lead owns those verified write receipts.
Scoped correctness and lower engineering validation cost do not establish
scientific benefit, reliable adaptation, source30s admission or release readiness.

Next product dependency: inspect the actual native CE inner-boundary call
multiplicity and compare remaining cost against the existing fitted task work.
Do not launch another task cohort, increase the original clock, omit/cached-pass
checks or add a new audit framework. Before any new scientific observation,
freeze a separately justified useful protocol with original roles/receipt/
consent/counter/RNG/clock gates; failed #344 cannot be resumed or renamed.

Evidence receipts for R7.1-TW.NATIVE.COST: [canonical closeout](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/346#issuecomment-6537848069); [parent #18](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/18#issuecomment-6537852050), [parent #21](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/21#issuecomment-6537852229), [parent #23](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/23#issuecomment-6537852383), [parent #31](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/31#issuecomment-6537852617).
Machine-readable receipt: artifacts/runs/native-owner-cost-20261030/closeout.json.
