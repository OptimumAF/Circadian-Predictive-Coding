# Native snapshot ownership with fewer physical copies

R7.1-TW.SNAPSHOT / #347, parents #18/#21/#23/#31 remain open. Previous #346
was progress: accepted scalar equality repair; #344 remains FAILED/SPENT with
useful F16 old4/fresh2 evidence and UNKNOWN complete return. No archived native
model/task/fit was reconstructed or scientific allowance renewed here.

## Delivered behavior

Existing `torch_feature_learner._state` now passes local detached BP/PC parameter,
gradient and PC traffic views directly into the existing snapshot constructor.
That constructor still validates all fields and deepcopies head/optimizer/
gradient state before returning. Previously these values were cloned once before
that owning copy. Public snapshots still own separate storage; no borrowed view
is returned. The CPC head snapshot branch is unchanged. There is no new state
format, public interface, cache, runner or validation framework.

Source inspection of the actual CE path resolved the previous next action:
`_fit_output_arrays` invokes require_data three times per closure. The fit boundary
checks its private owner before/after its external callback. CEProbeWork wraps
the protected application callback with two requires; the application callback
itself checks source, clocks and input ownership. Ordinary initialized-probe
requires also check prepared source and paid initialization. These checks surround
callbacks that can change state. They remain present; source cardinalities are
not observed whole-native execution counts or measured dominant cost. The repair
removes demonstrated duplicate data copying, not a validation boundary.

## Executed checks and engineering measurement

Sixteen focused cases passed on each platform, no skips:

- All four BP/PC/neutral CPC/active CPC snapshots match actual parameter values
  and own independent storage. Populated BP gradients/momentum and PC traffic
  have independent storage; every returned tensor/policy may be corrupted without
  changing the source or an older snapshot. Live parameter mutation cannot alter
  an already returned snapshot.
- Malformed live parameter, gradient, momentum, optimizer policy and head metadata
  are refused without restoration. The original finite parameter/gradient/momentum
  mutation guard still refuses foreign changes and leaves them observable.

Test-only fixture cleanup occurs after evidence assertions; production restores
nothing. Four fresh owners/platform (cold PC/CPC; BP zero-populated gradients/
momentum with synthetic12/72 counters) perform no training/forward/backward/step/
fit/task/final/physics/GPU/remote work. Those counters are schema fixtures, not
executed SGD or exposure. Model maxima are source-declared, not instrumented
correctness counters:<=4owners/8headconstructor entries/8normaldraw180values per
platform; <=256snapshots/50000tensorcopies5Mlogicalcells. One fresh BP measurement
owner adds2heads/2draw45values. Aggregate<=9owners/18head entries/18draw405values;
none is new scientific credit. All previous failed/spent costs remain preserved.

Windows checks EXIT0/3.185384100s/323710976B sampled RSS;
producer Linux checks EXIT0/7.432551211s/334639104B. Both below20s/512MiB per
body, external shell each below30s. No failed tests/check attempts this increment.
Existing mathematical/learner, baseline, broad, clean-clone and release suites
not repeated: arithmetic/update/role/clock paths unchanged; these scoped ownership
checks address the actual changed behavior. Prior #341/#342/#343 and #346 equality
evidence reused where relevant source/assumptions remain unchanged.

One producer measurement using the unchanged corrected #346 measurement script
and fixed fresh fixture/trials. Prior #346 AFTER is the saved component reference,
not a rerun. Its comparison code/source schema/seed17 dimensions and measurement
procedure are unchanged; only the native snapshot copy path changed. All 30
individual samples/component and one five-top-level-guard cProfile retained.
No favorable timing/seed selection, hidden warmup or sweep.

| Component, all-30-sample mean | Prior #346 AFTER ms | Current ms |
| --- | ---: | ---: |
| Complete native snapshot | 0.249756 | 0.233564 |
| RNG capture | 0.026020 | 0.026177 |
| RNG comparison | 0.057647 | 0.054125 |
| Complete owner guard | 0.406019 | 0.367371 |

Observed snapshot mean fell6.5% and guard9.5%, but these are one historical and
one current engineering observation, not replicated confidence bounds or native
task latency. Unmodified RNG components vary too. Do not infer a full-runtime
speedup, original30s admission, complete return/recovery or scientific advantage.
Profiled and unprofiled times are separate; nested cumulative times are not
disjoint. Raw current guard call/reccall6/1 (five nonrecursive) retained, no
invented execution count; equality still3750total/3735recursive calls.

Deterministic monitored copy counts supply the strongest cost evidence:
Tensor.clone/deepcopy calls1384→836, logical copied cells18684→11286, reductions
548calls/7398cells (~39.6%). Snapshot69/RNG70/root-comparison141/top-copy1 counts
stay unchanged, as do1owner/2heads/2normaldraw45values. No check was skipped.
Counts cover explicit clone/deepcopy traffic, not all allocations or complete
FLOPs. Tensor equality/read/schema validations remain paid time.

The prospective prediction was832copies/11232cells. It was wrong by4copies/
54cells: of69snapshots, the constructor-cold snapshot has no gradients and saves
four clones, while68populated snapshots save eight. Thus548, not552, calls are
removed. Frozen prediction is retained in PROTOCOL.md, with this explanation;
no rerun or threshold change to force agreement. This is not a failed ownership
criterion or a hidden scientific effect.

Sole measurement EXIT0/2.822931994s/322977792B sampled RSS, external4.7188353s,
budget_verifiedtrue, below20s/768MiB and30s external. Timing starts after stdlib/
SPENT preparation, includes Torch import, model setup, all operations and sampler
shutdown, excludes interpreter startup and final serialization/write. External
shell timing includes WSL/startup/output. RSS sampled, not OS-enforced hard cap.
No native344 retry/resume/clock extension/refund and no old baseline rerun.

## Exact commands and limits

Physical cwd D:/Dev/Repositories/Circadian. Windows TEMP/TMP=D:/Dev/_Temp;
WSL TEMP/TMP/TMPDIR=/mnt/d/Dev/_Temp, CPU threads1. Existing producer
Python3.12.3/NumPy1.26.4/Torch2.14CPU, no install/global setting change.

```text
.venv/Scripts/python.exe -B artifacts/runs/native-snapshot-copy-20261030/check.py windows
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp timeout --signal=KILL 20s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/native-snapshot-copy-20261030/check.py linux
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp timeout --signal=KILL 20s /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/native-snapshot-copy-20261030/measure.py after
.venv/Scripts/python.exe -B -m ruff format --check src/adapters/torch_feature_learner.py tests/test_torch_snapshot_ownership.py artifacts/runs/native-snapshot-copy-20261030/check.py artifacts/runs/native-snapshot-copy-20261030/measure.py
.venv/Scripts/python.exe -B -m ruff check src/adapters/torch_feature_learner.py tests/test_torch_snapshot_ownership.py artifacts/runs/native-snapshot-copy-20261030/check.py artifacts/runs/native-snapshot-copy-20261030/measure.py
.venv/Scripts/python.exe -B -m mypy --platform linux --explicit-package-bases --follow-imports=silent --cache-dir D:/Dev/_Temp/circadian347-mypy src/adapters/torch_feature_learner.py tests/test_torch_snapshot_ownership.py artifacts/runs/native-snapshot-copy-20261030/check.py artifacts/runs/native-snapshot-copy-20261030/measure.py
```

Final four-file scoped Ruff format/check/Linux-target mypy PASS, no diagnostics;
initial checks3.2692315s plus final recorded time, below90s engineering static
limit. Scoped cache64MiB and stage2MiB gates held. Exact log/JSON/cost receipts,
exclusive SPENT markers, raw samples/profile, initial production preimage and
protocol retained in artifacts/runs/native-snapshot-copy-20261030/. The larger
HEAD diff contains inherited unrelated work; only the local three clone sites,
comment and formatting are this repair. No dependencies/installs/deletes/commits/
pushes or unrelated source edits.

## Acceptance boundary and next action

Combined source/procedure/result self-review complete. Independent actual
patch/procedure/result review ACCEPT: four read-only commands total0.8513512s,
no imports/tests/models/numerical reruns/network/children. First discovery read
EXIT1 from two guessed missing filenames is retained; remaining reads use actual
paths and support the scoped verdict. Exact receipts/verdict in stage/REVIEW.md;
finite exception ended. Full unchanged CE caller/cardinality and GitHub chronology
were not independently retraced within that cap; those remain lead source/write
evidence. This review is not numerical replication.
Public ownership/refusal and monitored copy savings are scoped software evidence;
full task usefulness, return/recovery, matched science and release remain open.

The native runner and learnability/fresh-task machinery already exist; do not
implement another runner, provenance layer or automatic third component timing
increment. The next substantial outcome is end-to-end correctness qualification
of the repaired ordinary CE path under the original clocks. Before any new model
work, freeze one explicitly justified post-repair engineering protocol with
separate complete cost accounting, original tasks/roles/seed/metrics/guards,
honest repeat-data status and a fixed failure policy. Do not renew, resume or
relabel the spent #344 scientific attempt or claim an engineering retest is
independent scientific evidence. No new model allowance has been executed here.

Evidence receipts for R7.1-TW.SNAPSHOT: [canonical closeout](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/347#issuecomment-6537968123); [parent #18](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/18#issuecomment-6537972867), [parent #21](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/21#issuecomment-6537973232), [parent #23](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/23#issuecomment-6537973566), [parent #31](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/31#issuecomment-6537973872).
Machine-readable receipt: artifacts/runs/native-snapshot-copy-20261030/closeout.json.
