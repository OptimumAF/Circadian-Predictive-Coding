# Original train23 fits exactly; native reproduction remains open

[Canonical issue evidence and handoff](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/404#issuecomment-6560422847).

#404/R2.1-MEM.TRAIN delivers a separate evaluation-free nearest training probe
through the existing delayed-label runtime and one executed diagnosis. All18
original23demonstration rows fit exactly after one private append:

| Training readout | Normalized halfMSE | Selection |
| --- | ---: | --- |
|source7updates/220rows|.02021162526009419|source A indices0..14|
|private8updates/238rows|0|220..237, corresponding to probe rows0..17|

All four action axes have zero after-error. Training-fit gain+.02021162526009419.
No exact duplicate input pairs were found within these18rows or between them and
the220-row source, hence no exact raw/native32 label conflicts. This rules out
failed fitting and exact-label conflicts on these observed examples. It does NOT
identify the cause of #403native24failure: near-conflicts and off-demonstration
behavior remain untested, as does native23 learned-policy success. All previous
negative outcomes are retained; no model/dose/seed/threshold tuning follows here.

## Delivered behavior and boundaries

`src/adapters/continuous_probe.py` adds `run_nearest_training_probe` and a private
single-packet plan, routed through SAME `_run_single_probe`/ActorShadowRuntime/
delayed inbox. It accepts1..32same-spec probe_train D12rows and an exact nonempty
fixed-input nearest TRAIN source; evaluationNone, exactly1private append, no
gradient/evaluation/native/teacher/promotion/candidate release. Ordinary/final/test
roles, fitted/empty/over-capacity sources refuse. Existing pair/neural/default
entrypoints and update semantics remain unchanged.

`src/adapters/continuous_nearest_probe.py` composes unchanged nearest/residual/
native32 consistency kernels. Source chunks plus packet stay<=100rows; retain
each relevant exact pair once, raw/native deltas/conflicts, earliest selection and
source/probe provenance. Complete nested policy/spec/array/counter identities and
paid consistency/score/work/receipt/budget/clock/source/runtime proofs guard the
new path. Lost ACKs keep partial known and unknown work. No automatic data append
occurs during readouts. The explicit probe call alone admits its one private update.

`scripts/push_cube_reserved_probe_inputs.py` extracts the existing per-role byte,
decoded-value, identity and history qualifier. New `load_reserved_training_input`
opens original23 only; original two-packet API/defaultdiagnosticsTrue stays intact.
Why this: old nearest probes require evaluation input and reject gradient options;
neither could answer a strictly train-only fitting/conflict question.

## Executed checks and resources

Physical D:\Dev\Repositories\Circadian HEADd9119e80926a139a2f38460564d7eab1b6d2a2b0,
inherited dirty work retained. Stage `artifacts/runs/nearest-training-diagnostic-20261101/`
holds both frozen protocols, producer/host/21pins, failed/passing checks/SPENT,
actual/host/resources/cached scalars and reviews. [Usage](../nearest-training-diagnostic.md).

```powershell
.venv\Scripts\python.exe -I -B artifacts/runs/nearest-training-diagnostic-20261101/check.py initial
.venv\Scripts\python.exe -I -B artifacts/runs/nearest-training-diagnostic-20261101/check.py affected -k 'changed_source or original_training_role or pair_loader_defaults or equal_nonzero or consistency_work'
.venv\Scripts\python.exe -I -B artifacts/runs/nearest-training-diagnostic-20261101/check.py history -k 'original_training_role or pair_loader_defaults'
.venv\Scripts\python.exe -I -B artifacts/runs/nearest-training-diagnostic-20261101/check.py final
.venv\Scripts\python.exe -m ruff check --no-cache src/adapters/continuous_probe.py src/adapters/continuous_nearest_probe.py scripts/push_cube_reserved_probe_inputs.py tests/test_nearest_training_probe.py artifacts/runs/nearest-training-diagnostic-20261101/study.py
.venv\Scripts\python.exe -m ruff format --check src/adapters/continuous_probe.py src/adapters/continuous_nearest_probe.py scripts/push_cube_reserved_probe_inputs.py tests/test_nearest_training_probe.py artifacts/runs/nearest-training-diagnostic-20261101/study.py
.venv\Scripts\python.exe -m mypy --no-incremental --follow-imports=silent src/adapters/continuous_probe.py src/adapters/continuous_nearest_probe.py scripts/push_cube_reserved_probe_inputs.py tests/test_nearest_training_probe.py artifacts/runs/nearest-training-diagnostic-20261101/study.py
.\artifacts\runs\nearest-training-diagnostic-20261101\run-study.ps1
```

Final30combined cases PASS/1.6666388s/71368704B. Initial28=25pass3fail: fixtures
used an invalid grasp flag and corrupted a score too early. Affected8=6pass2fail:
loader fixture still omitted original fifth history-tail component; corrected to
original7+5 layout. History2PASS; final30PASS. No production guard relaxed. Added
equal-nonzero-distance earliest tie and lost-consistency-ACK cases. Cumulative
4commands68executions=63pass5fail/5.9652022s/max72261632B; all original caps held.
46literal nearest owners/8neural dispatch-only owners/18appends22rows/38retrievals
46queries870candidates10440features/50kernels9944pairs1128native32conversions/
1249public snapshots1975168known numeric bytes.0realinitializers/RNG/neuralmath/
recordeddata/native in software checks; four neural cases prove dispatch only.
Neural fixture-owner cap8is SPENT. Earlier400/402numerical evidence reused.

Scoped5file Ruff/format/mypy/hostsyntax/diff PASS. Initial4production annotations,
1fixture policy annotation and1study method-assignment error corrected without
ignores. Patch anchors that failed after formatter changes modified nothing;
simplified to exact text replacements. One guessed search path was missing,
corrected using actual source. All failed tests and costs remain preserved.

Actual command ran ONCE exit0/noexpiry, hostPID59788/8.977026s. Producer
5.221068419s/sampled60432384B; GNU5.35s58544KiBexit0. CachedLinuxPython3.12.3/
NumPy1.26.4/CPU1/BLAS1/Dtemp,0SDK/GPU. Source writer is nondurable, so NEW paid
preparation was necessary:1empty nearest initializer/7appends220rows, original
A15/B50/train20_15/Push18/corrections50/current_corrections50/originalB22_22 order,
only finaltick17/7-220 enters diagnostic. Other12observers0predictions. Private
one18-row delayed append ->8/238 discarded/source7/220 unchanged; original consent,
roles/receipts/budget/clock/paid-prefix/source identity+value proofs retained.

Actual totals8append attempts/returns238rows,2retrieval36queries8244distance
candidates98928features;4kernels sizes18/100/100/74,12754pairs1168native32scalar
conversions. Block work counters are CUMULATIVE; do not sum them.18forks97public
snapshots787known validator returns16163200known numeric copy bytes (public
snapshots included; generic deepcopy/Python/otherarrays additional).0NN/RNG/SGD/
gradient/SVD/coreforward/teacher/simulator/probe_test/final work; forbidden0/no
unknownappend. Logical prototype28160→30464B/privateowner(+2304).

Three original qualified inputs only/11opens-reads-returns2036362B/perfile3/3/5;
no test24file access, score rows probe_train36/probe_test0/train0/evaluationIDnull.
12unaffected original code pins matched before implementation qualification;
three intentional source changes and6additional pins made21, all matching at
four actual boundaries and independent POST. Original workstop40s768MiB/
GNU45TERM+5/50accept1GiB/host65s/stage8MiB/actual2MiB/log512KiB held unchanged.
Inner optional wall/RSS limits remain unconfigured; no false inner-limit claim.
Host end stage209518B/log654B before cached summaries/documentation.

Independent actual source/math/isolation/software/protocol PRE GO and oneNEWactual
POST outcomes/accounting/current21pins ACCEPT,4readonly batches2.6s/0imports/tests/
oldinputs/models/native/replication. Exception ENDED. Lead separately self-reviewed
actual combined source/procedure/results. NEWactual30756B SHA256
1898ddc3c4cff694bd41a8eda1fd7ccaad89c850bd9634c82437ac64d9d1fa1a.
Lead1+peer1raw reads SPENT; use LEAD-SCALARS, no raw reopen or work renewal.
Fullsuite/newLinuxfixtures/cleanclone/wheel/install/browser/device/GPU/remoteCI/
release skipped; actual diagnosis ran Linux. No installs/download/commit/push/
delete/global changes or unrelated dirty edits.

## Remaining useful outcome and next action

#404 scoped fit/conflict diagnosis COMPLETE. Preserve #403negative native24 and
old3retention/#399inner-limit deviation/all oldnegative-spent evidence; full
#18/#21/#23/#33/repository goal OPEN/ACTIVE. Exact demonstration fitting is not
closed-loop competence or a circadian/generalization/retention/recovery claim.

Next #405/R2.1-MEM.REPRO: qualify original training23 learned-policy closed-loop
reproduction before changing a dose/feature/algorithm. Inspect existing frozen
history/reset/role path and shared synchronous observer first. Underlying native
port already admits original probe_train23, while high-level frozen scoring is
24-only and the new training-only wrapper does not forward the shared observer.
Use an existing valid route or implement only that concrete minimal bridge, then
literal role/frozen-owner/control-failure checks and ONE separately frozen paired
23native protocol. No new framework, role conversion, labels/teacher, test24/final,
native retry or favorable seed selection. One task owns the useful reproduction
outcome and necessary code, instead of another infrastructure-only sequence.
