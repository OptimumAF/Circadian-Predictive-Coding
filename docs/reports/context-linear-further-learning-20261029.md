# R2.1-CTX.LS.P.X — further learning and recovery, with within-context forgetting

[Issue285](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/285).
ONE frozen known-template development observation COMPLETE/SPENT. Full goal and
parents18/21/23/33/reliable joint adaptation/science/release remain OPEN. This is
private probe behavior; source, benchmark and deployed learners are not updated.
All old negative/SPENT studies remain intact, including280UNKNOWN and283rawCLOSED.

## Observed usefulness and limitation

| Task | Initial source | After legacy probe | After old-TRAIN return |
|---|---|---|---|
| v1 invoice |success|wrong workflow|success|
| v1 password |success|wrong workflow|success|
| v2 invoice |success|wrong workflow|success|
| v2 parcel |success|success|success|
| v1 legacy |wrong workflow|success|wrong workflow|
| v2 legacy |success|success|success|

All18episodes complete. Each uses three actual typed decisions; failures terminate
`wrong_workflow`, rather than missing/incomplete observations. The initial six-task
reading matters: v1legacy is newly learned (gain+1), v2legacy already succeeds
(gain0/inherited ability). Three initially acquired old tasks are lost after
private legacy learning and all three observably recover on old-TRAIN return.
Parcel is retained, so its recovery is null. The newly acquired v1legacy task is
then lost on return. This demonstrates bounded further learning and actual old-task
recovery, with incomplete retention. It does not solve reliable joint retention.

Old-TRAIN full-class row fit is12/12initial→9/12afterprobe→12/12return; row fit is
not task success (old tasks4/4→1/4→4/4). Legacy v1 row fit2/3→3/3→2/3; v2 stays3/3.
Probe v1CE1.347560668→1.043591778, v2CE1.226348547→1.043591778. Old-TRAIN return
reverses those CE gains to their initial values. Curves have original0/4/8/16
points, no outcome-dependent selection/tuning. LS rank4source→3afterlegacy→4return;
native residual stays around10^-30, distinct from CE and not proof of retention.

The working two-head control avoids overwriting an inactive context, as283shows.
Here each context's active head is refitted on legacy packets, and later on old
packets. Code plus this observation identify a concrete within-context overwrite
limitation; one observation does not establish a universal forgetting mechanism.
Known TRAIN templates share workflow stages/labels, not unseen/final data. No
CPC/circadian/plasticity-cause/matched-capacity/totalcompute/method-superiority/
generalization/reliability/deployed-policy or release claim follows.

## Procedure and provenance

[Frozen protocol](../../artifacts/runs/context-linear-further-learning-20261029/PROTOCOL.md),
[GO6499617898](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/285#issuecomment-6499617898),
[outcome6499667417](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/285#issuecomment-6499667417).
Product source is unchanged. Existing281–284math/state/probe/return/guards reused;
no historical archive reconstruction, old raw read, control retrain or spent retry.

NEW source preparation is charged separately: existing contextual generator's
fixed packets0(firstA) and4(firstB), six rows each, through ordinary AttributionStream
training_only/labels+1/checkpoints0/2. Original source budget2updates30s512MiB,
root source and actor remain unfitted. Two actual causal receipts: observed0/
arrived1/applied1 and observed1/arrived2/applied2. The paid checkpoint2 is copied
as data to an owned in-memory probe source; no publication, source-authority
renewal or old283parameter reconstruction. Minimal preparation was fixed before
outcomes because this LS rule replaces each active head's current packet. It is
not283's12update native history or a matched comparison to that experiment.

Existing build_head_cohorts and284run_arm linear mode perform initial tasks,
16alternating3rowprobe_train updates48rows, then16alternating6rowoldTRAIN updates
96rows on the same disposable learner, with original checkpoint/metric/task order.
The two private16update30s512MiB sessions are born once:70.492208943 and70.492212986,
ending79.734029536/79.734292696. Return deadline includes probe time. Source budget
clock remains live70.438650553→79.773512579 with original2charges/limits. All source,
actor, candidate, original inputs, paid source evidence, NumPy/Python RNG and probe
source invariance checks pass. The owned probe source remains2/12, original stream
root/actor remain0/0 and preparation candidate stays2/12; private endpoint
counters2/12→18/60→34/156. No source/benchmark/deployed training, final evaluation,
replay, promotion or feedback from task/metric outcomes into training.

New stage-only `case.py` composes these existing interfaces; `meter.py` observes
physical API work, `execute.py` supplies one child/watchdog, and `producer_checks.py`
checks pure projection/capture/path behavior. No second product training runtime,
optimizer, dependency or source loader was added. Raw232453B written exclusively
once and NEVER opened afterward. Scalar endpoints/CE/accuracy/rank/residual/owned
failure/causal/work/budget metadata were emitted from the same memory before
parent success assertions. `case/actual.json` is CLOSED: use only scalar.json,
work.json and execution.json. No raw reread, late projection, rescoring, repair,
renewal or renamed capture. All29selected POSIX source/procedure pins matched
pre/post. Source artifact content remains frozen.

## Commands, costs and validation

Physical cwd `D:/Dev/Repositories/Circadian`; Windows TEMP/TMP=`D:/Dev/_Temp`,
Linux TEMP/TMP/TMPDIR=`/mnt/d/Dev/_Temp`; Python-B. Existing Linux3.12.3/NumPy1.26.4,
process-local OMP/MKL/OPENBLAS/NUMEXPRthreads1/CPU1requested/GPU0.

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /usr/bin/python3.12 -B artifacts/runs/context-linear-further-learning-20261029/execute.py
```

EXIT0. Parent11.879654229s, instrumented child11.538529054s, producer9.41926366s;
these are nested scopes, not additive. Parent sampled child peak50,462,720B;
child sampler50,466,816B. Same-process metadata.006078246s excludes the final scalar
write; total metadata10s cap checked after that write, no universal hard-memory/
FLOP evidence. Stage332695B before documentation, below8MiB; no large sweep.

| Scientific work, including preparation | Actual |
|---|---:|
| LS calls / physical solves / all exposure rows |34 /34 /156|
| Source TRAIN fits/rows |2 /12|
| Private probe fits/rows |16 /48|
| Private old-TRAIN return fits/rows |16 /96|
| Full-class prediction calls/rows |32 /144|
| Outer + selected typed calls |54 +54|
| Typed affine calls/rows; unfitted calls |54 /54;0|
| TRAIN+typed routing calls/rows |88 /210|
| All context projection calls/rows |120 /354|
| TRAIN teacher traces/steps |16 /48|
| Endpoint episodes/typed dispatches |18 /54|
| Total sandbox dispatches including teachers |102|

All original scientific caps held with no genuine failure or renewal. Logical
current model96float64coeff768B, three endpoint snapshots2304B plus observed source
checkpoint768B; Python/copies/metadata extra. Original generator+cohort arrays12000B;
detached probe arrays/stream/copies extra. Nominal typed affine terms2592, total
typed token visits576(fullquery288+selectedhead288); not all fitting/SVD/residual/
copy/validation/softmax computation.0Torch/neural/gradient/latent/randommodeldraws/
physics/final/replay/oldscientificraw work. No downloads/installs/commit/push/delete/
global changes; unrelated dirty270–284source and all failures remain.

Pure producer checks used no models/teachers/episodes: initial5/platform, final
9/platform after review repairs, no failed/skipped selected tests. Windows two
calls2.9004454s/49,541,120Bpeak; Linux two26.081842851s/47,833,088Bpeak, below original
2calls45s512MiB/20literalcase-runs bound. Commands were:

```text
.venv/Scripts/python.exe -B artifacts/runs/context-linear-further-learning-20261029/producer_checks.py artifacts/runs/context-linear-further-learning-20261029/win-final-pure.json
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-linear-further-learning-20261029/producer_checks.py artifacts/runs/context-linear-further-learning-20261029/linux-final-pure.json
```

Four stage Python files: Ruff check/format --check/default+Linux-target mypy
`--explicit-package-bases --follow-imports=silent` PASS. Whitespace PASS with
inherited CRLF advisories. Initial static annotation errors repaired before gates,
no scientific/model effect. Earlier284scope evidence reused precisely: its final
guard-only repair had9pure cases/platform plus independent source acceptance;
earlier integrations cover unchanged math/data/loop. This actual case now executes
the final combined in-memory API with its complete preparation/probe/return path.

Independent protocol/source review3readonlycalls found missing detached source
limit value pin and lost partial source evidence; also scalar rank omission.
All fixed before GO. Four new pure regressions preserve paid partial receipts/
checkpoints, explicitly unavailable corrupt receipts, frozen-limit mutation refusal,
rank/residual/failed-work projection without coefficients. One readonly repair
follow-up accepted the actual changes/protocol. All finite review exceptions ended;
no imports/tests/models/oldraw/writes by reviewer. Source acceptance is not
independent execution or scientific replication. Lead source/procedure/result
self-review complete. No genuine failure was repaired or rerun.
One postexecution independent emitted-scalar interpretation review found no blocker
in visible endpoints/curves/costs/provenance/claim limits. Its compact view omitted
individual terminal reasons, so `wrong_workflow` and per-episode three-decision
claims were checked by the lead only. One readonly call, no imports/models/tests/
raw reads/writes; finite exception ended. This is interpretation acceptance, not
replication or full source/receipt/release acceptance.

Unchanged full math/baseline/native suites, old raw-backed tests, fresh current
installed wheel/clean clone/remoteCI/GPU/physics/release checks were skipped outside
scope.279wheel predates source changes. No repository-wide readiness claim.

## Remaining outcome and next implementation action

Private further learning and observed old-task recovery are supported; simultaneous
retention of old and newly acquired same-context tasks is not. Plain sequential
replay on a replace-head learner can simply replace the new fit with the old fit.
Existing context_head_rehearsal composes the generic probe/StreamRehearsal and
already has explicit old-TRAIN admission/budgeting; do not build another runner.

Next concrete implementation: extend existing private rehearsal composition to
an opt-in bounded same-context **joint** LS packet, with current/probe and reused
old-TRAIN provenance/reuse authority kept explicit and matched current-only row
exposure available. Begin role/receipt/atomicity tests before any new experiment.
This targets the measured overwrite; it is not more update-count tuning or a
retry of285/280. Do not relabel probe/final data as replay or claim fair computation
from equal rows alone. No next scientific allowance is granted here. Broader
native learning, CPC benefit, generalization, deployment and release remain open.
Live handoff is285/tracker18; historical plans/log preserved perAGENTS.
