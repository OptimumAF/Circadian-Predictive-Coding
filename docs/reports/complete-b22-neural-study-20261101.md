# #399: complete220 fit observed; native qualification partial

Later evidence: the [inference-only Pick continuation](complete-b22-pick-continuation-20261101.md)
completed all three missing cases, each with zero success over50controls. This
original failed capture and its protocol deviation remain unchanged; its
then-unobserved Pick outcomes below describe this original attempt only.

One fixed cold H64 neural fit completed64mixed-SGD updates over all220TRAIN
examples. Both seven-group TRAIN readouts completed; error decreased in every
group. Push17 then completed50controls with zero success. Pick17/19/20 were
**not executed**: the observer falsely rejected the complete Push report because
it required a field absent from that existing report contract.

Overall study remains **FAILED/SPENT**, #399 remains OPEN. The fit and Push
negative are accepted partial evidence, not a complete four-endpoint result or
proof of acquisition/retention/recovery/generalization/CPC benefit. No fit retry,
new seed/dose/weights, native rerun or policy selection occurred.

## Declared question and actual learning evidence

The existing memory/teacher reader supplied its accepted198bank plus original
B22, explicitly remapped into the retained neural order. All seven identities
and weights were preserved: current50/corrections50/push18/a15/b50/train20_15/
b22_22. B22 remains distinct from derivedB50. The model never observes names or
teacher phase; features remain executedTS12. Memory's append order/objective/
exposure differ, so this is shared information, not a matched method comparison.

One D12/H64/seed17 cold initialization used1024normal values. Existing policy
defaults LR.01/momentum.9/no fitted statistics; existing private mixed-gripper
objective and fixed64update ceiling, no search or best-checkpoint selection.
Before diagnostics used an RNG-free mixed copy, so both endpoints' objective
is the same. Source and stable actor remained untrained/exact; the candidate
ends64updates/14080examples. Training global Python/NumPy RNG unchanged.

| TRAIN group | Rows | Before normalized half-MSE | After | After−before |
| --- | ---: | ---: | ---: | ---: |
| current_corrections |50|.600678413262|.021219126480|−.579459286782|
| corrections |50|.347749252003|.037939799742|−.309809452261|
| push |18|.481764793593|.019320726988|−.462444066605|
| a |15|.200022162266|.062178978988|−.137843183277|
| b |50|.156213435990|.077884736498|−.078328699492|
| train20 |15|.130510221765|.078222461011|−.052287760754|
| b22 |22|.163456202539|.048363359960|−.115092842578|

Unweighted row mean falls.32935383438036947→.04713623518574177. This is descriptive
TRAIN fit, not the weighted batch objective or native task success. Full native
update diagnostics and both conditioning/residual reports remain in raw output;
FIT-SUMMARY.json is a scalar projection from the sole lead cache, not a new model
evaluation. Lower error is the predeclared direction; no alternate endpoint.

Push17: complete/50returned controls/return0/any_success=false/final_success=false,
source_unchanged=true, no unknown native/prediction work, close1/1. Its250known
control-physics steps are paid. Three Pick outcomes stay unknown, never imputed
zero. Old memory220known successes15/22/15/18 and old198neural/affine failures
remain historical controls, not new reruns or compute-matched counterfactuals.

## Execution, resources and preserved mismatch

Exactly once from physical D:\Dev\Repositories\Circadian:

```powershell
.\artifacts\runs\complete-b22-neural-study-20261101\run-study.ps1
```

Terminal hostPID58452 exit1/child1/no expiry32.2544063s. Producer29.26859887s,
sampled520482816B; GNU29.68s/max514064KiB/exit1. Fit stage24.324958058s/64229376B,
within its observed60s/512MiB gate before native admission. Existing cached
LinuxPython3.12.3/NumPy1.26.4/CPU-Mesa, one BLAS/MKL/OMP/LP thread; physical
`/mnt/d/Dev/Repositories/Circadian` and D: temp paths. No installation/download.
SAPIEN's glvnd/NVIDIA discovery warning is preserved, not treated as GPU work.

The protocol incorrectly described the continuation's ORIGINAL owner as having
60s/512MiB configured limits. Actual `fit.budget` has max_training_updates64,
but max_wall_seconds/max_process_rss_bytes and other optional fields are **null**.
The original owner was preserved, not silently given new caps. Outer meter and
observed fit-stage admission checks were active; their passing observations do
not prove that the inner owner had those configured caps. Original protocol and
capture remain unchanged. Formal protocol qualification must retain this unmet
assertion; no retroactive configuration or budget/clock renewal is claimed.

| Selected actual work | Observed |
| --- | ---: |
| Init / normal calls / values |1 /2 /1024|
| Returned SGD / training exposures |64 /14080|
| Primary / reference exposure partition |3200 /10880|
| Complete TRAIN readouts / diagnostic predictions |2 /14 (440rows)|
| Core forwards / row visits |156 /15450|
| Gradients / row visits |78 /14520|
| Native predictions / controls |50 /50|
| Native creation / closure |1 /1|
| RNG-free forks / public snapshots |7 /1560|
| Accepted source opens / bytes |6 /1985562|
| Forbidden work/reads |0|

No LS/GN/PC initialization/replay buffer/probe/final input/teacher/native baseline
reconstruction. Reference data reuse is real paid joint-batch exposure; separate
replay_examples_completed remains0. Constructor/reset physics is unmeasured.
Selected counters exclude complete allocation/JSON/validation/SDK work, covered
only in observed RSS/time. Parameter+momentum payload17472B per state; two state
records34944logical bytes excluding JSON/Python/copies. No efficiency inference.

`native_update_counts` and direct meters describe executed work. The legacy
`native_event_row_weights` field multiplies update work by event row counts:
it reports14080forward/backward passes and3097600row visits. Existing source
documents those as exposure-weighted proxies, not executed arithmetic. Preserve
them verbatim; do not quote them as another3,097,600-row computation or silently
rewrite old reports. Actual training remained64×220, with diagnostics separately
accounted as above. This distinction matters for any later cost comparison.

All8source opens were prospectively bounded; the run reached6 (threepairs), with
two-file hashes verified initially/before models/after fit before Push. The
planned after-all-native pair was **not reached**. All14code pins passed initial
and pre-native validation; the producer never reached pins_after. Independent
and root post-terminal checks found14matches. These are direct pins, not a
whole-dependency or data-source after-native qualification.

## Failure, capture and review

Actual error: `RuntimeError: native endpoint incomplete/updated`. The underlying
Push report is complete and source-exact; unlike the affine wrapper, the neural
Push helper does not emit `optimizer_updates`. The observer's unconditional
`result.get("optimizer_updates") != 0` therefore rejected a missing field. This
must not be described as a model update or a failed Push runtime. The actual
Push policy result is independently zero-success and retained. No broad ignoring
of counter changes is justified: any successor must use the existing family-
specific frozen-state/counter contract and preserve actual paid prefixes.

Raw actual1031148B SHA256
9ae55c0d6aaf23481ba65acc39e2608773c0e696d5b7149a21d042d16363cc58.
Producer0rawreads,lead1+peer1 used; no further raw read allowance. Use
LEAD-SCALARS.json/FIT-SUMMARY.json. Host observed stage1068580B/log1188B before
review additions, within8MiB/512KiB; raw<4MiB. No Pick traces were created.
Study/protocol/markers/receipts remain FAILED/SPENT and frozen.

The same sole lead byte pass cached this NEW fitted final state (policy/full
parameters/momentum/counters), sources/recipe and original costs as
FITTED-STATE.json54461B SHA256
1d1df3dcf0d608a97aed745ca89165abd4c860f734886dba801adaaadd4a6b43.
Peer verified exact final_state/sources/recipe equality with actual in its one
POSTbatch. No restoration/prediction/new training authority was created by this
cache. Any future passive use needs a separate explicit prospective allowance;
do not refit or reconstruct a historical baseline to complete the cohort.

New actual-observer AST checks4PASS/.6160438s47616000B/0model/data/numerical/native
work; producer/test Ruffformat/check pass, host syntax/storage functions pass.
Previous21software/H64readout gates reused, SPENT3SGD660fixtures not rerun.

```text
python -I -B artifacts/runs/complete-b22-neural-study-20261101/check.py
python -m ruff format --check artifacts/runs/complete-b22-neural-study-20261101/study.py tests/test_complete_b22_study_observer.py
python -m ruff check --no-cache artifacts/runs/complete-b22-neural-study-20261101/study.py tests/test_complete_b22_study_observer.py
```

Independent PRE reviewed actual source/protocol/host in2batches; found the missing
pre-native code pin check. Exact fix accepted before launch,14pins frozen. POST
accepted actual fit+Push partial evidence and exact new export, identifying the
inner budget-description error and cost-proxy interpretation above.3read-only
batches/~.9s tool time total, stdlib compact JSON/hash inspection, no project
imports/tests/model execution or numerical replication. Review ENDED. Lead
combined source/procedure/results self-review, not independent replication.

Fullsuite/cleanclone/wheel/browser/device/GPU/remoteCI/release skipped. All old
positive/negative/partial cases, consent/receipts/charges/clocks and unrelated
dirty changes preserved; no Git publication/delete/global settings change.

## Still-open scope and precise next action

#399criteria1-3software retain accepted evidence. Criterion4 has executed fit
observations but incomplete protocol qualification because of the incorrect
inner-owner assertion; criterion5 has Push negative only, three Pick unknown.
Do not check either complete from this partial job. Full goal and parents OPEN.

Next SAME399: prepare a family-correct, **inference-only** continuation proposal
for the three missing fixed Pick17/19/20 cases using the peer-verified NEW state,
keeping Push17's50negative controls and ALL64SGD/14080exposures permanently paid.
Before any restoration/native work, explicitly reconcile the original null
inner limits versus measured outer gates and freeze cumulative time/IO/storage/
state-restore/native quotas. No training, new seed/weights/dose, repeat Push,
baseline reconstruction, checkpoint framework or automatic retry. Source and
prior-evidence changes must refuse admission. Preserve the original failed
protocol assertion rather than silently recasting it as a configured cap.
