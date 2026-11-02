# #408 — computed TRAIN diagnosis; failed phase capture, partial qualification

The one inference-only run computed all seven TRAIN residual/command readouts,
then failed the existing phase result serializer. **#408 remains OPEN, with the
actual attempt FAILED/SPENT/PARTIAL.** The captured observations are useful;
they do not establish full protocol conformance, improved native control or a
causal explanation. No inference, saved-model or native retry was performed.

The fixed neural checkpoint still has substantial action-fit limitations on its
original220TRAIN rows. Every four-value native command differs from its target.
There are37/54/84translation-direction disagreements on X/Y/Z, respectively,
and9negative analog gripper endpoint-sign margins. Direction disagreements include
zero versus nonzero, not only opposite nonzero directions. All seven primary
half-MSE values match the previously frozen published12decimal values.

| Original TRAIN group | Rows | Half-MSE | X/Y/Z direction disagreements | Negative gripper margins |
| --- | ---: | ---: | --- | ---: |
| current_corrections |50|.0212191264797546|0 /2 /7|0|
| corrections |50|.037939799741646005|0 /3 /44|0|
| push |18|.01932072698834951|0 /5 /0|0|
| a |15|.06217897898814366|5 /4 /2|1|
| b |50|.07788473649775597|22 /28 /22|5|
| train20 |15|.07822246101113833|3 /3 /2|2|
| b22 |22|.04836335996026541|7 /9 /7|1|

There are211positive and0zero gripper margins. These are target-sign times an
ANALOG gripper command, not a thresholded policy or grasp-success measure. Push
has no negative gripper margin yet its earlier native trial failed; this does not
identify gripper sign as a sufficient explanation. Likewise exact command
differences and direction mismatches do not prove what caused any native failure.
Do not change old metrics, tune reset24, or assert a circadian advantage.

## What ran, and why this was a distinct observation

The existing typed restorer loaded one immutable54461B qualified FITTED-STATE
export (SHA1d1df3dcf0d608a97aed745ca89165abd4c860f734886dba801adaaadd4a6b43),
with source origin9ae55c0d6aaf23481ba65acc39e2608773c0e696d5b7149a21d042d16363cc58.
Complete policy, four parameter/four momentum arrays and64/14080counters were
validated. The original qualified166439B bundled220TRAIN source was used with
unchanged roles, history and separate recorded replay admission. No replay
prediction, new label, source archive reconstruction or oldactual read occurred.

Original #399 and continuation training/native/raw/read allowances remain SPENT,
including all four negative native tasks and the inner-budget deviation. This
new bounded immutable-input use answered a changed TRAIN command-fit question;
it retrained no baseline and was not an independent scientific replication.
There was no expired live learner/clock/receipt/runtime resumption, promotion,
model publication or final/probe data use. The new private source/fork were
read-only. Original64updates/14080exposures remain inherited training cost;
the new work below is additional inference/diagnosis, not free reconstruction.

Exactly once, from physical D:\Dev\Repositories\Circadian:

```powershell
.\artifacts\runs\train-command-diagnosis-20261101\run-case.ps1
```

Existing LinuxPython3.12.3/NumPy1.26.4/CPU1/Dtemp matched the original producer.
HostPID69080/exit1/4.8927s/nohoststop; producer1.941610147s/samplepeak51261440B;
GNU elapsed2.03s/RSS49888KiB/exit1. No time, memory or work cap failed.

Paid1restoration/1fork,7predictions+7forwards220rows225280denseMACs,7command returns,
1760native scalar conversions+returns.62public snapshots/132validator returns/
2306304known numeric state-copy bytes; prediction/target seal copies14080B and
command-helper owned action arrays35200B additional. Scope counters can overlap:
public snapshots and phase copies are part of the validated state-copy paths,
not independent values to sum blindly. Python/JSON/normalization/temporary arrays/
validation and process/SDK installation IO are additional to these logical counts.

Source schema/history admission also paid926packing calls/1136scalar values;
total returned packed numeric bytes11584, separately from command conversions.
Actual3input reads387339B (bundle2/export1); the planned final two reads did not
run. Model initialization/RNG/training/fit/gradient/backward/conditioning/SVD/
SDK/native/teacher/evaluation/forbidden work0. New diagnostic allowance is SPENT;
unused postcheck read slots are not authority to resume a dead process.

## Failure and evidence limits

Internal diagnosis complete:14paid residual/command records/14ACKs, no unknown
unreturned operation. The observer returned tuple-bearing metadata and residual
dataclass containers to `PhaseReadouts`, whose unchanged native-JSON contract
correctly rejected them: `ValueError: phase result must contain native JSON values`.
The top-level writer encoded the captured records to a valid JSON artifact, so
the observations survive. Phase returns1/validated0/resultbytes0/failedtrue.

`PhaseReadouts` finally still ran rootguard, metadata, original source/detached
unchanged and no-alias checks. The raw result has no secondary_guard_error; source
review supports model/in-memory-input preservation at those observed boundaries.
It does not supply the unexecuted final input-file checks, pins_after or
source_unchanged/restored completion markers. They remain missing, not imputed
successful from the complete numerical work. Full original conformance cannot
be asserted or supplied retroactively by the repair.

Original producer/host/protocol/pins/raw are preserved. NEWactual46927B/SHA256
6fe510d80317c840da041598915d5b709aa0b65b5f1f9025db31160abc9c1126; lead1+peer1
bytepasses SPENT. Use LEAD-SCALARS.json and SCALAR-COMPARISON.json thereafter;
no dataset/export/model/raw re-open or inference retry. All seven scalar
comparisons pass at the declared printed precision only.

## Checks, correction and independent review

Before actual data:20initial synthetic procedure checksPASS/1.3062417s69054464B;
30finalPASS/1.155604s69976064B. They use typed generated rows and diagnostic doubles,
zero restorations/model math/real input reads. Final explicit forbidden-work/input
guards report0. Initial one typing annotation error was fixed; no numerical test
failed. New code was local procedure/test support, no production module edits.

PRE review found and corrected full nested owner identity before/after getters,
ACK/copy bytes before a late stop, exactint receipts and shared stdlib packing
accounting. Both state/owner policies/specs/unit-bound tuples and parameter/
momentum tuple containers/arrays are pinned before the first public getter. The
packing meter separates command'<f' from history'=f'/recorded replay'<4f'; it
did not increase the1760command-conversion allowance. Budgets froze before inputs.

The serializer mistake was not covered by that original suite. A separate
`capture_fixed.py` COPY now adds a native JSON bridge using the same pattern as
the existing readout: `json.loads(encoded(diagnosis))`. It changes no values or
numerical work. Its case.SPENT guard and original host prevent an accidental
rerun; it is a prepared, unexecuted producer copy, not a repaired actual result.
One metadata-only test reproduces rejection of the original tuple-bearing
synthetic payload and passes the unchanged strict64KiB gate after the bridge,
proving equal JSON values. It PASS/.9833503s70877184B/30deselected, zero real
model/data/inference. All3new fixture commands are SPENT (51executionsPASS,
3.445196s/max70877184B; deselections are not executed tests).

Exact fixture commands:

```powershell
.venv\Scripts\python.exe -I -B artifacts/runs/train-command-diagnosis-20261101/check.py initial --tb=short
.venv\Scripts\python.exe -I -B artifacts/runs/train-command-diagnosis-20261101/check.py final --tb=short
.venv\Scripts\python.exe -I -B artifacts/runs/train-command-diagnosis-20261101/check.py capture --tb=short -k metadata_bridge
```

Scoped Ruff/format/mypy on procedure/test sources and host syntax PASS; no
new numerical kernel cohort, full suite, clean clone/current wheel/install,
browser/device/GPU/remoteCI/native trial/comparative/release check. Existing407
math/default/ownership evidence was reused, not rerun. All18pins matched before
the actual run and at peerPOST. The later metadata-test addition changes only
the test-source pin: executed-tests.py preserves the exact executed prefix with
its original hash, and17other current pins still match. Original pin table was
not rewritten. One failed protocol-patch anchor was corrected by explicit append;
no data/model work was repeated to repair tooling.

One finite peer's PRE3batches1.1s/POST1batch.8s independently read actual source/
protocol/final30receipt and one new result/currentpins/resources. No imports/tests/
oldinputs/export/models/native/replication. POST accepted captured observations
and rejected full completion. Exact-text review then accepted the metadata-only
remedy/receipt, without another tool batch. Exception ENDED; lead separately
self-reviewed combined source, observed procedure and results. See stage/REVIEW.md.

## Next learner outcome, not another capture increment

#408 remains OPEN at original actual phase/postcheck conformance; no retry is
authorized or necessary to read the already-captured observations. Preserve the
failure and apply the tested JSON representation in future separately budgeted
procedures. Broader scientific/release and full repository goals remain ACTIVE.

The measured remaining TRAIN error justifies testing a stronger conventional fit
control, not asserting capacity/optimization is the proven cause. Source inspection
finds an existing head-only original-MSE Gauss-Newton control (hidden frozen,
<=130rows, mixed-gripper rejected), but no full-network Jacobian/proposal. Next
#409/R2.1-CTRL.FIT will reconcile consistency evidence and existing solver/loss/
commit contracts, then implement the smallest bounded full-network linearization
and strict-original-objective-descent proposal through existing learner paths.
Defaults, mixed objective, source/actor, scientific primary metrics and all failed
controls remain unchanged. No model/data/fit/native allowance comes from creation.

Exact next: read head `_linearize`/`_solve`/proposal plus adapter commit guards,
freeze full-network parameter ordering/Jacobian/mixed-loss derivative/curvature
and finite resource limits, then add tiny finite-difference tests. This advances
the learner; it does not claim improved control before a separate qualification.
