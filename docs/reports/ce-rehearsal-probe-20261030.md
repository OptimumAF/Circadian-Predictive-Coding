# Role-safe staged CE rehearsal updates

R2.1-TW.CE.PROBE / issue341. Previous340 was progress: the ordinary joint TRAIN
readout genuinely solved six workflows through final capture, without proving
chronological adaptation. The existing CE fitter intentionally rejects probe
roles. This increment adds a distinct staged proposal port and verifies its
numerical/state/role behavior before any existing-runner integration.

## Implemented and executed behavior

New `torch_ce_probe_update.fit_ce_probe_update` accepts one exact current
AttributionBatch plus bounded retained TRAIN batches. Fresh current role is
probe_train; return current role is TRAIN. All roles are checked before matrix
access or owner work. Hard targets, matching geometry and combined128-row bound
apply. Fresh IDs must be disjoint from anchors; duplicate return TRAIN IDs must
have matching context/features/targets and remain counted twice in fitting.
Original arrival and retention provenance remains the caller's responsibility.

The supplied native BP owner is never changed. A successful result is a
distinct CEProbeUpdate with owned before/derived state, immutable coefficients,
phase/row metadata, bounded losses and paid work. It is not a CategoricalCEFit
and cannot be consumed as TRAIN fit authority. The copy validator rejects
changed hidden state, fabricated SGD credit, coefficients, phases and role costs.
Only output parameters change in the derived state; original hidden parameters,
gradients, optimizer momentum, policy and native counters remain intact.

The proposal contains no probe-label bank or retained optimizer/input list. Fresh
labels are used only in this fit, without conversion to TRAIN or import into
replay. The returned coefficients, state and bounded metrics represent the current
fit; no replay bank or optimizer history is part of that proposal. No update is published
to the supplied learner, source, actor or benchmark. Future receiving code must
validate the before-state and original private-owner guards before application.

Tiny actual fits verify an independently owned private fork classifies all three
fixture rows correctly. Proposal coefficients, closure losses and final loss
exactly equal the TRAIN fitter from the SAME initial state and rows on each
platform. The objective decreases below0.01; no tolerance or seed was changed.
These are 2D/H3/K3 software fixtures, not the six-workflow native study or a
fresh-task acquisition claim. Return fitting checks same-fact TRAIN reuse and
its separate paid counts. Source state/ownership/RNG are preserved.

New tests also cover strict roles/types/phase, held-out labels and foreign role
callbacks, target/geometry/ID faults, equal-value anchor-array replacement,
current-data/role mutation before backward, lost optimizer return and foreign
source changes after a completed fit. Refusals preserve the changed evidence
and actual paid work; no hidden rollback erases a foreign modification.

`probe_gradient_rows`, `current_train_gradient_rows` and
`retained_train_gradient_rows` use backward attempts, including partial failures.
`native_sgd_updates` is0 for the proposal. The tiny fixture's native warmups and
validation controls still perform real SGD and are charged separately below.

## Shared arithmetic and preservation

The original TRAIN API still accepts only exact released TRAIN DecisionCases,
uses the same definition/result type and retains its owner/input guards. Its
detached optimizer block moved to private `_fit_output_arrays` in the same
module; existing observer hooks and all fixed float32 affine/LBFGS settings,
hard64-closure limit and work-counter semantics remain.

One source-only AST comparison against the exact preimage confirms the entire
numerical statement block is unchanged after case/feature/target aliases and
two role-neutral error-text substitutions. Coefficient/logit/loss/final-loss
return expressions exactly match the old result arguments. Evidence:
kernel-source-check.json; no numerical replication or framework was introduced
for that inspection. Legacy fit/reader/source/RNG/failure/hard-cap tests and two
existing TRAIN-initialization tests execute against the extracted implementation.

Implementation scope: existing torch_categorical_ce_readout.py and new torch_ce_probe_update.py /
test_torch_ce_probe_update.py changed. The original fitter preimage is retained;
new source/test/stage/report/guide/temp paths were checked absent first. No
native SGD port, original initializer, app runner or CLI changes were made.

## Checks, failures and resource costs

Windows:56 distinct passing checks, comprising23 unchanged legacy passes from
the original prefix and33 new/initialization passes from a targeted continuation.
Linux:full56PASS/0failed/0skipped in one cohort. No completed legacy Windows fit
cohort was repeated. The only behavioral fix after its23 passes was the new
proposal's batch comparator; original TRAIN/core code stayed unchanged.

The first Windows cohort failed at new-proposal setup: tensor-only
torch_decision._same_values could not compare NumPy matrices. It refused before
new proposal hidden/CE work. Original windows-work.json/SPENT and all charges remain; the reserved
temp path was never created and was never cleared. Replacing that new batch comparison with existing exact-type/dtype/shape
coupled_attribution._same_value preserves the separate identity and role checks.
This corrects input handling, not a weakened acceptance criterion.

Prospective amendment6536514398/ENGINEERING_AMENDMENT.md allowed one targeted
Windows software continuation while retaining all original per-platform pooled
work caps,180s aggregate testwall and120s statics. The checker deducts prior
platform charges and uses a distinct absent temp/SPENT path. The original
one-cohort condition was not retrospectively declared passed. No scientific
experiment or old performance allowance was renewed.

Commands from physical D:/Dev/Repositories/Circadian; PowerShell TEMP/TMP
D:/Dev/_Temp, WSL TEMP/TMP/TMPDIR=/mnt/d/Dev/_Temp:

```text
.venv/Scripts/python.exe -B artifacts/runs/ce-rehearsal-probe-20261030/check.py windows
.venv/Scripts/python.exe -B artifacts/runs/ce-rehearsal-probe-20261030/check.py windows-targeted tests/test_torch_ce_probe_update.py tests/test_torch_categorical_ce_initialization.py::test_derivation_preserves_full_source_sgd_rng_and_has_no_model_work tests/test_torch_categorical_ce_initialization.py::test_native_typed_predictions_equal_ce_reader_without_rescaling
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/ce-rehearsal-probe-20261030/check.py linux
```

Initial Windows EXIT1/23pass1setup-error:5.436727200s/327368704B. Targeted Windows
EXIT0/33pass0fail0skip:4.131315900s/325615616B. Linux EXIT0/56pass0fail0skip:
12.211097466s/343097344B; session15228 polled to terminal, never restarted.
Total process testwall21.779140566s under180s, all per-process90s512MiB limits.

Pooled Windows, including failure:49components98seeded draws735values,
7native SGD/backwards21rows,17CE optimizer/step attempts (7step returns),
232CEforwards696rows/225CEbackwards675rows/7final reads21rows;
25hidden reads59rows/8fitted affines8rows,15native head forwards35rows and one
deliberate NumPy RNG fault draw. Linux:47components94draw705values,6SGD18rows;
same CE/hidden/affine totals,14head forwards32rows and one deliberate RNG fault.
Those intentional faults are refused and preserved, never seed selection.
Forward counts name head/affine operations; loss-only diagnostic reductions,
state copies and validation add overhead measured by wall/RSS, not separate FLOPs.

All original pooled caps held:128components/8SGD24rows/24CEoptimizers/
1024CEforwards/backwards6144rows per platform and the remaining named limits in
work receipts. Across both platforms all attempts cost13nativeSGD39rows plus
450CEbackwards1350rows =1389gradient-row exposures. Model/copy/validation work is
real software cost, not zero models or an equal-compute certificate. Peak RSS
is per process, not additive. No LS/LP/SVD/PC-latent/physics/final/GPU/remote work.

Scoped Ruff/format four files and default+Linux-target mypy three source/test
files PASS. Initial missing set annotation was corrected before model fixtures,
with failed static receipt retained. Exact argv/results/timings:statics-*.json. Cumulative statics13.630186300s; scoped mypy cache16330981B plus shared Ruff
cache425447B (including inherited entries), below120s/64MiB. The unrelated
pre-existing global mypy cache is not part of this work and was preserved.
Named test-temp roots were never created (0file bytes); their absent names and
all three durable SPENT markers remain. Linux-target types ran on Windows.
A read-only PowerShell rg wildcard-path error during closeout was corrected
with directory/-g search; it performed no model work or file mutation.
No new dependencies, installs, global settings changes, deletes, commit/push or
unrelated reset. D:cache/temp/stage stay within declared64/16/4MiB allowances.

No D9 source12/F16-return16, genuine workflow/adaptation study, new baseline,
four-family benchmark, wheel/clean-clone/browser/GPU/release/sweep was launched.
Ordinary340/338 evidence is reused only for the unchanged numerical method;
331FAILED/SPENT,338's timing gap and323/337/328 results remain untouched.

## Acceptance and next implementation

Lead self-reviewed actual numerical extraction, strict role/type/data binding,
derived state and SGD invariance, paid failures, tests and pooled costs.
Independent actual math/state/role/result review ACCEPT for the staged-component
scope: five read-only commands, all EXIT0,0.7424521s under25s. Physical REVIEW.md
was verified after writing; exact commands and costs are retained there. No
numerical/project imports, test/model execution, child delegation or network
access occurred in review. The finite exception ended. This is distinct from
lead self-review and does not accept future caller integration or adaptation.

This delivers a functional staged update, not existing-runner publication or
demonstrated sequential adaptation. Next precise implementation: add an explicit
role-aware update callback to the existing disposable fresh/return probe path,
apply a validated proposal only to its still-matching private owner after the
original guards pass, and retain auxiliary fit/role costs separately from SGD.
Keep defaults, source/actor/benchmark isolation and TRAIN-only API unchanged;
do not add a probe-label replay bank. Start with bounded integration fixtures,
then freeze a small adaptation protocol. Full R2/R4/R5/R7/science/release goals
remain open; no claim that this optimizer will succeed on the original tasks.

Canonical closeout: [#341 evidence](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/341#issuecomment-6536738847), scoped issue CLOSED/completed.
Parent handoffs: #18 comment6536741962, #21 comment6536742209, #23 comment6536742489; parents/full goal remain open.
