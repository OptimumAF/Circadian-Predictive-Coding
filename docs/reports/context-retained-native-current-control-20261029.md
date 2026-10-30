# Native current-only control with accepted joint reuse, 2026-10-29

R2.1-CTX.LS.T.J.C, [#306](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/306).
One new control; the sole accepted #305 joint observation was reused, never fitted
again. No production source change, new learner/runner/loader or sweep. Full goal,
#21/#18 and broader scientific/release requirements remain open; #270 is a closed
scoped dependency. [Frozen protocol](../context-retained-native-current-control.md).

## Fixed-case result

| Arm | Old tasks before / after F / after return | Selected tasks before / after F / after return |
| --- | --- | --- |
| Joint, saved #305 | 4 / 4 / 4 | 0 / 2 / 0 |
| Current-only, new #306 | 4 / 2 / 4 | 0 / 2 / 0 |

All18 probe_test episodes per arm completed in three steps, zero task-readout
updates/external actions. Joint retains two more old tasks immediately after F
in this fixed comparison, at equal physical LS fits/rows. Both acquire the two
selected tasks and then lose BOTH on old-TRAIN return. Current-only recovers the
two lost old tasks; joint never lost old-task success, so has no recovery-from-
old-loss claim here. Old password-v1/parcel-v2 fail after current-only F.

This supports a scoped effect of including old TRAIN information in a joint fit
versus extra current-row exposure. It is not matched all-in compute, statistical
replication, unseen/disjoint generalization, CPC/model-family advantage, efficiency,
deployment or release evidence. Concepts/templates/public contexts are shared;
legacy skills outside the four-old/two-selected readout set remain unverified.

Fixed selected-task classification accuracy in both arms:2/3 initially,1 at F4/8/16,
2/3 at return4/8/16. Initial CE1.2580700206584228/1.2263485469698603; current-only
F CE1.0435917781858575 each; return1.22634854696986/1.2263485469698603. Joint F
CE1.0435917781858577 each. Tiny floating differences are preserved, not used to
select/retune/rank a winner. Full original CE/per-class trajectories/checkpoints
remain in each raw output. No seed/order/metric/direction/endpoint changes.

## Matching and isolation evidence

The stdlib comparison passed189 exact checks: all source checkpoints/weights/
policies/counters/TRAIN metadata, root configuration, retained candidate state,
initial donor state/task outcomes, recipe/policies/input geometry, selected task
slots, original budgets and fixed TRAIN bank/ordered episode selections match.
Independent reviewer also directly compared source/candidate/initial state/recipe
and ordered selected IDs/episodes. Comparison accepts ties and negative outcomes;
it does not require post-treatment model states or successes to match.

Probe-cohort identity additionally rests on the unchanged deterministic builder
and existing exact teacher/role admission. Raw probe arrays are NOT serialized
in comparison.json, so that file is not direct byte-for-byte array evidence.
Production source and accepted admissions were unchanged; source assumptions are
explicit rather than silently inferred from a metadata-only check.

Both runs independently passed original source/runtime/candidate/actor/donor/
owned/RNG isolation checks; root final/deployed probe updates zero. Retained
candidate stays23/144, private work48fits288rows with unknownwork false. Original
actor remains unchanged, live15/24, including its failures. No probe weights,
labels or updates enter retention, promotion, deployed/benchmark learner or final
evaluation. Process-local owners/elapsed times differ across runs; each original
owner/clock/charge remains protected within its own run.

Identical work geometry: source12/72 +ordinary retained11/72 +private48/288 =71LS/
432rows each, native readouts23/144→55/336→71/432. During F both pay16three-row wakes
and16nine-row consolidations. Joint fits96 current-row exposures and96 old TRAIN
exposures total; current-only fits192 current-row exposures and zero old TRAIN
rows during F. Both then fit16six-row old-return packets/96rows. Fixed bank four
TRAIN episodes12native transitions5216encodedB and selection are identical; old
TRAIN is available to both selectors but only joint uses it in the F fit. Probe
rows stay transient. Replay144rows/events16/attempted=returned=committed16, all
charges permanent. Same fits/rows does not establish equal numerical/total compute.

## Execution and budget ledger

Physical `D:\Dev\Repositories\Circadian`, HEAD0aae1d6 plus preserved inherited dirty/
untracked work. Existing Linux3.12.3/NumPy1.26.4 freshly confirmed. TEMP/TMP
`D:\Dev\_Temp`; Linux additionally TMPDIR `/mnt/d/Dev/_Temp`. Six affected production
files unchanged since accepted #304 statics/Linux40 checks and #305 gate/source.
The new envelope is EXACT #305 source except one joint→current_only argument.
No repeated joint/baseline/model-fit fixtures or changed production static checks.

Prelaunch bounded stdlib comparison gate:

```powershell
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-retained-native-current-control-20261029/gate.py
```

Five synthetic-metadata checks PASS, terminal0,0.101895519s24,633,344B,0physical
model work: same-source zero effect and negative post-treatment outcomes accepted;
source/role/selection corruption refused. Copied synthetic work counters are NOT
executed fits/control evidence. Three artifact AST checks/exact-envelope-difference
check passed. Accepted #305 producer gate/#304 tests/statics reused unchanged.

ONE new native invocation after independent source/protocol GO:

```powershell
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-retained-native-current-control-20261029/native.py
```

Terminal PROCESS EXIT0 plus receiptstatus0/errornull,71LS432rows71lstsq,
12.954211610s66,330,624B. Permanent native.attempted, no retry/renewal/amendment.
Receipt status precedes final assertions; actual bounds and terminal exit0 also
checked, not receiptstatus alone. Source120s age11.731153681; retained60s
age10.584558984; F/return30s ages9.087240481/9.087280943; replay144-row30s
age4.241305872. Original F16/48 and return16/96 limits unchanged. Phase-only
F2.802341969s/return0.821531399s exclude guards/readouts and do not replace ages.

All named counts exactly match #305:142nested training ports864rows (not142solves),
212affine212rows,38probability171rows,498routes1232rows,21authentic traces6training
builders72task episodes383sandbox dispatches248context typed496context+leaf
delegates38predict ports. Counters are before delegation with shared locks across
main/new threads; not every scalar/array operation or process-tree cost. All named
caps held, no Torch/neural/GPU/optimizer/latent/modelrandom/final access.

Zero-model-work comparison:

```powershell
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-retained-native-current-control-20261029/compare.py
```

Terminal0/statusmatched189checks true,0.038607725s25,776,128B, no model import/fit/
forward. Each named input limited4MiB; source mismatch would be inconclusive,
without relaxed tolerance/retry. Gate+comparison+new native sum13.094714854s.
Paired historical+new native work142LS864rows, summed invocation times26.081714471s;
this is NOT a new142-fit allowance or one simultaneous timed run. #304 software
273LS1656rows and #303/prior failures remain separately paid historical evidence.
No efficiency ranking from13.1275s versus12.9542s or equality of named operations.

Artifacts `artifacts/runs/context-retained-native-current-control-20261029/`:
8files2,159,389B including actual.json2,128,800B, below original8MiB. Native output,
work/marker, comparison.json, comparison-gate.json and scripts retained. #305
original files untouched. Local/unpublished. No native failure this increment.
Two non-model tooling failures preserved in tool/issue history: Windows Unicode
print error AFTER envelope write and an unmatched empty documentation patch
(zero writes). Source/AST checks confirmed the envelope and work proceeded using
the verified copy; no added tooling chain or scientific restart.

Lead self-reviewed source/assumptions/procedure/actual results, costs, clocks and
matching. Finite independent source/protocol GO preceded execution and DIRECT
result/common-state/selection/cap/clock/comparative review accepted the scoped
interpretation:3readonly calls0.4512089s, stdlibJSON only, no project imports/tests/
models/network/writes. Fourth unused, exception ended; no scientific replication.

Skipped repeated full software/math/default-native/baseline/raw-backed/model-family/
GPU/remoteCI/clean-clone/current-wheel/release checks; unchanged accepted evidence
reused. #279 wheel predates current code. No installs/downloads/deletions/global
changes/commit/push. All unrelated dirty/ignored work, #285 losses, #287 FAILED/
SPENT and every earlier failed/frozen capture preserved. No circumvention of
retired publication/counter/budget/consent history or final isolation.

## Next useful implementation action

Both arms still lose selected tasks on old return. Current generic
`LinearFitLearner.train_batch` always cold-refits a minimum-norm current-packet
solution; it does not supply previous coefficients. An existing minimum-change
residual solve already lives in `update_wide_categorical_linear`, preserving prior
components invisible to the new rows while not otherwise protecting old predictions.
Do not invent another algorithm or weaken its width/row/CLM-role boundaries.

Next: share that existing residual arithmetic and add an explicitly named guarded
minimum-change update port to the generic linear owner, preserving all default
minimum-norm behavior/snapshot/counter/ownership/failure contracts. Start with
behavior tests for invisible-prior-direction preservation, already-solved batch
stability, overlap limitations and atomic failure accounting under a prospective
software budget. This is a conventional learner hypothesis, not a proven remedy
or CPC benefit. No native experiment/new comparison allowance here. Freeze any
later model variant separately and preserve these baseline results unchanged.
