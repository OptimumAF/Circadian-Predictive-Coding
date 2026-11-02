# #410 — full-network conventional control solves three known Pick resets

The fixed16-update full-network control completed its amended experiment. It
solved PickCube17/19/20 in22/20/15controls; PushCube17 still failed after50controls.
Weighted original TRAIN loss fell from0.06507953223376839 to0.0017988678820405775.
Independent review accepted the completed measurement and usefulness on these
three known TRAIN resets. This is not reliability, generalization, chronological
retention/recovery, matched-compute optimizer superiority or a circadian benefit.

| Known TRAIN reset | Accepted original64-update checkpoint | After16full-network updates |
|---|---|---|
| Push17 | Failure,50controls,return0 | Failure,50controls,return0 |
| Pick17 | Failure,50controls,return0 | Success,22controls,return1 |
| Pick19 | Failure,50controls,return0 | Success,20controls,return1 |
| Pick20 | Failure,50controls,return0 | Success,15controls,return1 |

Original #399 outcomes were reused, not rerun. They came from the same qualified
checkpoint/environment; its inner-budget protocol deviation remains OPEN.
One fixed episode per reset does not establish success rates across new seeds.
Additional GGN computation/exposure makes this an unmatched-resource control.

## What changed, and what remained fixed

Existing #409pure GGN/owned BP API was used without production changes. The
saved D12/H64/seed17/TRAIN/mixed-gripper policy, original seven220-row TRAIN
groups, raw positive weights and XYZnormalized half-MSE/gripper2zCE objective
stayed fixed. No initializer/model RNG, teacher collection, final/probe labels,
policy promotion, seed/dose/weight/metric search or baseline retraining occurred.
The source checkpoint stayed unchanged; one private fork received16accepted
strict-descent updates, each with one damped norm-capped trial. All momentum
slots reset as declared. Final counters80updates/17600exposures retain original
64SGD/14080rows plus16GGN/3520rows; these are different optimizer operations.

Earlier [admission failures](full-network-native-control-20261101.md) remain
immutable FAILED/SPENT/PARTIAL. A prospective scientific protocol correction
removed the unnecessary exact-raw-interpolation prerequisite while preserving
the accepted raw1/native0 discrepancy, prior_completeFalse and its provenance.
The original acceptance asked strict loss decrease, not perfect interpolation.
No optimizer/native outcome had been observed when the correction was frozen.
The accepted consistency result was reused for unchanged bytes/code, not rerun.
True native-command contradictions remain refused; no labels were changed.

New explicit5input reads608239B plus earlier3/387339B=8reads995578B cumulative.
The old process was not resumed. Prior2.345914905s body/7.7717076s host, one
restore/fork, six consistency kernels and selected copy work remain charged.
The new private clock adds old body time to unchanged cumulative resource gates.
Old #410finalpostchecks, #408phase/file gaps and #399deviation are not repaired
retroactively. There is no automatic retry or future experiment allowance.

## Executed procedure and observed results

Exact ONCE command from physical `D:\Dev\Repositories\Circadian`:
`$env:TEMP='D:\Dev\_Temp'; $env:TMP='D:\Dev\_Temp'; $env:TMPDIR='D:\Dev\_Temp'; .\artifacts\runs\full-network-native-control-20261101\run-materiality.ps1`.
Cached LinuxPython3.12.3/NumPy1.26.4/ManiSkill3.0.1/Sapien3.0.3/softwareMesa,
CPUthreads1/GPU0/D:temp; no install/download/network/global change. Existing SDK
fixed-reset seeding is separate from MODEL RNG0. A GLVND ICD warning was logged;
the fixed software-rendered case nevertheless completed all native contracts.

HostPID35884/exit0/17.267444s/noexpiry; cumulative host25.0391516s. New producer
14.128814031s/cumulative16.474728936s; GNU14.50s/max528436KiB. Fit+diagnosis
cumulative8.4252576s/82,280,448B; overall sampled536,145,920B. These satisfy
90s/512MiB fit,125s/1792MiB workstop,145s/2GiB whole/cumulative time bounds and
newhost150s plus prior<165s. RSS is observed, not OS-enforced. All6combined logs
and full stage remain below512KiB/8MiB. Complete/budget/source_unchanged true,
error/permanentstop null;18pins before use/beforefit/beforenative/after, final
input-byte checks and original full source proofs passed in THIS case.

All seven original unweighted diagnostic half-MSE values fell:

| TRAIN group | Accepted checkpoint before | After16updates |
|---|---:|---:|
| current_corrections50 | .0212191264797546 | .0006716203148138398 |
| corrections50 | .037939799741646005 | .0006234502893042393 |
| push18 | .01932072698834951 | .001627184888515056 |
| a15 | .06217897898814366 | .0015257813261950288 |
| b50 | .07788473649775597 | .0026629619745648686 |
| train20_15 | .07822246101113833 | .0009462260756697378 |
| b22_22 | .04836335996026541 | .004241859426635779 |

Before values use accepted #408TRAIN scalar observations, whose own capture
qualification remained partial. Group readouts are unweighted diagnostics,
distinct from the weighted batch objective; do not average them into that loss.
Command diagnostics are descriptive: all220exact native-command rows still
differ from teachers. Gripper negative/zero margins are now0, but translation
directions are not uniformly better (PushXYZ0/5/0→2/9/3; B22Z7→10). Zero/nonzero
direction differences are included. These are not extra success metrics or a
causal explanation of Push failure/Pick improvement. Original native success
and unchanged imitation objective remain the declared outcomes.

## Paid work and preserved limits

| Work | New case | Cumulative #410 including earlier attempts |
|---|---:|---:|
| GGN updates/input-row exposures |16/3520|16/3520|
| Original gradient requests+returns/rows |39/7260|39/7260|
| Forwards+returns/rows |162/10887|162/10887|
| Square1092×1092least-squares solves |16|16|
| Native scenes closed/controls/known control-physics |4/107/535|4/107/535|
| Input reads/bytes |5/608239|8/995578|
| RNG-free restores/forks |1/4|2/5|
| Public snapshots/native validators |840/2568|843/2588|
| Selected returned state-copy bytes |59544576|59946432|
| New consistency calls |0|6earlier/9780paidpairs|

The optimizer pays48forwards10560rows/32backwards7040rows; seven diagnostics pay
7forwards220rows/7backwards220rows, even though their derivatives are unused.
Native107predictions add107forwards107rows. Every attempt returned. Selected
fit work:15,375,360Jacobian cells/19,079,424Gram cells/19,096,896solver-input
values,16,809,717,760known dense MACs and276,535,296selected numeric payloadB.
These are not complete FLOPs/allocation/RSS totals. Solver arithmetic/workspace,
elementwise/validation/other copies/Python, constructor/reset physics and SDK RNG
remain additional/unknown. Seven command calls pay1760native32 conversions;
no extra predictions. Original baseline computation remains paid and unmatched.

All four native lifetimes returned creation+close ACKs; no native optimizer
work or possible failed control-physics was reported. Pick owner/state/history
proofs and actual Push source/control contract passed; Push has no neural
optimizer_updates field. Private candidate full state stayed frozen through
rollouts; original source/inputs/paid prefixes remained exact.

## Validation, review and handoff

NEW literal boundary command:
`.venv\Scripts\python.exe -I -B artifacts/runs/full-network-native-control-20261101/check-materiality.py materiality --tb=short`
passed6cases/.57538209995s/48,615,424B/0model-input-physics. It refuses changed
provenance/coverage/Boolean counters/raw-erasure/native conflicts and retains
the original raw mismatch/prior_completeFalse on admission. One dedicated new
command marker is SPENT; old70literal executions/38distinct and failed Linux
capture/unknown RSS remain untouched. #40937numerical cases reused, not rerun.
One stale removed-consistency exception branch caused a lint/type failure,
then was removed; final Ruff/format3files/mypy2files/hostsyntax and9shared-helper
AST comparisons passed. New helper's executed AST is unchanged after that
unreachable stale branch cleanup. Full suite/Linux numerical/wheel/install/
cleanclone/remoteCI/browser/device/release checks were not rerun for this scoped
experiment. No product API/default/dependency changes, commit/push/deletion or
unrelated dirty edit; HEADd9119e8 preserved.

Independent actual delta/PRE/POST review accepted completed amended measurement
and useful-known-TRAIN scope,3readonlybatches/~.9s, oneNEWraw bytepass only,
0imports/tests/inputs/export/oldraw/models/native/replication/writes. Current18
pins match. Exception ENDED. Lead combined source/procedure/results self-review
is separate. Raw `materiality.json`251264B
SHA256`ee8b0276cf6c0fa247fd3ac18013da519223b7ef7927a2ac0bc384f0f5020cb7`;
lead1+peer1reads SPENT. Use MATERIALITY-SCALARS/MATERIALITY-COMPARISON caches.
Stage retains all protocols/originalfailures/copies/markers/receipts/pins/reviews.

#410all scoped criteria now evidenced under the prospectively amended protocol;
earlier failures remain failed. Broader18/21/23/33/fullgoal stay OPEN-ACTIVE.
Roadmap reassessment: a conventional neural policy now performs three existing
Pick resets, so the useful next outcome is an isolated fresh-task/old-task probe,
not another dose/capture increment or a Push4/4chase. #411 will reuse existing
ContinuousProbePlan/run_continuous_probe/delayed runtime and native endpoint
observer, with fixed23/24roles and old17/19/20readouts. First inspect neural
private-endpoint/native role admission; add only any demonstrated small missing
composition. Restore the qualified last80/17600record under a future explicit
input-use protocol, never retrain this baseline. No new input/model grant here.
Actual context-return recovery and reliable joint retention remain separate
open requirements; before/after endpoints alone do not prove recovery.
