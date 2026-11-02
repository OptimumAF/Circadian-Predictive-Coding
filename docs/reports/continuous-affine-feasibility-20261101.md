#398 — physical affine control and same198 TRAIN feasibility

Implemented and executed a separate bounded TS12/four-action weighted affine
fit/prediction control. The existing qualified reader supplies all198 original
TRAIN rows; the existing six-group weight recipe is unchanged. The core reuses
the validated fixed-cutoff LS algebra, without widening either old128-row API.
No new solver, loader, scheduler, trainable-owner hierarchy or dependency.

New production files: `src/core/continuous_affine_fit.py` and
`scripts/fit_aggregated_affine_control.py`. Immutable coefficients/rank metadata,
owned predictions, exact source and weight-recipe guards, original budget/clock
binding and permanently charged attempts are implemented. Source roles remain
TRAIN. No final/probe, old model restoration, statistics fitting, teacher,
simulator, actor update or promotion occurred.

**#398 remains open.** The pure control, qualified-data composition and one
feasibility case pass; the read-only native policy boundary and physical
qualification remain unfinished. TRAIN error is not evidence of robot success.

## Actual same-data result

One solve on198x13 augmented physical inputs, four target axes, square-root
normalized original weights and fixed1e-12 cutoff. Full rank13, condition and
effective condition26.578161878679158, coefficient norm2.8606365667032234.
Global weighted raw half-MSE0.02534324623126824; bounded0.02469080428985079.
Unweighted global raw0.026812068762864072; bounded0.026295835692691845.

All198rows/6groups remain:50current corrections,50original corrections,
18Push teacher,15Pick17,50derivedPick19supervision,15Pick20. Each group's raw
weight mass98; both correction initial rows retain49vs49remaining1s. This was
one cold batch fit, not chronological learning, recurrence, or a continuation
of the previous128-update neural learner. It minimizes **unclipped physical
MSE**, not that learner's mixed XYZ-MSE/gripper-BCE objective.

| Original group | Rows | Raw half-MSE | Bounded half-MSE | Historical neural128 reference |
| --- | ---: | ---: | ---: | ---: |
|current_corrections|50|0.003706238279|0.003518756032|0.02647018|
|corrections|50|0.020882833604|0.020772376544|0.03908475|
|push|18|0.012312125962|0.011843636205|0.01724678|
|a/Pick17|15|0.036935106471|0.035443936014|0.00912879|
|b/Pick19|50|0.055791024813|0.054948965205|0.03005605|
|train20|15|0.034275994390|0.033315072415|0.01962545|

These are within-group unweighted physical errors; do not substitute them for
the differently weighted global objective. All group weighted errors also remain
in LEAD-SCALARS.json. The old values are the accepted aggregated198 study's
reported common physical errors, rounded to its published precision. It was not
rerun. Source rows/weights match the original recipe; objectives, initialization,
capacity, previous exposure and total computation are not matched. The result
is mixed: Push/correction errors improve and all three Pick errors worsen. No
new correction data, dose/seed/weight search or favorable-group selection.

Prediction clipped coordinate counts(0,0,1,79): one z-coordinate and79gripper
values. These are coordinate counts, not necessarily80distinct rows. The map
has finite, well-conditioned coefficients, but fails to interpolate all targets.
This says nothing about nonlinear representation sufficiency, trajectory
stability, native acquisition/retention, reliability, safety or CPC advantage.

## Execution and cost

Physical working directory `D:\Dev\Repositories\Circadian`, process-local
TEMP/TMP/TMPDIR=`D:\Dev\_Temp`, CPU/BLAS1, existing NumPy2.4.6. Actual command:

```powershell
./artifacts/runs/continuous-affine-feasibility-20261101/run-fit.ps1
```

Exactly once: host PID79416 exit0/child0/noexpiry,0.6621634s. Observer including
output binding0.5416558s, sampled process peak66240512B. Source data5opens
(2accepted-study+3original-correction),10919961returned bytes; original public
loader's two reads are source-derived, the other three have explicit read-work
records. Existing byte/role/ID/history qualification and final file guards pass.
Both source SHA bindings unchanged. No historical model array was reconstructed
or executed. No source or caller RNG changed.

Numerical counts:1fit/return/198rows,1LAPACK/return/198rows,1prediction/return/
198rows,1matrix multiply;7metric calls/396rowvisits (global plus all six groups).
Two fitted-model wrappers(original+prediction validation),9enumerated array-copy
calls/6664float64 values(53312logical bytes),0forbidden work. These are selected
copy/work proxies, not all allocations/FLOPs. App's three input arrays26928B;
model52coefficients+13singular values520logical float64 bytes, excluding Python,
spec and metadata. JSON/episode/caller/guard/solver temporaries are additional.
RSS includes imports/data parsing/work, excludes the parent/process tree.

The app's original single-fit60s/512MiB budget was born before reading. One
attempt slot was charged before solving, observed slot1; original and observed
limits/start time match. This slot is an analytic fit attempt, not SGD. Outer
90s/512MiB, permanent75s/480MiB and parent105s held; no stop or unknown work.
Host stage34118B/log1139B at completion, later bounded metadata under8MiB;
actual11682B under1MiB. Host storage polling/final check/reserve4KiB is reused
from accepted397, not a new preservation framework.

Actual SHA256
`4d42e4dafe86713cb81221bc9b6867e692e270e2de8e8ef7418e92b4ad90fa3a`.
All three new result-byte passes(producer,lead,peer) are spent; do not reread the
raw file or refit this question. Use LEAD-SCALARS.json. The same lead pass wrote
the new `FITTED-MODEL.json`,2066B SHA256
`e07dd3c0e8783079afa5b4998c0ef7316ac00c2adba6c8a02b383088c24e0f4f`.
It exactly matches the fitted map, remains native/deployment-unqualified and
permits a later separately admitted frozen test without another fit. Thirteen
direct source/host/producer/protocol pins matched afterward; no whole-dependency,
archive authentication or durable checkpoint claim.

## Focused checks, failures and review

From the physical repository root, recorded launcher prefix
`./artifacts/runs/continuous-affine-control-20261101/run-check.ps1`:

| Exact arguments | Observed outcome | Body seconds | Process peak B |
| --- | --- | ---: | ---: |
|`-Attempt initial`|51PASS/1FAIL|1.8962684|80478208|
|`-Attempt source-capture -Selection 'exclusively or source_replacement or unreturned_reader'`|3affectedPASS|1.2697314|74166272|
|`-Attempt legacy-128`|1PASS,0solves|0.7254021|57303040|
|`-Attempt observer-literals`|3actual-helper literalsPASS|0.5932459|47259648|

56distinct now passing,59executions58PASS/1preservedFAIL/0SKIP; aggregate body
4.4846478s, no expiry/optional runtime imports, within180cases/120s/60s-per-body/
512MiB. Initial failure found a real report omission: replacing the outer report
dropped `source_bytes_unchanged=False` on final-source verification failure. Fixed
production default, not test criteria. Source-mutation/lost-fit/lost-charge,
nonTRAIN-role/exact-type/budget-clock/recipe/default128/ownership/rank/clipping
and analytical full-rank/weighted-mean cases qualify. Initial fixture genuine
LS work is source-derived21delegations/533rows, not retained final WORK totals;
live fixture caps64delegations/1024rows held. Later selections did no genuine LS.

Scoped final commands: `python -B -m ruff format --check` and `ruff check` on
the two production files, three new test modules and fit_observe.py;
`mypy --follow-imports=silent --cache-dir <software-stage>/type-cache` on the two
production files/three tests. All0/0.6713586s, host PowerShell parse0. Earlier
one heterogeneous-test-kwargs typing error is preserved/fixed with an explicit
annotation; no ignores or weakened numerical tolerances. Dynamic observer is
lint/syntax/literal/source-reviewed, not fully typed. Software stage/cache remain
under64MiB/32MiB. This is not repository-wide typing or a clean-clone gate.

Lead reviewed the actual source/procedure/results. Independent source ACCEPT,
PRE GO and POST ACCEPT: full new core/app, selected dependency functions, complete
observer/protocol/host and one new actual report read. Five compact batches,
reported0.775730s observed shell, no project imports/tests/fits/old inputs/edits/
native work or numerical replication. PRE looked for three check receipts in
the wrong stage; missing-path errors are preserved. Correct receipts were then
lead-forwarded from the software stage; those three were not independently file
read. POST checked rank/condition, coefficient norm, group/global metric and clip
aggregation(maxdifference1.39e-17), roles/rows/budget/counts/source IO/hash/pins/
storage/RNG and exported model equality. It did not recompute198residuals from
old inputs. Review allowance exhausted and finite exception ENDED.

Full suite/Linux/WSL/clean clone/wheel/browser/device/robot/GPU/CI/release skipped;
no install/download/Git publication/deletion/global setting changes. Accepted
PickCube/probes/consistency/old neural and MiniGrid controls were reused. All
failed/spent/negative results and unrelated dirty work remain unchanged.

## Still-open task and exact next action

Keep398open: core fit/prediction and TRAIN feasibility are complete; native
read-only policy admission and physical usefulness are not. Existing robot
rollout admission requires exactBC/PC/nearest owners. It cannot yet consume this
affine map, and no old learner snapshot may be fabricated to bypass it.

Next implement one explicit immutable affine reader through the EXISTING frozen
rollout admission/state comparison/task-space-history path. Preserve all default
families and zero update authority, use the cached accepted NEW model, and execute
focused port/budget/source/role tests. Then freeze a separate single native
qualification on ALL fixed Push17/Pick17/Pick19/Pick20 TRAIN endpoints, with no
additional fit or original source reread. No performance threshold/selection
rule is invented from this result. Parent/fullgoal/science/release remain OPEN.
