# #409 / R2.1-CTRL.FIT — bounded full-network control

Implemented an explicit conventional BP full-MLP GGN update through the existing
continuous learner. Its numerical/state behavior passes the final 37-case tiny
cohort. This is software correctness evidence, not useful native control or a
circadian advantage. Full repository/scientific/release requirements remain open.

## Contract and source

`src/core/continuous_full_gauss_newton.py` owns the full logit Jacobian, unchanged
original objective/gradient, normalized active-axis regression and optional
logit2z CE curvature, positive row masses, damped solve, norm cap and one strict
descent trial. `src/adapters/numpy_continuous_bc.py` adds explicit TRAIN-only
episode/raw-array entries and a result DTO. It pins native source owners and
values, commits atomically, resets all momentum on acceptance and reports paid
failed/rejected work without publication. See [API and cost semantics](../bp-full-network-fit.md).

No duplicate full-network equivalent was found. Existing head GN freezes hidden
features, allows130rows and excludes mixed loss; its limits/default SGD are not
changed. `default-preservation.json` compares the exact before/current AST of20
existing definitions/methods: all unchanged. The only old-file edits are imports,
the new DTO and new methods. No runner, dependency or release authority added.

Reuse #407/#408 cached evidence, the accepted198-row grouped consistency and
B22/B50 diagnostics. Full220 consistency is not inferred from those subsets;
no label-consistency prerequisite exists for this pure kernel. A future actual
fit must resolve any concrete contradiction with existing consistency code.
#408 remains FAILED/SPENT/PARTIAL: its observations were computed, but phase
serialization and final file postchecks did not complete. No rerun here. Existing
neural/affine native failures and the negative reserved24 result remain intact.

## Executed checks

All project commands used physical `D:\Dev\Repositories\Circadian`, TEMP/TMP/
TMPDIR `D:\Dev\_Temp`, bytecode disabled and numerical CPU threads1.
The frozen pre-test contract and runner are in
`artifacts/runs/full-network-fit-control-20261101/`.

Exact numerical commands (prefix `.venv\Scripts\python.exe -I -B`):

1. `artifacts/runs/full-network-fit-control-20261101/check.py initial --tb=short`:
   27PASS/3FAIL,0.981510200s,67,665,920B peak. Failures were fixture construction:
   reserved policies already rejected by the old constructor; episode vectors
   contained disallowed NumPy scalars. No numerical criterion was weakened.
2. `.../check.py corrective --tb=short -k 'no_nontrain_authority or episode_entrypoint'`:
   2PASS/2FAIL/26deselected,0.758337500s,62,201,856B. The role refusal occurred,
   but the fixture's uppercase-only message assertion was wrong. Match both
   existing constructor and new TRAIN boundary messages; refusal remains required.
3. `.../check.py final --tb=short`: **37PASS**,0.780565200s,64,876,544B.

Total71 executions:66PASS/5fixtureFAIL;2.520412900s, maximum67,665,920B.
All3command slots are permanently SPENT. JSON/log/SPENT records retain failures.
Cumulative actual toy work:452forwards/1,348rows,111original-gradient requests/
329rows,28solves(P<=64),52linearizations,44proposal calls,25adapter calls/75rows,
7accepted publications,28literal source owners,8public snapshots/2,304known array
payloadB and1,728finite-difference coordinates. Snapshot payload is selective;
internal validation/copies/temporaries/Python costs additional. One literal
H64/one-row linearization per full cohort, no H64 solve. Zero initializers/RNG,
real datasets/exports, SDK/native/teacher/probe/final work. All frozen limits held.

The final cases cover full Jacobian finite differences and directional original
gradients for MSE/mixed/weighted branches; CEfactor2 and saturated wrong signs;
constant axes/all-constant rejection; huge weight rescaling; strict original-loss
descent and hidden-weight change; no-improvement rejection; thrown, rank-invalid,
shape-invalid and nonfinite solve results; returned nonfinite objective ACK;
normalization exactly once; non-TRAIN policies/episodes; actor-fork preservation;
parameter/policy/optimizer-container/counter/effective-scale-owner corruption;
and malformed failure receipts retaining a typed failure with zero publication.

Self-review and the independent source review both identified malformed numerical
failure metadata escaping the typed paid envelope. Protected merge handling now
retains unknown partial work, rechecks source and raises TrainingUpdateFailed.
One later mypy error for effective-scale subtype access was corrected by explicit
subtype narrowing. These were engineering failures, not scientific results.

Final scoped Ruff/format, mypy, AST preservation and diff/whitespace results are
recorded in `final-statics.log` and `default-preservation.json`. Independent
source-review receipt and final verdict are retained separately in this stage.
Finite independent actual-source/math review ACCEPTED, no blocking findings,
4read-only batches/1.6s reported; exception ENDED. It independently compared18
preexisting functions/methods by AST (lead's20 includes class definitions).
No independent test execution/replication. Lead combined source/procedure/results
self-review complete; this is separate from the peer's scoped verdict.

## Limitations and next outcome

No real H64/220 fit, input/export reread, native rollout, Linux numerical cohort,
full suite, clean-clone/wheel/install, remote CI, GPU/device/browser or release
check was run: this increment changes one explicit CPU math/state operation.
Old numerical/scientific/native allowances remain spent; this new tiny cohort
does not renew them. No seed/metric/threshold/baseline tuning or comparative claim.
Inherited unrelated dirty work and HEADd9119e8 are preserved; no commit/push,
install/download, deletion or global configuration change.

Next useful outcome: freeze and review one small actual conventional full-network
TRAIN fit/control qualification through the existing native runner, using the
qualified checkpoint/bundle under distinct input-use/computation allowances.
Resolve full220 exact consistency with the existing core if needed; preserve
original groups, metrics, seed17 and all prior controls. No sweep or reserved24
selection. Measure both original TRAIN objective/command behavior and genuine
known-task native success, plus the full added cost. A failed control remains a
valid result. Do not start another capture/provenance increment.
