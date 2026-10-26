# Native corrective adaptation of the saved public21 MLP

## Accepted observed outcome, 2026-10-19

The single case completed within budget. Collection TRAIN19 and corrected
TRAIN17/19 each reached50 controls with return0/no success/truncation and no
action-bound violations. All six corrected-minus-uncorrected differences are0.
Eight recorded pre-update losses decreased, but final fit was not diagnosed and
native adaptation was not demonstrated. Exactly eight new updates/400 rows and
50 provisional labels were consumed; original source96/2816 remained exact.
50tests perplatform, scoped statics and independent actual acceptance pass.
All prior negative results remain. Next is a separately frozen matched public21
control using the existing mixed-gripper objective, with no baseline retraining.

This local case asks whether a fixed corrective intervention changes native
task outcomes after the accepted 96-update public21 model failed both known
TRAIN contexts. It uses the existing corrective pipeline and preserves the
original source model. Scientific outcome and final acceptance are recorded in
[RESULTS.md](../artifacts/runs/public-vector-corrective-native-20261019/RESULTS.md).

## Fixed composition

Restore the saved eight parameter/momentum arrays, preserving 96 updates,
2,816 training rows, seed17, hidden64, learning rate.01, momentum.9, default MSE
and A-only frozen statistics. The accepted loader rechecks the two TRAIN source
diagnostics against all42 saved scalar values before simulator creation.

Read the two accepted uncorrected p32 traces as references, then call the existing
`run_corrective_adaptation` with the explicit public projection:

1. Collect one source-policy TRAIN19 episode, preserving student controls and
   raw789 provisional geometry labels.
2. Apply exactly eight private updates to the arrived, internally derived21D
   correction view. Continue all source optimizer state without refitting.
3. Run the frozen candidate on TRAIN17, then TRAIN19.

All six corrected-minus-uncorrected sparse-return, any-success and final-success
differences are retained; positive is better. These are same-task TRAIN reset
contexts. Added exposure and geometry queries prevent an equal-compute or
query-necessity interpretation. No heldout, new-task, CPC or generalization claim
follows from this case. Earlier negative results remain unchanged.

## Source and evidence boundaries

```text
artifacts/runs/public-vector-corrective-native-20261019/
  native_case.py   existing loader/pipeline plus validation at stage boundaries
  native_guard.py  accepted observer with stage/module/cache binding only
  run_check.py     existing serial bounded command recorder
  PROTOCOL.md     frozen source, seeds, metrics, gates and resource limits
  REVIEW.md       independent source, gate and actual-result review
  RESULTS.md      exact commands, outcomes, failures, costs and next action
  adaptation/     collection, update, both frozen traces and pipeline summary
tests/test_public_vector_corrective_native.py  synthetic composition checks
```

Why this: the accepted correction implementation closes the raw/model input
gap, but its synthetic checks cannot establish native utility. This case composes
the existing runner rather than adding a new collection or learning framework.

Temporary wrappers only validate calls made by that pipeline. Returned work is
retained before assertions. Raw collection and labels are bound to the returned
batch; original clocks and all eight receipt/charge/work values validate. Prior
saved traces are reread after owner getters, including immediately before the
existing updater admits SGD. Retained paid update evidence validates before
candidate controls. Both original callables are restored in `finally`.

Failed paid progress is recorded from consistent native integer counters and
receipt counts. Inconsistent or unavailable progress remains unknown; it is not
reported as zero. Original reports and partial traces remain available. No
candidate archive or serving publication is created.

## Commands and budget

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m pytest -q tests/test_public_vector_corrective_native.py
.\.venv\Scripts\python.exe -X utf8 -B -m ruff check artifacts/runs/public-vector-corrective-native-20261019/native_case.py artifacts/runs/public-vector-corrective-native-20261019/native_guard.py artifacts/runs/public-vector-corrective-native-20261019/run_check.py tests/test_public_vector_corrective_native.py
.\.venv\Scripts\python.exe -X utf8 -B -m ruff format --check artifacts/runs/public-vector-corrective-native-20261019/native_case.py artifacts/runs/public-vector-corrective-native-20261019/native_guard.py artifacts/runs/public-vector-corrective-native-20261019/run_check.py tests/test_public_vector_corrective_native.py
.\.venv\Scripts\python.exe -X utf8 -B -m mypy artifacts/runs/public-vector-corrective-native-20261019/native_case.py
```

These focused tests use synthetic snapshots, receipts and simulator boundaries;
they perform no SGD or fitting. Unchanged correction276/source55/depth72 and
independent numerical proofs are reused. Exact Windows/Linux IDs, measured
constructor/restore/forward work and scoped checks belong to the evidence record.

The separately admitted actual command uses the pinned Linux stack, one CPU and
software Vulkan, with observer90s/launcher105s and no retry. Maximum new work is
150 controls,750 known control-physics steps,153 RGB observations,50 geometry
queries and8 SGD updates over400 training rows. Source diagnostic, projection,
normalization, repeated exposure and numeric memory costs are reported separately.

The observer enforces2GiB observed high-water sum including final process SELF,
64MiB cache,8MiB log and16MiB stage payload limits. Its sampled process tree plus
final SELF is not a whole-lifetime tree proof. Three traces are bounded to4MiB
each, update/summary to256KiB each and audit/outer summary to64KiB each. Hidden
SDK reset work, other physical constraints, copies and process overhead remain
unmeasured. No downloads, sweeps, baseline reruns, guard/probe/final access or
promotion are admitted. A failure or negative outcome remains valid evidence.

Future work must use these returned losses, native trajectories and costs to
choose a concrete learner dependency. Do not automatically repeat another count,
seed, capture or audit increment.
