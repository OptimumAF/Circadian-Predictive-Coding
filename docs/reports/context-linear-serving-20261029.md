# R4/R5-CTX.LS.S — owned LS policies now serve through the live route

[Issue #288](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/288),
parents #18/#21/#23/#24/#33 OPEN. Previous goal turn PROGRESS:287driver admission
repair verified, comparative case FAILED/SPENT before fits. That case and its
UNKNOWN scientific endpoints remain unchanged. No repaired comparison or old
scientific payload/model reconstruction occurred here.

## Implemented and executed outcome

The existing CLI now forwards the explicitly selected LS method through its
owned source, child learner, promotion guard, frozen serving readout and rollback.
The previous exact Perceptron source/path remains the default. No new runner,
optimizer, source-admission framework, dependency or product mathematics.

```text
src/adapters/context_serving.py             shared guard utility and typed-read hook
src/adapters/context_linear_serving.py      explicit LS record/guard/serving bindings
scripts/context_head_rehearsal.py           strict serving-only owned LS source opt-in
scripts/context_head_live.py                matching LS child/live bindings and paid failures
scripts/recurring_ticket_workflow_example.py API/CLI admission and actual method forwarding
tests/test_context_linear_serving.py        real admission, atomic slots, failures, full CLI
tests/test_context_linear_heads.py          joint no longer an unsupported mode; invalid still refuses
```

[Usage](../context-linear-serving.md). Source qualification preserves the original
TRAIN12fits72rows, head8/48+4/24, schema, report/row owners, source successes,
budget/clock/sampler and caller guard. The explicit LS branch accepts only the
ordinary serving source; private/probe/final or absent fabricated histories do
not gain authority. Whole validated record/digest includes LS coefficients,
fit rank/singular values/residual evidence, policies and all counters. Guard
utility/action/resource behavior is shared with the unchanged default binding.

Each typed request reads one detached atomic actor slot. Native aggregate
work is bounded rather than an unlimited read-record chain. A lost read pays
routing/affine work and latches refusal. A fit failure retains its owned
TrainingUpdateFailed definition/work in the child report. A started fit without
a returned model interval is explicitly unknown for overlap timing.

Both full **actual CLI** joint executions pass. Each freshly generates and fits
the ordinary TRAIN source12/72, then uses the existing static paired recipe:
child6fits48rows, first four9row fits learning v1parcel/v2password with old TRAIN
reuse, then two6row old-return fits. Total18fits120rows per integration. This is
an implementation/usefulness case, not287's two-arm dynamic episode comparison.
No current-only comparator ran and no causal/equal-work ranking is supported.

At fixed tick10 the unchanged guard accepts the first child window: actor new
TRAIN guard utility2/3, candidate1, old guard1→1; complete LS record2318B Windows/
2338B Linux below unchanged4096B cap. Max observed candidate guard prediction
time.0008722s/.001804678s, below the unchanged.25s cap; sampled local measurements
are not a general latency guarantee. Slot generation1 exposes the candidate
with16 cumulative fits/108rows (12source+4new). All six post-publication ordinary
workflows complete in3steps, including the two new head/task pairings and four
old tasks. These are known TRAIN templates, not an unseen distribution.

The live timeline has24opportunities/78complete typed worker requests/platform,
observing generations0/1. They are typed requests, not78closed-loop sandbox
episodes. Separate post-publication workflows establish task completion. The
candidate later reaches18/120 after the two old-return fits; no second return
publication was attempted. Do not claim that the final candidate retains the
new tasks, that it was republished, or that a rejected return guard was tested.
The actor remains the earlier published slot until rollback. Existing rollback
restores exact source state at generation2; four old workflows still succeed
and both new head/task workflows fail. No old task was lost during the source
context return, so source recovery remains null, not a new recovery gain.

Source/deployed-source authority, original source charges and caller RNG remain
unchanged. Serving publication changes only the explicit child's actor slot.
No probe/final updates, promotion from final data, scientific CPC comparison,
generalization, dynamic long-stream retention or release claim.

## Tests, review and failures

23distinct cases/platform qualified via scoped reuse and affected checks:
19core/admission cases,2CLI dispatch cases,1synthetic owned-failure forwarding
case,1full actual CLI case. This is not one final23case cohort.18new cases plus
5selected existing refusal cases. All passes,0failure/error/skip. The core source
fixture pays12 real source fits72rows,8TRAIN teachers24steps and4ordinary source
workflows per platform. Synthetic injected failed-work fields are fixture data,
not extra measured fits or evidence of a genuine lost native call.

Coverage includes real lifetime source admission, original budget and schema,
wrong roles/success/state/definition, replaced rows, foreign limits, default
Perceptron refusal of LS state, explicit LS refusal of Perceptron state, default
typed predictor, detached record identity, guarded atomic publication/rollback,
lost read costs/fault latch and unsupported API modes. Existing proof/source
assumptions outside the affected boundary were reused rather than rerun.

Independent source/state/isolation review used3readonly calls/90s allowance and
found two pre-integration defects: CLI accepted LS but failed to forward the
method (silently Perceptron), and the child's broad error handler dropped owned
LS failure receipts. Fixed both before the full case, added focused tests, and
re-reviewed actual source. The full integration was changed from direct API to
the actual parser/dispatcher/writer. Unfinished model intervals are now unknown.
No imported models/tests/old raw/writes in review; the finite exception ended.
Lead source/procedure/results self-review is complete, not independent review.
Final independent result review is separately limited to1readonly call60s.

Earlier scoped statics caught incompatible state-copy alias typing, native-work
type narrowing, unused imports, missing actor metadata and wrong test imports.
Repaired before numerical tests. No pytest failure, budget increase, full-case
retry, guard relaxation or favorable run selection. Guard/resource thresholds
and exact static schedule were unchanged throughout both platform cases.

## All engineering costs, including core fixtures

Stage `artifacts/runs/context-linear-serving-20261029/` contains scoped observer,
Win/Linux1..4work receipts and `win32-integration.json`/`linux-integration.json`.
Stage2,893,613B at initial closeout, within8MiB. No captures were overwritten.

| Cost | Windows | Linux |
|---|---:|---:|
| pytest invocations |4|4|
| observed pytest-body seconds |13.2541171|33.197049765|
| sampled process peak RSS bytes |77,221,888|87,093,248|
| physical LS attempts / solves / rows |30 /30 /192|30 /30 /192|
| original TRAIN teacher traces / steps |24 /72|24 /72|
| closed-loop workflow episodes |40|40|
| total teacher+task sandbox dispatches |218|218|
| main-thread affine calls / rows |112 /112|112 /112|
| extra worker affine calls / rows |78 /78|78 /78|
| all LS affine calls / rows after reconciliation |190 /190|190 /190|
| unfitted typed reads |38|38|
| main-thread outer+selected typed delegates |300|300|
| additional worker typed delegates |156|156|
| all LS outer+selected typed delegates |456|456|
| full-class guard/head prediction calls / rows |16 /96|16 /96|
| main-thread homogeneous route calls / rows |181 /343|181 /343|
| additional worker route calls / rows |78 /78|78 /78|

Why the reconciliation: `sys.setprofile` observes the main thread only. It does
not observe the live worker's native calls. The owned serving-native receipt
records114typed/affine/routing returns:36post-publication/rollback main-thread
reads plus78complete worker requests, corroborated by timeline request rows.
Add those78affines/routes and156outer+selected delegates to the immutable
observer receipts; do not present the main-thread observer as whole-process
inference accounting. Fits, full-class guard predictions and sandbox dispatches
occurred on the main thread; worker requests do not dispatch the environment.

The pure default-preservation check additionally serves one deterministic zero
Perceptron request: one outer+selected pair and one affine read, no fit/random
initialization. It is outside the LS-specific affine/delegate observer fields
(and its homogeneous route is already counted). Thus all-family delegate calls
are458 and affine reads191, not456/190. The LS totals remain as tabulated.

All native rows/fits/teachers/dispatch/time/RSS and reconciled inference totals
remain within the original per-platform limits; no additional work allowance.
Homogeneous route rows exclude full/mixed guard projections. Their features
also incur projection/validation/copy work; the source-inspected row accounting
adds96nonuniform head-prediction rows plus12uniform actor-guard rows, for529total
context-projection rows (421homogeneous+108guard). This is a derived bound/count,
not a separately instrumented full projection or FLOP measurement. Guard copies,
state evidence/serialization and Python allocations also cost time/memory.

The serving receipt includes5472nominal affine terms/1218token visits for114typed
calls/platform, counting full-context plus selected-head tokenization. Logical
coefficient bytes768 differ from full LS records2318/2338B and process RSS.
The observer body includes all pytest calls but excludes observer JSON output/
process launch. Full CLI case command wall8.8777678s Windows; Linux exec initially
yielded a live session, then the same handle completed successfully. No rerun
followed an observation yield. Process peak is sampled, not an exact peak proof.
Torch/neural/GPU/random model draws/probe/final/old scientific raw reads0.

Windows reports6worker request intervals intersecting model calls; Linux reports0.
No forced rendezvous or parallel CPU/speedup/general-overlap claim. Both have
78complete requests; scheduling differs and is retained rather than repeated.
Native coefficient digests differ by platform floating-point fit details; no
cross-platform bitwise snapshot equality is claimed. Task outcomes agree.

## Commands and skipped checks

Physical cwd `D:/Dev/Repositories/Circadian`, Windows TEMP/TMP=`D:/Dev/_Temp`;
Linux TEMP/TMP/TMPDIR=`/mnt/d/Dev/_Temp`. Python-B. Existing Windows Python3.14.7,
Linux Python3.12.3/NumPy1.26.4. Observer receipt argv preserves every exact call.

Initial scoped test selection, then the2pureCLI and1purefailure selectors:

```text
.venv/Scripts/python.exe -B artifacts/runs/context-linear-serving-20261029/check.py artifacts/runs/context-linear-serving-20261029/win-1-work.json tests/test_context_linear_serving.py tests/test_context_linear_heads.py::test_should_preserve_unsupported_separate_linear_modes_before_data -k "not full_existing_cli" --basetemp=D:/Dev/_Temp/circadian-ls-serving-win1
```

Final actual CLI integration, one/platform, exit0/1PASS:

```text
.venv/Scripts/python.exe -B artifacts/runs/context-linear-serving-20261029/check.py artifacts/runs/context-linear-serving-20261029/win-4-work.json tests/test_context_linear_serving.py::test_should_execute_full_existing_cli_owned_linear_live_route --basetemp=D:/Dev/_Temp/circadian-ls-serving-win4
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/context-linear-serving-20261029/check.py artifacts/runs/context-linear-serving-20261029/linux-4-work.json --capture=sys tests/test_context_linear_serving.py::test_should_execute_full_existing_cli_owned_linear_live_route --basetemp=/mnt/d/Dev/_Temp/circadian-ls-serving-linux4
```

Seven-file `python -B -m ruff check`, `ruff format --check`,
`mypy --explicit-package-bases --follow-imports=silent` (default and--platform
linux) pass. `git diff --check` final whitespace check. Changed files listed
above. No new dependencies/install/download/delete/global changes/commit/push.
HEAD0aae1d6 and unrelated dirty270–287 preserved. Existing legacy test file
restored in286 remains untouched. Old failed scientific captures unchanged.

Skipped broad unchanged math/native/BP-PC-CPC baselines, old raw-backed default
serving studies, full repository/clean clone/current installed wheel/remote CI/
GPU/physics/release checks. Selected new default Perceptron checks passed;
scoped checks do not establish whole-repository preservation or release readiness.
#279's installed wheel predates this implementation. Full LS `current` CLI is
unexecuted; only joint receives this complete integration qualification.

## Remaining full-goal work

Implemented code, executed scoped usefulness and source acceptance are separate
from scientific conclusions. Reliable autonomous adaptation, unforeseen-context
generalization, broader R2.1 retention/recovery, native CPC benefit, complete
continuation/persistence and release remain open.287comparison endpoints remain
UNKNOWN and its case permanently spent. No full-goal completion is claimed.

Exact next action: inspect the existing arrived-TRAIN shift observer and its
calibration/incident contract against this now-working LS serving route; identify
the smallest arrival-only trigger composition needed to replace the fixed
periodic schedule. Keep task readouts/final data out of trigger calibration and
decisions, preserve prior negative shift evidence, and implement/refusal-test
the dependency before any new adaptive comparison. No new model/study allowance
is granted by this serving closeout; do not repeat287or add a capture/count sequel.

GitHub remains live tracking; requested plan/log entries are archival evidence
pointers only, preserving original criteria and all unfinished parent tasks.


Final independent result review accepted the scoped interpretation and visible
receipt fields, including worker/main reconciliation, unpublished final candidate,
rollback and null recovery. One readonly call; no execution/writes. Its bounded
output truncated later episode details, so this is not independent episode-by-
episode rescoring or replication. No blocking correction; exception ended.
