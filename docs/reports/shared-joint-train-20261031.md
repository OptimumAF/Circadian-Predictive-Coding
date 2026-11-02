# Ordinary shared-head joint TRAIN replay — 2026-10-31

R5.3-LIN.SHARED / [#362](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/362).
The existing ordinary single-head stream now exposes explicit
`--replay coverage --replay-joint joint|current_only`. Current arrived TRAIN and
distinct retained TRAIN can fit together through existing consolidation. Cold
and minimum-change choices remain independent; omitted options preserve original
no-replay/sequential/contextual modes. No new solver, runner, bank or promotion.

Previous turn PROGRESS: #361 saved inspection accepted; #360 private fixed case
reused without another run. Ordinary shared sequential replay already exists;
#354 partial/SPENT old-task retention/new-invoice failure is preserved. Generic
joint replay required routed `ContextLinearFitSnapshot` and same-context packets;
shared mixed-context packet algebra existed only in the private probe route.
That concrete gap justified an explicit ordinary branch, not a repeated study.
Full goal and broader R2/R4/R5/R7/scientific/release requirements remain OPEN.

## Implementation and executed behavior

- `core/shared_linear_replay.py`: small exact frozen shared geometry/control
  policy, existing packet algebra unchanged.
- `app/context_joint_rehearsal.py`: explicit shared selection/preparation, using
  original certified coverage inputs, validating all selected TRAIN packets and
  removing exact whole-packet duplicates already covered by current wake.
- `app/stream_rehearsal.py`: narrow shared admission, original paid-prefix,
  budget, continuation and consolidation ports; bounded current bank identity/
  value pin; proposal applied/unknown readings separate from returns/publication.
- `app/coupled_attribution.py`: explicit replay consent is allowed with this exact
  shared TRAIN-only helper and required at its source birth. Other helper consent
  restrictions remain. Original consent cannot change before more work.
- `scripts/ticket_workflow_rehearsal.py` and `recurring_ticket_workflow_example.py`:
  API/CLI/default/refusal/intent binding and original fixed sequence progress.
- `tests/test_shared_joint_train.py`: 41 new cases; 38 on both platforms plus
  three Linux-only contextual-joint/continuation regressions. Existing51affected
  checks reused/executed. README/ARCHITECTURE/usage describe actual scope.

Why coverage: two-update cadence plus a two-episode recent bank can evict A
before a B replay opportunity. Existing declared-context coverage keeps the most
recent released packet of each of two keys. No outcome/loss/metric/seed selection
was added. Same-row conflicts remain refused by existing algebra. No final,
probe_test, monitor or private probe_train data enters ordinary retention.

The fixed source remains12wakefits72rows/30s512MiB, replay starts with72charged
matrix rows/30s512MiB before work. Two early no-distinct events skip with no fake
receipt. Four later events fit12rows each: current6+retained6, or current6twice in
the control. Expected complete work16fits120rows includes repeated current rows;
that is not matched all-in computation. These complete sequences were SCRIPTED,
with deliberately false task results; no ordinary joint learning success is
claimed. Tiny genuine one/two-row cases establish ports/application/charges and
unchanged source/actor, not new-task acquisition or scientific benefit.

Actual corruption checks refuse changed prior event values/replacements, bank/
array replacements, charged counters and chosen intent before more fits. Lost
proposal returns retain12rowcharge; observed proposal application and unreadable
application remain distinct from the uncommitted candidate. Failed captures do
not invent final source-preservation flags; tests compare actual owned snapshots.
Shared pause/resume keeps original clocks and the bounded bank/paid prefix.
Existing contextual joint controls still run. No unlimited bank or success chain:
ordinary sequence has two retained episodes/six predeclared events plus one bank
pin, and existing generic total-update/retention bounds remain.

## Exact commands and outcomes

Physical cwd `D:\Dev\Repositories\Circadian`, HEAD
d9119e80926a139a2f38460564d7eab1b6d2a2b0 plus preserved inherited dirty work.
All shells set TEMP/TMP `D:\Dev\_Temp`; Linux additionally TMPDIR
`/mnt/d/Dev/_Temp` and OMP/MKL/OpenBLAS threads1. Earlier Windows numeric-thread
settings were not explicitly pinned: exact thread conformance is unverified.
Do not infer a matched timing/CPU result. Stage
`artifacts/runs/shared-joint-train-20261031/` retains preimages, initial/final
diffs, checker/statics, raw logs and receipts. No C: project outputs.

```powershell
$env:TEMP='D:\Dev\_Temp'; $env:TMP='D:\Dev\_Temp'
.venv/Scripts/python.exe -B artifacts/runs/shared-joint-train-20261031/check.py windows D:/Dev/_Temp/circadian-shared-joint-win-20261031
.venv/Scripts/python.exe -B artifacts/runs/shared-joint-train-20261031/check.py windows-repair D:/Dev/_Temp/circadian-shared-joint-repair-20261031 tests/test_shared_joint_train.py
.venv/Scripts/python.exe -B artifacts/runs/shared-joint-train-20261031/check.py windows-final D:/Dev/_Temp/circadian-shared-joint-final-20261031
.venv/Scripts/python.exe -B artifacts/runs/shared-joint-train-20261031/check.py windows-affected D:/Dev/_Temp/circadian-shared-joint-affected-20261031 tests/test_shared_joint_train.py::test_scripted_ordinary_capture_has_skips_then_four_paid_shared_consolidations tests/test_shared_joint_train.py::test_scripted_lost_proposal_preserves_charge_and_application_scope
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/shared-joint-train-20261031/check.py linux-final /mnt/d/Dev/_Temp/circadian-shared-joint-linux-20261031
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/shared-joint-train-20261031/check.py linux-integration /mnt/d/Dev/_Temp/circadian-shared-joint-integration-20261031 tests/test_shared_joint_train.py::test_linux_integration_preserves_existing_contextual_joint_route tests/test_shared_joint_train.py::test_linux_integration_shared_continuation_preserves_bank_and_paid_prefix
.venv/Scripts/python.exe -B artifacts/runs/shared-joint-train-20261031/statics.py statics-initial.json
.venv/Scripts/python.exe -B artifacts/runs/shared-joint-train-20261031/statics.py statics-final.json
.venv/Scripts/python.exe -B artifacts/runs/shared-joint-train-20261031/statics.py statics-integration.json
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
.venv/Scripts/python.exe -B artifacts/runs/shared-joint-train-20261031/check.py windows-application-fix D:/Dev/_Temp/circadian-shared-application-win-20261031 tests/test_shared_joint_train.py::test_counter_only_return_is_unknown_and_cannot_publish
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/shared-joint-train-20261031/check.py linux-application-fix /mnt/d/Dev/_Temp/circadian-shared-application-linux-20261031 'tests/test_shared_joint_train.py::test_real_tiny_shared_fit_keeps_source_actor_and_original_authority[joint]' tests/test_shared_joint_train.py::test_scripted_ordinary_capture_has_skips_then_four_paid_shared_consolidations tests/test_shared_joint_train.py::test_counter_only_return_is_unknown_and_cannot_publish
.venv/Scripts/python.exe -B artifacts/runs/shared-joint-train-20261031/statics.py statics-application-fix.json
```

Exclusive destinations/budgets are consumed, not safe retry instructions. Raw
logs capture expected CLI refusals from passing regression assertions. Shell log
display commands can exit0 even when the underlying checker exits1; the checker
receipts/logged failures, not that display exit, determine outcomes.

| Invocation | Actual outcome | Seconds | Sampled RSS bytes |
|---|---|---:|---:|
| Windows initial |73pass/12fail|2.620470200|81,219,584|
| Windows repair |25pass/12fail; receipt write failed|body unknown; host≤2.0048673|unknown|
| Windows combined |82pass/6fail|3.257528700|79,937,536|
| Windows affected |6pass|1.566978600|71,950,336|
| Linux final |88pass|10.973022220|75,395,072|
| Linux integration |3pass|4.951987278|66,953,216|
| Windows application fix |1pass/scripted|0.975162000|70,889,472|
| Linux application fix |6pass|5.304488513|67,293,184|

Final qualification: Windows89distinct via82unchanged+6affected+1new counter-only
case; Linux92distinct (88fresh+3integration+1new counter-only; final six-case repair
also repeats five affected healthy cases). No skips/forbidden neural/physics imports. Seven-file Ruff/format/
default+Linux-target mypy PASS final; initial24typing diagnostics retained then
fixed (explicit arguments, collection/kwargs typing and readable test callback).
The final integration source change is test-only; production stayed fixed.

Failures: first explicit consent rejected every helper; repaired only the exact
new shared boundary, not by removing consent. Second scripted fixture used six
singular values for a12-row ten-column design; corrected the fixture, no numeric
production change. The one-off checker reused `name` while counting modules and
collided with its original receipt. Original receipt retained under
`test_shared_joint_train-checks.json`; repair raw stdout retained, exact work/RSS
unknown. Simplified to immutable run_name before further work. A conservative
reservation was charged before the next check:8LS16rows/32products64rowvisits/
8typedcalls/256scripted2048rows/96owners, NOT observed work or a refund.
Source paths bound that repair to two tiny cases (≤6solves8rows) and twelve
scripted cases (each at most16callbacks120rows, so192callbacks1440rows together);
reservation intentionally exceeds them. Exact sampledRSS remains unavailable, but
the final RSS/time/forbidden assertions passed before the exclusive write failed;
no exact value or complete thread conformance is inferred. Host2.0048673s includes
formatting/runner/log display, not a measured test-body duration.

Third run's six remaining failures were assertions: extra existing training_work
fields must be preserved, and failed runs lack final preservation flags. Corrected
tests assert all counts/additional fields and actual source/actor snapshots.
No production change for these assertions. Two patch anchors missed before editing; neither changed files.
All failures/costs/logs remain; no scientific allowance was renewed.

Agent-created engineering limits were amended prospectively in issue comments:
one extra Windows six-case SCRIPTED-only repair; owner cap512→640 justified by
measured171/full versus prior375inclreservation+planned171Linux+36affected=582.
Then three Linux-only integration cases were allowed within SAME64LS/384row cap.
No user-imposed limit was weakened. Later checks did not recreate a spent study.

Independent review found a P2: a counter-only proposal could retain its old6-row
model while claiming+1update/+12examples, pass snapshot history validation, and
be incorrectly classified applied/published. Initial CHANGES REQUIRED remains in
REVIEW.md. The repaired predicate validates original/post snapshots and requires
a non-None model with exact fitted-row evidence for the packet. Counter-only
returned work remains unknown/unpublished, with its12rowcharge paid. The new
scripted regression checks returned1/applied0/unknown1/committed0 and the candidate
staying at six wake updates. Prospectively allowed one Windows scripted fault and
six Linux affected cases consume SAME remaining quotas, not a renewal. One genuine
two-row joint case and all four scripted healthy operation/control combinations
also pass on the repaired tree. Counter/evidence consistency is not proof of
physical solver execution.

## Resources, review and next action

Across all eight invocations, observed plus conservative reservation:
62LS/78fitrows,140named affine products192rowvisits,20typed-logit calls,
1021scripted callbacks7214rows,633generic owner constructions. Observed physical
LS alone54/62rows; the remaining8/16 is reserved unknown repair work. Products
include32/64reserved, typed includes8reserved, scripted256/2048reserved, owners96
reserved. Total test body plus reserved host upper bound31.654504811s. Named counters can include delegated ports,
not distinct physical products or all wrapper-object constructions. Sampling
is not exact all-heap/FLOP/process-tree accounting. No genuine fit exceeded2rows;
no full ordinary native source/task/probe/comparison ran accidentally as a fixture.

Limits:64LS384rows,1024named products4096rowvisits,128typedcalls,1024scripted/
16384rows,640owners,60s512MiB/invocation/240saggregate; stage8MiB+cache64MiB.
Static25.768220600s/90s separate; static RSS unmeasured. Named counters do not measure every
validation, snapshot, temporary array or allocator operation. No random model
initialization, SGD/backprop/latent/native environment/final/promotion/GPU work.

Lead self-review includes actual combined source, source diffs, procedures,
failures/charges, unsupported/default routes and remaining scope. Independent
finite actual-source/evidence review ACCEPTED the scoped repaired software.
REVIEW.md preserves the initial P2/CHANGES REQUIRED and final resolution. Eight
readonly calls consumed; known measured subtotal0.9760410s plus unavailable third
receipt. Exact aggregate/verified60s timing conformance UNKNOWN; it was not refunded
or claimed inspected. Later bounded reads covered needed source. No reviewer
test/model/numeric/network/browser/child-agent execution. Scope is actual excerpts/
diffs/evidence, not exhaustive repository/all-test-body or numerical replication.
The finite exception ended. Lead updated stale totals.json with both final repair
receipts after reviewer feedback; original six-invocation summary is retained.
No additional review layer or gate.
Full suite/clean clone/current wheel/install/unchanged kernels/baseline cohorts/
new ordinary native learning/game/robot/physics/browser/GPU/final/remoteCI/release
skipped. Existing accepted scoped evidence is reused, not reconstructed; all
negative/partial/SPENT sources remain. Known inherited working files are preserved,
including the prior361 log-link typo corrected without changing other bytes.

One next substantive action, planned #363/allfiveunchecked/no numericalallowance: freeze and independently
review ONE small ordinary shared-head current_only→joint native protocol through
the existing command. Expected fixed12wake/4joint120rows per arm needs NEW matched
control; old108-row sequential replay cannot substitute. Pin source/selector/data/
seed/task/metric/directions and every source/replay clock, original consent, paid
skips/failures, diagnostics/teacher/prediction/copy/memory costs before either fresh
source. Validate preceding success before admitting second. No dose/weight/seed/
metric search, retry or source-authority renewal. Negative results are valid.
No experiment is authorized by this software increment or its fixtures.

Retained storage at closeout: stage excluding dedicated cache 488821B/8MiB; cache initially0→16798791B/64MiB. Owned/tracked whitespace checks pass (LF/CRLF advisories preserved). No commit/push/install/download/delete/global changes.

[Canonical scoped acceptance](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/362#issuecomment-6542393896); #362 CLOSED/fivecriteria checked. Next [#363](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/363) OPEN/fiveunchecked/no numericalallowance. Full goal ACTIVE.
