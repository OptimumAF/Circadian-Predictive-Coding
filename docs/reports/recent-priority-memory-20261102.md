# Optional recent-priority nearest memory — #425 / R2.3-MEM.RECENT

The existing bounded learner now has an explicit recent-priority policy. At total
capacity128,64recent distinct arrived IDs are protected on overflow; the worst
seeded-priority ID outside that window is evicted. Duplicate retained content
refreshes its order only in this mode. No original learner, deployed actor or
benchmark state is adopted from a private probe.

Why this: #424's fixed TRAIN/known-TRAIN observations showed FIFO forgetting and
reservoir rejecting useful return rows. The symmetric half split was fixed before
new outcomes, without readonly24 design input. This implements a conventional
control, not a circadian algorithm or selected winner. Existing controls, failures
and negative Push results remain frozen. No original inputs or simulator were
opened in this software qualification.

## Architecture and semantics

`RecentPriorityRetentionPolicy(seed)` is nearest-only. Its native integer uint64
seed is mandatory; capacity2..256, R=floor(capacity/2), P=capacity-R. `NearestRetentionPolicy`
is an exact-type union at nearest admissions. Generic `ReplayRetentionPolicy`
retains its old names/default and shared/CPC replay rejects the new type. Only the
pure SHA primitive is shared, with identical domain/seed/content-ID bytes.
Existing binary replay partitions do not own continuous-action pairs, so reusing
that buffer would alter data/identity contracts. No new runner/dependency exists.

The current ordered bank encodes recency without a history ledger. Every overflow
ranks capacity+1-R eligible IDs and removes the largest (rank,ID). The bank contains
the latest R distinct arrivals and the global best P priority IDs. These sets can
overlap: spare entries are history-dependent, not the best nonrecent complement
from all discarded history. Reappearing evicted IDs count as fresh current-bank
arrivals; duplicate counts describe retained-ID matches, not lifetime uniqueness.
Original content IDs/feature/target bytes are retained, including on duplicate
refresh. Published snapshots and source forks remain detached.

Exposure remains retained+duplicate+discarded. Logical rank evaluations equal
`discarded * (capacity+1-R)` (65 per discard at128); old reservoir129/FIFO0
formulas remain. Actual validation re-selection/hashing/copies are extra work,
not included in the logical count; fixture receipts separately instrument rank,
selected copying, hashing and prediction work. Current numeric/hex payload stays
capacity bounded; Python/RSS and transient validation cost are separate.

## Executed checks

Physical checkout D:/Dev/Repositories/Circadian, HEAD d9119e80926a139a2f38460564d7eab1b6d2a2b0,
inherited dirty tree preserved. Protocol and before copies were written before
model work at `artifacts/runs/recent-priority-memory-20261102/`. Lead alone wrote
code and ran literals; source-pins.json binds final7production files+test.

Exact bounded launch commands (PowerShell, process TEMP/TMP/TMPDIR on D:, threads1):

```powershell
& artifacts/runs/recent-priority-memory-20261102/run-checks.ps1 -Slot first
& artifacts/runs/recent-priority-memory-20261102/run-checks.ps1 -Slot second
```

Each wrapper runs `.venv/Scripts/python.exe -B -m pytest -q --tb=short tests/test_recent_priority_memory.py --basetemp D:/Dev/_Temp/circadian425-SLOT`.
First37PASS/exit0/7.4307763s; second40PASS/exit0/8.0994212s. 77case executions,
40distinct new cases. No test failures/skips; two remaining slots unspent and not
needed. Initial scoped static checks found a missing production import, then a
wrong test class/import and annotations; corrected before numerical execution.
A first textual import correction missed its formatted line; simplified direct
replacement fixed it. Harmless guessed source-path misses were replaced with
symbol search. No model retries or changed acceptance criteria resulted.

Final commands use these8files: replay_retention.py, nearest_retention.py,
numpy_retained_nearest_bc.py, continuous_stream_sequence.py, taskspace_memory_demo.py,
taskspace_memory_native.py, run_taskspace_memory_demo.py, test_recent_priority_memory.py
(full paths and stdout in statics.json):

```text
.venv/Scripts/python.exe -B -m ruff check FILES
.venv/Scripts/python.exe -B -m ruff format --check FILES
.venv/Scripts/python.exe -B -m mypy --platform win32 --follow-imports=silent FILES
.venv/Scripts/python.exe -B -m mypy --platform linux --follow-imports=silent FILES
```

All4 final checks exit0; Linux here is mypy platform analysis, not Linux runtime.
Final40 cases cover independent history membership oracle/cap2/odd capacities,
golden old hash bytes/ties/duplicate order, chunk continuity, atomic rank failure,
source/published snapshot/fork isolation, invalid seed/cap/role/policy admissions,
corrupted counters/owners, real literal D12 private acquisition+old return,
corrupted private endpoint refusal with permanent paid attempt, default/CPU/native
CLI metadata dispatch. Native dispatch is mocked and never opens an SDK.

Cumulative two slots:14nearest initializers/50attempted appends/140incoming rows,
80forks/74predictions/484query rows/1456candidates/17472coordinate upper bound,
9286snapshots/70350selected validators/18060960selected-copyB/21482968selected-content-hashB,
480residualrows/10fake endpoint calls;194pure selections/452IDarrivals/
2260candidate upper bound/990rank calls/63360rank-payloadB incl independent oracle.
Max process peak81477632B; wrapper sampled-child peak is incomplete and is not
used as process peak. All counters within original prospective aggregate limits.
Native/physics/gradient/replay/final/model-RNG counts0. Full exact counters in
second-counters.json; host time sums15.5301975s plus setup/statics/source review.

Independent changed-source/math/state review accepted seven production changes
and initial37 source/execution. Three readonly batches approximately1.048s,
no replication/models/SDK/data/writes; exception ended. Later3test additions and
final40 receipts were lead self-reviewed, not independently observed. REVIEW.md
records that exact distinction. Final combined source/pins/receipts self-reviewed.

## Qualification and next action

#425's software scope is complete. No actual new control arm, native success,
return improvement, reliability, CPC advantage, generalization, clean clone/wheel,
GPU, broad suite, remote CI or release qualification is established. Old numerical
cohorts and raw captures were not reopened; original423/424 failed attempts and
mixed outcomes remain. Parents18/21/23 and full roadmap remain open.

Next: freeze and review ONE recent_priority128/64recent/seed17 arm through the
existing adaptation command, same7TRAIN packets+18new+15old-return and five fixed
final scenes/metrics as423/424. Reuse accepted control scalars; no ratio, seed,
capacity, metric or readonly24 selection, no new sweep or default change. Separate
actual budget/prerequisites required before original-data/native work.
