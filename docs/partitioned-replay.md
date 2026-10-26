# Partitioned replay retention

`SharedReplayBuffer` can combine a refreshed recent FIFO with a seeded bottom-k
reservoir of distinct arrived labeled content. Defaults and the existing single-
policy behavior remain unchanged.

```python
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.replay_retention import ReplayRetentionPolicy
from src.core.shared_replay_schedule import ReplayAllocation, SharedReplayBuffer

buffer = SharedReplayBuffer(
    2,
    ReplayRetentionBudget(4, 96),
    ReplayRetentionPolicy("seeded_reservoir", 53),
    allocation=ReplayAllocation(
        ReplayRetentionBudget(2, 48),  # recent
        ReplayRetentionBudget(2, 48),  # reservoir
    ),
)
# Observe only arrived training arrays, then propose a detached common batch.
# buffer.observe_train_batch(inputs, targets)
# selection = buffer.select_partitioned(2)
# inputs_for_one_method = selection.training_batches()
```

Each partition has its own positive row/array-byte budgets. Their sums must fit
the existing global caps, and each must hold at least one float64 labeled row.
For width d, a stored row costs `8*(d+1)` bytes. Overlap pays both partition
quotas but stores one array pair per unique union ID. `partition_retention`
reports recent/reservoir/unique snapshots plus overlap; `retention` remains the
unique total. Python dictionaries, IDs and temporary/returned copies are outside
these array-byte figures. No unlimited seen-ID history is retained.

Duplicate arrival refreshes recency; the reservoir ranks distinct labeled
content using the existing seeded rank, independently of repeated exposure.
New label content is a new ID. No context or future label is consulted. Bottom-k
sampling does not guarantee every old context survives in finite memory.

`select_partitioned(n)` alternates recent newest-first and reservoir ascending
rank/ID, starting recent. Each turn scans past already selected IDs until its
next distinct row; an exhausted partition falls back to the other. For recent
`[A,B]` and reservoir `[A,C]`, selection is `A,C,B`. All returned row arrays are
detached, and each method requests its own copies. `select_recent` retains its
original union-order behavior and does not automatically mix partitions.

## Existing schedule and CLI

```powershell
.venv/Scripts/python.exe -m scripts.run_continual_matched_replay_schedule_smoke --partitioned --result schedule.json
```

The existing train-only schedule accepts `allocation=...`. Its new protocol and
configuration digest bind the base manifest, allocations and interleaved sampler;
default v9 manifest/digest/schema remain unchanged. The CLI uses existing seeds
17/19, reservoir seed53, two recent and two reservoir slots, A then B with two
epochs each. It writes exclusively and refuses occupied output before source
work. Every method receives the same proposed IDs and detached rows, with work
plans separated from executed work. It accesses arrived training roles only.

This command runs a schedule, **not model training or final evaluation**. The
legacy native matched runner checks equality with CPC's separate native buffer,
which currently supports single-policy retention. That check is preserved;
native partitioned replay integration is unfinished. No learning benefit,
equal-process-memory, context guarantee or scientific advantage follows from
retention/selection correctness.

The separate [coherent stream rehearsal](stream-rehearsal.md) integration trains
owned BP/PC/neutral CPC/active CPC forks through the existing consolidation hook.
It does not waive the legacy native/shared-buffer equality check described here.

### Optional bounded reservoir rotation

```powershell
.venv/Scripts/python.exe -m scripts.run_continual_matched_replay_schedule_smoke --partitioned --rotate-reservoir --result rotating-schedule.json
```

`select_partitioned_rotating(n)` is a separate core API. It stores only a frozen
`ReplayRotationCursor(next_partition, reservoir_after)`:0 means recent next,
1 reservoir next, and the optional last-scanned ID determines the next strict
rank/ID position with one wrap. The requested turn persists across calls,
including one-row requests. Each reservoir scan advances past duplicates that
were already proposed via recent in the same call. Detached result arrays must
be constructed successfully before cursor publication; invalid/empty/failed
selections do not spend a position. Existing selectors and observations do not
advance or reset this cursor.

For a **stable reservoir of R rows**, every member is proposed within at most2R
successful nonempty positive-budget calls. This is a reservoir exposure bound;
it does not cover changing membership or older recent-only rows, and cannot
guarantee old-context presence. One last-scanned ID remains bounded even after
its eviction. Trusted in-memory deepcopy/pickle preserves continuation; this is
not a new durable loader or cross-version authority guarantee.

The schedule opts in with `rotate_reservoir=True` and an allocation. New protocol
`continual_rotating_replay_schedule_v1` binds sampler
`recent_reservoir_rotating_interleave_v1`; old default/v1 identities and payloads
remain unchanged. The rotating CLI adds its post-selection cursor per boundary.
It still proposes work only, with no native training or final evaluation.

## Structure and checks

```text
src/core/shared_replay_schedule.py             retention, quotas, detached selection
src/app/continual_matched_replay_schedule.py   arrived roles and common work plans
scripts/run_continual_matched_replay_schedule_smoke.py   exclusive JSON boundary
tests/test_partitioned_replay_buffer.py
tests/test_partitioned_replay_schedule.py
```

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_partitioned_replay_buffer.py tests/test_partitioned_replay_schedule.py tests/test_shared_replay_schedule.py tests/test_continual_matched_replay_schedule.py
```

The last two modules include existing default/native parity checks. Prospective
local budgets, exact static/test commands and actual CLI evidence are in
`artifacts/runs/r23-partitioned-replay-20261016/`. Full R2.3 remains open for
learning integration and broader retention evidence. Any comparative learning
run needs a separate fixed protocol with replay exposure/memory/work costs.

## Executed evidence and limitation

The combined tree passes82 focused cases/246 phases and scoped Ruff/format plus
Windows/Linux mypy. Independent source/protocol review preceded one actual
partitioned CLI run. That run observed8 batches/88 training-row exposures,
produced8 boundaries and16 proposed rows per method, and executed zero optimizer
updates or final evaluations. The two repeated epochs in each phase count as
repeated observed exposure;40 generated rows per source are not all training
rows. The actual trace uses the original seeds17/19 and reservoir seed53.

| Seed | Phase | Unique stored rows | Array bytes | Overlap | End-A IDs retained | End-A IDs selected per boundary |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 17 | A (both epochs) | 3 | 72 | 1 | 3 | 2 |
| 17 | B (both epochs) | 3 | 72 | 1 | 1 | 0 |
| 19 | A (both epochs) | 4 | 96 | 0 | 4 | 2 |
| 19 | B (both epochs) | 4 | 96 | 0 | 2 | 1 |

These are intersections with the retained pool at the end of A. Seed17 shows a
useful negative: an old row survives in the reservoir but receives zero proposed
replay during B because each reservoir turn starts at the same lowest rank.
Retention does not imply exposure or learning. No seed, sampler or result was
changed to conceal this; full R2.3 remains open.

The opt-in rotation follow-up above is now implemented and separately qualified.
In its sole frozen CLI trace, retention/base settings match v1 exactly. Seed17
still proposes no old row at the first B boundary but proposes the retained old
row at the second; seed19 proposes both distinct retained old rows across its
two B boundaries. The original rank-first negative remains unchanged. This is
proposal coverage with16 rows per method and zero executed optimizer updates,
not learning benefit. Existing legacy CPC equality checks remain in force.

Qualification:127 current nodes across128 executions (initial126 pass/one root
test failure, separate corrected pure case pass), scoped static checks and
independent review. The failed root test assumed deepcopy of an entire schedule
whose sealed role metadata has mapping proxies; only the buffer promises copy/
pickle continuation, and those tests passed. Production stayed unchanged during
the correction. The format-check failure/repair and all failed receipts remain.

**Next implementation:** opt-in shared rehearsal on the existing coherent
four-arm attribution stream using its actor/shadow consolidation hook and
explicit replay budgets/receipts. Keep original no-replay behavior, owned forks,
final scoring after all training, and separate wake/replay/exposure/work costs.
The hook already transforms detached candidate state; use it instead of another
runner or weakening the legacy CPC native-buffer gate. Freeze/review a small
matched BP/PC/neutral CPC/active CPC protocol before comparative execution.

Exact commands, resources, failures and actual artifacts:
`artifacts/runs/r23-partitioned-replay-20261016/{SCOPE,RESULTS,REVIEW}.md`.
Rotation follow-up:
`artifacts/runs/r23-rotating-replay-20261016/{SCOPE,RESULTS,REVIEW}.md`.
