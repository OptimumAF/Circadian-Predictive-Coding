# Fixed old-A TRAIN rehearsal

## Explicit width selection

`wide_neural=False` preserves the original H8 policy. ExplicitTrue admits the
otherwise identical original H32 four-method native recipe; other policies and
non-native bool values refuse before replay charges. It still uses the same
TRAIN bank, existing stream and separate permanent replay allowance. See the
[public H32 composition](minigrid-probe-rehearsal.md#explicit-h32-composition).

`build_minigrid_old_a_rehearsal` composes the original public invariant BP, PC,
neutral CPC or active CPC learner with the existing `StreamRehearsal` and
`AttributionStream`. It admits two distinct complete nine-step A TRAIN episodes
with raw296 observations and seven actions. Current and next observations must
contain valid public frames and the same causal initial frame throughout.
The default native recipe remains H8/seed17; explicit H32 preserves its objective and hyperparameters.

The generic helper now accepts `fixed_training_episodes=()`. A nonempty bank
requires recent episode replay, complete TRAIN records and enough retention and
event capacity for **the entire bank**. Construction stores history without a
learner update or invented wake receipt. Incoming wake episodes are still
validated; they do not enter this fixed bank. Every configured event replays both
episodes in their original order through existing candidate consolidation.
Empty input preserves the previous arrived-only retention and report.

## Use with the existing stream

```python
from src.adapters.minigrid_fixed_rehearsal import build_minigrid_old_a_rehearsal
from src.app.coupled_attribution import AttributionStream

# learner: original InvariantTorchFeatureLearner; old_a: complete TRAIN tuple.
# wake_budget and replay_budget: distinct fresh ToyBudgetSessions with finite caps.
# training: complete TRAIN batches, not reserved probe/evaluation data relabeled.
plan = build_minigrid_old_a_rehearsal(
    learner, wake_budget, old_a,
    replay_budget=replay_budget,
    total_updates=len(training),
    every_updates=1,
)
stream = AttributionStream(
    learner, training, (),
    checkpoint_updates=(0, len(training)), label_delay=1,
    budget=wake_budget, rehearsal=plan.rehearsal, training_only=True,
)
stream.run_until(len(training))
report = stream.finish()
```

The composition adds no runner or public CLI flag. It does not accept
`probe_train`, `probe_test` or `final_test` as replay sources, convert roles,
evaluate a policy, choose a bank from scores or promote a learner. A caller must
bind the exact historical sources and collection/acquisition costs in any later
scientific protocol. Valid A metadata does not authenticate historical acquisition
or establish membership in the frozen original blue/red study cohort.

## Work and storage

- Bank:18 native transitions, two encoded episodes, **87,280 bytes** of retained
  float64 payload. This includes IDs/context headers, both observations, actions,
  rewards, terminal flags and dummy targets. No second full bank or success chain
  is retained by the fixed-bank authority.
- Public admission:36 current/next projection rows, **85,248 bytes** of projected
  array equivalent, exposed on `MiniGridRehearsalPlan`. This is validation work,
  not native training; raw transport remains raw and each native call projects once.
- Each event:two native updates/18 attempted row exposures. Original wake charges
  remain separate from the birth-bound replay budget. Attempts are charged before
  native calls; failed later episodes cannot publish partial replay work or refund
  earlier attempts. Reports distinguish attempted, returned and committed updates.
- `fixed_bank` report metadata identifies the stored episode IDs and payload.
  Existing content IDs plus bounded owners/configuration detect corrupted or
  replaced evidence before more wake/replay work. Ordinary same-owner pause/resume
  preserves original budgets, clocks, completed events and permanent charges.
- Python metadata, temporary validation/encoding/selection copies, event episode
  records, native snapshots and continuation copies are **not** included in the
  retained payload number. Existing total-updates caps bound event count; process
  RSS covers these other allocations. This is not a durable storage-byte bound.

## Executed verification

263 distinct scoped cases pass:251 fake/app regressions,8 public admission cases
and4 native methods. The native cases execute24updates/216rows in total, with
zero predictions, games, teachers or reserved probes. They verify original source,
actor, inputs and RNG remain unchanged, original budget/clock bindings and ordinary
same-owner resume. Four-path Ruff/format and two-source Windows/Linux mypy pass;
independent source/protocol/actual review accepts the scope. See
[exact commands and receipts](../artifacts/runs/minigrid-fixed-train-rehearsal-20261018/RESULTS.md).

Runnable check commands from the repository root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m pytest -q tests/test_fixed_training_stream_rehearsal.py tests/test_minigrid_fixed_rehearsal.py
.\.venv\Scripts\python.exe -m ruff check src/app/stream_rehearsal.py src/adapters/minigrid_fixed_rehearsal.py tests/test_fixed_training_stream_rehearsal.py tests/test_minigrid_fixed_rehearsal.py
.\.venv\Scripts\python.exe -m ruff format --check src/app/stream_rehearsal.py src/adapters/minigrid_fixed_rehearsal.py tests/test_fixed_training_stream_rehearsal.py tests/test_minigrid_fixed_rehearsal.py
.\.venv\Scripts\python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/app/stream_rehearsal.py src/adapters/minigrid_fixed_rehearsal.py
```

This establishes replay functionality, not a retention or circadian benefit.
Old A remains unacquired in the preserved scientific results. Next connect this
helper to the existing isolated F/A-return route, keeping reserved probe authority
at its current boundary. Before comparative execution freeze the exact old pair,
seeds, readouts and no-replay/matched-extra-exposure control and all costs.

## Structure and extension

```text
src/app/stream_rehearsal.py                 fixed bank and existing consolidation
src/adapters/minigrid_fixed_rehearsal.py    public admission and composition costs
tests/test_fixed_training_stream_rehearsal.py  generic behavior and failure guards
tests/test_minigrid_fixed_rehearsal.py      public schema and four native methods
```

Add a future CLI option at the existing MiniGrid route; reuse this factory and
existing stream. Do not relax TRAIN role checks or relabel reserved cohorts to
connect the isolated fresh-task probe. See [ADR0428](adr/ADR-0428-fixed-old-train-rehearsal.md).

## Later integration

The callable isolated F/replay/A-return integration is now implemented and verified
for all four original native methods. See [usage and evidence](isolated-probe-rehearsal.md).
The main MiniGrid pilot/CLI composition and frozen matched comparison remain open.
