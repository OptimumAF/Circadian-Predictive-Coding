# Isolated fresh-task rehearsal and old-task return

`run_fresh_task_probe(..., rehearsal_factory=...)` now drives its opt-in F updates
through the existing `AttributionStream` and fixed-bank `StreamRehearsal`. The
stream owns a disposable candidate. F batches, complete episodes, experiences and
labels retain `probe_train` throughout. Separately admitted old-A replay records
remain `train`; probe_test and final labels never reach an update.

The default probe still uses its original direct loop and report. The subsequent
[public MiniGrid pilot/CLI increment](minigrid-probe-rehearsal.md) exposes this API.

## Composition

Supply the existing F training/evaluation cohorts, original checkpoint, readonly
old-A TRAIN monitors, distinct fresh F/replay/return budgets and a fixed factory:

```python
from src.adapters.minigrid_fixed_rehearsal import build_minigrid_old_a_rehearsal
from src.adapters.minigrid_return_fit import return_training_sources
from src.app.fresh_task_probe import run_fresh_task_probe

plans = []
old_bank = tuple(packet.episode for packet in old_a_monitors)

def make_rehearsal(disposable, wake_budget, updates):
    plan = build_minigrid_old_a_rehearsal(
        disposable, wake_budget, old_bank,
        replay_budget=replay_budget, total_updates=updates,
        wake_role="probe_train",
    )
    plans.append(plan)  # admission projection costs remain available separately
    return plan.rehearsal

result = run_fresh_task_probe(
    source, checkpoint, f_training, f_evaluation,
    budget=f_budget, checkpoint_updates=(0, 4, 8, 16),
    training_readouts=old_a_monitors, decision_diagnostics=True,
    return_training=return_training_sources(old_a_monitors),
    return_budget=a_return_budget, rehearsal_factory=make_rehearsal,
)
```

For the original sixteen-update recipe, use exactly16 F update allowance,288
replay-row allowance (two9-step old-A episodes after each F wake) and16 A-return
updates. Keep all original wall/RSS limits, clock, progress and sampler objects.
The factory admits the original invariant four-method H8/seed17 recipe and complete
raw296/K7 causal histories; it does not authenticate prior acquisition or original
study membership. Freeze exact sources independently before a scientific run.

## Isolation and accounting

- `training_role="probe_train"` is an explicit birth-bound stream/runtime/inbox
  scope: at most16 complete episodes, exact fresh wake allowance, empty heldout
  data and no checkpoint/feature observers, capacity or renewal. A default TRAIN
  owner refuses probe events; probe owners refuse TRAIN events and final/heldout
  update roles. Probe experience permission cannot grant replay or evaluation.
- The existing stream owns wake receipts and consolidation. Its fixed TRAIN bank
  validates complete incoming F metadata without encoding, relabeling or retaining
  F in replay storage. Every event replays the whole old bank. Attempted rows are
  permanently charged before native calls, including a failed second episode.
- F counters remain16updates/144rows; replay has32native updates/288rows and its
  own budget; A return adds16updates/144rows. The same candidate and optimizer
  continue into A return after the F stream finishes; no further replay occurs.
  Source, actor and benchmark learner remain unchanged.
- `rehearsal` reports the bank, events, exposures and attempted/returned/committed
  replay. `probe_stream` reports native wake/consolidation receipts and zero final
  scoring. Complete selected episodes are not duplicated in both output branches.
  If rehearsal evidence cannot be read safely, the failed report says so and keeps
  the observed replay counter explicit; no zero work is inferred.
- Factory entry pins F budget authority before its callback and checks it before
  stream admission/scoring. Original replay limits/binding and observed charges
  are checked at protected boundaries, including return. Paid F evidence is frozen
  by the existing A-return guard; explicit Areturn0 must equal F16 scores.
- Ordinary same-owner probe pause/resume preserves scope, budget, receipts and
  fixed bank. Homogeneous probe cursors are data only; they do not authorize a
  default TRAIN owner to resume. Managed enrollment/capture remains TRAIN-only.
  Default TRAIN owners keep their exact historical field layout/capture bytes.

## Costs and observed verification

Each MiniGrid bank retains18native transitions in two encoded episodes:
**87,280 float64 payload bytes**. Public admission adds36 current/next projection
rows/**85,248 projected-array-equivalent bytes**, exposed on the returned plan.
Native projection still happens once per raw train/predict call. Existing
`reserved_data_bytes` counts original detached probe input arrays only: bank,
stream copies, selected-episode event reports, native/optimizer/checkpoint states,
Python objects and paid-report guard work are additional. Probe cap16 bounds event
count; RSS and invocation time cover those allocations/guards. This is no durable
storage-byte or performance guarantee.

Final combined regression check: **1,061 passed**, excluding14 already executed
native legacy cases. Four new native methods passed: **256updates/2,304rows and
128predictions/1,152rows**,44.1662465actual seconds, peakRSS355,770,368bytes.
Persisted per-case XML properties prove64updates/576rows/P32/288. Tests verify
F48-native→A64-native on the same candidate, source/actor/input/globalRNG isolation,
separate charges, original clocks and equality of F16/Areturn0 readouts. No game,
teacher or MiniGrid final evaluation was run. Scoped12-path Ruff/format and9-source
Windows/Linux mypy pass; independent source/protocol/execution GO recorded.

Initial7test failures, one later fixture failure,3capture teardown errors,2lint
and3typing errors remain preserved with their repairs. Legacy native regressions
were accidentally included before intended source review; that gate remains
failed. Their work was charged and the prospective engineering cap amended before
the new smoke, whose source review did pass. Including legacy work:390native
updates/2,548rows and260predictions/2,184rows, within revised400/3000/P280/2400.
Legacy toy final fixtures were scored only as regression oracles. No comparative
scientific run was performed. See [exact commands, failures and receipts](../artifacts/runs/isolated-probe-rehearsal-20261018/RESULTS.md).

Checks from repository root (set native thread environment variables to1):

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m pytest -q tests/test_probe_role_stream.py tests/test_rehearsed_fresh_task_probe.py
.\.venv\Scripts\python.exe -X utf8 -B -m pytest -q tests/test_minigrid_rehearsed_probe.py
```

Use `-o junit_family=legacy` when exporting new XML with `record_property`; the
executed run used xunit2 and emitted four warnings, while retaining all properties.
Do not rerun a model merely to remove those warnings. Full exact static commands
are in the linked results; no fullsuite/cleanclone/release claim is made.

## Structure and next step

```text
src/core/experience.py, inbox_cursor.py     typed probe roles and data-only cursors
src/app/experience_inbox.py, actor_shadow.py  bound update scope and owned candidate
src/app/coupled_attribution.py             existing stream, explicit probe scope
src/app/stream_rehearsal.py                fixed TRAIN bank, probe wake validation
src/app/fresh_task_probe.py                opt-in composition and same-candidate return
src/app/managed_experience.py              TRAIN-only managed admission
src/adapters/minigrid_fixed_rehearsal.py    raw public composition and costs
tests/test_probe_role_stream.py            transport/scope/continuation refusals
tests/test_rehearsed_fresh_task_probe.py    fake F/replay/return/failure routing
tests/test_minigrid_rehearsed_probe.py      four original native methods
```

The subsequent public CLI increment implements explicit selection, descriptions,
original replay budget binding and aggregate exposure/compute/memory accounting.
Next freeze/review exact old pair, fixed seed,
doses, readouts and matched extra-exposure controls before comparison. Old A is
still unacquired in preserved scientific results; this implementation proves no
retention/recovery or circadian advantage. See [ADR0429](adr/ADR-0429-preserve-probe-roles-in-stream-rehearsal.md).
