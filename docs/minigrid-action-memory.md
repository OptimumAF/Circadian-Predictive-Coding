# MiniGrid bounded action-memory control

## Optional finite-capacity FIFO control

The existing recurring runner also accepts `--invariant-fifo-memory`:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario minigrid-action-memory --invariant-fifo-memory --describe
```

This option is mutually exclusive with `--invariant-memory`. Both original raw
and invariant refusal defaults/previews remain unchanged. The new route composes
the existing invariant wrapper and FIFO backend: 36 entries, 87,264 logical table
bytes plus 16 bytes of retirement metadata. Hits retain insertion order; a new
key at capacity retires the oldest key and all its counts. This is a fixed
two-task capacity control, not a learned retention rule or a parameter sweep.

The same original six TRAIN traces, delayed A32/B32/C_train32/A32 stream, 128
updates, five checkpoints, 30 scores and 30 known games remain. A distinct typed
evaluator admits only FIFO36 and the original TRAIN inputs. A private explicit
training-arm option selects the 36-entry counter; the default counter still
requires the original 64-entry recipe. Raw, refusal and FIFO64 states cannot be
reinterpreted as the new control. Final/probe/replay and promotion remain excluded.

Each complete checkpoint records retained counts, retired examples and evictions;
retained plus retired exposure must equal lifetime training rows. Table bytes,
16-byte retirement metadata, copies and process memory are separate costs.
On failure, observed native commits and receipted charges remain distinct. For
example, a lost fourth return can leave four native commits/36 observed rows but
only three paid receipts/27 rows. Unavailable native readback stays unknown; its
clock cannot be inferred from the receipts. Existing paid-prefix and closing
source/budget guards still apply, with bounded report storage.

Why this: the accepted original TRAIN traces have 54 disjoint canonical keys,
18 per task. A predeclared 36-entry FIFO can expose finite-capacity loss and task
return through the existing learner and stream. Synthetic tests exercise that
behavior; they establish neither native game skill nor a circadian advantage.
Any native case needs its own frozen protocol and independent review. Preserve
all earlier failed grants; they do not authorize another run.

117 focused checks, scoped statics and independent source review pass. The single
native case is INCOMPLETE/SPENT at180 seconds, with all128updates and five typed
captures preserved. A succeeds at32 and remains successful at64. Independent
saved-state arithmetic verifies A's counts are retired at96 and relearned at128;
those later live readouts were not run. Full B/C acquisition, live return and
30-game acceptance remain open. No retry.
[Commands, costs, failures and partial evidence](../artifacts/runs/minigrid-recurring-fifo-20261017/RESULTS.md).

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest tests/test_minigrid_recurring_fifo_evaluator.py tests/test_minigrid_recurring_fifo_pilot.py tests/test_minigrid_recurring_fifo_cli.py -q
```

Extend this control through explicit typed admission and the existing stream;
keep queue order, retirement accounting and native/receipt clocks separate.

## Optional invariant recurring-task control

The existing runner also accepts the explicit `--invariant-memory` option:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario minigrid-action-memory --invariant-memory --describe
```

The preview allocates no learner or environment. Omit `--describe` only for a
separately budgeted execution. The flag is valid only for this scenario; omitting
it preserves the raw count recipe and CLI dispatch. The existing bound invariant
learner receives the same public initial/current observations, fixed seed17,
128 delayed A32/B32/C_train32/A32 updates and five original checkpoints. Its
existing public projection and count law are unchanged, with64keys/155136 logical
bytes and uniform-seven misses. This is a finite supervised control, not SGD.

`evaluate_invariant_memory_checkpoint` admits the complete bound policy, encoding
and native state before using the existing TRAIN evaluator. It accepts only the
original six teacher traces and three TRAIN layouts at0/32/64/96/128, and checks
the original untrained source around every external port. Raw/FIFO states cannot
be relabeled as invariant. Thirty scores and thirty live games occur after all
training; final/probe/replay data and promotion remain excluded.

The new route privately retains paid training and returned readouts before later
callbacks. A missing admitted evaluator receipt produces a failed/unknown marker
and lower bounds from earlier genuine reports. Closing sampler and clock ports
remain inside source/budget/evidence proofs. Five bounded private readouts and
typed checkpoints replace no original receipt, consent, budget or charge.
Returned canonical score/live arrays are counted separately; attempted allocations,
full copies, Python storage and process costs are additional or explicitly unknown.

Why this: reuse the working recurring stream to measure whether the public
representation can acquire successive tasks and preserve a returning task,
without introducing another orchestration layer or changing neural controls.
Acquisition still requires both teacher sequences exact and both live successes
at A32/B64/C96. Without those premises, forgetting/recovery remain inconclusive.
Even a successful known-TRAIN case does not establish generalization, reward
learning, equal-compute superiority or a circadian benefit.

Focused verification and all preserved failures are in the
[recurring invariant results](../artifacts/runs/minigrid-recurring-invariant-20261017/RESULTS.md).
The sole native case is FAILED/SPENT: a late wrapper deadline left its JSON empty,
so actual learner doses and outcomes are unknown. There is no retry. The two prior
incomplete A/F attempts and their permanent charges remain unchanged.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest tests/test_minigrid_recurring_memory_evaluator.py tests/test_minigrid_recurring_memory_pilot.py tests/test_minigrid_recurring_memory_cli.py -q
```

For extension, keep source admission in the adapter and compose an existing learner
through the current stream. Add a distinct frozen protocol before changing tasks,
exposures or evaluation roles; do not silently broaden this recipe's acceptance.

This opt-in recipe uses the existing MiniGrid source collector, public first/current
encoder, delayed AttributionStream and frozen policy evaluator. It supplies a
simple supervised count control for the actual game. Execution evidence is kept
in the [frozen local stage](../artifacts/runs/minigrid-action-memory-live-20261017/PROTOCOL.md).

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario minigrid-action-memory --describe
```

Preview constructs no learner, game or process sampler. Actual execution additionally
requires the existing `requirements-minigrid.txt` environment. Use the stage's
reviewed supervisor for its one prospectively budgeted case; a previous SPENT run
does not authorize a retry. Ordinary CLI execution is:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario minigrid-action-memory
```

## Boundaries and frozen behavior

```text
src/core/action_memory.py                       pure geometry/count/payload rules
src/adapters/action_memory_learner.py            owned native snapshots and forks
src/adapters/minigrid_action_memory_pilot.py     fixed control composition
src/adapters/minigrid_learning_pilot.py          existing source/training arm/guards
src/adapters/minigrid_policy_evaluation.py       frozen raw TRAIN/live evaluation
tests/test_minigrid_action_memory_evaluation.py  admission/isolation/paid failures
tests/test_minigrid_action_memory_pilot.py       real count/stream plus fake physics
```

The explicit296-feature/seven-action policy keeps64 exact keys within155136 logical
bytes,2424 per entry. This is an explicit allowance for larger public observations;
old16/5 recipes retain their10752B cap. Counts preserve conflicting labels; misses
return uniform seven. No masking, nearest-neighbor repair, eviction or hidden task
identifier is used. Storage updates are not SGD and this capacity is not matched
to neural parameters or computation. Logical bytes exclude Python objects,
episode/array/fork/report copies and process overhead measured through wall/RSS.

Six original TRAIN teacher traces provide the fixed seed17 A/B/C/A schedule,
32 updates per phase and checkpoints0/32/64/96/128. The stream delays labels by
one tick and keeps serving/source state unchanged. All128 updates/1152 rows finish
before30 teacher-forced TRAIN scores/270 rows and30 known TRAIN live episodes.
Live decisions use raw seven-action first argmax,16-step bounds and per-episode
public history reset. No final D/probe/replay/selection/promotion is admitted.

Original budgets/clock identity, permanent charges, receipts, source arrays,
checkpoints and prior readout identity/content remain guarded. Exactly five
immutable private readouts are retained during execution, then detached for
publication. Failed native reads keep known committed receipts/rows separately
from unavailable current state; failed live calls retain attempted/returned work.

Why this: the296/7 input capability now exists. Reusing the existing actual-game
ports tests a simple control before further mechanisms or neural tuning. Known
TRAIN recall does not establish unseen-layout generalization, recurrent reward
learning, causal recovery, a deployment benefit or circadian advantage. All
negative/inconclusive outcomes and acquisition premises remain visible.

## Checks and extension

The one reviewed local case is COMPLETE/SPENT:148final focused checks pass, scoped
statics and independent SOURCE/GO/ACTUAL review pass. A/B/C are acquired at32/64/96;
A remains successful through B/C and its return.54keys use130896 logical bytes.
The case costs128count updates/1152rows,270raw score rows,354live decisions and
408teacher+live physics steps,124.47supervisor seconds. External peak112226304B;
internal sampler peak117739520B.12cold/unseen live timeouts remain recorded.
No measured forgetting/recovery deficit, generalization or circadian advantage.
[Exact checks, failures, all outcomes and costs](../artifacts/runs/minigrid-action-memory-live-20261017/RESULTS.md).

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest tests/test_minigrid_action_memory_evaluation.py tests/test_minigrid_action_memory_pilot.py -q
.venv/Scripts/python.exe -m ruff check src/adapters/minigrid_action_memory_pilot.py src/adapters/minigrid_learning_pilot.py src/adapters/minigrid_policy_evaluation.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/minigrid_action_memory_pilot.py src/adapters/minigrid_learning_pilot.py src/adapters/minigrid_policy_evaluation.py
.venv/Scripts/python.exe -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/minigrid_action_memory_pilot.py src/adapters/minigrid_learning_pilot.py src/adapters/minigrid_policy_evaluation.py
```

Also run scoped linux-target typing; it does not establish Linux runtime support.
Engineering tests use explicit array fixtures/fake physics, with real count updates
and original stream/budget guards. Their success alone is not game skill evidence.
Freeze a new small protocol before changing doses, layouts, seeds, metrics, table
limits or evaluation roles. Extend existing ports instead of adding another runner.

## Isolated saved-state E16 capacity assay

The new adapter/script use the accepted0/end128 full snapshots through ordinary
bounded JSON and typed restore. No benchmark rerun or deployment update occurs.
Separate disposable forks retain the original64-key/155136B cap and16-update
budgets; the existing reserved MiniGrid collector/history/fresh-task helper supplies
E-blue/red train/test episodes and fixed relative0/4/8/16 readouts. No final data,
replay, live evaluation, promotion, baseline tuning or task/seed selection occurs.

```text
src/adapters/minigrid_action_memory_probes.py   fixed isolated assay composition
scripts/run_minigrid_memory_probe.py           saved snapshot loading and CLI
tests/test_minigrid_action_memory_probes.py    native probe/failure behavior
tests/test_minigrid_memory_probe_cli.py        bounded IO/preview/publication
```

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_memory_probe --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_memory_probe --source-artifact artifacts/runs/minigrid-action-memory-live-20261017/native.json --output NEW_RESULT.json
```

Use a new output path and a prospectively budgeted case. Existing SPENT evidence
never authorizes retry. Optional output refuses overwrite; incomplete probe exit2
is a result to preserve. The sole reviewed native case is incomplete: cold copy
fits E by relative4 and ends18keys; mature54-key copy commits one9-key batch,
reaches63, then refuses the next batch atomically. Mature later checkpoints/gain
are absent.17storage commits/153rows,10scores/90rows,36teacher physics steps and
3.053176supervisor seconds were charged. This demonstrates a capacity limit, not
neural plasticity loss, unseen-task generalization or CPC advantage.

[Protocol, exact checks, failures, costs and remaining task](../artifacts/runs/minigrid-action-memory-probes-20261017/RESULTS.md).
Use the four changed paths in that command list for scoped Ruff/format/both mypy
targets; final173focused cases pass. The existing generic helper/core learner and
all prior matched neural protocols remain unchanged. A future replacement control
must have distinct identity, deterministic documented policy and explicit retired
exposure, preserving this refusal control and the same capacity budget.

## Opt-in matched new-task and return comparison

The existing saved-state script also offers `--return-comparison`. It restores
accepted initial0/end128 states, explicitly converts owned FIFO states, and gives
both controls the same16 planned updates: eight E packets alternating blue/red,
then eight A packets from owned original TRAIN A-blue/red copies. Original A DTOs remain
recorded with their TRAIN role/source IDs. Separate helper-role copies support
known-task retention readouts and returning-task updates; A is not heldout data.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_memory_probe --return-comparison --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_memory_probe --return-comparison --source-artifact artifacts/runs/minigrid-action-memory-live-20261017/native.json --output NEW_RETURN_RESULT.json
```

The second command requires a prospectively reviewed local case. No prior SPENT
allowance or successful component test authorizes a new experiment. The original
E16 command, descriptor and refusal negative stay intact. This mode is exploratory
because E and A were already inspected; it is not independent confirmation.

Four fixed cohorts0/refusal,0/FIFO,128/refusal,128/FIFO each have a separate16-update
budget/fork. Relative0/4/8/12/16 checkpoints report E and known TRAIN A CE/accuracy
and decision diagnostics. New E gain is CE0−CE8; old A retention loss CE8−CE0;
A return gain CE8−CE16; E retention loss CE16−CE8. Missing endpoints have no gain.
FIFO adds16 logical retirement metadata bytes above the common retained-table
allowance; report retired exposure/evictions and process overhead separately.
Planned conditions match; actual paid dose and compute may differ after refusal.

A validated atomic exact-memory capacity refusal keeps failed status and may
precede the next independent cohort only while its native state still equals the
preceding observed paid state and original source/budget/report bindings hold.
Same-counter state changes, replaced/corrupt prior evidence and unknown costs stop
later admission. Any failed cohort keeps the overall report incomplete/CLI2,
including when the remaining independent cohorts execute. No budget is renewed.
Flat preceding checks avoid recursive checking; every native snapshot return is
revalidated by the original typed core before full-state comparison.

[Prospective protocol and engineering failures](../artifacts/runs/minigrid-memory-return-20261017/PROTOCOL.md).
The sole reviewed case has executed with a preserved mature exact-memory failure:
193focused checks plus12 type-only closing checks/scopedstatics pass. Mature FIFO
learns E, loses some A accuracy (1→11/18), then relearns A to1 while E stays correct.
Its26evictions retire704exposures and leave8originalkeys absent; B/C was not scored.
The overall result stays incomplete/CLI2. This is teacher-forced fitting, not live
policy success. [All results/costs/failures](../artifacts/runs/minigrid-memory-return-20261017/RESULTS.md).
This mode uses the existing adapter/helper and FIFO native learner; no second
orchestration framework, final evaluation, serving update, replay or tuning.

## Frozen saved-return game readout

The separate `evaluate_memory_return_checkpoint` entry point in the existing
MiniGrid evaluator accepts the actual saved initial/end states from that assay.
It runs four target-free games: known TRAIN A blue/red and the corresponding
E-blue/blue and E-red/red probe_test games. Public first/current history resets
per game; actions are raw first argmax over all seven inputs. No teacher, labels,
storage update, replay, final D or action repair is available on this route.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_memory_readout --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_memory_readout --source-artifact artifacts/runs/minigrid-action-memory-live-20261017/native.json --return-artifact artifacts/runs/minigrid-memory-return-20261017/native.json --output NEW_LIVE_READOUT.json
```

The run is a separately budgeted local case. Both source reports are read once
with the existing bounded JSON/typed admission; no original training budget is
restored. Four cohorts each read their initial then saved end state, eight
readouts total. The mature exact source remains failed even if its frozen game
readout completes. New game execution failure stops later work, preserves its
paid or unknown prefix and leaves planned slots explicitly not_run. Prior reports
and future owned input records are checked across callbacks; at most eight are
retained. The original evaluator's default admission remains unchanged.

There is no saved intermediate checkpoint8 native state. Initial/end games cannot
establish live recovery from a measured intermediate deficit. Known A/inspected E
success is exploratory memorization, not heldout generalization, independent
confirmation, neural plasticity or a circadian advantage. The frozen small
[protocol](../artifacts/runs/minigrid-memory-return-live-20261017/PROTOCOL.md)
limits new work to32 resets/512 predictions/steps and zero training updates.

The sole reviewed run recorded all eight readouts with32resets/379decisions/steps,
zero new training/raw-score forwards, and independent saved arithmetic max error0.
Cold exact/FIFO A/E successes0/0→2/2; mature failed exact prefix2/0→2/1; mature
FIFO2/0→2/2. Every success takes9steps/return.49375; negative games truncate at16.
The native qualification remains FAILED/SPENT: both processes0/outputcomplete,
supervisor1 for239B runpy warning on stderr. Executed child and negative gate stay
preserved. A separate fixed launcher calls imported main once; stdlib stub passes,
but no native retry ran. [Exact costs/failures/limits](../artifacts/runs/minigrid-memory-return-live-20261017/RESULTS.md).
