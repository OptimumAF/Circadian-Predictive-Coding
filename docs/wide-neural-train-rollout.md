# Width-32 known-TRAIN games

The saved-source fit CLI can play both original A cue games from its genuine
H32 checkpoint after all fixed TRAIN readouts. Why this: row accuracy does not
prove a policy can complete an episode. The four real H32 fits remain negative;
this feature observes actual target-free behavior through the existing evaluator.

## Use

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_linear_fit_control --wide-neural bp --train-a-rollout --fit-updates 128 --describe
```

Methods: `bp`, `pc`, `neutral_cpc`, `gated_cpc`. The option requires an explicit
128-pass H32 recipe; other families, cue-relative encoding and gradients refuse.
To execute after freezing/reviewing a small protocol, remove `--describe` and add
`--source TRAIN_REPORT --output NEW_RESULT.json --fit-seconds 60`. Outputs refuse
replacement. The input reader consumes complete original TRAIN records only.
No final or reserved probe data selects, trains, promotes or tunes this policy.

API: `minigrid_constructive_fit.diagnose_train_fit(episodes, wide_method="bp",
updates=128, train_a_rollout=True, max_wall_seconds=60)`. Omitting the flag keeps
the original behavior and descriptor. All original H8/catalogue/live recipes and
native algorithms remain unchanged.

## Ownership and execution

After128 complete54-row updates and both0/128 fixed TRAIN score readouts, the
runner passes its actual checkpoint to `run_train_rollout`. The evaluator accepts
only the exact original invariant D296/H32/K7/seed17 native policy, full owned
snapshot, endpoint128/6912 and unchanged source0/0. Structural snapshot validation
does not authenticate arbitrary external weights; this composition additionally
binds the checkpoint to the actual completed stream and original source/actor.

The existing evaluator forks a separately owned frozen reader and plays A-blue,
then A-red. Each game has at most16steps. Actions are first argmax of all seven
native probabilities, without masks, repair, teacher labels or target access.
Each game owns a new public initial-frame history; no cue or state carries over.
Complete actions/probabilities/observations, legal/illegal flags, success, return
and environment transitions are retained, including partial failure work.
No E/F/final game is admitted and no live learning or promotion occurs.

The existing helper owns a separate20-second budget with exactly zero training
and replay updates. The original training budget/clock/RSS limit and permanent
charges stay active throughout live games and final publication. Source, actor,
completed candidate, TRAIN packets and both budget identities remain guarded.
The readonly budget binding is checked after helper return and later readback.
A changed clock/budget/state causes an incomplete result; paid game evidence is
retained. If the helper return is lost, live work remains explicitly unknown.

## Reports and costs

`train_a_rollout` contains the helper status, original zero charges, binding/time
facts and complete or partial evaluator report. A failed rollout makes the
overall fit incomplete while preserving128 native/committed/charged TRAIN updates
and returned score/game evidence. The TRAIN fit witness remains a separate
teacher-forced metric and cannot certify game success.

`work.train_a_rollout` keeps live prediction, resets, physics, history/projection,
timing and unknown-work facts separate from TRAIN scoring. Aggregate prediction
calls/rows and environment steps include the additional game work. Maximum added
work is2resets/32predictioncalls/32physicssteps, zero optimizer/training/replay.
Native parameters/adaptive/optimizer/gradient/RNG copies, snapshot validation,
history/projection, game backend, Python and RSS add costs. The existing sampler
and external process supervision measure enclosing resource use.

These known-TRAIN games do not establish heldout generalization, acquired
retention/recovery, recurrent RL or CPC benefit. Both cue slots remain fixed even
when only one TRAIN trace fits. Preserve every failed or inconclusive run.

## Structure and checks

```text
src/adapters/minigrid_policy_evaluation.py   strict H32 readout, existing games
src/adapters/minigrid_train_rollout.py       original zero-update helper/binding
src/adapters/minigrid_constructive_fit.py    genuine endpoint and paid reporting
scripts/run_minigrid_linear_fit_control.py  explicit existing CLI option
tests/test_minigrid_wide_evaluation.py       native ownership on controlled frames
tests/test_minigrid_wide_train_rollout.py    routing/guards/public actual games
```

Focused: `python -m pytest -q tests/test_minigrid_wide_evaluation.py
tests/test_minigrid_wide_train_rollout.py`. Lint/format: `python -m ruff check`
and `python -m ruff format --check` on these six files. Types: `python -m mypy
--python-version 3.11 --follow-imports=silent --no-incremental --no-sqlite-cache
--cache-dir=NUL --platform win32` on the four production files; repeat for linux.
Exact executed selections, outcomes and budgets:
[RESULTS](../artifacts/runs/wide-neural-train-rollout-20261018/RESULTS.md).
No new dependency or product environment variable. Add future recipes through
explicit policy/clock/role admission while reusing existing frozen-reader and
zero-update guards.

## Observed engineering evidence

308 distinct scoped cases qualify: focused95PASS, combined304PASS/4artifact-save
FAIL, targeted4PASS after stage-only namespace repair. All failed attempts and
costs remain; no monolithic all-green rerun is claimed. Seven-path Ruff/format,
four-source Win/Linux mypy and independent SOURCE/repair/ACTUAL review pass.

The public BP128 synthetic fit ran twice on unchanged source. Each completes
108 fixed TRAIN rows plus32 actual target-free decisions/steps across both A
games, zero live updates/replay, zero success/return and false full-fit witness.
Native four-method evaluator fixtures use valid declared clocks and fake frames;
they are not executed training or real policy-benefit evidence.

All three sessions, including failed combined work:279nativecalls14886rows,
P248/1368,SGD264,actual4resets64steps. Aggregate external peakRSS701026304B covers
the combined test process. No original-source comparative game study ran.
