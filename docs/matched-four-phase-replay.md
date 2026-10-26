# Matched four-phase replay controls

Two explicit routes compose existing recent-only and partitioned rehearsal with
the four-phase stream and final evaluator. Both use256 wake updates plus128
replay updates, the same native learner/seed/objective and retention ceilings.
No new circadian mechanism, replay selector or training runner is introduced.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-four-phase-recent --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-four-phase-partitioned --describe
```

Omit `--describe` to execute the separately budgeted local arm. For the declared
comparison, run each fixed CLI arm in its own process, recent then partitioned;
never pass either final report into training. Describing allocates no learner,
environment or native work. No new dependencies or environment variables.

## Boundaries

```text
src/adapters/four_phase_replay.py       fixed policy/protocol composition
src/adapters/train_acquisition_pilot.py existing stream/final orchestration
src/adapters/return_replay_accounting.py fixed192/256 replay readbacks
scripts/run_coupled_attribution.py      explicit arm CLI dispatch
tests/test_four_phase_replay_*.py       accounting and real-stream fake-port tests
```

Why: actual C acquisition followed by forgetting makes retained-task behavior
the next outcome. Reuse RecentStreamReplayPolicy and StreamReplayPolicy, both
already supporting complete episodes. Only the accounting adapter needed
explicit256-wake admission (128events/1024 replay exposures). Its original192
default, full receipts/source validation and permanent charges remain binding.
Old acquisition/evaluation/probe/replay routes keep their defaults.

Both arms use period2,8 replay rows/event and16 retained transitions/4032 encoded
array bytes. Partitioned allocation is8recent+8reservoir transitions with seed53.
Recent has no reservoir seed. Each arm performs384 SGD updates/3072 training
row exposures; copies/report/optimizer storage is additional and appears in time
and sampled process RSS. Source/actor/candidate/selected checkpoints, full replay
events/receipts, released train content and original wake/replay budgets bind
fit and final calls. Wake clocks0/64/128/192/256 correspond to native0/96/192/288/
384. Final data exists only after training/finish; no promotion, tuning, probes,
RL change, checkpoint selection or final-driven update is allowed.

The primary fixed readout is pooled B/C_train live success and return at wake256
(four episodes), retaining acquisition at B128/C_train192 and every other
checkpoint/layout/cue. Report partitioned-minus-recent accuracy/success/return
and recent-minus-partitioned CE; positive favors partitioned. One fixed seed
does not establish replicated advantage. No-replay256 is an unequal-compute
reference. Equal update counts/retention ceilings do not match wall time or
selected information.

The original alternating cue order and period2 latest-one recent selector imply
blue-only recent replay. This is declared before native execution. Preserve the
existing control and report per-cue/context exposure; any difference also changes
cue coverage, so it cannot isolate a reservoir or memory-age causal effect.
No baseline/schedule/seed is tuned to manufacture a winner.

## Verification and extension

Prefix Python commands with `.venv/Scripts/python.exe -X utf8 -B`:

```text
-m pytest -q tests/test_four_phase_replay_accounting.py tests/test_four_phase_replay_pilot.py tests/test_return_episode_replay_pilot.py
-m ruff check src/adapters/four_phase_replay.py src/adapters/train_acquisition_pilot.py src/adapters/return_replay_accounting.py scripts/run_coupled_attribution.py tests/test_four_phase_replay_accounting.py tests/test_four_phase_replay_pilot.py
-m ruff format --check <same six paths>
-m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 <same six paths>
```

Also run mypy with `--platform linux`. Exact scoped outcomes, failed attempts and
experiment costs are recorded in stage RESULTS after execution. Add future
controls through existing policy/runner ports with separately fixed budgets;
keep final outcomes out of policy adjustment or favorable run selection.

## Observed frozen pair

200 distinct focused cases have passing latest results after two preserved
descriptor failures and a two-case correction. Scoped statics and the sole
frozen pair passed. Saved analysis initially failed a shadowed receipt variable;
the original source/failures remain, corrected saved-only arithmetic passes.

Both arms have zero B/C_train final live success and return (primary difference
zero). Both learn C then forget it after A return. Partitioned has one additional
intermediate A success and better final B/C cached accuracy, but all D attempts
fail and D cached accuracy is worse. This does not demonstrate useful retention
or circadian benefit. Recent has5 successes/40, partitioned6/40; all failures stay.

Each used384SGD/3072trainrows/16retainedtransitions/4032arrayB. Recent required
4318unroll/252predictions/188steps/24.8529parentseconds; partitioned4336unroll/
270predictions/207steps/26.3356seconds. Wall time/selected contents are not matched.
Recent replay is entirely blue cue; partitioned coverage is also uneven. These
declared composition limits prevent a pure memory-age/reservoir causal claim.
[Commands, costs, failures and all outcomes](../artifacts/runs/r42-matched-four-phase-replay-20261016/RESULTS.md),
[independent review](../artifacts/runs/r42-matched-four-phase-replay-20261016/REVIEW.md).
