# Complete-episode replay during context return

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-return-replay --describe
```

Omit `--describe` for the fixed local recipe. It uses the existing A64/B64/A64
recurrent mean-CE learner and StreamRehearsal: every two wakes replay one complete
eight-step train episode, with the existing seed53 recent/reservoir policy and
16-transition/4032-byte array retention cap. No new settings or dependencies.

Why this: B training-fit forgetting was observed before final evaluation. Reusing
the existing bounded replay service tests a concrete retention intervention while
keeping its extra exposure and computation visible. There is no policy or seed
selection from final outcomes.

## Source structure

```text
src/adapters/train_acquisition_pilot.py   existing recipe and explicit replay composition
src/adapters/return_replay_accounting.py  exact wake/native/event/receipt readbacks
scripts/run_coupled_attribution.py       existing CLI dispatch
tests/test_return_episode_replay_pilot.py independent behavioral/fault cases
```

The original192-wake budget remains; a distinct original768-exposure replay budget
shares the fixed invocation deadline and RSS sampler. At checkpoints the adapter
checks native updates against exact wake plus committed replay work, including
receipt identity, objective and released original train episode content. Reports
retain attempts, returns, commits and permanent charges separately on failure.
`updates`/`checkpoint_updates` identify wake counts; explicit clocks/native fields
identify the additional replay updates. Both budgets and completed evidence stay
bound during evaluation. Final packets appear only after training finishes; final
scores never train, promote, tune or select learners/probes.

## Observed fixed case

Native clocks at wake0/64/128/192 are0/96/192/288. At128, one A and one B cue
succeed; the other B cue gets wrong_key/return-1, yielding B mean return0. At192,
both A cues succeed but neither B cue does. All held-out C/D episodes fail.
B cached accuracy at192 is .875 with replay versus .5 in the accepted off case.
This does not establish useful retained B skill, generalization, recovery savings
or CPC advantage. Replay adds96 SGD updates/768 transition exposures; off/on are
not equal-compute controls. All failures remain in the report.

Measured costs:288 SGD/2304 train rows,3065 recurrent unrolls,173 predictions
(52 fit/32 cached/89 live),32 live sessions,44 environments/158 physical steps,
27.0327 recipe seconds and319754240-byte sampled process RSS. Array retention
excludes Python/event/copy/state allocations; total time and process RSS include
them. All fixed protocol limits passed; whole-repository/release gates remain open.

[Exact commands, failures, outcomes, costs and review](../artifacts/runs/r21-return-episode-replay-20261016/RESULTS.md).
Focused verification uses ordinary pytest on the six modules named in the stage
focused receipt; affected Ruff/format and both-platform mypy pass. The first mypy
keyword-inference failure is preserved. No baseline rerun or sweep was needed.

Safely extend through existing stream/probe services, retaining distinct budgets,
full state guards and explicit wake/native clocks. Next add the existing isolated
E-task probe at fixed initial/post192 checkpoints, before final data collection,
with a separately frozen small protocol; do not create another orchestration loop.
