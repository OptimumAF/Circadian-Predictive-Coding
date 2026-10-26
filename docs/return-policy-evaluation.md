# Cached and live evaluation after context return

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-return-policy --describe
```

Omit `--describe` to run the fixed local recipe. It preserves the accepted H8
seed17 mean-CE A64/B64/A64 training schedule and all13train-fit curves. Only after
all192updates and `stream.finish()` does it collect eight final-test teacher
packets and evaluate checkpoints0/64/128/192 on A/B/C/D, both cues. A/B are known
layouts; C/D were unavailable to training. Final data never train, promote, tune,
select a learner, or choose a probe. There is no replay or reward update.

The final readout uses the existing cached scorer and key/door live evaluator.
Each live episode starts a fresh zero-step owned recurrent session, chooses raw
argmax (first tie), and has at most12decisions. It receives observations, not
teacher targets or legal-action repair. All32episodes, cached scores, failures
and returns remain in the report, regardless of outcome quality.

## Observed fixed case

At checkpoint192, both A cue episodes succeed in8steps with return1 each. All
other30episodes fail with illegal actions: every earlier A attempt, all B attempts,
and every held-out C/D attempt. B cached accuracy.875 at128 does not translate to
live success. This verifies closed-loop behavior on the known A trace domain;
it does not establish broad generalization, retained B skill, recovery savings
or a circadian advantage. Initial A fit at64 was incomplete, and no equal-A-
exposure control ran. All earlier negative evidence remains valid.

All old training fields match the accepted train-only run exactly, excluding
protocol metadata/timing/RSS; relevant training protocol fields are also equal.
Recreating the prefix was necessary because the earlier JSON did not contain
restorable opaque optimizer checkpoints. This is not independent replication.
See [protocol, exact commands, costs and failures](../artifacts/runs/r42-return-policy-evaluation-20261016/RESULTS.md).

## Structure and boundaries

```text
src/adapters/train_acquisition_pilot.py       existing training and explicit final gate
src/adapters/return_policy_evaluation.py      final report/collection/accounting glue
src/adapters/key_door_pilot.py                existing live helper, explicit task sequence
src/app/coupled_attribution.py               existing cached helper, opt-in state guards
scripts/run_coupled_attribution.py            existing CLI dispatch
tests/test_return_evaluation_boundaries.py   common helper state/callback tests
tests/test_return_policy_evaluation_pilot.py integration/partial-failure tests
```

Why this: complete teacher-sequence fitting should be checked against actual
interaction, with final data unavailable until training is sealed. Four bounded
owned state copies reuse existing snapshot/session ports; no new checkpoint
format, publication framework or orchestration loop is introduced.

The route checks original budget/clock/cumulative charges, complete candidate,
receipts, saved checkpoints, and source/actor state around final callbacks.
Changes refuse before further work. Failed final evaluation keeps192training
charges, completed/attempted cached-call counts, partial live traces and unreached
slots. Cached checkpoint scores publish only after complete cached evaluation;
partial failed native work also requires its execution receipt. New helper checks
are opt-in, and earlier routes retain their public/default behavior.

The local case costs192SGD/1536train rows,52train-fit plus32cached plus74live
forwards (158total),2282unroll steps and44environments/140physical steps. Total
recipe25.5224seconds and sampledRSS316919808B include state/receipt checks, native
imports, data copies and diagnostic/report work. These costs are not pure training
latency; original train-input byte counts exclude these added allocations.

No new dependencies, settings or environment variables. Scoped check commands
and the failed initial fake-call allowance are preserved with the reviewed
prospective engineering amendment. Scientific budgets were unchanged. Whole-repo,
clean-clone and release gates remain open. Safely extend the recipe through
existing stream services, bind native and wake clocks explicitly, and freeze a
small protocol before any new experiment; do not tune from final outcomes.
