# Known old/new games from the retained after-B model

`--context-linear-retained-game` runs the existing native evaluator on the
saved issue109 after-B state. One frozen model plays A blue/red and B blue/red
with independent resets, seed17 and at most16steps per game. All seven actions
remain legal; the first largest raw probability selects the action.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_minigrid_linear_fit_control --context-linear-retained-game --describe
.\.venv\Scripts\python.exe -m scripts.run_minigrid_linear_fit_control --context-linear-retained-game --source artifacts/runs/arrived-cue-catalogue-20261024/actual.json --output new-retained-games.json
```

The second command needs the exact local qualified source and a prospective
budget. It refuses fit overrides and existing outputs. The original causal
six-game mode and defaults remain unchanged. No new dependencies or runner.

Why this: the earlier TRAIN scores showed both tasks classified correctly after
B, but imitation accuracy alone cannot establish successful closed-loop games.
This mode tests the genuine chronological retained state, without fitting again.

## Observed outcome

| Previously trained task | Cue | Success | Steps | Return | Closed |
| --- | --- | --- | ---: | ---: | --- |
| Old A | blue | yes | 9 | 0.49375 | yes |
| Old A | red | yes | 9 | 0.49375 | yes |
| New B | blue | yes | 9 | 0.49375 | yes |
| New B | red | yes | 9 | 0.49375 | yes |

All four terminated without truncation. Success means terminated, not truncated,
and positive return. This is one fixed known-TRAIN observation, not an estimate
of reliability or generalization. Before-B native performance was not measured;
this is after-B old/new task usefulness, not an isolated retention or replay
benefit comparison. Native return after further learning remains unmeasured.

The selected state has4past fits/54row exposures:36arrivals plus18extra replay
rows, two18-row heads. The full source history has6fits/90rows, including unused
later return-A work; those sunk costs remain paid. Only state4 is restored for
play; future state6 never reaches the scorer. Readout adds **zero fits, optimizer
updates, replay, probes, training-score calls or promotion**. Final evaluation
and the original TRAIN demonstration file are unavailable to this command.

The sole run took34.243823seconds instrumented body/35.846530seconds parent,
sampled peak80,678,912bytes. It used36physics steps/36core prediction calls,
72nested public projections,4environment constructions/resets/local Gym seed
calls,2learner owners,1,879snapshot copies and2,138model validations. Existing
source guards performed433qualified reads/226,242,500bytes of source IO.
Returned public-history arrays total85,248bytes; internal projection arrays
85,824bytes. These are cumulative numeric allocations, not peak process memory.

Original source bytes, untrained source, scorer state, caller NumPy RNG and
zero-update budget guards passed. No timeout or soft deadline fired. The fixed
150second/512MiB body,155second parent,64step/prediction and8MiB stage limits
held. No retry, seed selection or changed metrics.

32focused checks per platform passed with no skips here; scoped lint, formatting,
fresh Windows types and source/procedure/result self-review passed. This is
self-review, not independent acceptance. Six checks require local artifacts and
explicitly skip when absent elsewhere; no clean-clone or release claim.

Issue108's different six-current-phase native attempt remains failed/spent and
open, with outcomes unavailable. It was not retried, reconstructed or rescored.
The earlier current-task confidence loss and extra replay costs remain. There
is no matched-compute, CPC, efficiency or general adaptation advantage claim.

[Protocol](../artifacts/runs/retained-context-linear-games-20261024/PROTOCOL.md),
[results and exact commands](../artifacts/runs/retained-context-linear-games-20261024/RESULTS.md),
[issue110](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/110).

Extend safely through the existing catalogue composer and delayed stream: the
next useful question is learning C_train after A/B under a finite replay-bank
capacity, with old/new performance and exposure/storage costs. Freeze that
distinct protocol before any model work; do not start a checkpoint audit series.
