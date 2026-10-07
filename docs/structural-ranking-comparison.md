# P4.7b fixed structural-ranking comparison (v12)

This protocol was declared before any v12 training or final result. It
tests the *existing* reward-weighted importance signal, separated from
wake learning-rate scaling and the importance term in structural scores.
It does not add or select a new heuristic.

## Fixed roles and factors

Use new seeds 23 and 29, CPU NumPy shallow circadian and CPU Torch
shallow circadian heads, and the deterministic clean A→B source from
`difficulty_streams.py`. Per seed and phase, forty balanced development
rows are split 60% train, 20% inner guard, and 20% outer selection by
the existing four-role splitter; the independent forty-row final role
is deferred. Neither decision role selects a setting. Every factor cell
for a seed uses the same split IDs, train order, and A/B final identity.

The full factorial has wake modulation on/off, reward weighting of
per-neuron importance history on/off, and existing importance in
structural ranking on/off: eight cells per backend and seed, thirty-two
trials total. All heads enable the historical supervised-error factor
internally so its pre-update value and baseline are observed in every
cell. To isolate wake modulation off, divide the *applied parameter
delta* of that update by the observed factor after the core step;
chemistry, raw gradient, EMA, RNG, and counters remain from the same
pre-update state. To isolate importance-history reward weighting off,
divide only that step's EMA increment by the factor, leaving its prior
EMA and wake parameter update intact. Unit-factor cells are expected to
collapse. These are app-level counterfactual controls in v12 only; no
historical config or snapshot identity changes. Adaptive plasticity
sensitivity is off, so importance affects structural ranking but not
wake plasticity. Importance-score-off uses zero split/prune importance
mix only; the default on values are 0.20/0.35.

## Matched schedule and work

Each backend uses seed-matched initial 8-wide shallow heads and the same
24 A then 24 B train rows. Eight full-batch updates per phase use rate
0.02, two latent-inference iterations, and inference rate 0.1. There
is exactly one forced structural sleep after A training and before B.
The predeclared position is 8/16 completed updates, where both split
and prune budgets are active. Component sleep disables replay,
homeostasis and chemical reset. Split threshold 0 and prune threshold 1
make chemical eligibility nonselective at that event; ranking chooses
within one split and one prune slot. Initial width is 8, minimum 7,
maximum 9, so the global applied change cap is two and final width
should remain 8 when both apply. NumPy retains prune-before-split;
Torch retains its post-split prune planner. Compare factors only within
backend. Record stable chosen/applied IDs, clocks, examples, inference
loops, width, parameter count, and actual reward factors. No model
setting, seed, budget, metric, or stopping point changes from outcomes.

All thirty-two trials and their role/work/change-cap traces must pass
train-only preflight before the first final-test property is read. A
train-only feasibility failure is retained and stops final release;
it does not trigger seed or threshold tuning. After global freeze,
release A/B final roles once per seed and score A-after-A (the
post-sleep state), A-after-B, and B-after-B. Forgetting is signed
A-after-A minus A-after-B. Report every cell and factor contrast,
including zero or negative results. Deterministic local JSON is written
to a new path and repeated byte for byte. A candidate-order change
alone is insufficient to claim a learning benefit or justify another
reward-ranking term.

## Observed fixed run (after global release)

The train-only artifacts `data/structural-ranking-v12-train.json` and its
repeat are byte-identical (SHA-256
`f4137041aa152c9fca17d8d990a0f89b7329a25a9cfe9b907e2e5e54d29c0379`).
They contain all thirty-two preflighted trials and no final fields.
The full artifacts `data/structural-ranking-v12-result.json` and its
repeat are also byte-identical (SHA-256
`1f4bb8017f799fe5f3572ebb5830a51a65d654bde6e23764365121e49844a309`).
Manifest digest:
`7ef8e0cf5b27074fcae4425538bdb981dce89b824eed6799b6751635ce2e5491`.
Every trial applied one split and one prune at the A boundary, retained
width 8 and the same per-backend parameter count (NumPy 33, Torch 42),
and completed 16 wake updates, 384 example presentations, 32 inference
loops, 768 example-inference loops, one sleep, and zero replay updates.

Reward-weighted versus plain importance history selected identical
split/prune IDs and gave exactly identical held-out metrics in all
sixteen matched cells (two seeds × two backends × two wake factors ×
two score-mix factors). The chosen stable IDs were independent of wake
and history factors here. The score-mix factor did change prune IDs
for Torch seed 23 and NumPy seed 29:

| Seed | Backend | Split parent→child | Prune with score mix off | Prune with score mix on |
|---:|---|---|---:|---:|
| 23 | NumPy | 5→8 | 2 | 2 |
| 23 | Torch CPU | 3→8 | 3 | 7 |
| 29 | NumPy | 6→8 | 2 | 1 |
| 29 | Torch CPU | 4→8 | 0 | 0 |

The table below retains all distinct outcome rows after collapsing
only the exactly equal history-weighting pairs. `W` is wake scaling,
`I` is the existing importance score mix. Every row occurs twice in
the artifact, once with reward-weighted history and once with plain
history; both copies have the printed metrics.

| Seed | Backend | W | I | A after A | A after B | B after B | Forgetting |
|---:|---|---:|---:|---:|---:|---:|---:|
| 23 | NumPy | 0 | 0 | .100 | .050 | .000 | .050 |
| 23 | NumPy | 0 | 1 | .100 | .050 | .000 | .050 |
| 23 | NumPy | 1 | 0 | .100 | .050 | .000 | .050 |
| 23 | NumPy | 1 | 1 | .100 | .050 | .000 | .050 |
| 23 | Torch CPU | 0 | 0 | .000 | .200 | 1.000 | -.200 |
| 23 | Torch CPU | 0 | 1 | .000 | .150 | 1.000 | -.150 |
| 23 | Torch CPU | 1 | 0 | .000 | .200 | 1.000 | -.200 |
| 23 | Torch CPU | 1 | 1 | .000 | .150 | 1.000 | -.150 |
| 29 | NumPy | 0 | 0 | .800 | .850 | .000 | -.050 |
| 29 | NumPy | 0 | 1 | .775 | .800 | .000 | -.025 |
| 29 | NumPy | 1 | 0 | .800 | .925 | .000 | -.125 |
| 29 | NumPy | 1 | 1 | .775 | .825 | .000 | -.050 |
| 29 | Torch CPU | 0 | 0 | .975 | 1.000 | .000 | -.025 |
| 29 | Torch CPU | 0 | 1 | .975 | 1.000 | .000 | -.025 |
| 29 | Torch CPU | 1 | 0 | .975 | 1.000 | .025 | -.025 |
| 29 | Torch CPU | 1 | 1 | .975 | 1.000 | .025 | -.025 |

For explicit paired contrasts, each delta below is factor-on minus
factor-off at the same seed, backend, and other factor values. Each
contrast occurs identically for both history-weighting levels. Entries
list `(A after A, A after B, B after B, forgetting)`; every contrast
not listed is `(0, 0, 0, 0)`. In particular, the history-weighting
delta is zero for all sixteen matched pairs.

| Seed | Backend | Changed factor | Other fixed factor | Metric delta |
|---:|---|---|---|---|
| 23 | Torch CPU | importance score on | wake off or on | `(0, -.050, 0, +.050)` |
| 29 | NumPy | importance score on | wake off | `(-.025, -.050, 0, +.025)` |
| 29 | NumPy | importance score on | wake on | `(-.025, -.100, 0, +.075)` |
| 29 | NumPy | wake scaling on | importance score off | `(0, +.075, 0, -.075)` |
| 29 | NumPy | wake scaling on | importance score on | `(0, +.025, 0, -.025)` |
| 29 | Torch CPU | wake scaling on | importance score off or on | `(0, 0, +.025, 0)` |

This study does not show a benefit from reward weighting importance
history, despite the possible rank reversal in P4.7a. The existing
importance score mix reduced A-after-B accuracy in the cells where it
changed a prune choice; for NumPy seed 29 it also reduced A-after-A.
Signed forgetting can improve when its starting accuracy falls, so it
is not a benefit claim by itself. The small 40-row final roles,
eight updates per phase, NumPy's failure to learn B here, and weak A
learning for seed 23 limit any broader structural ranking inference.
The valid decision is to reject an *additional* reward-ranking term at
this gate. Preserve the historical weighted EMA and old identities;
future work would need a new prospective protocol with stronger train-
only feasibility and independent final data.
