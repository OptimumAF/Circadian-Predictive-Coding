# Fixed width-32 neural TRAIN fit

The existing saved-source TRAIN-fit runner offers one explicit width-32 recipe for
BP, PC, neutral CPC and active CPC (`gated_cpc`). Why this: useful acquisition is
still unproven on the original MiniGrid catalogue. The native policy already
supports width 32, while this runner previously admitted only width-8 BP. This
composition lets the existing learners test that capacity with owned native state
and ordinary budgets, receipts and scoring.

## Use

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_linear_fit_control --wide-neural bp --describe --fit-updates 128 --fit-seconds 30
```

Choices: `bp`, `pc`, `neutral_cpc`, `gated_cpc`. The resource-free preview declares
the chosen recipe. To execute after freezing/reviewing a small protocol, remove
`--describe` and add `--source TRAIN_REPORT --output NEW_RESULT.json`. Outputs
are create-exclusive. The reader decodes only complete original148-frame TRAIN
sources into raw296 histories. Existing full six-episode admission precedes model
construction. No final/probe records enter training, promotion, tuning or selection.

API: `minigrid_constructive_fit.diagnose_train_fit(episodes, wide_method="bp",
updates=128, max_wall_seconds=30)`. `None` keeps the original default. Other fit
families, cue-relative encoding and cue-turn gradient options cannot combine
with this recipe. The fixed policy is D296/H32/K7, seed17, LR0.03, momentum0.9,
PC inference2 steps at0.04. Width, seed and optimizer choices are not exposed.
Original width-8/default/least-squares/perceptron/kernel recipes are preserved.

Pass limits remain1..512 and wall limits(0,90] seconds. The fixed readouts are
0/min(128,updates)/updates, deduplicated, after all prescribed training. There is
no best-checkpoint, early-success stop, shuffle, seed or metric selection.

## Learners and state

The existing invariant wrapper projects every raw history before invoking the
unchanged native Torch learner. BP uses native unweighted CE and SGD/momentum;
PC uses its existing categorical relaxation and parameter rule. Neutral CPC
retains the matched neutral configuration; active CPC retains its original gates.
Native PC's returned legacy energy diagnostic is separate from prior-prediction
CE used for common readouts. This feature changes no loss or update algorithm.

Complete snapshots include the bound policy/encoding, parameters, native adaptive
state, optimizer/momentum, gradients, RNG and pass/row counters. The wide route
revalidates the full exact owned snapshot and fixed method/policy at each admitted
checkpoint and reader. Fork/restore preserves all native state. Source, actor,
candidate, inputs, clocks, budget identity and permanent charges retain existing
guards. Paid native work remains visible if a completed update's return is lost;
incomplete reports preserve the unknown-work flag and native prefix.

## Costs and limits

`work.wide_neural` reports observed completed native calls/rows. BP backward and
optimizer counts, PC parameter updates and nominal latent-row steps are explicitly
derived from the fixed algorithms and completed prefix. Two PC inference steps
mean2*completed-training-rows nominal latent-row steps. These are not an external
primitive-operation meter; failed work can be unobserved. Existing prediction
attempt/return calls and rows are reported separately.

The four shared trainable arrays contain9735 float32 values /38940 array-equivalent
bytes:296*32+32+32*7+7. Adaptive state, optimizer/momentum, gradients, RNG, temporary
arrays, validation/projection, checkpoint/runtime copies, Python and process RSS
add costs. The runner keeps at most three checkpoints plus bounded input copies,
receipts and readouts. Parameter bytes are not total memory. Equal data, seed,
width and exposure across methods do not imply equal computation.

No scientific fit, capacity improvement, acquired retention/recovery, generalization
or circadian advantage follows from implementation checks. A finite negative fit
is retained and does not prove width32 incapable. Comparative use requires a small
separately frozen protocol; preserve all original width-8 results and controls.

## Structure and checks

```text
src/adapters/minigrid_constructive_fit.py  fixed policy, owned checkpoints, costs
scripts/run_minigrid_linear_fit_control.py explicit family in existing saved CLI
tests/test_minigrid_wide_neural_fit.py     four native methods, state and failures
tests/test_minigrid_wide_neural_cli.py     resource-free CLI admission/dispatch
```

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_minigrid_wide_neural_fit.py tests/test_minigrid_wide_neural_cli.py
.venv/Scripts/python.exe -m ruff check src/adapters/minigrid_constructive_fit.py scripts/run_minigrid_linear_fit_control.py tests/test_minigrid_wide_neural_fit.py tests/test_minigrid_wide_neural_cli.py
.venv/Scripts/python.exe -m ruff format --check src/adapters/minigrid_constructive_fit.py scripts/run_minigrid_linear_fit_control.py tests/test_minigrid_wide_neural_fit.py tests/test_minigrid_wide_neural_cli.py
.venv/Scripts/python.exe -m mypy --python-version 3.11 --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 src/adapters/minigrid_constructive_fit.py scripts/run_minigrid_linear_fit_control.py
```

Repeat mypy with `--platform linux`. Exact observed checks and budgets are in
[RESULTS](../artifacts/runs/wide-neural-train-fit-20261018/RESULTS.md).
No new dependency or product environment variable. Extend explicit recipes through
the existing native ports and runner; preserve encoding/method/width ownership
and separately account for any diagnostics or disposable probes.

## Observed engineering evidence

194 distinct scoped cases pass,285 executions (91 focused +194 combined), with
no failures/errors/skips. Scoped Ruff/format and Win/Linux mypy pass. Independent
SOURCE/PROTOCOL/ACTUAL GO covers native ownership/CLI/isolation and saved metrics.
Each of the four synthetic routes completes2passes108rows/P12-108; these fixtures
contain conflicting keys and do not supply a real acquisition witness.

| Method | Final synthetic correct / 54 | Final mean CE | Full fit |
| --- | ---: | ---: | --- |
| bp | 30 | 1.664425568825 | False |
| pc | 27 | 1.817279921723 | False |
| neutral_cpc | 27 | 1.817279921723 | False |
| gated_cpc | 27 | 1.818983368398 | False |

Both commands retain separate reports. Cumulative native28/1152rows/P124-1116/
SGD10, including the old H8 regression and lost returns. External peakRSS
350154752B covers the combined test process; internal route RSS is a fixture.
Real-source fitting remains a separately frozen next case. No native algorithm or
original baseline was changed.

## Real-source study

The four frozen128-pass original TRAIN cases completed and qualified. Full fits
remain negative: BP51/54,3/6exact; PC/neutral/active30/54,0exact. PC/neutral raw
readouts match; active CE is0.070908higher at equal accuracy. All432probability
rows and512receipts were independently checked. No dose/seed/width retry.
[Full results, costs and next live-game implementation](reports/wide-neural-original-train-fit-20261018.md).
