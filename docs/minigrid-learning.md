# Actual MiniGrid learner checkpoints

The existing coupled-attribution CLI now has a fixed MiniGrid cue/key/door
scenario. It composes the accepted actual game, complete demonstration packets,
native learner ports and `AttributionStream`; it adds no trainer or learner loss.

```powershell
.venv/Scripts/python.exe -m scripts.run_coupled_attribution --scenario torch-minigrid-cue-key-door --describe
.venv/Scripts/python.exe -m scripts.run_coupled_attribution --scenario torch-minigrid-cue-key-door
```

Install `requirements-minigrid.txt` and the existing Torch requirements first.
`--describe` creates no environment or learner. The opt-in run is a single fixed
local case; use its printed JSON for all outcomes, including failed learning.

## Frozen protocol

- Five existing methods: BP, PC, neutral CPC, active CPC and recurrent BC.
- Seed17,148 public input coordinates,H8,K7, learning rate.03, momentum.9;
  original PC inference2/.04 and CPC defaults remain unchanged.
- Six TRAIN teacher sources collected once: A/B/C_train, blue/red each.
  Alternate cues through A32/B32/C_train32/A32:128 updates per method,640 total,
  5,760 row exposures. This is demonstration BC, with no reward update or replay.
- The existing training-only stream has no held-out packets and retains owned
  checkpoints0/32/64/96/128. Every method finishes training before evaluation.
- Score the original six TRAIN traces and run six known-layout live episodes
  per checkpoint. Construct held-out D only after all training; evaluate its two
  cues at endpoint128, without teacher actions or final labels.

Live policies receive only current public observations. All seven inputs remain
valid, including wasted moves/no-ops; use raw first argmax without action repair.
Recurrent policies get a fresh owned zero-step session for each episode. Learned
state, source, actor, previous completed evidence, original budgets and receipts
are checked before admitting work; failed attempts stay visible and charged.

## Metrics and interpretation

Raw CE improves downward; accuracy and exact sequence accuracy improve upward.
Live goal success and mean return improve upward. Preserve both cues and every
checkpoint. Signed metrics record initial A, new B/C, old A loss, A return gain
and the remaining return gap. A32 acquisition requires exact raw sequences and
both successful live episodes; if it was never acquired, later failure does not
establish forgetting of a learned task. B/C acquisition conditions are separate.

Primary comparisons are **active CPC minus PC** and **active CPC minus neutral
CPC**, endpoint128 pooled known-layout live success/return. Equal SGD/row dose is
not equal computation. The RNN has extra recurrence parameters and is an
architecture control. One fixed seed, familiar training layouts and two final
episodes do not establish statistical equivalence or broad generalization.
The D configuration was declared and exercised in earlier adapter/teacher smoke
checks. Final-only here describes this learner run's data access; it does not
describe a previously unknown or blinded external benchmark.

## Resource and failure accounting

The case allows180s body/210s parent,768MiB sampled process RSS,64MiB JSON,
CPU/BLAS/Torch intra/inter-op1 and GPU0. Planned work is640SGD/5,760 training rows,
150 raw score calls/1,350 rows,160 live episodes with at most2,560 predictions/
physics steps, plus54 teacher steps. RNN unroll is bounded by1,934 steps.
Full normalized before/after observations and decision probabilities remain;
redundant raw-next images in live copies are omitted to bound storage. The
normalization is invertible on the declared native integer encoding.

Unexpected faults stop further work and retain the observed prefix. Unknown
native work is labeled unknown rather than inferred as free work. No automatic
retry, seed search, endpoint selection, promotion or metric changes are performed.

## Files and next extension

```text
src/adapters/minigrid_learning_pilot.py     existing stream/native composition
src/adapters/minigrid_policy_evaluation.py frozen checkpoint/raw/live readouts
scripts/run_coupled_attribution.py          existing CLI, new explicit scenario
tests/test_minigrid_learning_pilot.py      fake boundary/isolation/failure checks
```

Native algorithms and the game are reused. Keep final D out of any future
training, replay, probe selection or tuning. Introduce new controls or doses
only through a separate prospectively reviewed protocol, preserving this case.
[Session commands/evidence](../artifacts/runs/r42-minigrid-learning-20261016/RESULTS.md).
Full useful acquisition, retention, recovery, held-out and recurrent RL outcomes
remain open; implementation correctness alone establishes no circadian advantage.

## Observed session result

The original418-case qualification and native failure remain saved. A measured
implementation correction now resolves Torch once per nested state comparison
and holds completed reports as private immutable payloads until final publication.
Native/copy/receipt/budget/corruption checks and output schema remain intact.

476 focused cases plus2 publication-edge cases pass (478 unique), with scoped
statics. Synthetic profiled comparisons improve, but a fresh separately budgeted
native case again failed180s after640SGD/5760rows. It reached47live starts versus
the earlier54; no whole-run speed benefit is established. All observed overlapping
receipts/scores/actions match; both failures and missing coverage remain saved.
Neither case reached final D or published a complete method comparison.
[Correction, failures and exact commands](../artifacts/runs/r42-minigrid-readout-cost-20261016/RESULTS.md).
Full native readout completion remains open. Next is the existing R2.1 probe's
exact148/probe-only E boundary prerequisite; it does not close this failed gate.
