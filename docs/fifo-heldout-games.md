# Final D-layout readout of FIFO36 benchmark128

The successful known-context memory learner now has an explicit held-out game
readout through the existing evaluator. This adds no training or adaptation.
The fixed observation is negative: both D cues fail, despite earlier A/C recall.

## Contract

`evaluate_heldout_fifo_checkpoint(source, state, *, budget, check_boundary)` in
`src/adapters/minigrid_policy_evaluation.py` accepts the original invariant
FIFO36/296/K7/87264-byte policy and genuine benchmark128/1152 state. The supplied
source must remain at its original0updates. Intermediate32/64/96 and probe160
states are refused. Full entries, order, counts, evictions and retired examples
remain part of the checked state; matching scalar clocks alone are insufficient.

Use the existing strict decoder with
`artifacts/runs/minigrid-recurring-fifo-20261017/native.json`, selecting its fixed
last benchmark snapshot. This is not the best-performing prefix or a probe-trained
state. The original incomplete180-second execution remains failed and charged;
its model history is reused, never its spent budget. No new model reconstruction,
source loader or training runner is required.

The readout requires `ToyBudgetSession(ToyExecutionBudget(max_training_updates=0,
max_replay_examples=0, ...), clock)` with a finite caller-owned deadline/RSS bound.
Budget identity, clock, progress and permanent charges are checked. The native
evaluator constructs exactly predefined D-blue and D-red, both `final_test`,
seed17/max16steps. It uses the frozen count predictor's raw all-seven-action first
argmax, without teacher targets, masks, repair, live updates, selection or promotion.
No TRAIN score arrays enter this path. Success and failure prefixes label public
initial frames FINAL. The old TRAIN-only API retains its exact admission/defaults.

The model and protocol must be frozen before final data is opened. D outcomes
cannot tune training, select replay/probes/controllers, or admit a promotion.
D was held out of this learner's A/B/C/A TRAIN history, but its public layout
family was inspected in older work; it is not a globally untouched benchmark.

## Observed result

| Final D cue | Success | Return | Steps | End | Stored-key matches |
| --- | ---: | ---: | ---: | --- | ---: |
| blue | 0 | 0 | 16 | truncated | 0/16 |
| red | 0 | 0 | 16 | truncated | 0/16 |

All32predictions are uniform1/7; raw first argmax chooses left0 every time.
All32native action indices are valid and all32steps apply. These are native
index/effect counts, not a geometric-progress or general safety claim. The raw
transition schema has no separate `legal` flag. Both environments close.

The exact-key learner's successful known-task recall does not transfer to this
fixed D case. No statistical/generalization/CPC/efficiency/retention advantage is
established. This does not change the accepted A→B→C→A acquisition/loss/return
evidence or imply every other held-out layout must fail. No D-driven model repair
or final-to-TRAIN conversion follows from this observation.

Source state/input bytes/Python+NumPyRNG remain unchanged; original model history
128updates/1152rows versus0new updates/rows/replay/probe/teacher/promotions.
Four memory learner constructor entries,32native queries/32returns,2resets32steps;
0Torch heads/SGD/PC. Table36entries87264numericB+16retirement metadata. Body76.7815s,
meter78.0732s,outer78.2934s,sampled peak86089728B. Validation/copies/projections/
posthoc key matching/startup add cost; these instrumented times are not an SLO or
an equal-compute comparison.

## Verification and handoff

```console
python -m pytest -q tests/test_minigrid_heldout_fifo.py tests/test_minigrid_recurring_fifo_evaluator.py
python -m ruff check src/adapters/minigrid_policy_evaluation.py tests/test_minigrid_heldout_fifo.py
python -m ruff format --check src/adapters/minigrid_policy_evaluation.py tests/test_minigrid_heldout_fifo.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent src/adapters/minigrid_policy_evaluation.py tests/test_minigrid_heldout_fifo.py
```

67checks qualified/platform; six affected cases reran after the final-frame report
label correction. Scoped statics pass. Scripted fixtures use literal snapshots
and fake completions, not native final data or demonstrated skill. The same agent
reviewed source/procedure/results, not independent acceptance. Full commands,
failures, skipped checks, budgets and immutable native evidence:
`artifacts/runs/fifo-heldout-d-20261023/RESULTS.md` and issue62.

## Inspect the saved final observations

```console
python -m scripts.build_key_door_inspector --source-artifact artifacts/runs/fifo-heldout-d-20261023/actual.json --output final-d.html
```

Use a fresh output filename; the writer refuses overwrites and source/output
overlap. Open the standalone HTML locally. Select either D cue, step through its
17 public frames, inspect all seven probabilities and open the exact decision,
transition, feature or cost source. The last frame has no next decision.
No model, environment, network connection or training is required.

Original TRAIN history remains incomplete/spent. New probe updates are recorded
as zero; TRAIN and probe outcome scores are absent in this artifact and remain
unknown. Final query diagnostics are not fit curves. Partial/failed records retain
their paid prefixes, false/unknown preservation flags and unknown unrun outcomes.
Malformed provenance, geometry, actions, counters or complete proofs are refused.

The new `minigrid_heldout_fifo_inspector_v1` projection is separate from the
existing F/A probe schema. It adds a 4 MiB final-source bound while keeping the
existing bounded envelope, 16 MiB rendered limit, duplicate/nonfinite refusal,
inert JSON escaping and exclusive output. No resource artifact can be mixed in.

This readonly view was selected before D was observed and grants no D-based
tuning. Focused saved-data checks:

```console
python -m pytest -q tests/test_minigrid_heldout_fifo_inspector.py tests/test_minigrid_inspector_projection.py tests/test_minigrid_saved_fifo_inspector.py tests/test_minigrid_saved_fifo_inspector_files.py
```

See ADR-0579 and `artifacts/runs/fifo-heldout-inspector-20261023/RESULTS.md` for
scoped check receipts, browser observations, failures and limitations.
Full R4.2, R4.7, scientific and release requirements remain open.
