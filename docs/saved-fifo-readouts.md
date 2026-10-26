# Saved FIFO36 TRAIN readouts

## Saved F16/A16 after-return game endpoints

`src.adapters.minigrid_policy_evaluation.evaluate_memory_probe_checkpoint` accepts
`saved_fifo=True` for genuine invariant FIFO36 source0/128, phase `after_return`,
native32/160 and rows288/1440. Supply an owned learner restored from that complete
state, an independent zero-training/zero-replay budget and the preceding-result
boundary guard. The evaluator forks its scorer and preserves both states. Every
port must match the full supplied endpoint, including count order and retirement.

The default remains the old refusal64 protocol. Explicit FIFO admission refuses
FIFO64, raw/noncanonical data, different clocks/policies, unavailable after_probe
states and non-boolean flags. It collects no teachers, raw scores or final data.
The accepted saved probe contains no complete F16 snapshot: never reconstruct one
or repeat training to label a new state as historical evidence.

The frozen local case succeeds in all8known A/F games after return, with zero new
training. [Report and limitations](reports/saved-fifo-probe-games-20261019.md).

The original recurring count control persisted five complete native checkpoints,
but exhausted its original 180-second allowance during readouts. Its incomplete
status, 128 updates, 1,152 training rows, and censored game remain historical facts.

The saved readout CLI validates those full snapshots, all original receipts and
six TRAIN demonstrations. It reuses completed cells and evaluates only the fixed
missing 12 scores and 16 games. This is a separate invocation with zero training,
zero replay, no teacher, and its own at-most-120-second body allowance.

```powershell
.venv/Scripts/python.exe -m scripts.run_minigrid_saved_fifo_readouts --describe
.venv/Scripts/python.exe -m scripts.run_minigrid_saved_fifo_readouts `
  --source artifacts/runs/minigrid-recurring-fifo-20261017/native.json `
  --output artifacts/runs/my-separately-budgeted-case.json --max-seconds 120
```

Run native evaluation only under a separately frozen local protocol and resource
supervisor. The CLI requires a new output file, bounds input/output at 16 MiB,
rejects nonfinite or duplicate-key JSON, and refuses altered historical evidence.
The JSON validates internal consistency; it cannot authenticate external origin.

## Boundaries and extension

- `minigrid_saved_fifo_inputs.py` converts bounded JSON data to owned typed records
  and validates historical completion against native counts and raw decisions.
- `minigrid_saved_fifo_readouts.py` composes the existing evaluator with a fresh
  zero-update budget, full source/RNG guards, and at most three result groups.
- `minigrid_policy_evaluation.py` admits explicit ordered score/game subsets of
  the original six TRAIN pairs. Its default route still reads all six pairs.
- The CLI handles file IO. It never restores the historical execution budget.

Why this: actual checkpoints already contain the trained state. Repeating their
training or completed games would spend work without resolving the missing cells.
A censored old attempt remains charged; its new whole episode is separately paid.

The present contract accepts this exact incomplete source matrix. A different
protocol needs explicit admission, tests, a prospective resource allowance, and
review. Do not expand it to arbitrary roles, final evaluation, outcome selection,
or unlimited success-record chains.

Targeted checks:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_minigrid_saved_fifo_inputs.py tests/test_minigrid_saved_fifo_readouts.py tests/test_minigrid_saved_fifo_cli.py tests/test_minigrid_recurring_fifo_evaluator.py tests/test_minigrid_memory_probe_cli.py
```

Known TRAIN count recall establishes neither neural adaptation nor a circadian
advantage. Comparative costs include historical training, the failed readout,
new readouts, direct validation arithmetic, snapshots, copying, and sampled RSS.
