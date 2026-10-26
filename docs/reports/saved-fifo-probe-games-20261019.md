# Saved FIFO36 after-return games — 2026-10-19

The existing count-memory evaluator now accepts the two genuine FIFO36 states
retained after the isolated F16/A16 probe. One frozen local case tested fixed A
and F games, for both cues, using raw seven-action argmax without teaching,
masking, repair or retries.

| Source cohort | Actual native endpoint | A blue/red | F blue/red | Steps | Return per game |
|---|---|---|---|---:|---:|
| Initial,0 |32updates/288rows|2/2 success|2/2 success|9 each|0.49375|
| Mature,128 |160updates/1440rows|2/2 success|2/2 success|9 each|0.49375|

These results support autonomous known-task recall after return for the finite
count control. Earlier accepted curves show F acquisition and mature A retention.
Cold A acquisition during return is not recovery from measured mature loss.
No F16 full snapshot was saved, so no after-F game result is inferred. No neural,
heldout, comparative-compute or circadian benefit is established.

## Cost and preservation

New work:72 predictions/72 rows,8 environment resets,72 steps. Zero training,
replay, teacher collection, raw teacher scores, SGD, benchmark updates or final
evaluation. Body55.2755s, outer55.5116s, external peakRSS86626304B. Each separate
30s endpoint budget used approximately27.13s, with zero update/replay charges.
The single100s allowance is spent; it was not retried.

Both tables retain36keys/87264logical payload bytes plus16retirement bytes.
The mature state retains its54evictions/864retired examples; the cold state has
zero retirements. History and canonical projection arrays each add170496bytes
over72returned predictions; Python/native/report copies are additional.
Elapsed time includes validation and preservation checks, not just inference.

The original180.0138s incomplete/spent benchmark,128updates/1152rows and censored
attempt remain unchanged. Prior F16/A16 phase charges remain separate. Source
file, complete endpoint states, restored sources and RNG checks pass. All72saved
probabilities were independently recomputed from frozen counts without a model call.

## Implementation and verification

Existing `src/adapters/minigrid_policy_evaluation.py` adds explicit
`evaluate_memory_probe_checkpoint(..., saved_fifo=True)`. It admits only fixed
FIFO36 after-return states and binds every source/scorer/checkpoint port to the
full endpoint. The original refusal64/default path remains unchanged. Existing
live games and budget checks perform execution; no new runner framework.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -o addopts= -q tests/test_minigrid_fifo_probe_live_evaluator.py tests/test_minigrid_probe_live_evaluator.py tests/test_minigrid_memory_return_evaluation.py tests/test_minigrid_recurring_fifo_evaluator.py
```

174 scoped cases pass, plus scoped lint/format, Windows/Linux types and independent
SOURCE/ACTUAL review. Full repository, release and interactive QA remain open.
[Exact commands, native artifacts and limits](../../artifacts/runs/saved-fifo-probe-games-20261019/RESULTS.md).

Next: compose existing TRAIN prior-gradient diagnostics with the explicit H32
joint54-row route to investigate partial neural acquisition. Preserve existing
baselines/objectives and require a separate frozen protocol for new native work.
