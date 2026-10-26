# Native neutral-CPC endpoint cost diagnosis

One frozen original-source H32/joint128/F16/A16 run stayed within unchanged
60-second endpoint/240-second case limits. It remains **incomplete**: the second
source0 endpoint timed out in its final observer check. No retries or cap changes.

| Source0 endpoint | Recorded elapsed | Preservation spans | Prediction spans | Outcome |
|---|---:|---:|---:|---|
| after F16 |55.17s |53.79s |0.249s |endpoint complete |
| after A16 |59.76s, partial |58.77s |0.250s |final deadline failed at60.048s |

Preservation-labelled inclusive ratios are97.50% and98.34% of the recorded
intervals. They are not exclusive CPU shares; spans may overlap and omit setup,
reset, report serialization and other work. Prediction includes adapter work.
The actual preservation subroutine responsible is not yet isolated.

All eight recorded source0 games ended without goals/return0 after16steps each;
eight source128 games remain unrun. A healthy timing profile does not qualify a
failed protected endpoint. No full adaptation/retention/recovery or circadian
benefit follows. Original quartet failures and negative results stay preserved.

Costs:160nativeupdates/7200trainingrows,190prediction calls/686rows,18resets/
218physics steps; all attempts returned, zero SGD/replay/final/gameupdates.
Public unknown aggregate counts remain null; external meter values remain separate.
Original sources,128receipts,270TRAINprobability rows and16F/A checkpoint prefixes
reproduce accepted controls exactly. Independent SOURCE and saved-data ACTUAL
review accepted the diagnosis only. No ambient-global-RNG before/after assertion
was added; unchanged implementation/isolation evidence was reused.

Next target is redundant fully validated state comparison inside the existing
common-H32 preservation path, preserving all mutation/port/clock fences and
measuring any proposed improvement before another native attempt.

[Exact commands, costs, failed gate and review](../../artifacts/runs/neutral-endpoint-profile-20261018/RESULTS.md).
