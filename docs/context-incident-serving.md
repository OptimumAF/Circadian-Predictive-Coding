# Owned LS serving with incident triggers

The normal workflow can run the accepted calibrated schedule through its
existing owned LS source, actor, guards, serving timeline and rollback path:

```text
python -B -m scripts.recurring_ticket_workflow_example --method linear_fit --context-return --separate-context-heads --paired-serving joint --adaptive-serving --incident-stream shift --calibrated-serving --output NEW.json
```

`--incident-stream stationary` declares the stationary control. Omit
`--calibrated-serving` to use the same stream with the existing supervised
error/ACK trigger. These options require adaptive owned LS joint mode. Existing
commands and paired recipes retain their defaults. Outputs must be new.

Each stream contains eight ordinary TRAIN packets with delay1 and the original
24 serving opportunities/eight-wake ceiling. Six stationary invoice-v1 packets
have3rows each. The shifted stream then has joint-v1 and joint-v2 packets, each
3new+6old rows. The stationary stream repeats the same3row task twice more.
Packet counts/row identities/order cannot be changed under those recipe names.
The two new recipes have a separate menu; saved paired/replay cohorts are intact.

Current-candidate diagnostics have a prospectively declared216row ceiling
(24opportunities times maximum9rows), never replenished. Correct packets may be
ACKed without fitting. Calibrated mode separately uses the unchanged default
six-packet calibrator with declared prefix IDs, model label context-ls-12 and
an owned initial frozen diagnostic reference. Neither stationarity nor model
identity is statistically attested by the label alone. Prefix observations,
extra frozen-reference predictions and scalar replay are paid diagnostic work.

An onset permits fitting only through its already-arrived FIFO boundary. Each
fit still requires a fresh error on the current learner and original credit.
Continued alarms do not renew it. The horizon stays fixed; missing alarms,
unfinished consumption and failed publication stay visible. Original source
budgets, native origin, receipt checks, guards and exact rollback still apply.
See [schedule contract](calibrated-shift-schedule.md).

These are known TRAIN/development task templates with public context cues and
explicit old-task reuse in joint fits. Guards and postpublication task episodes
are outside calibration and trigger selection. Private probe/final data remain
unavailable. Trigger comparisons include unequal diagnostic computation and
cannot establish CPC advantage, generalization or deployment readiness.

The four-case local protocol is frozen separately from correctness tests. See
[executed results, commands, costs and limitations](reports/context-incident-serving-20261029.md).
