# Native affine drift versus periodic: a scheduling tradeoff

[Issue194](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/194)
completed one fixed two-arm local observation. Both controls received the same
six TRAIN arrivals and six prequential prediction opportunities. All training
windows finished before any task readout. Neither control demonstrated complete
new-task acquisition, old-task retention and return recovery.

Each cell is old A3/new A4 successful games, each out of two, at fixed logical
phase ticks. These are familiar TRAIN layouts, not held-out generalization.
Both controls solve only one old goal at tick2, so complete initial-A acquisition
is itself absent at that declared phase.

| Control | Tick2 | Tick4 | Tick6 | Actual fits/rows | Pending packets |
| --- | --- | --- | --- | --- | --- |
| Fixed drift |1/0 |2/0 |2/0 |2/18 |4 |
| Fixed periodic2 |1/0 |2/0 |1/1 |3/27 |3 |

Both achieved5/12successful episodes across all readouts, total return2.46875
each. That aggregate hides the difference: drift retained both old goals but
never fitted a new-task packet; periodic reached only the first new-task packet,
at return time, losing the corresponding old goal. Neither acquired a new goal
during the B phase. Conditional recovery is unavailable because no old-task
decrease occurred before the return phase; periodic's decrease occurs at return.

The TRAIN trace identifies the concrete scheduling gap. Drift wakes at1/3;
the B-blue error2/9 at3 triggers fitting the older FIFO A-red packet. The B-red
prediction belongs to an older candidate and is correctly retired. Returning-A
errors are0/9, so drift then waits with both A4 packets and both returned-A
packets still pending. This is feedback/backlog misalignment, not evidence that
the learner lost plasticity. Periodic wakes at2/4/6, paying one additional fit.
The fixed policies, threshold.1, seed17 and metrics were not tuned after results.

Shared fresh preparation cost18teacher steps/two closed A4 environments. One
94705-byte packaged TRAIN export was read once; its pre-existing preparation
cost is not recreated or counted as newly measured work. Each arm paid6diagnostic
forks/predictions54rows. Across the case:5LSEfits45rows/5LAPACKcalls,326native
predictions422rows including108diagnostic rows,314game steps,332physics steps
including the teacher,26environments/closes,24empty native owners and5259native
snapshots. These counts do not measure all copies/allocations/FLOPs.

All24games completed and closed:10successes at9steps/.49375return/terminated,
14failures at16steps/0return/truncated. Native body83.770322900s, parent84.307731400s,
sampled peakRSS86,958,080bytes. Original per-arm6update90s512MiB owners and separate
0update/0replay150s512MiB readout owner retained original clocks/limits; caller
Python+NumPyRNG/source/actor/readout-state/paid-prefix checks passed. Readout-owner
elapsed80.5671171s includes shared preparation and windows as specified. No budget
renewal, retries, final/probe/replay/promotion, GPU or packages installed.

Equal information/opportunities are not equal actual training exposure, compute
or wall time. This single known-layout result establishes no CPC, efficiency,
generalization, statistical or release advantage. One-agent source/procedure/
result self-review, not independent acceptance. Earlier failed177/191cases and
all negative baselines remain unchanged.

[Frozen protocol, exact commands, original failures, costs and self-review](../../artifacts/runs/affine-drift-native-20261026/RESULTS.md).
The original actual.json was not reopened for review; analysis uses the emitted
same-process terminal scalar summary. No model evaluation or scoring was rerun.

Next implementation: add an explicit default-off diagnostic mode that scores
the oldest actually eligible FIFO TRAIN packet using the current candidate after
its label arrives. Bind packet/candidate/clock, charge all predictions, preserve
this prequential baseline and thresholds. Literal isolation/cost/FIFO tests
precede any separately frozen native comparison. This addresses the demonstrated
TRAIN scheduling gap; it does not guarantee a favorable result.
