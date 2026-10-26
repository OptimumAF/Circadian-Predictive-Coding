# Public-history live evaluation

The existing MiniGrid evaluator accepts `public_history=True` for the four
feedforward controls with296 inputs. Each live episode retains its own initial
public148-vector and concatenates the current public148-vector. Raw TRAIN scoring
uses the same episode transform. Recurrent BC keeps148 inputs and a fresh owned
session per episode. The default flag preserves the original148 evaluator.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_history_evaluation_smoke --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_history_evaluation_smoke
```

Install the existing optional `requirements-minigrid.txt` dependencies first.
There are no new environment variables or dependencies. The second command is
one budgeted local engineering case, not a benchmark or acquisition protocol.
It trains BP, PC, neutral CPC, active CPC and recurrent BC for32 alternating-cue
A updates each through the existing stream. All five finish before readout.
Each checkpoint scores six original TRAIN packets and runs six target-free live
episodes on known TRAIN layouts. B/C are untrained. No final D or E probe runs.

All seven actions remain available; the first raw argmax is sent to MiniGrid.
There is no action repair, teacher, label, context metadata or future observation
in live prediction. Each decision saves its raw observation/probabilities/action
and, for feedforward controls, its actual296 input. Timeouts are outcomes.

## Structure and boundaries

```text
src/core/minigrid_public_history.py         existing pure causal encoder
src/adapters/minigrid_policy_evaluation.py  opt-in owned raw/live evaluation
src/adapters/minigrid_learning_pilot.py     existing guarded32/128 training helper
scripts/run_minigrid_history_evaluation_smoke.py fixed runnable composition
tests/test_minigrid_history_evaluation.py   reset/causality/state/refusal behavior
tests/test_minigrid_history_smoke.py        CLI and stream integration
tests/test_minigrid_fixed_a_evaluation.py   exact A-only source/layout gates
tests/test_minigrid_fixed_a_acquisition.py  fixed dose, criteria and failure flow
```

Why: the accepted296 learner could not enter the original148-only live port.
Reuse the existing trainer with exact `(0,32)` or original128 checkpoints; other
recipes fail before learner construction. This avoids another training runner.
The CLI preserves preceding paid results and original source evidence with at
most five private copies, refusing corruption before admitting further work.

## Costs and checks

The frozen case allows90body/120parent seconds,CPU1/inter-op1,GPU0,768MiB sampled
RSS and16MiB output. Planned160SGD/1440rows,30raw forwards/270rows,30live episodes
with at most480 predictions/steps; six teacher episodes add54expected/max96
physics. Public-history counters disclose derived numeric arrays; RSS includes
Python/native copies. Each active FF episode stores148 values (1184float64
equivalent bytes), excluding Python overhead. Replay/probe/promotion/tuning are0.

Focused commands, static platform arguments, failed attempts and independent
review are preserved in [the run directory](../artifacts/runs/r21-history-live-20261016/).
Use `run_checks.py focused`, `statics`, or `describe` there for the frozen command
definitions; preserve their existing receipts before a corrective invocation.
The native controller's one-shot sentinel prevents automatic experiment retries.

Extend through these ports with a separately frozen protocol. Raw demonstration
fit and autonomous success are distinct; this engineering dose establishes no
acquisition, retention, recovery, generalization or circadian advantage.

## Executed evidence

92 unique focused bodies pass across88 original passes and four corrected fake
fixture cases. The original four failures and one diagnostic failure remain
recorded; no single92-pass run is claimed. Scoped static checks and independent
SOURCE/GO/ACTUAL review pass. The sole native case completes in68.18parent seconds
with160SGD/1440rows,30raw forwards/270rows,54teacher+480live physics steps.
All30live episodes time out with zero goals. No exact A teacher sequence is
learned at this dose. The port works; useful acquisition remains open.

[Exact commands, all outcomes, costs and review](../artifacts/runs/r21-history-live-20261016/RESULTS.md).

## Fixed-A128 acquisition control

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_history_evaluation_smoke --fixed-a-acquisition --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_history_evaluation_smoke --fixed-a-acquisition
```

The explicit option uses the existing guarded stream for128 alternating A
episodes per learner, with all five learners finished before any readout. Raw
TRAIN scores occur at0/32/64/96/128. The original0/128 checkpoints also run two
target-free A cue episodes each; midpoint scores reuse the existing raw scorer.
The narrow evaluator admission requires exact predefined A geometry and its two
original148 TRAIN source packets. Default three-layout and finalD gates remain.

This fixes the total dose prospectively at the original128 updates. A gets128
episodes rather than64 in the four-phase stream, so this condition cannot isolate
interference or recovery savings. FF2439 parameters versus RNN1319, different
native objectives and recurrent work also preclude matched-compute claims.

At128, raw both-cue exact sequences and live both-cue goals are reported
separately; their conjunction is the joint acquisition criterion. All five arms
run regardless of that criterion. Incomplete/refused cases retain paid records
and use null acquisition/gains, never zero imputation. CE gain is mean CE0 minus
mean CE128; active-minus-PC/neutral differences, live success-count change and
mean-return gain have fixed directions. Every checkpoint/episode is retained.
There is no dose extension until success or selection of favorable episodes.

The case permits180body/210parent seconds,768MiB sampled RSS,16MiB output,
CPU1/inter-op1/GPU0.640SGD/5760rows,50raw forwards/450rows,20live episodes/max320
steps. Existing six-source collection adds54expected/max96teacher steps,
including unused B/C, for26environments/max416physics. No finalD/E/probe/replay/
RL/health/selection/promotion/tuning. Four named retained input-array banks are
reported separately; temporary builders, prediction copies, native states and
Python/report copies contribute to elapsed time/RSS. At most25 paid readout
records are retained. The original A32 engineering case remains available.

[Frozen scope and executed evidence](../artifacts/runs/r21-fixed-a-acquisition-20261016/).
Extend through the existing evaluator/stream, preserving both-cue criteria and
failed acquisition. Retention/recovery requires a stated initial acquisition
premise and a separately frozen protocol; no scientific benefit is implied.

### Fixed-A executed evidence

The sole fixed-A case completes in111.30parent seconds:640SGD/5760rows,
50raw forwards/450rows,288live+54teacher=342physics. All134focused tests and
scoped static checks pass. BP/RNN each fit17/18raw actions and reach only the red
goal; blue terminates with−1return, giving mean-return gain−.253125 despite one
additional goal. PC/neutral/active fit10/18and reach neither goal. All five fail
the frozen joint both-cue acquisition criterion.16timeouts,2terminal failures
and2successes are retained. Active CEgain is smaller than PC/neutral by.059376.
No advantage, retention or recovery claim; no automatic dose extension.

[Exact commands, all curves/traces, costs and review](../artifacts/runs/r21-fixed-a-acquisition-20261016/RESULTS.md).
