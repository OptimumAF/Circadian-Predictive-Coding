# Native ready-feedback learning: acquisition and recovery, with forgetting

[Issue196](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/196)
completed its single frozen local observation. The existing stream diagnosed
the current candidate on the oldest arrived FIFO TRAIN packet, then consumed
that packet on a selected wake. All six updates completed without bypassing
pending work. All training finished before the twelve fixed games.

| Fixed logical phase | Fits / row exposures | Old A3 goals | New A4 goals |
| --- | ---: | ---: | ---: |
| After initial A3, tick2 | 2 / 18 | 2/2 | 0/2 |
| After A4, tick4 | 4 / 36 | 0/2 | 2/2 |
| After returned A3, tick6 | 6 / 54 | 2/2 | 0/2 |

The learner acquired both new goals and recovered both old goals after losing
them, paying eighteen returning-A training rows for recovery. It forgot both
new goals on return. Joint retention failed. These are known deterministic
TRAIN layouts at seed17, not held-out generalization or statistical reliability.
Six of twelve games succeeded; total return was2.9625. The aggregate hides the
complete swap in which task is solved.

At opportunities1..6, diagnosed mistakes were8/9,8/9,2/9,2/9,2/9,1/9. Every
fraction exceeded the frozen drift threshold.1. Each feedback packet matched
the FIFO update actually consumed, its original feature/label times and current
candidate update count. Six packets were consumed; none remained pending.
This addresses the concrete feedback/backlog mismatch seen in
[the prequential observation](affine-drift-native-20261026.md).

The historical [current-only control](a4-current-only-20261025.md) already
showed the same task endpoints under six fixed updates. Its accepted conclusions
were reused without a refit or raw/model reader. The historical joint-bank
control retained both tasks with six fits/ninety row exposures. Those are
descriptive references with different producer and diagnostic assumptions;
this run establishes neither parameter equality nor paired, matched-compute or
controller-efficiency superiority. No CPC model or circadian benefit was tested.

Measured new work:6native least-squares fits/54rows and6LAPACKcalls/54rows,
6diagnostic forks/predictions/54rows,150game predictions/steps,18teacher steps,
14environments created/closed,12empty learner constructors and2529snapshots.
Total native predictions156calls/204rows; total physics168steps. All twelve
games completed and closed: six successes at9steps/.49375return/terminated,
six failures at16steps/0return/truncated. Counters do not measure all allocation,
copy traffic or FLOPs. One94705-byte packaged TRAIN export was read once;
its54rows were decoded and18A3rows selected. One fresh A4 TRAIN pair supplied
18rows. Packaged preparation cost was not measured anew.

Native body15.2182166s, parent15.6602888s, sampled peakRSS83,972,096bytes.
The original six-update90s/512MiB training owner and separate zero-update/
zero-replay150s/512MiB readout owner retained their bindings and clocks.
Readout-owner elapsed13.7148269s includes preparation and the training window;
training-owner elapsed.3428310s is not isolated fitting time. Source, actor,
game states, paid-prefix values/identities and caller Python/NumPy RNG checks
passed. RNG baseline was taken after optional imports. No budget renewal,
retry, final/probe/replay/promotion/tuning, package installation or GPU work.

Source/procedure/result **self-review**, not independent acceptance. Reused
issue195's unchanged-source72checks/platform and scope statics. Its original
failed fake-owner gate remains failed; the prospective amended qualification
is separate. New case/supervisor Ruff, format and default/Linux-target mypy
passed in8.5664039s. Native attempt passed first launch. A later metadata-review
helper failed on absent optional psutil before reading results; switching to
the repository's existing RSS reader passed, with no installation or model
rerun. All original failed/spent studies and negative results remain unchanged.

[Frozen protocol, commands, costs, limitations and review evidence](../../artifacts/runs/affine-ready-native-20261026/RESULTS.md).
Review used only the same-process emitted scalar summary; actual.json was not
reopened. Full-suite, fresh install, independent review, Linux-native execution,
multiseed, unseen-task and release checks were outside this scoped observation.
The full repository goal and broad acquisition/retention/recovery claims remain
open.

The next useful outcome is retention during controller-selected learning.
Existing R2.1 health and isolated fresh-task probes already exist; another probe
or diagnostic horizon would duplicate accepted work. Existing arrived cue-bank
composition has native retention evidence, but the fixed-reference stream API
explicitly refuses that composition. Add a bounded opt-in composition through
that existing API and stream, with actual consumed replay exposures separated
from arrivals and diagnostic rows. Start with literal causal-label/FIFO/bank/
budget/isolation tests; keep current-only and prequential defaults unchanged.
This dependency addresses demonstrated forgetting without changing learner
math, thresholds or outcome metrics. No new comparative run was launched here;
freeze its own small protocol after software gates.
