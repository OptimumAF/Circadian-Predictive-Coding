# Selected TRAIN replay: native acquisition with retention

[Issue198](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/198)
completed its single frozen local observation. One conventional affine learner
with fixed ready-feedback drift and the existing arrived TRAIN bank learned
both new A4 goals while preserving both old A3 goals.

| Fixed logical phase | Actual fits / row exposures | Old A3 goals | New A4 goals | Readout |
| --- | ---: | ---: | ---: | --- |
| Initial A3, tick2 | 2 / 18 | 2/2 | 0/2 | Four fresh games |
| After A4, tick4 | 4 / 54 | 2/2 | 2/2 | Four fresh games |
| Returned A3, tick6 | 4 / 54 | 2/2 | 2/2 | Exact tick4 state/result reused |

The return phase makes no new performance observation: the complete checkpoint
state, including native counters, remained identical. Reuse was specified
before execution and validated before admitting more work. It avoids playing
duplicate episode IDs or inventing independent observations. Old performance
never decreased, so recovery is untested/null; this is retention, not relearning.
These are familiar deterministic TRAIN layouts at seed17, not generalization
or evidence of statistical reliability.

Only four of six queued packets were fitted:36current row exposures plus18prior
TRAIN replay exposures,54native rows total. Six diagnostics paid90rows. The
first four errors8/9,8/9,2/18,2/18 exceeded the frozen threshold.1. At ticks5/6,
the same oldest returning-A-blue joint packet was correct0/18, so drift waited.
Both return packets remain queued; red's return packet was not diagnosed.
The window is complete, but training_complete is false. No packet was skipped,
acknowledged, implicitly drained or granted renewed budget. This blocking
behavior remains a limitation for longer continuous streams.

The accepted [current-only trace](affine-ready-native-20261026.md) already
showed acquisition/recovery with forgetting; the older joint fixed-six control
retained both tasks with90rows. Their conclusions are reused descriptively,
without refits or raw/model reconstruction. Equal54row totals here and in the
current-only trace do not establish equal compute: packet composition, fit count,
diagnostic rows, producer assumptions and unique game work differ. No parameter
equivalence, paired comparison, controller-efficiency or CPC advantage follows.

Eight unique games ran and closed: six successes at9steps/.49375return and two
failures at16steps/0return. Unique executed return was2.9625 across86game steps.
There are twelve logical phase slots, four reused; do not pool them as twelve
independent games or compare their inflated aggregate with another study.
Fresh A4 preparation added18teacher steps/two closed environments. Total104
physics steps/10environments,4native LSE54rows/4LAPACK54rows,92predictions176rows
including6diagnostics90rows,11empty affine constructors/1525snapshot calls.
Counters and the conservative copy reserve are not complete allocations/FLOPs.

One94705-byte packaged TRAIN export was read once,54rows decoded/18A3selected;
one fresh A4 TRAIN pair supplied18rows. Plan construction paid213840input-array
bytes and18planned bank rows even though two packets remained unconsumed.
Numeric bank equivalents exclude metadata/Python overhead; whole RSS/time
includes construction, source checks, copies and diagnostics. Packaged source
preparation is pre-existing cost, not newly measured work.

Native body7.3363692s, parent7.7854912s, sampled peakRSS82,776,064bytes. Original
training6updates/90s/512MiB and separate readout0updates/0replay/150s/512MiB owners
retained their clocks and limits. Readout-owner elapsed6.0645710s includes all
preparation and training; training-owner elapsed.3700902s is not isolated fit
time. Source/actor/full readout states, paid-prefix values/identities and caller
Python/NumPy RNG checks passed. RNG baseline follows optional imports. No final,
probe, heldout, consolidation, promotion, tuning, retry or package installation.

This study directly composes production components for genuine A4 data; it does
not qualify the public A/B-only wrapper as an A4 command or establish installed
package/release readiness. Source/procedure/result **self-review**, not independent
acceptance. Reused unchanged issue197's85software checks/platform; new eight
pure reuse/summary checks/platform and scoped lint/type checks pass. Initial
fixture-only typing errors remain recorded; native attempt passed first launch.

[Frozen protocol, exact commands, costs, failures, skips and review](../../artifacts/runs/affine-ready-replay-native-20261026/RESULTS.md).
Review consumed only emitted same-process scalar evidence; actual.json was not
reopened. All earlier failed/spent studies and negative results remain unchanged.
Full repository/scientific/release goals remain open.

Before adding another controller mechanism, the next outcome is fair native
reference controls with this same bank. The existing retained-A4 readout admits
only2/4/6fits; no-learning, periodic and budget-aware references legitimately
produce0/1/3and other partial counts. The concrete next implementation is exact
0..6joint-prefix admission in the existing evaluator, preserving source/role/
budget guards and separate current-only defaults. Literal geometry/isolation/
cost tests precede a separately frozen small matched-opportunity control protocol.
This enables required baselines; it is not another capture/audit framework or
permission to rerun a spent cohort or tune from these TRAIN outcomes.
