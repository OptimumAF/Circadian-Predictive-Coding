# Stopping preset at a fixed allocated wall time

One prospectively frozen three-initialization comparison completed through the
existing timed application. Only the adaptive policy changed, using the min2/
gradient tolerance1.0 preset from the2026-10-14 no-heldout engineering example.
Fixed6 inference/.1 inference rate/.03parameter rate, all initialization tensors,
data, seed order and original10ms/11ms timing limits stayed fixed. Evidence
instrumentation and stopping checks were included in measured call/window time.

| Seed | Fixed/adaptive updates | Fixed/adaptive window, ms | Final BCE fixed/adaptive | Accuracy fixed/adaptive |
| --- | --- | --- | --- | --- |
| 23 | 11 / 19 | 10.491 / 10.140 | .764787 / .735252 | .390625 / .437500 |
| 41 | 11 / 18 | 10.725 / 10.318 | .582205 / .569236 | .929688 / .945313 |
| 59 | 12 / 17 | 10.689 / 10.006 | .608415 / .606263 | .757812 / .757812 |

All54 adaptive updates stopped at two steps, no fallback; fixed updates used6.
Adaptive-minus-fixed final BCE differences were−.0295353/−.0129683/−.0021515;
mean−.0148850, nominal marginal95% t interval[−.0491494,+.0193794]. Accuracy
mean difference+.0208333, interval[−.0384618,+.0801285]. Both intervals include0.
The complete comparison is inconclusive about a repeatable quality benefit.

Costs:88native updates/1408training rows;488coupled evaluations/7808rows
(400inference+88candidate),88training initializations/1408rows,162gradientnorms,
108reported energy comparisons,12final forwards/1536rows. No CPC/BP/replay/probe/
health/SVD work. All attempts returned; source learners unchanged and original
budget counters exact. All initial/update tensors and original probabilities
are saved. Whole command.7701649seconds; actual trainer peak49,569,792B sampled
externally. Pilot.0788603seconds, wrapper.3237876 incl imports/primaryexport.

Equal allocation permits different actual windows, updates and row exposures;
it is not equal CPUtime, FLOPs or exposure. Instrumentation overhead is part of
this workload, not a production speed estimate. Data are fixed and n3 mixes
initialization/runtime variability; intervals are not simultaneous. No general
speed, convergence, equivalence or circadian benefit established.

R1.5/R1.5b original bounded-inference/comparison criteria independently accepted
with this scope plus unchanged accepted R1.5a kernel evidence. Positive benefit
was not an acceptance requirement. Both earlier failed timing protocols, their
unscored cases and default-adaptive cap6 negative outcomes remain unchanged.

[Frozen protocol, exact commands, full results and failures](../../artifacts/runs/r15-engineering-preset-comparison-20261017/RESULTS.md).
Full adaptation, efficacy, release and repository goals remain open. Next add a
stationary sensor control through the existing fixture/monitor/CLI so that
post-calibration false-alarm rates have a genuine stationary denominator.
