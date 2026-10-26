# Fresh-task learnability after shared rehearsal

The existing rehearsal CLI now supports `--diagnose-learnability`, attaching
the existing reserved sixteen-step probe to copied post-rehearsal checkpoints.
An independent template creates disposable probe forks; benchmark/deployed
learners never train on probe data, and benchmark final scoring follows probes.

63focused cases/189phases, six-path lint/format/Windows+Linux types and independent
SOURCE/GO/ACTUAL review pass. The sole frozen run completed all376updates within
60s/256MiB/CPU1 limits. Full R2.1/R2.3 and release requirements remain open.

## What the fixed probe showed

Seed23, the original A→B→A stream and unchanged replay policy; reserved task C
is x0*x1>=0 on the original fixed data. Positive gain means C BCE decreased over
the fixed sixteen probe updates. Labels0/6/12/18 below count benchmark wakes;
their native states additionally contain0/4/8/12replay updates.

| Method | Wake0 | Wake6 | Wake12 | Wake18 |
| --- | ---: | ---: | ---: | ---: |
| BP | .00947800 | .00538653 | .00178348 | .00214041 |
| PC | .00002930 | -.00021568 | -.00052889 | -.00059157 |
| neutral CPC | .00002930 | -.00021568 | -.00052889 | -.00059157 |
| active CPC | .00001198 | -.00022490 | -.00049497 | -.00053885 |

All sixteen probes improved after steps1 and2. Positive counts at4/8/16 were
1/13/7. Thus a negative sixteen-step endpoint does not mean no learning occurred.
All96BCE/accuracy points are retained, with no endpoint selection or retuning.
The nine trained PC-family endpoints remain negative; no circadian advantage.

Absolute levels matter: every trained checkpoint has lower initial AND final C
loss than its retained no-replay counterpart, despite smaller within-probe gain.
For PC wake12, .72175357→.72228246 versus old .72898116→.72921041. Different
starting states/headroom and48extra benchmark rehearsal calls prevent a causal
plasticity-loss or replay-benefit conclusion. C is a fixed composition of A/B,
not independent task generalization. BP remains the strongest local control.

## Isolation and cost

All four COMPLETE benchmark reports/receipts/replay plans match the accepted
rehearsal run exactly. The four initial probe curves/health match the retained
sixteen-step experiment exactly; PC and neutral CPC full curves match. Benchmark
learning/retention/recovery outcomes remain unchanged by diagnostic work.

The run costs72wake+48replay+256probe calls,2672row exposures,128predictions,
1692coupled/188BP evaluations and32SVDs. Measured pilot0.7273175s, peak sampled
processRSS41672704B. Outer diagnostics0.0137735s/probes0.3783935s; these are local
telemetry with profiling overhead, not an equal-compute timing comparison.
Reserved arrays3840B/reference512B per observer; activations4096B/checkpoint;
snapshot arrays840B BP/PC or968B CPC; replay arrays<=384B. Copies, native forks,
Python metadata and reports cost additional memory. Importance/age remain
unavailable for coupled native models, without invented proxies.

Exact protocol, raw results, all curves, checks and independent reviews:
`artifacts/runs/r21-rehearsal-learnability-20261016/{SCOPE,RESULTS,REVIEW}.md`
and JSON files. See [usage](../stream-rehearsal.md).

Next: a recent-only rehearsal control with equal memory/exposure limits through
the existing helper/CLI, under a separately frozen protocol. This can test whether
historical-memory selection adds value beyond additional practice, before new
algorithmic features. All prior negative results and unfinished tasks remain.
