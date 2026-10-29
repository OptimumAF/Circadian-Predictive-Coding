# Public MiniGrid summary observability

[#233 / R2.7-MG.O](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/233).
One prospectively frozen data-only diagnosis completed. The existing eight
public feature summaries distinguish the four original A/B TRAIN packets at
the default feature metric's scale. **No feature replacement is justified by
this diagnosis.** Runtime alarms, uncertainty, residuals and learned outcomes
remain unmeasured. This is source/procedure/result self-review, not independent
acceptance or repository-wide readiness.

## Frozen question and observed result

Each of the four fixed original A-blue/A-red/B-blue/B-red episodes supplies
nine296-feature rows. The existing `public_feature_summary` produces eight
public coordinates per row: initial/current image-channel means and direction.
The per-packet mean of those coordinates is exactly the feature vector entering
the existing detector. No labels, predicted probabilities or context-conditioned
feature choices enter this reduction.

The default detector's distance is **mean absolute coordinate difference**, not
maximum coordinate difference. All six unordered packet pairs were declared and
reported; none was selected or omitted after inspection.

| Pair | Mean absolute difference over8 | Max coordinate difference | Raw row-aligned max | Summary row-aligned max |
| --- | ---: | ---: | ---: | ---: |
| A-blue / A-red | 0.0404383975813 | 0.296296296296 | 2 | 1.333333333333 |
| A-blue / B-blue | 0.536262282691 | 2 | 2 | 2 |
| A-blue / B-red | 0.502853363568 | 2 | 2 | 2 |
| A-red / B-blue | 0.495823885110 | 2 | 2 | 2 |
| A-red / B-red | 0.462414965986 | 2 | 2 | 2 |
| B-blue / B-red | 0.0406651549509 | 0.296296296296 | 2 | 1.333333333333 |

The default feature threshold remains0.25. Same-cue A/B distances exceed it;
within-context cue distances are smaller. These are descriptive source contrasts,
not independent samples, false-alarm rates or classifier accuracy. Row-aligned
maxima compare corresponding demonstration indices, not equivalent task states.
The source contrast includes public direction and color differences; this does
not isolate door-position information or demonstrate general layout detection.

For fixed packet mean vectors v_i, any two convex mixtures have distance at most
the largest pairwise mean-L1/8 distance: expand their difference as
sum(a_i b_j (v_i-v_j)) and apply the triangle inequality. The observed diameter
is0.5362622826908542. Adding the frozen64*float64-epsilon guard gives
0.5362622826908684. Therefore the frozen unreachable test is **false**: this
upper bound does not rule out threshold crossing. It does not establish a
runtime onset on any particular schedule or learner.

The eight-coordinate means and all full-precision reductions are in
`artifacts/runs/minigrid-shift-observability-20261027/terminal.json`.
No raw observations, targets, weights or prediction arrays were serialized.

## Execution, costs and checks

The sole data invocation was:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/minigrid-shift-observability-20261027/execute.py
```

Outer and checked child EXIT0, both terminal/work statuses complete. Body
0.95307509997s, checked parent1.3672485000s, whole tool command2.2754917s;
sampled peak RSS54,059,008B. RSS covers before optional imports through scalar
serialization/write and sampler exit; the final small work receipt is outside
RSS and inside parent time. Frozen caps30s body/60s parent/50s checked child/
256MiB all held. This result was not retried.

The existing reader performed **one** expected-SHA public TRAIN parse of94,705B,
decoding6episodes/54rows. Four original A/B episodes were selected; the other
C18rows were decoded and charged but not summarized. Four summaries/36rows,
six mean-distance pair reductions and six raw/summary contrast components of
those same pairs completed. All attempted/returned counters agree; no unreturned
work is unknown. The existing checked-launch preflight performed metadata/stat
checks, with no extra payload read. The SHA was reused from earlier metadata;
this task performed no preliminary opaque hash read.

Source SHA256:
`5a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508`.
Original source inputs and Python/NumPy RNG states remained unchanged. No seed,
metric, threshold, feature selection or baseline was adjusted after observation.

Retained feature/summary/mean arrays total87,808B. This is a named subset, not
total allocation or peak memory; decoded Python objects, targets, guard copies,
validation and temporary arrays are covered by RSS. Scalar terminal3,561B.
Learners, snapshots, budget owners, model initialization, updates, predictions,
fits, solvers, gradients, latent work, teacher, physics, final/probe/private data,
old scientific raw reads and detector/incident simulations are all0. Genuine
native construction, budget construction and LAPACK entry were patched to refuse.

Before data access, four pure checks/platform passed on Windows/cached Linux:
spatial-permutation summary invariance, equal-vector/boundary behavior,
mean-L1 versus coordinate-max distinction, and a larger contrast. Each used
two synthetic summary calls/18rows and24 mean-pair reductions;0 models/payloads.
Windows body0.00495119998s/parent0.82335790014s/RSS47,587,328B; Linux
0.00555403000/5.18418049999s/RSS48,295,936B. Caps30s body/60s parent/256MiB held.
Exact commands, CPU1 environment and terminal outputs are in
`protocol-{win,linux}-parent.json` and `protocol-{win32,linux}.json`.

Four helper files passed final Ruff format/check and default/Linux-target mypy.
Initial mypy reported2 binary/text file-handle errors in the producer; distinct
handle names fixed them without ignores, before the data attempt. The original
failed receipt remains in `statics.json`; final EXIT0 checks are in
`statics-final.json`. Total static time over all attempts14.9441880002s/60s.
The GO comment's incomplete shorthand is superseded by this exact receipt value.
One inherited111,280,128B cache file was mtime-selected, below160MiB; this is
touched storage, not growth or separate copies. Thirteen source/protocol pins
were frozen before execution and still match in postflight.

`PROTOCOL.md`, qualification, checked parent, `data-work.json`, scalar terminal,
pure/static receipts and postflight evidence remain in the same stage. No
production source was changed. No new dependency, framework, API, package, agent,
Git mutation, install/download, GPU or remote CI was used.

Full suite, new native learned experiment, clean clone, package/install, UI and
independent review were skipped. Unchanged #232/#230 software is reused at its
original scope. The separately spent #231 native attempt still has unknown
signal outcomes; this data-only result does not complete or replace that study.
#227, all earlier failures and all BP/PC/neutral CPC/active CPC negative results
remain unchanged. There is no CPC, learning, retention/recovery, deployability,
generalization, efficiency or release claim.

## Next action

Keep the existing summary and thresholds. The diagnosed source observability
prerequisite is satisfied, while the original six-arrival schedule's reference
contamination and earliest-onset6 limitation remain.

Next precise action: freeze and review a small **long-phase timing protocol** on
the existing stream/adapter, using a clean three-packet A reference and declared
four-arrival A/B/A phases, with fixed stationary/recurring and no_learning/
periodic controls. Record pending FIFO learning lag, false/onset/continued/clear
events, failed work and every fit/prediction/diagnostic cost. Use #232's already
qualified bounded horizon and the supported budget contract. Do not add another
API, capture or audit layer. Any execution needs that separate prospective
protocol and software gates first.

This would be a new exploratory phase-duration question informed by these TRAIN
contrasts, not a replication, continuation or confirmation of spent #231's
six-arrival study. Preserve its unknowns and budgets. Do not present development
data inspection as blind confirmation or change seeds, thresholds or comparison
directions to obtain a win. Broader R2.7/#21/#23/#33/full goal remain open; original
bounded R2.1 diagnostics/isolated probe and matched baselines already exist.
