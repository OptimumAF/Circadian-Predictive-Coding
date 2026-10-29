# Native long-phase shift timing

[#234 / R2.7-MG.L](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/234).
One exploratory four-case native observation completed. In both recurring
controls, the existing feature channel raises an onset at arrival6, one admitted
arrival after B starts, and clears at the first returned-A arrival9. It produces
**no new return onset**. Neither stationary control produces an onset in any
channel. Uncertainty and residual channels remain silent throughout.

This is source/procedure/result self-review, not independent acceptance. It
establishes a local timing observation, not useful training control, task success,
reliable recovery, CPC advantage or general deployability.

## Frozen procedure and results

Four cases, in declared order: stationary/no_learning, stationary/periodic,
recurring/no_learning, recurring/periodic. Each has12 delay-one TRAIN arrivals,
nine296-feature rows per arrival, seven classes, a separate cold conventional
affine source/actor/candidate/reader and original12-update/90s/512MiB budget.
Completed replay is0; the unsupported replay limit is unset. No signals feed
policy decisions. ReferencePolicy and ShiftPolicy defaults, public eight-feature
summary, metrics, seeds, FIFO and comparison directions stay unchanged.

Stationary repeats A-blue/A-red six times. Recurring is A-blue/A-red repeated
twice, B-blue/B-red repeated twice, then A-blue/A-red repeated twice: A1–4,
B5–8, returned A9–12. Only per-arrival identity aliases change. The first three
reference packets therefore contain only A. This is context-clean, not a fixed
model reference: periodic candidate updates at reference arrivals are0,0,1.

Primary detection requires a new onset within the B or return phase. Secondary
clear events were declared before execution and do not replace missed primary
onsets. Latency is measured in admitted **label arrivals**, not from unlabelled
feature observation; every callback occurs after its delay-one labels arrive.

| Condition / policy | Uncertainty onsets | Feature onsets | Residual onsets | Feature clears | Fits / rows | Pending |
| --- | --- | --- | --- | --- | ---: | ---: |
| Stationary / no_learning | none | none | none | none | 0 /0 | 12 |
| Stationary / periodic | none | none | none | none | 6 /54 | 6 |
| Recurring / no_learning | none | 6 | none | 9 | 0 /0 | 12 |
| Recurring / periodic | none | 6 | none | 9 | 6 /54 | 6 |

Feature B detection is2/2 recurring cases, latency1 each. Primary returned-A
onset detection is0/2; secondary return clear is2/2, latency0 each. Uncertainty
and residual miss both declared changes in both recurring cases. Stationary
false-onset counts are0 in both controls/all channels. These deterministic
repeated TRAIN streams are not stationary-noise calibration or false-positive
rate estimates.

Recurring feature drift at arrivals5–10 is
0.274870874276,0.506078357269,0.506078357269,0.506078357269,
0.237947215923,0.006739732930. The unchanged threshold is0.25. The feature
incident starts6, continues7–8 and clears9. Stationary drift is0.006739732930
through monitoring. These feature traces agree across policies because the
fixed public inputs are the same; that equality is not a method advantage.

### Learning lag is separate from detection

Periodic processes only the first six FIFO packets by the end of twelve
arrivals. Its update applications are at ticks2,4,6,8,10,12 for packets that
arrived at1,2,3,4,5,6; waiting delays are1,2,3,4,5,6 ticks. In the recurring
case, the first B fits occur at10 and12, after A has already returned at9.
Returned-A packets remain untrained. The feature alarm does not change this
queue or prove that learning follows the current context. No implicit drain,
acknowledgement, packet skip or budget renewal was introduced.

Both windows and training status remain explicit: all four windows complete;
no case fits every queued packet. Known FIFO lag is a limitation, not a reason
to change the frozen policy or claim recovery from the alarm clear.

## Costs, validation and preservation

Artifacts: `artifacts/runs/minigrid-shift-long-phase-20261027/`.

```powershell
.venv/Scripts/python.exe -B artifacts/runs/minigrid-shift-long-phase-20261027/execute.py
```

First/only launch, outer/checked child EXIT0. Body1.8890336999s, checked
parent2.0866688001s, whole tool command2.7468231s; sampled peak RSS58,220,544B.
Original CPU1/120s body/150s parent/140s child/512MiB caps held. RSS covers before
optional imports through scalar serialization/write and sampler exit; final
small work receipt is outside RSS and inside parent time. Fixed case order and
small wall times do not establish comparative efficiency.

All attempted counts returned:16 empty native owners,912 snapshot calls,
12 fits/108 training-row exposures,12 LAPACK solves/108rows,48 readonly
prediction calls/432rows,48 arrival callbacks and48 categorical metric calls.
Each periodic case pays6fits54rows; no_learning pays0. No equal-compute claim
follows from equal arrival/diagnostic access. Fit internals and state-copy work
are included in fits/time/RSS; these method counts are not exhaustive FLOPs.

One expected-SHA public TRAIN parse of94,705B decodes all6episodes/54rows;
the C18rows are decoded but unused by these schedules. There is no new opaque
hash read. Original collection is sunk work, not remeasured. Repeated A packets
are explicit stream exposures, not new independent examples or free replay.
No gradient/latent work, model random draw, teacher, physics, final/probe/private
task data or old scientific raw artifact read occurs. The actor is never trained
or published; only the isolated candidate changes.

Each adapter reports525,312 explicit input/guard/summary/prediction-array bytes
created. Peak scorer numeric float64-equivalent bytes are0 for cold readers and
33,424 for periodic readers. These are named accounting subsets; Python/state,
projection and validation temporaries are additional and included in RSS.
Observer-internal timed subtotals are .0352601993/.0643649008/.0342760000/
.0602397001s in case order, excluding surrounding stream guards/copies; they
are not complete diagnostic overhead or per-channel timing.

The existing finite producer/meter/projection/checked launcher were reused with
a twelve-arrival schedule and common `run_window` helper. Four literal fixtures
exercise that exact budget construction, stream, preservation and capture path.
The original spent producer was not edited; `producer-from-231.patch` records
the new producer's delta. This creates no new orchestration framework.

Twelve producer checks/platform pass: eight schedule/onset/continued/clear/
partial/projection checks and all four actual scripted windows. Per platform:
16 scripted owners,688 snapshots,12 scripted fits108rows,48 scripted predictions
432rows/48callbacks48metrics;72 synthetic packet encodings. Genuine Native
constructors/fit/predict and LAPACK are patched to refuse. Windows body/parent
1.9412058/2.2675917s, RSS69,537,792B; cached Linux3.087899814/8.0166596000s,
RSS67,661,824B. All engineering caps held. Exact commands/env/outputs are in
`protocol-{win,linux}-parent.json`, with receipts in `protocol-{win32,linux}.json`.

All four helpers pass Ruff format/check/default+Linux-target mypy, first attempt
with no errors. A final source-only decoded6/54 assertion was also checked by
final statics and the actual data run. Total formatting/static time over all
attempts20.0019125999s/120s. One inherited111,280,128B cache file was touched,
below160MiB; this is mtime-selected storage, not growth. Existing #232/#230
core/adapter evidence is reused without repeating its baseline cohort.

Source, actor, input episodes, global Python/NumPy RNG, original budget object/
limits/start/clock/sampler, prior paid-case identity/value proofs and native
bindings all pass final checks. All24 frozen source/protocol pins still match.
Scalar terminal77,473B/512KiB; stage146,063B before postflight/final receipts,
below10MiB. Four case reports/twelve slots are fixed; only the latest protected
checkpoint is retained for final-state checks. No weights, raw observations,
targets, prediction arrays or unbounded success-state chain is serialized.

No test, native, capture or static failure occurred in this increment. No retry,
scope/budget extension, installation, download, GPU, agent, Git mutation or
remote CI was used. No production source or unrelated dirty file changed.
Full suite, clean clone, wheel/install, UI, native Linux, independent review,
withheld/final evaluation and scientific CPC/release checks were skipped.

This is a distinct exploratory phase-duration question, informed by #233 TRAIN
contrasts, with a horizon implemented/tested before that diagnosis. It is not
blind confirmation, replication or continuation of spent #231's context-mixed
six-arrival study. Its unknowns and frozen budget correction proposal remain
unchanged. #227/#214 and all other failed, partial and negative evidence retain
their original meanings. No baseline, source, seed or metric was changed to
manufacture a circadian win.

## Critical path and next implementation

This scoped timing observation is complete; broader R2.7 stationary-noise,
general deployment, recovery and full-goal requirements remain open. Do not
automatically add another horizon, calibration, capture or provenance increment.

Source/evidence reassessment found useful conventional learning already exists:
the [current-only chooser case](minigrid-controller-learning-native-20261026.md)
learns, forgets and relearns; the [bank case](minigrid-controller-bank-native-20261026.md)
acquires both new cues while retaining both old cues. Existing forced-wake trials
already use `ReferencePolicy(period=1)`. Rebuilding that baseline would duplicate
accepted work. [Interleaved validation](minigrid-controller-validation-native-20261026.md)
is still known A3/A4 TRAIN geometry, not unseen-task transfer. The
[correct-ready prefix](correct-ready-window-native-20261026.md) supplies useful
periodic/drift outcomes but its fitted endpoint is unknown/spent, not a completed
cohort. Reuse these results and original bounded R2.1/matched-family evidence.

Next precise implementation action: add one explicitly declared **nuisance-shift
TRAIN validation recipe** to the existing controller-validation path, using
separate validation source episodes and matching game-layout bindings after
the chooser freezes. Preserve the known recipe/defaults and original fitting
episodes. Current `minigrid_controller_validation` only reindexes the same
owning study's episodes, so it cannot currently test an input domain withheld
from chooser fitting. Begin with literal isolation/source/readout-binding tests;
do not launch another native study as part of that software increment.

Why this: the present feature signal recognizes an input contrast, while a
useful controller must distinguish a contrast requiring learning from an
irrelevant input change. Existing `wall_color` layout variation offers a bounded
same-task nuisance test without inventing an environment or using reserved F
probe layouts. Keep both nuisance domains on the old A3 geometry so the
validation task itself does not become another A3-to-A4 geometry shift; bind
both TRAIN collection and game endpoints to that declared layout. Chooser
fitting remains the original grey A3/A4 case. Freeze the exact palette, TRAIN collection, controls, objective,
correct-acknowledgement/readout options and costs before any subsequent physics
or model experiment. Validation endpoints must not feed chooser fitting,
promotion, tuning or final/probe selection; this is out-of-fitting-source TRAIN
validation, not relabelled final or probe data. Preserve #214's missing fitted
outcome and every original negative. Full R2/R4/R5/R7/P7/P9/science/release and
the repository goal remain open.
