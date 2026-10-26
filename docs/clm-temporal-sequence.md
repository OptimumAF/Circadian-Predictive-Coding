# CLM learning on changing ticket rules

The fixed native temporal example connects the existing temporal-rule corpus to
released frozen CLM features, the owned head learner and delayed ExperienceInbox.
It measures changing-rule learning, stable-task retention and conditional recovery.

## Fixed behavior and isolation

Four paired updates on invoice/legacy/password precede four on invoice/legacy/
parcel and four on invoice/legacy/password again. The return retains current v2
rules. The paired group-masked native objective uses global catalog group codes;
duplicate positive actions remain positives. Only independent head/scale copies
learn. The encoder and released control stay frozen.

Eight query strings (bare TRAIN concepts and development concepts plus ` please`)
and four catalog descriptions are encoded once. Raw frozen features are reused;
head projections recompute. Four development readouts and an additional frozen-A
v2 readout are captured without labels during the stream. Scores are computed
only after all twelve updates. Original consent, clocks, delayed receipts and
permanent work charges remain. The default path has no final, probe, replay or promotion inputs.
The separate opt-in [reserved checkpoint probes](clm-reserved-probes.md) have
explicit roles, consent and budgets; the main learner is unchanged.

The released control is read once per candidate set. A frozen-A v2 observation
before B separates new learning from tool masking and existing knowledge. Report
all concepts and phases using the existing accuracy, raw NLL, Brier, five-bin ECE
and obsolete-rule metrics. Zero target probability retains infinite NLL status.
Already-correct parcel is not new acquisition. Password recovery requires correct
at A, wrong after B and correct on return. A removed legacy tool alone is not
learning the still-allowed invoice route-to-inspect rule.

**Why this:** the existing corpus supplies real rule changes and a new B task;
the preceding refund example was already correct before training. The tiny
suffix split remains a development diagnostic, not broad generalization,
workflow execution, consolidation, matched compute or a CPC advantage.

## Modules

```text
src/app/clm_temporal_sequence.py      fixed use case, receipts and raw readouts
scripts/run_clm_temporal_sequence.py  local assets, feature binding, scoring and CLI
tests/test_clm_temporal_sequence.py   literal-feature behavior and refusal checks
tests/test_clm_train_catalog_diagnostics.py  opt-in TRAIN diagnostics and refusals
```

App/core do not import backend adapters or infrastructure. The outer script
composes the established ports and existing temporal evaluator. No new generic
runner, checkpoint chain, dependency or projection cache.

## Commands

The CLI takes `--model-dir`, `--head`, `--output`, matching the
[frozen encoder example](clm-frozen-encoder.md#run-using-prepared-local-assets).
It refuses an existing output. Use prepared pinned local assets and the qualified
optional CPU environment; no download or GPU is performed. Reproduction requires
a new finite allowance, not automatic reuse of this completed scientific budget.

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_clm_temporal_sequence.py
.venv/Scripts/python.exe -m ruff check src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_temporal_sequence.py
.venv/Scripts/python.exe -m ruff format --check src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_temporal_sequence.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_temporal_sequence.py
```

The [frozen protocol](../artifacts/runs/clm-temporal-sequence-20261022/PROTOCOL.md)
contains prospective resource bounds, controls and directions. Independent
[source/software/result review](../artifacts/runs/clm-temporal-sequence-20261022/REVIEW.md)
separates implementation correctness from task usefulness. Extend through existing
corpus/learner ports, freeze a new small protocol before comparative work, and
preserve all previous negatives and failures. Full R5.1/R5.3/G5/release remain open.

## Observed development result

The sole native case completed all12updates/36pairedrows within its frozen
allowance. Learned correct counts were1/4→3/4→3/4→3/4 at baseline/A/B/return;
unchanged control was1/4 under v1 and3/4 under v2. Frozen-A v2 was0/4. Invoice
learned the v1 route but kept that obsolete choice through B/return; correct
inspect probability on return was.0664 versus route.9145. Learned v2 ties the
unchanged accuracy while its NLL/Brier/ECE and obsolete-rule errors are worse.

Parcel corrects relative to pre-B frozen-A and survives return, but unchanged
v2 already answers it correctly. Password is wrong after A and correct after
B/return; this supports neither initial A retention nor recovery after loss.
Current old-domain return is2/3, so reliable current-rule learning is unfinished.
This small mixed/negative result does not support a circadian advantage.

29focused cases/platform pass after one exact metric-status test correction;
scoped lint/format/mypy Windows+Linux pass. The original failed attempt and
source-only typing/format repairs remain recorded. Including repairs, tiny
fixtures use79logicalpairs/38updates/114rows. Sole native work:1encoderforward,
12sequences/95IDs/108paddedslots,20lifetimeheadpairs/38headforwards,
204.319512s wrapper and sampledRSS15,790,100,480B (nottruepeak).
[Full outcomes, all metrics and costs](../artifacts/runs/clm-temporal-sequence-20261022/RESULTS.md).

The original next action, TRAIN-only full-active-catalog diagnostic margins,
is now delivered below in this existing driver. Current B/return in-batch negatives omit the still-allowed route; that
could explain the low paired loss without global rule replacement, but the
suffix split is another possibility. Distinguish them before a coverage repair
or consolidation/probe experiment; freeze a new bounded protocol first. No
further model run is authorized by this completed case's allowance.


## Optional TRAIN catalog diagnosis

`--train-catalog-diagnostics` adds one readout of each phase's actual bare TRAIN
queries against all active candidates, at updates4/8/12. It adds no training or
encoder work. Diagnostic inputs must match the original phase TRAIN states and
active catalog; their role remains TRAIN. Default reports and work are unchanged.

Three raw3x3 matrices are retained unscored during training. After the fixed dose,
`train_catalog_diagnostics` reports all nine query choices, probabilities, correct
labels and target-minus-best-alternative margins. It also reports target-minus-
obsolete probability margins only when the obsolete candidate is still active;
removed tools have unavailable probability/margin, not zero or successful memory.
These are probability margins, not logits or a replacement scientific metric.

This diagnosis can distinguish observed full-catalog TRAIN fit from failure on
the minimal development suffix. It cannot by itself show that missing in-batch
negatives caused an error. The separate prospective
[diagnostic protocol](../artifacts/runs/clm-train-catalog-diagnosis-20261022/PROTOCOL.md)
keeps all original training settings and requires prior trajectory parity; its
software and sole native execution gates passed for the recorded case below.
No final/probe/replay or controller selection uses these readouts.


### Executed TRAIN catalog result

The independently reviewed diagnostic reproduces the original development
probabilities, loss and receipts exactly. Bare TRAIN full-catalog accuracy is
2/3 afterA,1/3 afterB,2/3 onreturn. Invoice still selects route on bare TRAIN:
inspect−route probability margin is−.734397 afterB and−.727767 onreturn. Thus
its failure is not solely caused by the development suffix. Missing route
negatives remain a hypothesis rather than a demonstrated cause.

Preserve the opposite cases: afterA password is correct on bare TRAIN but wrong
on development, legacy is wrong on TRAIN but correct on development; afterB
parcel is wrong on TRAIN but correct on development. No bare password observation
duringB exists, so these strings cannot be mixed into a retention/recovery curve.
All nine rows and unavailable obsolete choices remain in the
[diagnostic evidence](../artifacts/runs/clm-train-catalog-diagnosis-20261022/RESULTS.md).

54focused checks/platform (29default+25new) and scoped lint/format/type checks
pass on firstcohort. Sole diagnostic case207.114953s wrapper,1encoderforward/
108slots/23headpairs44headforwards/12updates36rows;sampledRSS15,793,131,520B,
nottruepeak. Prior weights/features were not saved, so the diagnosis paid for a
full reload and training trajectory: both temporal studies total24updates72rows/
2encoderforwards/411.434466s wrapper, not just six added head forwards.

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_clm_temporal_sequence.py tests/test_clm_train_catalog_diagnostics.py
.venv/Scripts/python.exe -m ruff check src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_temporal_sequence.py tests/test_clm_train_catalog_diagnostics.py
.venv/Scripts/python.exe -m ruff format --check src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_temporal_sequence.py tests/test_clm_train_catalog_diagnostics.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_temporal_sequence.py tests/test_clm_train_catalog_diagnostics.py
```

The optional explicit support implementation and its separately frozen comparison
are described below. Original diagnostic outputs, failures and provenance remain
unchanged. Full parent claims stay open.


## Optional catalog support recipes

`--catalog-support route` appends explicit NEW TRAIN query `original workflow`
with target route to each B/return packet. Original v2 cases retain their targets.
`--catalog-support repeat` instead duplicates invoice with current target inspect.
A packets stay three rows, B/return have four: twelve updates and44TRAIN exposures
per arm. Default none keeps the original36-row recipe and historical results.

`--compare-catalog-support` runs repeat then route with independent mutable heads/
scale/AdamW/OneCycle, one shared frozen encoder and two shared released control
readouts. It enables TRAIN catalog diagnostics. Both fixed trajectories finish
before any scoring. Development remains the same four concepts; support data is
TRAIN-only. The TRAIN diagnostic now reports3/4/4rows and31query/24candidate
charges per arm; disabled defaults retain their9rows/29query behavior.

The comparison tests two explicit learning recipes. Equal row/update counts still
differ in class/group mix and unique information; no pure negative-only cause or
CPC benefit follows from a result. All original targets, metric directions and
negative results remain. Frozen small
[protocol](../artifacts/runs/clm-catalog-support-20261022/PROTOCOL.md) and
[independent review](../artifacts/runs/clm-catalog-support-20261022/REVIEW.md)
gate the sole native comparison. No sweep or accuracy-based retry.

```text
src/core/temporal_catalog_support.py   pure new TRAIN semantics and recipe selection
src/app/clm_temporal_sequence.py       existing inbox stream with declared packet costs
scripts/run_clm_temporal_sequence.py   existing CLI, shared encoder, independent arms
tests/test_clm_catalog_support.py    new packet/owner/scoring/default checks
```

Run the existing CLI with prepared local --model-dir/--head/--output arguments,
adding either --catalog-support or --compare-catalog-support. Existing outputs are
refused. The guide describes reproduction; a new experiment still needs its own
finite protocol/allowance. No dependency, download, GPU or environment change.
[Decision rationale](adr/ADR-0539-explicit-train-catalog-support.md).

Catalog learning can also compose these existing recipes with
`--catalog-training --compare-catalog-support`; its own qualified software and
separate frozen observation are described in the
[catalog support guide](clm-catalog-learning.md#explicit-train-support-recipes).
This is a distinct objective/candidate composition from the historical paired
case below. All data-mix qualifications and support/probe restrictions remain.

### Executed support comparison and extension boundary

The fixed comparison completed once: repeat development correct1/4→3/4→3/4→3/4;
route-support1/4→3/4→4/4→3/4. The route recipe fixes invoice afterB but loses the
revision on developmentreturn, despite bareTRAIN invoice remaining correct.
Its new auxiliary route TRAIN example is wrong at both endpoints. Route improves
NLL/Brier/ECE versus repeat/unchangedv2, yet returnaccuracy ties and has one extra
obsolete error versusunchanged. Stable password startswrong afterA; no recovery
is demonstrated. All endpoints/TRAINrows/negatives and recipe/data-mix caveats:
[complete evidence](../artifacts/runs/clm-catalog-support-20261022/RESULTS.md).

94checks/platform pass; scoped Ruff/format/mypyWindows+Linux pass after a type-only
annotationrepair. Existing numerical passes were retained without runtime rerun.
24updates88rows/one13sequence encoder/84headforwards,227.479560s wrapper,
15,785,500,672BsampledRSS (nottruepeak). This is development evidence, no final/
CPC/consolidation/generalization/release claim.

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_clm_temporal_sequence.py tests/test_clm_train_catalog_diagnostics.py tests/test_clm_catalog_support.py
.venv/Scripts/python.exe -m ruff check src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_temporal_sequence.py tests/test_clm_train_catalog_diagnostics.py src/core/temporal_catalog_support.py tests/test_clm_catalog_support.py
.venv/Scripts/python.exe -m ruff format --check src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_temporal_sequence.py tests/test_clm_train_catalog_diagnostics.py src/core/temporal_catalog_support.py tests/test_clm_catalog_support.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent src/app/clm_temporal_sequence.py scripts/run_clm_temporal_sequence.py tests/test_clm_temporal_sequence.py tests/test_clm_train_catalog_diagnostics.py src/core/temporal_catalog_support.py tests/test_clm_catalog_support.py
```

Extend next through explicit probe roles and a disposable owned CLM fork, keeping
all benchmark/deployed heads and counters unchanged. The generic float64probe
and CLM4096Dfloat32 pairedTRAIN contract are incompatible; do not widen TRAIN or
convert heldout labels to force admission. Use tiny literal software checks first,
then a separately frozen small reserved-task protocol; no automatic further
native case or hyperparameter/data search under this completed stage.


## Optional reserved checkpoint probes

`--reserved-task-probes` runs four disposable warm-head six-update probes at
baseline/A/B/return. Each has independent consent, budget/clock/inbox/receipts
and TRAIN/suffix-test/current-old-task before/after readouts. Scores wait until
all benchmark and probe TRAIN finishes. Benchmark defaults remain12updates36rows.
The16-sequence encoder bound means this flag cannot combine with catalog support
or comparison; rejection occurs before loading. No probe weights are installed.
[Fixed task, cost protocol and commands](clm-reserved-probes.md).
