# Saved BP features support complete TRAIN separation

R2.1-TW.FR.S / [#319](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/319).
Both fixed multiclass feasibility checks found separating affine readouts for
all12 saved TRAIN rows. Independent exact-rational validation confirms every
required comparison is strictly positive. The unchanged H3 representation from
#318 therefore supports complete TRAIN separation with24 output coefficients.
The original BP8/12 and LS10/12 results were below that available solution.

This rules out an intrinsic lack of affine separability of **these saved rows**
as the explanation for those output errors. It does not establish a unique
optimizer failure, CE convergence, maximum learner capacity, native float32
behavior, task success, adaptation, generalization, matched compute or CPC benefit.
No new learner, native hidden forward, teacher labels or task rollout was run.

| Fixed comparison scope | Constraints | Solver iterations | Exact minimum margin | Accuracy in scope | Global accuracy |
|---|---:|---:|---:|---:|---:|
| Full six-class catalog |60|28|0.025165109618471033|12/12|12/12|
| Version-allowed catalog |48|24|0.028383514692311845|12/12|11/12|

Both solve statuses0, crossover iterations0, coefficient infinity norm1, maximum
reported primal residual1.1102230246251565e-16. Exact minimum margins:

```text
full:    486764012339233302841703 / 19342813113834066795298816
allowed: 34313563762946967321319 / 1208925819614629174706176
```

The allowed witness's11/12 global result is preserved: masked correctness is
not global correctness. The full-class witness separately establishes12/12 for
the complete catalog. No favorable task, seed, tolerance or solver selection.

## Frozen method and isolation

Read only accepted #318 `diagnosis.json`, its12 old TRAIN hidden rows/targets/order
and metadata. Input bytes were checked unchanged afterward. These are feature
data, not native model-state restore authority. Existing source snapshots and
all negative/SPENT runs remain untouched. PhysicalD HEAD
`d9119e80926a139a2f38460564d7eab1b6d2a2b0` plus inherited dirty tree; no production
code, dependency, runner, optimizer or default change.

With z=(h0,h1,h2,1), solve max rho subject to target logit minus each competitor
logit>=rho and every coefficient in[-1,1]. There are24 classifier coefficients
plus one auxiliary margin variable. Positive strict separation is scale
invariant, so bounding coefficient magnitude preserves existence. The auxiliary
margin does not add classifier capacity. A numerical zero or failure alone would
not have been an exact impossibility proof; both actual results instead have
positive exact certificates.

Two pure `workflow_request` calls, sealed v1/v2 with full capabilities, bind masks
to the unchanged actual source. v1 allows commit/inspect/open/retire/route;
v2 allows commit/defer/inspect/open/route. The source intentionally permits
phase-illegal actions in requests; no phase mask was added to conceal errors.
Both masks have five actions and remain constant across phases. No teacher,
dispatcher, private label source, probe, final evaluation or promotion access.

Linux Python3.12.3/NumPy1.26.4/installed SciPy1.15.3, linprog highs-ds, presolveTrue,
time_limit5s, maxiter500, primal/dual feasibility tolerances1e-7. HiGHS threads1
and random_seed0 supplied explicitly. SciPy's warning that these options are
forwarded verbatim to HiGHS is retained in every result. No fallback or tuning.
Every target-competitor margin was then checked using exact Fraction.from_float
products on saved binary feature values and returned coefficient values.

## Executed gates and costs

Stage `artifacts/runs/ticket-frozen-readout-separability-20261030/` contains frozen
PROTOCOL.md, small local feasibility/run scripts, full fixture/actual results,
static receipts and permanent per-process SPENT markers. No scientific retry.
All commands start in physical `D:/Dev/Repositories/Circadian`, with PowerShell
TEMP=TMP=D:/Dev/_Temp. Exact numeric commands, one invocation each:

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/ticket-frozen-readout-separability-20261030/run.py fixtures
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/ticket-frozen-readout-separability-20261030/run.py actual
```

Fixture process EXIT0, complete/budget_verifiedTrue,0.371058986s/75337728B. Four
two-row LPs: separable, contradictory, mask-full and mask-allowed. Margins2,0,0,2;
exact witnesses True,False,False,True, as declared. Last fixture allowed2/2 versus
global1/2. Four LP calls/eight diagnostic rows, ten constraints/62 matrix cells,
four float64 and four exact affine reads/16 rows/40 exact products. No real data.

Scientific process EXIT0, complete/budget_verifiedTrue, **0.230499616s/76111872B**.
Exactly two LPs/24 diagnostic-row exposures,108 constraints/2700 matrix cells,
two float64 and two exact affine reads/48 rows/576 exact products. Independent
reviewer added two exact reads/24 rows/576 products. Thus scientific validation
totals six reads/72 rows/1152 exact products, within the prospectively amended
independent-validation allowance (comment6524783319), not extra optimization.

Both numeric processes satisfy30s/256MiB; all individual solve5s/500iteration and
numerical work caps hold. CPU thread request1 and fixed solver seed0; internal
HiGHS arithmetic/random tie work is not instrumented. No matched-compute claim.
Zero native models/initializations, learner RNG calls, hidden forwards, SGD,
backward, latent work, replay, physics, probe/final updates or deployment.
LP optimization and affine verification are paid diagnostic computation, not free
metadata or native learner update counts.

Scoped static prefixes, each targeting stage `feasibility.py` and `run.py`:

```text
.venv/Scripts/python.exe -B -m ruff check
.venv/Scripts/python.exe -B -m ruff format --check
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent
.venv/Scripts/python.exe -B -m mypy --explicit-package-bases --follow-imports=silent --platform linux
```

Receipts preserve exact argv/output. First batch Ruff/format passed; both mypy
targets failed one reused-variable fixture/real-mask tuple inference error.
Renamed the fixture loop variable only. Batch times: first3.7249554s,
pre-guard final3.6975914s, guarded final3.7704054s. Final four checks all EXIT0.

**Engineering nonconformance:** the third static batch ran before the declared
two-batch limit was amended. Original static invocation limit FAILED, not met.
All12 commands total11.1929522s; preserve this overrun (comment6524852759) rather
than retroactively approving or deleting it. No further static batch, new LP,
scientific retry or numerical-cap renewal followed. This is a failed supporting
engineering qualification, not a change to the two scientific LP outcomes.

## Review, remaining work and next implementation

Prelaunch independent review accepted LP signs, bounds, masks and fixtures, but
caught a budget-result gate flaw: complete=True could be saved before cap checks.
Fixed before any LP: check time/RSS/conservative projected stage bytes before
writing, save budget_verifiedFalse/completeFalse on violation, and require
verified fixture status before admitting real data. Original failures remain
preserved. Final actual source and successful resource receipts were reviewed.

Independent reviewer verified all60+48 inequalities with exact rational
arithmetic and accepted the scoped scientific witness. Exactly four read-only
commands0.5529293s, zero project/numerical-library imports, solver calls, writes,
network or children; declared exact-validation cost above. Exception ended.
Lead separately reviewed source/procedure/costs. No self-review independence claim.

Skipped unchanged #317 native model tests, broad suite, clean clone, wheel,
new source/probe cohorts, native float32 inference, task rollouts and release
qualification: this local analysis changes no production behavior. Source history
and #318's unverified temporary ceiling remain unchanged/unchecked. All intentional
writes stay in the bounded D stage; no new library-temporary byte claim. No
install/download/delete/commit/push/global-setting or cache change.

Critical path now moves from feasibility diagnosis to **learning a readout that
actually reaches the available TRAIN solution**. Prefer a separately budgeted
disposable TRAIN-only CE readout fit on the unchanged H3 features, retaining24
output coefficients and source/actor/optimizer/RNG guards. Extend existing
`torch_categorical_readout_fit` and typed-reader ports rather than adding another
runner. First inspect equivalent bounded fitting/readout support; reuse existing
Torch optimization and normalization where applicable. Then qualify all four
known tasks together through the existing stream, with separate fit/readout costs
and frozen protocol. No capacity increase, source-dose/seed/weight variation or
further LP is needed first. This implementation/native qualification is unfinished;
no readout-learning or task-success result is claimed by319. Fullgoal/#21/#18/
R2.1 acquisition-retention-recovery/scientific comparison/release remain open.


Closeout storage: stage11files45,384B, below1MiB. Scoped tracked/new-document
whitespace checks emitted no whitespace errors (LF/CRLF advisories only).
The two scientific witness/cost claims above have direct receipts; the original
static-batch limit remains explicitly FAILED. Do not label this full engineering
or repository-wide acceptance. Scientific investigation criteria are evidenced,
including honest preservation of that failed supporting gate.
