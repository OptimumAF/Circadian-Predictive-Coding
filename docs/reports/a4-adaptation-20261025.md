# R2.1-A4 — native acquisition and retention, 2026-10-25

[#171](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/171).
Single `/root` implementation/tests/source-procedure-result SELF-REVIEW, not
independent acceptance. Full goal ACTIVE; broad parents OPEN. Current tracking
is GitHub; historical plans/log remain frozen under AGENTS. HEAD09ff811f and
unrelated dirty work preserved. [Usage](../minigrid-a4-adaptation.md).

## Observed behavior

The existing affine stream now supports explicit A3→A4→A3 TRAIN adaptation using
bundled A3 and fresh170 A4 TRAIN data. Original six-wake schedule, delayed labels,
one previous arrived episode per cue, learner math and game loop are reused.
Actual new full parameter states at2/4/6 pass to fixed read-only games after ALL
fits finish; no readout can select updates/data/checkpoints. The original A/B/
ABC/G/default/development modes remain. A3/A4 keep contextA but distinct native
episode IDs include branch row only in the new mode.

| Actual causal checkpoint | Cumulative fits / row exposures | Old A3 goals | New A4 goals |
| --- | ---: | ---: | ---: |
| After initial A3 | 2 / 18 | 2/2 | 0/2 |
| After A4 with joint previous TRAIN rows | 4 / 54 | 2/2 | 2/2 |
| After A3 returns | 6 / 90 | 2/2 | 2/2 |

Both initial A4 games completed16steps/return0; all ten successful games took
nine steps/return0.49375. All12games completed and closed. Source/scorer/owned
state, original cumulative budget/clock and paid source/training/phase-prefix
guards held. Process-global NumPy state comparison after imports matched.
Raw actions/probabilities/public inputs, end flags, full fitted parameters,
original receipts and source costs remain in actual.json. No extra raw reader.

Frozen acquisition gain+1.0, old retention delta0, return gain0 over two fixed
cues. **Recovery is false for both old cues because A3 was never lost.** This is
native known-TRAIN learning and retention under one conventional recipe, not
statistical reliability, unseen-layout generalization, live deployment/promotion,
matched computation, a circadian advantage or release readiness. It does not
show that joint replay was necessary: that needs the matched current-only control.
The earlier127model/168blue failure used a different recipe and remain unchanged;
this is not a paired repair or rerun of those actors.

## Data, computation and observed costs

Fresh170 TRAIN source428191B/SHA
`bce498c34c23d9ef128f9344bf7164a70edc6afa156d42e7534bb708b2e3c26f`.
Bundled TRAIN source94705B/SHA
`5a368cfa8d4d7326d5d3ab4d6663a0e8e35e1e660968a68212e627ba9203d508`.
One standard hash-only preparation read each,0.2902326s/0.5379819s separately,
then one scientific payload read each. The existing bundled decoder prepared
54TRAIN rows, selecting only18A3 rows; the unselected36 were not fitted/forwarded.
A4's18rows were TRAIN from birth, verified against original native148 and saved
296history. No168probe_test/private/final/D/E/F/bank/model/old127 inputs read.
Original170collection18teachersteps/2env/2.5406233s and clocks/budgets/counters
remain separate sunk cost, never new authority. No new teacher/source collection.

Exactly6least-squares fits/90row exposures:54arrivals (including18explicit
returning A3 rows) and36joint replay exposures, over36distinct rows. No optimizer,
backward, Torch or random model initialization/training.16TRAINscorecalls144rows
plus122live predictions/steps yielded138total predict calls/266rows.356port
projections,5031snapshot copies/15196617typed scalar-validation elements,9owners,
12environment constructions/returns/closes and122steps/returns. These counters
and reserves are not universal allocation/FLOPs accounting.

Sole scientific child EXIT0/body10.512506900005974s, parentEXIT0/10.89451310003642s,
sampledRSS85368832B. Original inner6-update90s512MiB training budget and ONE
original150s512MiB zero-update/replay readout budget spanning admission/training/
all phases were retained; no reset or renewal. Wholechild150/hard165/parent180s
caps held. Counted logicalJSON3333886B and explicit counting UTF8 copies3333886B
include16KiBconservative terminal bookkeeping reserve, well below1GiB each.
Actual physical file3123786B/log2709B/stage3177699B before final review append;
8MiB/256KiB/16MiB limits held. Physical file bytes differ from logical serialization.
Othercopy/source/RNG reserves remain separate from sampledRSS/complete allocation.

## Focused verification and preserved failures

29/29checks per platform,0skips, first numerical cohort attempt. One shared
actual CLI integration per platform executes6literal fits/90rows and the real
evaluator with12scripted two-step environments; captures reused by later tests.
Three unchanged default metadata bindings use0fits. Scripted successes are not
scientific outcomes. Causal joint-row membership/18→54→90native counters/delay1
receipts, role/order/source hashes/history/labels, wrong recipe/state/budget/
clock, changed/replaced paid prefix, describe/exclusive output/irrelevant CLI
options are covered. No real physics or actual170source input in qualification.

| Qualification | Body s | Parent s | Sampled RSS B |
| --- | ---: | ---: | ---: |
| Windows1 | 5.596421700087376 | 6.4014349 | 99651584 |
| Cached WSL Linux1 | 17.596676433 | 23.8020981 | 99328000 |

Each:6LSE90rows/40predictcalls168rows/258portprojections/1438snapshotcopies/
4089277typed validation scalars, JSON7565188B Windows/7565215B Linux with same
UTF8counter copies, RNGsame/0forbidden. Original120sbody150sparent/512MiB/call
limits held. Ordinary pytest/direct counters replace the prior slow profiler;
170's original timing failure and incomplete follow-up remain failures/unknown,
not retroactively passed by this different cohort.

First scoped Ruff/format passed but five typing errors remained in RNG stub
indexing/receipt/test-list annotations. Failure retained, fixed with explicit
types and the existing pickle state equality pattern, no suppression. Final
seven-file Ruff/format/default+Linux-targetmypy allPASS. Cumulative formatting/
failed/pass checks41.4678132s<150; touchedcache115717562B<160MiB, inherited global
603874408B separately. Product/tests unchanged since both passed cohorts and
native execution. Forced OS-timeout cleanup is source-reviewed only; normal
owned-Popen-tree driver path passed. No preflight/native retries or extensions.

Skipped fullsuite/cleanclone/newwheel/independent/multiseed/statistical/control/
matchedcompute/release qualification. No install/download/GPU/browser/CI/commit/
push/delete/unrelated reset.168's original missing global RNG witness remains
unobserved/unchecked; this newly captured comparison cannot qualify that prior
case.167timeout/166copy failure and every earlier negative/spent result preserved.

## Commands and evidence

From repository root, `.venv/Scripts/python.exe -B` and stage
`artifacts/runs/a4-adaptation-20261025/`:

- `qualify.py windows-1` →29PASS; exact parent command in windows-1-parent.json.
- Cached WSL CPU1 `timeout140s .../bin/python -B .../qualify.py linux-1` →29PASS;
  expanded command in linux-1-parent.json. No profiler or extra cohorts.
- `python -m ruff check`, `ruff format --check`, `mypy --follow-imports=silent`
  and `mypy --follow-imports=silent --platform linux` on changed files;
  format*.json/statics*.json preserve exact commands/durations and initial errors.
- `Get-FileHash -LiteralPath <declared TRAIN source> -Algorithm SHA256` once each;
  source-preparation.json/bundled-preparation.json preserve pre-run receipts.
- `execute.py` →sole admitted child EXIT0; execution.json/native.txt/work.json
  preserve command, counters, terminal metrics and actual per-stage outcomes.

PROTOCOL.md/REVIEW.md freeze data/budgets/metrics/gates before scientific access.
Only same-process terminal summary inspected, no extra raw actual/source reader.
Hash-preparation receipts describe their historical pre-run state; actual/work/
execution now prove both native payload allowances SPENT. No rerun authority.

## Critical path and exact next action

Useful native conventional acquisition/retention is now evidenced on this bounded
TRAIN pair. Before new algorithmic features, add the matched **current-only**
A3→A4→A3 control to the existing stream/composition/evaluator. Keep the SAME
source versions, cue order, seed17, max16, checkpoints, metrics/directions and
all12game outcomes; reuse171joint result without repeating it. Use the existing
non-retained six-wake path (6fits54rows/no jointreplay) with explicit A4 source
admission and exact checkpoint geometry; current code correctly requires
retention for A4 and needs this narrow, separately tested control branch.

Why: the present native gain does not isolate replay's contribution. Equal
updates with54versus90row exposures is not matched compute; preserve that cost
tradeoff. A simpler control matching or beating joint replay is valid. Freeze
one small new control protocol/source authority before fits/native data access;
no old171rerun, old127source, outcome-selected seeds, changed metrics, new
checkpoint/audit framework or invented CPC win. Learning recovery remains
unproved until an initially successful old task is actually lost and regained.
Full R2/R4/R5/G5/R7/P9 requirements and full goal remain OPEN/ACTIVE.
