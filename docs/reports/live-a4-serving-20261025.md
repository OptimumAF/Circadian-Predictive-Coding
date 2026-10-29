# R4.2-A4.L / R7.7 — fresh private learning with published serving

[Issue182](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/182)
connects the accepted A4 adaptation recipe to one live runtime and the published
actor's existing sharing gate. [Usage](../promotable-minigrid.md#fresh-a4-private-learning-with-published-serving).
Full repository goal and broader learning/science/release parents remain open.

## Delivered behavior

New `scripts/minigrid_live_adaptation.py` composes the existing collector,
arrived-cue catalogue, ActorShadowRuntime, ResourceSharedRuntime and native game
ports. Existing CLI/published script dispatch adds `--example-affine-live-a4`.
Default two-game mode remains unchanged. App/core/learner/math are not modified.

Original native6-update120s and readout0-update0-replay150s512MiB owners precede
data/models; same gate6admissions/1request/1perpoll and runtime max6/consolidation0
persist throughout. A3 two fits→fixed existing B guards→one publication, private
A3/A4 readout, fresh A4 teacher collection, private A4 fits/readout/actor serving,
retained A3 return fits/readout/actor serving. Existing delay-one record helper
releases at1..6. Fresh experience2 queues at tick2 after collection without
releasing label1 twice. No frozen stream crosses publication.

Six fits90row exposures comprise54arrival exposures+36replay-like exposures
from36unique original rows; eighteen fresh teacher steps are separate. Returning
A3 packets are retained exposure, not novel feedback. B18guard-only/C18unused.
Fixed private12games at2/4/6 and actual shared actor4games after4/6 use rawK7
firstargmax/seed17/max16/known TRAIN. Outcomes cannot select training or promotion.
Actor remains c0gen1/2fits18rows, candidate basea0 cannot authorize another
publication. No rebasing, checkpoint chain or additional publication attempt.

Original source/actor/control/receipt and finite paid output pins validate before
further work. Post-fit check refuses prior receipt mutation before accepting new
returned state. Failed native fit preserves permanent receipts/charges and
explicit unknown model readback; failed collection/game retains paid prefix.
At most2collection+3private+4actor+6poll+4named output records are retained, not
an unbounded serving-success chain. Live output8MiB is prospective because
accepted173's12game report was3.34MB plus this mode's4actor trajectories. Original
two-game4MiB cap stays; stage24MiB. No evaluation/probe/final authority is added.

## Software checks and failures

26distinct checks qualify/platform (24new+2explicit original default/describe
regressions). Real runtime/controller/collector/sharedactor/evaluator/episode
ports execute with literal fits/probabilities/environments;0genuine numerical/
physics/randommodel work. Literal success is routing evidence, not task utility.
No skips/foreign collection. Focused final prefix checks run on final source.

Windows initial21:8PASS9FAIL4setupERROR. Root literal train fixture passed NumPy
scalars to strict public codec; production correctly converts Python floats.
The setup failure cascaded into dependent checks. Scalar conversion fixed, all
13affectedPASS. Native work0; failed costs retained. Self-review then tightened
equalreport replacement/actualgate/source tamper checks3PASS and found post-fit
receipt-prefix absorption risk. New in-fit tamper+positive complete case2PASS.
Linux full22PASS, then those2PASS. Final self-review adds exact paid-list length,
logical-clock/inbox-budget and candidate-version pins; three directed refusal
checks run on that final tree. Source/procedure/result SELF-REVIEW, not
independent acceptance. Scoped seven-file Ruff/format/default+Linuxmypy pass.
Static-only unusedimport/types/missingmetadata/nativehelper errors retained and
fixed; no failed native work during software checks.

| Run | Body seconds | Parent seconds | Sampled RSS bytes |
| --- | ---: | ---: | ---: |
|Windows initial failed |5.776524400 |7.263146800 |106332160 |
|Windows affected13 |30.430566100 |31.518657800 |102940672 |
|Windows changed/new3 |1.449514500 |2.680597100 |93474816 |
|Windows final2 |20.277632200 |21.385258000 |95924224 |
|Linux full22 |33.467661135 |40.188775700 |97599488 |
|Linux final2 |21.945428035 |29.057660500 |95354880 |

Final combined source then passed three controls cases/platform and one complete
positive CLI case/platform. Windows positive18.333727500sbody/20.069105100sparent;
Linux positive21.223776340sbody/27.099078900sparent. Linux controls10.427285348sbody/
17.361889300sparent; Windows controls1.406893100sbody/2.435236300sparent.
Final7file statics-combined pass. No semantic source/test edits afterward.

Cumulative Windows77.674857800sbody/85.352001100sparent/RSS106332160B,
Linux87.064150858sbody/113.707404400sparent/RSS97599488B. Including legacy:
Windows164empty owners11516snapshots66scriptfit738rows221scriptpredict861rows;
Linux122owners11336snapshots52scriptfit612rows213scriptpredict789rows. Each has
71literalreadoutgames140steps and14charged collection episode slots. The
`teacher_steps=108` counter counts successful collection completions only;
one injected lost collection-step return adds one attempted teacher/step call
per platform by fixture source accounting, within126prospective cap. Original
failed prefix and repairs remain. No genuine numerical/physics/model RNG work.
Snapshot-record upper49,518,800/48,744,800 slots excludes copies/equality traffic.
Statics37.321418000s, touchedcache117283316B (mtime-selected, not growth; inherited
global separate), stage3681026B before metadata under24MiB/log256KiB/meta4MiB.
Final-metadata.json records versions/source hashes/commands/cost scopes.

Resource scope: RSS after imports, not process-tree/import peak. Snapshot proxy
is record slots, not all copies/equality/allocations. All original cumulative
software time/work/RSS and prospectively revised literal collection/affected
qualification caps pass. Exact command args/environment/stdout/XML/work/parent/
statics receipts in `artifacts/runs/live-a4-serving-20261025/`.

```console
.venv/Scripts/python.exe -B artifacts/runs/live-a4-serving-20261025/checks.py windows windows-1
.venv/Scripts/python.exe -B artifacts/runs/live-a4-serving-20261025/checks.py windows windows-repair -k "not describe and not unfrozen and not missing and not actual_command"
.venv/Scripts/python.exe -B artifacts/runs/live-a4-serving-20261025/checks.py windows windows-boundaries -k "refuse_mutation and (replace or gate or source)"
.venv/Scripts/python.exe -B artifacts/runs/live-a4-serving-20261025/checks.py linux linux-final
.venv/Scripts/python.exe -B artifacts/runs/live-a4-serving-20261025/checks.py windows windows-prefix -k "finish_actual_cli or prior_receipt_mutation_inside"
.venv/Scripts/python.exe -B artifacts/runs/live-a4-serving-20261025/checks.py linux linux-prefix -k "finish_actual_cli or prior_receipt_mutation_inside"
.venv/Scripts/python.exe -B artifacts/runs/live-a4-serving-20261025/checks.py statics statics-ready
.venv/Scripts/python.exe -B artifacts/runs/live-a4-serving-20261025/checks.py native native
.venv/Scripts/python.exe -B artifacts/runs/live-a4-serving-20261025/checks.py windows windows-controls -k "list_append or logical_clock or candidate_version"
.venv/Scripts/python.exe -B artifacts/runs/live-a4-serving-20261025/checks.py linux linux-controls -k "list_append or logical_clock or candidate_version"
.venv/Scripts/python.exe -B artifacts/runs/live-a4-serving-20261025/checks.py windows windows-combined -k finish_actual_cli
.venv/Scripts/python.exe -B artifacts/runs/live-a4-serving-20261025/checks.py linux linux-combined -k finish_actual_cli
.venv/Scripts/python.exe -B artifacts/runs/live-a4-serving-20261025/checks.py statics statics-combined
```

Fullsuite/cleanclone/currentinstalledwheel/other math-baseline-simulator cohorts/
external independent review/statistical multiseed/heldout/release checks are
outside this increment and skipped. Reused173/181 and historical negative,
failed and inconclusive results are unchanged. Historical plans/log remain
frozen per AGENTS; GitHub is the live handoff.

## Native integration result

The sole frozen `NATIVE_PROTOCOL.md` allowance is **FAILED-SPENT**, no retry.
Producer completed all work/export, then the outer helper failed at its first
JSON/native-object equality assertion. `asdict(SharingSnapshot).deferrals` is a
tuple, including the empty case; JSON decodes it as a list. Direct decoded JSON
equality against the native report is invalid even for unchanged correct output.
The original helper/protocol/failed summary remain unchanged. Later outer checks,
including both global RNG witnesses, were never executed and cannot be backfilled.
This is not an accepted native pass or independently accepted result.

After a prospective issue amendment, ONE readonly current-output diagnosis read
3575316B once in0.0245147s, with0models/physics/source/historicalreport reads.
It verified serialized complete producer stages/roles/closed games/receipts/
controls/counts and exported only scalar metadata. RSS was not sampled for this
small diagnostic; do not claim full readout resource qualification. It does not
recover the original Python object/stdout comparison or RNG state.

| Private updates | Learned old A3 | Fresh A4 |
| --- | ---: | ---: |
|2 |2/2 |0/2 |
|4 |2/2 |2/2 |
|6 |2/2 |2/2 |

Four published A3 games (two at4, two at6) all succeed, actorc0gen1/model2/18.
Successful games use9steps/.49375return; initial A4 failures use16steps/0return.
The saved producer reports acquisition gain1, retention delta0, return gain0,
recoveryfalse/false. This supports a limited known-TRAIN producer observation
of acquisition and retention; no lost-and-regained recovery occurred. Global
RNG preservation remains unverified. No multiseed/generalization/matchedcompute/
efficiency/CPC advantage/release conclusion follows.

Observed native attempts/returns:6fits90rows/6LAPACK,162predicts194rows INCLUDING
fourguardpredictions36rows and36publicactorpredictions36rows;18env/resets/closes,
176physicssteps INCLUDING18teachersteps. Nine empty owners,8667model snapshots/
25225227snapshotrecordslots5296gate snapshots; all work caps pass. Outer body
53.8179921s/parent54.5690117s/RSS80166912B; producer body53.2368921s/RSS80162816B.
Same native6charges/readout0/0 controls and starts preserved, sharing6admissions/
0active. Catalogue213840input-array bytes/18peak bank rows/85392float64-equivalent
bank bytes; this is not all process allocation. Counts/time/RSS include diagnostics,
readouts, state validation and serialization; source-generation sunk cost not
remeasured. No fresh optimizer/RNG initialization, SGD, final/probe or old-source
cohort repeated.

Final three guard additions postdate this native producer. Their directed
software checks do not retroactively qualify the native execution against new
source. Original positive-path observations remain evidence of executed behavior
with the stated source scope. Issue182 stays open for unfulfilled native/RNG
acceptance; failure cannot be repaired by relabeling or renewing this run.

## Exact next implementation action

Reconcile existing candidate/checkpoint retirement and original cumulative budget
handoff facilities to support a fresh candidate based on current publishedactorc0
after private A4 acquisition. Current basea0 correctly prevents deploying that new
skill. Implement the smallest genuine current-actor candidate continuation through
existing facilities, with original charges/consent/retired history preserved, then
literal second-publication checks before a separately scoped native deployment
protocol. Do not repeat this private-only cohort or start another audit chain.
Any future native helper must compare JSON-normalized records and record final
RNG evidence before fallible serialization checks; this does not erase182's gap.
