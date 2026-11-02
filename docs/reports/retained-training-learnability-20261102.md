# #427 / R2.1-RTN — retained TRAIN-only learnability

The existing continuous probe now accepts an explicit bounded-memory TRAIN-only
diagnostic. It performs one delayed private append, scores the supplied training
packet and up to four old TRAIN readouts, and reports content coverage without an
evaluation packet. Source and stable actor remain unchanged; no adapted learner
is returned or promoted. This is implemented and passing on literal software
cases, not yet an original-input or simulator demonstration of this new API.

Why this: #426's recent-priority control ended3/5, equalFIFO and belowreservoir4/5.
Restoring recorded training fit did not add native success. Before changing more
retention ratios, diagnose actual bounded-bank fit/conflicts and coverage without
readonly24 or final access. Existing append-only training-only diagnostics reject
retained sources; paired retained probes already exist but require evaluation.
Those earlier controls, failures and conclusions remain unchanged.

## Implementation and boundaries

`src/adapters/continuous_probe.py` adds `run_retained_nearest_training_probe` using
the existing `_NearestTrainingPlan`, `_run_single_probe`, delayed runtime, snapshot/
source/actor/owner/budget/clock/receipt guards, score sealing and permanent charges.
The new entry requires an exact nonempty `NumpyRetainedNearestBCLearner`, fixed-input
TRAIN policy, same-spec1..32TS12 probe_train rows and <=4same-spec distinct TRAIN
readouts. Other training roles, duplicate readout IDs/spec mismatch/wrong source
refuse before private work. The old append-only entry still rejects retained;
paired/return/native and BP/PC/CPC defaults keep their original routes/schemas.
Caller input qualification and release authority remain required.

`continuous_nearest_probe.summarize` handles zero evaluation bytes. New TRAIN-only
metrics are training_fit_gain (before-after, higher means improved fit) and
old_retention_loss (after-before, positive means interference). Evaluation_id is
None; probe_test score rows remain0. No fresh test metric is fabricated.

`continuous_retained_training.py` is a small data-only helper. It reuses original
content IDs and actual `NearestPrediction.selected_indices`, with no additional
prediction, new learner, role conversion, loader or orchestration. Positional
source-exposure offsets cannot describe a compacted bank. Every selected row now
records current index/content ID, matches_source, matches_incoming, query-content
presence, exact selected-content equality and equal feature values.

Origins are content membership in the source bank at this probe's admission and
the supplied incoming packet, not historical packet ownership or arrival time.
Both flags can be true, including before append for a retained duplicate. A
previously discarded old training row can be a query miss despite its history.
No unbounded provenance/seen ledger is kept. Coverage distinguishes occurrences
from distinct IDs, content present in bank from the actual selected row, and
feature ties with conflicting targets. A query can be stored yet not selected
because the original nearest kernel keeps earliest-row ties; no tie rule changes.

Hash attempts/returns/rows/payload and numeric materialization upper bounds remain
explicit. The latter counts2*(feature+target bytes) for existing casts+tobytes,
excluding Python concatenations/containers and other snapshot/validation copies.
Mapping attempts/returns/rows, input-ID set construction occurrences, semantic
membership-row and feature-comparison upper bounds are separate; they are not
Python hash-table probes or total computation. Failed hashes/mappings retain
attempted work and do not invent returns. Existing consistency kernel work,
retrieval/residual work, logical exposure/current bank payload and byte summaries
remain. Fixture receipts additionally meter selected validation/copy/hash work;
production summaries do not establish total RSS or memory efficiency.

## Executed evidence

Physical D:/Dev/Repositories/Circadian, dirtyHEADd9119e80926a139a2f38460564d7eab1b6d2a2b0
preserved. Before copies/prospective software protocol precede edits/models in
`artifacts/runs/retained-training-learnability-20261102/`. Only the two existing
adapters, new helper and new tests changed for code. No dependency/default/runner
or original source/actor authority was added. Source-pins.json binds final4files.

```powershell
& artifacts/runs/retained-training-learnability-20261102/run-checks.ps1 -Slot first
& artifacts/runs/retained-training-learnability-20261102/run-checks.ps1 -Slot second
& artifacts/runs/retained-training-learnability-20261102/run-checks.ps1 -Slot third
```

Each runs `.venv/Scripts/python.exe -B -m pytest -q --tb=short tests/test_retained_training_learnability.py --basetemp D:/Dev/_Temp/circadian427-SLOT`, D:temporarypaths,
threads1/no bytecode. First30PASS2.6009826s, second32PASS2.4710828s,
finalthird32PASS2.7379459s;94case executions/32distinct/0test failures or skips.
Fourth slot unused. Second added actual forgetting and oldappend-only compatibility;
third verifies explicit materialization/ID-visit counters including failed hash.

Passing behavior includes actual private fit, old-task interference+.5, source/
published/native owner isolation, FIFO/reservoir/recent-priority policy routes,
eviction/compaction, duplicate both-memberships, equal-input conflicting-target
coverage, invalid source/role/spec/readout admissions, fourTRAINreadouts with
no evaluation score view, source/endpoint/equalowner corruption refusals, deleted/
valuechanged/equalreplaced prior paid scores restored before refusal, lostappend
permanentattempt/unknownwork, hashfailureattempt-vs-return, oldappend-only actual
selected schema and pairedretained no-new-fields/dose compatibility. Existing
runtime/budget/return/native mathematical evidence is reused, not re-executed.

Final scoped commands on4files (exact full arguments/stdout in statics.json):

```text
.venv/Scripts/python.exe -B -m ruff check FILES
.venv/Scripts/python.exe -B -m ruff format --check FILES
.venv/Scripts/python.exe -B -m mypy --platform win32 --follow-imports=silent FILES
.venv/Scripts/python.exe -B -m mypy --platform linux --follow-imports=silent FILES
```

All exit0; Linux is type analysis, not Linux runtime. Initial unused test imports/
local and dict annotations were corrected before pytest. Two guessed test paths
were missing; switched to exact rg-listed source. No failed model attempt or
acceptance weakening. Self-review expanded materialization/set-visit accounting
before third checks; no algorithm, metric, role or experiment changed.

Cumulative all3slots95retained initializers+2literal plain owners,132append attempts/
224incoming rows/165forks/150predictions/156queryrows/307distance candidates/
3684featurevisits/5148snapshots/37386selectedvalidators/9355392selectedcopyB/
11178384selectedcontent-IDhashB/156residualrows/30fakeendpoints. Consistency104calls/
209rows/158pairs/836native conversions;134mappings140rows,193content-hashrows/
29336payloadB;30physicalrankcalls1920B,165pure selections260arrivals798candidate
upper bound. All frozen aggregate caps satisfied; maxprocesspeak74170368B,
hosttotal7.8100113s plus setup/statics/review. Wrapper's sampled-child peak is
incomplete; actual process peak comes from fixture receipt. No actual datasets,
oldraw/modelrestore/neuralinit/gradient/solve/modelRNG/SDK/physics/replay/final work.

Independent source/state/isolation/content-mapping review accepted final3production
files and final32-case result at its scoped review (no replication/imports/models/
data/SDK/writes). Three read batches .6825538stoolwall. Reporting limit failures
are preserved: first11222tokens/truncated(returnlimit9200), second2787exceeded2000
compacttarget, third1597within. Whitespace-normalized diff and targeted reads
recovered source coverage. Do not claim every review resource target passed.
Finalstatics were lead-run, not independently observed. Exception ENDED;
REVIEW.md records exact acceptance and limits. Final combined source/results/
procedure/costs self-reviewed separately from independent review.

## Remaining requirements and exact next action

#427's software scope is complete. No original23 diagnostic/new native task success,
reliability, generalization, CPC advantage, default selection, matchedcompute,
cleanclone/currentwheel/GPU/remoteCI/broad suite or release claim. Previous426
negative result and failed raw-review history remain unchanged. All fullroadmap/
parents18/21/23/R2.1/R2.3/R4/R5 and scientific/release requirements stayOPEN-ACTIVE.
No install/download/commit/push/reset/delete/unrelatededits.

Next useful outcome: ONE prospectively frozen TRAIN-only diagnosis from the
existing reservoir17/128 source using original23 training only and oldTRAIN
readouts, through existing stream observer/newAPI. It asks whether poor new fit
comes from rejected content or exact-input/target/tie conflicts. Reuse accepted
native/paired scores; do not retrain baselines to repeat them or reopen readonly24/
spent raw captures. Bound preparation/update/consistency/hash/copy/retrieval costs;
no SDK/native/return/ratio/seed search. Inspect actual source/reader and freeze
this necessary diagnostic protocol before any original-input/model work. Scientific
or native study authority is not granted by this software handoff.


Canonical scoped completion: https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/427#issuecomment-6569747616 . #427CLOSED with above review/reporting limits; #428PLANNED one necessary TRAIN-only bounded diagnosis. Fullgoal/broadparents OPEN-ACTIVE.
