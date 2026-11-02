# Known-TRAIN memory under storage pressure — completed #417 qualification

The corrected eight-endpoint qualification completed and received scoped
independent acceptance. FIFO retained Pick19 success; seeded reservoir retained
Pick17/Pick20 success. Both failed Push17, and neither retained all four tasks.
This is a fixed known-TRAIN trade-off, not a general winner, recovery result,
population reliability or circadian advantage.

The original failed attempt remains unchanged and separately reported in
[its historical report](retained-native-memory-20261101.md). Its missing terminal
gates are still missing. Completion here refers to the prospectively reviewed
engineering requalification on the same issue, with all prior costs included.

## Fixed outcomes

| Known TRAIN reset | Accepted unbounded220 reference | FIFO128 | Reservoir128, seed17 |
| --- | --- | --- | --- |
| Pick17 | success15 controls | failure50 | success14 |
| Pick19 | success22 | success22 | failure50 |
| Pick20 | success15 | failure50 | success13 |
| Push17 | success18 | failure50 | failure50 |

Every new endpoint is internally complete. Success has sparse return1; failure
has return0 and no success at the original50-control horizon. All299 controls
returned,1,495 known control-physics steps,0 possible failed steps,8 returned
scene constructions and closures. Pick frozen inference/original readout/history
flags pass, optimizer0; Push source unchanged and complete native-work returns
pass. This does not measure all reset physics or safety beyond these samples.

The historical unbounded reference is reused under unchanged TRAIN data/inference
assumptions; it was not rerun. Capacity, deduplication, metadata and total work
differ. The two new rules use the same seven released TRAIN packets/220 rows,
order, delays, fixed128 capacity and original task seeds/metrics. There is no
new label/teacher/query, probe/final access, optimizer dose, source promotion or
favorable capacity/seed/run selection. No equal-compute or overall memory-saving
claim follows from row capacity or these one-case outcomes.

## Working implementation and state

This session changes no production learner/native loop. It applies the already
qualified strict canonical JSON capture comparison in a corrected local producer
and adds the explicit cumulative engineering ledger. The existing stream/native
components from #416/#417 are used unchanged. Source-only AST checks prove the
whole TRAIN/native procedure unchanged except the added cumulative elapsed
receipt; capture helper and existing hooks match the qualified proposal.

Both streams finish7 appends/220 exposures, with6 runtime budget charges/205
rows after the warm15. Each candidate retains128 rows/90 discarded/2 duplicates.
Source and stable actor stay at1 update/15 rows; terminal native/data/owner/paid
prefix and input proofs pass. No private learner is published or exported.

Each final owner has16,384 numeric bytes and8,192 encoded hex-ID bytes,24,576
selected payload bytes together. The4,096 digest-payload figure describes the
same identities in raw digest form, not another stored bank. Validation copies,
Python strings/tuples/containers, transient allocation and SDK/RSS costs are
additional. Source/actor numeric payload is1,920B each. The measured process
peak is not a learner-only storage comparison.

Reservoir reports11,610 selected eviction-rank evaluations; FIFO0. The existing
stream separately repeats selection for validation, so another11,610 reservoir
rank evaluations are source-derived overhead, not counted by the selected
feature/target pair-hash meter. Model/SDK/validation costs are not interchangeable
with a claim of matched total computation.

## Why the engineering attempt was amended

Attempt1 completed FIFO Pick17 failure50, then rejected its149,479B capture by
comparing Python tuple fields directly with JSON lists. Its process died without
a learner export or whole-case final gates. Four pure-JSON checks qualified a
canonical comparison that still refuses altered exposure counts, bool/int and
list order. No original frozen file/marker/result was rewritten.

Seven-only continuation would leave the required whole-case terminal proofs
incomplete. Before new inputs/models/SDK, the same-issue engineering attempt
limit1→2 was explicitly amended for one full corrected qualification. It paid
for both new preparations and the repeated first scene; the old negative remains
visible and the corrected first FIFO Pick17 also fails50. No third attempt or
automatic retry allowance exists. This is requalification after an engineering
failure, not an independent statistical replication or a renamed science budget.

The new ledger uses min(original per-case cap, cumulative ceiling minus prior
paid work). Increased aggregate preparation/interaction ceilings were explicit:
3 initializers/21 appends660 exposures/9 scenes450 controls2,250 known physics.
Original cumulative selected-copy/hash/diagnostic/IO/RSS/parent-storage ceilings
were retained. New170s work/190s GNU/200s host fit original cumulative180/200/215
seconds after the old6.302/6.65/9.429 seconds. Complete negatives continue;
errors/corruption/caps stop. All choices and criteria were frozen and reviewed.

## Exact execution and observed costs

Physical cwd D:\Dev\Repositories\Circadian, process TEMP/TMP/TMPDIR D:\Dev\_Temp,
cached WSL CPU/Mesa runtime from the frozen protocol, numerical threads1,
no bytecode/install/download/network/global configuration change.

```powershell
.venv\Scripts\python.exe -I -B artifacts/runs/retained-native-20261101/prepare-requalification.py
.venv\Scripts\python.exe -I -B artifacts/runs/retained-native-20261101/requalification/qualify.py
.venv\Scripts\python.exe -m ruff check artifacts/runs/retained-native-20261101/requalification/study.py artifacts/runs/retained-native-20261101/requalification/qualify.py
.venv\Scripts\python.exe -m ruff format --check artifacts/runs/retained-native-20261101/requalification/study.py artifacts/runs/retained-native-20261101/requalification/qualify.py
.venv\Scripts\python.exe -m mypy --platform linux --follow-imports=silent --cache-dir D:\Dev\_Temp\circadian-mypy417rq artifacts/runs/retained-native-20261101/requalification/study.py
.\artifacts\runs\retained-native-20261101\requalification\run-case.ps1
```

The source/ledger/JSON metadata gate passes10 checks in .019678400s/23,183,360B,
zero project imports/old actual reads/data/models/native. Scoped Ruff/format2,
Linux producer mypy1 and host parser pass. The initial PowerShell source-pin
array concatenation error produced an incomplete metadata map; corrected before
any input/model/native work with literal path arrays and terminating errors.
Original24 source files match and stay frozen; new30 source/ledger/preservation
pins match before, after and independent POST. Old learning/math/boundary tests
were reused under unchanged sources, not renewed.

Corrected command ONCE: unified session62930, hostPID14660, exit0/17.9632929s,
cumulative host27.3924721s, no host stop. Producer14.657229449s/cumulative
20.958988204s, peak536,473,600B. GNU15.08s/526,500KiB, cumulative GNU21.73s.
All are within the reviewed ceilings. Peak is observed, not OS-enforced.

| Paid work | Corrected attempt | Cumulative including failed attempt |
| --- | --- | --- |
| Initializers; appends; row exposures |2;14;440 |3;21;660 |
| Forks |36 |52 |
| Predictions/query rows |299 |349 |
| Distance candidates; feature visits |38,272;459,264 |44,672;536,064 |
| Returned scenes/closures |8 |9 |
| Returned controls; known control physics |299;1,495 |349;1,745 |
| Public snapshots; validators |4,565;13,905 |5,558;16,837 |
| Selected numeric validation-copy bytes |221,590,016 |266,628,864 |
| Selected pair-hash calls; rows; bytes |13,931;1,732,022;263,267,344 |16,876;2,084,313;316,815,576 |
| Qualified public input opens; bytes |2;332,878 |3;499,317 |
| Selected SDK default_rng calls |1 |2 |
| Trace-read attempts; bytes |201;23,590,367 |202;23,739,846 |
| Prefix checks; selected UTF8 bytes |99;20,198,566 |139;20,298,113 |
| Source-pin payload bytes |1,334,650 |1,966,203 |

Complete/budget_ok true, error/permanent stop null, both streams complete,
30 source pins before/after, final original input digest/fileidentity and
trusted previous-result/capture proofs pass. No missing native returns or
possible failed physics. Original failed run's incomplete gates are not changed.

New raw1,140,325B SHA256
`2cb85a4286e18cbffbbb7fe1f042142b1627c5f2060c53dc65379b95ace4e465`.
Lead/peer one new raw read each SPENT; use requalification/LEAD-SCALARS.json.
Original raw208,216B SHA
`7e83d06cf0b198aada56341effe2d803622333dbd8b74bfaea87ae44fee73af3`
and its spent analysis allowances remain unchanged. Protocol, budget, pins,
metadata/PRE/POST, actual, six native captures, host/time and markers remain
under artifacts/runs/retained-native-20261101/requalification/ with the original
failed parent beside them. No project output was placed on C:.

## Review, limitations and next substantial outcome

Lead self-reviewed source/procedure/results and cumulative costs. One finite
independent exception used2 readonly batches~.417485s, no project imports,
tests/models/data/SDK/old raw/replication/writes. Source/PRE and completed-case
POST accepted; peer directly read one new raw, compact host/time/current30pins,
all eight outcome/state/history/closure fields and independently checked every
cumulative work key against prior+new and caps. Unused source-remedy allowance
closed; finite exception ENDED, no standing pool. Acceptance is scoped to this
qualification, not native reliability, recovery, statistical benefit or release.

No production changes, new learner test cohort, baseline/native comparison
sweep, GPU, clean clone/current wheel/install, whole suite or remote CI. Existing
working functionality/unrelated dirty source and all old negative/partial cases
remain. The SDK GLVND warning is retained. No commit/push/reset/delete. #417 can
close its scoped investigation with this evidence; broader R2.3/R4/R5/recovery/
scientific/CPC/release requirements and full goal remain OPEN/ACTIVE.

Critical path reassessed: do not tune capacity129 or reservoir seeds, or add
another capture/optimizer/audit increment. Existing four-family diagnostics,
memory acquisition/retention and default public demo already work in scope.
Inspection confirms the first-party memory CLI/API has no retention parameters:
run_demo/run_native_demo and its readout pins still accept only the old type.
The bounded learner is not accessible through that ordinary reproducible command.

Next #418 / R4.8-MEM.RETAIN exposes explicit retained CPU/native mode through
the existing API/CLI. Default exact70-command reproduction stays strict; explicit
retained CPU mode reports command fidelity/costs without treating saved returns
as fresh native success, and probe/final/publication rights remain unchanged.
**ONE next implementation action:** inspect the final observer/readout capture,
then add explicit retained policy/cap parameters and matching CLI options with
early incompatibility checks, reusing the existing stream. Freeze finite new
checks before model work; no new data/native experiment authority from this handoff.

Canonical completed scoped handoff: https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/417#issuecomment-6565821909 ; #417CLOSED allscopedcriteria/eightcurrentterminalproofs accepted; originalfailedattempt/missing oldgates preserved. #18updated/#418PLANNED/broaderparentsfullgoalOPEN-ACTIVE. Finaldoclinks/whitespacePASS; parentold+new74files2588640B/log4468B below8MiB/512KiB, old+newSPENTmarkersretained. No further raw/model/native run; LF/CRLFwarnings retained.
