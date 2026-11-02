# Bounded continued memory learning — 2026-11-01

[#416 / R2.3-MEM.CAP](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/416)
implements an explicit retained-memory option in the existing delayed TRAIN
stream. It accepts arrivals beyond prototype capacity while keeping prototypes
and IDs bounded and cumulative exposure paid. Default append-only learner,
nearest-distance math, replay-retention rules and native/probe admissions remain
unchanged. This is scoped software correctness, not new native usefulness or
scientific/CPC/release acceptance. The full repository goal remains ACTIVE.

## Changed boundaries

```text
src/core/nearest_retention.py               pure bounded row-ID selection
src/adapters/numpy_retained_nearest_bc.py    distinct owned state/learner and costs
src/adapters/continuous_stream_sequence.py  explicit option in existing runtime
tests/test_retained_nearest_memory.py       new literal behavioral cohort
docs/retained-nearest-memory.md             API, accounting and limitations
```

Why this: working source220/private238 memory approaches cap256; another50-row
arrival cannot fit. Repeating corrective neural fits would ignore #415's
negative evidence. Existing memory acquisition/retention/public workflow and
robot publication binding are reused, not rebuilt. No research raw files,
original checkpoint, teacher, native scenes, probes/final, neural optimizer,
model initialization RNG, data conversion or comparison baseline were used.

The new learner subclasses the interface solely to reuse prediction/episode
validation; its exact distinct snapshot prevents original admissions from
silently accepting cumulative counts as prototype rows. Fixed-input TRAIN only,
existing FIFO or seeded-reservoir rule, no future labels or role conversion.
Every row remains paid after discard; no unbounded seen-ID or success history.
Counters are native bounded63-bit integers. New stored IDs bind exact canonical
feature/target content and the snapshot includes all counters/rule/arrays.

## Executed behavior

Final27 distinct literal checks pass. A capacity2 sequence A→B→Areturn uses
four row exposures/three appends and stores two rows/two IDs. Warm A query gives
-.5; after two B rows A gives+.5 (forgotten) and B gives+.75; after returning A
the same queries give-.5 and+.75. Source/actor stay at the original one-row A
state. Labels arrive at ticks2/5 and use exactly two existing runtime charges.
No native controls or task-success/reliability claims follow from these literals.

The new mode also admits257 arrivals in the original two-packet stream while
retaining256; the default refuses the same overflow before fitting. Tests cover
duplicate paid exposure, conflicting targets/earliest ties, deterministic seeded
content ranks, full-state alias/fork/restore, bad IDs/counters/63-bit overflow,
real probe_train/probe_test/final_test label refusals, invalid options, delayed
labels, prior receipt corruption, callback failure and lost update return.

Final tiny state costs:48 numeric bytes,128 encoded hex-ID bytes,64 digest payload
bytes,192 selected incoming hash bytes including the24-byte domain,48 discarded
pair bytes and9 offered selection candidates. Validation/copies/Python/RSS are
additional. These are engineering accounting definitions, not new accuracy metrics.

Lost return evidence is explicit: two attempted incoming rows, observed native
candidate2 updates/3 exposures,0 returned appends and0 successful runtime budget
updates. The inbox stops and no next work is admitted. This is not called a
successful charge; source remains1/1 and native/attempt costs are preserved.
Corrupted prior receipt similarly refuses a second append while retaining the
first paid report and budget charge. No source budget or clock is renewed.

## Exact commands and finite qualification

Physical cwd `D:\Dev\Repositories\Circadian`; process TEMP/TMP/TMPDIR D:\Dev\_Temp,
no bytecode, numerical threads1. The prospectively frozen
[software contract](../../artifacts/runs/retained-memory-20261101/SOFTWARE-PROTOCOL.md)
permits three NEW literal invocations,30s each/90s aggregate, observed512MiB peak,
<=32 distinct cases,160 initializers/400 updates/2048 incoming rows/180 predictions/
1024 queries/32768 distance candidates cumulatively. No old fixture quota renewed.

```powershell
.\artifacts\runs\retained-memory-20261101\run-checks.ps1 -Slot first
.\artifacts\runs\retained-memory-20261101\run-checks.ps1 -Slot second
.\artifacts\runs\retained-memory-20261101\run-checks.ps1 -Slot third
```

Each invokes the ordinary repository command:

```powershell
.venv\Scripts\python.exe -B -m pytest -q --tb=short tests/test_retained_nearest_memory.py --basetemp D:/Dev/_Temp/circadian416-SLOT
```

| Slot | Outcome | Host seconds | Actual pytest lifetime peak |
| --- | --- | --- | --- |
| first |20 PASS/3 FAIL/2 ERROR; exit1 |1.1736629 | MISSING |
| second |27 PASS; exit0 |1.0581009 |65,556,480B |
| third, final combined source |27 PASS; exit0 |1.0450208 |65,376,256B |

The initial failures were a literal fixture with nonjoining next observations;
original episode validation correctly refused. The final role fixture was also
corrected from invalid `final` to actual `final_test`, so it tests the learner
boundary. Two extra counter/tamper cases remain within32. All failures persist.
First launcher sample5,472,256B is NOT pytest RSS. Do not claim all three memory
gates proven. Before second, real Windows process peak capture and cumulative
prior-slot work checks were added; first missing evidence is not reconstructed.
Host timeout kill covers its process tree; peak is observed, not OS-enforced.

All79 test executions:74 PASS/3 FAIL/2 ERROR;27 distinct final cases. Cumulative
34 nearest initializers,50 forks (no initializer/RNG),49 update attempts/588
incoming rows,21 predictions/34 queries/60 distance candidates. Total measured
host3.2767846s. All numerical quotas held, all three slots SPENT, no carry-forward.
No neural forward/gradient/solve/replay, model-random draws, teacher/SDK/physics.

Before final slot, review corrected32-byte raw digest accounting into explicit
64-byte stored hex plus32-byte digest payload, included the selected hash domain,
added attempted-row/unreturned flags and native policy-name type validation.
Earlier receipts preserve their earlier cost-field definitions. No prediction,
seed, comparison direction or scientific metric changed.

Final statics:

```powershell
.venv\Scripts\python.exe -m ruff check src/core/nearest_retention.py src/adapters/numpy_retained_nearest_bc.py src/adapters/continuous_stream_sequence.py tests/test_retained_nearest_memory.py
.venv\Scripts\python.exe -m ruff format --check src/core/nearest_retention.py src/adapters/numpy_retained_nearest_bc.py src/adapters/continuous_stream_sequence.py tests/test_retained_nearest_memory.py
.venv\Scripts\python.exe -m mypy --follow-imports=silent --cache-dir D:\Dev\_Temp\circadian-mypy416 src/core/nearest_retention.py src/adapters/numpy_retained_nearest_bc.py src/adapters/continuous_stream_sequence.py tests/test_retained_nearest_memory.py
```

Ruff/format4 and mypy4 PASS, host PowerShell parser PASS. Three initial mypy
diagnostics (optional-owner narrowing/empty dict annotation) were repaired
before final verification. Two patch anchors failed without partial writes;
small exact patches followed. No source or metric weakened to make tests pass.

## Review and remaining claims

Lead self-reviewed combined actual source, procedure and logs/receipts. One
finite independent reviewer used three read-only batches~.494s, no imports,
tests/models/data/SDK/RNG/replication/writes/delegation. ACCEPT scoped software
and directly observed compact new costs/failure/host receipts. Concrete findings
about cumulative slot limits, prior receipt tampering and hex-ID storage were
fixed and included in final checks. Initial diff output truncated one middle
section; batch2 inspected that specific missing code. Final27 PASS is lead-log
verified, not independently replicated; peer read final host exit0 and counters.
First actual RSS gap remains explicit. All review batches SPENT/exception ENDED.

No original nearest/native/probe mathematics or exact admissions changed. Old
accepted evidence is reused only for those unchanged implementations; it does
not qualify this new native type. Linux tests, whole suite, clean clone/current
wheel/install, native/scientific comparison, GPU/remoteCI/release checks skipped.
No installs/downloads/commits/push/reset/deletion/global settings/unrelated dirty
changes; inherited masterd9119e8 remains. Earlier #399/#408/#411 partial evidence
and #413/#415 completed negatives are preserved. Broader R2.3/retention/recovery/
science/release parents remain OPEN; finite state correctness is not competence.

All protocols/markers/three logs, counters and host receipts are in
`artifacts/runs/retained-memory-20261101/`. Requested plan/roadmap/log entries
are archival evidence; GitHub issues remain canonical.

**ONE next implementation action:** inspect the exact old nearest admissions
in the existing frozen native rollout, then qualify this exact retained type's
full-state/paid-control guards as the prerequisite to one separately frozen small
known-TRAIN storage-pressure study. Use the existing runner and unchanged native
seeds/metrics; do not launch another optimizer/probe/capture framework, republish
private candidates, retry old failed experiments or tune final evaluation. No
actual native/data allowance follows from this software closeout.

Recorder portability closeout: after all numerical slots closed, self-review found unconditional Windows-only memory APIs would break Linux tests. Saved before-portability.py; extracted identical Windows query into helper with standard POSIX getrusage fallback. ONE prospectively bounded metadata-only command .venv\Scripts\python.exe -I -B artifacts/runs/retained-memory-20261101/verify-portability.py PASS/.016167s/21,979,136B:21behavior definitions unchanged, Windows query AST equal,0projectimports/models/RNG/data/native. Metadata markerSPENT; no fourth numerical run. POSIX runtime unexecuted. Initial platform typing produced2Windows resource/2Linux windll diagnostics; switched the guard from os.name to type-checker-recognized sys.platform, Windows query/body and behavioral tests unchanged. Final Windows+Linux mypy4 PASS, source-qualified reuse of last27behavioral checks. Metadata helper also receives sys in namespace; no rerun of occupied marker. This final recorder portability delta is SELF-REVIEW ONLY, after finite peer ENDED; do not describe it as independently accepted or a Linux runtime observation. Production source is unchanged after final27PASS. Planned next owning issue417; no actual data/native quota follows from416closeout.
Canonical scoped software closeout: https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/416#issuecomment-6565157444 ; #416CLOSED/#18handoffupdated/#417PLANNED. FinalRuff-format5/Windows+Linuxmypy4/doclinks/scopedgitdiffcheckPASS; stage23files95952B/log18352B (below2MiB/256KiB),3numericalSPENT+1metadataSPENT. LF/CRLFwarnings retained. No further numerical/native execution.
