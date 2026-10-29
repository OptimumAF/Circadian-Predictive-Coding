# R4.2-A4.EC — repeated-current exposure control

[Issue186](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/186).
Previous184 was verified progress; full goal and broader R2/R4/P6/science/release
parents remain open. Single-agent implementation/testing/source-procedure-result
SELF-REVIEW, not independent acceptance. This is software behavior on literal
ports; no native learning or scientific comparison was executed.

## Delivered API behavior

`run_context_linear_stream(episodes, a4_current_repeated=True)` uses the existing
ordinary stream, original six A3blue/red→A4blue/red→A3blue/red arrivals, delay1,
six-update90s512MiB owner and0/2/4/6 checkpoints. Its pure descriptor is
`describe_context_linear_a4_current_repeated()`. Exact-bool and mode-conflict
gates refuse current-only/joint/ABC/recurring/reference mixing before owner work.
Source identity/context/role/nine-row/public-cue/causal validation stays intact.
All endpoint TRAIN scoring follows all six fits; complete native endpoint handoff
follows final successful receipt capture. No new CLI flag, evaluator or game path.

The existing catalogue module composes current-only multiplicities
`(1,1,2,2,2,2)`: batches9/9/18/18/18/18, six fits90rows. Counts are54arrival
exposures+36repeated-current, prior-packet replay0/bank0, original physical unique
source rows36. Unique copy-specific transport IDs and complete per-copy packet/
context spans make every duplicate explicit. Raw arrays repeat, not episode
records; no fictitious trajectory is created. Arrays are detached between wakes,
copies and caller sources. New `current_repetition` report distinguishes plan,
committed rows, latest verified receipts/rows, capture uncertainty and failed
attempt repetition upper bound. Old current-only54/joint90 reports stay unchanged.

Retained input-array bytes213840 count only owned output float64 arrays; six
temporary source templates total128304float64 bytes across construction, not a
retained-memory peak. Validation/projection/stream copies, Python overhead and
all allocations/FLOPs are not inferred from that field; sampled process RSS is
reported separately. Cumulative owned input arrays are bounded before allocation.
No implicit replay/consolidation budget event is invented for in-wake repetition.

Why this: existing171/172joint90 versuscurrent-only54 outcomes confound source
reuse with additional row exposure. The new control fixes fit count/batch geometry
without selecting seeds/doses/metrics from outcomes. It does not establish equal
wall time/FLOPs, replay benefit, CPC advantage, generalization or task usefulness.
All original unequal-work results and negatives remain unchanged.

## Failure and predecessor handling

Initial Windows run22PASS2FAIL. The row-limit test accidentally built a truncated
episode with a nonterminal last transition during arrange; corrected to a valid
eight-step packet. The other failure exposed real behavior: standalone stream
resource checks did not witness application-level preceding receipt values.
Corrupting an earlier receipt after update2 could reach later fits. That first
failed invocation's additional literal work remains fully counted.

Repair only in new mode: incremental existing `run_until`, verify preceding
receipt/candidate/source before next work, then verify old receipt prefix and
new receipt sample/times/update/native-row counts before admitting success.
Corrupt/replaced prior evidence stops before another fit. Unknown/corrupt capture
is null, retains the error and original budget charges, and exports the bounded
latest verified prefix/costs separately. No observation of a charged update is
fabricated. Failed fit retains its attempted18rows and uncertainty with two known
receipts; no fabricated complete endpoint or budget renewal. At most6verified
receipts/onecurrent state, no unlimited success-record chain.

## Checks and observed costs

| Check | Outcome | Body seconds | Parent seconds | RSS bytes |
| --- | --- | ---: | ---: | ---: |
| Initial Windows |22PASS/2FAIL |2.097913000 |3.246297600 |102375424 |
| Final Windows affected |7PASS/19deselected |1.412660900 |2.488187500 |93732864 |
| Full final Linux |26PASS |10.048969807 |18.730907800 |90669056 |

Windows26distinct qualified using19unchanged initial passes and7affected final
passes; both failed cases repaired, two new source/candidate tamper cases added.
No skipped tests. Real stream/native-port ownership, exact current rows/labels,
delay-one receipts,90row counters/scoring/handoff, source+actor and Python/NumPy
RNG preservation pass. Pure composer proofs cover copy IDs/spans/array detachment,
row/role/cue/history/identity/cap rejection. Old54current/joint90 positive API
regressions pass with literal ports. Scalar literal predictions remain uniform;
no task outcomes were tuned to reproduce prior native learning.

Cumulative Windows costs including failed work:45empty metadata owners427snapshot
calls42literalfit540rows80literalpredict720rows32validationpasses2260raw projected
rows. Linux31owners275snapshots27literalfit324rows48predict432rows26validationpasses
1702rows. Forbidden genuine solver/forward/physics/modelRNG/env/teacher/final/
probe/historical scientific reader calls0. All original90sbody120sparent/512MiB/
work caps pass. Source-validation pass versus raw-row counter units were clarified
prospectively before fixtures; no limit increased. Sampler starts after imports;
not exhaustive import/process-tree/allocation peak or FLOPs.

Initial Ruff/format passed; mypy found6errors from a missing span context and two
annotation issues. Fixed before fixtures. One multi-file patch failed an exact
formatter-changed anchor atomically; corrected with a small exact patch. All
failure logs/tool receipts retained. Final four-file Ruff/check/format--check and
default+Linuxmypy `--explicit-package-bases --follow-imports=silent` PASS. No code
change after final behavioral/static checks.

Commands at root (exact expanded parents/statics in finite stage):

```console
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/a4-current-repeated-20261026/qualify.py windows-1
.venv/Scripts/python.exe -X utf8 -B artifacts/runs/a4-current-repeated-20261026/qualify.py windows-repair -k "run_six_delayed or report_repeated_current or keep_two_receipts or corrupted_prior_receipt or changed_original or changed_candidate or (bad_composer_sources_before_allocating and rows)"
```

Linux uses WSL Ubuntu, existing cached
`/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B`
and same qualifier `linux-final`, timeout110s/parent120s/oneCPU. Explicit exact
collection, raw stdout/JUnit/work/parent/static/source/storage records and
PROTOCOL/REVIEW are in `artifacts/runs/a4-current-repeated-20261026/`.

No native mathematical/game/comparative cohort, fullsuite, real Torch, new CLI/
evaluator/game integration, cleanclone/install/currentwheel/browser/external
independent/scientific/release checks. Those claims remain unqualified. No new
deps/config/framework/download/GPU/CI/Gitmutation/delete/reset/agents.184original
time failure,182RNGgap,177/89spent comparisons and all neural/controller/CPC
negative results preserved. Both historical plans/log remain frozen perAGENTS;
issues are the live handoff. Inherited dirty functionality preserved.

## One next implementation action

Extend the EXISTING A4 adaptation composition/CLI to select this explicit
repeated-current stream control and qualify its six-fit90row endpoints through
the existing A3/A4 evaluator/summary. Keep old joint/current-only defaults and
source/readout roles, fixed game order/seed17/max16, all receipt/prefix/cost guards.
Begin literal composition/dispatch/admission/failure checks without solver or
physics. Then freeze ONE small current54/repeated90/joint90 native comparison
before any scientific execution. This is the missing usable comparative path,
not a new runner/checkpoint/provenance task or permission to renew old studies.
