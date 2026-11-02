# #368 / R4.2-MG.MIN.N — incremental recurring MiniGrid qualification

This case exercises the existing recurring workflow with explicit
`--train-update minimum_change`. It reuses the accepted #367 adapter and
stream/CLI software evidence, rather than repeating mathematical fixtures or
the cold-fit control. No production source changes are needed for this case.

The fixed recipe starts from untrained heads, admits the original ABC TRAIN
sources, collects the two original G teacher sources, and returns to original A.
Its existing final evaluator plays G blue/red and A blue/red. The TRAIN curves
and final native competence are separate observations: there are no earlier
native games in this path. Native acquisition and recovery after actual native
forgetting cannot be inferred from those curves.

The protocol and prospective allowance are recorded in [issue #368](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/368#issuecomment-6545030253)
and `artifacts/runs/minigrid-minimum-change-native-20261031/PROTOCOL.md`.
All old failed/SPENT results remain unchanged. Complete negative games are valid
observations; incomplete execution or capture remains partial and permanently
spent. This is one functional qualification, with no CPC, matched computation,
statistical reliability, heldout generalization or release claim.

## Engineering evidence before numerical admission

Commands use physical `D:\Dev\Repositories\Circadian` with process-local
`TEMP`/`TMP=D:\Dev\_Temp`; WSL additionally uses the corresponding D: `TMPDIR`.
Existing interpreters are used without installation or global changes:

- Windows `.venv/Scripts/python.exe`, Python 3.14.7, NumPy 2.4.6,
  MiniGrid 3.1.0, Gymnasium 1.2.3, pygame-ce 2.5.7.
- WSL `/home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python`,
  Python 3.12.3, NumPy 1.26.4. MiniGrid/pygame are absent; Linux native execution
  is skipped, with no installation.

`<python> -B <stage>/metadata.py <platform>` was executed once per platform,
without project/numerical imports or model/environment construction. Windows
used 0.081825500s and 23,724,032B peak; Linux 0.026729159s and 18,071,552B.
The native choice preserves the original Windows cold producer's two BLAS
threads. Its earlier four successful final games remain descriptive evidence,
not a newly executed or matched-total-compute comparison.

`<python> -B <stage>/check.py windows-1 <stage>/cache/windows-1` and the
corresponding WSL `linux-1` command each passed 35 pure cases, with no failures,
skips or model imports/work. Windows used 0.715391800s, 49,766,400B peak; Linux
4.889457662s, 38,281,216B. These validate literal protocol records, complete
negative outcomes, saved-file corruption/replacement refusal, exact named
source/driver rosters, and actual observer precharge/lost-return behavior.
They do not construct learners or simulate successful navigation.

Scoped Ruff check/format and default plus Linux-target mypy pass on the five
new producer/checker/analysis/check/test files (`statics-3.json`). Initial four
annotation errors per mypy target remain in `statics-1`; explicit annotations
and value types were corrected without changing validation expectations.
Three static invocations used 13.8926645s, plus initial formatting 0.0442113s,
within the original 90s allowance. Pure checks used 5.604849462s aggregate.

The sole Windows source-only command
`<python> -B <stage>/producer.py --preflight` completed with exit 0,
0.081624100s and 38,060,032B peak. It binds the exact 29 named source/data files
and four protocol/driver files, validates AST/data/installed metadata, and imports
no project/numerical modules. This is named source binding, not transitive Python
containment or authentication.

Exact executed gate/native/analysis commands (historical; exclusive markers now
refuse repeating them):

```powershell
$env:TEMP='D:\Dev\_Temp'; $env:TMP='D:\Dev\_Temp'
& .venv/Scripts/python.exe -B artifacts/runs/minigrid-minimum-change-native-20261031/metadata.py windows
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/minigrid-minimum-change-native-20261031/metadata.py linux
& .venv/Scripts/python.exe -B artifacts/runs/minigrid-minimum-change-native-20261031/check.py windows-1 artifacts/runs/minigrid-minimum-change-native-20261031/cache/windows-1
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/minigrid-minimum-change-native-20261031/check.py linux-1 artifacts/runs/minigrid-minimum-change-native-20261031/cache/linux-1
& .venv/Scripts/python.exe -B artifacts/runs/minigrid-minimum-change-native-20261031/producer.py --preflight
& .venv/Scripts/python.exe -B artifacts/runs/minigrid-minimum-change-native-20261031/launch.py
& .venv/Scripts/python.exe -B artifacts/runs/minigrid-minimum-change-native-20261031/analysis.py
```

Independent PRE GO consumed six readonly calls, 0.9447764s exact outer wall.
The final read verified all 29+4 current hashes and the unspent native boundary.
No reviewer model/test/native work ran. Original engineering/native limits
were not amended. See stage `REVIEW-PRE.md` and six timing receipts.

## Sole actual native case

`.venv/Scripts/python.exe -B <stage>/launch.py` completed once, terminal 0,
host 3.022430900s without expiry; its sole child also exited 0. The child invokes
the current CLI with `--example-affine-recurring --train-update minimum_change
--output <stage>/actual.json`. It used 2.812617600s and 91,000,832B sampled peak
including imports, observations and capture. No process restart or retry occurred.
`native.attempted` and `SPENT.txt` permanently close this case's allowance.

All observed attempts have known returns. Actual work:

| Work | Observed count |
| --- | ---: |
| Learner updates / row exposures | 10 / 252 |
| Cold fits / rows | 2 / 18 |
| Minimum-change fits / rows | 8 / 234 |
| Least-squares solves / rows | 10 / 252 |
| Scheduled training design products / row visits | 26 / 720 |
| Learner predictions / rows | 84 / 468 |
| Fitted core and probability predictions / rows, each | 76 / 396 |
| TRAIN metric calls / rows | 48 / 432 |
| Nested public projections | 2,540 |
| Owner construction / restore / fork | 7 / 11 / 4 |
| Snapshot copies / fit validations | 2,296 / 2,611 |
| Snapshot API calls / nominal coefficient visits | 1,412 / 2,345,112 |
| Teacher choices / physics steps | 18 / 54 |
| Native construction / reset / local Gym seed / successful close | 6 each |
| Observed pickle output calls / byte visits | 1 / 2,651 |
| Forbidden model/source-loader work | 0 |

These are named API/product/output visits, not total copies, allocations, FLOPs
or OS I/O. Source/integrity hashing, other serialization and Python work remain
inside time/RSS bounds. Six bounded environment references prevent lifetime-ID
reuse; all six unique owners were observed successfully closed.

There were 90 arrival-row exposures (72 unique rows and 18 A-return repeats)
plus 162 repeated joint TRAIN rows. Consolidation replay counter 0 does not mean
zero data reuse. Peak fit 36 rows, bank 54; input arrays 598,752B, peak joint
template 85,536B, bank observation-target float64 equivalent 256,176B. These
logical figures exclude metadata, copies and Python/RSS overhead.

Original source 10-update/90s/512MiB authority consumed 10 updates, zero separately
charged consolidation replay rows, and 0.252358s. Its owner, limits, birth clock,
source/deployed actor and permanent receipts remained unchanged. Readout retained
its original separate 150s/512MiB authority with zero updates/replay. Whole workflow
elapsed 2.2362552s within its original 300s/290s work deadline. None of these
nested clocks or allowances was renewed. NumPy/Python caller RNG was preserved;
six locally seeded Gym generators are separately charged.

All four final games completed successfully, each nine steps, return 0.49375,
terminated true, truncated false and closed true: G blue, G red, original A blue,
original A red. Both G teacher TRAIN collections completed and closed in nine
steps. No new fits, optimizer updates, promotion or learning occurred in readout.

| Completed updates | Training rows | A correct/18 | B correct/18 | C correct/18 | G correct/18 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 2 | 2 | 2 | 2 |
| 2 | 18 | 18 | 13 | 11 | 15 |
| 4 | 54 | 18 | 18 | 13 | 13 |
| 6 | 108 | 18 | 18 | 18 | 14 |
| 8 | 180 | 18 | 18 | 18 | 18 |
| 10 | 252 | 18 | 18 | 18 | 18 |

All 48 TRAIN scores and all six endpoints are saved, without selecting a preferred
endpoint. G changes from 14/18 after ABC to 18/18 after its two fits; old ABC
remains 18/18. A remains correct on return, so recovery after observed forgetting
is unavailable. Final A cross-entropy 1.189879070665448 is worse than its first
fit's 1.165422180485597; final C 1.1898790706654485 is worse than its first fit's
1.1812464973755996. Preserve these confidence regressions.

The earlier cold-fit recurring case has the same correctness curve and four
successful games. This is descriptive reuse, not a newly controlled comparison:
old observer profiling and execution conditions differ, so the large elapsed-time
difference cannot establish efficiency. Minimum-change training establishes no
additional navigation or retention benefit on this fixed recipe. The simpler
cold recipe remains a working option. No seed, threshold, metric, schedule,
replay dose or horizon was changed after results.

## Capture, review and remaining work

CLI output is 1,810,043B; captured original report is 61,185B compressed. Both
were admitted by canonical equality and same-file/content pins. Work receipt
shows CLI exit 0, qualified true, budget_ok true, no error/blocked imports and
all original owner/handoff/RNG checks true. `native-stderr.log` is empty.
Saved-only `<python> -B <stage>/analysis.py` exited 0 in 0.009432100s with
25,743,360B peak, rechecking pins and all actual gates without model/simulator
imports. One of two permitted analysis invocations was used; no unused native
allowance remains. Stage 3,781,668B before final report/review; dedicated cache
36,305,040B, within 16MiB/64MiB. Native stdout 1,813,670B is retained locally.

Lead source/procedure/result self-review passes. Independent POST ACCEPTS this
one frozen functional case and its bounded interpretation: four readonly calls,
1.2685977s exact outer wall. It checked actual games, all curves, source/driver
hashes, saved identity/content pins and canonical CLI/raw equality. A redundant
catalogue preview was truncated; its relevant scalar reuse/receipt fields were
explicitly checked in the final read. This is independent review, not numerical
replication. The later planned #369 report link is outside that review's scope.
See stage `REVIEW-POST.md` and timing receipts. The finite exception ended.

Skipped unchanged #367 math/software/native control fixtures, full suite,
clean-clone/current-wheel qualification, Linux native games (missing optional
packages), GPU/robot/browser/remote CI, heldout/final/probe/controller studies.
No installation, dependency download, Git publication, deletion or global changes.
Unrelated dirty/ignored work and every prior negative/PARTIAL/SPENT case remain.
Parents #18/#21/#23, reliable acquisition/retention/recovery, broader R2/R4/R5/R7,
scientific advantage and release readiness stay open. This final-only fixed-TRAIN
readout does not establish native-before/after acquisition or recovery,
generalization, reliable recurrence, matched total computation or CPC benefit.

Critical-path reassessment: repeating this jointly retained fixed recipe would
not answer a new adaptation question. Original R2.1 diagnostics and isolated F
probes already exist. The current affine probe's one disposable child still
cold-fits and has no selector for the newly implemented minimum-change update.
[Planned #369 / R2.1-MG.MIN.P](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/369)
is the next learner outcome: an explicit option in that existing isolated probe,
with source/default/role/budget preservation and separate old-task interference
readouts. Its six-update after-C source admission remains; the ten-update G
source is not silently substituted. This planned software task grants no new
native allowance and does not retry the failed retained-A4 #217 case. Exact next
action: inspect its existing source/fork/caller and generic probe dispatch,
freeze small fixture costs, then implement and execute default/selected/isolation/
paid-failure checks. No capture, packaging, update-count or audit sequel.

[Canonical scoped #368 acceptance and handoff](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/368#issuecomment-6545477112).
Issue #368 is CLOSED with all five scoped criteria evidenced; #369 is OPEN and
planned, with all five criteria unchecked. Parent update receipts:
#18/6545481797, #21/6545482083, #23/6545482273; all parents remain OPEN.
No unused native allowance carries forward.
Final documentation whitespace check initially exited 2 for added trailing blank
lines in the three archival files. Removing only those new EOF blanks produced
exit 0; existing LF/CRLF advisories remain. Native/source/driver bytes were
unchanged. Original handoff retains that initial failure; closeout.json records
the correction. No tests or model work were repeated.
