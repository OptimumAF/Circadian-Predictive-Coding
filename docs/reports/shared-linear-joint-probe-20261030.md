# Shared-head joint probe software evidence — #357

R5.3-LIN.JOINT.PROBE implements explicit joint/current_only shared-head LS updates
through the existing isolated ordinary fresh/return runner. This qualifies software,
not a scientific retention benefit. Full goal and parents #18/#21/#23/#24 remain
open. [Canonical scope and amendments](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/357).
Physical checkout HEAD d9119e80926a139a2f38460564d7eab1b6d2a2b0 has substantial
inherited changes, preserved. No Git publication, installation or global edits.

## Working behavior and boundaries

`src/core/shared_linear_replay.py` validates bounded owned arrays for both controls,
rejects contradictory identical inputs and enforces the existing128-row limit.
Joint concatenates; current_only repeats complete current packets to equal rows.
Equal rows do not establish equal SVD work. The existing LS kernel stays unchanged.

`scripts/ticket_linear_rehearsal.py` certifies the live source at12/72: explicit
birth-time replay consent, TRAIN-only roles, actor/learner lineage, actual arrived
labels/applied receipts, logical clock12, unchanged data and owner identity.
It retains fixed original packets0/4, without task-score selection. Diagnostic
monitors, probe_test and final evaluation supply no fitted labels; probe_train
never enters the source bank. Only detached private owners are updated.
Ordinary API/CLI opt in; existing shared/separate-head defaults remain. The cost
summary runs afterward without influencing fitting, selection or promotion.

One separate auxiliary32-admission/384-extra-row/30-second/512-MiB session charges
before construction/fitting. Original source12-update/30-second/512-MiB and F/return
authorities are not renewed. Source time continues. Only observed owned admissions
advance sharing counters. Changed source, public evidence or foreign counters
refuse further work; original limits/birth/known clock/paid records survive.
Known application requires a verified post-state. Lost returns preserve observed
application; unreadable state stays unknown, not zero. A returned diagnostic with
no declared application fails. Failed prior work cannot admit another update.
The receipt ledger is bounded32events. The callback intentionally keeps this one
admission/failure boundary together; no new orchestration framework is introduced.
[Usage, exact CLI and cost fields](../shared-linear-probe-rehearsal.md).

## Executed checks and preserved failures

Stage `artifacts/runs/shared-linear-joint-probe-20261030/` contains all7 terminal
process receipts, logs, SPENT markers, preimages, owned diffs and statics.
Per invocation60s/256MiB; cohort real-work caps16LS/256fit rows,48scheduled training
products/768rows and256prediction affines/2048rows. Source/full runner fixtures
use separately counted scripted fitting, avoiding accidental scientific runs.

| Receipt prefix | Result | Body seconds | Peak sampled RSS bytes | Real LS / rows | Scripted calls / rows |
|---|---|---:|---:|---:|---:|
| windows | 24 pass, 2 fail | 2.554930100 | 75624448 | 4 / 66 | 213 / 1287 |
| windows-affected | 1 pass, 1 fail | 1.497772600 | 69001216 | 0 / 0 | 24 / 144 |
| windows-generic | 1 pass | 1.487755700 | 67280896 | 0 / 0 | 44 / 600 |
| windows-final | 23 pass | 4.446961200 | 76664832 | 0 / 0 | 137 / 1503 |
| linux | 46 pass | 23.441781997 | 76419072 | 4 / 66 | 313 / 2559 |
| linux-clock | 2 pass | 8.988839592 | 62672896 | 0 / 0 | 24 / 144 |
| linux-application | 1 pass | 5.151944618 | 63537152 | 2 / 33 | 14 / 102 |

Initial failures were fixture errors: consent is enforced at the TRAIN handoff,
not a readonly snapshot getter; handwritten generic cohorts had wrong evaluation
cardinality, then mismatched contexts. Simplified to the accepted cohort builder;
no boundary weakening. All failed attempts/charges remain.

Windows qualifies37 distinct new plus8 legacy metadata cases via reuse/affected
checks; Linux40 new plus8 legacy. Later foreign-actor/clock/application edge cases
are Linux-only. Relevant positive/failure checks followed final guard changes;
unchanged broader cases were reused, not claimed as fresh full reruns everywhere.
Final combined-tree tests cover pure packets, consent/roles, ownership/receipts,
source/logical clocks, detached owner, paid failures/lost returns, evidence/counter
corruption, generic32-slot sequence, ordinary API/CLI/defaults and cost projection.

Totals:10real LS/165rows,30scheduled training products/495rows;160generic prediction
affines/720rows,0typed-logit calls;769scripted calls/6339rows;47.569985807check-body
seconds. Named ports/products are not complete physical copies/FLOPs. Existing
teacher/cohort helper work is included in time/RSS, not separately metered.

Scripted bounds changed prospectively, retaining failures: planned rows3072→4096
(comment6540033702), then768calls/6144rows (6540094261,6540164860); logical-clock
fixtures raised rows to6400 (6540255801); the final application-contract fixture
raised calls to800 (6540307396). Real numeric/time/RSS limits stayed. The last
fixture uses one certified source, two real tiny positive updates and two bad
private owners. Unused correctness allowance grants no future science allowance.

Final scoped Ruff/format/Linux-platform mypy on11 source/test/runner files PASS
(`statics-application.json`,0.7182524s). Initial12 then5 typing errors retained:
new annotations/fixture call corrections and existing skip-import renewal identity
narrowing. A non-None assertion after its existing identity check resolves the
latter without changing normal renewal behavior. Dedicated cache0→10700243B
under64MiB; stage excluding cache under4MiB. Earlier static successes/failures stay.

## Commands and reproducibility

Physical cwd `D:\Dev\Repositories\Circadian`; process-local TEMP/TMP=`D:\Dev\_Temp`.
Windows used `.venv/Scripts/python.exe -B artifacts/runs/shared-linear-joint-probe-20261030/check.py`
with prefix and D: pytest basetemp, then explicit selectors for affected/legacy
cases. Linux used the following prefix and the same stage runner:

```text
wsl.exe --cd /mnt/d/Dev/Repositories/Circadian env TEMP=/mnt/d/Dev/_Temp TMP=/mnt/d/Dev/_Temp TMPDIR=/mnt/d/Dev/_Temp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/avery/.cache/circadian/open-jev-native-smoke-20261022/env/bin/python -B artifacts/runs/shared-linear-joint-probe-20261030/check.py
```

Actual prefixes: windows, windows-affected, windows-generic, windows-final, linux,
linux-clock, linux-application. Each `PREFIX-checks.json` preserves every admitted
expanded node ID/result and cost. `executed-check-roster.json` records those exact
node rosters with the command family, not fabricated historical argv. Exclusive
SPENT prefixes must not be rerun. Runner disables plugin autoload/pytest cache and
forbids neural/physics imports. Same live process handles were polled to completion.

Statics: `.venv/Scripts/ruff.exe format`, `ruff.exe check`, and
`.venv/Scripts/mypy.exe --platform linux --follow-imports skip`, exact11 paths in
`statics-application.json`, stage/cache RUFF_CACHE_DIR and MYPY_CACHE_DIR.
Five owned diffs preserve inherited changes. Summary preimage was reconstructed
by reversing only the just-owned projection edit; the after-edit capture remains.
This is disclosed provenance, not an authenticated original historical snapshot.

Skipped: full suite, clean clone, wheel/build/install, unchanged kernels/baselines,
full genuine ordinary comparative case, GPU/physics/browser/remote CI, release and
numerical replication. Failed patch anchors made no source change; guessed paths
and Windows globbing were simplified to actual file discovery. No model restart.
The last archival whitespace check failed for newly appended blank lines at EOF
in both plans/log; only those owned trailing blanks were removed, and the scoped
recheck passed. Failure is saved in `whitespace-archive-initial.json`; all11
qualified source hashes still matched. No model/test execution was repeated.

## Critical path and claims still open

Reuse accepted repeated-update and R2.1/matched-method evidence, not another
capture/audit increment. #356 remains mixed unpaired evidence: old skills3→2→3;
designated fresh tasks already2/2 before updates, so no new acquisition established.
All adverse curves stay. No CPC advantage/generalization/release claim follows.

After software review, next implementation action is to freeze a small TWO-arm
ordinary current_only-versus-joint protocol using this entry/runner, with newly
born consented sources, duplicated source work, complete diagnostics, fixed seeds/
metrics/direction and failure policy. Review before fitting. Spent #356 authority
cannot be retrofitted; its144private rows cannot replace this528-row control.
This tests retention/interference under the existing fresh ceiling, not new task
acquisition or matched computation. No experiment allowance is granted here.

Finite independent actual-source/evidence review ACCEPTED the scoped software:
`artifacts/runs/shared-linear-joint-probe-20261030/REVIEW.md`. Four readonly commands,
all exit0,0.7411296s; all five owned diffs matched the current tree. Full core,
callback/source certification, integration changes, representative critical tests
and saved costs were inspected. This was neither numerical replication nor
exhaustive test-body/usage/hash/amendment-timestamp verification. Exception ended.
Lead self-review, qualified checks and this review jointly support only #357's
software scope. [Planned #358](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/358)
has five unchecked criteria and no numerical allowance. Canonical closeout receipt
is linked from the archival plan/log pointers and stage handoff.
