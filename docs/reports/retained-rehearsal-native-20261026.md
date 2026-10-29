# Retained A3/A4 rehearsal: software delivered, native comparison incomplete

Live task: [R2.1-A4.J.N / #227](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/227).
Parents21/23/33 and the full repository goal remain open. This turn made
progress: the existing fresh Python workflow now supports readonly games after
both equal-row rehearsal arms. The sole native comparison failed at the first
private episode identity boundary. Its scientific allowance is spent, with no
retry, seed selection or budget renewal. Source/procedure/result review is
**self-review**, not independent acceptance, under current AGENTS.md.

## Delivered behavior

`run_fresh_adaptation(reference_policy=ReferencePolicy("drift"),
retained_rehearsal=True, native_games=True)` uses the existing source, stream,
collector, comparator, evaluator and ordinary capture path. Default behavior
is unchanged. The formerly invalid rehearsal-plus-games combination is now
explicitly valid; mixed rehearsal/probe/return/controller recipes remain refused.

After both arms finish, their paid current-run endpoint dictionaries are turned
into exact typed snapshots with existing constructors, copy validation and full
`asdict` equality. There is no historical loader or refit. The evaluator admits
only the named retained arms, exact6updates/108rows/two27-row heads, no failed
work, and no short-return observer. Old matched8/180/head36 and short-return
6/72→8/108 admissions remain separate. No core learner/math/optimizer changes.

Games are fixed F blue/red probe_test, original A3 blue/red and A4 blue/red TRAIN,
seed17/max16/raw K7 argmax. Both arms fit before any post-fit game. The original
whole-workflow0-update/0-replay150-second512MiB owner spans all preparation and
readouts. Completed comparison/arm clocks are preserved. At most two paid game
records retain attempts, returns, known partials and unknown unreturned work.
Changed or replaced paid evidence, source or budget bindings stop later work.
No game result selects fitting, deployment, promotion or final evaluation.

Changed production boundaries: `src/adapters/minigrid_policy_evaluation.py`
(retained admission and two exact-role episode-ID branches),
`src/adapters/minigrid_affine_rehearsal.py` (fresh wrapper only), and
`scripts/minigrid_a4_adaptation.py` (descriptor/private composition only).
Existing225 comparator/classes and the pre-F reader have unchanged ASTs.
The public fresh forwarder already passed native_games and remains unchanged.

## The native attempt

Before GO, a prospective amendment added one target-free pre-F source reading
through the existing before_probe evaluator, before either private arm fits.
Its original whole owner, paid report and costs were guarded through later
work. The fixed arms, seeds, doses, metrics, comparison direction and native
caps stayed unchanged. Outcomes did not choose subsequent work.

| Observation | Source before F | Current-only endpoint | Joint A3/A4/F endpoint |
|---|---:|---:|---:|
| Native F goals | 0/2, complete | Unknown | Unknown |
| F diagnostic correct rows | 14/18 | 18/18 | 18/18 |
| Old TRAIN diagnostic correct rows | 36/36 | 31/36 | 36/36 |
| Endpoint updates/row exposures | 4/54 | 6/108 | 6/108 |
| Head rows per cue | 18 | 27 | 27 |

Both private fits per arm completed. Current-only adds54F exposures (18base,
36extra); joint adds18F plus36old TRAIN reuse exposures. Equal rows and fit
counts are not proof of equal computation. Joint old diagnostics include its
training rows. Even perfect diagnostic fit does not establish native goals.

The first current-only private constructor rejected its episode ID for exceeding
64UTF8bytes, before reset or a physics step. Its six slots are one failed plus
five not_run, all success values unknown. The joint readout was not attempted.
This is an implementation failure, not negative evidence about joint retention.

The software fixture had checked uniqueness but missed the real length limit.
It now calls the same production `_text(...,64)` validator before its literal
environment constructor. That reproduced the failure. Compact `current`/`joint`
tags replace only the two new long role names in IDs, preserving method,
update, context, branch and cue. All12planned IDs remain unique and fit the real
boundary; older IDs retain their original format. Native execution was not
repeated after this repair.

| Native cost, including failed work | Observed |
|---|---:|
| Fits / rows and LAPACK solves / rows | 8/162 and8/162 |
| Source fits / rows / prior-row exposures | 4/54/18 |
| Current fits / rows / extra F exposures | 2/54/36 |
| Joint fits / rows / reused old TRAIN rows | 2/54/36 |
| Predictions / rows | 148/532 |
| Environment constructors attempted / returned / closed | 17/16/16 |
| Physics steps attempted / returned | 172/172 |
| Packaged TRAIN reads / bytes | 1/94,705 |
| Empty native owners / snapshots | 21/2,393 |
| Body / checked parent seconds | 15.6687185999 /16.0606629001 |
| Sampled peak RSS bytes | 95,588,352 |

The baseline alone cost4.2259705001seconds/32predictions/32steps/two closed
environments; no fits or source reads. The failed current-only readout cost
0.1714766000seconds/one unreturned constructor/12snapshots, no predictions or
steps. Unreturned constructor work remains unknown. All measured counters stayed
within the frozen8fit/162row/30environment/480step/150body/180parent/512MiB caps.

The five original owners recorded updates0/4/4/2/2 and replay0/0/36/0/36.
The last three retain the same initial clock. Emitted source and actor flags
are true, as is the rehearsal source-state flag. The final success-only global
RNG, original-budget and restored-binding assertions were **not reached**;
their terminal flags are null. Do not promote those unexecuted assertions into
verified invariants.

Raw capture is1,371,706bytes, exclusive write-once, never reopened. Its digest
comes from the original in-memory write. Scalar capture is23,500bytes. Postflight
read only emitted scalar/work/parent records, source pins and raw file metadata.
The scalar contains no traces, target arrays, parameters or prediction arrays.
RSS sampling spans optional imports through raw/scalar writes and sampler exit;
the final<=16KiB work receipt/console is outside that RSS scope, inside parent
time. There is no instantaneous/whole-process peak claim.

## Executed checks and failures

Before GO,52focused literal tests passed per platform. Windows first produced
8pass/1fail: arm/branch IDs collided; the first exact-role ID change fixed
uniqueness. That change was too long for the real environment, causing the
native failure above. Afterwards Windows reproduced the constructor-boundary
error once, then two affected tests passed per platform: the complete ordinary
fresh workflow and all12IDs plus legacy-format preservation. Total53distinct
cases/platform; Windows64instances=62pass/1fail/1setup error, Linux54all pass.
No skipped cases. Unchanged checks were reused where only ID spelling changed.

All literal attempts count: Windows33.9995991seconds body/40.7639197parent,
RSS98,037,760bytes; Linux40.437060377/56.7545130,RSS87,384,064bytes. Windows
763literal owners/7,729snapshots/91scriptedfits2,025rows/526pred5,598rows/
64fakeenv126steps/30collections102closed144literalteachersteps; Linux627/
6,407/75/1,701/442/4,746/52/102/26/90/108. Readout/source calls60/45. Literal
models and simulated teacher/game ports perform no genuine fits, learned
forwards, LAPACK, physics, teacher collection or optional simulator imports.

Four pure projection checks passed three times/platform (initial/final plus
four affected baseline checks); these test positive, zero-goal, partial unknown,
excluded payload and cap behavior without native markers or optional imports.
Their parent times are separately retained and within software caps. Final
10-file Ruffformat/check/default+Linuxmypy passed before GO. After the repair,
the two changed files passed the same four static gates. All-static conservative
upper bound101.0389808seconds includes the asynchronous initial mypy interval
and intervening work, not an exact isolated runtime. Initial two list-annotation
errors, one metadata assertion expecting three app functions instead of two,
and a no-change protocol patch-context miss are preserved. No ignores added.

Exact commands are retained in `*-parent.json`, `gate-final.json`,
`repair-statics.json`, `software-final.json` and the issue GO comment. Main ones:

```text
.venv/Scripts/python.exe -B artifacts/runs/retained-rehearsal-native-20261026/supervise.py windows-2 .venv/Scripts/python.exe -B artifacts/runs/retained-rehearsal-native-20261026/qualify.py windows-2
.venv/Scripts/python.exe -B artifacts/runs/retained-rehearsal-native-20261026/execute.py
.venv/Scripts/python.exe -B artifacts/runs/retained-rehearsal-native-20261026/supervise.py windows-4 .venv/Scripts/python.exe -B artifacts/runs/retained-rehearsal-native-20261026/qualify.py windows-4 tests/test_retained_rehearsal_games.py::test_should_read_both_arms_only_after_all_fitting_on_the_original_owner tests/test_retained_rehearsal_games.py::test_should_fit_all_twelve_distinct_retained_ids_inside_the_real_boundary
.venv/Scripts/python.exe -m ruff format --check src/adapters/minigrid_policy_evaluation.py tests/test_retained_rehearsal_games.py
.venv/Scripts/python.exe -m ruff check src/adapters/minigrid_policy_evaluation.py tests/test_retained_rehearsal_games.py
.venv/Scripts/python.exe -m mypy --explicit-package-bases --follow-imports=silent src/adapters/minigrid_policy_evaluation.py tests/test_retained_rehearsal_games.py
.venv/Scripts/python.exe -m mypy --platform linux --explicit-package-bases --follow-imports=silent src/adapters/minigrid_policy_evaluation.py tests/test_retained_rehearsal_games.py
```

Linux uses the existing cached Python under WSL Ubuntu, CPU1, no install or
download; its exact environment/path/timeout command is in linux parent records.
No broad repository, installed-wheel, UI, release, CI, GPU, independent-agent
review, new BP/PC/CPC cohort or native retry was run or claimed passing.

## Evidence and handoff

Local stage: `artifacts/runs/retained-rehearsal-native-20261026/`. Original
qualification/protocol/GO pins remain frozen. All25spent source pins match,
using exact `.spent.py` snapshots for the two files repaired after the failure.
Other23pins remain current; the repair does not rebind earlier evidence.
`postflight.json` contains scalar-only result checks. Both repair patches and
original dirty snapshots are retained. Stage before closeout1,911,361bytes,
logs1,695bytes, below20MiB/256KiB limits; final receipts retain final footprint.
No Git mutations, dependencies or unrelated changes. Prior217/220spent failures,
222fit-only result and224F2/2old0/4→F0/2old4/4 mutual forgetting are unchanged.

The scoped software behavior and repair pass. The native acquisition-plus-joint-
retention question remains **unchecked/incomplete** in227, not silently weakened
to successful fitting. There is no CPC, generalization, matched-compute,
efficiency or release claim. Existing learnability and action-margin diagnostics
already exist; adding another diagnostic framework is not the critical path.

Next precise implementation action: expose the existing retained-rehearsal
Python recipe through the existing affine CLI's describe-before-IO route,
preserving defaults and failed/unknown outcomes, with literal argument,
dispatch and failure-capture checks. Why this: the current CLI only exposes the
ordinary A4 adaptation, while this fixed private adaptation is Python-only.
This makes the implemented learner workflow usable without another native
comparison, capture framework or spent-budget renewal. Freeze its small software
scope in an owning issue before edits. Joint native retention remains deferred;
any later comparative experiment needs a distinct useful scientific question,
prospective protocol and honest accounting, not a renamed retry of227.

Live issue handoff replaces duplicate plan/log updates under current AGENTS.md;
the historical plans and log remain frozen. Full goal ACTIVE and incomplete.
