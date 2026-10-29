# R7.5-MG.R: bounded validation readout sharing

Issue215 scoped SOFTWARE implementation passes54distinct checks per Windows and
cached Linux. Source/procedure/result SELF-REVIEW, not independent acceptance.
Full repository goal and R2/R4/R5/R7/P7/P9/scientific/release parents remain open.

The existing validation API now accepts exactbool `share_readouts=True`, only
with an explicit validation recipe. DefaultFalse retains the old schema and
readout behavior. Training, budgets, source loading, model math, four original
chooser-fitting trials and all-before-endpoints chronology are unchanged.

Why this: native214's completed periodic/drift readouts cost22.6823/21.8523s.
Its original150s allowance expired in the fitted readout. That run stays FAILED/
SPENT: fitted endpoint and final global checks UNKNOWN, no retry, reopening of
raw output, rescore or inferred state equality. This software change grants no
native allowance and cannot retroactively fill214's missing evidence.

## Behavior and boundaries

`scripts/minigrid_controller_learning.py` adds only the validation-only flag and
dispatch. `scripts/minigrid_controller_validation.py` keeps a local cache for one
call, warm plus at most five candidate entries, discarded on success/failure.
Existing warm-baseline reuse remains. Additional reuse requires equality of
the COMPLETE native snapshot and exact readout arguments, under the same study's
guarded source and fixed game protocol. Counts, predictions or digests alone
never authorize sharing. Parameter, policy, rank/residual, counter, failed-work
or readout-argument differences miss and reach normal admission/physical IO.

Original source/actor, budget/clock, frozen chooser, collection and paid-result
identity/value guards run before and after each endpoint. The readout method is
pinned in sharing mode, and a changed candidate during readout refuses before
cache admission. Only sealed successful physical results enter the cache;
unreturned/failed work cannot be reused. Corruption/equal container replacement
stops and preserves known paid prefix and unknown work. No new runner,
dependency, cache service, learned mechanism or execution authority.

`reused_from` names the actual origin. `readout_work` separates physical
four-game calls, logical episode slots and reused slots. `readout_seconds`
includes lookup/copy/guard time. Every reused endpoint, including baseline,
has `measures=None` and `utility=None`; shared evaluations cannot manufacture
a cheap controller ranking. Physically fresh endpoints retain the original
objective/weights and timing/cumulative-RSS qualifications. No endpoint feedback
reaches training, chooser fitting, probe selection, final evaluation or promotion.

## Executed evidence and costs

The full literal composition preserves all five independent windows and four
chooser trials. With correct-ready ACKs, periodic/drift/fitted each fit packets
0,2 and acknowledge1,3. Complete native-shaped states match, yielding one new
physical readout for five endpoints instead of the default three.20logical
episode slots include16reused slots; only the physically new endpoint has a
utility. Scripted learning/retention outcomes and all actual TRAIN charges are
preserved. Without ACKs, original pending counts4/2/3/4/3 remain and exact drift/
fitted equality shares one readout (two physical calls total).

This is fixture correctness, not measured native speedup or a completed native
controller comparison. The native214 equality was not observed. Old212partial,
209spent failure, 213unknown-counter debt and all negative studies remain valid.
Existing R2.1/fresh-task and matched BP/PC/neutral/active CPC evidence is reused;
no unchanged mathematical or baseline cohort was rerun.

Tests cover exact flags before models, default schema/IO, all windows first,
full-state and readout-argument misses, cross-call expiry, old pending work,
paid-value/equal-replacement/source/budget/clock/fitter/port corruption, changed
readout input and lost return. `tests/test_minigrid_readout_sharing.py` adds the
new checks through the existing actual stream/guards with literal fits/scripted
game IO; existing validation and ACK test files are in the selected cohort.

Windows54distinct/62passing instances: first48, then13affected, then1clock;
body154.3335678s/parent159.2557326s/RSS127,057,920B. Linux54distinct/54instances:
body120.4615131s/parent129.1449505s/RSS106,029,056B. All0failures/errors/skips.
All original240sbody/300sparent/512MiB and cumulative literal-work caps held.

Windows costs2149emptyowners/40115snapshots/519scriptedfits8712rows/
1387scriptedpredict24336rows/215scriptedgames/140counterfactual calls.
Linux1613owners/29863snapshots/391fits6552rows/1035predict18144rows/
165games/108counterfactual. Genuine native fit/predict/LAPACK/physics/teacher/
probe/final work0; native numerical methods and optional simulator imports
were forbidden, not merely omitted from counters. Sampler measures body RSS
after imports, not full process-tree initialization. No model fixtures in statics.

Initial default/Linux mypy each reported five test-only errors: inherited `run`
signature conflict and missing annotations. Renamed helper `run_shared`, added
annotations, no suppression/product change. Final five-file Ruff check/format
and default/Linux-target mypy all PASS. Statics including failures/formatting
48.2861861s below120s. Initial live cohort passed; source review extracted the
narrow cache-lookup helper to test arguments directly, AST-identical to the
original inline lookup. Remaining original behavior was reused where unchanged.
24of26native214source pins remain identical; only the two declared product files
changed, so old executed qualification capsules were not rewritten.
Stage `*.before` files are pre-fixture implementations, not pre-implementation
or native214 source backups. Touched mypy cache111,038,464B is size, not growth;
the inherited global cache is separate. Stage104,466B before final receipts,
metadata17,009B and each log below256KiB remain inside20MiB/2MiB caps.

Selection mistake retained: affected Windows `-k` filter deselected an explicitly
supplied clock case. Prospectively allowed exactly one clock-only invocation,
14affected instances total within original24/cumulative work/time limits;
no limit renewal. Clock-only check passed before Linux/final closeout. The
prospective exception is issue215 comment6427711108. The
missing legacy downloaded multi-agent prompt was recorded; current AGENTS and
AGENT_PROMPT expressly supersede that prospective workflow, so no new gate.

Exact fixture commands are in `windows-{1,2,3}-parent.json` and `linux-1-parent.json`
under `artifacts/runs/minigrid-readout-sharing-20261026/`. They invoke the reused
corrected literal qualifier through `supervise.py`; Linux uses the existing
cached WSL Python and timeout290s. `statics.json` retains every exact static
command/output, and XML/results retain counts/costs. The Windows second call's
filter mistake is preserved, not edited into a successful command.

For ordinary focused checks:

```powershell
$env:PYTHONPATH = ".;tests"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -B -m pytest -q tests/test_minigrid_readout_sharing.py tests/test_minigrid_controller_validation.py tests/test_minigrid_correct_ack.py
python -m ruff check scripts/minigrid_controller_learning.py scripts/minigrid_controller_validation.py tests/test_minigrid_readout_sharing.py
python -m ruff format --check scripts/minigrid_controller_learning.py scripts/minigrid_controller_validation.py tests/test_minigrid_readout_sharing.py
python -m mypy --explicit-package-bases --follow-imports=silent scripts/minigrid_controller_learning.py scripts/minigrid_controller_validation.py tests/test_minigrid_readout_sharing.py
```

Fullsuite, clean clone/current wheel, native game/time-savings study, new probe,
final, browser, independent review, comparative CPC/generalization and release
checks skipped as outside this software increment. No install/download/GPU/
CI/Git mutation/agents/unrelated reset/archive reconstruction. Historical plans/
log remain frozen under AGENTS; live issue/tracker carry current status.

## Critical path and next action

Do not start another equivalent native ACK/readout cohort or cache/audit sequel.
Useful known-TRAIN conventional learning already exists; R2.1 probe machinery
and four-family controls also exist. The next substantial outcome is an isolated
fresh-task learnability diagnostic from the conventional retained-A3/A4 source,
with old TRAIN monitors and all probe costs separate from its original learner.

Concrete code dependency found: `run_fresh_task_probe` already accepts296features,
disposable owned forks, old TRAIN monitors and endpoint observers, but the
existing affine F-game binding only admits before6/108/head27 and after8/126/head9.
It cannot admit an actual retained-A3/A4 source4/54/head18 without a new explicitly
declared recipe. Next implement one end-to-end composition using that existing
probe runner and a strict optional4/54-source F-readout recipe, preserving the
original6/108 recipe and all role/geometry/source/budget guards. Test isolated
probe updates, old-task monitoring, native readout admission and unknown failures
with literal fixtures before any separately frozen experiment. No old saved
model reader, new orchestration framework, broad counter relaxation, extra
TRAIN/promotion authority or automatic scientific allowance.
