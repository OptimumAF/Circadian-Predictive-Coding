# Nuisance episode identity repair — R7.5-MG.N.ID / #237

The native failure in #236 exposed a naming collision: the new nuisance role
fell through to context/cue-only episode names, so grey/red A3 shared IDs. The
existing duplicate validator correctly stopped the run. This repair changes
only `_live_episode_id` for `grey_red_A3_nuisance_TRAIN`: names bind context,
branch, wall colour and cue and stay within the native64-byte ASCII limit.
Original default, A3/A4 and private role names remain unchanged. No validation
rule, algorithm, source, policy, clock, budget or scientific outcome was changed.

The scripted nuisance endpoint now calls the real production name builder;
previously it invented unique fixture IDs and missed the integration defect.
Tests cover all five admitted native update counts2..6, separate four names,
native size limits, actual five-control integration, unchanged legacy names,
and continued refusal of duplicated nuisance evidence.

## Evidence and commands

`artifacts/runs/minigrid-nuisance-identity-20261027/` retains preedit source/tests,
combined patch, existing finite qualifier/supervisor, XML/logs/parent receipts,
static diagnostics/repair and final review pins. This is source/procedure/result
SELF-REVIEW, not independent acceptance. No native science was rerun.

Test-first command:

```powershell
.venv/Scripts/python.exe -B artifacts/runs/minigrid-nuisance-identity-20261027/supervise.py win-regression .venv/Scripts/python.exe -B artifacts/runs/minigrid-nuisance-identity-20261027/qualify.py win-regression tests/test_minigrid_nuisance_validation.py::test_should_distinguish_native_nuisance_episode_identities
```

It failed as expected at the first case under `--maxfail=1`; four parameter
cases were not run in that attempt. .69759760005s body/1.7410255s parent,
RSS82,345,984B, all scripted/genuine counters zero. After the six-line production
fix, final10 cases passed on Windows and cached WSL Ubuntu; exact five selectors,
cached Python path and CPU1 environment are in the two `*-final-parent.json`
receipts. Windows7.03763719997s body/8.2401083s parent/RSS85,942,272B; Linux
11.543696808s/19.7789443s/RSS77,570,048B. No final test failures/errors/skips.
Windows cumulative11 instances:10pass/1expected pre-fix failure. Linux10pass.

Each final cohort paid61 scripted owners/1546 snapshots/12 scripted fits198
rows/38 predictions666 rows/seven fake four-game calls/four counterfactual
calls/two fake collection environments/18 teacher-step upper bound. Genuine
Native constructors/fits/predicts, LAPACK, native world constructors and forbidden
imports were guarded and remained zero. No TRAIN payload/probe/final/oldraw
read, native teacher/physics, model random draws or benchmark execution occurred.

Four-file final Ruff format/check passed. Initial default and Linux-target
mypy both refused an inherited fixture module imported under two names. After
those two tooling failures, the ordinary `--explicit-package-bases` option
resolved mapping; both targets then passed without ignores or source changes.
All exact commands/outcomes are in `static-final.json` and
`static-module-repair.json`. Source review compares the actual combined patch
with the preserved dirty preedit, not just the new branch in isolation.
All-attempt static time29.38293559989s/90; mtime-selected touched cache
113,131,520B/160MiB (not growth). Windows cumulative software body7.73523480003s,
Linux11.543696808s; both remain within their original60s allowances.

Reused #235 guards/defaults/evaluation isolation remain supported within their
accepted scope; the changed identity boundary has the focused combined gates
above. Full original suite, whole repository, clean clone/current wheel/install,
independent review, native nuisance performance, CPC comparison, UI/GPU/remote
CI and release checks were skipped. No dependencies or framework were added;
unrelated dirty work, previous failures and frozen236 artifacts were preserved.

## Remaining claims

#235 may reclose its software regression after these gates; #236 remains
FAILED/SPENT with its native criterion unchecked. This fix does not establish
task acquisition, retained performance, return recovery, utility, efficiency,
CPC advantage or release readiness. Broader parents/full goal remain open.
Historical plans/log remain frozen under AGENTS; verified issue/tracker receipts
are the live handoff.

Next precise action: profile snapshot/copy work at the existing target-free
prediction boundary using its current native-free evaluator fixture, reusing
prior measurements if available. Measure before changing anything, preserve
strict lifetime/evidence guards, and do not retry the spent experiment.
