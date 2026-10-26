# F48-only exposure control

## Matched four-method observations

The six frozen PC-family cases now complete the accepted BP pair. All eight
qualify; PC-neutral matches270TRAIN probability rows and32F/A score+diagnostic
points perallocation. Source128PC-family stays5/9percue; activeendpointCEhigher,
noCPCadvantage. F48/replay tradeoffs and all costs/signs remain in the
[human report](reports/matched-exposure-controls-20261018.md). This is fixed
teacher-forced evidence, not live acquired-task recovery/generalization.

The existing learnability CLI now has an explicit extra-F control for BP, PC,
neutral CPC and active CPC. It matches the native calls/rows of fixed old-A replay
without replaying old A. Preview performs no collection or native work:

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-minigrid-history-learnability --invariant-history --qualification-method bp --joint-catalogue --distinct-probe --old-task-readouts --return-to-a --wide-neural --extra-f-exposure --describe
```

Remove `--describe` only within a prospectively budgeted local run. The option
requires the complete explicit H32 route, exact original method/seed/objective,
and refuses old-A replay, games, weighting and extra diagnostics. Omitting it
preserves existing routes.

## Behavior / storage / costs

The original six TRAIN sources fit128joint54-row updates. Source0/128 checkpoints
each seed a disposable learner:48 F updates followed by16 old-A TRAIN return
updates on that same native learner/optimizer. F keeps `probe_train`, evaluation
keeps `probe_test`, A monitors stay readonly TRAIN. Final evaluation is unavailable
to updates, promotion and selection. Source/actor/input/global RNG are guarded.

Validate and copy the original16 F packets once, then pass three reference-only
repetitions through the existing loop. The input bound remains65536entries and
per-probe F+evaluation reserved arrays384912B; monitors+return bring the existing
combined cohort to470448B. Tuple/episode/model/optimizer/copy/report/Python memory,
process RSS and wall time are additional. No48-packet input-copy envelope is built.

F48 has original checkpoints0/12/24/48; replay F16 has logical0/4/8/16, corresponding
to the same actual native0/12/24/48. A-return checkpoints0/4/8/16 stay unchanged.
Every F budget starts at maxupdates48 and every return budget at16; no renewal or
hidden wake counts. Total256calls8064rows/P94/846rows; BP256SGD or PC256updates/
16128nominal latent rowsteps. Replay0/bank0. This matches native training/scoring
counts with replay, not validation/projection/bank/process costs.

## Verification and result

785focused cases/scopedstatics/independent review pass. One original-data native
BP control and saved comparison with the accepted BP replay case verify full
state/native clocks, original budgets, source isolation and unchanged128receipts/
270TRAIN probability rows. Both cues and all32paired curvepoints/64cuecontrasts
are retained. Four-method admission is implemented; this experiment runs BP only.

At source128 F48 gets17/18 F actions after adaptation versus replay14/18, then
16/18 versus15/18 after Areturn. Its old-A CE is higher; both finish Areturn16/18
and fail to recover original exact A-red fit. No live games were measured. This
fixed single-seed teacher-forced tradeoff does not establish gameplay recovery,
generalization, a blanket winner or circadian advantage.

[Exact commands, evidence, negative results, costs and next action](../artifacts/runs/f48-exposure-control-20261018/RESULTS.md).
[Decision](adr/ADR-0442-f48-native-exposure-control.md).

## Extension

Keep defaultFalse call/report shapes and the original input bound. Add controls
through explicit recipe admission in `src/adapters/minigrid_f_exposure.py`, not
new runners or tolerance of changed counters. The two new test modules cover app
ownership/paid failures/evidence refusal and public metadata/CLI dispatch. Future
comparative cases require a frozen small protocol and actual native evidence;
all failed/inconclusive results and unfinished parent requirements remain.
