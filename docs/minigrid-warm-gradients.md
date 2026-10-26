# Gradient geometry at the accepted A128 state

This RNN-only use case reconstructs the accepted fixed A128 learner, then
measures the existing TRAIN cue-gradient diagnostic on its frozen state. The
reference JSON contains no native parameters or optimizer state, so reconstructing
128 updates is newly charged work. No other learners or probes are retrained.

## Preview and run

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_gradient_smoke --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_gradient_smoke --output artifacts/my-warm-gradient-run.json
```

The output parent must exist; an existing output file is refused before model
work. Each explicit run is a new experiment. The recorded case has a separate
[frozen protocol and allowance](../artifacts/runs/r21-warm-gradient-20261016/SCOPE.md);
its outcome is recorded in [RESULTS.md](../artifacts/runs/r21-warm-gradient-20261016/RESULTS.md).

The original seed17, native148/H8/K7 mean-CE learner, SGD rate.03/momentum.9 and
A-only blue/red training order stay fixed. The existing warm helper reuses cached
original TRAIN packets and the original stream. All five raw checkpoints and
both native cue-pair endpoint laws must match the reference before readouts.
No fresh teacher, reward update, weighted control, replay selection, probe or
final data enters this route.

The returned restored source, guard and six original TRAIN packets feed the
existing checkpoint scorer at128updates1152rows. It preserves six raw scores,
then reads A/B/C_train blue/red gradients on detached clones with zero recurrent
activation. The diagnostic applies zero SGD steps or publications. All signed
and undefined joint/per-parameter results are retained without selecting a seed,
context or favorable outcome.

## Limits and ownership

The recorded protocol uses75pilot-body/90parent seconds,45warm/20diagnostic,
768MiB observed RSS, CPU/BLAS1, GPU0 and32MiB output. The original warm guard's
45second deadline remains binding during readout. Its existing monotonic reads
may advance last_clock; budget config, clock identity/start, charges and paid
warm evidence are preserved. A separate max0-update diagnostic budget does not
renew the completed training allowance.

CLI Torch import/thread setup precedes the pilot body timer and sampler. The
recorded parent includes that latency; retained imports are in observed RSS,
but import peak and import latency are not measured by the pilot's body/RSS
record. Sampling is not a full allocation or model-only memory measurement.

Output separates the original warm work, six additional raw calls54rows, and
three gradient pairs54forward/54backwardrows,18geometry reductions and64368
logical float64 feature/target bytes. These bytes do not include Python records,
snapshots or tensor/reducer storage; total time/RSS covers that later work.
Native validation also builds temporary SGD configuration checkers without
stepping them. Failed warm acceptance prevents diagnostic admission. Returned
readout prefixes and private paid evidence survive later refusal with incomplete
status and a nonzero CLI exit; unknown model/input changes are never repaired.

## Isolated finite-update cue check

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_gradient_smoke --describe --cue-update-check
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_minigrid_gradient_smoke --cue-update-check --output artifacts/my-cue-update-run.json
```

The explicit option retains the original warm and gradient readouts, then forks
the same accepted A128 full native state twice. One original A-blue episode
updates one fork; one A-red episode updates the other. Original SGD momentum and
gradients are carried forward. All six TRAIN packets are scored on both129/1161
states. Neither updated learner is deployed, promoted or used as a benchmark.

`cue_update_readout` retains both arms, their native pre-update losses and full
raw readouts. `deltas.blue`/`red` contain per-packet after-minus-before CE;
`blue_minus_red` compares the two after CEs. Lower CE is better. Both arms and
all contexts stay in the output, without selecting a favorable result.

The separate max2-update/atmost20second budget accounts for disposable updates.
The original max0 diagnostic budget continues to limit baseline/raw scoring and
the entire extra phase; its20second deadline and the original45second warm guard
remain authoritative. The nested budget cannot extend either. Paid native
commits are reported separately from the last verified budget observation if
recording a charge fails. Failed readouts retain their prefixes, with null
deltas; unknown changes are refused without restoring caller state.

This adds two SGD updates18rows and12raw forwards108rows: total130newupdates/
1170rows including reconstruction. No new teacher, live episode, gradient pair,
probe or final data is added by the option. See the separately frozen
[protocol](../artifacts/runs/r21-cue-updates-20261016/SCOPE.md) and its recorded
[outcome](../artifacts/runs/r21-cue-updates-20261016/RESULTS.md); an explicit run spends its allowance even on failure. This tests finite
native updates with carried momentum, not pure-gradient causality, useful
acquisition, fresh-task adaptation, retention/recovery or CPC advantage.

## Files and checks

```text
src/adapters/minigrid_warm_gradient_pilot.py  existing warm+scorer composition
scripts/run_minigrid_gradient_smoke.py       describe/execute/new-file output
tests/test_minigrid_warm_gradient_pilot.py   bounded orchestration/refusal fakes
src/adapters/minigrid_cue_updates.py         two isolated original-CE updates
tests/test_minigrid_cue_updates.py           component isolation/failure behavior
tests/test_minigrid_cue_update_route.py      opt-in/default CLI composition
```

```powershell
.venv/Scripts/python.exe -X utf8 -B -m pytest -q tests/test_minigrid_warm_gradient_pilot.py tests/test_minigrid_cue_updates.py tests/test_minigrid_cue_update_route.py
$warmGradientPaths = @('src/adapters/minigrid_warm_gradient_pilot.py', 'scripts/run_minigrid_gradient_smoke.py', 'src/adapters/minigrid_cue_updates.py', 'tests/test_minigrid_cue_updates.py', 'tests/test_minigrid_cue_update_route.py')
.venv/Scripts/python.exe -X utf8 -B -m ruff check --no-cache @warmGradientPaths
.venv/Scripts/python.exe -X utf8 -B -m ruff format --check --no-cache @warmGradientPaths
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform win32 @warmGradientPaths
.venv/Scripts/python.exe -X utf8 -B -m mypy --follow-imports=silent --no-incremental --no-sqlite-cache --cache-dir=NUL --platform linux @warmGradientPaths
```

No dependencies or environment variable names are added. The CLI configures the
existing CPU/BLAS/GPU controls only for its explicit local process. Architecture
and extension rationale follow ADR-0362; this introduces no new trainer or runner.

## Interpretation

The accepted case measured A joint cosine−.99214785 (all five parameter
components negative), B+.57323365 and C_train+.32066771, with a negative C_train
output-weight component. Original A128 live parity still fails blue and succeeds
red. The run passed in10.6773pilot/13.6108parent seconds with31focused checks;
its allowance is spent. See the linked results for all costs and limitations.

The isolated update case passed68focused checks and one reviewed native case in
16.6003pilot/19.8710parent seconds. Each A cue update improved its own raw CE and
worsened the other: blue arm−.01399/+.01259, red arm+.04007/−.02469 (blue/red).
The blue arm changed teacher-forced exact sequence accuracy to blue1/red0;
the red arm kept blue0/red1. Both-cue fit remains false. No129fork live rollout
was run, and this is not useful adaptation or retention evidence. All B/C_train
effects, costs and unchanged baseline geometry remain in the linked results.

Negative whole-episode cosine shows local opposition for this pair at this state;
positive alignment may hide cue-decision conflicts or shared-label effects. A
zero norm remains undefined. Neither result establishes causality, useful skill,
retention/recovery or a circadian advantage. Use the original warm raw/live
outcomes as skill evidence, and preserve failed and inconclusive findings.

Future measurements should reuse these ports and declare their data, objective,
cost and interpretation before execution. Keep fresh-task probe updates separate
and final evaluation unavailable to training, tuning and selection.
