# Isolated fresh-task diagnostics for the learned affine model

The existing CLI now binds the after-C model to the original reserved Fprobe:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_minigrid_linear_fit_control --context-linear-probe --describe
.\.venv\Scripts\python.exe -m scripts.run_minigrid_linear_fit_control --context-linear-probe --source artifacts/runs/joint-catalogue-study-20261018/bp.txt --retained-source artifacts/runs/catalogue-capacity-new-c-20261024/actual.json --output new-affine-probe.json
```

The second command needs both qualified local sources, optional MiniGrid
dependencies already available, and a prospective budget. Existing outputs and
fit overrides are refused. Old CLI modes/defaults remain. No new dependency.

It collects the existing Fblue/Fred train/test teacher episodes, seed17/max16,
then fits only the firsttwo predetermined training packets on one disposable
fork. The original16train/2test release is validated before selection. All7
classes, same analytic objective, checkpoints0/1/2; no metric selects a dose,
seed, replay, promotion or training input. Probe_test labels stay in metrics.
Train/test share the same deterministic task and seed: this measures fit on
that reserved task, not heldout generalization or native learned Fgameplay.

Old A/B/C are readonly TRAIN monitors, grouped into two27row batches by public
cue. Full source spans/row IDs and existing row decisions preserve separate
task accuracy/CE. No fake27step episodes or increased monitor envelope; the
affine model uses independent296value initial/current frames. No monitor
labels reach training. All probe updates stay on the private fork; source
native state, benchmark and serving paths remain untouched. Fork0→1 is an
explicit ownership event, distinct from native parameter/counter equality.

Source6fits108rows remains separate from paid upstream10fits180rows and new
probe2fits18rows. Collection, diagnostic forwards/reductions, storage/copies,
wall and RSS count. Failures preserve closed collection prefixes, paid native
work/readouts and one private endpoint; budgets and clocks cannot renew.
No final/controller-validation data, replay, return phase or promotion. The
existing16step return option remains available elsewhere with its own criteria.

## Observed fixed diagnostic

| Private probe updates | F correct | A correct | B correct | C_train correct |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 15/18 | 18/18 | 18/18 | 18/18 |
| 1 | 16/18 | 17/18 | 16/18 | 16/18 |
| 2 | 18/18 | 16/18 | 14/18 | 14/18 |

Fmean CE improves1.413812→1.165422 while old-task CE worsens. The learner can
fit the reserved task after C training, with visible interference on its private
fork. **The original after-C model is unchanged.** Ftrain/test transitions are
identical for each cue under the fixed collector; this is fit, not generalization.

The sole run used2probe fits/18rows,12diagnostic predictions/216rows,12metrics
and12row-diagnostic calls,36teacher decisions/physics steps for4closed sources,
0benchmark/deployed/replay/return updates. Whole body21.742672seconds,
parent22.279737seconds, sampled peak95,539,200bytes. Input arrays406,944bytes;
helper additionally owns85,536Fbytes and128,304monitor bytes. Native source6/108
stays fixed, private endpoint8/126; one fork event explicitly recorded.

32distinct focused checks qualify per platform; scoped lint/format/types and
source/procedure/result self-review pass, not independent acceptance. One check
requires local accepted sources and explicitly skips elsewhere; no cleanclone
or bundled-input qualification yet. No failed numeric attempts or case retry.

No CPC, efficiency, reliability, general plasticity, full recovery, nativeFcontrol
or release claim follows. Earlier failed/spent evidence stays intact.

[Protocol](../artifacts/runs/affine-fresh-task-probe-20261024/PROTOCOL.md),
[results/commands/costs](../artifacts/runs/affine-fresh-task-probe-20261024/RESULTS.md),
[issue113](https://github.com/OptimumAF/Circadian-Predictive-Coding/issues/113).

Extend through existing role-separated probe ports and finite monitors. Keep
diagnostics from choosing deployment or altering benchmark/final evaluation.

## Retained A3/A4 source recipe

The same Python composition also accepts the exact retained A3/A4 endpoint:

```python
from src.adapters.minigrid_affine_probe import run_affine_fresh_probe

# Supply the actual in-memory native4/54/two18-row-head snapshot and original
# ordered A3blue/A3red/A4blue/A4red TRAIN episodes. Freeze native costs first.
diagnostic = run_affine_fresh_probe(
    a3_a4_train_episodes, retained_state,
    retained_a4=True, check_boundary=original_source_guard,
)
```

`retained_a4` is an exact bool, defaultFalse. The original CLI and after-C
recipe remain unchanged. This option validates the same16train/2test fresh F
release before fixed firsttwo selection, then uses the existing probe runner,
one disposable fork, two updates/18rows and checkpoints0/1/2. It monitors all
four old TRAIN sources in two18-row batches; monitor and probe_test labels
never update the source or choose replay, promotion, tuning or probe dose.
The original source stays native4/54 and the disposable endpoint becomes6/72
with two9-row heads. Per-source row diagnostics expose interference as well
as fresh-task fit. The F task/seed is already inspected; this is not unseen-task
generalization or a new circadian comparison.

`paid_source_fits`/`paid_source_rows` are unavailable for caller-supplied
endpoints: native4/54 counters cannot account for discarded fits or preparation
that happened upstream. The new probe's own charges are explicit.
`preparation_seconds` measures input/copy/monitor preflight before the original
90-second probe owner starts; preflight RSS is unavailable. The caller must
bound the whole invocation, including this work. `whole_diagnostic_seconds`
includes it. Original budgets/clocks are retained, not renewed.

Retained mode protects protocol, collection records, data evidence and completed
probe reports by identity and value. Corruption or equal-container replacement
stops continuation and preserves paid prefix/failed status. Failed second fits
retain actual committed updates and unknown work; no automatic retry.

The existing `evaluate_context_linear_probe_checkpoint` also accepts explicit
`retained_a4=True` for readonly F games: before4/54/head18 or after6/72/head9.
It uses the same native loop, layouts, probe_test roles, seed17/max16 and rawK7
actions, with an original zero-update/replay150s512MiB readout owner. A typed
same-run endpoint can be validated through the existing snapshot decoder;
the report grants no training or resume authority. The default6/108→8/126
recipe remains strict. Readout collection/calls/physics/time/RSS must be budgeted
and accounted separately; the diagnostic itself does not run these games.

[Software qualification](reports/retained-a4-fresh-probe-20261026.md) covers the
same-run diagnostic and actual game admission with literal learning/scripted IO.
Native usefulness on this source recipe remains unexecuted. It grants no retry
of the spent controller-validation experiment.

The first [native retained-source diagnostic attempt](reports/retained-a4-fresh-probe-native-20261026.md)
failed before any learner or F probe ran. It collected two A4 teacher episodes,
then the experiment driver's source-budget allowlist rejected the original
configuration. A preceding data-path preflight also failed but did not prevent
launch; the report corrects the mistaken GO claim. This attempt is spent, with
no retained-source F learnability result and no automatic retry.

For future separately budgeted work, `scripts.retained_probe_preflight` provides
two small engineering checks. `require_source_budget(original_session)` checks
the exact source6/None/90/512MiB limits and separately requires completed replay
to be integer0. It reads no clock and changes no session state. A configured
None is not permission for additional replay.

`python -m scripts.retained_probe_preflight` checks metadata through the existing
packaged TRAIN resolver and launches nothing. `run_checked(argv, timeout_seconds=...)`
keeps that resource alive, requires its expected file metadata, then runs the
child with `check=True` in the same call. A failed preflight prevents spawning;
a failed child raises instead of becoming a successful GO. The CLI accepts an
explicit child after `--`. Matching size does not qualify dataset contents or
authorize an experiment. Existing scientific limits and original owners still
apply; this helper does not retry or repair the frozen217case.
[No-model qualification](reports/retained-probe-preflight-20261026.md).

## Compose after a fresh A4 run

The existing Python workflow can now keep its source handoff in memory:

```python
from scripts.minigrid_a4_adaptation import run_fresh_adaptation
from src.core.reference_controller import ReferencePolicy

# Budget native execution before calling; this invokes real training and games.
result = run_fresh_adaptation(
    reference_policy=ReferencePolicy("drift"), retained_probe=True,
)
```

`retained_probe` defaults to False and requires an exact bool. True accepts only
the original default drift reference in the fresh workflow. After all original
source game phases, it hands the already owned TRAIN episodes and last admitted
4-update/54-row native source directly to the existing isolated probe. No extra
source read, source refit, learner framework or CLI option is introduced. An
unexpected source endpoint refuses before fresh F collection; the wrapper
records a lost return as unknown work rather than zero work.

The original zero-update/replay 150-second owner spans the whole workflow,
including preparation and the probe. The probe keeps its original separate
90-second, two-update/18-row owner and one disposable fork. These limits are
cumulative, never a promise of additional allowance. `result["probe"]` contains
the separate fresh-task fit and old-TRAIN interference diagnostics;
`probe_work` counts outer attempts/returns and unknown unreturned work. Source
game outcomes remain separate. Failed or corrupted probe evidence makes the
overall result incomplete while preserving trusted paid prefixes and counts.

No F native games, return-phase recovery, promotion, tuning or final evaluation
are added. This opt-in has passed literal end-to-end software checks on Windows
and Linux; native usefulness is not established by those checks.
[Qualification and limitations](reports/a4-fresh-probe-workflow-20261026.md).

One native engineering use of this first-party option completed: source A3/A4
games changed2/0→2/2 goals while preserving the source; private F fit changed
14/18→18/18, with old TRAIN fit36/36→31/36. It cost six total fits/72rows and
140simulator steps. These are fixed known-task observations, not F native
success or recovery. Full protocol qualification remains incomplete because
whole-capture RSS was not recorded and the emitted projection included current
teacher records beyond its declared scalar scope. No retry; earlier217 remains
failed/spent. [Native result and limitations](reports/a4-fresh-probe-workflow-native-20261026.md).

## Explicit old-TRAIN return after F

The same fresh Python workflow also accepts `return_old=True` together with
`retained_probe=True` and the original default drift reference:

```python
result = run_fresh_adaptation(
    reference_policy=ReferencePolicy("drift"), retained_probe=True, return_old=True,
)
```

This is a separate fixed short recipe. After the two F fits, the same private
learner fits each original18-row old-TRAIN cue template once. A separate
two-update/36-row/90-second owner starts before F; the original whole-workflow
150-second owner still spans all preparation, source games, probe and return.
No budget or clock is renewed. F and old-task diagnostics at return0/1/2 measure
both relearning and interference; no score selects the dose or templates.

In `result["probe"]`, `probe_endpoint` retains the paid F6/72 state, while
`disposable_state` is the actual final8/108 state. `probe` keeps F charges and
readouts, including its nested `return_phase`; `return_budget` and
`return_old_task_readouts` report the separate return work/results. Lost return
fits preserve the complete paid F prefix and known partial work, fail overall
completion and grant no retry. Corrupted paid evidence, inputs or owners refuse
before further work. The original source and deployed/benchmark learners stay
unchanged; only old TRAIN templates update the private child. Final and
probe_test labels never train it.

This option has passed literal integration checks on Windows/Linux. Native
return learning has not been executed or qualified, and source game recovery
remains null. The existing original16F/16return recipe and omitted/False defaults
retain their original admission. No new CLI option or native allowance.
[Software qualification](reports/short-affine-return-20261026.md).

The first fixed native return observation recovered old TRAIN fit31/36→36/36
with two updates/36rows, while F fit fell18/18→14/18. The full fresh run cost
eight fits/108rows and140simulator steps; original source stayed unchanged.
This is relearning with interference, not joint retention or verified native
game recovery. No F/post-return old games were executed.
[Native result, costs and limits](reports/short-affine-return-native-20261026.md).

## Read task outcomes around private return

The fresh Python API accepts `native_games=True` with the explicit retained
probe and short return recipe:

```python
result = run_fresh_adaptation(
    reference_policy=ReferencePolicy("drift"),
    retained_probe=True, return_old=True, native_games=True,
)
```

Budget native execution before invoking this option. It reads six fixed games
after F learning and again after return learning: F blue/red `probe_test`, then
original A3 blue/red and A4 blue/red TRAIN, seed17/max16/raw K7 argmax. It uses
the same private run, existing evaluator and protected endpoint observer. The
exact endpoints are6/72 with two9-row heads and8/108 with two18-row heads;
the original matched8/180 rehearsal admission stays separate.

The original whole-workflow zero-update/replay150-second/512MiB owner spans the
extra reads. No owner, clock or learning dose is renewed. The separate F and
return owners remain unchanged. No labels or game outcomes select updates,
controller decisions, promotion or final evaluation. Deployed/benchmark learners
and the original source remain separate from this private learner.

`result["probe"]["native_games"]` contains at most two paid phase records.
`native_game_work` reports call attempts, returns and unknown unreturned work;
individual reports retain the existing evaluator's attempted/returned game
costs and closed partial prefix, including `not_run` slots. A failed call or
changed paid evidence fails overall completion, preserves trusted bounded
records and grants no retry. Complete zero-goal outcomes are valid results.
Capture callbacks are checked before the final trusted records are detached.

This option passed51 distinct literal checks on Windows and Linux, including
the actual shared game loop and full fresh workflow. These checks establish
software behavior only. No native F/post-return game case has been executed
under this increment; the earlier fit recovery remains a separate observation.
No new CLI or experiment allowance is introduced.
[Software evidence and limitations](reports/short-affine-return-games-20261026.md).

One fixed native use now answers the task-outcome question: after F learning,
F2/2 and old A3/A4 0/4 goals; after two old-TRAIN return fits, F0/2 and old4/4.
All12private games completed and closed. The entire fresh run cost8fits108rows,
290simulator steps and51.5844seconds, with original source/actor/budgets/RNG
preserved. This demonstrates acquisition and old-task recovery with mutual
forgetting, not joint retention, generalization or a CPC advantage. No retry or
score-selected dose. [Native outcomes, costs and limits](reports/short-affine-return-games-native-20261026.md).

## Equal-row private rehearsal for retained A3/A4

The existing comparison API also accepts an explicit retained-A3/A4 recipe:

```python
from src.adapters.minigrid_affine_rehearsal import compare_private_rehearsal

result = compare_private_rehearsal(
    old_train_episodes, retained_state, selected_F_episodes, retained_a4=True,
    check_boundary=check_original_owners,
)
```

This is an in-memory API: the caller owns the actual4-update/54-row source with
two18-row cue heads, ordered original A3/A4 TRAIN episodes and the four already
qualified selected F sources (blue/red probe_train, then blue/red probe_test).
The caller must preserve the original complete-release qualification; the
comparison does not collect teachers or load a historical report. This option
also has the ordinary fresh Python workflow entry point described below.

Both private arms start from that same source and fit twice. `current_only`
uses three copies of the nine F rows per cue; `joint_A3_A4_F` uses18old-TRAIN
rows plus nine F rows per cue. Each arm adds54row exposures and ends at6updates/
108rows with two27-row heads. Extra F exposure36 and attempted/committed old
TRAIN reuse36 are separate costs. Equal rows and fit counts do not establish
equal computation.

The existing common90-second lifetime, two preallocated arm allowances, one
fork per arm, source/input guards and diagnostics remain. No budget is renewed.
Paid attempts precede fitting; lost fits retain fees and known native commits,
and a failed/corrupted first arm stops the next. Neither test nor final labels
enter either fit matrix. The original ABC36-row recipe and defaults remain.

57 distinct literal checks passed on Windows/Linux, with scoped statics and
self-review. These establish software behavior; joint retention and native game
success remain untested for this recipe. No new native allowance or CLI option.
[Evidence and next integration action](reports/retained-a4-rehearsal-20261026.md).

## Run rehearsal after an ordinary fresh source

```python
result = run_fresh_adaptation(
    reference_policy=ReferencePolicy("drift"), retained_rehearsal=True,
)
```

`retained_rehearsal` defaults to False and requires an exact bool. It accepts
the original default drift reference and is exclusive with `retained_probe`,
`return_old` and the controller recipes. The explicit `native_games=True`
combination described below is now supported. Describe-before-IO and
the original defaults remain available; this adds no CLI option.

After the source games, the workflow passes its already owned4/54 source and
original A3/A4 TRAIN episodes to the existing F collector. It validates the full
16training/2test release before selecting the fixed first2training+2test sources
for both private rehearsal arms. There is no additional F warmup or source fit,
historical report read or new learner. The original whole-workflow zero-update/
replay150-second/512MiB owner spans preparation, collection and comparison; the
existing study/two arm owners keep their shared90-second lifetime.

`result["rehearsal"]` holds bounded collection/source evidence, the two-arm
`comparison`, separate `comparison_work` attempt/return/unknown counts, and
preparation/validation array bytes. `rehearsal_work` records the outer call
separately. Partial failures and lost returns retain paid evidence/known charges
without a retry. Paid reports and costs are rechecked after the original resource
clock/source callbacks before completion and capture. No private scores select
the deployed learner, controller, promotion or final evaluation.

35 distinct literal checks pass on Windows/Linux with scoped statics and
self-review. This delivers the fresh-source comparison path; native joint
retention remains unverified. [Evidence and next native question](reports/a4-rehearsal-workflow-20261026.md).

## Read the retained rehearsal endpoints in native games

```python
result = run_fresh_adaptation(
    reference_policy=ReferencePolicy("drift"),
    retained_rehearsal=True, native_games=True,
)
```

Budget native execution before calling this option. After **both** private arms
finish fitting, it reads six fixed games per arm: F blue/red `probe_test`, then
original A3 blue/red and A4 blue/red TRAIN, seed17/max16/raw K7 argmax. Each
endpoint must be exactly6updates/108rows with two27-row heads. The existing
matched ABC and short-return admissions remain distinct. Current-run paid
endpoints use existing typed snapshot validation; there is no historical loader.

The original whole150-second/512MiB zero-update/replay owner spans all work.
Completed comparison/arm clocks and allowances are preserved. Game outcomes
do not affect either arm's training, deployment, promotion or final evaluation.
`rehearsal.native_games` retains at most two paid arm records;
`rehearsal.native_game_work` separates attempts, returns and unknown unreturned
work. A failure stops the next read, retaining closed partials and `not_run`
slots. Complete zero-success outcomes are valid.

The first native comparison **failed and spent its allowance**: both arms
fitted, but an oversized episode identifier stopped the first private game
before reset. The identifier repair passes focused Windows/Linux regressions
using the real64-byte validation; the native comparison was **not retried**.
The pre-F source baseline achieved0/2goals. Both arms fitted all18F diagnostic
rows; old TRAIN fit was31/36current-only versus36/36joint. These are diagnostic
fits, not post-training task success or demonstrated joint retention.
[Failure, repair, costs and remaining question](reports/retained-rehearsal-native-20261026.md).

## Run retained rehearsal from the normal command

The existing affine command now exposes the same fresh Python recipe:

```powershell
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --context-linear-retained-rehearsal diagnostics --describe
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --context-linear-retained-rehearsal games --describe
```

Both descriptions perform no collection, learning, games or source-file reads.
`diagnostics` fits both private arms and reports their existing F/old-TRAIN
diagnostics after the ordinary source workflow, which includes source games.
`games` additionally reads the fixed six native games per private arm after
**both** arms finish fitting. The inherited source summary describes A3/A4
source games; use `rehearsal.comparison` and `rehearsal.native_games` for private
outcomes. Missing/failed private games are unknown, never zero-success results.

For a separately budgeted execution, replace `--describe` with
`--output new-rehearsal.json`. Execution needs the existing optional MiniGrid
dependencies. Both modes require `--example-affine-a4`, use the original fixed
drift reference, and reject source paths, controller/reference combinations and
fit overrides. No new CLI seed, horizon or learning-dose selection is exposed.

Budget the complete fresh run before executing: original150-second/512MiB
whole-workflow owner, source owner plus preallocated common90-second comparison
and arm owners; private work adds four fits/108rows, including36extraF exposures
in the current-only control and36old-TRAIN replay exposures in the joint arm.
These are private candidate diagnostics, with no deployment or promotion.
The command does not add the experimental pre-F baseline from227's producer.

Output is complete finite compact JSON, UTF8 with LF, at most8MiB, exclusively
created. Existing files are refused before work; a file-creation race preserves
the competing file and never retries paid work. Complete reports exit0, including
complete zero-goal results; failed reports retain known costs/partials and exit2.
Serialization failures raise without retrying. The same bounded writer remains
in use for the existing controller command.

38focused literal checks pass per platform, including real command/workflow
composition and a failed second private readout. No native rehearsal command
was executed for this CLI increment;227's native comparison remains failed and
spent, with joint task retention unverified. [Command qualification](reports/retained-rehearsal-cli-20261026.md).
