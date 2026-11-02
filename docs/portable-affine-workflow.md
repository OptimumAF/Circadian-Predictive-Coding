# Fresh affine MiniGrid train and play

From a checkout with the optional MiniGrid requirements installed:

```powershell
python -m scripts.run_minigrid_linear_fit_control --example-affine-play --describe
python -m scripts.run_minigrid_linear_fit_control --example-affine-play --output fresh-affine.json
```

The installed wheel exposes the same arguments as `circadian-affine-game`.
The distinct `--example-affine-recurring` mode runs the existing original
A/B/C/G/A recipe and known G/A games; see [ordinary adaptation](ordinary-fourth-task.md).
Its `--describe` is metadata-only and the normal mode below remains unchanged.
An ordinary installation uses `pip install "circadian-predictive-coding[minigrid]"`
from your chosen local wheel or package source. NumPy is the only base dependency;
native gameplay requires the optional pinned MiniGrid/Gymnasium dependencies.
No Torch, GPU, research artifact archive or saved trained model is required.
Output must be new; failed and incomplete reports are retained and exit nonzero.
`--describe` performs no training, gameplay or source-data lookup.

The existing private adaptation command also supports one isolated incremental
F child from a fresh bundled A/B/C TRAIN endpoint:

```powershell
python -m scripts.run_minigrid_linear_fit_control --example-affine-adaptation --probe-update minimum_change --describe
python -m scripts.run_minigrid_linear_fit_control --example-affine-adaptation --probe-update minimum_change --output isolated-f.json
```

This explicit option uses six original source fits (108 row exposures), then
the existing disposable probe's two F updates (18 rows). The source stays at
6/108; the child ends at 8/126. The original four closed teacher collections
release 16 TRAIN and two test packets; only the fixed first two TRAIN packets
are fitted. Old A/B/C TRAIN diagnostics remain read-only. All of these costs
are separate from gameplay and the source catalogue's replay exposures.

The child is then frozen for eight target-free games: F blue/red with the
`probe_test` role, followed by every old A/B/C_train blue/red task with its
original `train` role. The fixed roster includes B and C because earlier F
fits interfered with their decisions. There is no final-test readout, probe
promotion, deployment update, or learning during games. These are known tasks;
the F train/test trajectories overlap. The original source/probe/readout
lifetime budgets remain separate, under the existing 180-second workflow
supervision and 170-second work deadline (512 MiB each). A failed or changed
paid source/probe prevents games; later failure retains preceding costs and
the known game prefix. No saved scientific source file is required.

Omitting `--probe-update` preserves the original two-arm private rehearsal
comparison. The new option requires private adaptation and cannot be combined
with ordinary TRAIN update selection, recurring training, or A4 adaptation.
The evaluator's explicit `include_old_tasks=True` option is limited to the
ordinary after-probe endpoint; it does not admit arbitrary states or tasks.
[Software evidence](reports/minigrid-isolated-affine-workflow-20261031.md)
covers scripted composition and isolation. One separately frozen
[native qualification](reports/minigrid-isolated-affine-native-20261031.md)
completed the fresh command: F2/2 and old A/B/C4/6 games succeeded. B-red and
C-red timed out at16 steps with zero return; all eight environments closed.
The case used six source solves/108 row exposures plus two child solves/18 rows,
with source and independent budgets conserved. Old-task retention is incomplete.
Reuse earlier qualified observations only within their source and protocol
assumptions; no native-before/recovery, added efficiency, generalization or
circadian benefit claim follows. The failed original results remain preserved.

The command starts with two unfitted seven-action affine heads. Six bundled
TRAIN demonstrations arrive in fixed A blue/red, B blue/red, C_train blue/red
order with label delay one. The existing arrived-cue catalogue retains two
previous complete episodes per public initial cue. Six joint least-squares fits
expose 108 rows: 54 arrivals and 54 additional replay exposures. The bounded
bank holds at most 36 rows; input templates, full model states and Python
overhead cost additional memory. All fits finish before fixed TRAIN diagnostic
readouts at updates 0/2/4/6, so diagnostic labels cannot select more work.

Only a complete observed six-receipt stream hands its actual in-memory endpoint
to the existing native evaluator. Four target-free games execute known B and
C_train blue/red contexts, seed 17, at most 16 decisions each. The evaluator
uses all seven action probabilities and first argmax. It makes no update or
replay call. Training and readout have distinct lifetime budgets (six updates,
90 seconds; zero updates/replay, 150 seconds; each 512 MiB sampled process RSS).
The readout stops new work at 140 seconds to reserve cleanup time. These are
new first-party execution budgets, never renewed historical study authority.

The JSON includes training receipts, checkpoints, scores, explicit row exposure
and storage accounting, source/actor invariance, budgets and actual game traces.
Training failure prevents gameplay. The preceding paid training report and
readout authority are guarded during games; changed values or replaced nested
containers are refused. A failed game retains the evaluator's known prefix.
Earlier scientific readback commands retain their original source qualification.

Bundled examples contain only project-generated synthetic TRAIN trajectories
under MIT; see [provenance](../examples/minigrid_affine_train/README.md).
Known-task execution is not unseen-layout generalization, reliable adaptation,
a recurrent RL result, matched computation or a circadian advantage. Existing
negative and incomplete studies remain authoritative within their scopes.

Extension: change task schedules through the existing stream/composer with
explicit role, receipt, native-counter and cost checks. Freeze a new small
scientific protocol before changing datasets, seeds, metrics or work doses.

Fresh ordinary and recurring workflows also accept the explicit API option
`train_update="minimum_change"`, or CLI `--train-update minimum_change`. This
selects the existing incremental learner operation through all owned TRAIN
forks; omitted options preserve the cold-fit recipe. Original source/readout
budgets, collection and episode protocols remain. Private/A4/reference profiles
do not accept this new option. `--describe` includes the selected objective scope.
[Software integration checks](reports/minigrid-minimum-change-stream-20261031.md)
use scripted fits and games; native navigation benefit is unqualified. The
historical installed run below predates this option and does not validate it.

One separately budgeted [source-checkout recurring qualification](reports/minigrid-minimum-change-native-20261031.md)
with this option completed 10 solves/252 row exposures and four successful
nine-step G/A games. Its TRAIN correctness curve matches the earlier cold
recipe, and confidence regressions remain. This does not qualify the option in
an installed wheel or establish added benefit, native recovery or generalization.

Executed qualification: one installed-only Windows run completed all six fits
and four known B/C games, each nine steps, return 0.49375, terminated and closed.
Body 57.2974 seconds, peak sampled RSS 83,611,648 bytes. This reused existing
NumPy/MiniGrid dependencies; no clean dependency installation or release claim.
Source/procedure/result self-review, not independent acceptance. Exact commands,
costs, failures and skipped checks: [evidence](../artifacts/runs/portable-affine-workflow-20261024/RESULTS.md).
