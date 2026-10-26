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

Executed qualification: one installed-only Windows run completed all six fits
and four known B/C games, each nine steps, return 0.49375, terminated and closed.
Body 57.2974 seconds, peak sampled RSS 83,611,648 bytes. This reused existing
NumPy/MiniGrid dependencies; no clean dependency installation or release claim.
Source/procedure/result self-review, not independent acceptance. Exact commands,
costs, failures and skipped checks: [evidence](../artifacts/runs/portable-affine-workflow-20261024/RESULTS.md).
