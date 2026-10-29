# A3 → A4 → A3 affine learning

The first-party workflow collects its own A4 TRAIN episodes and needs no saved
experiment file:

```powershell
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --describe
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --output fresh-a4.json
```

The installed entry point is `circadian-affine-game` with the same arguments.
The current checkout additionally supports fixed reference choices in this
fresh workflow; the existing command above keeps its six-update default:

```powershell
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --context-linear-reference periodic --ready-feedback --describe
python -m scripts.run_minigrid_linear_fit_control --example-affine-a4 --context-linear-reference periodic --ready-feedback --output fresh-reference.json
```

Choices are `no_learning`, `periodic`, `budget_aware` and `drift`, with their
original defaults. This explicit route requires `--ready-feedback` and uses
the existing arrived TRAIN bank inherently; do not add `--reference-retain-arrived`,
prequential, saved-source or other recipe/fit options. Describe performs no IO.
Execution needs a separately frozen local allowance; describe is safe to inspect.
The [software qualification](reports/a4-reference-workflow-20261026.md) covers
composition and isolation. The subsequent
[single native public-command trace](reports/a4-reference-cli-native-20261026.md)
acquired A4 and retained A3 with four fits/54rows, leaving two packets pending;
the return readout reused its unchanged complete state. This known-TRAIN result
does not establish recovery, generalization, controller/CPC advantage or an
installed-wheel qualification. Further executions need their own local budget.

Reference learning completes its seven-opportunity window before games, leaving
unconsumed packets pending. `training.stream.window_complete` and
`training.stream.training_complete` have different meanings. Each game phase
records logical `tick` 2/4/6 separately from actual `updates`; no-learning and
other controls may have zero, odd or repeated update counts. A complete identical
full checkpoint can reuse its earlier completed readout within this invocation;
`readout_reused_from_tick` identifies the original observation. Reuse adds no new
game or independent return sample. At most three phases/readout-cache entries
are retained. The summary reports acquisition/retention/return separately;
reference recovery is null for cues without observed loss after acquisition.

Only actual complete paid prefixes pass to the existing A3/A4 TRAIN evaluator.
The explicit A4 reference stream returns bounded `unscored_checkpoints` for
0..completed updates, including unfitted heads. Ordinary A/B reference exports
remain empty. Current/prior/native row exposures, planned storage, diagnostics
and game work stay separate; no scoring influences scheduling. Failed collection,
lost receipts, changed budgets/policy or corrupted/replaced prior evidence stop
later work and retain charges/partial output with exit2. Original lifetime
150s/512MiB readout and six-update/90s TRAIN limits apply, without renewal.

Use the repository's optional MiniGrid environment. One invocation collects two
fresh blue/red A4 TRAIN teacher episodes (at most32steps), requires their exact
nine-row contract, performs six fits/90exposures, and reads all three stages
through the existing game loop (at most192additional steps). Collection,
training and all games share the original150s/512MiB cumulative readout owner;
the stream retains its separate90s/six-update budget. Source provenance records
this run's native collection and costs, with no historical file hash. Two paid
collection records and three phase records are bounded and guarded against
changes/replacement before more work. Failed collection stops before fitting;
partial records remain with exit2. Describe performs no IO; output is exclusive
and8MiB bounded. The [installed-package qualification](reports/a4-installed-package-20261025.md)
checks the current wheel's command, resource, zero-model dispatch and early
runtime refusal; installed native games and deployment remain unqualified.
The [fresh-workflow qualification](reports/a4-fresh-workflow-20261025.md)
records an initial tuple/list handoff failure, its repair, and the explicit
corrected native run: A4 0/2→2/2 while A3 stayed2/2, both2/2 after return. Both
attempts' costs remain charged; see issue #173 and its protocols before reruns.

Replay a saved fresh-A4 result without training or simulation:

```powershell
python -m scripts.build_key_door_inspector --source-artifact fresh-a4.json --output a4-learning.html
```

The existing inspector now exports all three stages and twelve A3/A4 game
slots, TRAIN curves, original native outcome summary, causal receipts and
separate collection/replay/game costs. Initial failures remain visible; missing
phases stay unknown. [Export qualification and limitations](reports/a4-affine-inspector-20261025.md)
record the accepted saved run's twelve trajectories and134public-view frames.
HTML/payload/JavaScript checks passed; interactive browser and visual checks
remain pending under the existing local-file URL policy gate. Open the standalone
file to inspect it; export does not restore weights or grant training authority.

The existing affine CLI has an explicit adaptation mode:

```powershell
python -m scripts.run_minigrid_linear_fit_control --context-linear-a4-adaptation --describe
python -m scripts.run_minigrid_linear_fit_control --context-linear-a4-adaptation --source artifacts/runs/a4-train-collection-20261025/actual.json --output new-adaptation.json
```

Freeze a separate local allowance before any invocation that fits or plays. The
qualified171 case is one nonrenewable study, not permission to rerun a source or
select better outcomes. Describe performs no IO; output is exclusive and8MiB
bounded, failure/partial results are retained with exit2. Fit options/other modes
cannot combine. This is an advanced checkout mode, not a newly qualified wheel.

Why: the ordinary actor's changed-layout blue failure called for fresh TRAIN
coverage. The new mode uses the original bundled A3 pair and the exact qualified
fresh170 A4 TRAIN pair. A4 episodes retain their role from birth; no old
development/probe_test data enters fitting, replay or promotion. Source bytes
are read once, size/SHA pinned, native148 records checked against saved296 history
and original positive closed TRAIN evidence. Source collection costs/budgets/
clocks are provenance, not renewed training authority. Detached as-read data is
authoritative; this is not a live file watcher. No old127 fitted model is used.

The original delayed stream owns six updates: A3blue/red → A4blue/red → A3blue/red.
Each cue jointly fits its previous arrived TRAIN episode and current one. There
are90row exposures:54arrivals including18explicit old-task return rows, plus36
joint replay exposures, over36distinct rows. All labels arrive after delay1.
No optimizer, model-random initialization, additional catalogue or runner was
added. The new stream flag is `a4_training=True, retain_arrived=True`; other
catalogue modes are mutually exclusive. Original A/B/ABC/G defaults are unchanged.

After ALL fixed fits finish, complete actual parameters at updates2/4/6 pass to
the original game evaluator. This is full new-state transfer, not reconstruction
from scalar diagnostics or old experiment matrices. Every stage plays A3blue/red
then A4blue/red, TRAIN, seed17/max16/all7actions/raw firstargmax. A single original
150s/512MiB zero-update/replay budget spans source admission, fitting and all
three readouts; the inner six-update stream also retains its original90s budget.
The current actor is read-only during games. These are retrospective causal
checkpoint observations, not live deployment/promotion.

A3 and A4 retain contextA; branch-specific episode IDs distinguish the native
games only in this mode. Source/scorer/owned-state/clock/budget/effect/cleanup
guards remain. The original source/TRAIN evidence and at most three completed
phase records are checked for edits and equal-value container replacement before
more work. Corrupted/partial evidence stops later phases and keeps paid records.

`summary` reports new-A4 gain after its training, old-A3 retention before return,
and old-task return gain. A recovery flag requires an initially successful old
cue to fail after A4 and succeed after A3 returns. Retention alone is not recovery.
Incomplete phases yield unknown aggregate metrics; completed negative episodes
are valid observations. Goals/returns/steps and costs stay separate from fitting
diagnostics. No generalization/reliability/matchedcompute/CPC or release claim
follows from one fixed TRAIN scenario.

Changed implementation/tests:

```text
scripts/run_minigrid_linear_fit_control.py  opt-in descriptor/stream/CLI dispatch
scripts/minigrid_a4_adaptation.py           source admission/composition/summary
src/adapters/minigrid_policy_evaluation.py  exact new recipe/plan over old loop
tests/test_a4_adaptation.py                actual CLI + literal math/scripted IO
```

Focused validation uses `pytest -q tests/test_a4_adaptation.py`, plus the three
unchanged default binding cases from `tests/test_context_linear_causal_games.py`.
Scoped Ruff check/format and mypy `--follow-imports=silent` cover changed files.
Observe171's already-consumed fixture/native limits before running numerical
tests or games. Source/procedure/result review is self-review, not independent.
No new dependency or environment variable; existing optional MiniGrid setup and
single-thread native protocol apply. Extend through the existing stream and
evaluator with explicit roles/receipts/costs, rather than adding another framework.

The [qualified native result](reports/a4-adaptation-20261025.md) was A3 2/2 and
A4 0/2 before A4 training, then both2/2 after A4 training and after A3 returned.
Six fits/90exposures/122game steps took10.51s body with85.37MB sampledRSS. A3
was retained throughout; recovery was therefore false. The
[same-data current-only control](reports/a4-current-only-20261025.md) learned A4
but forgot A3, then relearned A3 using18returning TRAIN rows and forgot A4. It
used54fit exposures versus90with joint replay. Joint replay retained both known
tasks in this fixed pair; this is not matched-compute, broad generalization,
live deployment, CPC advantage or release readiness.

The opt-in control uses the same qualified source:

```powershell
python -m scripts.run_minigrid_linear_fit_control --context-linear-a4-current-only --describe
python -m scripts.run_minigrid_linear_fit_control --context-linear-a4-current-only --source artifacts/runs/a4-train-collection-20261025/actual.json --output new-control.json
```

Observe172's consumed study allowance before fits/games. The stream flag is
`a4_current_only=True`, retention/joint false; composition keyword
`current_only=True`. Source/role/seed/metric/clock/cleanup/output contracts remain.
Recovery here costs supervised old-task retraining. Both research modes retain
their strict historical source requirements; the first-party fresh workflow
above uses the same joint learning/game sequence with its own collection.

For an exposure control, use
`run_context_linear_stream(episodes, a4_current_repeated=True)` with the same four
ordered typed A3/A4 TRAIN source packets. Its pure descriptor is
`describe_context_linear_a4_current_repeated()`. Six ordinary wakes use current
packet multiplicities `(1,1,2,2,2,2)`:6fits90row exposures,54arrival+36repeated
current/0prior-packet replay/0bank. Per-copy spans and unique transport IDs preserve
provenance without inventing complete trajectories. Source/state/previous
receipts are checked before more work; failed/uncertain prefixes retain original
charges and latest verified rows. Existing modes are unchanged.

This matches the joint recipe's fit/batch row shapes, not wall time or FLOPs.
[Software evidence and limitations](reports/a4-current-repeated-20261026.md)
cover literal ports only. The [command composition is now software-qualified](reports/a4-current-repeated-cli-20261026.md):

```powershell
python -m scripts.run_minigrid_linear_fit_control --context-linear-a4-current-repeated --describe
python -m scripts.run_minigrid_linear_fit_control --context-linear-a4-current-repeated --source artifacts/runs/a4-train-collection-20261025/actual.json --output new-repeated-control.json
```

Composition keyword is `current_repeated=True`; it is distinct from `current_only`.
The existing evaluator uses `a4_current_repeated=True` and an explicit repeated-
current role,6fit90row geometry and the same three fixed A3/A4 TRAIN phases. Before
games, complete protocol/costs/verified receipts are checked; changed/uncertain
prefixes retain original charges and refuse more games. Original cumulative
readout budget and acquisition/retention/learning-recovery meanings remain.
No native comparison was executed for this new mode. The second command requires
a fresh frozen protocol/allowance before numerical work; reuse valid old controls.
Original171/172unequal-work results remain unchanged, with no CPC/efficiency claim.

The [sole native repeated-current observation](reports/a4-current-repeated-native-20261026.md)
has now completed: A3/A4 goals were2/0→0/2→2/0 over2cues/task, matching the
historical54row current-only goal endpoints despite6fits90exposures. Repeating
current data did not preserve A3 after A4; A3 relearning on return cost36fit-row
exposures and lostA4. All12games closed,150steps,8.512sbody/86.29MBsampledRSS;
both RNG witnesses unchanged. Historical joint90retained both afterA4/return.
Historical package/source-byte receipts limit this to a descriptive exploratory
comparison, not exact paired replication/matchedwall/FLOPs/CPC/generalization.
The one new native allowance is SPENT; do not rerun or extend it automatically.
