# A3 → A4 → A3 affine learning

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
was retained throughout; recovery was therefore false. Joint replay necessity
and matched control/cost tradeoffs remain untested, as do broad generalization,
live deployment, CPC advantage and release readiness.
