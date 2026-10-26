# Explicit live TRAIN recipes and marked label corruption

The existing native live-serving modes can use a declared TRAIN packet recipe:

```console
python -m scripts.run_bp_ticket_serving --source artifacts/runs/bp-ticket-fit-qualification-20261023/actual.json --publish-during-serving --live-training-recipe live_new_retained_noise --output noisy-live.json
```

Use a fresh output path. `--live-training-recipe` requires either
`--serving-during-updates` or `--publish-during-serving`. Other modes retain their
defaults. Existing comparative `stream_recipe` still requires deferred evaluation;
the live input cannot bypass that contract or enroll final/probe data.

## Data contract

`LiveTrainingRecipe` composes the existing immutable `TicketStreamRecipe` and an
explicit retained-label corruption flag. It leaves the original recipe schema,
WITHHELD_RECIPES and builder semantics intact. `build_live_stream` reuses
`build_stream`: each owned12row packet is selected TRAIN6rows plus retained old
TRAIN6rows,7features/6classes. Original roles, bounded unique IDs and private array
copies remain; final/probe/guard roles cannot be relabelled TRAIN.

| CLI recipe | Packet and deliberate error |
| --- | --- |
| `noisy_return` | Existing old+old, delay2; original row1 invoice inspect->defer |
| `live_new_retained_noise` | New+old, delay2; retained row7 invoice inspect->defer |

The new mixed recipe uses eight packets. Only the copied retained old target at
zero-based row7 changes. The selected new six rows, all features and original
teacher buffers remain clean. The copied retained target must originally be
inspect; double corruption is refused. One false label per12rows gives eight
explicitly false exposures over eight updates. This is an experimental poisoning
fixture, **not verified supervision or a useful augmentation claim**.

The Python API accepts `live_training_recipe=LiveTrainingRecipe(base,
retained_invoice_noise=True)` with an explicit `LiveServingPlan` and the original
drift mode. It validates the finite plan/recipe/delay before source/model work.
Original recipe values are copied/pinned; main and worker source checks refuse
later changes. The actual packet composition/delay is recorded in
`live_training_recipe` and `training_stream`, alongside exact `training_corruption`
metadata. `training_context` continues to identify the original dispatch argument.

Why this: old row1 noise is safe only for old-only packets. A separate live wrapper
can mark retained-row noise while preserving the frozen comparative schema and
evaluation isolation. No new runner, native rule, guard policy or controller fit.

## Execution and isolation

Existing delayed labels, shared serving gate, one worker, finite choices, original
budgets, guarded drain and rollback remain. Only causally arrived TRAIN labels
feed the unchanged drift0.1 controller. Corrupted targets do not change the clean
teacher used by guards or the workflow dispatcher's task truth. Live outcomes,
final labels and probes never enter training, promotion or policy selection.

Repeated old anchors are ordinary TRAIN exposures included in the row ledger;
there is no extra replay-executor call or concealed exposure allowance. Every
failure/refusal stays paid. A rejected candidate does not trigger a later guard
retry, budget renewal or actor replacement.

## Observed negative result

The sole frozen mixed-noise trace committed8updates/96rows:48new and48retained
exposures, including8wrong retained labels. Delay2 feedback reported2/12mistakes;
the original drift controller consumed its finite eight-update budget. At tick11
the unchanged guard rejected the candidate for `new_task_utility` and `new_task_gain`:

- New guard accuracy stayed5/6; old guard accuracy stayed6/6.
- Candidate completed invoice, parcel and newly learned legacy, but failed the
  previously working password workflow.
- Actor stayed generation0/qualified-96/full96state. All six live password
  episodes kept working; all six live legacy episodes remained wrong workflows.
- Live15/24successes; three stale invoice episodes recovered, three revoked parcel
  commits were refused, and no permission/revision violation was applied.

This demonstrates one useful refusal under a declared corrupted stream. It does
not prove general poisoning robustness, nor isolate corruption from mixed-packet
learning effects. No matched clean control was run in this increment. Prior clean
six-row traces have different exposure/weighting and are not that control. No
guard, threshold, dose, seed or metric was changed to force acceptance.

Candidate104updates/672rows; source and caller RNG unchanged. Physical8SGD/
96rows/8backwards,424head constructor entries16536constructor matrix-slot upper,
128forward invocations979snapshots131184logical elements,108total typed attempts/
132local dispatches,0PC/newfit/finalTRAIN/probe/replay-executor. Internal7.0522s,
meter8.5490s,outer13.0006s,sampled peak334131200B. Constructor/element counters
are partial/conservative, RSS current-process sampled, and instrumented timings
are not a latency SLO or efficiency comparison.

## Checks and handoff

```console
python -m pytest -q tests/test_live_training_recipes.py tests/test_ticket_controller_streams.py
python -m ruff check scripts/run_bp_ticket_serving.py scripts/ticket_controller_streams.py tests/test_live_training_recipes.py
python -m ruff format --check scripts/run_bp_ticket_serving.py scripts/ticket_controller_streams.py tests/test_live_training_recipes.py
python -m mypy --no-incremental --explicit-package-bases --follow-imports=silent scripts/run_bp_ticket_serving.py scripts/ticket_controller_streams.py tests/test_live_training_recipes.py
```

133selected combined checks/platform and scoped statics passed, including native
delayed packets, clean guard labels, source-mutation refusal and default/deferred
regressions. Seven unchanged reference-cohort/failure cases were explicitly reused.
Single-agent source/procedure/result self-review, not independent acceptance.
Exact commands/costs/protocol/evidence:
`artifacts/runs/live-training-recipes-20261023/RESULTS.md`.

## Subsequent clean matched-dose diagnosis

Issue60 subsequently ran one private clean control through the existing deferred
stream, reusing the noisy result. The same source/full optimizer, seed17, settings,
eight updates/96rows, mixed packet features and row order were used. Only retained
row7 changed back to its correct inspect target. All four ordinary workflows
succeeded: legacy learned and password retained; invoice/parcel also retained.
The original noisy candidate's password failure remains preserved.

This supports a harmful effect from that label corruption in this fixed
diagnostic. It is one source/data/seed, with repeated development templates.
The clean control used an explicit fixed dose via ContextualChoice(wake1,wake1),
delay2/horizon10, without live contention/guard/promotion. Production drift0.1
remains unchanged (clean1/12feedback would not trigger it). Native BP has no tick
input; optimizer data/order/dose match, total computation/scheduling/latency do
not. No controller/robustness/CPC/generalization or promotion claim follows.

Readouts open only after all updates; no benchmark/deployed/probe/final mutation.
Physical8SGD96rows8backwards/32head constructors1248matrix-slot upper28forwards/
85snapshots11856logical/12typed48localdispatch; internal6.1386s,meter7.6665s,
outer13.2254s,sampled334778368B. Full protocol, preserved engineering failures,
30pure checks/platform and final case statics, all costs and next action:
`artifacts/runs/clean-mixed-control-20261023/RESULTS.md`.

The label question is now answered within this scope. Next reassess the existing
MiniGrid TRAIN-fit-to-live acquisition gap through its existing readouts before
another ticket knob/trace. Full R4.2, R7.7 long-duration/broader utility, R7.6
validated augmentation, R7.1 gate/sleep/combinations and scientific/release
requirements remain open. Single-agent self-review, not independent acceptance.
