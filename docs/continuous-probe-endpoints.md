# Native scoring at disposable continuous-probe endpoints

## Accepted five-scene retention/transfer case

At the rehearsed48/2896 source, one existing private SGD1x18 probe leaves test24
unsuccessful before/after50controls. Subsequent frozen TRAIN17/19/20 scenes remain
successful17/19/16controls, matching cached pre-update results. All signed native
contrasts are zero; all21 imitation contrasts and earlier source32 failures stay.
Independent actual review accepts strict common-probe equality, roles/initials,
paid prefixes and costs.152controls760knownphysics/196pred1244rows in one41.04s
run; no new production API. This supports these fixed retention cases only.
[Protocol, exact commands, outcomes and limitations](../artifacts/runs/rehearsed-probe-native-transfer-20261021/RESULTS.md).

The existing `run_continuous_probe` accepts an optional synchronous
`endpoint_observer(readout, phase, guard)`. It borrows the existing private
readout at `before_probe` and `after_probe`. It receives no episodes or labels;
updates remain limited to the original reserved `probe_train` plan. The returned
report contains scores, receipts and work, never a learner. Do not retain,
train, restore, export, deploy or promote the borrowed readout.

For the existing Panda raw42/executedTS12 setup, a callback can call the ordinary
frozen rollout with explicit `probe_test_seed=24`:

```python
def observe(readout, phase, guard):
    def score():
        result = run_frozen_rollout(
            output_directory / f"{phase}.json",
            learner=readout,
            probe_test_seed=24,
            action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
        )
        physical_results[phase] = result.report
        if not result.report["complete"]:
            raise RuntimeError("physical endpoint incomplete")
    guard(score)

report = run_continuous_probe(
    checkpoint, fixed_probe_plan,
    training_readouts=original_train_readouts,
    endpoint_observer=observe,
)
```

Import the probe from `src.adapters.continuous_probe`, frozen rollout from
`scripts.maniskill_continuous_smoke`, and recipe from
`src.core.pick_cube_taskspace_history`. The caller supplies a fresh exclusive
output directory, a prospectively budgeted callback and separately retained
physical results/costs. A complete endpoint may have zero task success. No score
can alter the fixed training plan, dose or learner selection. Reserved test24
is unavailable to training, promotion, replay and final evaluation.

## Boundaries and failure behavior

The before callback follows initial imitation scoring. The after callback runs
only after the original native updates and receipts validate. Both callbacks
must return `None`; read ports passed to `guard` also return `None`. A guard
refusal remains latched even if caught. An exception/refusal before training
stops updates; an after failure retains already-paid updates and valid imitation
scores while marking the overall result incomplete. Original report values and
container identities, native source/runtime/readout owners, budgets and clocks
are checked around opaque observations. No counter restoration or budget renewal.

Selected reports add at most two `endpoint_observations` entries with phase,
attempted and completed flags. Physical callback work is separate from the
probe's existing imitation prediction ledger. Probe readout snapshots, guards,
copies and callback time add work and memory. Invalid observed budget/progress
values on refusal are not proof of additional validated updates: original native
returns and receipts retain the trusted paid prefix. No unbounded record chain.

`endpoint_observer=None` preserves the default schema and delegated call shape.
An observer combined with `zero_velocity_control=True` refuses before work;
that existing pair has four endpoints and no implicit callback allocation.

Native test serving accepts exact native integer24, a BP/PC-family learner and
executedTS12 only. It retains role `probe_test`, fixed reserved context, and a
distinct `pick-cube-probe-test-smoke-24` rollout ID. It grants no teacher
provenance. Guard and probe options are exclusive; TRAIN seed stays default17.
Queries, correction collection, actors, clocks, visual/drift/projection/decoder/
monitor/intervention options refuse before native work. Existing TRAIN/guard
defaults and native32 executed-history checks remain.

## Source and focused checks

```text
src/adapters/continuous_probe.py         # existing runtime plus borrowed endpoints
src/app/action_history_encoding.py      # explicit scoring-only reset/history binding
scripts/maniskill_continuous_smoke.py    # existing frozen native serving
tests/test_continuous_probe_endpoints.py # private owners, paid failures, latched guards
tests/test_probe_frozen_rollout.py       # reserved serving and fake native boundary
```

```powershell
python -m pytest -q tests/test_continuous_probe.py tests/test_guard_frozen_rollout.py tests/test_continuous_probe_endpoints.py tests/test_probe_frozen_rollout.py
python -m ruff check src/adapters/continuous_probe.py src/app/action_history_encoding.py scripts/maniskill_continuous_smoke.py tests/test_continuous_probe_endpoints.py tests/test_probe_frozen_rollout.py
python -m ruff format --check src/adapters/continuous_probe.py src/app/action_history_encoding.py scripts/maniskill_continuous_smoke.py tests/test_continuous_probe_endpoints.py tests/test_probe_frozen_rollout.py
python -m mypy --platform linux --follow-imports silent src/adapters/continuous_probe.py src/app/action_history_encoding.py scripts/maniskill_continuous_smoke.py tests/test_continuous_probe_endpoints.py tests/test_probe_frozen_rollout.py
```

Use the prospective stage commands/ledgers for bounded instrumented acceptance.
The combined180 checks passed on Windows and pinned Linux, with final scoped
lint/format/types and independent actual acceptance. This covers the new observer
and reserved serving plus the existing probe/guard defaults.
[Executed evidence and costs](../artifacts/runs/continuous-probe-endpoints-20261021/RESULTS.md).
Software correctness does not establish task success, reliable learning, old-task
retention, transfer, safety or circadian advantage. Extend physical scoring through
this existing callback and serving path; freeze separate scientific budgets.


The separately frozen native case completed and was independently accepted:
reserved24 fails50controls both before/after the original SGD1/18 update; all
three physical gains zero despite exact positive imitation gains.100controls/
500knowncontrolphysics paid, no retry. This is a negative fixed same-task-reset
result, not physical learning/new-task/retention/reliability/CPC evidence.
[Native outcomes, commands and costs](../artifacts/runs/native-fresh-probe-endpoints-20261021/RESULTS.md).


A separate fixed retention readout now succeeds on old TRAIN17/19 after the SAME
private SGD1/18 update:17/19controls versus cached17/18; all three success-score
changes zero. Whole source/probe report equivalence and both initial observations
validate; independent acceptance.36controls180knownphysics, no native24 or baseline
rerun. This is one known-context retention result, not general reliability or
physical new-task acquisition.
[Retention outcomes and costs](../artifacts/runs/native-probe-old-retention-20261021/RESULTS.md).
