# Explicit mean and weighted key/door demonstrations

The existing key/door recipe now exposes BC-only conditions using the same
typed learner, actor/inbox runtime, complete ordered episodes, native optimizer
and original eight-update session. No new orchestration layer is introduced.

```powershell
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-bc --describe
.venv/Scripts/python.exe -X utf8 -B -m scripts.run_coupled_attribution --scenario torch-key-door-weighted-bc --describe
```

Omit `--describe` to execute the chosen fixed local condition. Each condition
trains eight episodes/64 rows and scores checkpoints0/8 on all eight A/B/C/D cue
pairs, including bounded live episodes. All training finishes before scoring.
Source and serving actor remain unchanged; failed updates retain original budget
charges. No reward updates, replay, probes, promotion or final-driven selection.

`src/adapters/key_door_reward_pilot.py` owns recipe orchestration;
`scripts/run_coupled_attribution.py` owns CLI dispatch;
`tests/test_weighted_bc_pilot.py` covers routing/isolation/failures.
The public Python entry points are `describe_demonstration_protocol` and
`run_demonstration_pilot`, with an exact-bool keyword `class_weighted=False`.
Explicit `True` uses the accepted derived typed weighted packet. See
[weighted objective and native method](class-weighted-demonstrations.md).

Why this: training-label imbalance motivated an explicit experimental condition.
The default mean objective and all earlier routes remain available. Requested
objective identity is bound before collection and checked across updates/finals;
unexpected changes fail before more work. Each weighted slot records its derived
class-weight tuple and named native diagnostic. Weights use only that released
training packet; final/probe labels never derive them.

The frozen local protocol and results live in
[the run directory](../artifacts/runs/r42-weighted-bc-control-20261016/SCOPE.md).
Its accepted historical mean-CE control retains original costs and all negatives.
Historical comparison is not an independent concurrent replication. Successful
execution establishes routing/cost accounting, not useful adaptation or a
circadian advantage. Run new comparative work only under a separately frozen,
reviewed budget; do not use these final outcomes to tune seeds or objectives.

Focused checks use the run directory's `run_statics.py` and `run_focused.py`.
They record exact commands and receipts; these are scoped gates, not clean-clone
or repository-wide release acceptance. Safely extend this route by adding an
explicit typed train objective and train-only rationale, preserving final
isolation and existing defaults, then testing its owned runtime boundary.
