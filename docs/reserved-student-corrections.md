# Reserved-training corrections on learner-visited states

One fixed real corrective-data case has executed:50 new student-state labels
plus16 private full-network updates sharply improved their own fit but failed
both fresh resets and all three old resets. Source/deployed state stayed intact.
Independent source/PRE passed; actual results are lead-verified with independent
POST incomplete. [Negative result, commands, costs and limits](reports/student-corrective-native-20261101.md).
No further query/dose/experiment authority follows from this result.

`run_frozen_rollout` accepts a separate `probe_correction_consent` for an exact
BP learner on explicit `probe_train_seed=23` with executed TS12 history. This
permits new provisional goal-consistent observed-grasp teacher queries, bounded
by1..50steps. The student alone executes commands; teacher targets are labels.
The original TRAIN17/19/20 correction APIs and18row reserved teacher reader are
unchanged. Reserved24, final and mixed interventions cannot collect these labels.

```python
from src.core.pick_cube_probe_corrections import ProbeCorrectionConsent
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.adapters.continuous_probe import ContinuousProbePlan, run_continuous_probe
from scripts.maniskill_continuous_smoke import run_frozen_rollout

collected = run_frozen_rollout(
    output,
    learner=source,
    probe_train_seed=23,
    action_history_recipe=PickCubeTaskSpaceHistoryRecipe(),
    probe_correction_consent=ProbeCorrectionConsent(max_queries=50),
)
if not collected.report["complete"] or collected.corrections is None:
    raise RuntimeError("Incomplete collection releases no usable corrective batch")
batch = collected.corrections
plan = ContinuousProbePlan(batch.model_episode(), reserved_readout, updates=1)
result = run_continuous_probe(
    source,
    plan,
    correction_batch=batch,
    correction_release_tick=release_tick,
)
```

This is an API pattern, not permission to run an experiment. Declare new query,
input, private-update, diagnostic, simulator, copy, storage and time budgets first.
`reserved_readout` must be the released probe_test packet; it supplies no updates.
`release_tick` belongs to the collection's label chronology and must cover all
original arrivals (2×step+1). It does not advance or renew an existing clock.
The disposable runtime owns a separate delayed-label clock and update budget.

`ProbeCorrectionBatch.episode` retains the complete raw42 executed trajectory.
`labels` retains proposed targets/stages and original grasp feedback separately.
The model episode preserves executed commands when deriving TS12 history.
`training_pairs(now=...)` gates labels before projection and never substitutes
teacher targets into that executed history. No synthetic expert trajectory or
reserved teacher export is created. Private native-reference seals reject
corruption and equal replacement after admission; they are not serialized proof
or expert qualification. Imported data would require its own admission contract.

Collection reports include original query consent, `provisional_training_labels_collected`,
separate teacher provenance/work, actual native controls and frozen-state proofs.
`scoring_only` still means no optimizer/publication during the rollout; this
explicit new mode also collects provisional training labels. No teacher-forced
action, warmup refit, source/deployed update or implicit old TRAIN replay occurs.
New label consent does not follow from the original18row reader's consent.

The corrective probe currently requires exact BP and a single plan without
SGD-direction diagnostics or zero-velocity controls. Its training episode must
match the actual projected student trajectory. New targets enter only the
existing probe_train label runtime; public TRAIN-only fitting APIs stay separate.
Default SGD or the explicit private full-network option uses the same admitted
pairs; the newly executed literal composition used SGD. Reports contain metrics,
receipts and costs, not the adapted learner or weights. Corrective SGD preserves
all returned paid updates after history erasure/injection and permanently refuses
more work; partial/corrupt query sequences release no usable batch.

[Software evidence and limits](reports/reserved-student-corrections-20261101.md).
39distinct checks qualified, including one fake-scene collection-to-private-update
case. This is no real native adaptation, teacher quality, retention/recovery,
generalization, CPC or release claim. The final narrow capture hardening received
self-review/statics plus a metadata-only failure check, without another
independent pass. A frozen reviewed native protocol remains outstanding.
