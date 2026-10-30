# Joint same-context replay for fixed LS heads

`StreamRehearsal` has an explicit `joint_context` option for the owned contextual
LS learner. It uses the existing delayed stream, complete TRAIN episode buffer,
replay budget and atomic consolidation. Defaults and other learner paths remain.

```python
from src.app.stream_rehearsal import StreamRehearsal, RecentStreamReplayPolicy
from src.core.context_joint_replay import JointContextReplayPolicy
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import EpisodeReplaySchema

helper = StreamRehearsal(
    source, wake_budget,
    RecentStreamReplayPolicy(1, 12, ReplayRetentionBudget(30, 65536)),
    replay_budget=replay_budget, total_updates=len(training), input_dim=9,
    episode_schema=EpisodeReplaySchema(9, 3, 6),
    joint_context=JointContextReplayPolicy(9, 6, (4, 5), "joint"),
)
```

Pass this helper to the existing `AttributionStream(..., rehearsal=helper)`.
`source` is an owned ContextLinearFitLearner; budgets are distinct original
ToyBudgetSessions. Attach an active RSS sampler when declaring an RSS limit. Run
from the physical D: checkout with D: temporary storage. Training batches must
retain original TRAIN roles and canonical complete-episode metadata/row IDs.
The helper does not attest external provenance or grant new replay consent.

Each ordinary wake first completes its original fit. At a replay event, the
helper selects newest distinct complete TRAIN episodes from the **same public
context**, excluding exact copies of the current packet. It adds eligible old
episodes within `rows_per_event`, then performs **one additional LS fit** on
current+old rows through the existing consolidation. No prior eligible episode
means a reported skip, not a fabricated update. Unknown/mixed public context or
conflicting labels on identical selected features refuse before fitting.

The `"current_only"` control repeats the current packet to the same matrix row
count. Both modes pay the same native fit/row counts under the same fixed data
schedule; SVD rank, data diversity, copying and total computation can differ.
No matched-compute or method-superiority claim follows from row matching.

All additional matrix rows are charged permanently to the replay budget before
delegation—including the repeated current rows. For example, a3row wake followed
by a9row joint fit pays12physical row exposures in two solves. Its replay quota
charge is9, not only the6old rows. Event metadata separates current role/rows,
retained TRAIN IDs/rows used, total matrix rows, attempt/return/commit counts and
owned failed-fit work. Lost returns or post-fit corruption retain costs without
publishing a partial consolidation. Source and actor remain unchanged.

Normal TRAIN mode retains arrived TRAIN episodes under the existing bounded
episode policy, including new tasks. Private `wake_role="probe_train"` still
requires an explicit nonempty `fixed_training_episodes` TRAIN bank. Private
current probe inputs are transient; they never enter retention. Original bank
roles are preserved. The generic old-TRAIN return must not replay probe labels;
this option does not convert private probe data into product TRAIN authority.

Only recent categorical episode replay with a matching ContextLinearFitSnapshot
is supported. Distillation, incident gating, coverage and other joint option
combinations refuse. New policy and budget identities/values are bound; changed
limits, deadlines, counters or paid records stop further work. Paid event
containers/rows are restored on corruption. Same-owner continuation preserves
charges and refreshes references only after its validated commit.

[Qualification and all costs/failures](reports/context-joint-rehearsal-20261029.md):
the corrected original-TRAIN stream fixture runs10wakes/8consolidations,
18solves96rows, and executes all six known workflows after old/new/old-return.
This is scoped implementation evidence, not a new frozen comparative study,
generalization, circadian benefit or release qualification.


A thin outcome-free preview is available via
`python -B -m scripts.compare_joint_context_retention`. The frozen #287 comparison
failed before fitting because its driver supplied a zero replay limit on the
wake budget; the driver now uses the required absent limit and real zero-fit
admission checks pass. Its original numerical case remains FAILED/SPENT and
must not be rerun. No comparative retention result follows. See the
[failure and admission-repair report](reports/joint-context-retention-comparison-20261029.md).
