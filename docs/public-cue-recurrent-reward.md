# Reward updates for the public-cue recurrent learner

The existing joint TRAIN example has two recurrent heads, selected by the
initial public cue. Its complete checkpoint cannot restore into the historical
single-head reward learner. `PublicCueRewardLearner` preserves that two-head
architecture and delegates each TRAIN reward update to the existing native
gamma1 policy-gradient implementation. It adds no new reward mathematics.

The input is an exact `PublicCueRewardUpdate(episode, behavior_state)` and the
matching episode identifier as the target. The target releases the trajectory;
it is not a class label. The behavior state contains both heads, including
parameters, optimizer momentum, gradients and counters. Both heads must match
the current learner. The initial public frame selects the one updated head;
the other head stays unchanged. Probe and final roles are refused.

```python
from src.adapters.minigrid_public_cue_reward import (
    PublicCueRewardLearner, PublicCueRewardUpdate,
)

# Inside an already budgeted current-run workflow, after validating its paid
# joint TRAIN endpoint and collected TRAIN trajectory:
candidate = PublicCueRewardLearner(current_state.policy)
candidate.restore_state(current_state)
update = PublicCueRewardUpdate(collection["episode"], collection["behavior_state"])
# Release through the existing runtime/inbox before calling train_batch.
diagnostic = candidate.train_batch(update, update.episode.episode_id)
```

This example does not supply consent, collection, budgets, delayed releases,
comparison controls or evaluation. Use the existing `ActorShadowRuntime` and
MiniGrid reward collector for those boundaries. Do not restore historical JSON
weights or feed final observations into this path. Constructor/staging/copy and
collection costs must be budgeted before genuine use.

Source, packet and nested native owners/values are checked around callbacks.
The final proof reads actual live values after snapshot callbacks. A returned
diagnostic is pinned before later callbacks. Corruption or lost native work
retains a failed prefix with attempted/returned counters and permanently blocks
reuse or restoration. Forks and episode sessions delegate to the existing owner.

The same port also accepts the existing `MomentumOnlyUpdate` with the original
TRAIN trajectory and matching release identifier. This selects the existing
native zero-loss method, preserving optimizer momentum while producing dense
zero gradients. The diagnostic must have the distinct zero-loss definition and
value zero. Actions, rewards and recorded probabilities stay unchanged evidence;
they are not replaced with control predictions or claimed to be on-policy.
Only the initial public cue selects the updated head. The reward packet still
requires its full matching behavior checkpoint and uses its original gamma1
objective. Collection/release/budget/consent authority stays with the caller.
[Control dispatch and mixed-runtime checks](reports/joint-recurrent-momentum-port-20261031.md)
pass; a full joint-warm native comparison remains open.

[Scoped software evidence](reports/joint-recurrent-reward-port-20261031.md)
covers literal native ports and the actual collector/runtime composition.
The [joint-warm command](recurrent-joint-example.md) now composes these ports
through the existing runner with scoped software qualification. Genuine native
continuation remains unqualified: the [single attempted comparison](reports/joint-recurrent-reward-native-20261031.md)
completed learning but failed its serialization-work cap during readouts, leaving
no retained comparison or final preservation proof. Its allowance remains spent.
The historical A-only command and all its negative
results remain unchanged. Full #390, recurrent-RL usefulness and release remain
open; this component alone establishes no learning or circadian benefit.
