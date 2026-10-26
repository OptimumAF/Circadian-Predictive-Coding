# Explicit class-balanced public-cue training

`PublicCueRecurrentLearner.train_class_balanced_episode(episode)` updates the
head selected from the first public observation using the existing native
weighted recurrent loss. Supply one complete released `train` episode, raw148
coordinates and seven action classes. Other roles are refused.

```python
from src.adapters.minigrid_public_cue_recurrent_learner import PublicCueRecurrentLearner

learner = PublicCueRecurrentLearner()
diagnostic = learner.train_class_balanced_episode(released_train_episode)
```

Weights are fixed from this episode alone: `N / (present_classes * class_count)`;
absent classes receive1. Native weighted mean CE divides by observed target
weight mass. There are no new weight settings. `train_batch` and `train_episode`
retain ordinary mean CE. Prediction and fresh recurrent sessions are unchanged.
See [the existing weighted objective](class-weighted-demonstrations.md).

The selected head continues its full parameters, gradients and SGD momentum;
the other head remains unchanged. The call validates original and owned episode
data around native work and final state reads. A failed native call keeps its
known/unknown paid prefix and permanently refuses continuation or restore that
would erase it. Snapshot format, policy and failure schema remain unchanged.
The caller still owns consent, label release and execution budgets.

This explicit method is implemented and qualified:45software cases plus one
native parity case per Windows/Linux platform. The native case mixes ordinary
and weighted updates, matches existing direct-native state exactly, and verifies
source/real actor/RNG and frozen sessions. It is not a task-success experiment.
The original weighted control's17/18teacher fit and1/2native goals supplied no
joint acquisition benefit; those negatives and issue136's spent incomplete run
remain unchanged.

The existing `TorchTrainWeightedLearner` schedule adapter now admits this family
through the [normal delayed-label stream](public-cue-weighted-stream.md). It binds
arrays to the original complete TRAIN episode, validates per-head prefix clocks,
and retains public-cue paid failures separately from its old atomic rollback
branch. This integration has its own scoped69distinct scripted+1native parity evidence;
it does not imply native task success or retry the spent experiment.

Exact bounded commands, costs and skipped checks are in
`artifacts/runs/public-cue-weighted-train-20261025/RESULTS.md`. New comparative
acquisition/adaptation work requires a fixed prospective protocol after those
software gates. No circadian, retention/recovery, generalization or release
benefit follows from native objective parity.
