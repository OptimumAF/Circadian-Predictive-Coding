"""Compose the original public MiniGrid learner with fixed old-A TRAIN replay.

Inputs are two complete causal raw296/K7 A TRAIN episodes, the original invariant
native learner and independent wake/replay budgets. Output is an existing rehearsal
helper plus admission costs. No role conversion, learner update, evaluation, task
selection, runner, persistence or comparative claim belongs here.
"""

from dataclasses import dataclass

from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.minigrid_train_catalogue import _validate_sources
from src.adapters.torch_feature_learner import TorchFeaturePolicy
from src.app.stream_rehearsal import RecentStreamReplayPolicy, StreamRehearsal
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.circadian_predictive_coding import ReplayRetentionBudget
from src.core.demonstration_episode import DemonstrationEpisode, EpisodeReplaySchema

SCHEMA = EpisodeReplaySchema(296, 9, 7)
BANK_TRANSITIONS = 18
BANK_ARRAY_BYTES = 2 * (SCHEMA.encoded_dim + 1) * 8


@dataclass(frozen=True)
class MiniGridRehearsalPlan:
    """Composition and logical admission costs; process RSS includes other copies."""

    rehearsal: StreamRehearsal
    validation_projection_rows: int
    validation_projected_array_equivalent_bytes: int


def build_minigrid_old_a_rehearsal(
    learner: InvariantTorchFeatureLearner,
    budget: ToyBudgetSession,
    old_training_episodes: tuple[DemonstrationEpisode, ...],
    *,
    replay_budget: ToyBudgetSession,
    total_updates: int,
    every_updates: int = 1,
    wake_role: str = "train",
    wide_neural: bool = False,
) -> MiniGridRehearsalPlan:
    """Retain/replay both original old-A episodes through existing consolidation.

    Why: fixed membership prevents new-task wakes evicting the old-task control.
    Seeding is storage only. Actual prior acquisition and collection costs remain
    the caller's responsibility; neither is inferred or charged as new training.
    Raw transport is preserved and the native bound learner projects exactly once.
    Explicit wake_role="probe_train" is for the existing disposable fresh-task
    boundary; old_training_episodes must still be TRAIN and are never relabeled.
    Width is declared explicitly: the default original H8 recipe stays unchanged,
    while wide_neural=True admits only the otherwise identical original H32.
    """
    if type(wide_neural) is not bool:
        raise ValueError("wide_neural must be an exact bool")
    if type(old_training_episodes) is not tuple or len(old_training_episodes) != 2:
        raise ValueError("MiniGrid old-A replay requires exactly two complete TRAIN episodes")
    witness, rows = _validate_sources(old_training_episodes, 1)
    if rows != BANK_TRANSITIONS or any(
        episode.context != "A" or len(episode.transitions) != 9 for episode in old_training_episodes
    ):
        raise ValueError("MiniGrid old-A replay requires two nine-step A TRAIN episodes")
    if type(learner) is not InvariantTorchFeatureLearner:
        raise ValueError("MiniGrid old-A replay requires the original invariant native learner")
    policy = learner.policy
    hidden_dim = 32 if wide_neural else 8
    expected = InvariantTorchFeaturePolicy(
        TorchFeaturePolicy(policy.native_policy.method, 296, hidden_dim, 17, num_classes=7)
    )
    if policy != expected:
        raise ValueError(
            f"MiniGrid old-A replay requires the original H{hidden_dim}/seed17 native recipe"
        )
    witness.check(old_training_episodes)
    helper = StreamRehearsal(
        learner,
        budget,
        RecentStreamReplayPolicy(
            every_updates,
            BANK_TRANSITIONS,
            ReplayRetentionBudget(BANK_TRANSITIONS, BANK_ARRAY_BYTES),
        ),
        replay_budget=replay_budget,
        total_updates=total_updates,
        input_dim=296,
        episode_schema=SCHEMA,
        fixed_training_episodes=old_training_episodes,
        wake_role=wake_role,
    )
    return MiniGridRehearsalPlan(helper, rows * 2, rows * 2 * 296 * 8)
