"""Atomic complete-TRAIN-episode weighted PC/neutral feature updates.

This adapter admits released raw296/K7 episodes and explicit class weights,
stages a detached numerical proposal and complete native state transition, then
publishes one owned payload. It supplies no schedule, weight selection, active
gating, replay, evaluation access, budget renewal or comparative experiment.
"""

from copy import deepcopy
from typing import Any, Callable, TypeVar

import numpy as np

from src.adapters import torch_feature_learner as native
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_feature_weighted import _autograd_marks, _record, _rng
from src.core.categorical_pc_weighted import OBJECTIVE, class_weighted_pc_update
from src.core.demonstration_class_weights import validate_class_weights
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    copy_episode,
    episode_arrays,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.shared.torch_runtime import require_torch

TORCH_INVARIANT_WEIGHTED_PC_ID = f"torch_invariant_{OBJECTIVE}"
_StageResult = TypeVar("_StageResult")


def _admit(
    learner: InvariantTorchFeatureLearner,
    episode: DemonstrationEpisode,
    class_weights: tuple[float, ...],
    *,
    expected_role: str = "train",
) -> tuple[DemonstrationEpisode, InvariantTorchFeaturePolicy, tuple[float, ...]]:
    if type(expected_role) is not str or expected_role not in {"train", "probe_train"}:
        raise ValueError("unsupported weighted learning source role")
    if (
        type(learner) is not InvariantTorchFeatureLearner
        or type(getattr(learner, "_native", None)) is not native.TorchFeatureLearner
    ):
        raise ValueError("weighted PC requires an exact encoding-bound native learner")
    policy = learner.policy
    if (
        policy.native_policy.method not in {"pc", "neutral_cpc"}
        or learner._native.policy != policy.native_policy
    ):
        raise ValueError("weighted PC requires the bound PC or neutral_cpc policy")
    _record(episode, DemonstrationEpisode)
    if type(episode.transitions) is not tuple or not 1 <= len(episode.transitions) <= 16:
        raise ValueError("weighted PC requires a complete1..16-row TRAIN episode")
    for transition in episode.transitions:
        _record(transition, DemonstrationTransition)
        if any(
            type(row) is not tuple or len(row) != 296
            for row in (transition.observation, transition.next_observation)
        ):
            raise ValueError("weighted PC requires raw296 public observations")
    owned = copy_episode(episode)
    if owned.role != expected_role or owned.num_classes != policy.num_classes:
        raise ValueError("weighted PC requires complete bound-class TRAIN episodes")
    # Terminal next views are admission evidence, never training inputs.
    for transition in owned.transitions:
        canonicalize_public_history(transition.next_observation)
    return owned, policy, validate_class_weights(class_weights, policy.num_classes)


def _stage(head, values, labels, policy, weights):
    torch = require_torch()
    native_policy = policy.native_policy
    with torch.no_grad():
        proposal = class_weighted_pc_update(
            values,
            labels,
            tuple(getattr(head, name) for name in native._PARAMETERS),
            class_weights=weights,
            learning_rate=native_policy.learning_rate,
            inference_steps=native_policy.inference_steps,
            inference_learning_rate=native_policy.inference_learning_rate,
            role="train",
        )
        _apply_proposal(head, values, labels, policy, proposal)
    return TrainingDiagnostic(TORCH_INVARIANT_WEIGHTED_PC_ID, proposal.energy)


def _charge(work, name):
    if work is not None:
        work[name] += 1


def _apply_proposal(head, values, labels, policy, proposal, *, work=None):
    """Apply the complete existing native transition on a detached staged head."""
    torch = require_torch()
    _charge(work, "staging_context_evaluations_attempted")
    prior = torch.tanh(values @ head.weight_feature_hidden + head.bias_hidden)
    logits = proposal.hidden_state @ head.weight_hidden_output + head.bias_output
    error = torch.softmax(logits, dim=1) - torch.nn.functional.one_hot(
        labels, num_classes=policy.num_classes
    ).to(values.dtype)
    # Why: native chemistry, ages, traffic and legacy history must travel with
    # the parameters. Both fixed and bounded proposals use this same transition.
    _charge(work, "native_transitions_attempted")
    head._apply_training_gradients(
        proposal.hidden_state,
        error,
        proposal.hidden_state - prior,
        proposal.gradients,
        policy.native_policy.learning_rate,
    )
    _charge(work, "native_transitions_completed")


def train_weighted_episode(
    learner: InvariantTorchFeatureLearner,
    episode: DemonstrationEpisode,
    class_weights: tuple[float, ...],
) -> TrainingDiagnostic:
    """Publish one weighted partial-gradient update on complete detached state.

    All-one weights preserve full native PC/neutral continuation. Legacy native
    history keeps its original MSE statistic; this call reports actual weighted
    CE-plus-hidden energy. Foreign changes cause refusal and are not rolled back.
    """
    return _execute_episode(learner, episode, class_weights, _stage)


def _execute_episode(
    learner: InvariantTorchFeatureLearner,
    episode: DemonstrationEpisode,
    class_weights: tuple[float, ...],
    stage: Callable[..., _StageResult],
    *,
    prepared: tuple[DemonstrationEpisode, InvariantTorchFeaturePolicy, tuple[float, ...]]
    | None = None,
    work: dict[str, Any] | None = None,
    guard: Callable[[], bool] | None = None,
    expected_role: str = "train",
) -> _StageResult:
    """Share the original atomic boundary; fixed calls add no work or guard."""
    owned, policy, weights = (
        _admit(learner, episode, class_weights, expected_role=expected_role)
        if prepared is None
        else prepared
    )
    torch = require_torch()
    if torch.is_inference_mode_enabled():
        raise ValueError("weighted PC requires non-inference mode")
    mode = torch.is_grad_enabled()
    owner = learner._native
    payload_owner, policy_owner = owner._payload, learner._policy
    native_policy_owner = owner._policy
    saved = learner.snapshot_state()
    marks = _autograd_marks(owner)
    rng = deepcopy(_rng(torch))
    features, targets = episode_arrays(owned)
    _charge(work, "feature_projection_attempts")
    values = native._features(learner._features(features), policy.feature_dim)
    labels = torch.from_numpy(targets[:, 0].astype(np.int64))
    updates = saved.completed_updates + 1
    examples = saved.training_examples + len(features)
    native._count(updates, "completed_updates")
    native._count(examples, "training_examples")
    _charge(work, "native_state_restores_attempted")
    head, optimizer = native._restore_components(saved.native_state)
    diagnostic = stage(head, values, labels, policy, weights)
    _charge(work, "candidate_validations_attempted")
    native._state(head, optimizer, policy.native_policy, updates, examples)
    _charge(work, "candidate_validations_completed")
    if (
        learner._native is not owner
        or learner._policy is not policy_owner
        or owner._policy is not native_policy_owner
        or owner._payload is not payload_owner
        or _autograd_marks(owner) != marks
        or not _same_state(saved, learner.snapshot_state())
        or not _same_state(rng, _rng(torch))
        or _admit(learner, episode, class_weights, expected_role=expected_role)
        != (owned, policy, weights)
        or torch.is_grad_enabled() != mode
        or torch.is_inference_mode_enabled()
        or (guard is not None and not guard())
    ):
        raise ValueError("weighted PC donor, episode, policy, gradients, RNG or mode changed")
    owner._payload = (head, optimizer, updates, examples)
    _charge(work, "published_updates")
    return diagnostic
