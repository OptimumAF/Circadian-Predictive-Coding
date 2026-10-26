"""Explicit weighted TRAIN-episode CE on an encoding-bound native BP owner.

Inputs are a complete raw296 episode and bounded explicit class weights. Output
is a separately named diagnostic and one atomic native SGD update. The caller
owns weight derivation, release authority and budgets. There is no scheduling,
evaluation, new snapshot format, PC/CPC update law or automatic weight selection.
"""

from copy import deepcopy
from dataclasses import fields
import random
from typing import Any, cast

import numpy as np

from src.adapters import torch_feature_learner as native
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.torch_feature_rehearsal_pilot import _same_state
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

TORCH_INVARIANT_WEIGHTED_CE_ID = "torch_invariant_feature_episode_class_weighted_cross_entropy_v1"


def _record(value: Any, kind: type) -> None:
    if type(value) is not kind or set(vars(value)) != {field.name for field in fields(kind)}:
        raise ValueError(f"an exact complete {kind.__name__} is required")


def _admit(
    learner: InvariantTorchFeatureLearner,
    episode: DemonstrationEpisode,
    class_weights: tuple[float, ...],
) -> tuple[DemonstrationEpisode, InvariantTorchFeaturePolicy, tuple[float, ...]]:
    if (
        type(learner) is not InvariantTorchFeatureLearner
        or type(getattr(learner, "_native", None)) is not native.TorchFeatureLearner
    ):
        raise ValueError("weighted updates require an exact encoding-bound native BP learner")
    policy = learner.policy
    if policy.native_policy.method != "bp" or learner._native.policy != policy.native_policy:
        raise ValueError("weighted feature CE requires the bound BP policy")
    _record(episode, DemonstrationEpisode)
    if type(episode.transitions) is not tuple or not 1 <= len(episode.transitions) <= 16:
        raise ValueError("weighted updates require a complete1..16-row TRAIN episode")
    for transition in episode.transitions:
        _record(transition, DemonstrationTransition)
        if any(
            type(row) is not tuple or len(row) != 296
            for row in (transition.observation, transition.next_observation)
        ):
            raise ValueError("weighted feature episodes require raw296 public observations")
    owned = copy_episode(episode)
    if owned.role != "train" or owned.num_classes != policy.num_classes:
        raise ValueError("only complete bound-class TRAIN episodes may use weighted BP")
    # The terminal next view is still complete episode evidence, although it is
    # never a CE input. Refuse malformed public fields before candidate work.
    for transition in owned.transitions:
        canonicalize_public_history(transition.next_observation)
    return owned, policy, validate_class_weights(class_weights, policy.num_classes)


def _rng(torch: Any) -> tuple[Any, ...]:
    numpy = cast(tuple[Any, ...], np.random.get_state())
    # The native-state comparator handles scalars/bytes/Torch tensors, not NumPy
    # array truth values. Preserve exact array geometry and bytes for RNG equality.
    numpy_state = (numpy[0], numpy[1].dtype.str, numpy[1].shape, numpy[1].tobytes(), *numpy[2:])
    return torch.get_rng_state(), numpy_state, random.getstate()


def _autograd_marks(owner: native.TorchFeatureLearner) -> tuple[Any, ...]:
    return tuple(
        (id(parameter), parameter.requires_grad, id(parameter.grad))
        for parameter in (getattr(owner._head, name) for name in native._PARAMETERS)
    )


def train_weighted_episode(
    learner: InvariantTorchFeatureLearner,
    episode: DemonstrationEpisode,
    class_weights: tuple[float, ...],
) -> TrainingDiagnostic:
    """Publish one sum(w_target*CE)/sum(w_target) update on detached native state.

    All-one weights follow ordinary BP mathematics. RNG and donor changes are
    refused before publication; observed foreign changes are never rolled back.
    The episode's reward/next frame remain evidence, not prediction inputs.
    """
    owned, policy, weights = _admit(learner, episode, class_weights)
    torch = require_torch()
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("weighted BP requires grad-enabled non-inference mode")
    owner = learner._native
    payload_owner, policy_owner = owner._payload, learner._policy
    native_policy_owner = owner._policy
    saved = learner.snapshot_state()
    marks = _autograd_marks(owner)
    rng = deepcopy(_rng(torch))
    features, targets = episode_arrays(owned)
    projected = learner._features(features)
    values = native._features(projected, policy.feature_dim)
    updates = saved.completed_updates + 1
    examples = saved.training_examples + len(features)
    native._count(updates, "completed_updates")
    native._count(examples, "training_examples")
    head, optimizer = native._restore_components(saved.native_state)
    optimizer.zero_grad(set_to_none=True)
    logits = head.forward_logits(values)
    # Why: normalize by observed target-weight mass, as in the existing weighted
    # recurrent objective. Absent-class weights cannot change the update scale.
    loss = torch.nn.functional.cross_entropy(
        logits,
        torch.from_numpy(targets[:, 0].astype(np.int64)),
        weight=torch.tensor(weights, dtype=torch.float32, device="cpu"),
    )
    diagnostic = TrainingDiagnostic(TORCH_INVARIANT_WEIGHTED_CE_ID, float(loss.item()))
    loss.backward()
    optimizer.step()
    native._state(head, optimizer, policy.native_policy, updates, examples)
    if (
        learner._native is not owner
        or learner._policy is not policy_owner
        or owner._policy is not native_policy_owner
        or owner._payload is not payload_owner
        or _autograd_marks(owner) != marks
        or not _same_state(saved, learner.snapshot_state())
        or not _same_state(rng, _rng(torch))
        or _admit(learner, episode, class_weights) != (owned, policy, weights)
        or not torch.is_grad_enabled()
        or torch.is_inference_mode_enabled()
    ):
        raise ValueError("weighted BP donor, episode, policy, gradients, RNG or mode changed")
    # Candidate validation and donor/input proof both finish before this one write.
    owner._payload = (head, optimizer, updates, examples)
    return diagnostic
