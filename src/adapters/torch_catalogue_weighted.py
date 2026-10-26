"""Atomic weighted TRAIN catalogues on existing BP/PC/neutral feature owners.

Inputs are complete raw296/K7 sources with their original spans and explicit
weights. Outputs are one native update and a distinct objective diagnostic.
Catalogue rows are independent; they never form a fictitious episode. This
module owns no schedule, budget, evaluation, probe, replay or promotion.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from src.adapters import torch_feature_learner as native
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
)
from src.adapters.minigrid_train_catalogue import (
    CatalogueSourceSpan,
    compose_catalogue_training,
)
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_feature_weighted import _autograd_marks, _record, _rng
from src.adapters.torch_pc_weighted import _stage as _stage_pc
from src.core.categorical_pc_weighted import OBJECTIVE
from src.core.demonstration_class_weights import validate_class_weights
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    copy_episode,
)
from src.core.learner_ports import TrainingDiagnostic
from src.shared.torch_runtime import require_torch

TORCH_CATALOGUE_WEIGHTED_CE_ID = "torch_invariant_catalogue_class_weighted_cross_entropy_v1"
TORCH_CATALOGUE_WEIGHTED_PC_ID = f"torch_invariant_catalogue_{OBJECTIVE}"


@dataclass(frozen=True)
class TrainCatalogue:
    """Owned complete source evidence and half-open spans, not an episode."""

    episodes: tuple[DemonstrationEpisode, ...]
    source_spans: tuple[CatalogueSourceSpan, ...]


def prepare_train_catalogue(episodes: tuple[DemonstrationEpisode, ...]) -> TrainCatalogue:
    """Own at most six complete sources and54 rows; validate unused next views."""
    if type(episodes) is not tuple or not 1 <= len(episodes) <= 6:
        raise ValueError("weighted catalogue requires an exact tuple of1..6 TRAIN sources")
    for episode in episodes:
        _record(episode, DemonstrationEpisode)
        if type(episode.transitions) is not tuple or not 1 <= len(episode.transitions) <= 16:
            raise ValueError("catalogue requires complete1..16-row source episodes")
        for transition in episode.transitions:
            _record(transition, DemonstrationTransition)
    if sum(len(episode.transitions) for episode in episodes) > 54:
        raise ValueError("weighted catalogue exceeds54 rows")
    owned = tuple(copy_episode(episode) for episode in episodes)
    plan = compose_catalogue_training(owned, 1)
    return TrainCatalogue(owned, plan.source_spans)


def _catalogue(catalogue: TrainCatalogue) -> TrainCatalogue:
    _record(catalogue, TrainCatalogue)
    if type(catalogue.source_spans) is not tuple:
        raise ValueError("catalogue spans require an exact tuple")
    for span in catalogue.source_spans:
        _record(span, CatalogueSourceSpan)
        if (
            type(span.episode_id) is not str
            or type(span.context) is not str
            or type(span.start) is not int
            or type(span.stop) is not int
        ):
            raise ValueError("catalogue spans require exact string identities and integer bounds")
    owned = prepare_train_catalogue(catalogue.episodes)
    if catalogue.source_spans != owned.source_spans:
        raise ValueError("catalogue source spans differ from complete TRAIN evidence")
    return owned


def catalogue_class_weights(catalogue: TrainCatalogue) -> tuple[float, ...]:
    """Equalize class mass across this complete catalogue, not per episode.

    Why: each wake contains all source rows. Its explicitly separate weighting
    rule uses their total counts; no episode boundary or reserved label is lost.
    """
    owned = _catalogue(catalogue)
    counts = [0] * 7
    for episode in owned.episodes:
        for transition in episode.transitions:
            counts[transition.action] += 1
    present = sum(count > 0 for count in counts)
    rows = sum(counts)
    return tuple(float(rows / (present * count)) if count else 1.0 for count in counts)


def _admit(
    learner: InvariantTorchFeatureLearner,
    catalogue: TrainCatalogue,
    class_weights: tuple[float, ...],
) -> tuple[TrainCatalogue, InvariantTorchFeaturePolicy, tuple[float, ...]]:
    if (
        type(learner) is not InvariantTorchFeatureLearner
        or type(getattr(learner, "_native", None)) is not native.TorchFeatureLearner
    ):
        raise ValueError("weighted catalogue requires an exact encoding-bound native learner")
    policy = learner.policy
    if policy.native_policy.method not in {"bp", "pc", "neutral_cpc"} or (
        learner._native.policy != policy.native_policy or policy.num_classes != 7
    ):
        raise ValueError("weighted catalogue requires the bound BP/PC/neutral K7 policy")
    return _catalogue(catalogue), policy, validate_class_weights(class_weights, 7)


def _stage_bp(head, optimizer, values, labels, weights):
    torch = require_torch()
    optimizer.zero_grad(set_to_none=True)
    # Why: constant observed weights cancel algebraically. Keep ordinary native
    # reduction order exactly, including when only absent-class weights differ.
    uniform = len({weights[index] for index in labels.tolist()}) == 1
    loss = torch.nn.functional.cross_entropy(
        head.forward_logits(values),
        labels,
        weight=None if uniform else torch.tensor(weights, dtype=torch.float32, device="cpu"),
    )
    diagnostic = TrainingDiagnostic(TORCH_CATALOGUE_WEIGHTED_CE_ID, float(loss.item()))
    loss.backward()
    optimizer.step()
    return diagnostic


def train_weighted_catalogue(
    learner: InvariantTorchFeatureLearner,
    catalogue: TrainCatalogue,
    class_weights: tuple[float, ...],
) -> TrainingDiagnostic:
    """Publish one54-row-bounded weighted update on detached complete state.

    Explicit all-one weights preserve ordinary native continuation. The caller
    owns released TRAIN authority and budgets; foreign changes cause refusal
    without rollback. Reward/next frames/spans are evidence, not feature inputs.
    """
    owned, policy, weights = _admit(learner, catalogue, class_weights)
    torch = require_torch()
    bp = policy.native_policy.method == "bp"
    if torch.is_inference_mode_enabled() or (bp and not torch.is_grad_enabled()):
        raise ValueError("weighted catalogue requires non-inference mode and BP gradients")
    mode = torch.is_grad_enabled()
    owner = learner._native
    payload_owner, policy_owner = owner._payload, learner._policy
    native_policy_owner = owner._policy
    saved = learner.snapshot_state()
    marks, rng = _autograd_marks(owner), deepcopy(_rng(torch))
    batch = compose_catalogue_training(owned.episodes, 1).training[0]
    values = native._features(learner._features(batch.features), policy.feature_dim)
    labels = torch.from_numpy(batch.targets[:, 0].astype(np.int64))
    updates, examples = saved.completed_updates + 1, saved.training_examples + len(values)
    native._count(updates, "completed_updates")
    native._count(examples, "training_examples")
    head, optimizer = native._restore_components(saved.native_state)
    if bp:
        diagnostic = _stage_bp(head, optimizer, values, labels, weights)
    else:
        result = _stage_pc(head, values, labels, policy, weights)
        diagnostic = TrainingDiagnostic(TORCH_CATALOGUE_WEIGHTED_PC_ID, result.value)
    native._state(head, optimizer, policy.native_policy, updates, examples)
    if (
        learner._native is not owner
        or learner._policy is not policy_owner
        or owner._policy is not native_policy_owner
        or owner._payload is not payload_owner
        or _autograd_marks(owner) != marks
        or not _same_state(saved, learner.snapshot_state())
        or not _same_state(rng, _rng(torch))
        or _admit(learner, catalogue, class_weights) != (owned, policy, weights)
        or torch.is_grad_enabled() != mode
        or torch.is_inference_mode_enabled()
    ):
        raise ValueError(
            "weighted catalogue donor, sources, policy, gradients, RNG or mode changed"
        )
    owner._payload = (head, optimizer, updates, examples)
    return diagnostic
