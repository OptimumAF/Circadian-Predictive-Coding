"""Explicit weighted episode CE/BPTT on the existing owned recurrent state.

Inputs are a train-role complete episode and bounded explicit class weights.
The output is a named native diagnostic and one atomic update of the same owner.
Weight derivation, release authority, scheduling, evaluation and budgets live at
the caller boundary. No snapshot or policy format changes are introduced here.
"""

from dataclasses import fields

import numpy as np

from src.adapters import torch_recurrent_learner as native
from src.core.demonstration_class_weights import validate_class_weights
from src.core.demonstration_episode import DemonstrationEpisode, copy_episode, episode_arrays
from src.core.learner_ports import TrainingDiagnostic
from src.shared.torch_runtime import require_torch

TORCH_RECURRENT_WEIGHTED_CE_ID = "torch_recurrent_episode_class_weighted_cross_entropy_v1"


def train_weighted_episode(
    learner: native.TorchRecurrentLearner,
    episode: DemonstrationEpisode,
    class_weights: tuple[float, ...],
) -> TrainingDiagnostic:
    """Commit one sum(weight*CE)/sum(target weight) update, preserving state on error."""
    if type(learner) is not native.TorchRecurrentLearner:
        raise ValueError("weighted updates require an exact TorchRecurrentLearner")
    if type(episode) is not DemonstrationEpisode or set(vars(episode)) != {
        field.name for field in fields(DemonstrationEpisode)
    }:
        raise ValueError("weighted updates require an exact complete DemonstrationEpisode")
    owned = copy_episode(episode)
    policy = learner.policy
    if owned.role != "train" or owned.num_classes != policy.num_classes:
        raise ValueError("only bound-class train-role demonstrations may train")
    weights = validate_class_weights(class_weights, policy.num_classes)
    features, targets = episode_arrays(owned)
    values = native._features(features, policy)
    saved = learner.snapshot_state()
    updates, examples = saved.completed_updates + 1, saved.training_examples + len(features)
    native._count(updates, "completed_updates")
    native._count(examples, "training_examples")
    parameters, optimizer = native._restore(saved)
    torch = require_torch()
    optimizer.zero_grad(set_to_none=True)
    logits, _ = native._forward(
        parameters,
        values,
        torch.zeros((1, policy.hidden_dim), dtype=torch.float32, device="cpu"),
    )
    # Why: native weighted mean divides by observed target-weight mass. Dividing
    # by timestep count would change the step scale when class support changes.
    loss = torch.nn.functional.cross_entropy(
        logits,
        torch.from_numpy(targets[:, 0].astype(np.int64)),
        weight=torch.tensor(weights, dtype=torch.float32, device="cpu"),
    )
    diagnostic = TrainingDiagnostic(TORCH_RECURRENT_WEIGHTED_CE_ID, float(loss.item()))
    loss.backward()
    optimizer.step()
    native._capture(parameters, optimizer, policy, updates, examples)
    # Work stays on detached parameters/optimizer until every validation passes.
    learner._payload = (parameters, optimizer, updates, examples)
    return diagnostic
