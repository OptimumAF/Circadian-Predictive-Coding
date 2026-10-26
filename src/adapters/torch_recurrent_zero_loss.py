"""Explicit zero-loss SGD control on original observed TRAIN trajectory rows.

Inputs are one owned recurrent learner and an unchanged policy episode. Outputs
are a distinct diagnostic and one atomic staged native SGD step. Only observation
rows enter this objective: recorded actions/rewards/behavior remain evidence,
never relabeled on-policy targets. Release, provenance, sampling, budgets, replay
exposure accounting and comparisons remain the caller's responsibility.
"""

import numpy as np

from src.adapters import torch_recurrent_learner as native
from src.core.learner_ports import TrainingDiagnostic
from src.core.policy_gradient_episode import PolicyGradientEpisode, copy_policy_gradient_episode
from src.shared.torch_runtime import require_torch

TORCH_RECURRENT_ZERO_LOSS_ID = "torch_recurrent_observed_episode_zero_loss_sgd_v1"


def train_zero_loss_episode(
    learner: native.TorchRecurrentLearner, episode: PolicyGradientEpisode
) -> TrainingDiagnostic:
    """Commit dense zero gradients, allowing existing SGD momentum to continue.

    Why: a detached constant or skipped backward leaves gradients None and skips
    momentum. Elementwise zero before summation also avoids reduction overflow.
    Full staged validation precedes publication, preserving live state on failure.
    """
    if type(learner) is not native.TorchRecurrentLearner:
        raise ValueError("zero-loss updates require an exact TorchRecurrentLearner")
    owned = copy_policy_gradient_episode(episode)
    if owned.role != "train":
        raise ValueError("only train-role observed episodes may train the zero-loss control")
    policy = learner.policy
    if (
        owned.num_classes != policy.num_classes
        or len(owned.transitions[0].observation) != policy.feature_dim
        or len(owned.transitions) > policy.max_episode_steps
    ):
        raise ValueError("zero-loss trajectory geometry differs from bound policy")
    saved = learner.snapshot_state()
    rows = len(owned.transitions)
    updates, examples = saved.completed_updates + 1, saved.training_examples + rows
    native._count(updates, "completed_updates")
    native._count(examples, "training_examples")
    values = native._features(
        np.array([transition.observation for transition in owned.transitions], dtype=np.float64),
        policy,
    )
    parameters, optimizer = native._restore(saved)
    torch = require_torch()
    optimizer.zero_grad(set_to_none=True)
    logits, _ = native._forward(
        parameters,
        values,
        torch.zeros((1, policy.hidden_dim), dtype=torch.float32, device="cpu"),
    )
    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError("zero-loss forward produced nonfinite logits")
    loss = (logits * 0.0).sum()
    diagnostic = TrainingDiagnostic(TORCH_RECURRENT_ZERO_LOSS_ID, float(loss.item()))
    if diagnostic.value != 0.0:
        raise FloatingPointError("zero-loss objective must be exactly zero")
    loss.backward()
    if set(parameters) != set(native._NAMES):
        raise ValueError("zero-loss control requires all five recurrent parameters")
    for name, parameter in parameters.items():
        gradient = parameter.grad
        if gradient is None or gradient.is_sparse or gradient.layout != torch.strided:
            raise ValueError("zero-loss control requires dense zero gradients, never None")
        native._tensor(gradient, tuple(parameter.shape), f"zero-loss gradient {name}")
        if int(torch.count_nonzero(gradient).item()) != 0:
            raise ValueError("zero-loss control requires exactly zero parameter gradients")
    optimizer.step()
    native._capture(parameters, optimizer, policy, updates, examples)
    learner._payload = (parameters, optimizer, updates, examples)
    return diagnostic
