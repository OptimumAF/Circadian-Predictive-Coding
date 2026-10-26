"""Explicit reward-gradient update for the owned recurrent adapter family.

Inputs are complete policy trajectories plus the full behavior checkpoint.
Outputs are a native diagnostic and one atomic SGD commit. Data collection,
provenance authority, budgets, replay, sampling and evaluation live outside here.
Why this: rewards are not supervised class targets; keep the CE API unchanged.
"""

from typing import Any

import numpy as np

from src.adapters.torch_recurrent_learner import (
    TorchRecurrentLearner,
    TorchRecurrentSnapshot,
    _capture,
    _count,
    _equal_tensors,
    _features,
    _forward,
    _probabilities,
    _restore,
    _snapshot,
)
from src.core.learner_ports import TrainingDiagnostic
from src.core.policy_gradient_episode import (
    PolicyGradientEpisode,
    copy_policy_gradient_episode,
    policy_gradient_weights,
)
from src.core.legal_policy_gradient import (
    LegalPolicyGradientEpisode,
    copy_legal_policy_gradient_episode,
    condition_action_probabilities,
)
from src.shared.torch_runtime import require_torch

TORCH_RECURRENT_REWARD_ID = "torch_recurrent_reward_logprob_sum_v1"
TORCH_RECURRENT_LEGAL_REWARD_ID = "torch_recurrent_legal_reward_logprob_sum_v1"


def _same_snapshot(left: TorchRecurrentSnapshot, right: TorchRecurrentSnapshot) -> bool:
    return (
        left.policy == right.policy
        and left.completed_updates == right.completed_updates
        and left.training_examples == right.training_examples
        and left.optimizer_state["param_groups"] == right.optimizer_state["param_groups"]
        and _equal_tensors(left.parameters, right.parameters)
        and _equal_tensors(left.gradients, right.gradients)
        and _equal_tensors(left.optimizer_state["state"], right.optimizer_state["state"])
    )


def train_reward_episode(
    learner: TorchRecurrentLearner,
    episode: PolicyGradientEpisode | LegalPolicyGradientEpisode,
    behavior_state: TorchRecurrentSnapshot,
    *,
    discount: float = 1.0,
) -> TrainingDiagnostic:
    if type(learner) is not TorchRecurrentLearner:
        raise ValueError("reward updates require an exact TorchRecurrentLearner")
    legal: LegalPolicyGradientEpisode | None = None
    if isinstance(episode, LegalPolicyGradientEpisode):
        legal = copy_legal_policy_gradient_episode(episode)
        owned = legal.episode
    else:
        owned = copy_policy_gradient_episode(episode)
    if owned.role != "train":
        raise ValueError("only train-role policy trajectories may train rewards")
    policy = learner.policy
    if (
        owned.num_classes != policy.num_classes
        or len(owned.transitions[0].observation) != policy.feature_dim
        or len(owned.transitions) > policy.max_episode_steps
    ):
        raise ValueError("reward trajectory geometry differs from bound policy")
    weights = policy_gradient_weights(owned, discount)
    expected = _snapshot(behavior_state)
    saved = learner.snapshot_state()
    if not _same_snapshot(saved, expected):
        raise ValueError("reward trajectory behavior checkpoint is stale or differs")
    updates, examples = saved.completed_updates + 1, saved.training_examples + len(weights)
    _count(updates, "completed_updates")
    _count(examples, "training_examples")
    features = _features(
        np.array([step.observation for step in owned.transitions], dtype=np.float64), policy
    )
    parameters, optimizer = _restore(saved)
    loss = (
        _reward_loss(parameters, features, owned, weights, policy.hidden_dim)
        if legal is None
        else _reward_loss(parameters, features, owned, weights, policy.hidden_dim, legal=legal)
    )
    definition = TORCH_RECURRENT_REWARD_ID if legal is None else TORCH_RECURRENT_LEGAL_REWARD_ID
    diagnostic = TrainingDiagnostic(definition, float(loss.item()))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    _capture(parameters, optimizer, policy, updates, examples)
    # The staged full state is validated before any live state is published.
    learner._payload = (parameters, optimizer, updates, examples)
    return diagnostic


def _reward_loss(
    parameters: dict[str, Any],
    features: Any,
    episode: PolicyGradientEpisode,
    weights: tuple[float, ...],
    hidden_dim: int,
    *,
    legal: LegalPolicyGradientEpisode | None = None,
) -> Any:
    torch = require_torch()
    logits, _ = _forward(
        parameters, features, torch.zeros((1, hidden_dim), dtype=torch.float32, device="cpu")
    )
    probabilities = _probabilities(logits)
    expected_raw = episode.behavior_probabilities if legal is None else legal.raw_probabilities
    if not np.allclose(probabilities, np.array(expected_raw), atol=1e-6, rtol=0.0):
        raise ValueError("recorded behavior probabilities differ from current policy")
    if legal is not None:
        current = tuple(
            condition_action_probabilities(tuple(float(p) for p in row), support)
            for row, support in zip(probabilities, legal.legal_actions, strict=True)
        )
        if any(
            row[step.action] <= 0 for row, step in zip(current, episode.transitions, strict=True)
        ) or not np.allclose(
            np.array(current), np.array(episode.behavior_probabilities), atol=1e-6, rtol=0.0
        ):
            raise ValueError("recorded conditional behavior differs from current public policy")
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for row, support in enumerate(legal.legal_actions):
            mask[row, list(support)] = True
        logits = logits.masked_fill(~mask, float("-inf"))
        # Raw logits were checked above; -inf now represents excluded actions.
        # A public float32 underflow law that disagrees with this stable law is
        # refused, rather than silently repaired or assigned another distribution.
        conditional = torch.softmax(logits, dim=1)
        if not bool(torch.isfinite(conditional).all().item()) or not np.allclose(
            conditional.detach().cpu().numpy(),
            np.array(episode.behavior_probabilities),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("recorded conditional behavior differs from masked native policy")
    actions = torch.tensor([step.action for step in episode.transitions], dtype=torch.int64)
    selected = torch.log_softmax(logits, dim=1)[torch.arange(len(weights)), actions]
    coefficients = torch.tensor(weights, dtype=torch.float32, device="cpu")
    # gamma^t G_t gives the discounted episodic-return estimator. Variable length
    # must not silently introduce a mean, advantage normalization or bootstrap.
    return -(coefficients * selected).sum()
