"""Explicit class-weighted CE for the existing generic native BP array owner.

Inputs are admitted float64 arrays and bounded explicit class weights. One
validated detached SGD candidate becomes live atomically. Caller owns TRAIN
roles, label release, weight derivation, work accounting and budgets. No data
loading, scheduling, selection, final/probe access or new state format lives here.
Why: the invariant-game weighted port must retain its raw296 episode boundary.
"""

import random
from typing import Any

import numpy as np

from src.adapters import torch_feature_learner as native
from src.adapters.torch_decision import _same_values
from src.core.coupled_energy import Array
from src.core.demonstration_class_weights import validate_class_weights
from src.core.learner_ports import TrainingDiagnostic
from src.shared.torch_runtime import require_torch

TORCH_ARRAY_WEIGHTED_CE_ID = "torch_feature_array_class_weighted_cross_entropy_v1"


def _rng(torch: Any) -> tuple[Any, ...]:
    state = np.random.get_state()
    assert isinstance(state, tuple)
    numpy = (state[0], state[1].dtype.str, state[1].shape, state[1].tobytes(), *state[2:])
    return torch.get_rng_state(), numpy, random.getstate()


def _ownership(learner: native.TorchFeatureLearner) -> tuple[Any, ...]:
    optimizer = learner._optimizer
    parameters = tuple(getattr(learner._head, n) for n in native._PARAMETERS)
    if (
        len(optimizer.param_groups) != 1
        or type(optimizer.param_groups[0].get("params")) is not list
        or len(optimizer.param_groups[0]["params"]) != len(parameters)
        or any(
            p is not q for p, q in zip(optimizer.param_groups[0]["params"], parameters, strict=True)
        )
    ):
        raise ValueError("weighted BP donor optimizer binding changed")
    return (
        id(learner._policy),
        id(learner._payload),
        id(learner._head),
        id(optimizer),
        id(optimizer.state),
        id(optimizer.param_groups),
        tuple(id(group) for group in optimizer.param_groups),
        tuple(
            (id(group["params"]), tuple(id(p) for p in group["params"]))
            for group in optimizer.param_groups
        ),
        tuple((id(p), p.requires_grad, id(p.grad)) for p in parameters),
    )


def train_class_weighted_batch(
    learner: native.TorchFeatureLearner,
    features: Array,
    targets: Array,
    class_weights: tuple[float, ...],
) -> TrainingDiagnostic:
    """Apply sum(w_target*CE)/sum(w_target); never alter base policy or donor on failure.

    All-one weights use the original unweighted kernel for exact continuation.
    Foreign donor/input/RNG changes are refused, never erased by rollback.
    Subclasses retain the generic array contract (e.g. counted stream forks).
    """
    if not isinstance(learner, native.TorchFeatureLearner) or learner.policy.method != "bp":
        raise ValueError("weighted arrays require a native BP learner")
    policy = learner.policy
    weights = validate_class_weights(class_weights, policy.num_classes)
    torch = require_torch()
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("weighted BP requires grad-enabled non-inference mode")
    rng = _rng(torch)
    values = native._features(features, policy.feature_dim)
    if (
        type(targets) is not np.ndarray
        or targets.dtype != np.float64
        or targets.shape != (len(features), 1)
        or not np.isfinite(targets).all()
        or not ((targets >= 0) & (targets < policy.num_classes)).all()
        or not (targets == np.floor(targets)).all()
    ):
        raise ValueError("weighted targets require finite float64 Bx1 hard class indices")
    labels = torch.from_numpy(targets[:, 0].astype(np.int64))
    input_pin = (features.copy(), targets.copy())
    saved = learner.snapshot_state()
    ownership = _ownership(learner)
    updates, examples = saved.completed_updates + 1, saved.training_examples + len(features)
    native._count(updates, "completed_updates")
    native._count(examples, "training_examples")
    head, optimizer = native._restore_components(saved)
    optimizer.zero_grad(set_to_none=True)
    logits = head.forward_logits(values)
    # Preserve the original kernel when weighting is exactly neutral. General
    # weights normalize by target mass, not by all catalog entries or batch size.
    loss = (
        torch.nn.functional.cross_entropy(logits, labels)
        if all(w == 1.0 for w in weights)
        else torch.nn.functional.cross_entropy(
            logits, labels, weight=torch.tensor(weights, dtype=torch.float32, device="cpu")
        )
    )
    diagnostic = TrainingDiagnostic(TORCH_ARRAY_WEIGHTED_CE_ID, float(loss.item()))
    loss.backward()
    optimizer.step()
    native._state(head, optimizer, policy, updates, examples)
    try:
        unchanged = (
            _ownership(learner) == ownership
            and _same_values(saved, learner.snapshot_state())
            and _same_values(rng, _rng(torch))
            and features.dtype == input_pin[0].dtype
            and targets.dtype == input_pin[1].dtype
            and features.shape == input_pin[0].shape
            and targets.shape == input_pin[1].shape
            and np.array_equal(features, input_pin[0])
            and np.array_equal(targets, input_pin[1])
            and torch.is_grad_enabled()
            and not torch.is_inference_mode_enabled()
        )
    except Exception as error:
        raise ValueError("weighted BP donor, inputs, RNG or mode changed") from error
    if not unchanged:
        raise ValueError("weighted BP donor, inputs, RNG or mode changed")
    learner._payload = (head, optimizer, updates, examples)
    return diagnostic
