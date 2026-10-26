"""Owned CPU recurrent behavior cloning through the existing native learner port.

Each array call is one ordered episode, starting from zero hidden activation.
The optional record entry point enforces complete train-role demonstrations.
Learned/SGD/gradient state is owned and atomic; live activation belongs only to a
bounded independent session. Reward updates use an explicit separate trajectory
method; no environment, replay or role release lives here.
"""

from copy import deepcopy
from dataclasses import dataclass, fields, replace
from math import isfinite
from typing import Any

import numpy as np

from src.core.coupled_energy import Array
from src.core.demonstration_episode import DemonstrationEpisode, copy_episode, episode_arrays
from src.core.learner_ports import TrainingDiagnostic
from src.core.policy_gradient_episode import PolicyGradientEpisode
from src.core.legal_policy_gradient import LegalPolicyGradientEpisode
from src.shared.torch_runtime import require_torch

TORCH_RECURRENT_CE_ID = "torch_recurrent_episode_cross_entropy_mean_v1"
_NAMES = ("weight_input", "weight_recurrent", "bias_hidden", "weight_output", "bias_output")
_MAX_COUNT = 2**53 - 1


@dataclass(frozen=True)
class TorchRecurrentPolicy:
    feature_dim: int
    hidden_dim: int
    num_classes: int
    seed: int
    learning_rate: float = 0.03
    momentum: float = 0.9
    max_episode_steps: int = 16

    def __post_init__(self) -> None:
        # Preserve toy/raw contracts;296 is exactly public first+current frames.
        # This changes neither episode length nor hidden width.
        if type(self.feature_dim) is not int or not (
            1 <= self.feature_dim <= 16 or self.feature_dim in (148, 296)
        ):
            raise ValueError("feature_dim must be a native integer in 1..16 or exactly148 or296")
        for name, lo, hi in (
            ("hidden_dim", 1, 32),
            ("num_classes", 2, 16),
            ("seed", 0, 2**31 - 1),
            ("max_episode_steps", 1, 16),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be a native integer in {lo}..{hi}")
        for name in ("learning_rate", "momentum"):
            value = getattr(self, name)
            try:
                valid = type(value) in (int, float) and isfinite(value)
            except OverflowError:
                valid = False
            if not valid or (value <= 0 if name == "learning_rate" else not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite and within its declared range")


def _policy(value: TorchRecurrentPolicy) -> TorchRecurrentPolicy:
    if type(value) is not TorchRecurrentPolicy or set(vars(value)) != {
        field.name for field in fields(TorchRecurrentPolicy)
    }:
        raise ValueError("an exact complete TorchRecurrentPolicy is required")
    return replace(value)


def _shapes(policy: TorchRecurrentPolicy) -> dict[str, tuple[int, int]]:
    d, h, k = policy.feature_dim, policy.hidden_dim, policy.num_classes
    return dict(zip(_NAMES, ((d, h), (h, h), (1, h), (h, k), (1, k)), strict=True))


def _tensor(value: Any, shape: tuple[int, int], name: str) -> None:
    torch = require_torch()
    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.float32
        or value.device != torch.device("cpu")
        or tuple(value.shape) != shape
        or not bool(torch.isfinite(value).all().item())
    ):
        raise ValueError(f"{name} must be a finite CPU float32 tensor with shape {shape}")


def _count(value: Any, name: str) -> None:
    if type(value) is not int or not 0 <= value <= _MAX_COUNT:
        raise ValueError(f"{name} must be a bounded nonnegative native integer")


@dataclass(frozen=True)
class TorchRecurrentSnapshot:
    format_version: int
    policy: TorchRecurrentPolicy
    parameters: dict[str, Any]
    optimizer_state: dict[str, Any]
    gradients: dict[str, Any]
    completed_updates: int
    training_examples: int


def _sgd_options(policy: TorchRecurrentPolicy) -> dict[str, Any]:
    # Why this: bind the existing SGD defaults explicitly so full snapshot
    # validation needs neither temporary parameters nor an optimizer per read.
    return dict(
        lr=policy.learning_rate,
        momentum=policy.momentum,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        maximize=False,
        foreach=None,
        differentiable=False,
        fused=None,
    )


def _optimizer(parameters: dict[str, Any], policy: TorchRecurrentPolicy) -> Any:
    return require_torch().optim.SGD(list(parameters.values()), **_sgd_options(policy))


def _validate_optimizer(snapshot: TorchRecurrentSnapshot) -> None:
    state, policy = snapshot.optimizer_state, snapshot.policy
    if type(state) is not dict or state.keys() != {"state", "param_groups"}:
        raise ValueError("optimizer state must contain exactly state and param_groups")
    # Keep exact keys/types/order, rather than permissive load_state_dict.
    expected = dict(_sgd_options(policy), params=list(range(len(_NAMES))))
    groups = state["param_groups"]
    if type(groups) is not list or len(groups) != 1 or type(groups[0]) is not dict:
        raise ValueError("optimizer requires exactly one canonical parameter group")
    if groups[0].keys() != expected.keys() or any(
        type(groups[0][key]) is not type(value) or groups[0][key] != value
        for key, value in expected.items()
    ):
        raise ValueError("optimizer configuration differs from bound native SGD")
    if any(type(index) is not int for index in groups[0]["params"]):
        raise ValueError("optimizer parameter IDs must be native integers")
    values = state["state"]
    active = bool(snapshot.completed_updates and policy.momentum)
    if type(values) is not dict or set(values) != (set(range(5)) if active else set()):
        raise ValueError("optimizer momentum state differs from update clocks")
    if any(type(index) is not int for index in values):
        raise ValueError("optimizer momentum IDs must be native integers")
    for index, (name, shape) in enumerate(_shapes(policy).items()):
        if active:
            entry = values[index]
            if type(entry) is not dict or entry.keys() != {"momentum_buffer"}:
                raise ValueError("optimizer momentum fields differ")
            _tensor(entry["momentum_buffer"], shape, f"momentum {name}")


def _snapshot(value: TorchRecurrentSnapshot) -> TorchRecurrentSnapshot:
    if type(value) is not TorchRecurrentSnapshot or set(vars(value)) != {
        field.name for field in fields(TorchRecurrentSnapshot)
    }:
        raise ValueError("an exact complete TorchRecurrentSnapshot is required")
    if type(value.format_version) is not int or value.format_version != 1:
        raise ValueError("unsupported recurrent snapshot version")
    policy = _policy(value.policy)
    _count(value.completed_updates, "completed_updates")
    _count(value.training_examples, "training_examples")
    if (
        not value.completed_updates
        <= value.training_examples
        <= policy.max_episode_steps * value.completed_updates
    ):
        raise ValueError("example clock differs from complete episode updates")
    for name in ("parameters", "gradients"):
        mapping = getattr(value, name)
        if type(mapping) is not dict or mapping.keys() != set(_NAMES):
            raise ValueError(f"snapshot {name} must contain exactly the recurrent fields")
        for key, shape in _shapes(policy).items():
            if name == "gradients" and value.completed_updates == 0 and mapping[key] is None:
                continue
            _tensor(mapping[key], shape, f"{name} {key}")
            if name == "gradients" and value.completed_updates == 0:
                raise ValueError("initial gradients must be None")
    _validate_optimizer(value)
    return deepcopy(replace(value, policy=policy))


def _capture(
    parameters: dict[str, Any],
    optimizer: Any,
    policy: TorchRecurrentPolicy,
    updates: int,
    examples: int,
) -> TorchRecurrentSnapshot:
    return _snapshot(
        TorchRecurrentSnapshot(
            1,
            policy,
            {key: value.detach().clone() for key, value in parameters.items()},
            deepcopy(optimizer.state_dict()),
            {
                key: None if value.grad is None else value.grad.detach().clone()
                for key, value in parameters.items()
            },
            updates,
            examples,
        )
    )


def _restore(snapshot: TorchRecurrentSnapshot) -> tuple[dict[str, Any], Any]:
    parameters = {
        name: snapshot.parameters[name].detach().clone().requires_grad_(True) for name in _NAMES
    }
    optimizer = _optimizer(parameters, snapshot.policy)
    optimizer.load_state_dict(deepcopy(snapshot.optimizer_state))
    for name, parameter in parameters.items():
        gradient = snapshot.gradients[name]
        parameter.grad = None if gradient is None else gradient.detach().clone()
    return parameters, optimizer


def _features(features: Array, policy: TorchRecurrentPolicy, *, single: bool = False) -> Any:
    if (
        type(features) is not np.ndarray
        or features.dtype != np.float64
        or features.ndim != 2
        or features.shape[1] != policy.feature_dim
        or not 1 <= features.shape[0] <= (1 if single else policy.max_episode_steps)
        or not np.isfinite(features).all()
    ):
        raise ValueError("features must be finite exact float64 episode rows with bound geometry")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = features.astype(np.float32)
    if not np.isfinite(converted).all():
        raise ValueError("features must remain finite as float32")
    return require_torch().from_numpy(converted)


def _forward(
    parameters: dict[str, Any], features: Any, hidden: Any, *, check_finite: bool = False
) -> tuple[Any, Any]:
    if type(check_finite) is not bool:
        raise ValueError("check_finite must be an exact bool")
    torch = require_torch()
    outputs = []
    for row in features:
        linear = (
            row.reshape(1, -1) @ parameters["weight_input"]
            + hidden @ parameters["weight_recurrent"]
            + parameters["bias_hidden"]
        )
        # Diagnostic callers must observe affine overflow before tanh can hide
        # it behind finite saturation. The default retains native training math.
        if check_finite and not bool(torch.isfinite(linear).all().item()):
            raise FloatingPointError("nonfinite recurrent hidden preactivation")
        hidden = torch.tanh(linear)
        outputs.append(hidden @ parameters["weight_output"] + parameters["bias_output"])
    return torch.cat(outputs, dim=0), hidden


def _probabilities(logits: Any) -> Array:
    torch = require_torch()
    values = torch.softmax(logits, dim=1)
    if not bool(torch.isfinite(logits).all().item()) or not bool(
        torch.isfinite(values).all().item()
    ):
        raise FloatingPointError("nonfinite recurrent prediction")
    return values.detach().cpu().numpy().astype(np.float64, copy=True)


class TorchRecurrentLearner:
    """Mean episode CE/BPTT with atomic native SGD and no persistent activation."""

    def __init__(self, policy: TorchRecurrentPolicy) -> None:
        self._policy = _policy(policy)
        torch = require_torch()
        generator = torch.Generator(device="cpu").manual_seed(self._policy.seed)
        parameters = {
            name: (
                torch.zeros(shape, dtype=torch.float32, device="cpu")
                if name.startswith("bias")
                else torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")
                * 0.1
            ).requires_grad_(True)
            for name, shape in _shapes(self._policy).items()
        }
        optimizer = _optimizer(parameters, self._policy)
        _capture(parameters, optimizer, self._policy, 0, 0)
        self._payload = (parameters, optimizer, 0, 0)

    @property
    def policy(self) -> TorchRecurrentPolicy:
        return _policy(self._policy)

    @property
    def completed_updates(self) -> int:
        return self._payload[2]

    @property
    def training_examples(self) -> int:
        return self._payload[3]

    def snapshot_state(self) -> TorchRecurrentSnapshot:
        p, optimizer, updates, examples = self._payload
        return _capture(p, optimizer, self.policy, updates, examples)

    def restore_state(self, snapshot: TorchRecurrentSnapshot) -> None:
        owned = _snapshot(snapshot)
        if owned.policy != self.policy:
            raise ValueError("snapshot policy differs from this learner")
        parameters, optimizer = _restore(owned)
        self._payload = (parameters, optimizer, owned.completed_updates, owned.training_examples)

    def fork(self) -> "TorchRecurrentLearner":
        snapshot = self.snapshot_state()
        learner = TorchRecurrentLearner(snapshot.policy)
        learner.restore_state(snapshot)
        return learner

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        policy = self.policy
        values = _features(features, policy)
        if (
            type(targets) is not np.ndarray
            or targets.dtype != np.float64
            or targets.shape != (len(features), 1)
            or not np.isfinite(targets).all()
            or not ((targets >= 0) & (targets < policy.num_classes)).all()
            or not (targets == np.floor(targets)).all()
        ):
            raise ValueError("targets must be finite exact float64 Tx1 hard class indices")
        saved = self.snapshot_state()
        updates, examples = saved.completed_updates + 1, saved.training_examples + len(features)
        _count(updates, "completed_updates")
        _count(examples, "training_examples")
        parameters, optimizer = _restore(saved)
        torch = require_torch()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = _forward(
            parameters,
            values,
            torch.zeros((1, policy.hidden_dim), dtype=torch.float32, device="cpu"),
        )
        loss = torch.nn.functional.cross_entropy(
            logits, torch.from_numpy(targets[:, 0].astype(np.int64))
        )
        diagnostic = TrainingDiagnostic(TORCH_RECURRENT_CE_ID, float(loss.item()))
        loss.backward()
        optimizer.step()
        _capture(parameters, optimizer, policy, updates, examples)
        self._payload = (parameters, optimizer, updates, examples)
        return diagnostic

    def train_episode(self, episode: DemonstrationEpisode) -> TrainingDiagnostic:
        owned = copy_episode(episode)
        if owned.role != "train" or owned.num_classes != self.policy.num_classes:
            raise ValueError("only bound-class train-role demonstrations may train")
        return self.train_batch(*episode_arrays(owned))

    def train_reward_episode(
        self,
        episode: PolicyGradientEpisode | LegalPolicyGradientEpisode,
        behavior_state: TorchRecurrentSnapshot,
        *,
        discount: float = 1.0,
    ) -> TrainingDiagnostic:
        """One explicit on-policy reward update; supervised array semantics stay CE."""
        from src.adapters.torch_recurrent_reward import train_reward_episode

        return train_reward_episode(self, episode, behavior_state, discount=discount)

    def train_weighted_episode(
        self, episode: DemonstrationEpisode, class_weights: tuple[float, ...]
    ) -> TrainingDiagnostic:
        """One explicit weighted CE update; ordinary array/episode training stays mean CE."""
        from src.adapters.torch_recurrent_weighted import train_weighted_episode

        return train_weighted_episode(self, episode, class_weights)

    def train_zero_loss_episode(self, episode: PolicyGradientEpisode) -> TrainingDiagnostic:
        """Explicit momentum control; original reward and CE objectives stay unchanged."""
        from src.adapters.torch_recurrent_zero_loss import train_zero_loss_episode

        return train_zero_loss_episode(self, episode)

    def predict(self, features: Array) -> Array:
        policy = self.policy
        values = _features(features, policy)
        self.snapshot_state()
        torch = require_torch()
        with torch.no_grad():
            logits, _ = _forward(
                self._payload[0],
                values,
                torch.zeros((1, policy.hidden_dim), dtype=torch.float32, device="cpu"),
            )
            return _probabilities(logits)

    def start_episode(self) -> "RecurrentEpisodePredictor":
        return RecurrentEpisodePredictor(self)


class RecurrentEpisodePredictor:
    """One owned bounded rollout; hidden activation is separate from learned state."""

    def __init__(self, learner: TorchRecurrentLearner) -> None:
        # Constructor also owns its argument: direct construction cannot share a
        # trainable owner. start_episode avoids any dependence on caller discipline.
        self._learner = learner.fork()
        self._frozen = self._learner.snapshot_state()
        self._hidden = require_torch().zeros(
            (1, self._learner.policy.hidden_dim), dtype=require_torch().float32, device="cpu"
        )
        self._steps = 0

    @property
    def completed_steps(self) -> int:
        return self._steps

    def snapshot_state(self) -> TorchRecurrentSnapshot:
        """Actual frozen learned state only; activation is intentionally separate."""
        current = self._learner.snapshot_state()
        before = self._frozen
        if (
            current.policy != before.policy
            or current.completed_updates != before.completed_updates
            or current.training_examples != before.training_examples
            or not _equal_tensors(current.parameters, before.parameters)
            or not _equal_tensors(current.gradients, before.gradients)
            or not _equal_tensors(current.optimizer_state["state"], before.optimizer_state["state"])
        ):
            raise ValueError("episode learned/optimizer state changed")
        return current

    def predict(self, features: Array) -> Array:
        policy = self._learner.policy
        values = _features(features, policy, single=True)
        if type(self._steps) is not int or not 0 <= self._steps < policy.max_episode_steps:
            raise ValueError("episode prediction limit reached or counter invalid")
        _tensor(self._hidden, (1, policy.hidden_dim), "episode activation")
        self.snapshot_state()
        torch = require_torch()
        with torch.no_grad():
            logits, hidden = _forward(
                self._learner._payload[0], values, self._hidden.detach().clone()
            )
            result = _probabilities(logits)
            _tensor(hidden, (1, policy.hidden_dim), "next episode activation")
        self.snapshot_state()
        self._hidden, self._steps = hidden.detach().clone(), self._steps + 1
        return result


def _equal_tensors(left: Any, right: Any) -> bool:
    """Compare the already validated tensor mappings without weakening state guards."""
    torch = require_torch()
    if type(left) is dict and type(right) is dict:
        return left.keys() == right.keys() and all(_equal_tensors(left[k], right[k]) for k in left)
    if left is None or right is None:
        return left is right
    return bool(torch.equal(left, right))
