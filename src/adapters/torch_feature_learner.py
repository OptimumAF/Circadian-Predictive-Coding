"""Owned CPU Torch heads behind the existing NumPy learner port.

Inputs are bounded float64 feature arrays and hard class-index labels. Snapshots own
parameters, native adaptive/RNG state, optimizer momentum and gradient buffers.
Training stages an independent head before publishing one complete payload, so
failed native work cannot partly change this learner. No sleep, backbone, device
selection, application budget or durable whole-stream recovery is provided.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from math import isfinite
from typing import Any

import numpy as np

from src.core.coupled_energy import Array
from src.core.learner_ports import (
    TrainingDiagnostic,
    validate_learning_rate_multiplier,
    validate_gate_setting,
)
from src.core.resnet50_variants import (
    TORCH_PC_ENERGY_ID,
    BackpropMLPHead,
    CircadianHeadConfig,
    CircadianPredictiveCodingHead,
    PredictiveCodingHead,
)
from src.shared.torch_runtime import require_torch

TORCH_CE_ID = "torch_feature_cross_entropy_mean_v1"
_PARAMETERS = ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output")
_MAX_COUNT = 2**53 - 1


def _number(value: Any, name: str, *, positive: bool = True) -> None:
    try:
        valid = type(value) in {int, float} and isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (value <= 0 if positive else not 0 <= value <= 1):
        raise ValueError(f"{name} must be native finite {'positive' if positive else 'in [0,1]'}")


@dataclass(frozen=True)
class TorchFeaturePolicy:
    method: str
    feature_dim: int
    hidden_dim: int
    seed: int
    learning_rate: float = 0.03
    inference_steps: int = 2
    inference_learning_rate: float = 0.04
    momentum: float = 0.9
    num_classes: int = 2

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method not in {
            "bp",
            "pc",
            "neutral_cpc",
            "gated_cpc",
        }:
            raise ValueError("method must be bp, pc, neutral_cpc or gated_cpc")
        # Exact MiniGrid public-frame widths admit the raw view and initial-frame
        # history; hidden/action limits and native algorithms retain their bounds.
        if type(self.feature_dim) is not int or not (
            1 <= self.feature_dim <= 16 or self.feature_dim in (148, 296)
        ):
            raise ValueError("feature_dim must be a native integer in 1..16, exactly 148 or 296")
        for name, maximum, minimum in (
            ("hidden_dim", 32, 1),
            ("seed", 2**31 - 1, 0),
            ("inference_steps", 64, 1),
            ("num_classes", 16, 2),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be a native integer in {minimum}..{maximum}")
        _number(self.learning_rate, "learning_rate")
        _number(self.inference_learning_rate, "inference_learning_rate")
        _number(self.momentum, "momentum", positive=False)


def _policy(policy: TorchFeaturePolicy) -> TorchFeaturePolicy:
    if type(policy) is not TorchFeaturePolicy or set(vars(policy)) != {
        item.name for item in fields(TorchFeaturePolicy)
    }:
        raise ValueError("an exact complete TorchFeaturePolicy is required")
    return replace(policy)


def _config(policy: TorchFeaturePolicy) -> CircadianHeadConfig:
    return (
        CircadianHeadConfig.matched_pc_control()
        if policy.method == "neutral_cpc"
        else CircadianHeadConfig()
    )


def _tensor(value: Any, shape: tuple[int, ...], dtype: Any, name: str) -> None:
    torch = require_torch()
    if (
        type(value) is not torch.Tensor
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != torch.device("cpu")
    ):
        raise ValueError(f"snapshot {name} has incompatible CPU tensor geometry or dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"snapshot {name} must be finite")


def _parameter_shapes(policy: TorchFeaturePolicy) -> dict[str, tuple[int, ...]]:
    d, h, c = policy.feature_dim, policy.hidden_dim, policy.num_classes
    return dict(zip(_PARAMETERS, ((d, h), (1, h), (h, c), (1, c)), strict=True))


def _count(value: Any, name: str) -> None:
    if type(value) is not int or not 0 <= value <= _MAX_COUNT:
        raise ValueError(f"snapshot {name} must be a bounded nonnegative native integer")


def _cpc_state(
    state: dict[str, Any], policy: TorchFeaturePolicy, updates: int, examples: int
) -> None:
    torch = require_torch()
    tensors = CircadianPredictiveCodingHead._SNAPSHOT_TENSORS
    counters = CircadianPredictiveCodingHead._SNAPSHOT_COUNTERS
    metadata = {
        "format_version",
        "feature_dim",
        "num_classes",
        "min_hidden_dim",
        "max_hidden_dim",
        "device",
        "generator_device",
        "config",
        "split_generator_state",
        "energy_history",
        "reward_error_ema",
        "last_reward_scale",
    }
    if state.keys() != set(tensors) | set(counters) | metadata:
        raise ValueError("snapshot CPC head fields are incompatible")
    expected = dict(
        format_version=2,
        feature_dim=policy.feature_dim,
        num_classes=policy.num_classes,
        min_hidden_dim=policy.hidden_dim,
        max_hidden_dim=policy.hidden_dim,
        device="cpu",
        generator_device="cpu",
    )
    for name, value in expected.items():
        if type(state[name]) is not type(value) or state[name] != value:
            raise ValueError(f"snapshot CPC {name} is incompatible")
    config = asdict(_config(policy))
    if (
        type(state["config"]) is not dict
        or state["config"].keys() != config.keys()
        or any(
            type(state["config"][name]) is not type(value) or state["config"][name] != value
            for name, value in config.items()
        )
    ):
        raise ValueError("snapshot CPC config differs from the bound policy")
    shapes = _parameter_shapes(policy)
    for name in tensors:
        dtype = (
            torch.int64
            if name in {"neuron_ids", "parent_ids"}
            else torch.int32
            if name in {"split_cooldown", "prune_cooldown"}
            else torch.float32
        )
        _tensor(state[name], shapes.get(name, (policy.hidden_dim,)), dtype, name)
        if (
            name not in shapes
            and name not in {"parent_ids"}
            and bool((state[name] < 0).any().item())
        ):
            raise ValueError(f"snapshot CPC {name} must be nonnegative")
    expected_counters = dict(
        traffic_steps=updates,
        wake_examples=examples,
        sleep_events=0,
        steps_since_sleep=updates,
        next_neuron_id=policy.hidden_dim,
    )
    for name, expected_count in expected_counters.items():
        _count(state[name], name)
        if state[name] != expected_count:
            raise ValueError(f"snapshot CPC {name} differs from native update clocks")
    # Native no-sleep ages add float32 one each call, which saturates at 2**24;
    # retain that existing arithmetic rather than inventing a new clock limit.
    if not bool((state["neuron_age"] == min(updates, 2**24)).all().item()):
        raise ValueError("snapshot CPC neuron ages differ from native update clocks")
    if not torch.equal(
        state["neuron_ids"], torch.arange(policy.hidden_dim, dtype=torch.int64)
    ) or not bool((state["parent_ids"] == -1).all().item()):
        raise ValueError("snapshot CPC lineage differs from fixed no-sleep topology")
    history = state["energy_history"]
    max_history = max(_config(policy).sleep_energy_window * 4, 16)
    if (
        type(history) is not list
        or len(history) != min(updates, max_history)
        or any(type(value) is not float or not isfinite(value) or value < 0 for value in history)
    ):
        raise ValueError("snapshot CPC energy history is incompatible")
    reward, scale = state["reward_error_ema"], state["last_reward_scale"]
    if reward is not None and (type(reward) is not float or not isfinite(reward) or reward < 0):
        raise ValueError("snapshot CPC reward baseline must be finite or None")
    if type(scale) is not float or not isfinite(scale) or scale <= 0:
        raise ValueError("snapshot CPC reward scale must be positive finite")
    generator = state["split_generator_state"]
    if (
        type(generator) is not torch.Tensor
        or generator.dtype != torch.uint8
        or generator.device != torch.device("cpu")
        or generator.ndim != 1
        or not 1 <= generator.numel() <= 16384
    ):
        raise ValueError("snapshot CPC split generator state is incompatible")
    try:
        torch.Generator(device="cpu").set_state(generator)
    except (RuntimeError, ValueError) as error:
        raise ValueError("snapshot CPC split generator state is invalid") from error


def _head_state(state: Any, policy: TorchFeaturePolicy, updates: int, examples: int) -> None:
    if type(state) is not dict:
        raise ValueError("snapshot head_state must be an exact dictionary")
    if policy.method in {"neutral_cpc", "gated_cpc"}:
        _cpc_state(state, policy, updates, examples)
        return
    torch = require_torch()
    expected = dict(
        format_version=1,
        feature_dim=policy.feature_dim,
        hidden_dim=policy.hidden_dim,
        num_classes=policy.num_classes,
        device="cpu",
    )
    extra = {"traffic_sum", "traffic_steps"} if policy.method == "pc" else set()
    if state.keys() != set(expected) | set(_PARAMETERS) | extra:
        raise ValueError("snapshot head fields are incompatible")
    for name, value in expected.items():
        if type(state[name]) is not type(value) or state[name] != value:
            raise ValueError(f"snapshot head {name} is incompatible")
    for name, shape in _parameter_shapes(policy).items():
        _tensor(state[name], shape, torch.float32, name)
    if policy.method == "pc":
        _tensor(state["traffic_sum"], (policy.hidden_dim,), torch.float32, "traffic_sum")
        _count(state["traffic_steps"], "traffic_steps")
        if state["traffic_steps"] != updates or bool((state["traffic_sum"] < 0).any().item()):
            raise ValueError("snapshot PC traffic is incompatible with update clock")


def _optimizer_state(state: Any, gradients: Any, policy: TorchFeaturePolicy, updates: int) -> None:
    if policy.method != "bp":
        if state is not None or gradients is not None:
            raise ValueError("snapshot PC/CPC optimizer and gradients must be None")
        return
    torch = require_torch()
    if type(gradients) is not dict or gradients.keys() != set(_PARAMETERS):
        raise ValueError("snapshot BP gradient fields are incompatible")
    for name, shape in _parameter_shapes(policy).items():
        value = gradients[name]
        if value is None and updates == 0:
            continue
        _tensor(value, shape, torch.float32, f"gradient {name}")
    if type(state) is not dict or state.keys() != {"state", "param_groups"}:
        raise ValueError("snapshot BP optimizer fields are incompatible")
    groups = state["param_groups"]
    if type(groups) is not list or len(groups) != 1 or type(groups[0]) is not dict:
        raise ValueError("snapshot BP optimizer requires one parameter group")
    expected = dict(
        lr=float(policy.learning_rate),
        momentum=float(policy.momentum),
        dampening=0,
        weight_decay=0,
        nesterov=False,
        maximize=False,
        foreach=None,
        differentiable=False,
        fused=None,
        params=[0, 1, 2, 3],
    )
    group = groups[0]
    if group.keys() != expected.keys() or any(
        type(group[name]) is not type(value) or group[name] != value
        for name, value in expected.items()
    ):
        raise ValueError("snapshot BP optimizer parameter-group policy differs")
    if any(type(index) is not int for index in group["params"]):
        raise ValueError("snapshot BP optimizer parameter IDs must be native integers")
    buffers = state["state"]
    expected_ids = set(range(4)) if updates > 0 and policy.momentum > 0 else set()
    if (
        type(buffers) is not dict
        or len(buffers) != len(expected_ids)
        or set(buffers) != expected_ids
        or any(type(k) is not int for k in buffers)
    ):
        raise ValueError("snapshot BP optimizer momentum membership is incompatible")
    for index, name in enumerate(_PARAMETERS):
        if index not in buffers:
            continue
        value = buffers[index]
        if type(value) is not dict or value.keys() != {"momentum_buffer"}:
            raise ValueError("snapshot BP momentum fields are incompatible")
        _tensor(
            value["momentum_buffer"],
            _parameter_shapes(policy)[name],
            torch.float32,
            "momentum_buffer",
        )


@dataclass(frozen=True)
class TorchFeatureSnapshot:
    format_version: int
    policy: TorchFeaturePolicy
    head_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    gradients: dict[str, Any] | None
    completed_updates: int
    training_examples: int

    def __post_init__(self) -> None:
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError("Torch feature snapshot format must be native integer 1")
        policy = _policy(self.policy)
        _count(self.completed_updates, "completed_updates")
        _count(self.training_examples, "training_examples")
        if not self.completed_updates <= self.training_examples <= 128 * self.completed_updates:
            raise ValueError("snapshot training examples differ from bounded native updates")
        _head_state(self.head_state, policy, self.completed_updates, self.training_examples)
        _optimizer_state(self.optimizer_state, self.gradients, policy, self.completed_updates)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "head_state", deepcopy(self.head_state))
        object.__setattr__(self, "optimizer_state", deepcopy(self.optimizer_state))
        object.__setattr__(self, "gradients", deepcopy(self.gradients))


def _snapshot(snapshot: TorchFeatureSnapshot) -> TorchFeatureSnapshot:
    if type(snapshot) is not TorchFeatureSnapshot or set(vars(snapshot)) != {
        item.name for item in fields(TorchFeatureSnapshot)
    }:
        raise ValueError("an exact complete TorchFeatureSnapshot is required")
    return replace(snapshot)


def _components(policy: TorchFeaturePolicy) -> tuple[Any, Any]:
    torch = require_torch()
    args = dict(
        feature_dim=policy.feature_dim,
        hidden_dim=policy.hidden_dim,
        num_classes=policy.num_classes,
        device=torch.device("cpu"),
        seed=policy.seed,
    )
    if policy.method == "bp":
        head = BackpropMLPHead(**args)
        return head, torch.optim.SGD(
            head.trainable_parameters(),
            lr=float(policy.learning_rate),
            momentum=float(policy.momentum),
        )
    if policy.method == "pc":
        return PredictiveCodingHead(**args), None
    return CircadianPredictiveCodingHead(
        **args,
        config=_config(policy),
        min_hidden_dim=policy.hidden_dim,
        max_hidden_dim=policy.hidden_dim,
    ), None


def _state(
    head: Any, optimizer: Any, policy: TorchFeaturePolicy, updates: int, examples: int
) -> TorchFeatureSnapshot:
    torch = require_torch()
    kind = (
        BackpropMLPHead
        if policy.method == "bp"
        else PredictiveCodingHead
        if policy.method == "pc"
        else CircadianPredictiveCodingHead
    )
    if (
        type(head) is not kind
        or type(head.feature_dim) is not int
        or head.feature_dim != policy.feature_dim
        or type(head.num_classes) is not int
        or head.num_classes != policy.num_classes
        or head.device != torch.device("cpu")
    ):
        raise ValueError("native head metadata differs from the bound Torch feature policy")
    if isinstance(head, CircadianPredictiveCodingHead):
        state = head.snapshot_state()
    else:
        state = dict(
            format_version=1,
            feature_dim=policy.feature_dim,
            hidden_dim=policy.hidden_dim,
            num_classes=policy.num_classes,
            device="cpu",
        )
        state.update({name: getattr(head, name).detach().clone() for name in _PARAMETERS})
        if isinstance(head, PredictiveCodingHead):
            state.update(
                traffic_sum=head._traffic_sum.detach().clone(), traffic_steps=head._traffic_steps
            )
    gradients = (
        {
            name: None
            if getattr(head, name).grad is None
            else getattr(head, name).grad.detach().clone()
            for name in _PARAMETERS
        }
        if policy.method == "bp"
        else None
    )
    return TorchFeatureSnapshot(
        1,
        policy,
        state,
        None if optimizer is None else optimizer.state_dict(),
        gradients,
        updates,
        examples,
    )


def _restore_components(snapshot: TorchFeatureSnapshot) -> tuple[Any, Any]:
    policy = snapshot.policy
    head, optimizer = _components(policy)
    if policy.method in {"neutral_cpc", "gated_cpc"}:
        head.restore_state(snapshot.head_state)
    elif policy.method == "pc":
        for name in _PARAMETERS:
            setattr(head, name, snapshot.head_state[name].detach().clone())
        head._traffic_sum = snapshot.head_state["traffic_sum"].detach().clone()
        head._traffic_steps = snapshot.head_state["traffic_steps"]
    else:
        torch = require_torch()
        with torch.no_grad():
            for name in _PARAMETERS:
                getattr(head, name).copy_(snapshot.head_state[name])
        optimizer.load_state_dict(deepcopy(snapshot.optimizer_state))
        for name in _PARAMETERS:
            value = snapshot.gradients[name] if snapshot.gradients is not None else None
            getattr(head, name).grad = None if value is None else value.detach().clone()
    return head, optimizer


def _features(features: Array, width: int) -> Any:
    if (
        type(features) is not np.ndarray
        or features.dtype != np.float64
        or features.ndim != 2
        or not 1 <= features.shape[0] <= 128
        or features.shape[1] != width
    ):
        raise ValueError("features must be exact float64 arrays with 1..128 rows and bound width")
    if not np.isfinite(features).all():
        raise ValueError("features must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = features.astype(np.float32)
    if not np.isfinite(converted).all():
        raise ValueError("features must remain finite after float32 conversion")
    return require_torch().from_numpy(converted)


class TorchFeatureLearner:
    """Fixed-policy native heads; only complete owned candidates become live."""

    def __init__(self, policy: TorchFeaturePolicy) -> None:
        self._policy = _policy(policy)
        head, optimizer = _components(self._policy)
        _state(head, optimizer, self._policy, 0, 0)
        self._payload = (head, optimizer, 0, 0)

    @property
    def policy(self) -> TorchFeaturePolicy:
        return _policy(self._policy)

    @property
    def _head(self) -> Any:
        return self._payload[0]

    @property
    def _optimizer(self) -> Any:
        return self._payload[1]

    @property
    def completed_updates(self) -> int:
        return self._payload[2]

    @property
    def training_examples(self) -> int:
        return self._payload[3]

    def snapshot_state(self) -> TorchFeatureSnapshot:
        policy = self.policy
        return _state(
            self._head, self._optimizer, policy, self.completed_updates, self.training_examples
        )

    def restore_state(self, snapshot: TorchFeatureSnapshot) -> None:
        owned = _snapshot(snapshot)
        if owned.policy != self.policy:
            raise ValueError("snapshot policy differs from this learner")
        head, optimizer = _restore_components(owned)
        self._payload = (head, optimizer, owned.completed_updates, owned.training_examples)

    def fork(self) -> "TorchFeatureLearner":
        snapshot = self.snapshot_state()
        learner = TorchFeatureLearner(snapshot.policy)
        learner.restore_state(snapshot)
        return learner

    def train_batch(self, features: Array, targets: Array) -> TrainingDiagnostic:
        return self._train_batch(features, targets, 1.0)

    def train_batch_with_rate_multiplier(
        self, features: Array, targets: Array, multiplier: float
    ) -> TrainingDiagnostic:
        """Apply one bounded effect on staged state; never change the base policy.

        Why: optimizer LR is pinned by complete snapshot validation. Restore the
        base rate before publishing while retaining the real momentum/gradients.
        Unity dispatches the original public port, including subclass observers.
        """
        validate_learning_rate_multiplier(multiplier)
        if multiplier == 1:
            return self.train_batch(features, targets)
        return self._train_batch(features, targets, multiplier)

    def _train_batch(
        self, features: Array, targets: Array, multiplier: float, gate_setting: str = "unchanged"
    ) -> TrainingDiagnostic:
        policy = self.policy
        effective_rate = policy.learning_rate * multiplier
        _number(effective_rate, "effective learning rate")
        values = _features(features, policy.feature_dim)
        if (
            type(targets) is not np.ndarray
            or targets.dtype != np.float64
            or targets.shape != (features.shape[0], 1)
            or not np.isfinite(targets).all()
            or not ((targets >= 0) & (targets < policy.num_classes)).all()
            or not (targets == np.floor(targets)).all()
        ):
            raise ValueError(
                f"targets must be finite exact float64 Bx1 hard class indices in [0, {policy.num_classes})"
            )
        torch = require_torch()
        labels = torch.from_numpy(targets[:, 0].astype(np.int64))
        saved = self.snapshot_state()
        _count(saved.completed_updates + 1, "completed_updates")
        _count(saved.training_examples + features.shape[0], "training_examples")
        head, optimizer = _restore_components(saved)
        if policy.method == "bp":
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head.forward_logits(values), labels)
            diagnostic = TrainingDiagnostic(TORCH_CE_ID, float(loss.item()))
            loss.backward()
            optimizer.param_groups[0]["lr"] = float(effective_rate)
            optimizer.step()
            optimizer.param_groups[0]["lr"] = float(policy.learning_rate)
        else:
            energy = head.train_step(
                values,
                labels,
                effective_rate,
                policy.inference_steps,
                policy.inference_learning_rate,
                **({} if gate_setting == "unchanged" else {"gate_setting": gate_setting}),
            )
            diagnostic = TrainingDiagnostic(TORCH_PC_ENERGY_ID, energy)
        updates, examples = saved.completed_updates + 1, saved.training_examples + features.shape[0]
        _state(head, optimizer, policy, updates, examples)
        self._payload = (head, optimizer, updates, examples)
        return diagnostic

    def validate_gate_setting(self, setting: str) -> None:
        """Validate scalar capability without reading/staging model payloads."""
        validate_gate_setting(setting)
        if setting == "unchanged":
            return
        if self.policy.method not in {"neutral_cpc", "gated_cpc"} or (
            setting == "active" and self.policy.method != "gated_cpc"
        ):
            raise ValueError("gate setting is unsupported by the bound native policy")

    def train_batch_with_gate_setting(
        self, features: Array, targets: Array, setting: str
    ) -> TrainingDiagnostic:
        """One staged plasticity effect; chemistry and the base config stay native."""
        self.validate_gate_setting(setting)
        if setting == "unchanged":
            return self.train_batch(features, targets)
        return self._train_batch(features, targets, 1.0, setting)

    def train_batch_with_rate_and_gate(
        self, features: Array, targets: Array, multiplier: float, setting: str
    ) -> TrainingDiagnostic:
        """Compose temporary effects in the existing single staged native step.

        Why: sequential rate and gate calls would apply two gradients and spend
        two updates. Preserve the original public dispatch for degenerate cases.
        """
        validate_learning_rate_multiplier(multiplier)
        self.validate_gate_setting(setting)
        if setting == "unchanged":
            return self.train_batch_with_rate_multiplier(features, targets, multiplier)
        if multiplier == 1:
            return self.train_batch_with_gate_setting(features, targets, setting)
        return self._train_batch(features, targets, multiplier, setting)

    def predict_logits(self, features: Array) -> Array:
        """Return owned finite BxK raw logits, before any candidate masking.

        Why: a typed caller must mask logits before softmax; globally normalized
        probabilities can underflow all permitted actions to zero. Keep predict's
        established float32 probability arithmetic and binary shape unchanged.
        """
        policy = self.policy
        values = _features(features, policy.feature_dim)
        self.snapshot_state()
        torch = require_torch()
        with torch.no_grad():
            logits = (
                self._head.forward_logits(values)
                if policy.method == "bp"
                else self._head.predict_logits(values)
            )
            _tensor(logits, (features.shape[0], policy.num_classes), torch.float32, "logits")
            return logits.cpu().numpy().astype(np.float64, copy=True)

    def predict(self, features: Array) -> Array:
        """Return owned Bx1 class-one probabilities for K=2, otherwise full BxK.

        Why: preserve the existing binary stream contract while categorical
        callers receive every action probability without implicit action choice.
        """
        policy = self.policy
        values = _features(features, policy.feature_dim)
        self.snapshot_state()  # Refuse corrupt complete state before native prediction.
        torch = require_torch()
        with torch.no_grad():
            logits = (
                self._head.forward_logits(values)
                if policy.method == "bp"
                else self._head.predict_logits(values)
            )
            if tuple(logits.shape) != (features.shape[0], policy.num_classes):
                raise ValueError("native logits differ from bound batch/class geometry")
            probabilities = torch.softmax(logits, dim=1)
            if policy.num_classes == 2:
                probabilities = probabilities[:, 1:2]
            if not bool(torch.isfinite(logits).all().item()) or not bool(
                torch.isfinite(probabilities).all().item()
            ):
                raise FloatingPointError("nonfinite Torch feature prediction")
            return probabilities.cpu().numpy().astype(np.float64, copy=True)
