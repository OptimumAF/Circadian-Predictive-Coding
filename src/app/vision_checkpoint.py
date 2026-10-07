"""Identity and runner state for trusted unmatched-vision checkpoints.

The runner supplies completed model outcomes and the three development
loaders. This module hashes their raw labeled examples without evaluating
augmentations or opening final test. File encoding belongs to infra.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from json import dumps
import random
from typing import Any, Protocol

import numpy as np

from src.core.resnet50_variants import (
    BackpropResNet50Classifier,
    CircadianClassifierSnapshot,
    CircadianPredictiveCodingResNet50Classifier,
    PredictiveCodingResNet50Classifier,
)
from src.app.seeded_vision_loader import SeededEpochLoaderState
from src.app.shared_vision_loader import SharedEpochLoaderState
from src.app.sleep_schedule import SleepRollbackCooldownState
from src.core.sleep_telemetry import SleepEventTelemetry


@dataclass(frozen=True)
class BaselineVisionModelState:
    """Backbone and baseline output state without a live Torch module object."""

    backbone_state: dict[str, Any]
    backbone_training: dict[str, bool]
    backbone_requires_grad: dict[str, bool]
    output_state: dict[str, Any]
    output_training: dict[str, bool] | None
    output_requires_grad: dict[str, bool] | None


@dataclass(frozen=True)
class CompletedVisionModelState:
    """Tagged detached state for one fully trained image classifier."""

    format_version: int
    variant: str
    state: BaselineVisionModelState | CircadianClassifierSnapshot


_PREDICTIVE_HEAD_FIELDS = (
    "weight_feature_hidden",
    "bias_hidden",
    "weight_hidden_output",
    "bias_output",
    "_traffic_sum",
    "_traffic_steps",
)


@dataclass(frozen=True)
class VisionRunnerCheckpoint:
    """Detached completed-model progress for one unmatched benchmark run."""

    format_version: int
    protocol_id: str
    config_digest: str
    data_digest: str
    training_order: tuple[str, ...]
    next_variant_index: int
    completed_outcomes: tuple[Any, ...]
    completed_hashes: tuple[str, ...]
    python_random_state: Any
    numpy_random_state: Any
    torch_cpu_random_state: Any
    active_circadian: VisionCircadianProgress | None = None
    shared_train_generator_state: Any | None = None
    torch_cuda_device: str | None = None
    torch_cuda_random_state: Any | None = None


@dataclass(frozen=True)
class VisionCircadianProgress:
    """Classifier and runner state at a complete wake or sleep boundary."""

    stage: str
    completed_epoch: int
    next_batch_index: int
    classifier_state: CircadianClassifierSnapshot
    loader_state: SeededEpochLoaderState | SharedEpochLoaderState
    retry_state: SleepRollbackCooldownState
    outer_entry_torch_state: Any
    initial_hidden_dim: int
    epochs_ran: int
    wake_batches: int
    seen_samples: int
    sleep_attempts: int
    sleep_rollbacks: int
    sleep_splits: int
    sleep_prunes: int
    final_energy: float
    step_times_ms: tuple[float, ...]
    elapsed_seconds: float
    outer_entry_torch_cuda_state: Any | None = None
    sleep_events: tuple[SleepEventTelemetry, ...] = ()


class VisionCheckpointStore(Protocol):
    """App port for one replaceable trusted local vision checkpoint."""

    def load(self) -> VisionRunnerCheckpoint: ...

    def save(self, checkpoint: VisionRunnerCheckpoint) -> None: ...


@dataclass(frozen=True)
class VisionCircadianCheckpointContext:
    """Immutable run identity and prior outcomes while circadian trains."""

    store: VisionCheckpointStore
    config: Any
    data_digest: str
    training_order: tuple[str, ...]
    completed_outcomes: tuple[Any, ...]
    completed_hashes: tuple[str, ...]
    outer_entry_torch_state: Any
    outer_entry_torch_cuda_state: Any | None = None
    resume_checkpoint: VisionRunnerCheckpoint | None = None


def snapshot_completed_vision_outcome(outcome: Any, variant: str) -> Any:
    """Detach a completed model while retaining its report progress."""
    model = outcome.model
    if variant == "circadian":
        state = model.snapshot_full_state()
    elif variant in {"backprop", "predictive"}:
        output = model.classifier if variant == "backprop" else model.head
        output_state = (
            deepcopy(output.state_dict())
            if variant == "backprop"
            else {name: deepcopy(getattr(output, name)) for name in _PREDICTIVE_HEAD_FIELDS}
        )
        state = BaselineVisionModelState(
            backbone_state=deepcopy(model.backbone.state_dict()),
            backbone_training=_module_training(model.backbone),
            backbone_requires_grad=_module_requires_grad(model.backbone),
            output_state=output_state,
            output_training=_module_training(output) if variant == "backprop" else None,
            output_requires_grad=(_module_requires_grad(output) if variant == "backprop" else None),
        )
    else:
        raise ValueError("unknown vision checkpoint model variant")
    return replace(
        outcome, model=CompletedVisionModelState(format_version=1, variant=variant, state=state)
    )


def restore_completed_vision_outcome(
    stored: Any,
    *,
    variant: str,
    config: Any,
    device: Any,
    num_classes: int,
    circadian_config: Any,
) -> Any:
    """Construct and validate a fresh classifier before exposing an outcome."""
    snapshot = stored.model
    if (
        not isinstance(snapshot, CompletedVisionModelState)
        or type(snapshot.format_version) is not int
        or snapshot.format_version != 1
        or snapshot.variant != variant
    ):
        raise ValueError("incompatible vision checkpoint completed model snapshot")
    model: Any
    if variant == "backprop":
        model = BackpropResNet50Classifier(
            num_classes=num_classes,
            device=device,
            freeze_backbone=config.backprop_freeze_backbone,
            backbone_weights=config.backbone_weights,
        )
    elif variant == "predictive":
        model = PredictiveCodingResNet50Classifier(
            num_classes=num_classes,
            device=device,
            head_hidden_dim=config.predictive_head_hidden_dim,
            seed=config.seed + 11,
            freeze_backbone=True,
            backbone_weights=config.backbone_weights,
        )
    elif variant == "circadian":
        model = CircadianPredictiveCodingResNet50Classifier(
            num_classes=num_classes,
            device=device,
            head_hidden_dim=config.circadian_head_hidden_dim,
            seed=config.seed + 23,
            freeze_backbone=True,
            backbone_weights=config.backbone_weights,
            circadian_config=circadian_config,
            min_hidden_dim=config.circadian_min_hidden_dim,
            max_hidden_dim=config.circadian_max_hidden_dim,
        )
    else:
        raise ValueError("unknown vision checkpoint model variant")
    if variant == "circadian":
        if not isinstance(snapshot.state, CircadianClassifierSnapshot):
            raise ValueError("incompatible vision checkpoint full classifier snapshot")
        model.restore_full_state(snapshot.state)
    else:
        if not isinstance(snapshot.state, BaselineVisionModelState):
            raise ValueError("incompatible vision checkpoint baseline snapshot")
        _restore_module_state(
            model.backbone,
            snapshot.state.backbone_state,
            snapshot.state.backbone_training,
            snapshot.state.backbone_requires_grad,
        )
        if variant == "backprop":
            assert isinstance(model, BackpropResNet50Classifier)
            if (
                snapshot.state.output_training is None
                or snapshot.state.output_requires_grad is None
            ):
                raise ValueError("incompatible vision checkpoint classifier modes")
            _restore_module_state(
                model.classifier,
                snapshot.state.output_state,
                snapshot.state.output_training,
                snapshot.state.output_requires_grad,
            )
        else:
            assert isinstance(model, PredictiveCodingResNet50Classifier)
            if (
                snapshot.state.output_training is not None
                or snapshot.state.output_requires_grad is not None
            ):
                raise ValueError("incompatible vision checkpoint predictive head modes")
            _restore_predictive_head(model.head, snapshot.state.output_state)
    return replace(stored, model=model)


def _module_training(module: Any) -> dict[str, bool]:
    return {name: child.training for name, child in module.named_modules()}


def _module_requires_grad(module: Any) -> dict[str, bool]:
    return {name: parameter.requires_grad for name, parameter in module.named_parameters()}


def _restore_module_state(
    module: Any,
    state: dict[str, Any],
    training: dict[str, bool],
    requires_grad: dict[str, bool],
) -> None:
    current = module.state_dict()
    children = dict(module.named_modules())
    parameters = dict(module.named_parameters())
    if (
        not isinstance(state, dict)
        or state.keys() != current.keys()
        or not isinstance(training, dict)
        or training.keys() != children.keys()
        or not isinstance(requires_grad, dict)
        or requires_grad.keys() != parameters.keys()
        or any(type(value) is not bool for value in training.values())
        or any(type(value) is not bool for value in requires_grad.values())
    ):
        raise ValueError("incompatible vision checkpoint module state")
    for name, reference in current.items():
        value = state[name]
        if (
            not hasattr(value, "shape")
            or value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device != reference.device
        ):
            raise ValueError(f"incompatible vision checkpoint module tensor {name}")
    module.load_state_dict(deepcopy(state), strict=True)
    for name, child in children.items():
        child.training = training[name]
    for name, parameter in parameters.items():
        parameter.requires_grad_(requires_grad[name])


def _restore_predictive_head(head: Any, state: dict[str, Any]) -> None:
    if not isinstance(state, dict) or state.keys() != set(_PREDICTIVE_HEAD_FIELDS):
        raise ValueError("incompatible vision checkpoint predictive head fields")
    for name in _PREDICTIVE_HEAD_FIELDS:
        reference = getattr(head, name)
        value = state[name]
        if name == "_traffic_steps":
            if type(value) is not int or value < 0:
                raise ValueError("incompatible vision checkpoint predictive traffic count")
        elif (
            not hasattr(value, "shape")
            or value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device != reference.device
        ):
            raise ValueError(f"incompatible vision checkpoint predictive tensor {name}")
    for name in _PREDICTIVE_HEAD_FIELDS:
        setattr(head, name, deepcopy(state[name]))


def vision_config_digest(config: Any, training_order: tuple[str, ...]) -> str:
    """Bind the whole benchmark configuration and requested execution order."""
    encoded = dumps(
        {"config": asdict(config), "training_order": training_order},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def vision_development_data_digest(loaders: Any) -> str:
    """Hash exact raw train/guard/validation examples without touching test."""
    digest = sha256()
    for role in ("train", "guard", "validation"):
        loader = getattr(loaders, f"{role}_loader")
        split_hash = loaders.split_hashes.get(
            role, loaders.split_hashes.get("validation") if role == "guard" else None
        )
        if split_hash is None:
            raise ValueError(f"vision checkpoint requires a {role} split identity")
        digest.update(f"{role}:{split_hash}".encode("utf-8"))
        _hash_raw_dataset(digest, loader.dataset)
    return digest.hexdigest()


def _hash_raw_dataset(digest: Any, dataset: Any) -> None:
    indices = np.arange(len(dataset))
    while hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        indices = np.asarray(dataset.indices, dtype=np.int64)[indices]
        dataset = dataset.dataset
    if hasattr(dataset, "images") and hasattr(dataset, "labels"):
        images, labels = dataset.images, dataset.labels
    elif hasattr(dataset, "data") and hasattr(dataset, "targets"):
        images, labels = dataset.data, dataset.targets
    else:
        raise ValueError("vision checkpoint requires raw labeled development data")
    image_array = _as_cpu_array(images)
    label_array = _as_cpu_array(labels)
    if (
        len(image_array) != len(label_array)
        or np.any(indices < 0)
        or np.any(indices >= len(image_array))
    ):
        raise ValueError("vision checkpoint development dataset is incompatible")
    digest.update(
        dumps(
            {
                "type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
                "count": len(indices),
                "image_dtype": str(image_array.dtype),
                "image_shape": image_array.shape[1:],
                "label_dtype": str(label_array.dtype),
                "label_shape": label_array.shape[1:],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for start in range(0, len(indices), 1024):
        selected = indices[start : start + 1024]
        digest.update(np.ascontiguousarray(image_array[selected]).tobytes())
        digest.update(np.ascontiguousarray(label_array[selected]).tobytes())


def _as_cpu_array(value: Any) -> np.ndarray[Any, Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def capture_vision_checkpoint(
    *,
    config: Any,
    data_digest: str,
    training_order: tuple[str, ...],
    completed_outcomes: tuple[Any, ...],
    completed_hashes: tuple[str, ...],
    torch: Any,
    active_circadian: VisionCircadianProgress | None = None,
    shared_train_generator_state: Any | None = None,
    cuda_device: str | None = None,
) -> VisionRunnerCheckpoint:
    """Capture complete model-boundary progress and process RNG."""
    return VisionRunnerCheckpoint(
        format_version=2,
        protocol_id=config.protocol_id,
        config_digest=vision_config_digest(config, training_order),
        data_digest=data_digest,
        training_order=training_order,
        next_variant_index=len(completed_outcomes),
        completed_outcomes=completed_outcomes,
        completed_hashes=completed_hashes,
        python_random_state=deepcopy(random.getstate()),
        numpy_random_state=deepcopy(np.random.get_state()),
        torch_cpu_random_state=torch.get_rng_state().clone(),
        active_circadian=active_circadian,
        shared_train_generator_state=(
            shared_train_generator_state.detach().clone()
            if shared_train_generator_state is not None
            else None
        ),
        torch_cuda_device=cuda_device,
        torch_cuda_random_state=(
            torch.cuda.get_rng_state(cuda_device).clone() if cuda_device is not None else None
        ),
    )


def validate_vision_checkpoint(
    checkpoint: VisionRunnerCheckpoint,
    *,
    config: Any,
    data_digest: str,
    training_order: tuple[str, ...],
    torch: Any,
    shared_train_loader: bool = False,
    cuda_device: str | None = None,
) -> None:
    """Reject incompatible model-boundary files before restoring process state."""
    if (
        not isinstance(checkpoint, VisionRunnerCheckpoint)
        or type(checkpoint.format_version) is not int
        or checkpoint.format_version != 2
    ):
        raise ValueError("incompatible vision checkpoint format")
    if (
        checkpoint.protocol_id != config.protocol_id
        or checkpoint.config_digest != vision_config_digest(config, training_order)
        or checkpoint.data_digest != data_digest
        or checkpoint.training_order != training_order
    ):
        raise ValueError("incompatible vision checkpoint config, order, or data")
    if (
        type(checkpoint.next_variant_index) is not int
        or not 0 <= checkpoint.next_variant_index <= len(training_order)
        or not isinstance(checkpoint.completed_outcomes, tuple)
        or not isinstance(checkpoint.completed_hashes, tuple)
        or len(checkpoint.completed_outcomes) != checkpoint.next_variant_index
        or len(checkpoint.completed_hashes) != checkpoint.next_variant_index
        or any(not isinstance(value, str) for value in checkpoint.completed_hashes)
    ):
        raise ValueError("incompatible vision checkpoint model progress")
    if checkpoint.active_circadian is not None and (
        not isinstance(checkpoint.active_circadian, VisionCircadianProgress)
        or checkpoint.next_variant_index >= len(training_order)
        or training_order[checkpoint.next_variant_index] != "circadian"
    ):
        raise ValueError("incompatible vision checkpoint active variant")
    try:
        random.Random().setstate(deepcopy(checkpoint.python_random_state))
        np.random.RandomState().set_state(deepcopy(checkpoint.numpy_random_state))
        state = checkpoint.torch_cpu_random_state
        if not torch.is_tensor(state) or state.dtype != torch.uint8:
            raise ValueError("invalid Torch CPU RNG state")
        torch.Generator(device="cpu").set_state(state.detach().clone())
        shared_state = checkpoint.shared_train_generator_state
        if shared_train_loader:
            if (
                shared_state is None
                or not torch.is_tensor(shared_state)
                or shared_state.dtype != torch.uint8
            ):
                raise ValueError("invalid shared train-loader generator state")
            torch.Generator(device="cpu").set_state(shared_state.detach().clone())
        elif shared_state is not None:
            raise ValueError("unexpected shared train-loader generator state")
        saved_cuda_device = getattr(checkpoint, "torch_cuda_device", None)
        cuda_state = getattr(checkpoint, "torch_cuda_random_state", None)
        if cuda_device is None:
            if saved_cuda_device is not None or cuda_state is not None:
                raise ValueError("unexpected CUDA process RNG state")
        else:
            if saved_cuda_device != cuda_device:
                raise ValueError("incompatible CUDA checkpoint device")
            if (
                cuda_state is None
                or not torch.is_tensor(cuda_state)
                or cuda_state.dtype != torch.uint8
                or cuda_state.device.type != "cpu"
            ):
                raise ValueError("invalid CUDA process RNG state")
            torch.Generator(device=cuda_device).set_state(cuda_state.detach().clone())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("incompatible vision checkpoint process RNG") from exc


def restore_vision_process_rng(checkpoint: VisionRunnerCheckpoint, torch: Any) -> None:
    """Apply process streams after all model/data preflight checks pass."""
    random.setstate(deepcopy(checkpoint.python_random_state))
    np.random.set_state(deepcopy(checkpoint.numpy_random_state))
    torch.set_rng_state(checkpoint.torch_cpu_random_state.detach().clone())
    cuda_device = getattr(checkpoint, "torch_cuda_device", None)
    if cuda_device is not None:
        cuda_state = checkpoint.torch_cuda_random_state
        assert cuda_state is not None
        torch.cuda.set_rng_state(cuda_state.detach().clone(), cuda_device)
