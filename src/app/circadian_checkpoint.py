"""Combine model, runner retry, and caller RNG state for a seeded resume.

This in-memory boundary accepts a NumPy circadian network or a Torch
circadian head. Runners supply their protocol, configuration, data identity,
and progress; durable storage and data-order replay belong to the runners.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
import random
import re
from typing import Any

import numpy as np

from src.app.sleep_schedule import SleepRollbackCooldown, SleepRollbackCooldownState
from src.core.circadian_predictive_coding import (
    CircadianNetworkSnapshot,
    CircadianPredictiveCodingNetwork,
)
from src.core.resnet50_variants import CircadianPredictiveCodingHead
from src.shared.torch_runtime import require_torch

CircadianCheckpointModel = CircadianPredictiveCodingNetwork | CircadianPredictiveCodingHead


@dataclass(frozen=True)
class CircadianResumePosition:
    """Wake cursor or completed wake epoch at a sleep transaction boundary."""

    completed_epoch: int
    stage: str
    wake_batches: int
    next_batch_index: int = 0

    def __post_init__(self) -> None:
        if type(self.completed_epoch) is not int or self.completed_epoch < 0:
            raise ValueError("checkpoint completed_epoch must be a nonnegative integer")
        if self.stage not in {"wake", "before_sleep", "after_sleep"}:
            raise ValueError("checkpoint stage must be wake, before_sleep, or after_sleep")
        if type(self.wake_batches) is not int or self.wake_batches < 0:
            raise ValueError("checkpoint wake_batches must be a nonnegative integer")
        if type(self.next_batch_index) is not int or self.next_batch_index < 0:
            raise ValueError("checkpoint next_batch_index must be a nonnegative integer")
        if self.stage != "wake" and self.next_batch_index != 0:
            raise ValueError("sleep-stage checkpoint next_batch_index must be zero")


@dataclass(frozen=True)
class CircadianRunCheckpoint:
    """Detached state for one compatible in-memory circadian continuation."""

    format_version: int
    backend: str
    protocol_id: str
    config_digest: str
    data_digest: str
    position: CircadianResumePosition
    model_state: CircadianNetworkSnapshot | dict[str, Any]
    retry_state: SleepRollbackCooldownState | None
    python_random_state: Any
    numpy_random_state: Any
    torch_cpu_random_state: Any | None
    torch_cuda_device: str | None = None
    torch_cuda_random_state: Any | None = None


def capture_circadian_checkpoint(
    model: CircadianCheckpointModel,
    *,
    retry: SleepRollbackCooldown | None,
    position: CircadianResumePosition,
    protocol_id: str,
    config: Any,
    data_digest: str,
) -> CircadianRunCheckpoint:
    """Capture a detached model/runner/process state at a caller-owned boundary."""
    backend = _model_backend(model)
    _validate_identity(model, position, protocol_id, config, data_digest)
    if retry is not None and not isinstance(retry, SleepRollbackCooldown):
        raise TypeError("checkpoint retry state must use SleepRollbackCooldown")
    torch = require_torch() if backend.startswith("torch_head") else None
    torch_state = torch.get_rng_state().clone() if torch is not None else None
    cuda_device = _cuda_device(model) if backend == "torch_head_cuda" else None
    cuda_state = require_torch().cuda.get_rng_state(cuda_device).clone() if cuda_device else None
    return CircadianRunCheckpoint(
        format_version=1,
        backend=backend,
        protocol_id=protocol_id,
        config_digest=_config_digest(config),
        data_digest=data_digest,
        position=position,
        model_state=deepcopy(model.snapshot_state()),
        retry_state=retry.snapshot_state() if retry is not None else None,
        python_random_state=deepcopy(random.getstate()),
        numpy_random_state=deepcopy(np.random.get_state()),
        torch_cpu_random_state=torch_state,
        torch_cuda_device=cuda_device,
        torch_cuda_random_state=cuda_state,
    )


def restore_circadian_checkpoint(
    model: CircadianCheckpointModel,
    checkpoint: CircadianRunCheckpoint,
    *,
    retry: SleepRollbackCooldown | None,
    protocol_id: str,
    config: Any,
    data_digest: str,
    expected_stage: str,
) -> CircadianResumePosition:
    """Reject all known incompatibilities before restoring any live state."""
    backend = _model_backend(model)
    if (
        not isinstance(checkpoint, CircadianRunCheckpoint)
        or type(checkpoint.format_version) is not int
        or checkpoint.format_version != 1
    ):
        raise ValueError("incompatible checkpoint format")
    if not isinstance(checkpoint.position, CircadianResumePosition):
        raise ValueError("incompatible checkpoint position")
    _validate_identity(
        model, checkpoint.position, protocol_id, config, data_digest, check_wake_batches=False
    )
    if (
        checkpoint.backend != backend
        or checkpoint.protocol_id != protocol_id
        or checkpoint.config_digest != _config_digest(config)
        or checkpoint.data_digest != data_digest
        or checkpoint.position.stage != expected_stage
    ):
        raise ValueError("incompatible checkpoint protocol, config, data, backend, or stage")
    if (retry is None) != (checkpoint.retry_state is None):
        raise ValueError("incompatible checkpoint retry state presence")
    retry_state = checkpoint.retry_state
    if retry is not None:
        if not isinstance(retry, SleepRollbackCooldown):
            raise TypeError("checkpoint retry state must use SleepRollbackCooldown")
        if retry_state is None:
            raise ValueError("incompatible checkpoint retry state presence")
        SleepRollbackCooldown(retry.cooldown_epochs).restore_state(retry_state)

    _validate_random_state(
        checkpoint,
        backend,
        expected_cuda_device=_cuda_device(model) if backend == "torch_head_cuda" else None,
    )
    # Why this: a temporary model catches malformed topology or RNG state
    # before the first live model or runner value is changed.
    candidate = object.__new__(type(model))
    candidate.__dict__ = model.__dict__.copy()
    _restore_model(candidate, checkpoint.model_state)
    if candidate.get_sleep_clocks().wake_batches != checkpoint.position.wake_batches:
        raise ValueError("incompatible checkpoint wake batch progress")

    _restore_model(model, checkpoint.model_state)
    if retry is not None:
        assert retry_state is not None
        retry.restore_state(retry_state)
    random.setstate(deepcopy(checkpoint.python_random_state))
    np.random.set_state(deepcopy(checkpoint.numpy_random_state))
    if backend.startswith("torch_head"):
        torch_state = checkpoint.torch_cpu_random_state
        assert torch_state is not None
        require_torch().set_rng_state(torch_state.detach().clone())
    if backend == "torch_head_cuda":
        cuda_state = checkpoint.torch_cuda_random_state
        cuda_device = checkpoint.torch_cuda_device
        assert cuda_state is not None and cuda_device is not None
        require_torch().cuda.set_rng_state(cuda_state.detach().clone(), cuda_device)
    return checkpoint.position


def _model_backend(model: CircadianCheckpointModel) -> str:
    if isinstance(model, CircadianPredictiveCodingNetwork):
        return "numpy"
    if isinstance(model, CircadianPredictiveCodingHead):
        device = model.weight_feature_hidden.device
        if device.type == "cpu":
            return "torch_head"
        if device.type == "cuda":
            return "torch_head_cuda"
        raise ValueError("checkpoint Torch head must use a CPU or CUDA device")
    raise TypeError("checkpoint model must be a circadian NumPy network or Torch head")


def _restore_model(
    model: CircadianCheckpointModel,
    state: CircadianNetworkSnapshot | dict[str, Any],
) -> None:
    if isinstance(model, CircadianPredictiveCodingNetwork):
        if not isinstance(state, CircadianNetworkSnapshot):
            raise ValueError("incompatible checkpoint NumPy model state")
        model.restore_state(state)
    else:
        if not isinstance(state, dict):
            raise ValueError("incompatible checkpoint Torch head state")
        model.restore_state(state)


def _validate_identity(
    model: CircadianCheckpointModel,
    position: CircadianResumePosition,
    protocol_id: str,
    config: Any,
    data_digest: str,
    *,
    check_wake_batches: bool = True,
) -> None:
    if not isinstance(position, CircadianResumePosition):
        raise TypeError("checkpoint position must be CircadianResumePosition")
    CircadianResumePosition(
        position.completed_epoch,
        position.stage,
        position.wake_batches,
        position.next_batch_index,
    )
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError("checkpoint protocol_id must be nonempty")
    if not isinstance(data_digest, str) or re.fullmatch(r"[0-9a-f]{64}", data_digest) is None:
        raise ValueError("checkpoint data_digest must be a lowercase SHA-256 hex digest")
    if not is_dataclass(config) or isinstance(config, type) or config != model.config:
        raise ValueError("checkpoint config must match the model config dataclass")
    if check_wake_batches and model.get_sleep_clocks().wake_batches != position.wake_batches:
        raise ValueError("checkpoint wake batch progress does not match model state")


def _config_digest(config: Any) -> str:
    name = f"{type(config).__module__}.{type(config).__qualname__}"
    encoded = json.dumps(
        {"type": name, "fields": asdict(config)},
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _cuda_device(model: CircadianCheckpointModel) -> str:
    assert isinstance(model, CircadianPredictiveCodingHead)
    # Why this: a configured "cuda" alias can point to a different index later.
    return str(model.weight_feature_hidden.device)


def _validate_random_state(
    checkpoint: CircadianRunCheckpoint,
    backend: str,
    *,
    expected_cuda_device: str | None,
) -> None:
    try:
        random.Random().setstate(deepcopy(checkpoint.python_random_state))
        np.random.RandomState().set_state(deepcopy(checkpoint.numpy_random_state))
        if backend.startswith("torch_head"):
            torch = require_torch()
            state = checkpoint.torch_cpu_random_state
            if (
                not isinstance(state, torch.Tensor)
                or state.dtype != torch.uint8
                or state.device.type != "cpu"
            ):
                raise ValueError("invalid Torch CPU RNG state")
            torch.Generator(device="cpu").set_state(state.detach().clone())
        elif checkpoint.torch_cpu_random_state is not None:
            raise ValueError("unexpected Torch CPU RNG state")
        cuda_state = getattr(checkpoint, "torch_cuda_random_state", None)
        cuda_device = getattr(checkpoint, "torch_cuda_device", None)
        if backend == "torch_head_cuda":
            if cuda_device != expected_cuda_device:
                raise ValueError("incompatible checkpoint CUDA device")
            if (
                not isinstance(cuda_state, torch.Tensor)
                or cuda_state.dtype != torch.uint8
                or cuda_state.device.type != "cpu"
            ):
                raise ValueError("invalid Torch CUDA RNG state")
            torch.Generator(device=cuda_device).set_state(cuda_state.detach().clone())
        elif cuda_device is not None or cuda_state is not None:
            raise ValueError("unexpected Torch CUDA RNG state")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("incompatible checkpoint process random state") from exc
