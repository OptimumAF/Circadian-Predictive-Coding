"""Runner-owned identity and progress for the NumPy toy comparison.

Inputs are seeded training/validation roles and the three-model cursor.
The infra adapter owns file encoding; final-test data and scoring are not
part of this training checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Protocol

from src.app.circadian_checkpoint import CircadianResumePosition, CircadianRunCheckpoint
from src.app.numpy_checkpoint_validation import (
    LabeledRole,
    digest_labeled_roles,
    validate_numpy_baseline_model,
)
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianNetworkSnapshot
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.sleep_telemetry import SleepEventTelemetry, SleepReplayUsage, SleepStructuralChanges
import numpy as np


@dataclass(frozen=True)
class ToyRunnerCheckpoint:
    """All runner state outside the combined circadian checkpoint."""

    format_version: int
    runner_config_digest: str
    data_digest: str
    backprop_model: BackpropMLP
    predictive_model: PredictiveCodingNetwork
    losses: tuple[list[float], list[float], list[float]]
    sleep_event_count: int
    total_splits: int
    total_prunes: int
    hidden_dim_start: int
    sleep_events: tuple[SleepEventTelemetry, ...]
    combined: CircadianRunCheckpoint


class ToyCheckpointStore(Protocol):
    """App port for a replaceable trusted local toy checkpoint."""

    def load(self) -> ToyRunnerCheckpoint: ...

    def save(self, checkpoint: ToyRunnerCheckpoint) -> None: ...


def toy_config_digest(config: Any) -> str:
    """Bind every runner option, including order and epoch count."""
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def toy_data_digest(train: LabeledRole, validation: LabeledRole | None) -> str:
    """Bind the seeded development roles without reading final-test arrays."""
    return digest_labeled_roles(
        "numpy_toy_development_roles_v1",
        (("train", train), ("validation", validation)),
    )


def toy_checkpoint_width_work(checkpoint: ToyRunnerCheckpoint) -> tuple[int, int]:
    """Return current and historical transient circadian width from saved work."""
    snapshot = checkpoint.combined.model_state
    if not isinstance(snapshot, CircadianNetworkSnapshot) or type(snapshot.state) is not dict:
        raise ValueError("incompatible toy checkpoint circadian width state")
    weights = snapshot.state.get("weight_input_hidden")
    if not isinstance(weights, np.ndarray) or weights.ndim != 2 or weights.shape[1] <= 0:
        raise ValueError("incompatible toy checkpoint circadian width state")
    if type(checkpoint.hidden_dim_start) is not int or checkpoint.hidden_dim_start <= 0:
        raise ValueError("incompatible toy checkpoint initial width")
    current_width = int(weights.shape[1])
    peak_width = max(checkpoint.hidden_dim_start, current_width)
    for event in checkpoint.sleep_events:
        if not isinstance(event, SleepEventTelemetry) or not isinstance(
            event.changes, SleepStructuralChanges
        ):
            raise ValueError("incompatible toy checkpoint sleep width history")
        try:
            event.changes.__post_init__()
            event.__post_init__()
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("incompatible toy checkpoint sleep width history") from exc
        peak_width = max(peak_width, event.before_width + len(event.changes.applied_split_pairs))
    return current_width, peak_width


def validate_toy_checkpoint(
    checkpoint: ToyRunnerCheckpoint,
    *,
    config_digest: str,
    data_digest: str,
    protocol_id: str,
    epoch_count: int,
    model_order: tuple[str, ...],
    hidden_dims: tuple[int, ...],
) -> CircadianResumePosition:
    """Reject bad identity/cursors before restoring a fresh model or RNG."""
    if (
        not isinstance(checkpoint, ToyRunnerCheckpoint)
        or type(checkpoint.format_version) is not int
        or checkpoint.format_version != 2
    ):
        raise ValueError("incompatible toy checkpoint format")
    if checkpoint.runner_config_digest != config_digest or checkpoint.data_digest != data_digest:
        raise ValueError("incompatible toy checkpoint config or data")
    if not isinstance(checkpoint.combined, CircadianRunCheckpoint):
        raise ValueError("incompatible toy combined checkpoint")
    position = checkpoint.combined.position
    if (
        not isinstance(position, CircadianResumePosition)
        or checkpoint.combined.protocol_id != protocol_id
        or checkpoint.combined.data_digest != data_digest
        or checkpoint.combined.retry_state is not None
    ):
        raise ValueError("incompatible toy checkpoint protocol or position")
    if position.stage == "wake":
        if not 0 <= position.completed_epoch < epoch_count or not (
            1 <= position.next_batch_index < len(model_order)
        ):
            raise ValueError("incompatible toy checkpoint wake cursor")
        completed = position.completed_epoch
        expected_lengths = tuple(
            completed + int(name in model_order[: position.next_batch_index])
            for name in ("backprop", "predictive_coding", "circadian_predictive_coding")
        )
    elif position.stage in {"before_sleep", "after_sleep"}:
        if not 1 <= position.completed_epoch <= epoch_count or position.next_batch_index != 0:
            raise ValueError("incompatible toy checkpoint sleep cursor")
        completed = position.completed_epoch
        expected_lengths = (completed, completed, completed)
    else:
        raise ValueError("incompatible toy checkpoint stage")
    expected_events = completed - int(position.stage == "before_sleep")
    if (
        type(checkpoint.sleep_events) is not tuple
        or len(checkpoint.sleep_events) != expected_events
        or any(
            not isinstance(event, SleepEventTelemetry)
            or event.format_version != 1
            or event.completed_epoch != index
            or event.wake_batches != index
            or event.guard is not None
            for index, event in enumerate(checkpoint.sleep_events, start=1)
        )
    ):
        raise ValueError("incompatible toy checkpoint sleep event history")
    # Why this: a replay budget restored from telemetry must reject a
    # deserialized negative or over-applied count before any new work.
    try:
        for event in checkpoint.sleep_events:
            if type(event.replay) is not SleepReplayUsage:
                raise ValueError("replay usage is not typed")
            event.replay.__post_init__()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("incompatible toy checkpoint replay usage") from exc
    if (
        not isinstance(checkpoint.losses, tuple)
        or len(checkpoint.losses) != 3
        or any(
            not isinstance(history, list)
            or len(history) != expected
            or any(type(value) not in {int, float} or not np.isfinite(value) for value in history)
            for history, expected in zip(checkpoint.losses, expected_lengths)
        )
    ):
        raise ValueError("incompatible toy checkpoint metric progress")
    counters = (
        checkpoint.sleep_event_count,
        checkpoint.total_splits,
        checkpoint.total_prunes,
        checkpoint.hidden_dim_start,
    )
    if any(type(value) is not int or value < 0 for value in counters) or (
        checkpoint.sleep_event_count > completed or checkpoint.hidden_dim_start != hidden_dims[-1]
    ):
        raise ValueError("incompatible toy checkpoint sleep counters")
    if not isinstance(checkpoint.backprop_model, BackpropMLP) or not isinstance(
        checkpoint.predictive_model, PredictiveCodingNetwork
    ):
        raise ValueError("incompatible toy checkpoint baseline types")
    backprop = checkpoint.backprop_model
    predictive = checkpoint.predictive_model
    validate_numpy_baseline_model(backprop, hidden_dims, expected_lengths[0])
    validate_numpy_baseline_model(predictive, hidden_dims, expected_lengths[1])
    expected_wake = expected_lengths[2]
    if position.wake_batches != expected_wake:
        raise ValueError("incompatible toy checkpoint wake progress")
    return position
