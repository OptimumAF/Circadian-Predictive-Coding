"""Runner-owned file payload and preflight for continual NumPy resume.

The payload binds the configured seed list, active phase, model-step cursor,
completed seed reports, and detached model state. File IO and final-test
scoring belong to separate layers.
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
from src.core.sleep_telemetry import SleepEventTelemetry


@dataclass
class ContinualRunnerState:
    """Current and phase-A-frozen models plus per-seed sleep report counters."""

    backprop_model: BackpropMLP
    predictive_model: PredictiveCodingNetwork
    backprop_after_a: BackpropMLP | None = None
    predictive_after_a: PredictiveCodingNetwork | None = None
    circadian_after_a: CircadianNetworkSnapshot | None = None
    sleep_event_count: int = 0
    total_splits: int = 0
    total_prunes: int = 0
    hidden_dim_start: int = 0


@dataclass(frozen=True)
class ContinualUnscoredSeed:
    """Format-5 trained state and arrived-role identity, without test data."""

    seed: int
    data_digest: str
    split_hashes: tuple[tuple[str, str], ...]
    state: ContinualRunnerState
    circadian_final: CircadianNetworkSnapshot
    sleep_events: tuple[SleepEventTelemetry, ...] = ()


@dataclass(frozen=True)
class ContinualRunnerCheckpoint:
    """One seed/phase transaction and the reports already committed."""

    format_version: int
    runner_config_digest: str
    seeds: tuple[int, ...]
    seed_index: int
    phase: str
    phase_epoch_completed: int
    data_digest: str
    split_hashes: tuple[tuple[str, str], ...]
    completed_results: list[Any]
    completed_data_digests: tuple[str, ...]
    completed_test_digests: tuple[str, ...]
    state: ContinualRunnerState
    combined: CircadianRunCheckpoint
    unscored_seeds: tuple[ContinualUnscoredSeed, ...] = ()
    sleep_event_history_version: int = 1
    sleep_events: tuple[SleepEventTelemetry, ...] = ()


class ContinualCheckpointStore(Protocol):
    """App port for one replaceable trusted local checkpoint."""

    def load(self) -> ContinualRunnerCheckpoint: ...

    def save(self, checkpoint: ContinualRunnerCheckpoint) -> None: ...


def continual_config_digest(config: Any, seeds: tuple[int, ...]) -> str:
    """Bind all runner settings and the exact ordered seed list."""
    payload = json.dumps(
        {"config": asdict(config), "seeds": seeds},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def continual_data_digest(
    phase_a_train: LabeledRole,
    phase_a_validation: LabeledRole | None,
    phase_b_train: LabeledRole,
    phase_b_validation: LabeledRole | None,
) -> str:
    """Bind both development phases without opening either final test."""
    return digest_labeled_roles(
        "numpy_continual_development_roles_v1",
        (
            ("phase_a_train", phase_a_train),
            ("phase_a_validation", phase_a_validation),
            ("phase_b_train", phase_b_train),
            ("phase_b_validation", phase_b_validation),
        ),
    )


def continual_phase_a_data_digest(
    phase_a_train: LabeledRole,
    phase_a_validation: LabeledRole | None,
) -> str:
    """Bind only arrived Phase A development roles for v2 checkpoints."""
    return digest_labeled_roles(
        "numpy_continual_phase_a_development_roles_v2",
        (("phase_a_train", phase_a_train), ("phase_a_validation", phase_a_validation)),
    )


def continual_test_digest(phase_a_test: LabeledRole, phase_b_test: LabeledRole) -> str:
    """Bind completed results only after both phases have finished."""
    return digest_labeled_roles(
        "numpy_continual_final_tests_v1",
        (("phase_a_test", phase_a_test), ("phase_b_test", phase_b_test)),
    )


def validate_continual_checkpoint(
    checkpoint: ContinualRunnerCheckpoint,
    *,
    config_digest: str,
    seeds: tuple[int, ...],
    data_digest: str,
    split_hashes: tuple[tuple[str, str], ...],
    protocol_id: str,
    model_order: tuple[str, ...],
    phase_a_epochs: int,
    phase_b_epochs: int,
    hidden_dims: tuple[int, ...],
    expected_format_version: int = 1,
) -> CircadianResumePosition:
    """Check identity, progress, counters, and baseline state before restore."""
    if (
        not isinstance(checkpoint, ContinualRunnerCheckpoint)
        or type(checkpoint.format_version) is not int
        or checkpoint.format_version != expected_format_version
        or vars(checkpoint).get("sleep_event_history_version") != 1
    ):
        raise ValueError("incompatible continual checkpoint format")
    if (
        checkpoint.runner_config_digest != config_digest
        or checkpoint.seeds != seeds
        or checkpoint.data_digest != data_digest
        or checkpoint.split_hashes != split_hashes
    ):
        raise ValueError("incompatible continual checkpoint config, seeds, or data")
    if type(checkpoint.seed_index) is not int or not 0 <= checkpoint.seed_index < len(seeds):
        raise ValueError("incompatible continual checkpoint seed cursor")
    completed_count = checkpoint.seed_index + int(checkpoint.phase == "seed_complete")
    if expected_format_version == 5:
        if (
            not isinstance(checkpoint.completed_results, list)
            or len(checkpoint.completed_results) != 0
            or not isinstance(checkpoint.completed_data_digests, tuple)
            or len(checkpoint.completed_data_digests) != 0
            or not isinstance(checkpoint.completed_test_digests, tuple)
            or len(checkpoint.completed_test_digests) != 0
            or not isinstance(checkpoint.unscored_seeds, tuple)
            or len(checkpoint.unscored_seeds) != completed_count
            or any(
                not isinstance(item, ContinualUnscoredSeed)
                or item.seed != seeds[index]
                or any("test" in role for role, _ in item.split_hashes)
                for index, item in enumerate(checkpoint.unscored_seeds)
            )
        ):
            raise ValueError("incompatible continual checkpoint unscored seed states")
    elif (
        not isinstance(checkpoint.completed_results, list)
        or len(checkpoint.completed_results) != completed_count
        or not isinstance(checkpoint.completed_data_digests, tuple)
        or len(checkpoint.completed_data_digests) != completed_count
        or not isinstance(checkpoint.completed_test_digests, tuple)
        or len(checkpoint.completed_test_digests) != completed_count
        or bool(getattr(checkpoint, "unscored_seeds", ()))
        or any(
            getattr(result, "seed", None) != seeds[index]
            or getattr(result, "training_order", None) != model_order
            for index, result in enumerate(checkpoint.completed_results)
        )
    ):
        raise ValueError("incompatible continual checkpoint completed seed reports")
    if not isinstance(checkpoint.combined, CircadianRunCheckpoint):
        raise ValueError("incompatible continual combined checkpoint")
    position = checkpoint.combined.position
    if (
        not isinstance(position, CircadianResumePosition)
        or checkpoint.combined.protocol_id != protocol_id
        or checkpoint.combined.data_digest != data_digest
        or checkpoint.combined.retry_state is not None
    ):
        raise ValueError("incompatible continual checkpoint protocol or position")
    if checkpoint.phase == "a":
        offset, phase_epochs = 0, phase_a_epochs
    elif checkpoint.phase in {"b", "seed_complete"}:
        offset, phase_epochs = phase_a_epochs, phase_b_epochs
    else:
        raise ValueError("incompatible continual checkpoint phase")
    local = checkpoint.phase_epoch_completed
    if type(local) is not int or not 0 <= local <= phase_epochs:
        raise ValueError("incompatible continual checkpoint phase epoch")
    if position.completed_epoch != offset + local:
        raise ValueError("incompatible continual checkpoint global epoch")
    if position.stage == "wake":
        if (
            checkpoint.phase == "seed_complete"
            or local >= phase_epochs
            or not (1 <= position.next_batch_index < len(model_order))
        ):
            raise ValueError("incompatible continual checkpoint wake cursor")
        prefix = model_order[: position.next_batch_index]
    elif position.stage == "before_sleep":
        if checkpoint.phase == "seed_complete" or local < 1 or position.next_batch_index != 0:
            raise ValueError("incompatible continual checkpoint before-sleep cursor")
        prefix = model_order
    elif position.stage == "after_sleep":
        if (
            (local < 1 and checkpoint.phase != "b")
            or (checkpoint.phase == "seed_complete" and local != phase_b_epochs)
            or position.next_batch_index != 0
        ):
            raise ValueError("incompatible continual checkpoint after-sleep cursor")
        prefix = ()
    else:
        raise ValueError("incompatible continual checkpoint stage")
    completed_global = offset + local
    expected_backprop = completed_global + int("backprop" in prefix)
    expected_predictive = completed_global + int("predictive_coding" in prefix)
    expected_circadian = completed_global + int("circadian_predictive_coding" in prefix)
    if position.stage == "before_sleep":
        expected_backprop = expected_predictive = expected_circadian = completed_global
    if position.wake_batches != expected_circadian:
        raise ValueError("incompatible continual checkpoint wake progress")
    state = checkpoint.state
    if not isinstance(state, ContinualRunnerState):
        raise ValueError("incompatible continual checkpoint runner state")
    counters = (
        state.sleep_event_count,
        state.total_splits,
        state.total_prunes,
        state.hidden_dim_start,
    )
    if any(type(value) is not int or value < 0 for value in counters) or (
        state.sleep_event_count > completed_global or state.hidden_dim_start != hidden_dims[-1]
    ):
        raise ValueError("incompatible continual checkpoint counters")
    expected_events = position.completed_epoch - int(position.stage == "before_sleep")
    validate_continual_sleep_events(checkpoint.sleep_events, expected_events)
    if not isinstance(state.backprop_model, BackpropMLP) or not isinstance(
        state.predictive_model, PredictiveCodingNetwork
    ):
        raise ValueError("incompatible continual checkpoint baseline types")
    validate_numpy_baseline_model(state.backprop_model, hidden_dims, expected_backprop)
    validate_numpy_baseline_model(state.predictive_model, hidden_dims, expected_predictive)
    frozen = (state.backprop_after_a, state.predictive_after_a, state.circadian_after_a)
    if checkpoint.phase == "a":
        if any(value is not None for value in frozen):
            raise ValueError("incompatible continual checkpoint premature phase-A state")
    else:
        if (
            not isinstance(state.backprop_after_a, BackpropMLP)
            or not isinstance(state.predictive_after_a, PredictiveCodingNetwork)
            or not isinstance(state.circadian_after_a, CircadianNetworkSnapshot)
        ):
            raise ValueError("incompatible continual checkpoint missing phase-A state")
        validate_numpy_baseline_model(state.backprop_after_a, hidden_dims, phase_a_epochs)
        validate_numpy_baseline_model(state.predictive_after_a, hidden_dims, phase_a_epochs)
    if checkpoint.phase == "seed_complete":
        if expected_format_version == 5:
            completed_state = checkpoint.unscored_seeds[-1].state
            counters_match = (
                isinstance(completed_state, ContinualRunnerState)
                and completed_state.sleep_event_count == state.sleep_event_count
                and completed_state.total_splits == state.total_splits
                and completed_state.total_prunes == state.total_prunes
                and completed_state.hidden_dim_start == state.hidden_dim_start
            )
        else:
            result = checkpoint.completed_results[-1]
            report = getattr(result, "circadian_predictive_coding", None)
            counters_match = (
                getattr(report, "sleep_event_count", None) == state.sleep_event_count
                and getattr(report, "total_splits", None) == state.total_splits
                and getattr(report, "total_prunes", None) == state.total_prunes
                and getattr(report, "hidden_dim_start", None) == state.hidden_dim_start
            )
        if not counters_match:
            raise ValueError("incompatible continual checkpoint completed counters")
    return position


def validate_continual_sleep_events(
    events: list[SleepEventTelemetry] | tuple[SleepEventTelemetry, ...],
    expected_epochs: int,
) -> None:
    """Require one unguarded historical event per finished global epoch."""
    if (
        type(events) not in {list, tuple}
        or len(events) != expected_epochs
        or any(
            not isinstance(event, SleepEventTelemetry)
            or event.format_version != 1
            or event.completed_epoch != index
            or event.wake_batches != index
            or event.guard is not None
            for index, event in enumerate(events, start=1)
        )
    ):
        raise ValueError("incompatible continual checkpoint sleep event history")
