"""Unscored v6 continual checkpoint values and run header checks.

Inputs are fixed run identity, detached seed/active state, and an A/B cursor.
Outputs are typed, final-test-free checkpoint records. This module does not
load files, train models, or release held-out source fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import TYPE_CHECKING, Protocol

from src.app import continual_shift_benchmark as base
from src.app.circadian_checkpoint import CircadianResumePosition
from src.app.continual_checkpoint import ContinualRunnerState, continual_config_digest
from src.core.circadian_predictive_coding import CircadianNetworkSnapshot
from src.core.sleep_telemetry import SleepEventTelemetry

if TYPE_CHECKING:
    from src.app.continual_arrived_benchmark import (
        GuardDecision,
        MethodTaskInformation,
        RoleAccessEvent,
    )


ARRIVED_CHECKPOINT_FORMAT = 6


@dataclass(frozen=True)
class ArrivedUnscoredSeed:
    """Completed models and observed development roles, with no final data."""

    seed: int
    role_ids: tuple[tuple[str, tuple[str, ...]], ...]
    role_hashes: tuple[tuple[str, str], ...]
    state: base._ContinualTrainingState
    role_accesses: tuple[RoleAccessEvent, ...]
    guard_decisions: tuple[GuardDecision, ...]
    method_task_information: tuple[MethodTaskInformation, ...]
    event_digest: str
    sleep_events: tuple[SleepEventTelemetry, ...] = ()
    sleep_history_digest: str = ""
    sleep_event_history_version: int = 1


@dataclass(frozen=True)
class ArrivedRunnerCheckpoint:
    """Separate v6 run transaction for completed or active A/B state."""

    format_version: int
    config_digest: str
    seeds: tuple[int, ...]
    seed_index: int
    phase: str
    phase_epoch_completed: int
    stage: str
    next_model_index: int
    unscored_seeds: tuple[ArrivedUnscoredSeed, ...]
    active_role_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()
    active_role_hashes: tuple[tuple[str, str], ...] = ()
    active_role_accesses: tuple[RoleAccessEvent, ...] = ()
    active_guard_decisions: tuple[GuardDecision, ...] = ()
    active_task_information: tuple[MethodTaskInformation, ...] = ()
    active_event_digest: str = ""
    active_state: ContinualRunnerState | None = None
    active_circadian: CircadianNetworkSnapshot | None = None
    active_position: CircadianResumePosition | None = None
    sleep_event_history_version: int = 1
    active_sleep_events: tuple[SleepEventTelemetry, ...] = ()
    active_sleep_history_digest: str = ""


class ArrivedCheckpointStore(Protocol):
    """App port for a trusted replaceable local v6 checkpoint."""

    def load(self) -> ArrivedRunnerCheckpoint: ...

    def save(self, checkpoint: ArrivedRunnerCheckpoint) -> None: ...


def arrived_config_digest(config: object, seeds: tuple[int, ...]) -> str:
    """Bind every setting and ordered seed before source access."""
    return continual_config_digest(config, seeds)


def arrived_event_digest(
    accesses: tuple[RoleAccessEvent, ...],
    decisions: tuple[GuardDecision, ...],
    information: tuple[MethodTaskInformation, ...],
) -> str:
    """Bind the completed release/update/guard cursor without final data."""
    payload = json.dumps(
        {
            "accesses": [asdict(item) for item in accesses],
            "decisions": [asdict(item) for item in decisions],
            "task_information": [asdict(item) for item in information],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def arrived_sleep_history_digest(
    events: tuple[SleepEventTelemetry, ...], role_event_digest: str
) -> str:
    """Bind measured sleep facts to the independently retained role ledger."""
    payload = json.dumps(
        {"role_event_digest": role_event_digest, "sleep_events": [asdict(item) for item in events]},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def validate_arrived_checkpoint_header(
    checkpoint: ArrivedRunnerCheckpoint,
    *,
    config_digest: str,
    seeds: tuple[int, ...],
    phase_a_epochs: int,
    phase_b_epochs: int,
    model_order: tuple[str, ...],
) -> None:
    """Reject wrong run, final data, and impossible cursor before source access."""
    if (
        type(checkpoint) is not ArrivedRunnerCheckpoint
        or type(checkpoint.format_version) is not int
        or checkpoint.format_version != ARRIVED_CHECKPOINT_FORMAT
        or checkpoint.config_digest != config_digest
        or checkpoint.seeds != seeds
    ):
        raise ValueError("incompatible v6 checkpoint format, config, or seeds")
    if vars(checkpoint).get("sleep_event_history_version") != 1:
        raise ValueError("incompatible v6 checkpoint sleep history version")
    if type(checkpoint.seed_index) is not int or not 0 <= checkpoint.seed_index < len(seeds):
        raise ValueError("incompatible v6 checkpoint seed cursor")
    completed_count = checkpoint.seed_index + int(checkpoint.phase == "seed_complete")
    if (
        not isinstance(checkpoint.unscored_seeds, tuple)
        or len(checkpoint.unscored_seeds) != completed_count
        or any(
            type(record) is not ArrivedUnscoredSeed
            or record.seed != seeds[index]
            or any("final_test" in role for role, _ in record.role_hashes)
            or any("final_test" in role for role, _ in record.role_ids)
            or any(event.role == "final_test" for event in record.role_accesses)
            for index, record in enumerate(checkpoint.unscored_seeds)
        )
    ):
        raise ValueError("incompatible v6 unscored seed records or final-data seal")
    if checkpoint.phase == "seed_complete":
        if (
            checkpoint.phase_epoch_completed != phase_b_epochs
            or checkpoint.stage != "after_sleep"
            or checkpoint.next_model_index != 0
            or checkpoint.active_role_ids
            or checkpoint.active_role_hashes
            or checkpoint.active_role_accesses
            or checkpoint.active_guard_decisions
            or checkpoint.active_task_information
            or checkpoint.active_event_digest
            or checkpoint.active_state is not None
            or checkpoint.active_circadian is not None
            or checkpoint.active_position is not None
            or vars(checkpoint).get("active_sleep_events") != ()
            or vars(checkpoint).get("active_sleep_history_digest") != ""
        ):
            raise ValueError("incompatible v6 completed-seed cursor or final-data seal")
        return
    if checkpoint.phase not in {"a", "b"}:
        raise ValueError("incompatible v6 active checkpoint phase")
    phase_epochs = phase_a_epochs if checkpoint.phase == "a" else phase_b_epochs
    local = checkpoint.phase_epoch_completed
    if (
        type(local) is not int
        or not 0 <= local <= phase_epochs
        or checkpoint.stage not in {"wake", "before_sleep", "after_sleep"}
        or type(checkpoint.next_model_index) is not int
        or (
            checkpoint.stage == "wake"
            and (local >= phase_epochs or not 1 <= checkpoint.next_model_index < len(model_order))
        )
        or (checkpoint.stage == "before_sleep" and (local < 1 or checkpoint.next_model_index != 0))
        or (checkpoint.stage == "after_sleep" and checkpoint.next_model_index != 0)
        or not isinstance(checkpoint.active_role_ids, tuple)
        or not isinstance(checkpoint.active_role_hashes, tuple)
        or not checkpoint.active_role_ids
        or not checkpoint.active_role_hashes
        or any("final_test" in role for role, _ in checkpoint.active_role_ids)
        or any("final_test" in role for role, _ in checkpoint.active_role_hashes)
        or any(event.role == "final_test" for event in checkpoint.active_role_accesses)
        or not isinstance(checkpoint.active_state, ContinualRunnerState)
        or not isinstance(checkpoint.active_circadian, CircadianNetworkSnapshot)
        or not isinstance(checkpoint.active_position, CircadianResumePosition)
        or type(vars(checkpoint).get("active_sleep_events")) is not tuple
        or type(vars(checkpoint).get("active_sleep_history_digest")) is not str
        or checkpoint.active_position.stage != checkpoint.stage
        or checkpoint.active_position.next_batch_index != checkpoint.next_model_index
        or checkpoint.active_position.completed_epoch
        != (0 if checkpoint.phase == "a" else phase_a_epochs) + local
    ):
        raise ValueError("incompatible v6 active checkpoint cursor or final-data seal")
