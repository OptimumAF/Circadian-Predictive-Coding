"""Unscored completed and active checkpoint values for the v8 policy comparison.

Inputs are a frozen manifest digest and completed policy/seed records. Output
is a typed format-9 cursor; this module does not train, open final roles, or
read files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Protocol

from src.app.circadian_checkpoint import CircadianResumePosition
from src.app.continual_arrived_checkpoint import ArrivedRunnerCheckpoint, ArrivedUnscoredSeed
from src.app.continual_checkpoint import ContinualRunnerState
from src.core.circadian_predictive_coding import CircadianNetworkSnapshot
from src.core.replay_retention import ReplayExposureSnapshot


REPLAY_POLICY_CHECKPOINT_FORMAT = 9


@dataclass(frozen=True)
class ReplayPolicyUnscoredSeed:
    """One complete policy/seed without final-test values or hashes."""

    policy_index: int
    seed_index: int
    arrived: ArrivedUnscoredSeed
    exposure_digest: str


@dataclass(frozen=True, kw_only=True)
class ReplayPolicyActiveTransaction(ArrivedRunnerCheckpoint):
    """Distinct format-9 A/B cursor inside one declared policy trial."""

    policy_index: int


@dataclass(frozen=True)
class ReplayPolicyRunnerCheckpoint:
    """Completed prefix plus at most one active policy/seed transaction."""

    format_version: int
    manifest_digest: str
    next_trial_index: int
    unscored_seeds: tuple[ReplayPolicyUnscoredSeed, ...]
    active: ReplayPolicyActiveTransaction | None = None


class ReplayPolicyCheckpointStore(Protocol):
    """App port for one trusted replaceable local format-9 file."""

    def load(self) -> ReplayPolicyRunnerCheckpoint: ...

    def save(self, checkpoint: ReplayPolicyRunnerCheckpoint) -> None: ...


def replay_exposure_digest(phase_a: ReplayExposureSnapshot, after_b: ReplayExposureSnapshot) -> str:
    """Bind the two cumulative ledgers independently of model pickle bytes."""
    payload = json.dumps(
        {"phase_a": asdict(phase_a), "after_b": asdict(after_b)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def validate_replay_policy_checkpoint_header(
    checkpoint: ReplayPolicyRunnerCheckpoint,
    *,
    manifest_digest: str,
    policy_count: int,
    seeds: tuple[int, ...],
    phase_a_epochs: int,
    phase_b_epochs: int,
    model_order: tuple[str, ...],
) -> None:
    """Reject changed identity and a non-prefix cursor before any source use."""
    seed_count = len(seeds)
    trial_count = policy_count * seed_count
    if (
        type(checkpoint) is not ReplayPolicyRunnerCheckpoint
        or type(checkpoint.format_version) is not int
        or checkpoint.format_version != REPLAY_POLICY_CHECKPOINT_FORMAT
        or checkpoint.manifest_digest != manifest_digest
    ):
        raise ValueError("incompatible v8 policy checkpoint format or manifest")
    if (
        type(checkpoint.next_trial_index) is not int
        or not 0 <= checkpoint.next_trial_index <= trial_count
        or type(checkpoint.unscored_seeds) is not tuple
        or len(checkpoint.unscored_seeds) != checkpoint.next_trial_index
        or (checkpoint.active is None and checkpoint.next_trial_index == 0)
    ):
        raise ValueError("incompatible v8 policy checkpoint trial cursor")
    for index, record in enumerate(checkpoint.unscored_seeds):
        if (
            type(record) is not ReplayPolicyUnscoredSeed
            or record.policy_index != index // seed_count
            or record.seed_index != index % seed_count
            or type(record.arrived) is not ArrivedUnscoredSeed
            or any("final_test" in role for role, _ in record.arrived.role_ids)
            or any("final_test" in role for role, _ in record.arrived.role_hashes)
            or any(event.role == "final_test" for event in record.arrived.role_accesses)
        ):
            raise ValueError("incompatible v8 policy checkpoint record or final-data seal")
    active = checkpoint.active
    if active is None:
        return
    if (
        type(active) is not ReplayPolicyActiveTransaction
        or checkpoint.next_trial_index >= trial_count
        or active.format_version != REPLAY_POLICY_CHECKPOINT_FORMAT
        or active.config_digest != manifest_digest
        or active.policy_index != checkpoint.next_trial_index // seed_count
        or active.seed_index != checkpoint.next_trial_index % seed_count
        or type(active.policy_index) is not int
        or type(active.seed_index) is not int
        or active.seeds != seeds
        or active.unscored_seeds != ()
        or active.phase not in {"a", "b"}
        or any("final_test" in role for role, _ in active.active_role_ids)
        or any("final_test" in role for role, _ in active.active_role_hashes)
        or any(event.role == "final_test" for event in active.active_role_accesses)
        or not isinstance(active.active_state, ContinualRunnerState)
        or not isinstance(active.active_circadian, CircadianNetworkSnapshot)
        or not isinstance(active.active_position, CircadianResumePosition)
    ):
        raise ValueError("incompatible v8 active policy cursor or final-data seal")
    frozen = (
        active.active_state.backprop_after_a,
        active.active_state.predictive_after_a,
        active.active_state.circadian_after_a,
    )
    if (active.phase == "a" and any(item is not None for item in frozen)) or (
        active.phase == "b" and any(item is None for item in frozen)
    ):
        raise ValueError("incompatible v8 active phase or frozen Phase A state")
    phase_epochs = phase_a_epochs if active.phase == "a" else phase_b_epochs
    local = active.phase_epoch_completed
    if (
        type(local) is not int
        or not 0 <= local <= phase_epochs
        or active.stage not in {"wake", "before_sleep", "after_sleep"}
        or type(active.next_model_index) is not int
        or (
            active.stage == "wake"
            and (local >= phase_epochs or not 1 <= active.next_model_index < len(model_order))
        )
        or (active.stage == "before_sleep" and (local < 1 or active.next_model_index != 0))
        or (active.stage == "after_sleep" and active.next_model_index != 0)
        or active.active_position.stage != active.stage
        or active.active_position.next_batch_index != active.next_model_index
        or active.active_position.completed_epoch
        != (0 if active.phase == "a" else phase_a_epochs) + local
    ):
        raise ValueError("incompatible v8 active policy progress cursor")
