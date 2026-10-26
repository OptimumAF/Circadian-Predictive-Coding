"""Finite adaptation choices and hard envelopes, with no executable payloads.

An owner declares a catalog once; a controller selects only its ID. The app
enforces supported effects and original runtime budgets. These records neither
train nor grant access to model/data/permissions, and establish no utility claim.
"""

from dataclasses import dataclass
from math import isfinite

from src.core.experience import require_identifier
from src.core.actor_ports import AppliedConsolidation
from src.core.resource_sharing import TrainingPoll
from src.core.learner_ports import GATE_SETTINGS


def _count(value: int, name: str) -> None:
    if type(value) is not int or not 0 <= value <= 2**31 - 1:
        raise ValueError(f"{name} must be a bounded nonnegative exact integer")


def _rate(value: float, name: str) -> None:
    try:
        valid = type(value) in (int, float) and isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class AdaptationAction:
    action_id: str
    wake_updates: int = 0
    replay_examples: int = 0
    learning_rate_multiplier: float = 1.0
    gate_setting: str = "unchanged"
    sleep_updates: int = 0

    def __post_init__(self) -> None:
        require_identifier(self.action_id, "action_id")
        for name in ("wake_updates", "replay_examples", "sleep_updates"):
            _count(getattr(self, name), name)
        _rate(self.learning_rate_multiplier, "learning_rate_multiplier")
        if type(self.gate_setting) is not str or self.gate_setting not in GATE_SETTINGS:
            raise ValueError("gate_setting must be unchanged, neutral or active")

    @property
    def needs_extended_executor(self) -> bool:
        return (
            self.replay_examples != 0
            or self.learning_rate_multiplier != 1.0
            or self.gate_setting != "unchanged"
            or self.sleep_updates != 0
        )


@dataclass(frozen=True)
class AdaptationEnvelope:
    actions: tuple[AdaptationAction, ...]
    max_decisions: int
    max_wake_updates: int
    max_replay_examples: int = 0
    max_sleep_updates: int = 0
    max_learning_rate_multiplier: float = 1.0
    gate_settings: tuple[str, ...] = ("unchanged",)

    def __post_init__(self) -> None:
        for name in (
            "max_decisions",
            "max_wake_updates",
            "max_replay_examples",
            "max_sleep_updates",
        ):
            _count(getattr(self, name), name)
        _rate(self.max_learning_rate_multiplier, "max_learning_rate_multiplier")
        if (
            type(self.gate_settings) is not tuple
            or not self.gate_settings
            or any(type(g) is not str or g not in GATE_SETTINGS for g in self.gate_settings)
            or len(set(self.gate_settings)) != len(self.gate_settings)
        ):
            raise ValueError("gate_settings must be unique declared finite settings")
        if type(self.actions) is not tuple or not 1 <= len(self.actions) <= 32:
            raise ValueError("actions must be an immutable catalog of 1..32 choices")
        ids = []
        for action in self.actions:
            if type(action) is not AdaptationAction:
                raise ValueError("actions must be exact AdaptationAction records")
            action.__post_init__()
            if (
                action.wake_updates > self.max_wake_updates
                or action.replay_examples > self.max_replay_examples
                or action.sleep_updates > self.max_sleep_updates
                or action.learning_rate_multiplier > self.max_learning_rate_multiplier
                or action.gate_setting not in self.gate_settings
            ):
                raise ValueError("declared action exceeds its hard envelope")
            ids.append(action.action_id)
        if len(set(ids)) != len(ids):
            raise ValueError("action IDs must be unique")

    def select(self, action_id: str) -> AdaptationAction:
        require_identifier(action_id, "action_id")
        for action in self.actions:
            if action.action_id == action_id:
                return action
        raise ValueError("action ID is not in the declared finite catalog")


@dataclass(frozen=True)
class AdaptationProgress:
    decisions_attempted: int = 0
    decisions_returned: int = 0
    decisions_refused: int = 0
    wake_admissions: int = 0
    native_attempts: int = 0
    native_completions: int = 0
    poll_updates_returned: int = 0
    unreturned_work_unknown: bool = False
    replay_admissions: int = 0
    replay_examples_charged: int = 0
    replay_native_attempts: int = 0
    replay_native_completions: int = 0
    replay_commits: int = 0
    replay_publication_unknown: bool = False
    sleep_admissions: int = 0
    sleep_native_attempts: int = 0
    sleep_native_completions: int = 0
    sleep_commits: int = 0
    sleep_publication_unknown: bool = False


@dataclass(frozen=True)
class AdaptationSnapshot:
    envelope: AdaptationEnvelope
    progress: AdaptationProgress


@dataclass(frozen=True)
class AdaptationPoll:
    action_id: str
    requested_wake_updates: int
    training: TrainingPoll
    progress: AdaptationProgress
    requested_replay_examples: int = 0
    replay_receipt: AppliedConsolidation | None = None
    requested_learning_rate_multiplier: float = 1.0
    requested_gate_setting: str = "unchanged"
    requested_sleep_updates: int = 0
    sleep_receipt: AppliedConsolidation | None = None
