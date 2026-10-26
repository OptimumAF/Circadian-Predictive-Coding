"""Provisional labels paired with an unchanged executed TRAIN robot trajectory.

Inputs/outputs are owned immutable records and arrival-gated training pairs.
No geometry, simulator, expert qualification, reward relabeling, fitting or IO.
"""

from dataclasses import dataclass

from src.core.continuous_demonstration import (
    ContinuousEpisode,
    _text,
    _vector,
    copy_continuous_episode,
)
from src.core.experience import require_tick
from src.core.pick_cube_train_context import pick_cube_training_context

PROVENANCE = "provisional_fixed_pick_cube_schedule_v1"
_UNITS = (
    "normalized_root_dx",
    "normalized_root_dy",
    "normalized_root_dz",
    "normalized_gripper",
)


@dataclass(frozen=True)
class PickCubeCorrectionLabel:
    step: int
    observation: tuple[float, ...]
    target: tuple[float, ...]
    stage: str

    def __post_init__(self) -> None:
        if type(self.step) is not int or not 0 <= self.step < 50:
            raise ValueError("correction step must be a native integer in0..49")
        observation = _vector(self.observation, "correction observation", 1024)
        target = _vector(self.target, "correction target", 4)
        if len(target) != 4 or any(not -1.0 <= value <= 1.0 for value in target):
            raise ValueError("correction target must have four normalized bounded axes")
        expected = (
            "align"
            if self.step < 8
            else "descend"
            if self.step < 16
            else "grasp"
            if self.step < 22
            else "lift"
            if self.step < 40
            else "settle"
        )
        if _text(self.stage, "correction stage", 16) != expected:
            raise ValueError("correction stage differs from fixed teacher schedule")
        if target[3] != (1.0 if self.step < 16 else -1.0):
            raise ValueError("correction gripper differs from fixed teacher schedule")
        if expected in ("grasp", "settle") and target[:3] != (0.0, 0.0, 0.0):
            raise ValueError("grasp/settle correction translations must be zero")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "target", target)

    @property
    def observed_at(self) -> int:
        return 2 * self.step

    @property
    def labels_arrived_at(self) -> int:
        return 2 * self.step + 1


@dataclass(frozen=True)
class PickCubeCorrectionBatch:
    train_seed: int
    episode: ContinuousEpisode
    labels: tuple[PickCubeCorrectionLabel, ...]

    def __post_init__(self) -> None:
        context = pick_cube_training_context(self.train_seed)
        episode = copy_continuous_episode(self.episode)
        if (
            episode.role != "train"
            or episode.context != context
            or episode.episode_id != f"pick-cube-smoke-{self.train_seed}"
            or episode.spec.action_units != _UNITS
            or episode.spec.lower_bounds != (-1.0,) * 4
            or episode.spec.upper_bounds != (1.0,) * 4
        ):
            raise ValueError("corrections require the declared executed TRAIN robot episode")
        if (
            type(self.labels) is not tuple
            or not 1 <= len(self.labels) <= 50
            or len(self.labels) != len(episode.transitions)
        ):
            raise ValueError("one correction is required for every executed transition, at most50")
        labels = []
        for index, (label, record) in enumerate(zip(self.labels, episode.transitions, strict=True)):
            if type(label) is not PickCubeCorrectionLabel:
                raise ValueError("expected exact PickCubeCorrectionLabel records")
            owned = PickCubeCorrectionLabel(
                label.step, label.observation, label.target, label.stage
            )
            if owned.step != index or owned.observation != record.observation:
                raise ValueError(
                    "correction must match the exact executed step and public observation"
                )
            labels.append(owned)
        object.__setattr__(self, "episode", episode)
        object.__setattr__(self, "labels", tuple(labels))

    def training_pairs(self, *, now: int):
        """Release arrived provisional targets, never a rewritten expert episode."""
        require_tick(now, "correction release time")
        # Why: frozen records can be corrupted through object.__setattr__; re-own
        # at the release boundary before exposing any target to an optimizer.
        owned = PickCubeCorrectionBatch(self.train_seed, self.episode, self.labels)
        if any(label.labels_arrived_at > now for label in owned.labels):
            raise ValueError("correction labels have not all arrived")
        return (
            tuple(label.observation for label in owned.labels),
            tuple(label.target for label in owned.labels),
        )
