"""Distinct provisional readiness labels on an unchanged TRAIN student trace.

Own finite public observation/target pairs and release them after their original
label ticks. No geometry, expert qualification, reward relabeling, learner or IO.
The original fixed-schedule schema remains unchanged: these stages are geometric,
so they cannot be inferred from elapsed student steps.
"""

from dataclasses import dataclass

from src.core.continuous_demonstration import (
    ContinuousEpisode,
    ContinuousActionSpec,
    ContinuousTransition,
    _text,
    _vector,
    copy_continuous_episode,
)
from src.core.experience import require_tick
from src.core.pick_cube_corrections import _UNITS
from src.core.pick_cube_train_context import pick_cube_training_context
from src.core.pick_cube_readiness_teacher import (
    DEFINITION,
    OBSERVED_GRASP_DEFINITION,
    GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
)

PROVENANCE = "provisional_geometry_readiness_v1"
OBSERVED_GRASP_PROVENANCE = "provisional_geometry_observed_grasp_readiness_v1"
GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE = "provisional_geometry_observed_grasp_goal_consistent_v2"


@dataclass(frozen=True)
class ReadinessCorrectionLabel:
    step: int
    observation: tuple[float, ...]
    target: tuple[float, ...]
    stage: str

    def __post_init__(self) -> None:
        if type(self.step) is not int or not 0 <= self.step < 50:
            raise ValueError("readiness correction step must be a native integer in0..49")
        observation = _vector(self.observation, "readiness correction observation", 1024)
        target = _vector(self.target, "readiness correction target", 4)
        stage = _text(self.stage, "readiness correction stage", 16)
        if len(target) != 4 or any(not -1.0 <= value <= 1.0 for value in target):
            raise ValueError("readiness target must have four normalized bounded axes")
        if stage not in ("align", "descend", "grasp", "lift", "settle"):
            raise ValueError("unknown geometry-readiness correction stage")
        if target[3] != (1.0 if stage in ("align", "descend") else -1.0):
            raise ValueError("readiness correction gripper differs from its command stage")
        if stage in ("grasp", "settle") and target[:3] != (0.0, 0.0, 0.0):
            raise ValueError("readiness grasp/settle translation must be zero")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "target", target)

    @property
    def observed_at(self) -> int:
        return 2 * self.step

    @property
    def labels_arrived_at(self) -> int:
        return 2 * self.step + 1


@dataclass(frozen=True)
class ReadinessCorrectionBatch:
    train_seed: int
    episode: ContinuousEpisode
    labels: tuple[ReadinessCorrectionLabel, ...]

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
            raise ValueError("readiness corrections require the executed TRAIN robot episode")
        if (
            type(self.labels) is not tuple
            or not 1 <= len(self.labels) <= 50
            or len(self.labels) != len(episode.transitions)
        ):
            raise ValueError("one readiness label is required per executed transition, at most50")
        labels = []
        for index, (label, row) in enumerate(zip(self.labels, episode.transitions, strict=True)):
            if type(label) is not ReadinessCorrectionLabel:
                raise ValueError("expected exact ReadinessCorrectionLabel records")
            owned = ReadinessCorrectionLabel(
                label.step, label.observation, label.target, label.stage
            )
            if owned.step != index or owned.observation != row.observation:
                raise ValueError("readiness label must match the exact executed public observation")
            labels.append(owned)
        object.__setattr__(self, "episode", episode)
        object.__setattr__(self, "labels", tuple(labels))

    def training_pairs(self, *, now: int):
        """Revalidate ownership and original arrival; never rewrite student actions."""
        require_tick(now, "readiness correction release time")
        owned = ReadinessCorrectionBatch(self.train_seed, self.episode, self.labels)
        if any(label.labels_arrived_at > now for label in owned.labels):
            raise ValueError("readiness correction labels have not all arrived")
        return (
            tuple(label.observation for label in owned.labels),
            tuple(label.target for label in owned.labels),
        )


def _observed_grasp_flags(episode: ContinuousEpisode) -> tuple[bool, ...]:
    """Validate original owned row flags before copying can normalize corrupt values."""
    if type(episode) is not ContinuousEpisode or type(episode.spec) is not ContinuousActionSpec:
        raise ValueError("observed grasp corrections require an exact continuous episode/spec")
    dimensions = episode.spec.observation_dim
    if type(dimensions) is not int or dimensions not in (12, 42, 47):
        raise ValueError("observed grasp requires official state42, history47 or taskspace12")
    index = 0 if dimensions == 12 else 18
    if type(episode.transitions) is not tuple or not 1 <= len(episode.transitions) <= 50:
        raise ValueError("observed grasp requires one to fifty exact transitions")
    flags = []
    for row in episode.transitions:
        if type(row) is not ContinuousTransition:
            raise ValueError("observed grasp requires exact transition records")
        for observation in (row.observation, row.next_observation):
            if (
                type(observation) is not tuple
                or len(observation) != dimensions
                or type(observation[index]) is not float
                or observation[index] not in (0.0, 1.0)
            ):
                raise ValueError("official current/next grasp flag must be native float0 or1")
        flags.append(row.observation[index] == 1.0)
    return tuple(flags)


@dataclass(frozen=True)
class ObservedGraspCorrectionBatch(ReadinessCorrectionBatch):
    """Distinct current-observation feedback; next-state feedback may change normally."""

    observed_grasp: tuple[bool, ...]

    def __post_init__(self) -> None:
        flags = _observed_grasp_flags(self.episode)
        if (
            type(self.observed_grasp) is not tuple
            or len(self.observed_grasp) != len(flags)
            or any(type(value) is not bool for value in self.observed_grasp)
            or self.observed_grasp != flags
        ):
            raise ValueError("observed grasp feedback must exactly match each current row flag")
        super().__post_init__()
        for grasped, label in zip(self.observed_grasp, self.labels, strict=True):
            if (label.stage in ("lift", "settle")) is not grasped:
                raise ValueError(
                    "observed grasp feedback differs from its transport/approach stage"
                )
        object.__setattr__(self, "observed_grasp", tuple(value for value in self.observed_grasp))

    def training_pairs(self, *, now: int):
        """Revalidate the selected family and feedback before original arrival release."""
        require_tick(now, "readiness correction release time")
        owned = copy_readiness_correction_batch(self)
        if any(label.labels_arrived_at > now for label in owned.labels):
            raise ValueError("readiness correction labels have not all arrived")
        return (
            tuple(label.observation for label in owned.labels),
            tuple(label.target for label in owned.labels),
        )


@dataclass(frozen=True)
class GoalConsistentObservedGraspCorrectionBatch(ObservedGraspCorrectionBatch):
    """Versioned Euclidean transport labels; original feedback rules are unchanged."""


def copy_readiness_correction_batch(
    batch: ReadinessCorrectionBatch, *, episode=None, labels=None
) -> ReadinessCorrectionBatch:
    """Own an exact known family; feature views retain the original feedback tuple."""
    owned: ReadinessCorrectionBatch
    if type(batch) is GoalConsistentObservedGraspCorrectionBatch:
        owned = GoalConsistentObservedGraspCorrectionBatch(
            batch.train_seed, batch.episode, batch.labels, batch.observed_grasp
        )
    elif type(batch) is ObservedGraspCorrectionBatch:
        owned = ObservedGraspCorrectionBatch(
            batch.train_seed, batch.episode, batch.labels, batch.observed_grasp
        )
    elif type(batch) is ReadinessCorrectionBatch:
        owned = ReadinessCorrectionBatch(batch.train_seed, batch.episode, batch.labels)
    else:
        raise ValueError("expected an exact known readiness correction batch")
    if episode is None and labels is None:
        return owned
    target_episode = owned.episode if episode is None else episode
    target_labels = owned.labels if labels is None else labels
    if type(owned) is GoalConsistentObservedGraspCorrectionBatch:
        return GoalConsistentObservedGraspCorrectionBatch(
            owned.train_seed, target_episode, target_labels, owned.observed_grasp
        )
    if type(owned) is ObservedGraspCorrectionBatch:
        return ObservedGraspCorrectionBatch(
            owned.train_seed, target_episode, target_labels, owned.observed_grasp
        )
    return ReadinessCorrectionBatch(owned.train_seed, target_episode, target_labels)


def readiness_correction_provenance(batch: ReadinessCorrectionBatch) -> str:
    owned = copy_readiness_correction_batch(batch)
    if type(owned) is GoalConsistentObservedGraspCorrectionBatch:
        return GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE
    return OBSERVED_GRASP_PROVENANCE if type(owned) is ObservedGraspCorrectionBatch else PROVENANCE


def readiness_correction_definition(batch: ReadinessCorrectionBatch) -> str:
    owned = copy_readiness_correction_batch(batch)
    if type(owned) is GoalConsistentObservedGraspCorrectionBatch:
        return GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
    return OBSERVED_GRASP_DEFINITION if type(owned) is ObservedGraspCorrectionBatch else DEFINITION
