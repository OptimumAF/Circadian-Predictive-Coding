"""Post-hoc TRAIN labels with explicit curation and original availability clocks.

Inputs are unchanged raw state42 student trajectories and provisional targets.
Outputs are owned records and arrival-gated pairs, never rewritten actions. No IO,
geometry extraction, teacher queries, simulator, model or legacy clock conversion.
Why: saved trajectories need new query/arrival times without inventing historical
observation ticks or backdating the later curation into the original collection.
"""

from dataclasses import dataclass

from src.core.continuous_demonstration import (
    ContinuousEpisode,
    _text,
    _vector,
    copy_continuous_episode,
)
from src.core.experience import require_tick
from src.core.pick_cube_action_history import _native_action
from src.core.pick_cube_corrections import _UNITS
from src.core.pick_cube_readiness_corrections import (
    _observed_grasp_flags,
)
from src.core.pick_cube_readiness_teacher import (
    GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
)
from src.core.pick_cube_train_context import pick_cube_training_context

PROVENANCE = "provisional_saved_state42_goal_consistent_v1"
GEOMETRY_DEFINITION = "official_state42_relative_world_axes_panda_table_v1"
DEFINITION = GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION


@dataclass(frozen=True)
class CuratedReadinessLabel:
    step: int
    observation: tuple[float, ...]
    target: tuple[float, ...]
    stage: str
    source_observed_at: int | None
    queried_at: int
    arrived_at: int

    def __post_init__(self) -> None:
        if type(self.step) is not int or not 0 <= self.step < 50:
            raise ValueError("curated readiness step must be a native integer in0..49")
        observation = _vector(self.observation, "curated observation", 42)
        target = _vector(self.target, "curated target", 4)
        stage = _text(self.stage, "curated stage", 16)
        if len(observation) != 42:
            raise ValueError("curated readiness labels require an exact state42 observation")
        if len(target) != 4 or any(not -1.0 <= value <= 1.0 for value in target):
            raise ValueError("curated readiness target requires four normalized bounded axes")
        if stage not in ("align", "descend", "grasp", "lift", "settle"):
            raise ValueError("unknown curated geometry-readiness stage")
        if target[3] != (1.0 if stage in ("align", "descend") else -1.0):
            raise ValueError("curated gripper differs from its command stage")
        if stage in ("grasp", "settle") and target[:3] != (0.0, 0.0, 0.0):
            raise ValueError("curated grasp/settle translation must be zero")
        require_tick(self.queried_at, "curated query time")
        require_tick(self.arrived_at, "curated arrival time")
        if self.source_observed_at is not None:
            require_tick(self.source_observed_at, "original observation time")
            if self.source_observed_at > self.queried_at:
                raise ValueError("original observation must precede its later curation query")
        if self.queried_at > self.arrived_at:
            raise ValueError("curated query must precede its label arrival")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "target", target)


def _copy_label(value: CuratedReadinessLabel) -> CuratedReadinessLabel:
    if type(value) is not CuratedReadinessLabel:
        raise ValueError("expected an exact CuratedReadinessLabel")
    return CuratedReadinessLabel(
        value.step,
        value.observation,
        value.target,
        value.stage,
        value.source_observed_at,
        value.queried_at,
        value.arrived_at,
    )


@dataclass(frozen=True)
class CuratedReadinessBatch:
    train_seed: int
    episode: ContinuousEpisode
    labels: tuple[CuratedReadinessLabel, ...]
    source_available_at: int

    def __post_init__(self) -> None:
        context = pick_cube_training_context(self.train_seed)
        require_tick(self.source_available_at, "saved source availability time")
        # Inspect native flags before the episode copy can normalize integers.
        flags = _observed_grasp_flags(self.episode)
        episode = copy_continuous_episode(self.episode)
        if (
            episode.role != "train"
            or episode.context != context
            or episode.episode_id != f"pick-cube-smoke-{self.train_seed}"
            or episode.spec.observation_dim != 42
            or episode.spec.action_units != _UNITS
            or episode.spec.lower_bounds != (-1.0,) * 4
            or episode.spec.upper_bounds != (1.0,) * 4
        ):
            raise ValueError("curation requires the exact executed TRAIN state42 robot episode")
        if type(self.labels) is not tuple or len(self.labels) != len(episode.transitions):
            raise ValueError("one curated label is required per executed row, at most fifty")
        for row in episode.transitions:
            _native_action(row.action)
        labels = tuple(_copy_label(label) for label in self.labels)
        _validate_bindings(episode, labels, flags, self.source_available_at)
        object.__setattr__(self, "episode", episode)
        object.__setattr__(self, "labels", labels)

    def training_pairs(
        self, *, now: int
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
        """Release all arrived targets after revalidating the complete owned source."""
        require_tick(now, "curated label release time")
        owned = copy_curated_readiness_batch(self)
        if any(label.arrived_at > now for label in owned.labels):
            raise ValueError("curated readiness labels have not all arrived")
        return (
            tuple(label.observation for label in owned.labels),
            tuple(label.target for label in owned.labels),
        )


def _validate_bindings(
    episode: ContinuousEpisode,
    labels: tuple[CuratedReadinessLabel, ...],
    flags: tuple[bool, ...],
    available_at: int,
) -> None:
    previous: CuratedReadinessLabel | None = None
    for index, (label, row, grasped) in enumerate(
        zip(labels, episode.transitions, flags, strict=True)
    ):
        if label.step != index or label.observation != row.observation:
            raise ValueError("curated label must bind the exact original step and observation")
        if (label.stage in ("lift", "settle")) is not grasped:
            raise ValueError("curated transport stage must match current observed grasp")
        if (
            label.source_observed_at is not None and label.source_observed_at > available_at
        ) or available_at > label.queried_at:
            raise ValueError("source observation and availability must precede curation query")
        if previous is not None and (
            label.queried_at <= previous.queried_at
            or label.arrived_at <= previous.arrived_at
            or label.queried_at < previous.arrived_at
        ):
            raise ValueError("curation queries and arrivals must progress in original row order")
        previous = label


def copy_curated_readiness_batch(batch: CuratedReadinessBatch) -> CuratedReadinessBatch:
    """Re-own exact records; frozen instances may have been corrupted externally."""
    if type(batch) is not CuratedReadinessBatch:
        raise ValueError("expected an exact CuratedReadinessBatch")
    return CuratedReadinessBatch(
        batch.train_seed, batch.episode, batch.labels, batch.source_available_at
    )
