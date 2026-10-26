"""Provisional PushCube labels bound to actual student-visited physical states.

Inputs are observed states, executed student transitions and already returned
geometry labels. Outputs are owned release-gated targets, never a qualified
expert trajectory. No teacher decision, geometry query, model, reward inference,
role conversion, simulation or IO occurs here.
"""

from dataclasses import dataclass

from src.core.continuous_demonstration import (
    ContinuousEpisode,
    ContinuousTransition,
    _vector,
    copy_continuous_episode,
)
from src.core.cube_taskspace_state import (
    CubeTaskSpaceState,
    copy_cube_taskspace_state,
    cube_taskspace_observation,
    cube_taskspace_spec,
)
from src.core.experience import require_tick
from src.core.pick_cube_action_history import ActionHistoryState, advance_action_history
from src.core.push_cube_teacher import PushCubeDecision

PROVENANCE = "provisional_push_cube_on_policy_geometry_v1"


@dataclass(frozen=True)
class PushCubeCorrectionLabel:
    step: int
    state: CubeTaskSpaceState
    observation: tuple[float, ...]
    target: tuple[float, ...]
    stage: str

    def __post_init__(self) -> None:
        if type(self.step) is not int or not 0 <= self.step < 50:
            raise ValueError("PushCube correction step must be a native integer in0..49")
        state = copy_cube_taskspace_state(self.state)
        observation = _vector(self.observation, "PushCube correction observation", 12)
        if len(observation) != 12:
            raise ValueError("PushCube correction observation must have exactly12 values")
        decision = PushCubeDecision(self.stage, self.target)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "target", decision.action)

    @property
    def observed_at(self) -> int:
        return 2 * self.step

    @property
    def labels_arrived_at(self) -> int:
        return 2 * self.step + 1


def _original_episode(episode: ContinuousEpisode, seed: int) -> None:
    if (
        type(episode) is not ContinuousEpisode
        or type(episode.role) is not str
        or episode.role != "train"
        or episode.context != f"push-cube-train-{seed}"
        or episode.episode_id != f"push-cube-student-{seed}"
        or episode.spec != cube_taskspace_spec()
        or type(episode.transitions) is not tuple
        or not 1 <= len(episode.transitions) <= 50
    ):
        raise ValueError("PushCube corrections require the complete declared TRAIN student episode")
    # Why: inspect original flags before an episode copier can normalize forged
    # int/bool values into floats. The physical port emits observed float0/1.
    history = ActionHistoryState()
    for row in episode.transitions:
        if type(row) is not ContinuousTransition:
            raise ValueError("PushCube corrections require exact native transition records")
        for observation in (row.observation, row.next_observation):
            if (
                type(observation) is not tuple
                or len(observation) != 12
                or type(observation[0]) is not float
                or observation[0] not in (0.0, 1.0)
            ):
                raise ValueError("PushCube grasp flags must be original observed float0/1")
        history = advance_action_history(history, row.action)


def _copy_label(label: PushCubeCorrectionLabel) -> PushCubeCorrectionLabel:
    if type(label) is not PushCubeCorrectionLabel:
        raise ValueError("expected an exact PushCubeCorrectionLabel")
    return PushCubeCorrectionLabel(
        label.step, label.state, label.observation, label.target, label.stage
    )


@dataclass(frozen=True)
class PushCubeCorrectionBatch:
    seed: int
    episode: ContinuousEpisode
    labels: tuple[PushCubeCorrectionLabel, ...]

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in (17, 19, 20):
            raise ValueError("PushCube corrections allow only TRAIN seeds17,19,20")
        _original_episode(self.episode, self.seed)
        if (
            type(self.labels) is not tuple
            or not 1 <= len(self.labels) <= 50
            or len(self.labels) != len(self.episode.transitions)
        ):
            raise ValueError("one PushCube label is required per student transition, at most50")
        episode = copy_continuous_episode(self.episode)
        labels = tuple(_copy_label(label) for label in self.labels)
        history = ActionHistoryState()
        for index, (label, row) in enumerate(zip(labels, episode.transitions, strict=True)):
            expected = cube_taskspace_observation(label.state, history)
            if (
                label.step != index
                or row.step != index
                or label.observation != expected
                or row.observation != expected
            ):
                raise ValueError(
                    "PushCube label state must match its actual student observation/history"
                )
            history = advance_action_history(history, row.action)
            if row.next_observation[7:] != (*history.previous_action, history.closing_count / 6):
                raise ValueError(
                    "PushCube next observation must retain actual executed student history"
                )
        object.__setattr__(self, "episode", episode)
        object.__setattr__(self, "labels", labels)

    def _released(self, now: int) -> "PushCubeCorrectionBatch":
        require_tick(now, "PushCube correction release time")
        owned = PushCubeCorrectionBatch(self.seed, self.episode, self.labels)
        if any(label.labels_arrived_at > now for label in owned.labels):
            raise ValueError("PushCube correction labels have not all arrived")
        return owned

    def training_pairs(self, *, now: int):
        """Return released targets without changing original student commands."""
        owned = self._released(now)
        return (
            tuple(label.observation for label in owned.labels),
            tuple(label.target for label in owned.labels),
        )

    def training_episode(self, *, now: int) -> ContinuousEpisode:
        """Return an explicit target packet, not a native expert trajectory.

        Why: observations retain student-command history even though packet
        actions are teacher targets. Replaying targets did not produce this path.
        Original student commands, physical states and rewards remain unchanged.
        """
        owned = self._released(now)
        rows = tuple(
            ContinuousTransition(
                row.step,
                row.observation,
                label.target,
                row.reward,
                row.next_observation,
                row.terminated,
                row.truncated,
            )
            for row, label in zip(owned.episode.transitions, owned.labels, strict=True)
        )
        return ContinuousEpisode(
            f"push-cube-correction-targets-{owned.seed}",
            owned.episode.context,
            "train",
            owned.episode.spec,
            rows,
        )
