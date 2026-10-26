"""One bounded TRAIN PushCube supervision collection through the existing port.

Inputs: a fixed admitted TRAIN seed. Output: explicit result, owned physical
inputs/labels/returned transitions and paid work, including failures. Only final
native success permits a training export. No learner, fitting, evaluation-role
conversion, retry, deployment, serialization or new stream orchestration.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.continuous_demonstration import (
    ContinuousEpisode,
    ContinuousTransition,
    copy_continuous_episode,
)
from src.core.cube_taskspace_state import (
    CUBE_TASKSPACE_DEFINITION,
    CubeTaskSpaceState,
    copy_cube_taskspace_state,
    cube_taskspace_observation,
    cube_taskspace_spec,
)
from src.core.pick_cube_action_history import ActionHistoryState, advance_action_history
from src.core.push_cube_teacher import DEFINITION, PushCubeDecision, PushCubeTeacher
from src.infra.maniskill_push_cube import ManiSkillPushCubeEpisode, PushCubeWork


def _native_command(action: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(value) for value in np.array(action, dtype=np.float32))


def _check_input(port, state, history, seed) -> None:
    if (
        port.seed != seed
        or port.role != "train"
        or port.context != f"push-cube-train-{seed}"
        or port.spec != cube_taskspace_spec()
        or port.observation != cube_taskspace_observation(state, history)
    ):
        raise ValueError("Push teacher input differs from its admitted physical TRAIN state")


@dataclass(frozen=True)
class PushCubeTeacherResult:
    episode: ContinuousEpisode | None
    definition: str
    seed: int
    role: str
    states: tuple[CubeTaskSpaceState, ...]
    decisions: tuple[PushCubeDecision, ...]
    transitions: tuple[ContinuousTransition, ...]
    attempted_actions: tuple[tuple[float, ...], ...]
    returned_actions: tuple[tuple[float, ...], ...]
    decision_attempts: int
    decision_returns: int
    native_work: PushCubeWork | None
    metadata: dict[str, Any] | None
    error: str | None
    unreturned_work_possible: bool

    @property
    def final_success(self) -> bool:
        return bool(
            self.transitions
            and (self.transitions[-1].terminated or self.transitions[-1].truncated)
            and self.transitions[-1].reward == 1.0
        )

    def training_episode(self) -> ContinuousEpisode:
        """Release successful complete TRAIN data without re-querying the oracle."""
        if self.error is not None or self.unreturned_work_possible or not self.final_success:
            raise ValueError("Push supervision has not qualified for training")
        if (
            self.definition != DEFINITION
            or type(self.seed) is not int
            or self.seed not in (17, 19, 20)
            or self.role != "train"
            or type(self.episode) is not ContinuousEpisode
            or self.metadata is None
            or self.metadata.get("task_id") != "PushCube-v1"
            or self.metadata.get("seed") != self.seed
            or self.metadata.get("role") != "train"
        ):
            raise ValueError("Push supervision lost its original task/role binding")
        episode = copy_continuous_episode(self.episode)
        count = len(self.transitions)
        work = self.native_work
        if (
            episode.episode_id != f"push-cube-teacher-{self.seed}"
            or episode.context != f"push-cube-train-{self.seed}"
            or episode.role != "train"
            or episode.spec != cube_taskspace_spec()
            or episode.transitions != self.transitions
            or not 1 <= count <= 50
            or self.decision_attempts != count
            or self.decision_returns != count
            or len(self.states) != count
            or len(self.decisions) != count
            or type(work) is not PushCubeWork
            or work
            != PushCubeWork(
                1,
                1,
                1,
                1,
                count,
                count,
                count * 5,
                0,
                count + 1,
                count + 1,
                count + 1,
                count + 1,
                2 * (count + 1),
                2 * (count + 1),
                1,
                1,
            )
        ):
            raise ValueError("Push supervision differs from its complete paid collection")
        history = ActionHistoryState()
        actions = tuple(row.action for row in self.transitions)
        if self.attempted_actions != actions or self.returned_actions != actions:
            raise ValueError("Push supervision native command receipts changed")
        for state, decision, row in zip(self.states, self.decisions, self.transitions, strict=True):
            if (
                type(decision) is not PushCubeDecision
                or PushCubeDecision(decision.stage, decision.action) != decision
                or _native_command(decision.action) != row.action
                or cube_taskspace_observation(state, history) != row.observation
            ):
                raise ValueError("Push supervision label/input differs from its executed row")
            history = advance_action_history(history, row.action)
            if row.next_observation[7:] != (*history.previous_action, history.closing_count / 6):
                raise ValueError("Push supervision executed action history changed")
        return episode


def collect_push_cube_teacher(*, seed: int = 17, role: str = "train") -> PushCubeTeacherResult:
    """Collect once; complete negatives and partial failures keep all known costs."""
    if (
        type(seed) is not int
        or seed not in (17, 19, 20)
        or type(role) is not str
        or role != "train"
    ):
        raise ValueError("Push supervision admits only TRAIN seeds 17/19/20")
    states: list[CubeTaskSpaceState] = []
    decisions: list[PushCubeDecision] = []
    transitions: list[ContinuousTransition] = []
    attempted: list[tuple[float, ...]] = []
    returned: list[tuple[float, ...]] = []
    attempts = returns = 0
    port = None
    episode = None
    metadata = None
    work = None
    error_text = None
    unknown = False
    history = ActionHistoryState()
    teacher = PushCubeTeacher()
    try:
        port = ManiSkillPushCubeEpisode(f"push-cube-teacher-{seed}", seed=seed, role=role)
        metadata = deepcopy(port.metadata)
        if (
            metadata.get("task_id") != "PushCube-v1"
            or metadata.get("context") != f"push-cube-train-{seed}"
            or metadata.get("seed") != seed
            or metadata.get("role") != "train"
            or metadata.get("horizon") != 50
            or metadata.get("control_physics_steps_per_control") != 5
            or metadata.get("recipe", {}).get("definition") != CUBE_TASKSPACE_DEFINITION
        ):
            raise ValueError(
                "Push teacher native metadata differs from its frozen task/physics contract"
            )
        for _ in range(50):
            state = copy_cube_taskspace_state(port.state)
            _check_input(port, state, history, seed)
            if teacher.steps != returns or (decisions and teacher.phase != decisions[-1].stage):
                raise ValueError("Push teacher preceding decision count changed")
            states.append(copy_cube_taskspace_state(state))
            attempts += 1
            decision = teacher.decide(state, history=history)
            returns += 1
            if (
                type(decision) is not PushCubeDecision
                or teacher.steps != returns
                or teacher.phase != decision.stage
            ):
                raise ValueError("Push teacher returned an inconsistent decision")
            decision = PushCubeDecision(decision.stage, decision.action)
            decisions.append(decision)
            _check_input(port, state, history, seed)
            sent = _native_command(decision.action)
            attempted.append(sent)
            row = port.step(decision.action)
            if type(row) is not ContinuousTransition:
                unknown = True
                raise ValueError("Push teacher control did not return a native transition")
            row = ContinuousTransition(
                row.step,
                row.observation,
                row.action,
                row.reward,
                row.next_observation,
                row.terminated,
                row.truncated,
            )
            returned.append(row.action)
            if (
                row.step != len(transitions)
                or row.action != sent
                or row.observation != cube_taskspace_observation(state, history)
                or (transitions and transitions[-1].next_observation != row.observation)
            ):
                raise ValueError("Push teacher control returned an inconsistent transition")
            history = advance_action_history(history, row.action)
            if row.next_observation[7:] != (*history.previous_action, history.closing_count / 6):
                raise ValueError(
                    "Push teacher returned history differs from executed native action"
                )
            transitions.append(row)
            current = copy_cube_taskspace_state(port.state)
            _check_input(port, current, history, seed)
            if row.next_observation != cube_taskspace_observation(current, history):
                raise ValueError(
                    "Push teacher returned next observation differs from current native state"
                )
            if row.terminated or row.truncated:
                break
        episode = port.episode()
        if episode.transitions != tuple(transitions):
            raise ValueError("Push teacher exported a different native transition prefix")
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
        work = getattr(error, "push_cube_work", port.work if port is not None else None)
        attempted = list(getattr(error, "push_cube_attempted_actions", attempted))
        returned = list(getattr(error, "push_cube_returned_actions", returned))
        unknown = (
            unknown
            or attempts > returns
            or work is None
            or any(
                first > last
                for first, last in (
                    (work.construction_attempts, work.construction_returns),
                    (work.reset_attempts, work.reset_returns),
                    (work.step_attempts, work.completed_steps),
                    (work.grasp_query_attempts, work.grasp_query_returns),
                )
            )
        )
    finally:
        if port is not None:
            try:
                port.close()
            except Exception as error:
                message = f"cleanup {type(error).__name__}: {error}"
                error_text = message if error_text is None else f"{error_text}; {message}"
                unknown = True
            work = port.work
    return PushCubeTeacherResult(
        episode,
        DEFINITION,
        seed,
        role,
        tuple(states),
        tuple(decisions),
        tuple(transitions),
        tuple(attempted),
        tuple(returned),
        attempts,
        returns,
        work,
        metadata,
        error_text,
        unknown,
    )
