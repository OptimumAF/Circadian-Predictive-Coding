"""TRAIN-only fixed or readiness teacher queries in a frozen robot rollout.

Privileged geometry stays here. Queries never execute teacher actions, reset an
environment, fit a learner, qualify an expert or publish a serving model.
"""

from dataclasses import asdict
from typing import cast

from src.core.continuous_demonstration import ContinuousEpisode, _vector
from src.core.pick_cube_corrections import (
    PROVENANCE,
    PickCubeCorrectionBatch,
    PickCubeCorrectionLabel,
)
from src.core.pick_cube_teacher import PickCubeTeacher
from src.core.pick_cube_readiness_teacher import (
    DEFINITION as READINESS_DEFINITION,
    ReadinessDecision,
    ReadinessPickCubeTeacher,
    ObservedGraspPickCubeTeacher,
    OBSERVED_GRASP_DEFINITION,
    GoalConsistentObservedGraspPickCubeTeacher,
    GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
)
from src.core.pick_cube_readiness_corrections import (
    PROVENANCE as READINESS_PROVENANCE,
    ReadinessCorrectionBatch,
    ReadinessCorrectionLabel,
    ObservedGraspCorrectionBatch,
    OBSERVED_GRASP_PROVENANCE,
    GoalConsistentObservedGraspCorrectionBatch,
    GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE,
)
from src.core.pick_cube_train_context import pick_cube_training_context
from src.infra.maniskill_continuous import ACTION_UNITS, ManiSkillPickCubeEpisode
from src.infra.maniskill_task_teacher import _geometry


def _official_grasp(observation: tuple[float, ...]) -> bool:
    """Read current official state feedback; commands never establish contact."""
    if (
        type(observation) is not tuple
        or len(observation) != 42
        or type(observation[18]) is not float
        or observation[18] not in (0.0, 1.0)
    ):
        raise ValueError("observed grasp requires official raw42 float0/1 feedback")
    return observation[18] == 1.0


class PickCubeCorrectionQuery:
    """One bounded query sequence; any failed query permanently closes admission."""

    def __init__(
        self,
        train_seed: int,
        *,
        readiness_corrections: bool = False,
        observed_grasp_corrections: bool = False,
        goal_consistent_corrections: bool = False,
    ) -> None:
        if type(readiness_corrections) is not bool:
            raise ValueError("readiness_corrections must be an exact bool")
        if type(observed_grasp_corrections) is not bool:
            raise ValueError("observed_grasp_corrections must be an exact bool")
        if observed_grasp_corrections and not readiness_corrections:
            raise ValueError("observed grasp requires explicit readiness corrections")
        if type(goal_consistent_corrections) is not bool:
            raise ValueError("goal_consistent_corrections must be an exact bool")
        if goal_consistent_corrections and not (
            readiness_corrections and observed_grasp_corrections
        ):
            raise ValueError("goal-consistent corrections require observed-grasp readiness")
        self._goal_consistent_corrections = self._admitted_goal_consistent = (
            goal_consistent_corrections
        )
        self._context = pick_cube_training_context(train_seed)
        self._seed = train_seed
        self._readiness_corrections = readiness_corrections
        self._observed_grasp_corrections = self._admitted_observed_grasp = (
            observed_grasp_corrections
        )
        self._observed_grasp_receipts: tuple[bool, ...] = ()
        self._teacher = (
            GoalConsistentObservedGraspPickCubeTeacher()
            if goal_consistent_corrections
            else ObservedGraspPickCubeTeacher()
            if observed_grasp_corrections
            else ReadinessPickCubeTeacher()
            if readiness_corrections
            else PickCubeTeacher()
        )
        self._labels: list[PickCubeCorrectionLabel | ReadinessCorrectionLabel] = []
        self._attempts = self._geometry_returns = 0
        self._teacher_attempts = self._teacher_returns = 0
        self._failed = False
        self._port: ManiSkillPickCubeEpisode | None = None
        self._admitted_readiness = readiness_corrections
        self._admitted_teacher = self._teacher
        self._readiness_port: ManiSkillPickCubeEpisode | None = None
        self._readiness_state = (0, "align", 0)
        self._readiness_paid: tuple[int, ...] = (0, 0, 0, 0)
        self._readiness_receipts: tuple[tuple, ...] = ()

    def _check_readiness_binding(self) -> None:
        if not self._admitted_readiness:
            return
        expected_teacher = (
            GoalConsistentObservedGraspPickCubeTeacher
            if self._admitted_goal_consistent
            else ObservedGraspPickCubeTeacher
            if self._admitted_observed_grasp
            else ReadinessPickCubeTeacher
        )
        if type(self._admitted_teacher) is not expected_teacher:
            self._failed = True
            raise ValueError("admitted readiness correction teacher changed type")
        teacher = self._admitted_teacher
        state = (teacher._steps, teacher._phase, teacher._closing_steps)
        paid = (
            self._attempts,
            self._geometry_returns,
            self._teacher_attempts,
            self._teacher_returns,
        )
        if (
            self._readiness_corrections is not True
            or type(self._admitted_goal_consistent) is not bool
            or self._goal_consistent_corrections is not self._admitted_goal_consistent
            or (self._admitted_goal_consistent and not self._admitted_observed_grasp)
            or type(self._admitted_observed_grasp) is not bool
            or self._observed_grasp_corrections is not self._admitted_observed_grasp
            or self._teacher is not teacher
            or self._port is not self._readiness_port
            or type(teacher) is not expected_teacher
            or type(state[0]) is not int
            or type(state[1]) is not str
            or type(state[2]) is not int
            or state != self._readiness_state
            or any(type(count) is not int for count in paid)
            or paid != self._readiness_paid
            or type(self._labels) is not list
            or len(self._labels) != len(self._readiness_receipts)
        ):
            self._failed = True
            raise ValueError("admitted readiness correction binding changed")
        if self._admitted_observed_grasp and (
            type(self._observed_grasp_receipts) is not tuple
            or len(self._observed_grasp_receipts) != len(self._labels)
            or any(type(value) is not bool for value in self._observed_grasp_receipts)
        ):
            self._failed = True
            raise ValueError("original observed grasp feedback changed")
        for label, receipt in zip(self._labels, self._readiness_receipts, strict=True):
            if (
                type(label) is not ReadinessCorrectionLabel
                or type(label.step) is not int
                or type(label.observation) is not tuple
                or type(label.target) is not tuple
                or any(type(value) is not float for value in label.observation + label.target)
                or type(label.stage) is not str
                or (label.step, label.observation, label.target, label.stage) != receipt
            ):
                self._failed = True
                raise ValueError("preceding readiness correction label changed")
        if self._admitted_observed_grasp:
            for label, grasped in zip(self._labels, self._observed_grasp_receipts, strict=True):
                if (
                    _official_grasp(label.observation) is not grasped
                    or (label.stage in ("lift", "settle")) is not grasped
                ):
                    self._failed = True
                    raise ValueError("preceding observed grasp label binding changed")

    def _pay_readiness(self, index: int) -> None:
        if self._admitted_readiness:
            paid = list(self._readiness_paid)
            paid[index] += 1
            self._readiness_paid = tuple(paid)

    def _freeze_readiness_decision(self, stage: str) -> None:
        expected_teacher = (
            GoalConsistentObservedGraspPickCubeTeacher
            if self._admitted_goal_consistent
            else ObservedGraspPickCubeTeacher
            if self._admitted_observed_grasp
            else ReadinessPickCubeTeacher
        )
        if type(self._admitted_teacher) is not expected_teacher:
            raise ValueError("admitted readiness correction teacher changed type")
        teacher = self._admitted_teacher
        if self._readiness_corrections is not True or self._teacher is not teacher:
            raise ValueError("admitted readiness correction teacher was replaced")
        state = (teacher._steps, teacher._phase, teacher._closing_steps)
        if (
            type(teacher) is not expected_teacher
            or type(state[0]) is not int
            or state[0] != self._readiness_state[0] + 1
            or not 1 <= state[0] <= 50
            or type(state[1]) is not str
            or state[1] != stage
            or state[1] not in ("align", "descend", "grasp", "lift", "settle")
            or type(state[2]) is not int
            or not 0 <= state[2] <= 6
        ):
            raise ValueError("readiness decision returned invalid scalar state")
        self._readiness_state = state
        self._check_readiness_binding()

    def _check_port(self, port, step, observation, *, label_offset: int = 0) -> None:
        if (
            type(port) is not ManiSkillPickCubeEpisode
            or type(port._seed) is not int
            or port._seed != self._seed
            or port._role != "train"
            or port._context != self._context
            or port._episode_id != f"pick-cube-smoke-{self._seed}"
            or port._closed
            or port._failed
            or (self._port is not None and port is not self._port)
            or type(step) is not int
            or step != len(self._labels) - label_offset
            or not 0 <= step < 50
            or len(port._records) != step
            or port.work.completed_steps != step
            or (port._records and (port._records[-1].terminated or port._records[-1].truncated))
            or port.observation != observation
        ):
            raise ValueError("correction query must match the current TRAIN port step/observation")
        _vector(port.observation, "current correction observation", 1024)
        if self._admitted_observed_grasp:
            _official_grasp(observation)
        spec = port.spec
        if (
            spec.action_units != ACTION_UNITS
            or spec.lower_bounds != (-1.0,) * 4
            or spec.upper_bounds != (1.0,) * 4
            or len(observation) != spec.observation_dim
        ):
            raise ValueError("correction query requires the normalized public robot spec")
        # Finish from owned native authority after all public metadata getters.
        if (
            type(port) is not ManiSkillPickCubeEpisode
            or type(port._seed) is not int
            or port._seed != self._seed
            or port._role != "train"
            or port._context != self._context
            or port._episode_id != f"pick-cube-smoke-{self._seed}"
            or (self._port is not None and port is not self._port)
            or port._closed
            or port._failed
            or len(port._records) != step
            or (port._records and (port._records[-1].terminated or port._records[-1].truncated))
        ):
            raise ValueError("correction port authority changed during metadata observation")
        if self._admitted_observed_grasp:
            if (
                _official_grasp(port._observation) is not _official_grasp(observation)
                or port._observation != observation
            ):
                raise ValueError("current observed grasp changed during metadata observation")

    def query(self, port, step: int, observation: tuple[float, ...]) -> None:
        if self._failed:
            raise ValueError("failed correction sequence cannot resume")
        try:
            self._check_readiness_binding()
            if self._admitted_observed_grasp:
                _official_grasp(observation)
            observation = _vector(observation, "correction query observation", 1024)
            self._check_port(port, step, observation)
            self._check_readiness_binding()
            if self._teacher.steps != step:
                raise ValueError("correction teacher schedule changed")
            self._check_readiness_binding()
            self._port = port
            if self._admitted_readiness:
                self._readiness_port = port
            self._attempts += 1
            self._pay_readiness(0)
            geometry = _geometry(port)
            self._geometry_returns += 1
            self._pay_readiness(1)
            self._check_readiness_binding()
            self._check_port(port, step, observation)
            self._check_readiness_binding()
            if self._readiness_corrections:
                expected_teacher = (
                    GoalConsistentObservedGraspPickCubeTeacher
                    if self._admitted_goal_consistent
                    else ObservedGraspPickCubeTeacher
                    if self._admitted_observed_grasp
                    else ReadinessPickCubeTeacher
                )
                if type(self._teacher) is not expected_teacher:
                    raise ValueError("readiness correction teacher was replaced")
                self._teacher_attempts += 1
                self._pay_readiness(2)
                executed_gripper = port._records[-1].action[3] if step else 1.0
                if self._admitted_observed_grasp:
                    decision = cast(ObservedGraspPickCubeTeacher, self._teacher).decide(
                        geometry,
                        grasped=_official_grasp(observation),
                        executed_gripper=executed_gripper,
                    )
                else:
                    decision = cast(ReadinessPickCubeTeacher, self._teacher).decide(
                        geometry,
                        executed_gripper=executed_gripper,
                    )
                self._teacher_returns += 1
                self._pay_readiness(3)
                if type(decision) is not ReadinessDecision:
                    raise ValueError("readiness correction returned an unexpected decision")
                stage, target = decision.stage, decision.action
                self._freeze_readiness_decision(stage)
            else:
                if type(self._teacher) is not PickCubeTeacher:
                    raise ValueError("fixed correction teacher was replaced")
                stage = self._teacher.stage
                self._teacher_attempts += 1
                target = self._teacher.action(geometry)
                self._teacher_returns += 1
            label = (
                ReadinessCorrectionLabel(step, observation, target, stage)
                if self._readiness_corrections
                else PickCubeCorrectionLabel(step, observation, target, stage)
            )
            receipt = (label.step, label.observation, label.target, label.stage)
            try:
                self._check_port(port, step, observation)
                self._check_readiness_binding()
            finally:
                # Returned valid labels remain raw evidence even when a later
                # pairing/state check fails; the poisoned sequence cannot release.
                self._labels.append(label)
                if self._admitted_readiness:
                    self._readiness_receipts += (receipt,)
                if self._admitted_observed_grasp:
                    self._observed_grasp_receipts += (_official_grasp(observation),)
            if self._teacher.steps != step + 1:
                raise ValueError("correction teacher did not advance exactly once")
            self._check_readiness_binding()
        except Exception:
            self._failed = True
            raise

    def latest_readiness_label(
        self, step: int, observation: tuple[float, ...]
    ) -> ReadinessCorrectionLabel:
        """Own the current successful query result without querying or executing.

        The port must still be at the queried pre-control observation. Advancing
        the teacher, collecting a new label or executing this target is outside
        this accessor; callers record their own explicit behavior decision.
        """
        try:
            if self._admitted_readiness is not True or self._failed:
                raise ValueError("latest readiness label requires a successful readiness query")
            self._check_readiness_binding()
            if type(step) is not int or not 0 <= step < 50 or step != len(self._labels) - 1:
                raise ValueError("latest readiness label requires the exact current queried step")
            raw = _vector(observation, "latest readiness observation", 1024)
            label = self._labels[-1]
            if type(label) is not ReadinessCorrectionLabel or label.observation != raw:
                raise ValueError("latest readiness label differs from the queried raw observation")
            port = self._readiness_port
            if port is None or self._teacher.steps != step + 1:
                raise ValueError("latest readiness label lost its teacher or port position")
            self._check_readiness_binding()
            self._check_port(port, step, raw, label_offset=1)
            self._check_readiness_binding()
            owned = ReadinessCorrectionLabel(
                label.step, label.observation, label.target, label.stage
            )
            # Public port getters above may run instrumentation callbacks. End
            # with original native pairing and cached teacher/label checks only.
            if (
                port is not self._readiness_port
                or port is not self._port
                or len(port._records) != step
                or port._observation != raw
            ):
                raise ValueError("latest readiness query pairing changed during observation")
            self._check_readiness_binding()
            return owned
        except Exception:
            self._failed = True
            raise

    def finish(
        self, episode: ContinuousEpisode
    ) -> PickCubeCorrectionBatch | ReadinessCorrectionBatch:
        if self._failed:
            raise ValueError("failed correction sequence releases no usable batch")
        try:
            self._check_readiness_binding()
        except Exception:
            self._failed = True
            raise
        if (
            self._port is None
            or type(self._port._seed) is not int
            or self._port._seed != self._seed
            or self._port._context != self._context
            or self._port._role != "train"
            or episode != self._port.episode()
        ):
            raise ValueError("correction episode differs from the queried port trajectory")
        self._check_readiness_binding()
        # Each constructor re-owns exact family labels and rejects cross-types.
        if self._admitted_observed_grasp:
            batch_type = (
                GoalConsistentObservedGraspCorrectionBatch
                if self._admitted_goal_consistent
                else ObservedGraspCorrectionBatch
            )
            return batch_type(
                self._seed,
                episode,
                cast(tuple[ReadinessCorrectionLabel, ...], tuple(self._labels)),
                self._observed_grasp_receipts,
            )
        if self._readiness_corrections:
            return ReadinessCorrectionBatch(
                self._seed, episode, cast(tuple[ReadinessCorrectionLabel, ...], tuple(self._labels))
            )
        return PickCubeCorrectionBatch(
            self._seed, episode, cast(tuple[PickCubeCorrectionLabel, ...], tuple(self._labels))
        )

    def report(self) -> dict:
        if self._admitted_readiness:
            try:
                self._check_readiness_binding()
            except Exception:
                self._failed = True
            attempts, geometry_returns, teacher_attempts, teacher_returns = self._readiness_paid
            result = dict(
                provenance=GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE
                if self._admitted_goal_consistent
                else OBSERVED_GRASP_PROVENANCE
                if self._admitted_observed_grasp
                else READINESS_PROVENANCE,
                train_seed=self._seed,
                geometry_attempts=attempts,
                geometry_returns=geometry_returns,
                teacher_action_attempts=teacher_attempts,
                teacher_actions=teacher_returns,
                teacher_definition=GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
                if self._admitted_goal_consistent
                else OBSERVED_GRASP_DEFINITION
                if self._admitted_observed_grasp
                else READINESS_DEFINITION,
                teacher_decision_steps=self._readiness_state[0],
                labels=[
                    dict(
                        step=step,
                        observation=observation,
                        target=target,
                        stage=stage,
                        observed_at=2 * step,
                        labels_arrived_at=2 * step + 1,
                    )
                    for step, observation, target, stage in self._readiness_receipts
                ],
                batch=None,
            )
            if self._admitted_observed_grasp:
                result.update(observed_grasp=list(self._observed_grasp_receipts))
            return result
        result = dict(
            provenance=READINESS_PROVENANCE if self._readiness_corrections else PROVENANCE,
            train_seed=self._seed,
            geometry_attempts=self._attempts,
            geometry_returns=self._geometry_returns,
            teacher_action_attempts=self._teacher_attempts,
            teacher_actions=self._teacher_returns,
            **(
                dict(
                    teacher_definition=READINESS_DEFINITION,
                    teacher_decision_steps=self._teacher.steps,
                )
                if self._readiness_corrections
                else dict(teacher_schedule_steps=self._teacher.steps)
            ),
            labels=[
                dict(
                    asdict(label),
                    observed_at=label.observed_at,
                    labels_arrived_at=label.labels_arrived_at,
                )
                for label in self._labels
            ],
            batch=None,
        )
        return result
