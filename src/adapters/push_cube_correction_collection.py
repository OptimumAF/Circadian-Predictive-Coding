"""Frozen student Push collection with provisional labels on actual visited states.

Own one bounded TRAIN interaction; retain paid failures and release only complete,
causally aligned correction packets. No training, expert-action execution, final
access, promotion, persistence or generic orchestration belongs here.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from time import monotonic
from typing import Any

import numpy as np

from src.adapters.continuous_learnability_modes import same_continuous_state
from src.adapters.numpy_continuous_bc import NumpyContinuousBCLearner, _snapshot
from src.core.continuous_demonstration import ContinuousTransition
from src.core.cube_taskspace_state import (
    copy_cube_taskspace_state,
    cube_taskspace_observation,
    cube_taskspace_spec,
)
from src.core.pick_cube_action_history import ActionHistoryState, advance_action_history
from src.core.push_cube_corrections import (
    PROVENANCE,
    PushCubeCorrectionBatch,
    PushCubeCorrectionLabel,
)
from src.core.push_cube_teacher import PushCubeDecision, PushCubeTeacher
from src.infra.maniskill_push_cube import ManiSkillPushCubeEpisode, PushCubeWork


@dataclass(frozen=True)
class PushCubeCorrectionResult:
    report: dict[str, Any]
    corrections: PushCubeCorrectionBatch | None


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def collect_push_cube_corrections(
    learner: NumpyContinuousBCLearner,
    *,
    seed: int = 17,
    role: str = "train",
    check_boundary: Callable[[], None] | None = None,
) -> PushCubeCorrectionResult:
    """Query geometry labels but execute the frozen student, at most50 times."""
    if (
        type(seed) is not int
        or seed not in (17, 19, 20)
        or type(role) is not str
        or role != "train"
        or type(learner) is not NumpyContinuousBCLearner
        or learner._policy.spec != cube_taskspace_spec()
        or learner._policy.training_role != "train"
        or learner._policy.hidden_dim > 64
        or (check_boundary is not None and not callable(check_boundary))
    ):
        raise ValueError("Push corrections require native same-spec TRAIN BP and seed17/19/20")
    native, policy = learner._state, learner._policy
    initial = _snapshot(native, policy)
    teacher = PushCubeTeacher()
    teacher_pin = (0, "align")
    history = ActionHistoryState()
    labels: list[PushCubeCorrectionLabel] = []
    transitions: list[ContinuousTransition] = []
    decisions: list[PushCubeDecision] = []
    report: dict[str, Any] = dict(
        definition=PROVENANCE,
        complete=False,
        error=None,
        seed=seed,
        role=role,
        source_updates=initial.completed_updates,
        source_training_rows=initial.training_examples,
        episode=None,
        transitions=[],
        states=[],
        labels=[],
        label_receipts=[],
        metadata=None,
        work=None,
        prediction_outputs=[],
        source_unchanged=False,
        training_release=False,
    )
    counts = dict(
        teacher_query_attempts=0,
        teacher_query_returns=0,
        prediction_attempts=0,
        prediction_returns=0,
        control_attempts=0,
        control_returns=0,
    )
    report.update(counts)
    seal = _json(report)
    records_seal = _json([[], [], []])
    port = None
    corrections = None
    correction_seal = None
    paid_return = None
    started = monotonic()

    def require_source():
        if (
            learner._state is not native
            or learner._policy is not policy
            or not same_continuous_state(initial, _snapshot(learner._state, learner._policy))
        ):
            raise RuntimeError("Push correction frozen student changed")

    def guard(*, teacher_pending=False):
        if check_boundary is not None:
            check_boundary()
        require_source()
        if corrections is not None and _json(asdict(corrections)) != correction_seal:
            raise RuntimeError("Push correction released batch changed")
        if not teacher_pending and (teacher.steps, teacher.phase) != teacher_pin:
            raise RuntimeError("Push correction preceding teacher state changed")
        if (
            _json(report) != seal
            or _json(
                [
                    [asdict(value) for value in labels],
                    [asdict(value) for value in transitions],
                    [asdict(value) for value in decisions],
                ]
            )
            != records_seal
        ):
            raise RuntimeError("Push correction preceding evidence changed")

    def save():
        nonlocal seal, records_seal
        report.update(counts)
        seal = _json(report)
        records_seal = _json(
            [
                [asdict(value) for value in labels],
                [asdict(value) for value in transitions],
                [asdict(value) for value in decisions],
            ]
        )

    def require_input(state):
        assert port is not None
        if (
            port.seed != seed
            or port.role != role
            or port.context != f"push-cube-train-{seed}"
            or port.spec != cube_taskspace_spec()
            or port.state != state
            or port.observation != cube_taskspace_observation(state, history)
        ):
            raise RuntimeError("Push correction physical state or actual history changed")

    try:
        guard()
        port = ManiSkillPushCubeEpisode(f"push-cube-student-{seed}", seed=seed, role=role)
        guard()
        metadata = port.metadata
        guard()
        if (
            metadata["task_id"] != "PushCube-v1"
            or metadata["seed"] != seed
            or metadata["role"] != role
            or metadata["context"] != f"push-cube-train-{seed}"
            or metadata["horizon"] != 50
            or metadata["control_physics_steps_per_control"] != 5
        ):
            raise ValueError("Push correction native task/physics metadata differs")
        report["metadata"] = deepcopy(metadata)
        save()
        for step in range(50):
            guard()
            state = copy_cube_taskspace_state(port.state)
            observation = cube_taskspace_observation(state, history)
            require_input(state)
            guard()
            if teacher.steps != counts["teacher_query_returns"] or (
                decisions and teacher.phase != decisions[-1].stage
            ):
                raise RuntimeError("Push correction preceding teacher state changed")
            state_seal = asdict(state)
            report["states"].append(asdict(state))
            counts["teacher_query_attempts"] += 1
            save()
            paid_return = None
            decision = teacher.decide(state, history=history)
            counts["teacher_query_returns"] += 1
            paid_return = dict(
                stage="teacher",
                value=asdict(decision)
                if type(decision) is PushCubeDecision
                else repr(decision)[:1024],
            )
            guard(teacher_pending=True)
            if (
                type(decision) is not PushCubeDecision
                or asdict(state) != state_seal
                or teacher.steps != counts["teacher_query_returns"]
                or teacher.phase != decision.stage
            ):
                raise RuntimeError("Push correction teacher return or physical state differs")
            require_input(state)
            guard(teacher_pending=True)
            if (
                asdict(state) != state_seal
                or asdict(decision) != paid_return["value"]
                or teacher.steps != counts["teacher_query_returns"]
                or teacher.phase != decision.stage
            ):
                raise RuntimeError("Push correction late teacher return or input changed")
            teacher_pin = (teacher.steps, teacher.phase)
            label = PushCubeCorrectionLabel(
                step, state, observation, decision.action, decision.stage
            )
            labels.append(label)
            decisions.append(decision)
            report["labels"].append(asdict(label))
            report["label_receipts"].append(
                dict(
                    step=step,
                    observed_at=label.observed_at,
                    labels_arrived_at=label.labels_arrived_at,
                )
            )
            save()
            paid_return = None
            guard()
            counts["prediction_attempts"] += 1
            save()
            prediction = learner.predict(np.asarray([observation], dtype=np.float64))
            counts["prediction_returns"] += 1
            prediction_payload = (
                dict(
                    shape=list(prediction.shape),
                    dtype=str(prediction.dtype),
                    values=[repr(value) for value in prediction.flat[:16]],
                )
                if type(prediction) is np.ndarray
                else dict(type=type(prediction).__name__)
            )
            paid_return = dict(stage="prediction", value=prediction_payload)
            guard()
            report["prediction_outputs"].append(prediction_payload)
            save()
            if (
                type(prediction) is not np.ndarray
                or prediction.shape != (1, 4)
                or not np.isfinite(prediction).all()
                or np.any(np.abs(prediction) > 1.0)
            ):
                raise ValueError("Push correction student must return one finite bounded action")
            require_input(state)
            guard()
            # Why: preserve actual float32 controller commands, never teacher targets.
            sent = tuple(float(value) for value in np.asarray(prediction[0], dtype=np.float32))
            counts["control_attempts"] += 1
            save()
            paid_return = None
            row = port.step(sent)
            counts["control_returns"] += 1
            paid_return = dict(
                stage="control",
                value=asdict(row) if type(row) is ContinuousTransition else repr(row)[:1024],
            )
            guard()
            if type(row) is not ContinuousTransition:
                raise ValueError("Push correction control returned no native transition")
            report["transitions"].append(asdict(row))
            save()
            if (
                row.step != step
                or row.action != sent
                or row.observation != observation
                or (transitions and transitions[-1].next_observation != row.observation)
            ):
                raise ValueError("Push correction returned student trajectory differs")
            history = advance_action_history(history, row.action)
            current = copy_cube_taskspace_state(port.state)
            require_input(current)
            guard()
            if row.next_observation != cube_taskspace_observation(current, history):
                raise ValueError("Push correction next input differs from executed student history")
            native_work = port.work
            guard()
            transitions.append(row)
            report["work"] = asdict(native_work)
            save()
            paid_return = None
            if row.terminated or row.truncated:
                break
        guard()
        episode = port.episode()
        guard()
        if episode.transitions != tuple(transitions):
            raise ValueError("Push correction exported native prefix differs")
        corrections = PushCubeCorrectionBatch(seed, episode, tuple(labels))
        correction_seal = _json(asdict(corrections))
        report["episode"] = asdict(episode)
        save()
    except Exception as error:
        # Restore the last intended report; retain corruption and all local paid counts.
        try:
            changed = _json(report) != seal
        except (TypeError, ValueError):
            changed = True
        report = json.loads(seal)
        report["error"] = f"{type(error).__name__}: {error}"
        if changed:
            report["rejected_evidence"] = "preceding report changed"
        attached = getattr(error, "push_cube_work", None)
        if attached is not None:
            report["work"] = asdict(attached)
        if paid_return is not None:
            try:
                report["refused_return"] = json.loads(_json(paid_return))
            except (TypeError, ValueError):
                report["refused_return"] = dict(representation=repr(paid_return)[:4096])
        for name in ("push_cube_attempted_actions", "push_cube_returned_actions"):
            if hasattr(error, name):
                report[name] = getattr(error, name)
        corrections = None
    finally:
        if port is not None:
            if corrections is not None:
                try:
                    guard()
                except Exception as error:
                    report["evidence_error"] = f"{type(error).__name__}: {error}"
                    corrections = None
            try:
                port.close()
                if corrections is not None:
                    guard()
            except Exception as error:
                report["close_error"] = f"{type(error).__name__}: {error}"
                corrections = None
            try:
                work = port.work
                if corrections is not None:
                    guard()
                report["work"] = asdict(work)
                n = counts["control_returns"]
                if (
                    type(work) is not PushCubeWork
                    or work.construction_returns != 1
                    or work.reset_returns != 1
                    or work.step_attempts != n
                    or work.completed_steps != n
                    or work.completed_control_physics_steps != 5 * n
                    or work.possible_failed_physics_steps != 0
                    or work.close_attempts != 1
                    or work.close_returns != 1
                    or work.state_encoding_attempts != n + 1
                    or work.state_encoding_returns != n + 1
                    or work.grasp_query_attempts != n + 1
                    or work.grasp_query_returns != n + 1
                    or work.source_contact_force_query_attempts_upper_bound != 2 * (n + 1)
                    or work.source_contact_force_query_returns != 2 * (n + 1)
                ):
                    report["work_error"] = "Push correction complete paid prefix differs"
                    corrections = None
            except Exception as error:
                report["work_error"] = f"{type(error).__name__}: {error}"
                corrections = None
        try:
            require_source()
            report["source_unchanged"] = True
        except Exception as error:
            report["state_error"] = f"{type(error).__name__}: {error}"
            corrections = None
        report.update(counts)
        report["unreturned_query_possible"] = (
            counts["teacher_query_attempts"] != counts["teacher_query_returns"]
        )
        report["unreturned_prediction_possible"] = (
            counts["prediction_attempts"] != counts["prediction_returns"]
        )
        work_payload = report.get("work")
        report["unreturned_native_work_unknown"] = (
            work_payload is None
            or counts["control_attempts"] != counts["control_returns"]
            or any(
                work_payload[a] != work_payload[b]
                for a, b in (
                    ("construction_attempts", "construction_returns"),
                    ("reset_attempts", "reset_returns"),
                    ("step_attempts", "completed_steps"),
                    ("state_encoding_attempts", "state_encoding_returns"),
                    ("grasp_query_attempts", "grasp_query_returns"),
                    ("close_attempts", "close_returns"),
                )
            )
        )
        if any(
            report[key]
            for key in (
                "unreturned_query_possible",
                "unreturned_prediction_possible",
                "unreturned_native_work_unknown",
            )
        ):
            corrections = None
        if corrections is not None:
            save()
        elapsed = monotonic() - started
        try:
            if corrections is not None:
                guard()  # Even the opaque final clock cannot change prior evidence.
            else:
                require_source()
            if corrections is not None:
                PushCubeCorrectionBatch(seed, corrections.episode, corrections.labels)
        except Exception as error:
            if corrections is not None:
                report = json.loads(seal)
                report["rejected_evidence"] = "final callback changed preceding evidence"
            report["state_error"] = f"{type(error).__name__}: {error}"
            report["source_unchanged"] = False
            corrections = None
        report["elapsed_seconds"] = elapsed
        report["complete"] = corrections is not None
        report["training_release"] = corrections is not None
    return PushCubeCorrectionResult(report, corrections)
