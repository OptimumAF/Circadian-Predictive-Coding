"""Student prefix then observed-v2 teacher completion, with truthful paid work.

One BP D12/H2 initializer32 normal values, one manual trained-state restore.
Five fake scenes:8 one-row predictions/normalizations96coordinates/256denseMAC,
8 geometry/teacher returns,7 control attempts6returns,5reset/close. No fit,
SGD/replay/PC/native physics. Driver cases reuse the paid result with stubs.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json
import struct
import sys
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters import numpy_continuous_bc as bp
from src.adapters.continuous_learnability_demo import ContinuousLearnabilityResult
from src.app import readiness_roll_in as behavior
from src.core import pick_cube_action_history as history
from src.core import pick_cube_readiness_roll_in as selector
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.pick_cube_action_history_corrections import action_history_correction_metadata
from src.core.pick_cube_readiness_corrections import (
    GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE,
    GoalConsistentObservedGraspCorrectionBatch,
)
from src.core.pick_cube_readiness_teacher import GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
from src.core.pick_cube_taskspace_history import PickCubeTaskSpaceHistoryRecipe
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_bp_binary_gripper_robot_learning import assert_state
from test_maniskill_joint_positions import StateEnvironment


RECIPE = PickCubeTaskSpaceHistoryRecipe()
STUDENT = (0.3, -0.25, 0.0, -0.75)
TARGET = (-0.2, 0.0, 0.1, 1.0)


def float32(command):
    return tuple(struct.unpack("!f", struct.pack("!f", value))[0] for value in command)


@pytest.mark.parametrize(
    "prefix,step,source",
    [(1, 0, "student"), (1, 1, "teacher"), (49, 48, "student"), (49, 49, "teacher")],
)
def test_should_use_only_the_declared_student_prefix_and_keep_both_original_commands(
    prefix, step, source
):
    plan = selector.ReadinessTeacherCompletionPlan(prefix)
    owned = selector.validate_readiness_roll_in_plan(plan)
    assert (
        type(owned) is selector.ReadinessTeacherCompletionPlan
        and owned == plan
        and owned is not plan
    )
    value = selector.readiness_roll_in_decision(plan, step, STUDENT, TARGET)
    assert value.step == step and value.behavior_source == source
    assert value.student_action == STUDENT and value.teacher_target == TARGET
    assert value.executed_action == float32(STUDENT if source == "student" else TARGET)
    assert value.student_action is not STUDENT and value.teacher_target is not TARGET
    with pytest.raises(FrozenInstanceError):
        setattr(value, "step", 9)


def test_should_distinguish_completion_metadata_from_the_unchanged_old_teacher_prefix():
    assert selector.readiness_roll_in_metadata(selector.ReadinessTeacherCompletionPlan(2)) == {
        "definition": "pick_cube_readiness_student_prefix_then_teacher_v1",
        "student_steps": 2,
    }
    assert selector.readiness_roll_in_metadata(selector.ReadinessRollInPlan(2)) == {
        "definition": selector.DEFINITION,
        "teacher_steps": 2,
    }
    assert selector.COMPLETION_DEFINITION != selector.DEFINITION


@pytest.mark.parametrize("value", [True, 0, 50, 1.0, None])
def test_should_refuse_untyped_or_unbounded_student_prefix_without_coercion(value):
    with pytest.raises(ValueError):
        selector.ReadinessTeacherCompletionPlan(value)


@pytest.mark.parametrize("fault", ["mapping", "subclass", "changed", "extra_old_field"])
def test_should_reown_only_the_exact_completion_plan_and_refuse_coherent_replacements(fault):
    plan: Any = selector.ReadinessTeacherCompletionPlan(1)
    if fault == "mapping":
        plan = asdict(plan)
    elif fault == "subclass":

        class Derived(selector.ReadinessTeacherCompletionPlan):
            pass

        plan = Derived(1)
    elif fault == "changed":
        object.__setattr__(plan, "student_steps", 1.0)
    else:
        object.__setattr__(plan, "teacher_steps", 1)
    with pytest.raises(ValueError):
        selector.validate_readiness_roll_in_plan(plan)


@pytest.mark.parametrize("step", [True, -1, 50, 0.0])
def test_should_refuse_invalid_observed_steps_for_the_new_selection_mode(step):
    with pytest.raises(ValueError):
        selector.readiness_roll_in_decision(
            selector.ReadinessTeacherCompletionPlan(1), step, STUDENT, TARGET
        )


@pytest.mark.parametrize("side", ["student", "teacher"])
def test_should_validate_unselected_commands_as_well_as_the_executed_command(side):
    bad = (0.0, 0.0, float("nan"), -1.0)
    student, teacher = (bad, TARGET) if side == "student" else (STUDENT, bad)
    with pytest.raises(ValueError):
        selector.readiness_roll_in_decision(
            selector.ReadinessTeacherCompletionPlan(1), 0, student, teacher
        )


def test_should_report_teacher_takeover_only_after_a_return_and_keep_detached_control_receipts():
    ledger = behavior.ReadinessRollInLedger(selector.ReadinessTeacherCompletionPlan(1))
    report = ledger.report()
    assert report["teacher_takeover_returned"] is False
    assert "handed_back" not in report and "teacher_steps" not in report
    for step in range(3):
        observation = (float(step), 0.0)
        decision = selector.readiness_roll_in_decision(ledger.plan, step, STUDENT, TARGET)
        ledger.attempt(decision, observation, STUDENT, TARGET, report)
        report = ledger.report()
        assert report["teacher_takeover_returned"] is (step > 1)
        row = SimpleNamespace(step=step, observation=observation, action=decision.executed_action)
        ledger.returned(row, report)
        report = ledger.report()
        assert report["teacher_takeover_returned"] is (step >= 1)
    assert report["student_control_attempts"] == report["student_control_returns"] == 1
    assert report["teacher_control_attempts"] == report["teacher_control_returns"] == 2
    changed = deepcopy(report)
    changed["controls"][0]["executed_action"] = (0.0,) * 4
    with pytest.raises(ValueError):
        ledger.check(changed)
    assert ledger.report() == report
    with pytest.raises(ValueError, match="failed"):
        ledger.check(report)


def test_should_keep_the_original_paid_return_when_a_private_prior_observation_was_replaced():
    ledger = behavior.ReadinessRollInLedger(selector.ReadinessTeacherCompletionPlan(1))
    decision = selector.readiness_roll_in_decision(ledger.plan, 0, STUDENT, TARGET)
    ledger.attempt(decision, (0.0, 1.0), STUDENT, TARGET, ledger.report())
    report = ledger.report()
    record = list(ledger._records[0])
    record[5] = (99.0, 1.0)
    ledger._records = (tuple(record),)
    row = SimpleNamespace(step=0, observation=(0.0, 1.0), action=decision.executed_action)
    with pytest.raises(ValueError):
        ledger.returned(row, report)
    paid = ledger.report()
    assert paid["student_control_attempts"] == paid["student_control_returns"] == 1
    assert paid["controls"][0]["control_returned"] is True
    assert paid["controls"][0]["executed_action"] == decision.executed_action
    with pytest.raises(ValueError, match="failed"):
        ledger.check(paid)


def raw(step):
    values = [0.0] * 42
    values[7:9] = (0.02, 0.02)
    values[18] = float(step >= 2)
    values[36:42] = (
        (0.0, 0.0, -0.08, 0.018, 0.018, 0.0)
        if step == 0
        else (0.0, 0.0, 0.0, 0.018, 0.018, 0.0)
        if step == 1
        else (0.3, 0.0, -0.25, 0.018, 0.018, 0.0)
    )
    return tuple(values)


class CompletionEnvironment(StateEnvironment):
    def __init__(self, *, steps=3, fail_second=False):
        super().__init__()
        self.steps, self.fail_second = steps, fail_second
        self.reset_observation = np.asarray([raw(0)], dtype=np.float64)

    def step(self, command):
        if self.fail_second and len(self.commands) == 1:
            self.step_error = RuntimeError("teacher takeover control failed")
        returned = super().step(command)
        count = len(self.commands)
        return (
            np.asarray([raw(count)], dtype=np.float64),
            np.asarray([float(count == self.steps)], dtype=np.float64),
            *returned[2:],
        )


def geometry(port):
    index = len(port._records)
    tcp = (0.0, 0.0, 0.08) if index == 0 else (0.0, 0.0, 0.0) if index == 1 else (-0.3, 0.0, 0.25)
    return PickCubeGeometry(tcp, (0.0, 0.0, 0.0), (0.018, 0.018, 0.0))


@pytest.fixture(scope="module")
def source():
    spec = ContinuousActionSpec(12, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    owner = bp.NumpyContinuousBCLearner(bp.ContinuousBCPolicy(spec, hidden_dim=2, seed=17))
    initial = owner.snapshot_state()
    parameters = cast(bp.Parameters, tuple(np.zeros_like(value) for value in initial.parameters))
    parameters[3][3] = -1.0
    velocity = cast(bp.Parameters, tuple(np.full_like(value, 0.125) for value in initial.velocity))
    owner.restore_state(
        replace(
            initial,
            parameters=parameters,
            velocity=velocity,
            completed_updates=3,
            training_examples=6,
        )
    )
    return owner


def install(patch, scene, geometry_calls):
    def factory():
        return scene

    def query_geometry(port):
        geometry_calls.append(len(port._records))
        return geometry(port)

    patch.setattr(native, "_make_environment", factory)
    patch.setattr(queries, "_geometry", query_geometry)
    patch.setattr(smoke, "version", lambda _: "fake-student-prefix-completion")
    patch.setattr(bp.NumpyContinuousBCLearner, "train_batch", forbidden)
    patch.setattr(bp.observations, "fit_observation_standardization", forbidden)


def run(source, output, *, steps=3, fail_second=False, patch):
    scene = CompletionEnvironment(steps=steps, fail_second=fail_second)
    queries_observed: list[int] = []
    install(patch, scene, queries_observed)
    owner = source.fork()
    before = owner.snapshot_state()
    predictions = []
    predict = bp.NumpyContinuousBCLearner.predict

    def record(model, features):
        predictions.append(features.copy())
        return predict(model, features)

    patch.setattr(bp.NumpyContinuousBCLearner, "predict", record)
    result = smoke.run_frozen_rollout(
        output,
        learner=owner,
        train_seed=17,
        corrective_queries=True,
        readiness_corrections=True,
        observed_grasp_corrections=True,
        goal_consistent_corrections=True,
        action_history_recipe=RECIPE,
        readiness_roll_in=selector.ReadinessTeacherCompletionPlan(1),
    )
    assert_state(before, owner.snapshot_state())
    return result, scene, queries_observed, predictions


@pytest.fixture(scope="module")
def successful(source, tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        return run(
            source,
            tmp_path_factory.mktemp("student-teacher-completion") / "success.json",
            patch=patch,
        )


def test_should_switch_to_the_observed_teacher_without_claiming_student_only_success(successful):
    result, scene, calls, predictions = successful
    report, receipt = result.report, result.report["readiness_roll_in"]
    assert (
        report["complete"] is True
        and type(result.corrections) is GoalConsistentObservedGraspCorrectionBatch
    )
    assert report["policy"] == "mixed_readiness_teacher_and_frozen_student"
    assert report["behavior_scope"] == "frozen_student_prefix_then_teacher_completion"
    assert report["student_only_success_available"] is False
    assert report["any_success"] is report["final_success"] is True
    assert receipt["definition"] == selector.COMPLETION_DEFINITION and receipt["student_steps"] == 1
    assert receipt["teacher_takeover_returned"] is True and "handed_back" not in receipt
    assert [row["behavior_source"] for row in receipt["controls"]] == [
        "student",
        "teacher",
        "teacher",
    ]
    assert receipt["student_control_attempts"] == receipt["student_control_returns"] == 1
    assert receipt["teacher_control_attempts"] == receipt["teacher_control_returns"] == 2
    assert report["prediction_attempts"] == report["prediction_returns"] == len(predictions) == 3
    assert calls == [0, 1, 2] and len(scene.commands) == 3 and scene.close_calls == 1
    assert report["original_owner_exact"] is report["frozen_state_exact"] is True
    assert report["optimizer_updates"] == 0
    assert (report["source_updates"], report["source_training_rows"]) == (3, 6)
    assert report["action_history_exact"] is True
    assert report["action_history"]["recipe"] == asdict(RECIPE)
    assert report["action_history"]["work"] == dict(
        encoding_attempts=3,
        encoding_returns=3,
        committed_controls=3,
        source_values_validated=126,
        model_values_emitted=36,
        model_numeric_bytes_emitted=288,
    )


def test_should_keep_exact_executed_float32_history_distinct_from_student_proposals_and_teacher_targets(
    successful,
):
    result, scene, _, predictions = successful
    batch = result.corrections
    controls = result.report["readiness_roll_in"]["controls"]
    assert batch.observed_grasp == (False, False, True)
    assert [label.stage for label in batch.labels] == ["descend", "grasp", "lift"]
    assert batch.labels[2].target == pytest.approx((0.18, 0.18, 0.0, -1.0))
    assert result.report["corrections"]["provenance"] == GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE
    assert (
        result.report["corrections"]["teacher_definition"]
        == GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION
    )
    assert predictions[0][0, -5:].tolist() == [0.0] * 5
    for index, (row, control, command) in enumerate(
        zip(batch.episode.transitions, controls, scene.commands, strict=True)
    ):
        assert row.observation == raw(index) and row.next_observation == raw(index + 1)
        assert (
            row.action
            == tuple(control["executed_action"])
            == tuple(float(value) for value in command[0])
        )
        assert row.action == float32(
            control["student_action"] if index == 0 else control["teacher_target"]
        )
        assert tuple(control["teacher_target"]) == batch.labels[index].target
        assert tuple(control["student_action"]) == pytest.approx(
            (0.0, 0.0, 0.0, -0.7615941559557649)
        )
        if index:
            assert predictions[index][0, -5:-1].tolist() == list(
                batch.episode.transitions[index - 1].action
            )
            assert predictions[index][0, -1] == pytest.approx(index / 6)
    assert batch.labels[0].target != batch.episode.transitions[0].action
    assert batch.labels[2].target != batch.episode.transitions[2].action
    assert controls[2]["student_action"] != controls[2]["executed_action"]
    assert [(label.observed_at, label.labels_arrived_at) for label in batch.labels] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]
    json.dumps(result.report, allow_nan=False)


def test_should_end_during_the_student_prefix_without_claiming_teacher_takeover(
    source, tmp_path, monkeypatch
):
    result, scene, calls, predictions = run(
        source, tmp_path / "early.json", steps=1, patch=monkeypatch
    )
    report, receipt = result.report, result.report["readiness_roll_in"]
    assert report["complete"] is True and result.corrections is not None
    assert receipt["teacher_takeover_returned"] is False
    assert receipt["student_control_returns"] == 1 and receipt["teacher_control_attempts"] == 0
    assert report["student_only_success_available"] is False
    assert len(predictions) == len(scene.commands) == 1 and calls == [0]


def test_should_retain_the_failed_teacher_attempt_and_both_paid_predictions_and_labels(
    source, tmp_path, monkeypatch
):
    result, scene, calls, predictions = run(
        source, tmp_path / "failed-control.json", fail_second=True, patch=monkeypatch
    )
    report, receipt = result.report, result.report["readiness_roll_in"]
    assert report["complete"] is False and result.corrections is None
    assert receipt["student_control_returns"] == 1
    assert receipt["teacher_control_attempts"] == 1 and receipt["teacher_control_returns"] == 0
    assert receipt["teacher_takeover_returned"] is False
    assert [row["control_returned"] for row in receipt["controls"]] == [True, False]
    assert report["prediction_returns"] == len(predictions) == 2
    assert len(report["corrections"]["labels"]) == 2 and calls == [0, 1]
    assert len(scene.commands) == 2 and scene.close_calls == 1


def test_should_refuse_pending_receipt_corruption_from_the_last_owner_callback_before_control(
    source, tmp_path, monkeypatch
):
    original = smoke._check_state_projection_frozen
    changed: list[bool] = []

    def corrupt(*args, **options):
        value = original(*args, **options)
        receipt = args[3].get("readiness_roll_in")
        if receipt is not None and receipt["controls"] and not changed:
            assert receipt["controls"][-1]["control_returned"] is False
            receipt["controls"][-1]["executed_action"] = (0.5, 0.0, 0.0, -1.0)
            changed.append(True)
        return value

    monkeypatch.setattr(smoke, "_check_state_projection_frozen", corrupt)
    result, scene, calls, predictions = run(
        source, tmp_path / "changed-prefix.json", patch=monkeypatch
    )
    assert result.report["complete"] is False and result.corrections is None
    assert changed == [True]
    assert len(predictions) == 1 and calls == [0] and scene.commands == []
    assert len(result.report["corrections"]["labels"]) == 1 and scene.close_calls == 1


def test_should_pin_the_original_completion_plan_before_the_initial_timer_callback(
    source, tmp_path, monkeypatch
):
    plan = selector.ReadinessTeacherCompletionPlan(2)
    timer = smoke.time.monotonic
    calls: list[bool] = []
    before = source.snapshot_state()

    def changed_timer():
        if not calls:
            object.__setattr__(plan, "student_steps", 3)
        calls.append(True)
        return timer()

    monkeypatch.setattr(smoke.time, "monotonic", changed_timer)
    monkeypatch.setattr(source, "fork", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "predict", forbidden)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    monkeypatch.setattr(smoke, "version", forbidden)
    result = smoke.run_frozen_rollout(
        tmp_path / "changed-original-plan.json",
        learner=source,
        corrective_queries=True,
        readiness_corrections=True,
        observed_grasp_corrections=True,
        goal_consistent_corrections=True,
        action_history_recipe=RECIPE,
        readiness_roll_in=plan,
    )
    assert calls and plan.student_steps == 3
    assert result.report["complete"] is False and result.corrections is None
    assert result.report["creation_attempts"] == result.report["prediction_attempts"] == 0
    assert_state(before, source.snapshot_state())


def test_should_retain_a_returned_control_when_subsequent_action_history_commit_raises(
    source, tmp_path, monkeypatch
):
    def fail(*args, **options):
        raise RuntimeError("history commit failed after native return")

    monkeypatch.setattr(smoke.ActionHistoryEncoding, "commit", fail)
    result, scene, calls, predictions = run(
        source, tmp_path / "post-return-failure.json", patch=monkeypatch
    )
    receipt = result.report["readiness_roll_in"]
    assert result.report["complete"] is False and result.corrections is None
    assert "history commit failed" in result.report["error"]
    assert len(predictions) == len(scene.commands) == 1 and calls == [0]
    assert receipt["student_control_attempts"] == receipt["student_control_returns"] == 1
    assert receipt["controls"][0]["control_returned"] is True
    assert receipt["teacher_takeover_returned"] is False
    assert result.report["work"]["completed_steps"] == 1 and scene.close_calls == 1


def forbidden(*args, **options):
    pytest.fail("teacher-completion refusal/stub admitted extra learner or native work")


@pytest.mark.parametrize(
    "fault", ["queries", "observed", "goal", "zero_history", "visual", "actor"]
)
def test_should_refuse_unsupported_completion_before_fork_prediction_or_scene_creation(
    source, tmp_path, monkeypatch, fault
):
    options: dict[str, Any] = dict(
        learner=source,
        train_seed=17,
        corrective_queries=True,
        readiness_corrections=True,
        observed_grasp_corrections=True,
        goal_consistent_corrections=True,
        action_history_recipe=RECIPE,
        readiness_roll_in=selector.ReadinessTeacherCompletionPlan(1),
    )
    if fault == "queries":
        options["corrective_queries"] = False
    elif fault == "observed":
        options["observed_grasp_corrections"] = False
    elif fault == "goal":
        options["goal_consistent_corrections"] = False
    elif fault == "zero_history":
        options["action_history_recipe"] = PickCubeTaskSpaceHistoryRecipe(mode="zero")
    elif fault == "visual":
        options["visual_observation"] = True
    else:
        options["actor"] = object()
    monkeypatch.setattr(source, "fork", forbidden)
    monkeypatch.setattr(bp.NumpyContinuousBCLearner, "predict", forbidden)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    monkeypatch.setattr(smoke, "version", forbidden)
    before = source.snapshot_state()
    result = smoke.run_frozen_rollout(tmp_path / f"refused-{fault}.json", **options)
    assert result.report["complete"] is False and result.corrections is None
    assert result.report["creation_attempts"] == result.report["prediction_attempts"] == 0
    assert_state(before, source.snapshot_state())


@pytest.mark.parametrize("selected", [False, True])
def test_should_forward_new_cli_plan_only_when_selected_and_leave_default_kwargs_absent(
    monkeypatch, tmp_path, selected
):
    calls = []

    def execute(args, **options):
        calls.append((args, options))
        return smoke.FrozenRolloutResult({"complete": False}, None)

    monkeypatch.setattr(smoke, "_execute_frozen_rollout", execute)
    flags = ["--readiness-teacher-completion-student-steps", "1"] if selected else []
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(tmp_path / "cli.json"), *flags])
    assert smoke.main() == 1 and len(calls) == 1
    if selected:
        assert calls[0][1]["readiness_roll_in"] == selector.ReadinessTeacherCompletionPlan(1)
    else:
        assert "readiness_roll_in" not in calls[0][1]


def test_should_refuse_simultaneous_opposite_prefix_cli_flags_before_execution(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(smoke, "_execute_frozen_rollout", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke",
            "--output",
            str(tmp_path / "mutual.json"),
            "--readiness-roll-in-steps",
            "1",
            "--readiness-teacher-completion-student-steps",
            "1",
        ],
    )
    with pytest.raises(SystemExit) as error:
        smoke.main()
    assert error.value.code == 2


@pytest.mark.parametrize("fault", ["none", "changed_prefix", "mixed_return"])
def test_should_pass_completion_only_to_collection_then_guard_a_pure_student_return(
    source,
    successful,
    tmp_path,
    monkeypatch,
    fault,
):
    # Boundary composition only: reuse the already returned mixed raw episode;
    # model continuation and serving reports are stubs, not new SGD or physics.
    collected = deepcopy(successful[0])
    raw_batch = collected.corrections
    assert type(raw_batch) is GoalConsistentObservedGraspCorrectionBatch
    owner = source.fork()
    initial = owner.snapshot_state()
    candidate = owner.fork()
    candidate.restore_state(
        replace(candidate.snapshot_state(), completed_updates=11, training_examples=30)
    )
    candidate_before = candidate.snapshot_state()
    update = ContinuousLearnabilityResult(
        dict(
            complete=True,
            provenance=GOAL_CONSISTENT_OBSERVED_GRASP_PROVENANCE,
            teacher_definition=GOAL_CONSISTENT_OBSERVED_GRASP_DEFINITION,
            observed_grasp=list(raw_batch.observed_grasp),
            input_projection=action_history_correction_metadata(raw_batch, RECIPE),
        ),
        candidate,
    )
    pure = deepcopy(collected.report)
    for name in (
        "readiness_roll_in",
        "behavior_scope",
        "student_only_success_available",
        "corrections",
        "observed_grasp_corrections",
        "goal_consistent_corrections",
    ):
        pure.pop(name, None)
    pure.update(policy="frozen_continuous_bc", source_updates=11, source_training_rows=30)
    state = history.ActionHistoryState()
    for row, control in zip(
        pure["episode"]["transitions"],
        collected.report["readiness_roll_in"]["controls"],
        strict=True,
    ):
        row["action"] = float32(control["student_action"])
        state = history.advance_action_history(state, row["action"])
    pure["action_history"]["state"] = asdict(state)
    if fault == "changed_prefix":
        collected.report["readiness_roll_in"]["student_steps"] = 2
    elif fault == "mixed_return":
        pure["policy"] = "mixed_readiness_teacher_and_frozen_student"
    calls, updates = [], []

    def rollout(*args, **options):
        calls.append(options)
        return collected if len(calls) == 1 else smoke.FrozenRolloutResult(pure, None)

    def correction(*args, **options):
        updates.append((args, options))
        assert args == (owner, raw_batch)
        assert options == {"now": 5, "action_history_recipe": RECIPE}
        return update

    monkeypatch.setattr(smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(smoke, "run_continuous_correction_update", correction)
    report = smoke.run_corrective_adaptation(
        owner,
        output_directory=tmp_path / fault,
        visual_observation=False,
        readiness_corrections=True,
        observed_grasp_corrections=True,
        goal_consistent_corrections=True,
        acquisition_only=True,
        action_history_recipe=RECIPE,
        readiness_roll_in=selector.ReadinessTeacherCompletionPlan(1),
    )
    assert report["complete"] is (fault == "none")
    assert calls[0]["readiness_roll_in"] == selector.ReadinessTeacherCompletionPlan(1)
    assert calls[0]["goal_consistent_corrections"] is calls[0]["observed_grasp_corrections"] is True
    if fault == "changed_prefix":
        assert len(calls) == 1 and updates == []
    else:
        assert len(updates) == 1 and [call["train_seed"] for call in calls] == [17, 17]
        assert "readiness_roll_in" not in calls[1] and "corrective_queries" not in calls[1]
        assert report["update"] == update.report
    if fault == "none":
        assert (
            report["protocol_id"]
            == "corrective_robot_student_prefix_teacher_completion_train_17_acquisition_v1"
        )
        assert (
            report["collection_behavior_scope"] == "frozen_student_prefix_then_teacher_completion"
        )
        assert report["return_behavior_scope"] == "frozen_student_only"
    assert_state(initial, owner.snapshot_state())
    assert_state(candidate_before, candidate.snapshot_state())
