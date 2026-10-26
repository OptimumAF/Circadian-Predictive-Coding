"""Fixed teacher-prefix behavior, with every student prediction still paid.

One D42/H2 BP initializer draws92 normal values, then deterministic state and
frozen statistics are restored. Eight one-row predictions normalize336 values
and use736 denseMACs. Seven fake scenes reset/close, attempt six controls (five
returns), and make ten geometry queries/teacher decisions. Two are query-only
scenes and two refuse corruption before control. No fit/SGD/gradient/physics.
The full module uses28 RNG-free forks and six explicit state restores,
including five cached driver-wiring cases with no additional predictions.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json
import sys
from typing import Any, cast

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from src.adapters.continuous_learnability_demo import ContinuousLearnabilityResult
from src.adapters import numpy_continuous_bc as bp
from src.core.continuous_demonstration import ContinuousActionSpec
from src.core.observation_standardization import ObservationStandardization
from src.core import pick_cube_readiness_roll_in as roll_in
from src.core.pick_cube_readiness_corrections import PROVENANCE, ReadinessCorrectionLabel
from src.core.pick_cube_readiness_teacher import DEFINITION as TEACHER_DEFINITION
from src.core.pick_cube_state_quaternions import state_quaternion_projection
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as native
from src.infra import maniskill_corrections as queries
from test_continuous_robot_learning import assert_state
from test_state_quaternion_robot_learning import QuaternionEnvironment


def test_should_preserve_target_and_record_exact_native_float32_command():
    decision = roll_in.readiness_roll_in_decision(
        roll_in.ReadinessRollInPlan(1), 0, (0.3, 0.0, 0.0, -1.0), (0.1, -0.1, 0.0, 1.0)
    )
    assert decision.teacher_target == (0.1, -0.1, 0.0, 1.0)
    assert decision.student_action == (0.3, 0.0, 0.0, -1.0)
    assert decision.executed_action == (0.10000000149011612, -0.10000000149011612, 0.0, 1.0)


@pytest.mark.parametrize(
    "count,step,expected",
    [(1, 0, "teacher"), (1, 1, "student"), (49, 48, "teacher"), (49, 49, "student")],
)
def test_should_choose_only_fixed_prefix_and_record_both_owned_commands(count, step, expected):
    plan = roll_in.ReadinessRollInPlan(count)
    student, teacher = (0, 0.25, -0.5, -1), (0.5, 0, 1, 1)
    owned = roll_in.validate_readiness_roll_in_plan(plan)
    assert owned == plan and owned is not plan
    decision = roll_in.readiness_roll_in_decision(plan, step, student, teacher)
    assert decision.step == step and decision.behavior_source == expected
    assert decision.student_action == tuple(float(x) for x in student)
    assert decision.teacher_target == tuple(float(x) for x in teacher)
    assert decision.executed_action == (
        decision.teacher_target if expected == "teacher" else decision.student_action
    )
    assert decision.student_action is not student and decision.teacher_target is not teacher
    with pytest.raises(FrozenInstanceError):
        setattr(decision, "step", 0)


@pytest.mark.parametrize("count", [True, False, 0, -1, 50, 1.0, None, "1"])
def test_should_refuse_invalid_prefix_length_without_numeric_coercion(count):
    with pytest.raises(ValueError):
        roll_in.ReadinessRollInPlan(count)


@pytest.mark.parametrize("fault", ["mapping", "subclass", "replaced_count"])
def test_should_refuse_changed_or_foreign_plans(fault):
    value: Any = roll_in.ReadinessRollInPlan(1)
    if fault == "mapping":
        value = asdict(value)
    elif fault == "subclass":

        class Derived(roll_in.ReadinessRollInPlan):
            pass

        value = Derived(1)
    else:
        object.__setattr__(value, "teacher_steps", 1.0)
    with pytest.raises(ValueError):
        roll_in.validate_readiness_roll_in_plan(value)


@pytest.mark.parametrize("step", [True, -1, 50, 0.0, None])
def test_should_refuse_unobserved_or_untyped_step(step):
    with pytest.raises(ValueError):
        roll_in.readiness_roll_in_decision(
            roll_in.ReadinessRollInPlan(1), step, (0.0,) * 4, (1.0,) * 4
        )


@pytest.mark.parametrize("side", ["student", "teacher"])
@pytest.mark.parametrize(
    "command",
    [
        [0.0, 0.0, 0.0, 1.0],
        (),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, True),
        (0.0, 0.0, float("nan"), 1.0),
        (0.0, 0.0, float("inf"), 1.0),
        (0.0, 0.0, -1.01, 1.0),
    ],
)
def test_should_refuse_nonfinite_unbounded_or_non_tuple_commands(side, command):
    student, teacher = (command, (0.0,) * 4) if side == "student" else ((0.0,) * 4, command)
    with pytest.raises(ValueError):
        roll_in.readiness_roll_in_decision(roll_in.ReadinessRollInPlan(1), 0, student, teacher)


class Environment(QuaternionEnvironment):
    def __init__(self, *, steps=3, fail_second=False):
        super().__init__()
        self.steps = steps
        self.fail_second = fail_second

    def step(self, command):
        if self.fail_second and len(self.commands) == 1:
            self.step_error = RuntimeError("second paid control failed")
        return super().step(command)


def geometry(port):
    return PickCubeGeometry(
        (0.0, 0.0, 0.08) if not port._records else (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.1, 0.0, 0.2)
    )


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environments, geometry_calls = [], []
    settings = dict(steps=3, fail_second=False)

    def factory():
        value = Environment(**settings)
        environments.append(value)
        return value

    def observed_geometry(port):
        geometry_calls.append(len(port._records))
        return geometry(port)

    monkeypatch.setattr(native, "_make_environment", factory)
    monkeypatch.setattr(queries, "_geometry", observed_geometry)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-readiness-roll-in")
    return environments, geometry_calls, settings, tmp_path


def open_port():
    return native.ManiSkillPickCubeEpisode("pick-cube-smoke-17", "pick-cube", role="train", seed=17)


def test_should_read_latest_owned_raw_label_without_new_teacher_or_geometry_work(boundary):
    _, geometry_calls, _, _ = boundary
    port = open_port()
    query = queries.PickCubeCorrectionQuery(17, readiness_corrections=True)
    try:
        raw = port.observation
        query.query(port, 0, raw)
        before = deepcopy(query.report())
        first = query.latest_readiness_label(0, raw)
        second = query.latest_readiness_label(0, raw)
        assert type(first) is ReadinessCorrectionLabel and first == second and first is not second
        assert first.observation[22] == -0.5
        assert first.target == tuple(before["labels"][0]["target"])
        assert first.target == pytest.approx((0.0, 0.0, -0.8, 1.0))
        object.__setattr__(first, "target", (0.0, 0.0, 0.0, -1.0))
        assert query.latest_readiness_label(0, raw).target == tuple(before["labels"][0]["target"])
        assert query.report() == before and geometry_calls == [0]
        with pytest.raises(ValueError):
            query.latest_readiness_label(1, raw)
        with pytest.raises(ValueError):
            query.latest_readiness_label(0, raw)
    finally:
        port.close()


def test_should_poison_latest_label_after_prior_receipt_integrity_failure_even_if_repaired(
    boundary,
):
    _, geometry_calls, _, _ = boundary
    port = open_port()
    query = queries.PickCubeCorrectionQuery(17, readiness_corrections=True)
    try:
        raw = port.observation
        query.query(port, 0, raw)
        original = query._labels[0].target
        object.__setattr__(query._labels[0], "target", (0.0, 0.0, -0.5, 1.0))
        with pytest.raises(ValueError):
            query.latest_readiness_label(0, raw)
        object.__setattr__(query._labels[0], "target", original)
        with pytest.raises(ValueError):
            query.latest_readiness_label(0, raw)
        assert query.report()["teacher_actions"] == 1 and geometry_calls == [0]
    finally:
        port.close()


@pytest.fixture(scope="module")
def source():
    spec = ContinuousActionSpec(42, native.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)
    model = bp.NumpyContinuousBCLearner(
        bp.ContinuousBCPolicy(
            spec,
            hidden_dim=2,
            seed=17,
            observation_standardization=ObservationStandardization((0.0,) * 42, (1.0,) * 42, 2),
        )
    )
    initial = model.snapshot_state()
    parameters = cast(bp.Parameters, tuple(np.zeros_like(value) for value in initial.parameters))
    parameters[3][3] = -1.0
    model.restore_state(
        replace(
            initial,
            parameters=parameters,
            velocity=cast(
                bp.Parameters, tuple(np.full_like(value, 0.125) for value in initial.velocity)
            ),
            completed_updates=3,
            training_examples=6,
        )
    )
    return model


def run(source, boundary, *, teacher_steps=1):
    environments, calls, _, folder = boundary
    owner = source.fork()
    before = owner.snapshot_state()
    result = smoke.run_frozen_rollout(
        folder / "roll-in.json",
        learner=owner,
        corrective_queries=True,
        readiness_corrections=True,
        observation_projection=state_quaternion_projection(),
        readiness_roll_in=roll_in.ReadinessRollInPlan(teacher_steps),
    )
    assert_state(before, owner.snapshot_state())
    return result, environments[0], calls


@pytest.fixture(scope="module")
def successful(source, tmp_path_factory):
    environments, calls = [], []
    settings = dict(steps=3, fail_second=False)
    folder = tmp_path_factory.mktemp("readiness-roll-in")

    def factory():
        value = Environment(**settings)
        environments.append(value)
        return value

    def observed_geometry(port):
        calls.append(len(port._records))
        return geometry(port)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native, "_make_environment", factory)
        patch.setattr(queries, "_geometry", observed_geometry)
        patch.setattr(smoke, "version", lambda _: "fake-readiness-roll-in")
        return run(source, (environments, calls, settings, folder))


def test_should_pay_student_predictions_on_teacher_prefix_and_hand_back_with_truthful_raw_actions(
    successful,
):
    result, environment, calls = successful
    report = result.report
    assert report["complete"] is True and result.corrections is not None
    assert report["behavior_scope"] == "teacher_prefix_then_frozen_student"
    assert report["student_only_success_available"] is False
    record = report["readiness_roll_in"]
    assert record["definition"] == roll_in.DEFINITION and record["teacher_steps"] == 1
    assert record["teacher_control_attempts"] == record["teacher_control_returns"] == 1
    assert record["student_control_attempts"] == record["student_control_returns"] == 2
    assert record["handed_back"] is True
    assert [row["behavior_source"] for row in record["controls"]] == [
        "teacher",
        "student",
        "student",
    ]
    assert all(row["control_returned"] is True for row in record["controls"])
    assert report["prediction_attempts"] == report["prediction_returns"] == 3
    assert calls == [0, 1, 2] and len(environment.commands) == 3
    for row, transition in zip(
        record["controls"], result.corrections.episode.transitions, strict=True
    ):
        assert row["student_action"] == pytest.approx((0.0, 0.0, 0.0, -0.7615941559557649))
        assert transition.action == tuple(row["executed_action"])
    assert result.corrections.labels[1].target != result.corrections.episode.transitions[1].action
    assert record["controls"][0]["student_action"] != record["controls"][0]["executed_action"]
    assert result.corrections.episode.transitions[0].observation[22] == -0.5
    assert (
        report["projection_exact"]
        is report["original_owner_exact"]
        is report["frozen_state_exact"]
        is True
    )
    json.dumps(report, allow_nan=False)


def test_should_report_teacher_only_termination_without_claiming_student_handback(source, boundary):
    boundary[2]["steps"] = 1
    result, environment, _ = run(source, boundary, teacher_steps=2)
    record = result.report["readiness_roll_in"]
    assert result.report["complete"] is True
    assert record["handed_back"] is False and record["student_control_attempts"] == 0
    assert record["teacher_control_returns"] == 1 and len(environment.commands) == 1
    assert result.report["prediction_returns"] == 1
    assert result.report["student_only_success_available"] is False


def test_should_preserve_both_predictions_queries_and_failed_student_control_attempt(
    source, boundary
):
    boundary[2]["fail_second"] = True
    result, environment, calls = run(source, boundary)
    report, record = result.report, result.report["readiness_roll_in"]
    assert report["complete"] is False and result.corrections is None
    assert record["teacher_control_returns"] == 1
    assert record["student_control_attempts"] == 1 and record["student_control_returns"] == 0
    assert [row["control_returned"] for row in record["controls"]] == [True, False]
    assert report["prediction_returns"] == 2 and len(report["corrections"]["labels"]) == 2
    assert len(environment.commands) == 2 and calls == [0, 1] and environment.close_calls == 1


def forbidden(*args, **kwargs):
    pytest.fail("invalid roll-in request admitted more model/native work")


@pytest.mark.parametrize(
    "fault", ["plan_type", "no_projection", "no_queries", "fixed_teacher", "visual", "actor"]
)
def test_should_refuse_unsupported_roll_in_mode_before_fork_or_sdk(
    source, monkeypatch, tmp_path, fault
):
    owner = source.fork()
    monkeypatch.setattr(owner, "fork", forbidden)
    monkeypatch.setattr(native, "_make_environment", forbidden)
    monkeypatch.setattr(smoke, "version", forbidden)
    options: dict[str, Any] = dict(
        learner=owner,
        corrective_queries=True,
        readiness_corrections=True,
        observation_projection=state_quaternion_projection(),
        readiness_roll_in=roll_in.ReadinessRollInPlan(1),
    )
    changes: dict[str, dict[str, Any]] = {
        "plan_type": dict(readiness_roll_in=True),
        "no_projection": dict(observation_projection=None),
        "no_queries": dict(corrective_queries=False),
        "fixed_teacher": dict(readiness_corrections=False),
        "visual": dict(visual_observation=True),
        "actor": dict(actor=object()),
    }
    options.update(changes[fault])
    result = smoke.run_frozen_rollout(tmp_path / "refused.json", **options)
    assert result.report["complete"] is False and result.report["creation_attempts"] == 0


@pytest.mark.parametrize("selected", [None, 1])
def test_should_forward_cli_plan_only_when_explicit_without_new_default_kwargs(
    monkeypatch, tmp_path, selected
):
    calls = []

    def execute(args, **options):
        calls.append((args, options))
        return smoke.FrozenRolloutResult(dict(complete=False), None)

    monkeypatch.setattr(smoke, "_execute_frozen_rollout", execute)
    flags = ["--readiness-roll-in-steps", "1"] if selected else []
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(tmp_path / "cli.json"), *flags])
    assert smoke.main() == 1
    if selected:
        assert calls[0][0].readiness_roll_in_steps == 1
    else:
        assert "readiness_roll_in" not in calls[0][1]


@pytest.mark.parametrize("fault", ["owned_plan", "public_receipt"])
def test_should_refuse_changed_plan_or_receipt_after_query_before_any_control(
    source, boundary, monkeypatch, fault
):
    decision = smoke.readiness_roll_in_decision
    frozen = smoke._check_frozen
    armed = []

    def choose(plan, *args):
        returned = decision(plan, *args)
        if fault == "owned_plan":
            object.__setattr__(plan, "teacher_steps", 2)
        armed.append(True)
        return returned

    def check(*args, **options):
        returned = frozen(*args, **options)
        if armed and fault == "public_receipt":
            args[3]["readiness_roll_in"]["student_control_returns"] = 1
        return returned

    monkeypatch.setattr(smoke, "readiness_roll_in_decision", choose)
    monkeypatch.setattr(smoke, "_check_frozen", check)
    result, environment, calls = run(source, boundary)
    assert result.report["complete"] is False and result.corrections is None
    assert result.report["prediction_returns"] == 1 and calls == [0]
    assert len(result.report["corrections"]["labels"]) == 1
    assert environment.commands == [] and environment.close_calls == 1


@pytest.mark.parametrize(
    "fault", [None, "ignored_plan", "wrong_prefix", "mixed_return", "mixed_metadata"]
)
def test_should_forward_roll_in_only_to_collection_then_return_a_pure_student(
    source, successful, monkeypatch, tmp_path, fault
):
    # Wiring test only: candidate advancement and update report are stubs, no SGD.
    collected = deepcopy(successful[0])
    raw_batch = collected.corrections
    assert collected.report["complete"] is True and raw_batch is not None
    owner = source.fork()
    candidate = owner.fork()
    candidate.restore_state(
        replace(candidate.snapshot_state(), completed_updates=11, training_examples=30)
    )
    projection = state_quaternion_projection()
    update = ContinuousLearnabilityResult(
        dict(
            complete=True,
            provenance=PROVENANCE,
            teacher_definition=TEACHER_DEFINITION,
            input_projection=smoke._canonical_correction_metadata(raw_batch, projection),
        ),
        candidate,
    )
    pure = deepcopy(collected.report)
    for key in (
        "readiness_roll_in",
        "behavior_scope",
        "student_only_success_available",
        "corrections",
    ):
        pure.pop(key, None)
    pure.update(policy="frozen_continuous_bc", source_updates=11, source_training_rows=30)
    for row, decision in zip(
        pure["episode"]["transitions"],
        collected.report["readiness_roll_in"]["controls"],
        strict=True,
    ):
        row["action"] = tuple(decision["student_action"])
    if fault == "ignored_plan":
        collected.report.pop("readiness_roll_in")
    elif fault == "wrong_prefix":
        collected.report["readiness_roll_in"]["teacher_steps"] = 2
    elif fault == "mixed_return":
        pure["policy"] = "mixed_readiness_teacher_and_frozen_student"
    elif fault == "mixed_metadata":
        pure["readiness_roll_in"] = deepcopy(collected.report["readiness_roll_in"])
    calls = []

    def rollout(output, **options):
        calls.append(options)
        return collected if len(calls) == 1 else smoke.FrozenRolloutResult(pure, None)

    monkeypatch.setattr(smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(smoke, "run_continuous_correction_update", lambda *args, **options: update)
    summary = smoke.run_corrective_adaptation(
        owner,
        output_directory=tmp_path / "driver",
        visual_observation=False,
        readiness_corrections=True,
        acquisition_only=True,
        observation_projection=projection,
        readiness_roll_in=roll_in.ReadinessRollInPlan(1),
    )
    if fault is not None:
        assert summary["complete"] is False
        assert [options["train_seed"] for options in calls] == (
            [17] if fault in ("ignored_plan", "wrong_prefix") else [17, 17]
        )
        if fault in ("mixed_return", "mixed_metadata"):
            assert summary["update"] == update.report
            assert len(summary["rollouts"]) == 1
        return
    assert summary["complete"] is True
    assert summary["protocol_id"] == "corrective_robot_mixed_train_17_acquisition_v1"
    assert summary["collection_behavior_scope"] == "teacher_prefix_then_frozen_student"
    assert summary["return_behavior_scope"] == "frozen_student_only"
    assert [options["train_seed"] for options in calls] == [17, 17]
    assert calls[0]["readiness_roll_in"] == roll_in.ReadinessRollInPlan(1)
    assert "readiness_roll_in" not in calls[1] and "corrective_queries" not in calls[1]


@pytest.mark.parametrize("selected", ["omitted", "explicit_none"])
def test_should_keep_default_driver_collection_callshape_without_a_roll_in_plan(
    source, monkeypatch, tmp_path, selected
):
    calls = []

    def rollout(*args, **options):
        calls.append(options)
        return smoke.FrozenRolloutResult(dict(complete=False), None)

    monkeypatch.setattr(smoke, "run_frozen_rollout", rollout)
    monkeypatch.setattr(smoke, "run_continuous_correction_update", forbidden)
    # None is the omitted API default; a non-plan bool is invalid, not False.
    options: dict[str, Any] = {} if selected == "omitted" else dict(readiness_roll_in=None)
    summary = smoke.run_corrective_adaptation(
        source.fork(), output_directory=tmp_path / "legacy", visual_observation=False, **options
    )
    assert summary["protocol_id"] == "corrective_robot_train_19_17_19_v1"
    assert "readiness_roll_in" not in calls[0] and calls[0]["train_seed"] == 19
