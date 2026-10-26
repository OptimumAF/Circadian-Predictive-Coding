"""Typed lighting acquisition and frozen inference through fake RGB IO only.

Per full module invocation: two constructors, three explicit restores, 1,586
seeded normal values, 21 prediction rows, and no fitting, SGD, or appends.
The 26 fake scenes budget 131 reads, 25 writes, 25 refreshes, and 27 controls;
factory/reset/close each occur 26 times (286 boundary calls total).
"""

from dataclasses import FrozenInstanceError, asdict, replace
import json
import sys
from typing import Any, cast

import numpy as np
import pytest

from scripts import maniskill_continuous_smoke as smoke
from scripts import maniskill_task_teacher_smoke as teacher_cli
from src.adapters.numpy_continuous_bc import ContinuousBCPolicy, NumpyContinuousBCLearner
from src.adapters.numpy_nearest_continuous_bc import (
    NearestContinuousBCPolicy,
    NearestContinuousBCSnapshot,
    NumpyNearestContinuousBCLearner,
)
from src.app.actor_shadow import StableActor
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)
from src.core.observation_standardization import ObservationStandardization
from src.core.pick_cube_camera_context import pick_cube_camera_context
from src.core.pick_cube_mass_context import pick_cube_mass_context
from src.core.pick_cube_lighting_context import pick_cube_lighting_context
from src.core.pick_cube_public_vectors import public_vector_projection
from src.core.pick_cube_rgb_features import pick_cube_rgb_recipe
from src.core.pick_cube_teacher import PickCubeGeometry
from src.infra import maniskill_continuous as port
from src.infra import maniskill_lighting as lighting
from src.infra import maniskill_task_teacher as teacher
from test_maniskill_bc_smoke import assert_same_state
from test_maniskill_rgb_observations import packet, public_feature_oracle
from test_pick_cube_lighting_context import CONTEXT, Environment


def visual_spec():
    return ContinuousActionSpec(789, port.ACTION_UNITS, (-1.0,) * 4, (1.0,) * 4)


def readback():
    return lighting.ManiSkillLightingReadback(
        pick_cube_lighting_context(), (1.0, 0.5, 0.25), (0.5, 0.25, 0.125)
    )


def lighting_work(reads=8):
    return lighting.ManiSkillLightingWork(reads, reads, 1, 1, 1, 1, 1)


def qualified_result() -> Any:
    observations = tuple(tuple(public_feature_oracle(packet(index))) for index in (20, 1, 2))
    episode = ContinuousEpisode(
        f"{CONTEXT}-teacher",
        CONTEXT,
        "train",
        visual_spec(),
        tuple(
            ContinuousTransition(
                i,
                observations[i],
                (0.2, -0.3, 0.9, 1.0),
                float(i),
                observations[i + 1],
                False,
                i == 1,
            )
            for i in range(2)
        ),
    )
    return teacher.TaskTeacherResult(
        candidate_episode=episode,
        observed_success=True,
        final_success=True,
        creation_attempts=1,
        geometry_attempts=2,
        geometry_returns=2,
        teacher_actions=2,
        stages=("align", "align"),
        work=port.ManiSkillWork(2, 2, 10, 0),
        error=None,
        traceback=None,
        observation_recipe=pick_cube_rgb_recipe(),
        visual_work=port.ManiSkillVisualWork(
            3, 3, 3 * 49152, 3 * 789, 3 * 49152, 3 * 789, 3 * 6312, 49152, 0, 0
        ),
        lighting_context=pick_cube_lighting_context(),
        lighting_readback=readback(),
        lighting_work=lighting_work(),
    )


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    environment, calls = Environment(), []

    def factory(**kwargs):
        calls.append(kwargs)
        return environment

    monkeypatch.setattr(port, "_make_environment", factory)
    monkeypatch.setattr(smoke, "version", lambda _: "fake-lighting-interface")
    monkeypatch.setattr(teacher_cli, "version", lambda _: "fake-lighting-interface")
    monkeypatch.setattr(teacher_cli.faulthandler, "enable", lambda: None)
    monkeypatch.setattr(
        teacher,
        "_geometry",
        lambda _: PickCubeGeometry((0.0,) * 3, (0.02, -0.03, 0.01), (0.05, 0.02, 0.08)),
    )
    yield environment, calls, tmp_path
    if calls and environment.close_calls == 0:
        environment.close()


@pytest.fixture(scope="module")
def source():
    model = NumpyContinuousBCLearner(ContinuousBCPolicy(visual_spec(), hidden_dim=2, seed=17))
    state = model.snapshot_state()
    parameters = tuple(np.zeros_like(value) for value in state.parameters)
    parameters[3][:] = (0.2, -0.3, 0.4, -0.5)
    model.restore_state(
        replace(
            state,
            parameters=cast(Any, parameters),
            velocity=cast(Any, tuple(np.full_like(value, 0.125) for value in state.velocity)),
            completed_updates=13,
            training_examples=377,
        )
    )
    return model


def test_should_own_strict_original_readback_and_represent_partial_failed_work():
    value = readback()
    owned = lighting.validate_lighting_readback(value, value.context)
    assert owned == value and owned is not value and owned.context is not value.context
    with pytest.raises(FrozenInstanceError):
        setattr(owned, "source_ambient_rgb", (0.0,) * 3)
    object.__setattr__(value.context, "seed", 19)
    assert owned.context == pick_cube_lighting_context()
    for work in (
        lighting.ManiSkillLightingWork(1, 0, 0, 0, 0, 0, 1),
        lighting.ManiSkillLightingWork(1, 1, 1, 0, 0, 0, 1),
        lighting.ManiSkillLightingWork(2, 2, 1, 1, 1, 0, 1),
        lighting_work(3),
    ):
        copied = lighting.validate_lighting_work(work)
        assert copied == work and copied is not work


@pytest.mark.parametrize(
    "changes",
    [
        dict(source_ambient_rgb=[1.0, 0.5, 0.25]),
        dict(source_ambient_rgb=(1, 0.5, 0.25)),
        dict(source_ambient_rgb=(np.float64(1.0), 0.5, 0.25)),
        dict(source_ambient_rgb=(1.0, 0.5)),
        dict(source_ambient_rgb=(0.0, 0.5, 0.25)),
        dict(source_ambient_rgb=(1.01, 0.5, 0.25)),
        dict(source_ambient_rgb=(float("nan"), 0.5, 0.25)),
        dict(applied_ambient_rgb=(0.25, 0.25, 0.125)),
        dict(applied_ambient_rgb=(0.5, 0.25, float("inf"))),
        dict(context=replace(pick_cube_lighting_context(), seed=cast(Any, 17.0))),
    ],
)
def test_should_refuse_untyped_nonfinite_or_nonhalved_original_readback(changes: dict[str, Any]):
    with pytest.raises(ValueError):
        lighting.validate_lighting_readback(
            replace(readback(), **changes), pick_cube_lighting_context()
        )


@pytest.mark.parametrize(
    "changes",
    [
        dict(read_attempts=True),
        dict(read_returns=8.0),
        dict(read_returns=9),
        dict(write_attempts=2),
        dict(write_returns=2),
        dict(refresh_attempts=2),
        dict(refresh_returns=2),
        dict(discarded_reset_observations=0),
        dict(discarded_reset_observations=True),
        dict(read_attempts=2**63),
        dict(read_attempts=0, read_returns=0),
        dict(write_attempts=0, write_returns=0),
        dict(read_attempts=1, read_returns=1),
    ],
)
def test_should_refuse_untyped_or_impossible_paid_lighting_work(changes: dict[str, Any]):
    with pytest.raises(ValueError):
        lighting.validate_lighting_work(replace(lighting_work(), **changes))


@pytest.mark.parametrize(
    "kind", ["readback_dict", "work_dict", "readback_subclass", "work_subclass"]
)
def test_should_require_exact_metadata_record_classes(kind):
    class Readback(lighting.ManiSkillLightingReadback):
        pass

    class Work(lighting.ManiSkillLightingWork):
        pass

    original, work = readback(), lighting_work()
    if kind == "readback_dict":
        value = cast(Any, asdict(original))
    elif kind == "work_dict":
        value = cast(Any, asdict(work))
    elif kind == "readback_subclass":
        value = Readback(
            original.context, original.source_ambient_rgb, original.applied_ambient_rgb
        )
    else:
        value = Work(*asdict(work).values())
    with pytest.raises(ValueError):
        if kind.startswith("readback"):
            lighting.validate_lighting_readback(cast(Any, value), pick_cube_lighting_context())
        else:
            lighting.validate_lighting_work(cast(Any, value))


def test_should_collect_actual_lighting_teacher_and_release_owned_qualified_rgb(boundary):
    environment, calls, _ = boundary
    declared = pick_cube_lighting_context()
    result = teacher.collect_task_teacher(visual_observation=True, lighting_context=declared)
    assert result.qualified and result.error is None and result.traceback is None
    assert result.observed_success is result.final_success is True
    assert calls == [dict(visual_observation=True)] and environment.reset_calls == [17]
    assert result.lighting_context == declared and result.lighting_context is not declared
    assert result.lighting_readback == readback() and result.lighting_work == lighting_work()
    assert result.geometry_attempts == result.geometry_returns == result.teacher_actions == 2
    assert result.work == port.ManiSkillWork(2, 2, 10, 0)
    value = result.training_episode()
    original_reward = value.transitions[0].reward
    assert (
        value.episode_id == f"{CONTEXT}-teacher"
        and value.context == CONTEXT
        and value.role == "train"
    )
    assert value.transitions[0].observation == tuple(
        public_feature_oracle(environment.refreshed_packet)
    )
    assert value.spec == visual_spec() and result.observation_recipe == pick_cube_rgb_recipe()
    object.__setattr__(declared, "seed", 19)
    object.__setattr__(value.transitions[0], "reward", 0.5)
    assert result.training_episode().transitions[0].reward == original_reward
    assert environment.close_calls == 1


def test_should_admit_synthetic_valid_lighting_qualification_before_corrupt_variants():
    result = qualified_result()
    assert result.qualified is True and result.training_episode() == result.candidate_episode


@pytest.mark.parametrize(
    "damage",
    [
        "context",
        "readback",
        "work",
        "all",
        "identity",
        "role",
        "recipe",
        "pixels",
        "controller",
        "read_count",
        "visual_count",
        "stage",
        "stage_prefix",
        "stage_type",
        "observed",
        "traceback",
        "guard",
        "mass",
        "camera",
    ],
)
def test_should_refuse_lost_relabelled_corrupted_or_mixed_lighting_teacher(damage):
    result = qualified_result()
    if damage in ("context", "readback", "work"):
        result = replace(result, **{"lighting_" + damage: None})
    elif damage == "all":
        result = replace(result, lighting_context=None, lighting_readback=None, lighting_work=None)
    elif damage in ("identity", "role"):
        field, value = (
            ("episode_id", "pick-cube-task-teacher-17")
            if damage == "identity"
            else ("role", "probe_train")
        )
        result = replace(
            result, candidate_episode=replace(result.candidate_episode, **{field: value})
        )
    elif damage == "recipe":
        result = replace(
            result, observation_recipe=replace(pick_cube_rgb_recipe(), pixel_divisor=255)
        )
    elif damage == "pixels":
        value = cast(Any, result.candidate_episode)
        row = value.transitions[-1]
        result = replace(
            result,
            candidate_episode=replace(
                value,
                transitions=value.transitions[:-1]
                + (
                    replace(
                        row,
                        next_observation=row.next_observation[:21]
                        + (1.01,)
                        + row.next_observation[22:],
                    ),
                ),
            ),
        )
    elif damage == "controller":
        result = replace(result, work=replace(result.work, completed_control_physics_steps=9))
    elif damage == "read_count":
        result = replace(result, lighting_work=lighting_work(7))
    elif damage == "visual_count":
        result = replace(result, visual_work=replace(result.visual_work, frame_export_copies=1))
    elif damage == "stage":
        result = replace(result, stages=("align",))
    elif damage == "stage_prefix":
        result = replace(result, stages=("descend", "align"))
    elif damage == "stage_type":
        result = replace(result, stages=("align", cast(Any, 1)))
    elif damage == "observed":
        result = replace(result, observed_success=cast(Any, 1))
    elif damage == "traceback":
        result = replace(result, traceback="retained failure")
    elif damage == "guard":
        result = replace(result, guard_seed=31, labels_arrived_at=2)
    elif damage == "mass":
        result = replace(result, mass_context=pick_cube_mass_context())
    else:
        result = replace(result, camera_context=pick_cube_camera_context())
    assert result.qualified is False
    with pytest.raises(ValueError):
        result.training_episode()


@pytest.mark.parametrize("failure", ["geometry", "native_step", "refresh", "cleanup"])
def test_should_retain_teacher_paid_costs_and_original_constructor_failure(
    boundary, monkeypatch, failure
):
    environment, _, _ = boundary
    if failure == "geometry":
        monkeypatch.setattr(
            teacher, "_geometry", lambda _: (_ for _ in ()).throw(RuntimeError("geometry failed"))
        )
    elif failure == "native_step":
        environment.step_failure_at = 1
    elif failure == "refresh":
        environment.refresh_error = RuntimeError("refresh failed")
    else:
        original = vars(port.ManiSkillPickCubeEpisode)["lighting_readback"].fget
        monkeypatch.setattr(
            port.ManiSkillPickCubeEpisode,
            "lighting_readback",
            property(
                lambda owner: (
                    replace(
                        original(owner),
                        source_ambient_rgb=(0.5, 0.25, 0.125),
                        applied_ambient_rgb=(0.25, 0.125, 0.0625),
                    )
                    if environment.close_calls
                    else original(owner)
                )
            ),
        )
    result = teacher.collect_task_teacher(
        visual_observation=True, lighting_context=pick_cube_lighting_context()
    )
    assert result.qualified is False and result.error is not None
    assert environment.close_calls == 1 and result.creation_attempts == 1
    assert result.lighting_context == pick_cube_lighting_context()
    assert result.lighting_readback == readback()
    if failure == "refresh":
        assert result.lighting_work == lighting.ManiSkillLightingWork(2, 2, 1, 1, 1, 0, 1)
        assert (
            result.work is None
            and result.geometry_attempts == result.geometry_returns == result.teacher_actions == 0
        )
        assert "refresh failed" in result.error
    else:
        assert result.work is not None and result.lighting_work is not None
        attempts = dict(geometry=0, native_step=1, cleanup=2)[failure]
        assert result.work.step_attempts == len(environment.commands) == attempts
        assert result.work.completed_steps == (0 if failure == "native_step" else attempts)
        assert result.lighting_work.write_returns == result.lighting_work.refresh_returns == 1
    with pytest.raises(ValueError):
        result.training_episode()


@pytest.mark.parametrize(
    "changes",
    [
        dict(train_seed=19),
        dict(train_seed=True),
        dict(visual_observation=False),
        dict(camera_context=pick_cube_camera_context()),
        dict(mass_context=pick_cube_mass_context()),
        dict(lighting_context={}),
    ],
)
def test_should_refuse_teacher_misallocation_before_factory(boundary, changes: dict[str, Any]):
    _, calls, _ = boundary
    options: dict[str, Any] = dict(
        visual_observation=True, lighting_context=pick_cube_lighting_context()
    )
    options.update(changes)
    with pytest.raises(ValueError):
        teacher.collect_task_teacher(**options)
    assert calls == []


@pytest.mark.parametrize("damage", ["context", "readback", "work"])
def test_should_refuse_teacher_geometry_callback_replacement_before_control(
    boundary, monkeypatch, damage
):
    environment, _, _ = boundary
    armed: list[bool] = []
    name = "lighting_" + damage
    original = getattr(port.ManiSkillPickCubeEpisode, name).fget

    def getter(owner):
        value = original(owner)
        if armed:
            if damage == "context":
                return replace(value, seed=17.0)
            if damage == "readback":
                return replace(
                    value,
                    source_ambient_rgb=(0.5, 0.25, 0.125),
                    applied_ambient_rgb=(0.25, 0.125, 0.0625),
                )
            return replace(value, read_returns=2)
        return value

    def geometry(_owner):
        armed.append(True)
        return PickCubeGeometry((0.0,) * 3, (0.02, -0.03, 0.01), (0.05, 0.02, 0.08))

    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, name, property(getter))
    monkeypatch.setattr(teacher, "_geometry", geometry)
    result = teacher.collect_task_teacher(
        visual_observation=True, lighting_context=pick_cube_lighting_context()
    )
    assert result.qualified is False and result.error is not None
    assert result.geometry_attempts == result.geometry_returns == 1 and result.teacher_actions == 0
    assert result.work is not None
    assert environment.commands == [] and result.work.step_attempts == 0
    assert result.lighting_readback == readback()


def test_should_run_actual_frozen_rgb_predictions_without_changing_any_source_array(
    boundary, source
):
    environment, calls, directory = boundary
    before = source.snapshot_state()
    outcome = smoke.run_frozen_rollout(
        directory / "lighting.json",
        learner=source,
        visual_observation=True,
        lighting_context=pick_cube_lighting_context(),
    )
    report = outcome.report
    assert report["complete"] is True and report["lighting_context_exact"] is True
    assert report["context"] == CONTEXT and report["seed"] == 17
    assert report["lighting_context"] == asdict(pick_cube_lighting_context())
    assert report["lighting_readback"] == asdict(readback()) and report["lighting_work"] == asdict(
        lighting_work()
    )
    assert report["source_updates"] == 13 and report["source_training_rows"] == 377
    assert (
        report["prediction_attempts"]
        == report["prediction_returns"]
        == report["validated_prediction_rows"]
        == 2
    )
    assert report["validated_prediction_bytes"] == 64 and report["optimizer_updates"] == 0
    assert report["frozen_state_exact"] is report["original_owner_exact"] is True
    assert outcome.corrections is None and environment.close_calls == 1
    expected = np.tanh(np.array([[0.2, -0.3, 0.4, -0.5]])).astype(np.float32)
    for command in environment.commands:
        np.testing.assert_array_equal(command, expected)
    assert calls == [dict(visual_observation=True)]
    assert report["episode"]["transitions"][0]["observation"] == tuple(
        public_feature_oracle(environment.refreshed_packet)
    )
    assert_same_state(before, source.snapshot_state())
    assert json.loads((directory / "lighting.json").read_text()) == json.loads(json.dumps(report))


@pytest.mark.parametrize("phase", ["after_control", "cleanup"])
@pytest.mark.parametrize("damage", ["context", "readback", "work"])
def test_should_clear_success_on_late_lighting_metadata_without_losing_paid_controls(
    boundary, source, monkeypatch, phase, damage
):
    environment, _, directory = boundary
    before = source.snapshot_state()
    name = "lighting_" + damage
    original = getattr(port.ManiSkillPickCubeEpisode, name).fget

    def getter(owner):
        value = original(owner)
        corrupt = (
            bool(environment.commands)
            if phase == "after_control"
            else bool(environment.close_calls)
        )
        if corrupt:
            if damage == "context":
                return replace(value, seed=17.0)
            if damage == "readback":
                return replace(
                    value,
                    source_ambient_rgb=(0.5, 0.25, 0.125),
                    applied_ambient_rgb=(0.25, 0.125, 0.0625),
                )
            return replace(value, refresh_returns=True)
        return value

    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, name, property(getter))
    report = smoke.run_frozen_rollout(
        directory / "late.json",
        learner=source,
        visual_observation=True,
        lighting_context=pick_cube_lighting_context(),
    ).report
    assert report["complete"] is False and report["lighting_context_exact"] is not True
    paid = 1 if phase == "after_control" else 2
    assert len(environment.commands) == report["work"]["completed_steps"] == paid
    assert (
        report["prediction_attempts"]
        == report["prediction_returns"]
        == report["validated_prediction_rows"]
        == paid
    )
    assert report["lighting_readback"] == asdict(readback())
    assert environment.close_calls == 1
    assert_same_state(before, source.snapshot_state())


@pytest.mark.parametrize(
    "changes",
    [
        dict(train_seed=19),
        dict(visual_observation=False),
        dict(camera_context=pick_cube_camera_context()),
        dict(mass_context=pick_cube_mass_context()),
        dict(corrective_queries=True),
        dict(observation_projection=public_vector_projection()),
        dict(lighting_context={}),
    ],
)
def test_should_refuse_invalid_rollout_context_before_fork_versions_or_factory(
    boundary, source, monkeypatch, changes: dict[str, Any]
):
    _, calls, directory = boundary
    monkeypatch.setattr(
        NumpyContinuousBCLearner, "fork", lambda _: pytest.fail("invalid lighting forked source")
    )
    monkeypatch.setattr(
        smoke, "version", lambda _: pytest.fail("invalid lighting queried SDK version")
    )
    options: dict[str, Any] = dict(
        learner=source, visual_observation=True, lighting_context=pick_cube_lighting_context()
    )
    options.update(changes)
    report = smoke.run_frozen_rollout(directory / "invalid.json", **options).report
    assert (
        report["complete"] is False
        and report["creation_attempts"] == report["prediction_attempts"] == 0
    )
    assert calls == []


def test_should_keep_original_constructor_lighting_cost_in_failed_frozen_report(boundary, source):
    environment, _, directory = boundary
    environment.refresh_error = RuntimeError("refresh failed before encoding")
    report = smoke.run_frozen_rollout(
        directory / "constructor.json",
        learner=source,
        visual_observation=True,
        lighting_context=pick_cube_lighting_context(),
    ).report
    assert (
        report["complete"] is False
        and report["creation_attempts"] == 1
        and report["creation_returns"] == 0
    )
    assert report["prediction_attempts"] == report["prediction_returns"] == 0
    assert report["lighting_work"] == asdict(lighting.ManiSkillLightingWork(2, 2, 1, 1, 1, 0, 1))
    assert report["lighting_readback"] == asdict(readback())
    assert report["lighting_context_exact"] is not True and environment.close_calls == 1


def test_should_retain_unreturned_native_step_and_prediction_cost(boundary, source):
    environment, _, directory = boundary
    before = source.snapshot_state()
    environment.step_failure_at = 1
    report = smoke.run_frozen_rollout(
        directory / "step-failure.json",
        learner=source,
        visual_observation=True,
        lighting_context=pick_cube_lighting_context(),
    ).report
    assert report["complete"] is False
    assert (
        report["prediction_attempts"]
        == report["prediction_returns"]
        == report["validated_prediction_rows"]
        == 1
    )
    assert report["work"] == asdict(port.ManiSkillWork(1, 0, 0, 5))
    assert report["lighting_work"] == asdict(lighting_work(4))
    assert report["lighting_readback"] == asdict(readback())
    assert len(environment.commands) == environment.close_calls == 1
    assert_same_state(before, source.snapshot_state())


@pytest.mark.parametrize(
    "direction", ["source", "binding_before", "binding_final", "visual_before"]
)
def test_should_close_both_callback_ordering_gaps_and_refuse_late_false_success(
    boundary, source, monkeypatch, direction
):
    environment, _, directory = boundary
    before = source.snapshot_state()
    original_init = port.ManiSkillPickCubeEpisode.__init__
    original_predict = NumpyContinuousBCLearner.predict
    original_snapshot = NumpyContinuousBCLearner.snapshot_state
    original_readback = vars(port.ManiSkillPickCubeEpisode)["lighting_readback"].fget
    ports, predicted, armed, mutated = [], [], [], []

    def initialize(owner, *args, **kwargs):
        original_init(owner, *args, **kwargs)
        ports.append(owner)

    def predict(owner, features):
        values = original_predict(owner, features)
        predicted.append(True)
        return values

    def getter(owner):
        value = original_readback(owner)
        if predicted:
            armed.append(True)
            if direction == "source":
                getattr(source, "_state").parameters[0][0, 0] = 99.0
                mutated.append(True)
        return value

    def snapshot(owner):
        value = original_snapshot(owner)
        phase = direction in ("binding_before", "visual_before") or environment.close_calls > 0
        if owner is source and armed and not mutated and direction != "source" and phase:
            if direction == "visual_before":
                setattr(ports[0], "_visual_returns", getattr(ports[0], "_visual_returns") + 1)
            else:
                setattr(ports[0], "_lighting_context", None)
            mutated.append(True)
        return value

    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "__init__", initialize)
    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "lighting_readback", property(getter))
    monkeypatch.setattr(NumpyContinuousBCLearner, "predict", predict)
    monkeypatch.setattr(NumpyContinuousBCLearner, "snapshot_state", snapshot)
    try:
        report = smoke.run_frozen_rollout(
            directory / "callback.json",
            learner=source,
            visual_observation=True,
            lighting_context=pick_cube_lighting_context(),
        ).report
        assert (
            mutated and report["complete"] is False and report["lighting_context_exact"] is not True
        )
        predictions, controls = (2, 2) if direction == "binding_final" else (1, 0)
        assert (
            report["prediction_attempts"]
            == report["prediction_returns"]
            == report["validated_prediction_rows"]
            == predictions
        )
        assert report["work"]["completed_steps"] == len(environment.commands) == controls
        if direction == "source":
            assert report["original_owner_exact"] is not True
        else:
            assert report["original_owner_exact"] is report["frozen_state_exact"] is True
    finally:
        if direction == "source":
            source.restore_state(before)
    assert_same_state(before, source.snapshot_state())


def test_should_refuse_wrong_returned_visual_cost_without_new_controls(
    boundary, source, monkeypatch
):
    environment, _, directory = boundary
    before = source.snapshot_state()
    original = vars(port.ManiSkillPickCubeEpisode)["visual_work"].fget

    def getter(owner):
        value = original(owner)
        return replace(value, returned_feature_values=value.returned_feature_values + 1)

    monkeypatch.setattr(port.ManiSkillPickCubeEpisode, "visual_work", property(getter))
    report = smoke.run_frozen_rollout(
        directory / "visual-cost.json",
        learner=source,
        visual_observation=True,
        lighting_context=pick_cube_lighting_context(),
    ).report
    assert report["complete"] is False and report["lighting_context_exact"] is not True
    assert report["prediction_attempts"] == report["prediction_returns"] == 0
    assert environment.commands == [] and report["work"]["completed_steps"] == 0
    assert report["lighting_work"] == asdict(lighting_work(3))
    assert environment.close_calls == 1
    assert_same_state(before, source.snapshot_state())


def test_should_serve_actual_stable_nearest_actor_without_changing_source_or_actor_arrays(boundary):
    environment, _, directory = boundary
    statistics = ObservationStandardization((0.0,) * 789, (1.0,) * 789, 2)
    policy = NearestContinuousBCPolicy(visual_spec(), statistics)
    model = NumpyNearestContinuousBCLearner(policy)
    features = np.asarray(
        [public_feature_oracle(packet(index)) for index in (20, 1)], dtype=np.float64
    )
    actions = np.array([[0.125, -0.25, 0.5, -1.0], [-0.5, 0.25, 0.0, 1.0]], dtype=np.float64)
    model.restore_state(NearestContinuousBCSnapshot(policy, features, actions, 1, 2))
    initial = model.snapshot_state()
    actor = StableActor(model, version="lighting-actor-0")
    report = smoke.run_frozen_rollout(
        directory / "actor.json",
        actor=actor,
        visual_observation=True,
        lighting_context=pick_cube_lighting_context(),
    ).report
    assert report["complete"] is True and report["lighting_context_exact"] is True
    assert (
        report["actor_version"] == "lighting-actor-0"
        and report["actor_version_exact"] is report["actor_backend_exact"] is True
    )
    assert report["actor_prediction_attempts"] == report["actor_prediction_returns"] == 2
    assert report["frozen_state_exact"] is report["original_owner_exact"] is True
    assert report["optimizer_updates"] == 0 and report["lighting_work"] == asdict(lighting_work())
    for command, action in zip(environment.commands, actions, strict=True):
        np.testing.assert_array_equal(command[0], action.astype(np.float32))
    for observed in (model.snapshot_state(), actor.snapshot_state().state):
        assert (
            observed.policy == initial.policy
            and observed.completed_updates == 1
            and observed.training_examples == 2
        )
        np.testing.assert_array_equal(observed.prototype_features, initial.prototype_features)
        np.testing.assert_array_equal(observed.prototype_actions, initial.prototype_actions)


def test_should_keep_default_visual_reports_and_existing_output_unchanged(boundary):
    _, calls, directory = boundary
    output = directory / "default.json"
    report = smoke.run_frozen_rollout(output, visual_observation=True).report
    assert report["complete"] is True
    assert (
        not {"lighting_context", "lighting_readback", "lighting_work", "lighting_context_exact"}
        & report.keys()
    )
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        smoke.run_frozen_rollout(
            output, visual_observation=True, lighting_context=pick_cube_lighting_context()
        )
    assert output.read_bytes() == before and len(calls) == 1


@pytest.mark.parametrize("entry", ["teacher", "rollout"])
def test_should_forward_explicit_lighting_cli_and_keep_owned_returned_metadata(
    boundary, source, monkeypatch, entry
):
    _, _, directory = boundary
    output = directory / "cli.json"
    monkeypatch.setattr(
        sys, "argv", [entry, "--output", str(output), "--visual-observation", "--lighting-drift"]
    )
    assert (teacher_cli.main() if entry == "teacher" else smoke.main(learner=source)) == 0
    report = json.loads(output.read_text())
    assert report["lighting_context"] == asdict(pick_cube_lighting_context())
    assert report["lighting_readback"] == json.loads(json.dumps(asdict(readback())))
    assert report["lighting_work"] == asdict(lighting_work())
    assert report["context"] == CONTEXT and report["seed"] == 17
    assert report["qualified" if entry == "teacher" else "complete"] is True


def test_should_omit_new_default_teacher_cli_fields_but_keep_python_none_values(
    boundary, monkeypatch
):
    _, _, directory = boundary
    result = qualified_result()
    episode = replace(
        result.candidate_episode, episode_id="pick-cube-task-teacher-17", context="pick-cube"
    )
    default = replace(
        result,
        candidate_episode=episode,
        lighting_context=None,
        lighting_readback=None,
        lighting_work=None,
    )
    assert all(
        getattr(default, name) is None and asdict(default)[name] is None
        for name in ("lighting_context", "lighting_readback", "lighting_work")
    )
    monkeypatch.setattr(teacher_cli, "collect_task_teacher", lambda **_options: default)
    output = directory / "default-teacher.json"
    monkeypatch.setattr(sys, "argv", ["teacher", "--output", str(output), "--visual-observation"])
    assert teacher_cli.main() == 0
    report = json.loads(output.read_text())
    assert not {"lighting_context", "lighting_readback", "lighting_work"} & report.keys()


def test_should_reject_qualified_teacher_returned_under_a_different_requested_lighting_context(
    boundary, monkeypatch
):
    _, calls, directory = boundary
    result = qualified_result()
    episode = replace(
        result.candidate_episode, episode_id="pick-cube-task-teacher-17", context="pick-cube"
    )
    default = replace(
        result,
        candidate_episode=episode,
        lighting_context=None,
        lighting_readback=None,
        lighting_work=None,
    )
    assert default.qualified is True
    monkeypatch.setattr(teacher_cli, "collect_task_teacher", lambda **_options: default)
    output = directory / "misqualified-teacher.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["teacher", "--output", str(output), "--visual-observation", "--lighting-drift"],
    )
    assert teacher_cli.main() == 1
    report = json.loads(output.read_text())
    assert report["qualified"] is False and report["lighting_request_error"]
    assert calls == []


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--train-seed", "19", "--visual-observation"],
        ["--guard-seed", "31", "--labels-arrived-at", "2", "--visual-observation"],
        ["--visual-observation", "--camera-drift"],
        ["--visual-observation", "--mass-drift"],
    ],
)
def test_should_refuse_invalid_teacher_cli_lighting_allocation_before_output(
    boundary, monkeypatch, flags
):
    _, calls, directory = boundary
    output = directory / "invalid-cli.json"
    monkeypatch.setattr(
        sys, "argv", ["teacher", "--output", str(output), "--lighting-drift", *flags]
    )
    with pytest.raises(SystemExit) as caught:
        teacher_cli.main()
    assert caught.value.code == 2 and not output.exists() and calls == []
